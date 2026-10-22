#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#  ---------------------------------------------------------------------
#
#  _____    _      _              _         _____ _____
# | ____|__| | ___| |_      _____(_)___ ___|  ___| ____|
# |  _| / _` |/ _ \ \ \ /\ / / _ \ / __/ __| |_  |  _|
# | |__| (_| |  __/ |\ V  V /  __/ \__ \__ \  _| | |___
# |_____\__,_|\___|_| \_/\_/ \___|_|___/___/_|   |_____|
#
#
#  Unit of Strength of Materials and Structural Analysis
#  University of Innsbruck,
#  2017 - today
#
#  Matthias Neuner matthias.neuner@uibk.ac.at
#
#  This file is part of EdelweissFE.
#
#  This library is free software; you can redistribute it and/or
#  modify it under the terms of the GNU Lesser General Public
#  License as published by the Free Software Foundation; either
#  version 2.1 of the License, or (at your option) any later version.
#
#  The full text of the license can be found in the file LICENSE.md at
#  the top level directory of EdelweissFE.
#  ---------------------------------------------------------------------
# Created on Sun Jan  8 20:37:35 2017

# @author: Matthias Neuner

import json
from dataclasses import dataclass
from time import perf_counter

import numpy as np
from scipy.sparse import csr_matrix

import edelweissfe.utils.performancetiming as performancetiming
from edelweissfe.config.linsolve import getDefaultLinSolver, getLinSolverByName
from edelweissfe.constraints.base.constraintbase import ConstraintBase
from edelweissfe.models.femodel import FEModel
from edelweissfe.numerics.csrgeneratorv2 import CSRGenerator
from edelweissfe.numerics.dofmanager import DofManager, DofVector, VIJSystemMatrix
from edelweissfe.numerics.parallelizationutilities import forEachSliceInParallel
from edelweissfe.outputmanagers.base.outputmanagerbase import OutputManagerBase
from edelweissfe.solvers.base.convergencecriteria import (
    AbaqusConvergenceCriterion,
    ConvergenceCriterionBase,
    LegacyConvergenceCriterion,
)
from edelweissfe.solvers.base.convergencetable import ConvergenceTable
from edelweissfe.solvers.base.dirichlet import applyDirichletToStiffness
from edelweissfe.solvers.base.nonlinearsolverbase import NonlinearSolverBase
from edelweissfe.stepactions.base.stepactionbase import StepActionBase
from edelweissfe.timesteppers.timestep import TimeStep
from edelweissfe.utils.exceptions import (
    CutbackRequest,
    DivergingSolution,
    IncrementFailed,
    ReachedMaxIterations,
)
from edelweissfe.utils.fieldoutput import FieldOutputController
from edelweissfe.utils.schema import schemaField


@dataclass(frozen=True)
class NISTSchema:
    """The options of the ``*solver`` datalines and of an ``>>options`` block routed to this
    solver, owned by this module and never mutated from outside it.

    Mirrors :attr:`NIST.SolverSpecificOptions` one-for-one; that dict remains the actual source of
    truth consulted at runtime (``self.options``, a plain mutable dict) -- this schema exists so the
    registry and the name-based ``>>options`` override mechanism have a typed description of what
    this solver accepts, without requiring every internal ``self.options[...]`` access to become a
    dataclass attribute access.
    """

    defaultMaxIter: int | None = schemaField(
        description="The default maximum number of iterations.", dtype=int, default=10
    )
    defaultCriticalIter: int | None = schemaField(
        description="The default number of critical iterations.", dtype=int, default=5
    )
    defaultMaxGrowingIter: int | None = schemaField(
        description="The default number of allowed residual growths.", dtype=int, default=10
    )
    extrapolation: str | None = schemaField(
        description="The extrapolation strategy for new increments (off|linear).", dtype=str, default="linear"
    )
    extrapolateAfterModelChange: bool | None = schemaField(
        description=(
            "Whether to extrapolate the predictor on the increment FOLLOWING a model change (adaptive mesh "
            "refinement). Set False to start that increment from a zero predictor, avoiding extrapolation of "
            "the one-off warm-start/remesh settling transient."
        ),
        dtype=bool,
        default=True,
    )
    equilibrateAfterModelChange: bool | None = schemaField(
        description=(
            "Whether to insert one constant-load, zero-time re-equilibration increment immediately after an "
            "adaptive mesh refinement, before advancing the load. When True, the warm-started refined mesh is "
            "first settled to equilibrium at the last converged load level (no load advance, no Dirichlet "
            "increment, zero time increment) so the subsequent load-advancing increment starts from an "
            "equilibrated state. Intended for softening problems where remeshing near the process zone "
            "otherwise couples the load advance with the warm-start settling transient in one solve. Note: "
            "the equilibration solve integrates materials with dT=0, which suits rate-independent models; "
            "rate-dependent materials see no time advance during it (by design)."
        ),
        dtype=bool,
        default=False,
    )
    linsolver: str | None = schemaField(description="The linear solver to be used.", dtype=str, default="pardiso")
    linsolverConfigFile: str | None = schemaField(
        description="A JSON configuration file for the linear solver.", dtype=str, default=""
    )
    pruneCondensedMatrixZeros: bool | None = schemaField(
        description=(
            "Compact explicitly stored zeros out of the multi-point-constraint-condensed system "
            "matrix before solving. Pruning removes whichever entries happen to be exactly zero on "
            "this particular iteration, so the pattern changes on almost every Newton iteration -- "
            "which defeats any linear solver that relies on a stable pattern across iterations (e.g. "
            "blockamg reusing a per-field AMG hierarchy built for the old one). Left unset (the "
            "default), the choice is automatic: pruned unless the linear solver in force reports that "
            "it needs the pattern stable "
            "(:attr:`~edelweissfe.linsolve.base.LinearSolver.needsStableSparsityPattern`), in which "
            "case pruning is skipped so that solver's own reuse can engage. An explicit True/False "
            "here always wins over the automatic choice. Pruning exists in the first place because it "
            "was needed for PARDISO: its reordering is sensitive to the extra structural entries on "
            "these path-dependent condensed systems, and keeping them has been observed to drift from "
            "the converged reference path -- only turn it off for a solver that does not share that "
            "sensitivity, and verify the load path against a reference run."
        ),
        dtype=bool,
        default=None,
        documentedDefault="auto",
    )
    useThreadedMPCCondensation: bool | None = schemaField(
        description=(
            "Condense the multi-point-constraint system matrix, T^T K T + C, with an OpenMP-threaded "
            "kernel instead of SciPy's single-threaded CSR routines: every row of the result is computed "
            "directly from the rows of K it combines, with no intermediate matrix, and rows of K that no "
            "constraint touches are copied unchanged. The result does not depend on the number of "
            "threads. Leaves more raw nnz than the "
            "plain expression (explicit zeros are kept, SciPy's products drop exact-cancellation "
            "zeros); applyDirichletToStiffness prunes them right after, gated by "
            "pruneCondensedMatrixZeros (default auto: pruned for PARDISO, kept for blockamg, whose "
            "hierarchy reuse needs a stable pattern). Off by default."
        ),
        dtype=bool,
        default=False,
    )
    reportPerformanceFrequency: int | None = schemaField(
        description=(
            "Print the cumulative performance table (totals since the start of the step) every this "
            "many increments, in addition to the one printed at the end of the step. 0 (default) "
            "disables the periodic report."
        ),
        dtype=int,
        default=0,
        optionName="report-performance-frequency",
    )
    convergenceCriterion: str | None = schemaField(
        description=(
            "The convergence criterion of the Newton iterations, legacy (the original EdelweissFE one) or abaqus "
            "(the one of Abaqus/Standard); see edelweissfe.solvers.base.convergencecriteria."
        ),
        dtype=str,
        default="legacy",
    )
    abaqusResidualTolerance: float | None = schemaField(
        description="abaqus criterion: R_n, the flux residual tolerance relative to the time average flux.",
        dtype=float,
        default=5e-3,
    )
    abaqusCorrectionTolerance: float | None = schemaField(
        description="abaqus criterion: C_n, the correction tolerance relative to the largest increment of the field.",
        dtype=float,
        default=1e-2,
    )
    abaqusResidualToleranceAlternative: float | None = schemaField(
        description="abaqus criterion: R_p, the flux residual tolerance used from iteration I_p on.",
        dtype=float,
        default=2e-2,
    )
    abaqusIterationsForAlternativeTolerance: int | None = schemaField(
        description="abaqus criterion: I_p, the iteration from which on R_p is used.", dtype=int, default=9
    )
    abaqusZeroFluxThreshold: float | None = schemaField(
        description="abaqus criterion: epsilon, below which (relative to the time average flux) a field carries no flux.",
        dtype=float,
        default=1e-5,
    )
    abaqusZeroFluxCorrectionTolerance: float | None = schemaField(
        description="abaqus criterion: C_epsilon, the relative correction tolerance of a field without flux.",
        dtype=float,
        default=1e-3,
    )
    abaqusLinearResidualTolerance: float | None = schemaField(
        description="abaqus criterion: R_l, below which a field is converged irrespective of its correction.",
        dtype=float,
        default=1e-8,
    )
    abaqusInitialTimeAveragedFlux: float | None = schemaField(
        description="abaqus criterion: q~0, the time average flux of a field that has not carried flux yet (first step).",
        dtype=float,
        default=1e-2,
    )


class NIST(NonlinearSolverBase):
    """This is the Nonlinear Implicit STatic -- solver.

    Parameters
    ----------
    jobInfo
        A dictionary containing the job information.
    journal
        The journal instance for logging.
    """

    identification = "NISTSolver"

    supportsMPC = True
    supportsModelModifiers = True

    #: Option schema for this solver, per OptionSchemaProvider.
    schema = NISTSchema

    SolverSpecificOptions = {
        "defaultMaxIter": 10,
        "defaultCriticalIter": 5,
        "defaultMaxGrowingIter": 10,
        "extrapolation": "linear",
        "extrapolateAfterModelChange": True,
        "equilibrateAfterModelChange": False,
        "linsolver": "",
        "linsolverConfigFile": "",
        "pruneCondensedMatrixZeros": None,
        "useThreadedMPCCondensation": False,
        "report-performance-frequency": 0,
        "convergenceCriterion": "legacy",
        "abaqusResidualTolerance": 5e-3,
        "abaqusCorrectionTolerance": 1e-2,
        "abaqusResidualToleranceAlternative": 2e-2,
        "abaqusIterationsForAlternativeTolerance": 9,
        "abaqusZeroFluxThreshold": 1e-5,
        "abaqusZeroFluxCorrectionTolerance": 1e-3,
        "abaqusLinearResidualTolerance": 1e-8,
        "abaqusInitialTimeAveragedFlux": 1e-2,
    }

    #: The convergence criterion of the current step.
    convergenceCriterion = None

    #: The predictor's state between increments: the last accepted increment and its dU.
    checkpointedState = {"prevTimeStep": TimeStep, "dU": np.ndarray}

    def __init__(self, jobInfo, journal, **kwargs):
        self.journal = journal

        self.fieldCorrectionTolerances = jobInfo["fieldCorrectionTolerance"]
        self.fluxResidualTolerances = jobInfo["fluxResidualTolerance"]
        self.fluxResidualTolerancesAlt = jobInfo["fluxResidualToleranceAlternative"]

        self.options = self.SolverSpecificOptions.copy()
        # the *solver datalines belong only to this solver: unknown entries are typos
        self._updateOptions(kwargs, journal, strict=True)

        #: The last accepted increment, from which the predictor extrapolates (None: no extrapolation).
        self.prevTimeStep = None
        #: The solution increment of the last accepted increment.
        self.dU = None

    def beginStep(
        self,
        step,
        model: FEModel,
        fieldOutputController: FieldOutputController,
        outputmanagers: dict[str, OutputManagerBase],
    ):
        """Set up the linear solver, and start the step; see
        :meth:`~edelweissfe.solvers.base.nonlinearsolverbase.NonlinearSolverBase.beginStep`.

        Parameters
        ----------
        step
            The step to be solved.
        model
            The model tree.
        fieldOutputController
            The field output controller.
        outputmanagers
            The output managers.
        """

        # self.options already reflects every >>options block applied so far (see applyOptionsOverride).
        linsolverOptions = self.options["linsolverConfigFile"]
        linsolverOptionDict = json.load(open(linsolverOptions, "r")) if linsolverOptions else ""
        self.linSolver = (
            getLinSolverByName(self.options["linsolver"], linsolverOptionDict)
            if self.options["linsolver"]
            else getDefaultLinSolver()
        )
        self.linSolver.setJournal(self.journal)

        convergenceCriterion = self.createConvergenceCriterion()
        convergenceCriterion.startStep(self.convergenceCriterion)
        self.convergenceCriterion = convergenceCriterion

        # The equation system is (re)built lazily in the first increment, and whenever the topology
        # update reports a change of the mesh or of a constraint's DOF footprint.
        self.theDofManager = None
        self.mpcTransformation = None
        self._U = self._dU = self._P = self._K = None

        # A step starts without a predictor state; a resumed step takes over the checkpointed one.
        self.prevTimeStep = None
        self.dU = None

        self.validateModelCapabilities(model)

        self.applyStepActionsAtStepStart(model, step)

        # The timing table and the step's wall clock must start together.
        performancetiming.reset()
        self._stepWallClockTic = perf_counter()

    def prepareIncrement(self, step, model: FEModel, isRetry: bool):
        """The topology update, at the start of every increment; see
        :meth:`~edelweissfe.solvers.base.nonlinearsolverbase.NonlinearSolverBase.prepareIncrement`.

        Parameters
        ----------
        step
            The step being solved.
        model
            The model tree.
        isRetry
            True if the increment is retried after a cutback: the model modifiers already decided on
            the state the retry starts from.
        """

        # Before the topology update, so that its messages head this increment's block.
        self.journal.printSeperationLine()

        topologyUpdate = self.updateTopologyAndConnectivity(model, step, offerModelModifiers=not isRetry)
        self._modelHasChanged = topologyUpdate.topologyChanged
        self._connectivityHasChanged = (
            topologyUpdate.meshDependentsRefreshed or topologyUpdate.constraintConnectivityChanged
        )

    def attemptIncrement(self, step, model: FEModel, timeStep: TimeStep):
        """Solve the increment by Newton's method, rebuilding the equation system first if the topology
        update changed the model, and re-equilibrating a changed model first if so configured; see
        :meth:`~edelweissfe.solvers.base.nonlinearsolverbase.NonlinearSolverBase.attemptIncrement`.

        Parameters
        ----------
        step
            The step being solved.
        model
            The model tree.
        timeStep
            The increment.

        Raises
        ------
        IncrementFailed
            If Newton's method does not converge, or a material requests a cutback.
        """

        extrapolation = self.options["extrapolation"]
        equilibrateAfterModelChange = self.options["equilibrateAfterModelChange"]
        reportPerformanceFrequency = self.options["report-performance-frequency"]
        stepWallClockTic = self._stepWallClockTic
        maxIter = step.maxIter
        maxGrowingIter = step.maxGrowIter
        cutbackFactor = step.cutbackFactor
        modelHasChanged = self._modelHasChanged
        connectivityHasChanged = self._connectivityHasChanged
        U, dU, P, K = self._U, self._dU, self._P, self._K

        # The header of this increment's block; all messages below are nested one level deeper.
        self.journal.message(
            "increment {:}: {:8f}, {:8f}; time {:10f} to {:10f}".format(
                timeStep.number,
                timeStep.stepProgressIncrement,
                timeStep.stepProgress,
                timeStep.totalTime - timeStep.timeIncrement,
                timeStep.totalTime,
            ),
            self.identification,
            level=1,
        )

        if reportPerformanceFrequency and timeStep.number > 0 and timeStep.number % reportPerformanceFrequency == 0:
            self.journal.printPrettyTable(
                performancetiming.makePrettyTable(wallTime=perf_counter() - stepWallClockTic),
                self.identification,
            )

        if modelHasChanged or connectivityHasChanged or self.theDofManager is None:
            with performancetiming.timeit("equation system rebuild"):
                with performancetiming.timeit("dof manager"):
                    self.theDofManager = DofManager(
                        model.nodeFields.values(),
                        model.scalarVariables.values(),
                        model.elements.values(),
                        model.constraints.values(),
                        model.nodeSets.values(),
                    )
                # the cache is keyed on the DofManager; drop it so the old manager can be freed
                self._dirichletIndicesCache = None
                self.journal.message(
                    "eq. system rebuilt: {:} dof".format(self.theDofManager.nDof),
                    self.identification,
                    2,
                )

                # Fields are laid out in contiguous slices: report the block structure per field.
                for fieldName, fieldIndices in self.theDofManager.idcsOfFieldsInDofVector.items():
                    self.journal.message(
                        "field '{:}': {:} dof, [{:}, {:})".format(
                            fieldName,
                            fieldIndices.stop - fieldIndices.start,
                            fieldIndices.start,
                            fieldIndices.stop,
                        ),
                        self.identification,
                        2,
                    )

                # Re-pushed on every (re)build so the linear solver tracks the mesh across AMR.
                with performancetiming.timeit("linear solver setup"):
                    self.linSolver.setModel(model, self.theDofManager)

                presentVariableNames = list(self.theDofManager.idcsOfFieldsInDofVector.keys())

                if self.theDofManager.idcsOfScalarVariablesInDofVector:
                    presentVariableNames += [
                        "scalar variables",
                    ]

                self.convergenceTable = ConvergenceTable(presentVariableNames, self.linSolver.reportsSolveSummary)

                with performancetiming.timeit("system matrix pattern"):
                    K = self.theDofManager.constructVIJSystemMatrix()
                    self.csrGenerator = CSRGenerator(K)

                U = self.theDofManager.constructDofVector()
                P = self.theDofManager.constructDofVector()
                dU = self.theDofManager.constructDofVector()

                for fieldName, field in model.nodeFields.items():
                    U = self.theDofManager.writeNodeFieldToDofVector(U, field, "U")

                for variable in model.scalarVariables.values():
                    U[self.theDofManager.idcsOfScalarVariablesInDofVector[variable]] = variable.value

                with performancetiming.timeit("mpc transformation build"):
                    self.mpcTransformation = self.buildMPCTransformation(model, step.actions)
                with performancetiming.timeit("mpc dirichlet conflict check"):
                    self.checkMPCDirichletConflicts(self.mpcTransformation, step.actions)

                # After a topology change, the last dU belongs to a different DOF layout: do not
                # extrapolate this increment.
                if modelHasChanged or connectivityHasChanged:
                    self.prevTimeStep = None
                elif self.dU is not None:
                    dU[:] = self.dU
                self._U, self._dU, self._P, self._K = U, dU, P, K

        self.incrementStatus = {
            "step": step.number,
            "inc": timeStep.number,
            "iters": None,
            "converged": False,
            "time inc": timeStep.timeIncrement,
            "time end": timeStep.totalTime,
            "notes": "",
        }

        self.journal.preformatted(self.convergenceTable.headerLines(), self.identification, level=2)

        if modelHasChanged and equilibrateAfterModelChange:
            # Settle the refined mesh at the LAST converged load: a time step with zero step-progress
            # and time increment holds all loads and yields a zero Dirichlet increment.
            equilibrationTimeStep = TimeStep(
                timeStep.number,
                0.0,
                timeStep.stepProgress - timeStep.stepProgressIncrement,
                0.0,
                timeStep.stepTime - timeStep.timeIncrement,
                timeStep.totalTime - timeStep.timeIncrement,
            )
            self.journal.message(
                "Model changed: re-equilibrating at constant load before advancing",
                self.identification,
                1,
            )
            try:
                U, dU, P, _, _ = self.solveIncrement(
                    U,
                    dU,
                    P,
                    K,
                    step.actions,
                    model,
                    equilibrationTimeStep,
                    None,
                    extrapolation,
                    maxIter,
                    maxGrowingIter,
                )
            except (CutbackRequest, ReachedMaxIterations, DivergingSolution) as e:
                self.journal.message(
                    "Re-equilibration after model change failed ({:}); cutting back".format(str(e)),
                    self.identification,
                    1,
                )
                self.prevTimeStep = None
                self.incrementStatus["iters"] = np.inf
                self.incrementStatus["notes"] = "re-equilibration failed: {:}".format(str(e))
                raise IncrementFailed(str(e), cutbackFactor) from e

            # Commit the settled state as a converged sub-increment (no output): elements integrate
            # incrementally from the committed state, so otherwise U and the internal state disagree.
            for fieldName, field in model.nodeFields.items():
                self.theDofManager.writeDofVectorToNodeField(U, field, "U")
                self.theDofManager.writeDofVectorToNodeField(P, field, "P")
                self.theDofManager.writeDofVectorToNodeField(dU, field, "dU")
            for variable in model.scalarVariables.values():
                variable.value = U[self.theDofManager.idcsOfScalarVariablesInDofVector[variable]]
            model.advanceToTime(equilibrationTimeStep.totalTime)
            self._U, self._dU, self._P = U, dU, P

        try:
            U, dU, P, iterationCounter, incrementResidualHistory = self.solveIncrement(
                U,
                dU,
                P,
                K,
                step.actions,
                model,
                timeStep,
                self.prevTimeStep,
                extrapolation,
                maxIter,
                maxGrowingIter,
            )

        except CutbackRequest as e:
            self.prevTimeStep = None
            self.incrementStatus["iters"] = np.inf
            self.incrementStatus["notes"] = str(e)
            raise IncrementFailed(str(e), max(e.cutbackSize, cutbackFactor)) from e

        except (ReachedMaxIterations, DivergingSolution) as e:
            self.prevTimeStep = None
            self.incrementStatus["iters"] = np.inf
            self.incrementStatus["notes"] = str(e)
            raise IncrementFailed(str(e), cutbackFactor) from e

        self._U, self._dU, self._P = U, dU, P
        self._iterationCounter = iterationCounter

    @performancetiming.timeit("accept increment")
    def acceptIncrement(self, step, model: FEModel, timeStep: TimeStep):
        """Commit the converged increment to the model, and keep the predictor's state; see
        :meth:`~edelweissfe.solvers.base.nonlinearsolverbase.NonlinearSolverBase.acceptIncrement`.

        Parameters
        ----------
        step
            The step being solved.
        model
            The model tree.
        timeStep
            The converged increment.
        """

        U, dU, P = self._U, self._dU, self._P
        iterationCounter = self._iterationCounter

        # After a model change, dU contains the remesh settling transient: optionally do not extrapolate it.
        if self._modelHasChanged and not self.options["extrapolateAfterModelChange"]:
            self.prevTimeStep = None
        else:
            self.prevTimeStep = timeStep
        self.dU = dU
        self.convergenceCriterion.acceptIncrement()

        if iterationCounter >= step.criticalIter:
            step.timeStepper.preventIncrementIncrease()

        with performancetiming.timeit("write results to nodes"):
            for fieldName, field in model.nodeFields.items():
                self.theDofManager.writeDofVectorToNodeField(U, field, "U")
                self.theDofManager.writeDofVectorToNodeField(P, field, "P")
                self.theDofManager.writeDofVectorToNodeField(dU, field, "dU")

            for variable in model.scalarVariables.values():
                variable.value = U[self.theDofManager.idcsOfScalarVariablesInDofVector[variable]]

        self.updateRigidBodies(model, timeStep)

        with performancetiming.timeit("advance model time"):
            model.advanceToTime(timeStep.totalTime)

        self.journal.message(
            "Converged in {:} iteration(s)".format(iterationCounter),
            self.identification,
            1,
        )

        self.incrementStatus["iters"] = iterationCounter
        self.incrementStatus["converged"] = True

    def endStep(self, step, model: FEModel):
        """Report the step's performance timing.

        Parameters
        ----------
        step
            The step that was solved.
        model
            The model tree.
        """

        prettyTable = performancetiming.makePrettyTable(wallTime=perf_counter() - self._stepWallClockTic)
        self.journal.printPrettyTable(prettyTable, self.identification)
        performancetiming.reset()

    def createConvergenceCriterion(self) -> ConvergenceCriterionBase:
        """Create the convergence criterion selected by the option ``convergenceCriterion``.

        Returns
        -------
        ConvergenceCriterionBase
            The convergence criterion.
        """
        criterion = self.options["convergenceCriterion"]

        if criterion == "legacy":
            return LegacyConvergenceCriterion(
                self.fluxResidualTolerances, self.fluxResidualTolerancesAlt, self.fieldCorrectionTolerances
            )

        if criterion == "abaqus":
            return AbaqusConvergenceCriterion(
                residualTolerance=self.options["abaqusResidualTolerance"],
                correctionTolerance=self.options["abaqusCorrectionTolerance"],
                residualToleranceAlternative=self.options["abaqusResidualToleranceAlternative"],
                iterationsForAlternativeTolerance=self.options["abaqusIterationsForAlternativeTolerance"],
                zeroFluxThreshold=self.options["abaqusZeroFluxThreshold"],
                zeroFluxCorrectionTolerance=self.options["abaqusZeroFluxCorrectionTolerance"],
                linearResidualTolerance=self.options["abaqusLinearResidualTolerance"],
                initialTimeAveragedFlux=self.options["abaqusInitialTimeAveragedFlux"],
            )

        raise ValueError("Unknown convergence criterion '{:}', use 'legacy' or 'abaqus'".format(criterion))

    def solveIncrement(
        self,
        U_n: DofVector,
        dU: DofVector,
        P: DofVector,
        K: VIJSystemMatrix,
        stepActions: list,
        model: FEModel,
        timeStep: TimeStep,
        prevTimeStep: TimeStep,
        extrapolation: str,
        maxIter: int,
        maxGrowingIter: int,
    ) -> tuple[DofVector, DofVector, DofVector, int, dict]:
        """Standard Newton-Raphson scheme to solve for an increment.

        Parameters
        ----------
        Un
            The old solution vector.
        dU
            The old solution increment.
        P
            The old reaction vector.
        K
            The system matrix to be used.
        elements
            The dictionary containing all elements.
        stepActions
            The list of active step actions.
        model
            The model tree.
        increment
            The increment.
        lastIncrementSize
            The size of the previous increment.
        extrapolation
            The type of extrapolation to be used.
        maxIter
            The maximum number of iterations to be used.
        maxGrowingIter
            The maximum number of growing residuals until the Newton-Raphson is terminated.

        Returns
        -------
        tuple[DofVector,DofVector,DofVector,int,dict]
            A tuple containing
                - the new solution vector
                - the solution increment
                - the new reaction vector
                - the number of required iterations
                - the history of residuals per field
        """

        iterationCounter = 0
        incrementResidualHistory = dict.fromkeys(self.theDofManager.idcsOfFieldsInDofVector, (0.0, 0))

        elements = model.elements
        constraints = model.constraints

        R = self.theDofManager.constructDofVector()
        F = self.theDofManager.constructDofVector()
        FConstraints = self.theDofManager.constructDofVector()
        PExt = self.theDofManager.constructDofVector()
        U_np = self.theDofManager.constructDofVector()
        ddU = None

        dirichlets = stepActions["dirichlet"].values()
        nodeforces = stepActions["nodeforces"].values()
        distributedLoads = stepActions["distributedload"].values()
        bodyForces = stepActions["bodyforce"].values()

        with performancetiming.timeit("step actions at increment start"):
            # Find which global DOFs the Dirichlet BCs constrain, once up front.
            self.locateConstrainedDofs(dirichlets)

            self.applyStepActionsAtIncrementStart(model, timeStep, stepActions)

        self.initializeIncrement(U_n, stepActions, model, timeStep)

        with performancetiming.timeit("predictor"):
            dU, isExtrapolatedIncrement = self.extrapolateLastIncrement(
                extrapolation, timeStep, dU, dirichlets, prevTimeStep, model
            )

        while True:
            for geostatic in stepActions["geostatic"].values():
                geostatic.applyAtIterationStart()

            with performancetiming.timeit("reset residual and tangent"):
                U_np[:] = U_n
                U_np += dU

                P[:] = F[:] = FConstraints[:] = PExt[:] = 0.0
                # K holds the entries of all element matrices, far too many to zero on a single thread
                forEachSliceInParallel(lambda s: K[s].fill(0.0), len(K))

            P, K, F = self.computeElements(elements, U_np, dU, P, K, F, timeStep)
            PExt, K = self.assembleLoads(nodeforces, distributedLoads, bodyForces, U_np, PExt, K, timeStep)
            PExt, K = self.assembleConstraints(constraints, U_np, dU, PExt, K, timeStep, FConstraints)

            R[:] = -P
            R += PExt

            self.assembleAdditionalTerms(dU, R, F, K, timeStep)

            # Condense BEFORE the Dirichlet handling: T^T folds slave rows into master rows, which may
            # carry a prescribed delta.
            if self.mpcTransformation is not None:
                R[:] = self.mpcTransformation.transformResidual(R, dU)

            # --- Impose the Dirichlet (prescribed-value) boundary conditions ---
            # For each constrained DOF i we overwrite its row of the linearized system
            # K ddU = R  so that the linear solve returns a *known* value for the
            # increment ddU[i]:
            #     R: set R[i] to the value ddU[i] must take
            #     K: zero row i, set K[i, i] = 1, and eliminate column i: R[r] -= K[r, i] R[i]
            #        for every other row r, then K[r, i] = 0   (see applyDirichletToStiffness)
            # Together these give  ddU[i] = R[i]  exactly, and keep a symmetric K symmetric.
            if iterationCounter == 0 and not isExtrapolatedIncrement and dirichlets:
                # First iteration: the constrained DOFs still move by their prescribed increment.
                R = self.applyDirichletToResidual(timeStep, R, dirichlets)
            else:
                # Later iterations: zero increment, which also removes them from the convergence check.
                for dirichlet in dirichlets:
                    R[dirichlet.constrainedDofIndices] = 0.0

                converged, nodesWithLargestResidual = self.checkConvergence(
                    R, ddU, dU, F, FConstraints, iterationCounter, incrementResidualHistory
                )

                if converged:
                    break

                if self.checkDivergingSolution(incrementResidualHistory, maxGrowingIter):
                    self.printResidualOutlierNodes(nodesWithLargestResidual)
                    raise DivergingSolution("Residual grew {:} times, cutting back".format(maxGrowingIter))

                if iterationCounter == maxIter:
                    self.printResidualOutlierNodes(nodesWithLargestResidual)
                    raise ReachedMaxIterations("Reached max. iterations in current increment, cutting back")

            K_ = self.assembleStiffnessCSR(K)

            if self.mpcTransformation is not None:
                K_ = self.mpcTransformation.transformSystemMatrix(K_)

            # identity rows, and the columns eliminated into R
            K_ = self.applyDirichletToStiffness(K_, dirichlets, R)

            ddU = self.linearSolve(K_, R)

            # add the master-interpolated part at the slave DOFs
            if self.mpcTransformation is not None:
                ddU = self.mpcTransformation.recoverSlaveSolution(ddU)

            dU += ddU
            iterationCounter += 1

        self.finalizeIncrement(model)

        return U_np, dU, P, iterationCounter, incrementResidualHistory

    def initializeIncrement(self, U_n: DofVector, stepActions: dict, model: FEModel, timeStep: TimeStep):
        """Prepare what :meth:`assembleAdditionalTerms` needs during this increment's Newton
        iterations. Called by :meth:`solveIncrement` once per increment, after the step actions have
        been applied at the increment's start and before the increment is extrapolated.

        A static analysis needs nothing here. A dynamic one, for example, assembles its mass matrix
        here and computes the terms the time integration adds to the stiffness for this time
        increment -- see :class:`~edelweissfe.solvers.nonlinearimplicitdynamic.NonlinearImplicitDynamic`.

        Parameters
        ----------
        U_n
            The solution at the start of the increment.
        stepActions
            The active step actions.
        model
            The model tree.
        timeStep
            The time step.
        """

    def assembleAdditionalTerms(
        self, dU: DofVector, R: DofVector, F: DofVector, K: VIJSystemMatrix, timeStep: TimeStep
    ):
        """Assemble terms of the equation system beyond the internal and external forces. Called
        by :meth:`solveIncrement` in every Newton iteration, next to :meth:`computeElements`,
        :meth:`assembleLoads` and :meth:`assembleConstraints`: right after the residual
        :math:`R = P_\\mathrm{ext} - P_\\mathrm{int}` is formed, and before the
        multi-point-constraint condensation, the Dirichlet handling, the convergence test and the
        linear solve -- which therefore all act on what is added here.

        A static analysis has no such terms. A dynamic one, for example, subtracts the inertia force
        :math:`M \\ddot{u}` from the residual and adds the corresponding mass term to the stiffness
        -- see :class:`~edelweissfe.solvers.nonlinearimplicitdynamic.NonlinearImplicitDynamic`.

        Parameters
        ----------
        dU
            The current trial solution increment.
        R
            The residual, to be augmented in place.
        F
            The reference flux scale of the convergence test, to be augmented in place.
        K
            The tangent in VIJ layout, to be augmented in place.
        timeStep
            The time step.
        """

    def finalizeIncrement(self, model: FEModel):
        """Store what the converged increment leaves behind. Called by :meth:`solveIncrement` once
        its Newton loop has converged -- only for an accepted increment, since a failing one leaves
        by exception before it.

        A static analysis stores nothing here. A dynamic one, for example, stores the velocity and
        acceleration of the converged increment, as the starting point of the next one.

        Parameters
        ----------
        model
            The model tree.
        """

    def _shouldPruneCondensedMatrixZeros(self) -> bool:
        """Whether to compact exact zeros out of the multi-point-condensed system matrix before this
        solve -- see ``pruneCondensedMatrixZeros`` (:class:`NISTSchema`). An explicit True/False in
        the input file always wins; left unset (``None``), the currently active ``self.linSolver``
        decides, polymorphically, via
        :attr:`~edelweissfe.linsolve.base.LinearSolver.needsStableSparsityPattern`: pruned unless that
        solver says it needs the pattern stable across iterations."""
        explicit = self.options["pruneCondensedMatrixZeros"]
        if explicit is not None:
            return explicit
        return not self.linSolver.needsStableSparsityPattern

    @performancetiming.timeit("dirichlet K on CSR")
    def applyDirichletToStiffness(self, K: csr_matrix, dirichlets: list[StepActionBase], rhs=None) -> csr_matrix:
        """Impose the Dirichlet BCs on the system matrix -- and, given the right-hand side(s) ``rhs``
        (modified in place), eliminate the constrained DOFs from the columns as well; see
        :func:`edelweissfe.solvers.base.dirichlet.applyDirichletToStiffness`."""
        K = applyDirichletToStiffness(K, dirichlets, rhs)

        # Pruning zeros is only safe on the disposable MPC-condensed matrix: without MPC, K is the
        # CSRGenerator's persistent buffer, which a fixed assembly map writes into every iteration.
        # PARDISO needs the pruning on condensed systems; solvers that reuse a stable pattern do not
        # (see _shouldPruneCondensedMatrixZeros).
        if self.mpcTransformation is not None and self._shouldPruneCondensedMatrixZeros():
            K.eliminate_zeros()

        return K

    @performancetiming.timeit("elements")
    def computeElements(
        self,
        elements: list,
        U_np: DofVector,
        dU: DofVector,
        P: DofVector,
        K: VIJSystemMatrix,
        F: DofVector,
        timeStep: TimeStep,
    ) -> tuple[DofVector, VIJSystemMatrix, DofVector]:
        """Loop over all elements, and evalute them.
        Is called by solveIncrement() in each iteration.

        Parameters
        ----------
        elements
            The list of finite elements.
        U_np
            The current solution vector.
        dU
            The current solution increment vector.
        P
            The reaction vector.
        K
            The system matrix.
        F
            The vector of accumulated fluxes for convergence checks.
        timeStep
            The time step.

        Returns
        -------
        tuple[DofVector,VIJSystemMatrix,DofVector]
            - The modified reaction vector.
            - The modified system matrix.
            - The modified accumulated flux vector.
        """

        time = timeStep.totalTime
        dT = timeStep.timeIncrement

        for el in elements.values():
            Ke = K[el]
            Pe = np.zeros(el.nDof)

            el.computeKernels(Ke, Pe, U_np[el], dU[el], time, dT)

            P[el] += Pe
            F[el] += abs(Pe)

        return P, K, F

    @performancetiming.timeit("assemble constraints")
    def assembleConstraints(
        self,
        constraints: list[ConstraintBase],
        U_np: DofVector,
        dU: DofVector,
        PExt: DofVector,
        K: VIJSystemMatrix,
        timeStep: TimeStep,
        FConstraints: DofVector = None,
    ) -> tuple[DofVector, VIJSystemMatrix]:
        """Loop over all constraints, and evaluate them.
        Is called by solveIncrement() in each iteration.

        Parameters
        ----------
        constraints
            The list of constraints.
        U_np
            The current solution vector.
        dU
            The current solution increment vector.
        PExt
            The external load vector.
        K
            The system matrix.
        timeStep
            The time step.
        FConstraints
            The vector of accumulated absolute constraint fluxes for convergence checks, if given.

        Returns
        -------
        tuple[DofVector,VIJSystemMatrix]
            - The modified external load vector.
            - The modified system matrix.
        """

        for constraint in constraints.values():
            Kc = K[constraint]
            Pc = np.zeros(constraint.nDof)

            constraint.applyConstraint(U_np[constraint], dU[constraint], Pc, Kc, timeStep)

            # instead of PExt[constraint] += Pe, np.add.at allows for repeated indices
            np.add.at(PExt, PExt.entitiesInDofVector[constraint], Pc)
            if FConstraints is not None:
                np.add.at(FConstraints, FConstraints.entitiesInDofVector[constraint], np.abs(Pc))

        return PExt, K
