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

from collections.abc import Callable
from dataclasses import dataclass
from time import perf_counter

import numpy as np
from scipy.sparse import csr_matrix

import edelweissfe.utils.performancetiming as performancetiming
from edelweissfe.config.linsolve import createLinearSolver
from edelweissfe.constraints.base.constraintbase import ConstraintBase
from edelweissfe.models.femodel import FEModel
from edelweissfe.numerics.csrgeneratorv2 import CSRGenerator
from edelweissfe.numerics.dofmanager import DofVector, VIJSystemMatrix
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


class QuasiNewtonFallback:
    """The quasi-Newton fallback of one attempt of an increment (solver option ``fallBackToQuasiNewton``): once the
    residual has grown more often than the step option ``maxGrowIter`` allows, the increment continues with the tangent
    of its first iteration instead of being cut back. A tangent that is not updated converges only linearly, but it
    cannot switch abruptly either, as a tangent of materials switching between loading and unloading does.

    Parameters
    ----------
    isEnabled
        Whether the option is on; only then is the tangent of the first iteration kept.
    """

    def __init__(self, isEnabled: bool):
        self.isEnabled = isEnabled
        #: True once the increment continues with the tangent of its first iteration.
        self.isEngaged = False
        #: The tangent of the first iteration, in CSR layout and condensed, without the Dirichlet rows and columns.
        self.tangentOfFirstIteration = None


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
        description=(
            "The prediction a new increment starts its Newton iterations from (off|linear|quadratic): none, the "
            "last accepted increment scaled to the new size, or the parabola through the last three accepted states "
            "(linear while only one accepted increment of the step is available)."
        ),
        dtype=str,
        default="linear",
    )
    retryWithoutExtrapolation: bool | None = schemaField(
        description=(
            "If an extrapolated increment fails, attempt it once more from a zero prediction before cutting it back. "
            "A failed increment then also keeps the accepted history the next attempt extrapolates from."
        ),
        dtype=bool,
        default=False,
    )
    extrapolateAcrossConnectivityChange: bool | None = schemaField(
        description=(
            "Keep extrapolating across a rebuild of the equation system caused by a change of the connectivity alone "
            "(e.g., a contact search), as long as the rebuilt system numbers the degrees of freedom as before. Off: "
            "every rebuild starts the increment from a zero prediction."
        ),
        dtype=bool,
        default=False,
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
    linsolver: str | None = schemaField(
        description=(
            "The linear solver: the name of a *linearSolver block, or a type of linear solver with its defaults (e.g. "
            "pardiso, superlu, blockamg)."
        ),
        dtype=str,
        default="pardiso",
    )
    pruneCondensedMatrixZeros: bool | None = schemaField(
        description=(
            "Compact explicitly stored zeros out of the multi-point-constraint-condensed system "
            "matrix before solving. Pruning removes whichever entries happen to be exactly zero on "
            "this particular iteration, so the pattern changes on almost every Newton iteration -- "
            "which defeats any linear solver that relies on a stable pattern across iterations (e.g. "
            "blockamg reusing a per-field AMG hierarchy built for the old one). Left unset (the "
            "default), the choice is automatic: pruned if the linear solver in force reports that it "
            "needs it "
            "(:attr:`~edelweissfe.linsolve.base.LinearSolver.needsPrunedSystemMatrix`), which a "
            "direct factorization does -- PARDISO's reordering is sensitive to the extra structural "
            "entries on these path-dependent condensed systems, and keeping them has been observed to "
            "drift from the converged reference path -- and a Krylov solver with an AMG preconditioner "
            "does not: it gives the same iterations with and without the stored zeros, and pruning "
            "costs it a copy of the matrix and a pass over it in every Newton iteration. An explicit "
            "True/False here always wins over the automatic choice; turn pruning off for a direct "
            "solver only after verifying the load path against a reference run."
        ),
        dtype=bool,
        default=None,
        documentedDefault="auto",
    )
    useThreadedMPCCondensation: bool | None = schemaField(
        description=(
            "Condense the multi-point-constraint system matrix, T^T K T + C, in place, with an "
            "OpenMP-threaded kernel, instead of with SciPy's single-threaded CSR routines, which build a "
            "second matrix. The CSR pattern of K gains, once per equation system, the entries where a "
            "master collects the couplings of its slaves; the result does not depend on the number of "
            "threads. Explicit zeros (the slave columns among them) are pruned from a copy according to "
            "pruneCondensedMatrixZeros. Off by default."
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
    lineSearch: bool | None = schemaField(
        description=(
            "Scale the Newton correction by the step length at which a parabola, fitted to the largest residual "
            "evaluated at the trial step lengths lineSearchStepLengths, is smallest (a quadratic line search)."
        ),
        dtype=bool,
        default=False,
    )
    lineSearchAfterIterations: int | None = schemaField(
        description=(
            "The line search starts after this many iterations of an increment, at least 0: the first correction of an "
            "increment, which makes the prescribed increments, is never scaled."
        ),
        dtype=int,
        default=5,
    )
    lineSearchEveryIterations: int | None = schemaField(
        description="From then on, the line search is done in every iteration whose number is a multiple of this.",
        dtype=int,
        default=2,
    )
    lineSearchStepLengths: str | None = schemaField(
        description="The trial step lengths of the line search, comma-separated, at least three.",
        dtype=str,
        default="0.8, 1.0, 1.2",
    )
    fallBackToQuasiNewton: bool | None = schemaField(
        description=(
            "Instead of cutting an increment back once the residual grew more often than maxGrowIter allows, "
            "continue it with the tangent of its first iteration (a quasi-Newton method); maxIter still applies."
        ),
        dtype=bool,
        default=False,
    )
    convergenceCriterion: str | None = schemaField(
        description=(
            "The convergence criterion of the Newton iterations, legacy (the original EdelweissFE one) or abaqus "
            "(the one of Abaqus/Standard); see edelweissfe.solvers.base.convergencecriteria."
        ),
        dtype=str,
        default="legacy",
    )
    iterationsForAlternativeTolerance: int | None = schemaField(
        description=(
            "legacy criterion: I, the iteration from which on the flux residual is checked against "
            "fluxResidualToleranceAlternative instead of fluxResidualTolerance."
        ),
        dtype=int,
        default=15,
    )
    zeroFluxThreshold: float | None = schemaField(
        description=(
            "legacy criterion: a field whose spatial average flux does not exceed this value carries no flux its "
            "residual could be compared with, so its flux residual counts as converged; needs minimumIterations >= 1, "
            "and refuses a smaller one. Off if not set."
        ),
        dtype=float,
        default=None,
        documentedDefault="off",
    )
    zeroIncrementThreshold: float | None = schemaField(
        description=(
            "legacy criterion: a field whose largest increment does not exceed this value has not moved, so there is "
            "no increment to relate its correction to, and its correction counts as converged. Off if not set."
        ),
        dtype=float,
        default=None,
        documentedDefault="off",
    )
    minimumIterations: int | None = schemaField(
        description=(
            "legacy criterion: no flux residual counts as converged before this iteration, so an increment takes at "
            "least this many Newton iterations; 0 accepts a prediction that already satisfies the criterion."
        ),
        dtype=int,
        default=0,
    )
    countFirstResidualAsGrowth: bool | None = schemaField(
        description=(
            "Residual growths (the step option maxGrowIter): the residual history of a field starts from zero, so the "
            "first checked residual of an increment counts as a growth; if False, counting starts with the second."
        ),
        dtype=bool,
        default=True,
    )
    countGrowthOfConvergedFields: bool | None = schemaField(
        description=(
            "Residual growths (the step option maxGrowIter): count a growth of the flux residual of a field also in "
            "an iteration in which this field is converged and only other fields keep the iterations going."
        ),
        dtype=bool,
        default=True,
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
        "retryWithoutExtrapolation": False,
        "extrapolateAcrossConnectivityChange": False,
        "extrapolateAfterModelChange": True,
        "equilibrateAfterModelChange": False,
        "linsolver": "",
        "pruneCondensedMatrixZeros": None,
        "useThreadedMPCCondensation": False,
        "report-performance-frequency": 0,
        "lineSearch": False,
        "lineSearchAfterIterations": 5,
        "lineSearchEveryIterations": 2,
        "lineSearchStepLengths": "0.8, 1.0, 1.2",
        "fallBackToQuasiNewton": False,
        "convergenceCriterion": "legacy",
        "iterationsForAlternativeTolerance": 15,
        "zeroFluxThreshold": None,
        "zeroIncrementThreshold": None,
        "minimumIterations": 0,
        "countFirstResidualAsGrowth": True,
        "countGrowthOfConvergedFields": True,
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

    #: The predictor's state between increments: the last accepted increment and its dU, and, for a quadratic
    #: extrapolation, the accepted increment before it and its dU.
    checkpointedState = {
        "prevTimeStep": TimeStep,
        "dU": np.ndarray,
        "timeStepBeforePrev": TimeStep,
        "dUBeforePrev": np.ndarray,
    }

    def __init__(self, jobInfo, journal, **kwargs):
        self.journal = journal

        # the tolerances of the legacy criterion per field, see edelweissfe.solvers.base.convergencecriteria
        self.fluxResidualTolerances = jobInfo["fluxResidualTolerance"]
        self.fluxResidualTolerancesAlternative = jobInfo["fluxResidualToleranceAlternative"]
        self.absoluteFluxResidualTolerances = jobInfo["absoluteFluxResidualTolerance"]
        self.fieldCorrectionTolerances = jobInfo["fieldCorrectionTolerance"]
        self.relativeFieldCorrectionTolerances = jobInfo["relativeFieldCorrectionTolerance"]

        self.options = self.SolverSpecificOptions.copy()
        # the *solver datalines belong only to this solver: unknown entries are typos
        self._updateOptions(kwargs, journal, strict=True)
        #: The ``*linearSolver`` blocks of the input file by name, which linsolver= may name (none without an input file).
        self.linearSolverDefinitions = jobInfo.get("linearSolvers", {})
        #: The linear solver, and the ``linsolver`` it was made for (see :meth:`selectLinearSolverForStep`).
        self.linSolver = None
        self.linSolverName = None

        #: The last accepted increment, from which the predictor extrapolates (None: no extrapolation).
        self.prevTimeStep = None
        #: The solution increment of the last accepted increment.
        self.dU = None
        #: The accepted increment before :attr:`prevTimeStep`, kept for a quadratic extrapolation only (else None).
        self.timeStepBeforePrev = None
        #: The solution increment of :attr:`timeStepBeforePrev`.
        self.dUBeforePrev = None

    def selectLinearSolverForStep(self) -> None:
        """Make :attr:`linSolver` the linear solver the step about to begin names (``linsolver``): the one of the
        previous step if it names the same, else a new one.

        A kept solver builds anew for the step's equation system (``setModel``) and frees what it built before. A new
        solver per step would leave the previous one to Python's cyclic garbage collector whenever its tree refers
        back to itself (a function in a solver script, whose globals hold the tree), and the steps' matrices would
        pile up."""
        if self.linSolver is not None and self.options["linsolver"] == self.linSolverName:
            return
        # the previous one goes before the new one is made
        self.linSolver = None
        self.linSolver = createLinearSolver(self.options["linsolver"], self.linearSolverDefinitions)
        self.linSolverName = self.options["linsolver"]
        self.linSolver.setJournal(self.journal)

    def forgetEquationSystem(self) -> None:
        """Drop the equation system of the last build -- the DOF manager, the system matrix and its CSR generator,
        the multi-point-constraint condensation, the DOF vectors -- so that it is gone before the next one is built,
        never both at once."""
        self.theDofManager = None
        self.csrGenerator = None
        self.mpcTransformation = None
        self._U = self._dU = self._P = self._K = None

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
        self.selectLinearSolverForStep()

        convergenceCriterion = self.createConvergenceCriterion()
        convergenceCriterion.startStep(self.convergenceCriterion)
        self.convergenceCriterion = convergenceCriterion

        # The equation system (DofManager, VIJ pattern, CSR structure) is (re)built lazily in the first increment,
        # and whenever the topology update reports a change of the mesh or of a constraint's DOF footprint.
        self.forgetEquationSystem()

        if self.options["extrapolation"] not in ("off", "linear", "quadratic"):
            raise ValueError(
                "Unknown extrapolation '{:}', use off, linear or quadratic".format(self.options["extrapolation"])
            )

        if self.options["lineSearch"]:
            stepLengths = self.lineSearchStepLengths()
            if len(set(stepLengths)) < 3 or min(stepLengths) <= 0.0:
                raise ValueError(
                    "lineSearchStepLengths: a parabola needs at least three different positive step lengths, got "
                    "{:}".format(stepLengths)
                )
            # The first correction of an increment makes the prescribed increments, which a line search would scale.
            if self.options["lineSearchAfterIterations"] < 0:
                raise ValueError("lineSearchAfterIterations must not be negative")
            if self.options["lineSearchEveryIterations"] < 1:
                raise ValueError("lineSearchEveryIterations must be at least 1")

        # A step starts without a predictor state; a resumed step takes over the checkpointed one.
        self.prevTimeStep = None
        self.dU = None
        self.timeStepBeforePrev = None
        self.dUBeforePrev = None

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
            # The prediction may survive a rebuild caused by the connectivity alone, if the rebuilt system numbers the
            # degrees of freedom as the last one: remember that numbering, before the last system is dropped.
            numberingBeforeRebuild = (
                self.theDofManager.dofNumbering()
                if connectivityHasChanged
                and not modelHasChanged
                and self.theDofManager is not None
                and self.options["extrapolateAcrossConnectivityChange"]
                else None
            )
            with performancetiming.timeit("equation system rebuild"):
                self.forgetEquationSystem()
                with performancetiming.timeit("dof manager"):
                    self.theDofManager = self.createDofManager(model)
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

                U = self.theDofManager.writeScalarVariablesToDofVector(U)

                with performancetiming.timeit("mpc transformation build"):
                    self.mpcTransformation = self.buildMPCTransformation(model, step.actions)
                with performancetiming.timeit("mpc dirichlet conflict check"):
                    self.checkMPCDirichletConflicts(self.mpcTransformation, step.actions)
                if self.mpcTransformation is not None:
                    # Room in the CSR pattern for the entries the condensation adds, so that it can be done in place
                    with performancetiming.timeit("mpc condensed pattern"):
                        self.csrGenerator.addToPattern(
                            self.mpcTransformation.condensedPatternFillIn(self.csrGenerator.csrMatrix)
                        )

                # After a topology change, the last dU belongs to a different DOF layout: do not extrapolate this
                # increment. After a change of the connectivity alone, it may still fit.
                if modelHasChanged or (
                    connectivityHasChanged and not self.predictionSurvivesRebuild(numberingBeforeRebuild)
                ):
                    self.prevTimeStep = None
                self._U, self._dU, self._P, self._K = U, dU, P, K
        # the equation system of this increment, just built or kept from the last one
        U, dU, P, K = self._U, self._dU, self._P, self._K

        self.incrementStatus = {
            "step": step.number,
            "inc": timeStep.number,
            "iters": None,
            "converged": False,
            "time inc": timeStep.timeIncrement,
            "time end": timeStep.totalTime,
            "notes": "",
        }

        self.tableHeaderWritten = False

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
            self.theDofManager.writeDofVectorToScalarVariables(U)
            model.advanceToTime(equilibrationTimeStep.totalTime)
            self._U, self._dU, self._P = U, dU, P

        retryWithoutExtrapolation = self.options["retryWithoutExtrapolation"]

        try:
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
            except (CutbackRequest, ReachedMaxIterations, DivergingSolution) as e:
                if not (retryWithoutExtrapolation and self.extrapolates(extrapolation, self.prevTimeStep)):
                    raise
                # The prediction may have led Newton's method astray: the same increment, from a zero prediction.
                self.journal.message(
                    "{:}; retrying the increment without extrapolation".format(str(e)), self.identification, 1
                )
                U, dU, P, iterationCounter, incrementResidualHistory = self.solveIncrement(
                    U, dU, P, K, step.actions, model, timeStep, None, extrapolation, maxIter, maxGrowingIter
                )
                self.incrementStatus["notes"] = "retried without extrapolation"

        except CutbackRequest as e:
            self.forgetPredictionAfterFailure()
            self.incrementStatus["iters"] = np.inf
            self.incrementStatus["notes"] = str(e)
            raise IncrementFailed(str(e), max(e.cutbackSize, cutbackFactor)) from e

        except (ReachedMaxIterations, DivergingSolution) as e:
            self.forgetPredictionAfterFailure()
            self.incrementStatus["iters"] = np.inf
            self.incrementStatus["notes"] = str(e)
            raise IncrementFailed(str(e), cutbackFactor) from e

        self._U, self._dU, self._P = U, dU, P
        self._iterationCounter = iterationCounter

    def predictionSurvivesRebuild(self, numberingBeforeRebuild: tuple | None) -> bool:
        """Whether the last accepted increment still fits the equation system rebuilt after a change of the
        connectivity alone (e.g., a contact search), so that this increment may extrapolate it: only with
        ``extrapolateAcrossConnectivityChange``, and only if the rebuilt system numbers the degrees of freedom as the
        last one. At the first increment of a resumed step there is no last system, and the numbering of the checkpoint
        is assumed, as for every resumed prediction.

        Parameters
        ----------
        numberingBeforeRebuild
            The :meth:`~edelweissfe.numerics.dofmanager.DofManager.dofNumbering` of the last equation system, or None
            if there was none.

        Returns
        -------
        bool
            True if the prediction may extrapolate the last accepted increment.
        """

        if not self.options["extrapolateAcrossConnectivityChange"]:
            return False

        return numberingBeforeRebuild is None or self.theDofManager.dofNumbering() == numberingBeforeRebuild

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

        # The predictor's history: a quadratic extrapolation needs the accepted increment before the last one, too.
        if self.options["extrapolation"] == "quadratic" and self.prevTimeStep is not None:
            self.timeStepBeforePrev, self.dUBeforePrev = self.prevTimeStep, self.dU
        else:
            self.timeStepBeforePrev, self.dUBeforePrev = None, None
        # After a model change, dU contains the remesh settling transient: optionally do not extrapolate it.
        if self._modelHasChanged and not self.options["extrapolateAfterModelChange"]:
            self.prevTimeStep = None
        else:
            self.prevTimeStep = timeStep
        # A copy: the next increment iterates on dU in place, and may fail.
        self.dU = dU.copy()
        self.convergenceCriterion.acceptIncrement()

        if iterationCounter >= step.criticalIter:
            step.timeStepper.preventIncrementIncrease()

        with performancetiming.timeit("write results to nodes"):
            for fieldName, field in model.nodeFields.items():
                self.theDofManager.writeDofVectorToNodeField(U, field, "U")
                self.theDofManager.writeDofVectorToNodeField(P, field, "P")
                self.theDofManager.writeDofVectorToNodeField(dU, field, "dU")

            self.theDofManager.writeDofVectorToScalarVariables(U)

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

        legacyCriterion = LegacyConvergenceCriterion(
            fluxResidualTolerances=self.fluxResidualTolerances,
            fluxResidualTolerancesAlternative=self.fluxResidualTolerancesAlternative,
            absoluteFluxResidualTolerances=self.absoluteFluxResidualTolerances,
            fieldCorrectionTolerances=self.fieldCorrectionTolerances,
            relativeFieldCorrectionTolerances=self.relativeFieldCorrectionTolerances,
            iterationsForAlternativeTolerance=self.options["iterationsForAlternativeTolerance"],
            zeroFluxThreshold=self.options["zeroFluxThreshold"],
            zeroIncrementThreshold=self.options["zeroIncrementThreshold"],
            minimumIterations=self.options["minimumIterations"],
        )

        if criterion == "legacy":
            return legacyCriterion

        if criterion == "abaqus":
            return AbaqusConvergenceCriterion(
                scalarVariableCriterion=legacyCriterion,
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

    def forgetPredictionAfterFailure(self):
        """A failed increment drops the prediction, which may have caused the failure, unless
        ``retryWithoutExtrapolation`` has answered that already: then the accepted history is kept, and the smaller
        increment that follows the cutback extrapolates it."""

        if not self.options["retryWithoutExtrapolation"]:
            self.prevTimeStep = None

    def extrapolates(self, extrapolation: str, prevTimeStep: TimeStep | None) -> bool:
        """Whether an increment that follows the accepted increment ``prevTimeStep`` starts from an extrapolated
        prediction.

        Parameters
        ----------
        extrapolation
            The type of extrapolation, ``off``, ``linear`` or ``quadratic``.
        prevTimeStep
            The last accepted increment, or None if there is none to extrapolate.

        Returns
        -------
        bool
            True if the increment is extrapolated.
        """

        return extrapolation != "off" and prevTimeStep is not None and bool(prevTimeStep.timeIncrement)

    def extrapolateLastIncrement(
        self,
        extrapolation: str,
        timeStep: TimeStep,
        dU: DofVector,
        dirichlets: list,
        prevTimeStep: TimeStep,
        model,
    ) -> tuple[DofVector, bool]:
        """Predict the solution increment, the point Newton's method starts from, by extrapolating the last
        accepted increments of the step. With :math:`h` the size (step progress) of the new increment and
        :math:`h_n, \\Delta U_n` the size and the solution increment of the last accepted one:

        * ``off``: no prediction, :math:`\\Delta U = 0`.
        * ``linear``: the last increment, scaled to the new size, :math:`\\Delta U = \\Delta U_n \\, h / h_n`.
        * ``quadratic``: the parabola through the last three accepted states, evaluated at the end of the new
          increment. With the increment before the last one, :math:`h_{n-1}, \\Delta U_{n-1}`,

          .. math::

              \\Delta U = \\Delta U_n \\frac{h}{h_n}
              + \\left( \\frac{\\Delta U_n}{h_n} - \\frac{\\Delta U_{n-1}}{h_{n-1}} \\right)
              \\frac{h \\, (h + h_n)}{h_{n-1} + h_n},

          the linear prediction plus the change of the slope of the path. It reproduces a quadratic path exactly. While
          only one accepted increment of the step is available, the prediction is linear.

        The prescribed degrees of freedom take their prescribed increment.

        Parameters
        ----------
        extrapolation
            The type of extrapolation.
        timeStep
            The new increment.
        dU
            The solution increment to predict, overwritten.
        dirichlets
            The active dirichlet boundary conditions.
        prevTimeStep
            The last accepted increment, or None for no prediction.
        model
            The model tree.

        Returns
        -------
        tuple[DofVector,bool]
            - The predicted solution increment.
            - True if an extrapolation was performed.
        """

        if not self.extrapolates(extrapolation, prevTimeStep):
            dU[:] = 0.0
            return dU, False

        weightOfLast, weightOfBeforeLast = self.extrapolationWeights(extrapolation, timeStep, prevTimeStep)
        np.multiply(self.dU, weightOfLast, out=dU)
        if weightOfBeforeLast is not None:
            dU += weightOfBeforeLast * self.dUBeforePrev

        dU = self.applyDirichletToResidual(timeStep, dU, dirichlets)

        return dU, True

    def extrapolationWeights(
        self, extrapolation: str, timeStep: TimeStep, prevTimeStep: TimeStep
    ) -> tuple[float, float | None]:
        """The weights of the last accepted increment and of the one before it in the prediction of
        :meth:`extrapolateLastIncrement`: :math:`\\Delta U = w_n \\, \\Delta U_n + w_{n-1} \\, \\Delta U_{n-1}`, with

        .. math::

            w_n = \\frac{h}{h_n} + \\frac{c}{h_n}, \\qquad w_{n-1} = -\\frac{c}{h_{n-1}}, \\qquad
            c = \\frac{h \\, (h + h_n)}{h_{n-1} + h_n}

        for a quadratic extrapolation, and :math:`w_n = h / h_n` alone for a linear one. Every quantity that follows
        the path of the solution is predicted with these weights, e.g. the load parameter of the arc length method,
        too.

        Parameters
        ----------
        extrapolation
            The type of extrapolation, ``linear`` or ``quadratic``.
        timeStep
            The new increment.
        prevTimeStep
            The last accepted increment, which the prediction extrapolates.

        Returns
        -------
        tuple[float,float|None]
            - The weight of the last accepted increment.
            - The weight of the accepted increment before it, None for a linear prediction.
        """

        h = timeStep.stepProgressIncrement
        h_n = prevTimeStep.stepProgressIncrement

        if extrapolation == "quadratic" and self.extrapolates(extrapolation, self.timeStepBeforePrev):
            h_nm1 = self.timeStepBeforePrev.stepProgressIncrement
            curvatureFactor = h * (h + h_n) / (h_nm1 + h_n)
            return h / h_n + curvatureFactor / h_n, -(curvatureFactor / h_nm1)

        return h / h_n, None

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
        incrementResidualHistory = {field: [] for field in self.theDofManager.idcsOfFieldsInDofVector}

        R = self.theDofManager.constructDofVector()
        F = self.theDofManager.constructDofVector()
        FConstraints = self.theDofManager.constructDofVector()
        PLoadsAndConstraints = self.theDofManager.constructDofVector()
        U_np = self.theDofManager.constructDofVector()
        ddU = None

        dirichlets = stepActions["dirichlet"].values()

        with performancetiming.timeit("step actions at increment start"):
            # Find which global DOFs the Dirichlet BCs constrain, once up front.
            self.locateConstrainedDofs(dirichlets)

            self.applyStepActionsAtIncrementStart(model, timeStep, stepActions)

        self.initializeIncrement(U_n, stepActions, model, timeStep)

        with performancetiming.timeit("predictor"):
            dU, isExtrapolatedIncrement = self.extrapolateLastIncrement(
                extrapolation, timeStep, dU, dirichlets, prevTimeStep, model
            )

        quasiNewton = QuasiNewtonFallback(self.options["fallBackToQuasiNewton"])

        while True:
            P, K, F = self.assembleResidualAndTangent(
                U_n, dU, U_np, R, P, PLoadsAndConstraints, F, FConstraints, K, stepActions, model, timeStep
            )

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

                self.checkDivergingSolutionOrFallBack(
                    quasiNewton, incrementResidualHistory, maxGrowingIter, nodesWithLargestResidual
                )

                if iterationCounter == maxIter:
                    self.printResidualOutlierNodes(nodesWithLargestResidual)
                    raise ReachedMaxIterations("Reached max. iterations in current increment, cutting back")

            K_ = self.assembleSystemMatrix(K, quasiNewton)

            # identity rows, and the columns eliminated into R
            K_ = self.applyDirichletToStiffness(K_, dirichlets, R)

            ddU = self.linearSolve(K_, R)

            # add the master-interpolated part at the slave DOFs
            if self.mpcTransformation is not None:
                ddU = self.mpcTransformation.recoverSlaveSolution(ddU)

            if self.lineSearchIsDue(iterationCounter):

                def largestResidualAt(stepLength):
                    # the trial solution increment along the correction, without the Dirichlet degrees of freedom
                    self.assembleResidualAndTangent(
                        U_n,
                        dU + stepLength * ddU,
                        U_np,
                        R,
                        P,
                        PLoadsAndConstraints,
                        F,
                        FConstraints,
                        K,
                        stepActions,
                        model,
                        timeStep,
                    )
                    for dirichlet in dirichlets:
                        R[dirichlet.constrainedDofIndices] = 0.0
                    return np.linalg.norm(R, np.inf)

                ddU = self.quadraticLineSearch(largestResidualAt) * ddU

            dU += ddU
            iterationCounter += 1

        self.finalizeIncrement(model)

        return U_np, dU, P, iterationCounter, incrementResidualHistory

    def assembleResidualAndTangent(
        self,
        U_n: DofVector,
        dU: DofVector,
        U_np: DofVector,
        R: DofVector,
        P: DofVector,
        PLoadsAndConstraints: DofVector,
        F: DofVector,
        FConstraints: DofVector,
        K: VIJSystemMatrix,
        stepActions: dict,
        model: FEModel,
        timeStep: TimeStep,
    ) -> tuple[DofVector, VIJSystemMatrix, DofVector]:
        """Evaluate the model at the trial solution :math:`U_{n+1} = U_n + \\Delta U`: the right-hand side
        :math:`R = f_\\mathrm{ext} - f_\\mathrm{int}` (see :ref:`residual-and-tangent-convention`) with the terms of
        :meth:`assembleAdditionalTerms`, condensed by the multi-point constraints, and the tangent. The one place
        where the model is evaluated during the Newton iterations: called by :meth:`solveIncrement` -- and by the
        arc length method's -- once per iteration, and by their line searches once per trial step length. The loads
        are those at the end of the increment, see :meth:`assembleLoadsAtIncrementEnd`. The Dirichlet boundary
        conditions are imposed by the callers.

        Parameters
        ----------
        U_n
            The solution at the start of the increment.
        dU
            The trial solution increment.
        U_np
            The trial solution, overwritten.
        R
            The residual, overwritten.
        P
            The internal forces (the reaction), overwritten.
        PLoadsAndConstraints
            The residual contributions of the loads and the constraints, overwritten.
        F
            The accumulated absolute element fluxes, the reference of the convergence test, overwritten.
        FConstraints
            The accumulated absolute constraint fluxes, overwritten.
        K
            The tangent in VIJ layout, overwritten.
        stepActions
            The active step actions.
        model
            The model tree.
        timeStep
            The time step.

        Returns
        -------
        tuple[DofVector,VIJSystemMatrix,DofVector]
            The internal forces, the tangent and the accumulated element fluxes.
        """

        for geostatic in stepActions["geostatic"].values():
            geostatic.applyAtIterationStart()

        with performancetiming.timeit("reset residual and tangent"):
            U_np[:] = U_n
            U_np += dU

            P[:] = F[:] = FConstraints[:] = PLoadsAndConstraints[:] = 0.0
            # K holds the entries of all element matrices, far too many to zero on a single thread
            forEachSliceInParallel(lambda s: K[s].fill(0.0), len(K))

        # Every element, load and constraint adds its residual contribution and the tangent of it.
        # The elements' sum P is kept apart: it is the reaction force this increment reports.
        P, K, F = self.computeElements(model.elements, U_np, dU, P, K, F, timeStep)
        PLoadsAndConstraints, K = self.assembleLoadsAtIncrementEnd(
            stepActions, U_np, dU, PLoadsAndConstraints, K, timeStep
        )
        PLoadsAndConstraints, K = self.assembleConstraints(
            model.constraints, U_np, dU, PLoadsAndConstraints, K, timeStep, FConstraints
        )

        # The right-hand side of K ddU = R is the negative of the summed contributions: the out-of-balance
        # force R = f_ext - f_int.
        R[:] = -P
        R -= PLoadsAndConstraints

        self.assembleAdditionalTerms(dU, R, F, K, timeStep)

        # Condense BEFORE the Dirichlet handling: T^T folds slave rows into master rows, which may
        # carry a prescribed delta.
        if self.mpcTransformation is not None:
            R[:] = self.mpcTransformation.transformResidual(R, dU)

        return P, K, F

    def assembleLoadsAtIncrementEnd(
        self,
        stepActions: dict,
        U_np: DofVector,
        dU: DofVector,
        P: DofVector,
        K: VIJSystemMatrix,
        timeStep: TimeStep,
    ) -> tuple[DofVector, VIJSystemMatrix]:
        """Assemble the residual contributions of the loads acting at the end of the increment, and their tangents;
        called by :meth:`assembleResidualAndTangent`. Here, these are the loads of the step at the time of the end of
        the increment (see
        :meth:`~edelweissfe.solvers.base.nonlinearsolverbase.NonlinearSolverBase.assembleLoads`). The arc length method
        replaces them by the dead load plus the reference load scaled by its load parameter, see
        :class:`~edelweissfe.solvers.nonlinearimplicitstaticparallelarclength.NISTPArcLength`.

        Parameters
        ----------
        stepActions
            The active step actions.
        U_np
            The trial solution.
        dU
            The trial solution increment.
        P
            The residual contribution vector to assemble into.
        K
            The tangent in VIJ layout to assemble into.
        timeStep
            The time step.

        Returns
        -------
        tuple[DofVector,VIJSystemMatrix]
            The updated residual contribution vector and tangent.
        """

        return self.assembleLoads(self.collectLoads(stepActions), U_np, dU, P, K, timeStep)

    #: The step lengths a line search may choose from: a shorter step would hardly move the iterate, a longer one would
    #: trust the fitted parabola far beyond the trial step lengths.
    lineSearchStepLengthRange = (0.1, 1.5)

    def lineSearchIsDue(self, iterationCounter: int) -> bool:
        """Whether the correction of this iteration is scaled by a line search (``lineSearch``): after
        ``lineSearchAfterIterations`` iterations, in every iteration whose number is a multiple of
        ``lineSearchEveryIterations``.

        Parameters
        ----------
        iterationCounter
            The current iteration number.

        Returns
        -------
        bool
            True if a line search is due.
        """

        return (
            self.options["lineSearch"]
            and iterationCounter > self.options["lineSearchAfterIterations"]
            and iterationCounter % self.options["lineSearchEveryIterations"] == 0
        )

    def lineSearchStepLengths(self) -> list[float]:
        """The trial step lengths of the line search, from the option ``lineSearchStepLengths``.

        Returns
        -------
        list[float]
            The trial step lengths.
        """

        return [float(stepLength) for stepLength in self.options["lineSearchStepLengths"].split(",")]

    @performancetiming.timeit("line search")
    def quadraticLineSearch(self, largestResidualAt: Callable[[float], float]) -> float:
        """The step length :math:`\\alpha` by which a Newton correction is scaled, for a Newton iteration that
        overshoots or oscillates (e.g., materials switching between loading and unloading).

        The largest residual along the correction, :math:`\\| R(\\alpha) \\|_\\infty` with the Dirichlet degrees of
        freedom left out, is evaluated at the trial step lengths ``lineSearchStepLengths``, and a parabola is fitted to
        these values (least squares for more than three). The step length minimizes the parabola on
        :attr:`lineSearchStepLengthRange`: it is the vertex of a parabola that opens upwards, limited to that range,
        and otherwise the end of the range where the parabola is smaller. Each trial costs one evaluation of the model;
        the next Newton iteration evaluates the chosen one.

        Two kinds of trials fit no parabola:

        * A trial state at which the residual is not finite, which the model could not evaluate: the trials decide
          themselves, and the step length is the trial step length of the smallest finite residual (of equal ones, the
          shortest), limited to the range. If no trial residual is finite, the increment is cut back.
        * Trial residuals that are equal to round-off, which do not tell the step lengths apart: the correction is
          taken in full, the step length is 1.

        What the correction is, the caller says: :meth:`solveIncrement` scales the solution correction
        :math:`\\Delta\\Delta U`, and the arc length method the pair of the solution correction and the correction of
        the load parameter, with the same step length, so that the trial states stay on the Newton direction of the
        coupled problem.

        Parameters
        ----------
        largestResidualAt
            Evaluates the model at the trial state of a step length, and returns its largest residual, without the
            Dirichlet degrees of freedom.

        Returns
        -------
        float
            The step length.

        Raises
        ------
        DivergingSolution
            If the residual is not finite at any trial step length.
        """

        stepLengths = self.lineSearchStepLengths()
        largestResiduals = []

        for stepLength in stepLengths:
            largestResiduals.append(largestResidualAt(stepLength))
            self.journal.message(
                "line search: step length {:7.4f}, largest residual {:11.4e}".format(stepLength, largestResiduals[-1]),
                self.identification,
                2,
            )

        smallest, largest = self.lineSearchStepLengthRange

        if not np.all(np.isfinite(largestResiduals)):
            # no parabola fits through a residual that is not finite: the smallest finite one decides
            finiteTrials = [
                (largestResidual, stepLength)
                for stepLength, largestResidual in zip(stepLengths, largestResiduals)
                if np.isfinite(largestResidual)
            ]
            if not finiteTrials:
                raise DivergingSolution("Residual not finite at any trial step length of the line search, cutting back")
            stepLength = min(max(min(finiteTrials)[1], smallest), largest)
        elif max(largestResiduals) - min(largestResiduals) <= 1e-12 * max(largestResiduals):
            # a parabola fitted to round-off would choose any step length: the full correction
            stepLength = 1.0
        else:
            # the minimum of the parabola on the allowed range: its vertex if it opens upwards, else an end of the range
            parabola = np.polyfit(stepLengths, largestResiduals, 2)
            curvature, slope, _ = parabola
            if curvature > 0.0:
                stepLength = min(max(-slope / (2.0 * curvature), smallest), largest)
            else:
                stepLength = smallest if np.polyval(parabola, smallest) <= np.polyval(parabola, largest) else largest

        self.journal.message("line search: step length {:7.4f}".format(stepLength), self.identification, 2)

        return stepLength

    def checkDivergingSolution(self, incrementResidualHistory: dict, maxGrowingIter: int) -> bool:
        """Check if the iterative solution scheme is diverging: if the flux residual of a field grew, from one
        checked iteration to the next, more than ``maxGrowingIter`` times in this increment.

        Two solver options decide what counts as a growth:

        * ``countFirstResidualAsGrowth``: the residual history of a field starts from zero, so the first checked
          residual of an increment is a growth already (EdelweissFE's original counting); otherwise, counting starts
          with the second checked iteration.
        * ``countGrowthOfConvergedFields``: a field that is converged, while other fields keep the Newton iterations
          going, may still grow at the level of its round-off; such growths count too (EdelweissFE's original
          counting), or are ignored.

        Parameters
        ----------
        incrementResidualHistory
            The convergence state of every field in each checked iteration of the increment, see
            :meth:`~edelweissfe.solvers.base.nonlinearsolverbase.NonlinearSolverBase.checkConvergence`.
        maxGrowingIter
            The maximum allowed number of growths of a residual during the iterative solution scheme.

        Returns
        -------
        bool
            True if solution is diverging.
        """

        countFirstResidualAsGrowth = self.options["countFirstResidualAsGrowth"]
        countGrowthOfConvergedFields = self.options["countGrowthOfConvergedFields"]

        for fieldHistory in incrementResidualHistory.values():
            nGrew = 0
            previousFluxResidual = 0.0 if countFirstResidualAsGrowth else None
            for fieldConvergence in fieldHistory:
                if previousFluxResidual is not None and fieldConvergence.fluxResidual > previousFluxResidual:
                    fieldConverged = fieldConvergence.fluxResidualConverged and fieldConvergence.correctionConverged
                    if countGrowthOfConvergedFields or not fieldConverged:
                        nGrew += 1
                previousFluxResidual = fieldConvergence.fluxResidual

            if nGrew > maxGrowingIter:
                return True

        return False

    def checkDivergingSolutionOrFallBack(
        self,
        quasiNewton: QuasiNewtonFallback,
        incrementResidualHistory: dict,
        maxGrowingIter: int,
        nodesWithLargestResidual: dict,
    ):
        """Cut an increment back whose residual grew more often than ``maxGrowingIter`` allows (see
        :meth:`checkDivergingSolution`) -- or, with the option ``fallBackToQuasiNewton``, let it continue with the tangent
        of its first iteration (a quasi-Newton method), see :meth:`assembleSystemMatrix`. Called by the Newton loops after
        every check that does not converge.

        Parameters
        ----------
        quasiNewton
            The quasi-Newton fallback of this attempt of the increment.
        incrementResidualHistory
            The convergence state of every field in each checked iteration of the increment.
        maxGrowingIter
            The maximum allowed number of growths of a residual.
        nodesWithLargestResidual
            The nodes with the largest residual, per field, reported if the increment diverges.

        Raises
        ------
        DivergingSolution
            If the residual grew too often, and the increment does not fall back to the quasi-Newton method.
        """

        if quasiNewton.isEngaged or not self.checkDivergingSolution(incrementResidualHistory, maxGrowingIter):
            return

        self.printResidualOutlierNodes(nodesWithLargestResidual)
        if not quasiNewton.isEnabled:
            raise DivergingSolution("Residual grew {:} times, cutting back".format(maxGrowingIter))

        self.journal.message(
            "Residual grew {:} times, continuing with the tangent of the first iteration".format(maxGrowingIter),
            self.identification,
            1,
        )
        quasiNewton.isEngaged = True

    def assembleSystemMatrix(self, K: VIJSystemMatrix, quasiNewton: QuasiNewtonFallback) -> csr_matrix:
        """The system matrix of this Newton iteration, before the Dirichlet boundary conditions are imposed: the
        tangent in CSR layout, condensed by the multi-point constraints -- or, once the increment has fallen back to the
        quasi-Newton method, the tangent of its first iteration.

        Parameters
        ----------
        K
            The tangent of this iteration in VIJ layout.
        quasiNewton
            The quasi-Newton fallback of this attempt of the increment, which keeps the tangent of the first iteration.

        Returns
        -------
        csr_matrix
            The system matrix.
        """

        if quasiNewton.isEngaged and quasiNewton.tangentOfFirstIteration is not None:
            # Kept without the Dirichlet rows and columns: they are imposed anew, as they eliminate into this R.
            return quasiNewton.tangentOfFirstIteration.copy()

        K_ = self.assembleStiffnessCSR(K)

        if self.mpcTransformation is not None:
            K_ = self.mpcTransformation.transformSystemMatrix(K_)

        # the first tangent of the increment: the one of iteration 0
        if quasiNewton.isEnabled and quasiNewton.tangentOfFirstIteration is None:
            quasiNewton.tangentOfFirstIteration = K_.copy()

        return K_

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
        by :meth:`assembleResidualAndTangent` in every evaluation of the model, next to :meth:`computeElements`,
        :meth:`assembleLoadsAtIncrementEnd` and :meth:`assembleConstraints`: right after the right-hand side
        :math:`R = f_\\mathrm{ext} - f_\\mathrm{int}` is formed (see
        :ref:`residual-and-tangent-convention`), and before the multi-point-constraint condensation,
        the Dirichlet handling, the convergence test and the linear solve -- which therefore all act
        on what is added here.

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
        :attr:`~edelweissfe.linsolve.base.LinearSolver.needsPrunedSystemMatrix`."""
        explicit = self.options["pruneCondensedMatrixZeros"]
        if explicit is not None:
            return explicit
        return self.linSolver.needsPrunedSystemMatrix

    @performancetiming.timeit("dirichlet K on CSR")
    def applyDirichletToStiffness(self, K: csr_matrix, dirichlets: list[StepActionBase], rhs=None) -> csr_matrix:
        """Impose the Dirichlet BCs on the system matrix -- and, given the right-hand side(s) ``rhs``
        (modified in place), eliminate the constrained DOFs from the columns as well; see
        :func:`edelweissfe.solvers.base.dirichlet.applyDirichletToStiffness`."""
        K = applyDirichletToStiffness(K, dirichlets, rhs)

        # Compacting the explicit zeros out of a condensed K (see _shouldPruneCondensedMatrixZeros): PARDISO's
        # reordering of these poorly conditioned, path-dependent (contact/friction) systems is sensitive enough to
        # the extra structural entries to drift from the reference path; but the pattern then changes from one
        # Newton iteration to the next, so a solver like blockamg, which reuses its hierarchy, keeps them.
        #
        # The CSR generator's own matrix (the threaded condensation works in place on it) must never be compacted:
        # the generator scatters every update into its full-length buffer via a fixed map. A copy is pruned instead.
        if self.mpcTransformation is not None and self._shouldPruneCondensedMatrixZeros():
            if K is self.csrGenerator.csrMatrix:
                K = K.copy()
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
        P: DofVector,
        K: VIJSystemMatrix,
        timeStep: TimeStep,
        FConstraints: DofVector = None,
    ) -> tuple[DofVector, VIJSystemMatrix]:
        """Loop over all constraints, and evaluate the active ones: each adds its residual contribution and the
        tangent of it, see :meth:`~edelweissfe.constraints.base.constraintbase.ConstraintBase.computeKernels`.
        A constraint that is switched off (see
        :attr:`~edelweissfe.constraints.base.constraintbase.ConstraintBase.active`) is not part of the equation
        system and is skipped. Is called by solveIncrement() in each iteration.

        Parameters
        ----------
        constraints
            The list of constraints.
        U_np
            The current solution vector.
        dU
            The current solution increment vector.
        P
            The residual contribution vector to assemble into.
        K
            The system matrix.
        timeStep
            The time step.
        FConstraints
            The vector of accumulated absolute constraint fluxes for convergence checks, if given.

        Returns
        -------
        tuple[DofVector,VIJSystemMatrix]
            - The modified residual contribution vector.
            - The modified system matrix.
        """

        for constraint in constraints.values():
            # a constraint that is switched off has no place in the equation system
            if not constraint.active:
                continue

            Kc = K[constraint]
            Pc = np.zeros(constraint.nDof)

            constraint.computeKernels(U_np[constraint], dU[constraint], Pc, Kc, timeStep)

            # instead of P[constraint] += Pc, np.add.at allows for repeated indices
            np.add.at(P, P.entitiesInDofVector[constraint], Pc)
            if FConstraints is not None:
                np.add.at(FConstraints, FConstraints.entitiesInDofVector[constraint], np.abs(Pc))

        return P, K
