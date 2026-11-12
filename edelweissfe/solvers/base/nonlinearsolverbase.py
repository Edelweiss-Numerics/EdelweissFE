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

import dataclasses
from abc import ABC, abstractmethod

import numpy as np
from numpy import ndarray
from scipy.sparse import csr_matrix

import edelweissfe.utils.performancetiming as performancetiming
from edelweissfe.models.change import Change
from edelweissfe.models.discretesystem import DiscreteSystem
from edelweissfe.models.femodel import FEModel
from edelweissfe.numerics.dofvector import DofVector
from edelweissfe.numerics.mpctransformation import (
    MultiPointConstraintTransformation,
    _flattenChainedRecords,
)
from edelweissfe.numerics.systemmatrixassembly import (
    SystemMatrixAssembly,
    SystemMatrixBlocks,
)
from edelweissfe.solvers.base.convergencetable import ConvergenceTable
from edelweissfe.stepactions.base.loadbase import LoadBase
from edelweissfe.stepactions.base.stepactionbase import StepActionBase
from edelweissfe.timesteppers.timestep import TimeStep
from edelweissfe.utils.checkpointedstate import packState, unpackState
from edelweissfe.utils.exceptions import DivergingSolution, TopologyError
from edelweissfe.utils.schema import (
    OptionSchemaProvider,
    coerceValue,
    fieldSchemaMeta,
    optionNames,
)


class NonlinearSolverBase(OptionSchemaProvider, ABC):
    """This is the base class for all nonlinear solvers.

    Parameters
    ----------
    jobInfo
        A dictionary containing the job information.
    journal
        The journal instance for logging.
    """

    identification = "NonlinearSolverBase"

    SolverSpecificOptions = {}

    #: Whether this solver supports master-slave condensation / multi-point constraints
    #: (e.g. surface ties). Subclasses supporting MPCs must set this to True.
    supportsMPC = False

    #: Whether this solver runs the topology update (e.g. h-adaptivity) at all; without it a modifier never runs.
    #: A solver that runs it only before its increment loop sets this too, and refuses the modifiers that would
    #: need it later (see ModelModifierBase.actsOnlyAtSimulationStart and NED.validateModelCapabilities).
    supportsModelModifiers = False

    #: The current discrete system of the model (see :meth:`~edelweissfe.models.femodel.FEModel.buildDiscreteSystem`),
    #: with its numbering, its locations, and how the solution passes between the model and the vectors of the solver.
    #: Replaced by a new one, never changed (see :meth:`updateDiscreteSystem`); None until the first one is built.
    system: DiscreteSystem | None = None

    #: What the topology update changed at the start of the increment, which the increment reacts on (see
    #: :meth:`updateTopologyAndConnectivity`).
    _change: Change = Change.NONE

    #: The actions of the step being solved, set when it begins. The multi-point constraints are reconciled with its
    #: Dirichlet boundary conditions whenever the discrete system gets a new numbering.
    _stepActions: dict | None = None

    #: The active multi-point-constraint (hanging node / tie) condensation, or None without MPCs. Lets
    #: applyDirichletToStiffness tell an MPC-transformed (disposable) system matrix apart from the
    #: assembler's persistent, in-place-updated one.
    mpcTransformation = None

    #: The layout of the iteration table that :meth:`checkConvergence` writes, one column per field; set up
    #: by the solver whenever it (re)builds its discrete system.
    convergenceTable: ConvergenceTable | None = None

    #: Whether the header of the iteration table is written in this attempt of the increment: it comes with the
    #: first row, so that what the linear solver reports while it solves for the first time (e.g. its set-up) stands
    #: above the table, not inside it.
    tableHeaderWritten: bool = False

    #: The force buffer of each constraint, by constraint, with the DOF indices it was made for, which
    #: :meth:`assembleConstraintForces` reuses from increment to increment as long as the constraint has these indices.
    _constraintForcePlans: dict | None = None

    def __init__(self, jobInfo, journal, **kwargs):
        pass

    def validateModelCapabilities(self, model: FEModel):
        """Validate whether the solver supports the active features/constraints of the model.

        Parameters
        ----------
        model
            The model tree.
        """
        if model.multiPointConstraints and not self.supportsMPC:
            raise NotImplementedError(
                f"Multi-point constraints (e.g. surface ties) are not supported by the {self.identification} solver."
            )

        # Only self-starting modifiers matter: a purely reactive one (e.g. the surface-facet retiling) never acts
        # unless another modifier changed the mesh first.
        selfStarting = sorted(name for name, m in model.modelModifiers.items() if m.initiatesTopologyChanges)
        if selfStarting and not self.supportsModelModifiers:
            raise NotImplementedError(
                "The {:} solver does not run the topology update, so the model modifier(s) {:} "
                "would never modify the model.".format(self.identification, ", ".join(selfStarting))
            )

    def _updateOptions(self, updatedOptions: dict, journal, strict: bool = False):
        """Update options of the solver using a string dict

        Parameters
        ----------
        updatedOptions
            The options dictionary.
        journal
            The journal module.
        strict
            If True, an unrecognised option raises an AttributeError instead of being ignored. Use it
            for option sources which are exclusively owned by this solver (i.e. the datalines of the
            *solver keyword), such that typos are not silently swallowed.

        Returns
        -------
        set[str]
            The names of the options updated.
        """

        # Option keys arrive lowercased: match them case-insensitively against the camelCase names. A >>options
        # block carries every solver's options plus meta keys, so keys of other solvers are skipped.
        canonicalByLower = {key.lower(): key for key in self.SolverSpecificOptions}
        updatedNames = set()
        for k, v in updatedOptions.items():
            canonicalKey = canonicalByLower.get(k.lower())
            if canonicalKey is None:
                if strict:
                    raise AttributeError("Invalid option {:} for {:}".format(k, self.identification))
                continue
            updatedNames.add(canonicalKey)
            journal.message("Updating option {:}={:}".format(canonicalKey, v), self.identification)
            defaultValue = self.SolverSpecificOptions[canonicalKey]
            if isinstance(defaultValue, bool):
                # bool("False") is truthy, so parse the string explicitly rather than via bool(...)
                self.options[canonicalKey] = str(v).strip().lower() in ("true", "1", "yes", "on")
            elif isinstance(defaultValue, list):
                self.options[canonicalKey] = self.options[canonicalKey] + [item.strip() for item in str(v).split(",")]
            elif defaultValue is None:
                # an option that decides for itself while unset (``bool | None``): no default to take its type from,
                # so the schema says it
                dtype = fieldSchemaMeta(optionNames(self.schema)[canonicalKey]).dtype
                self.options[canonicalKey] = coerceValue(v, dtype)
            else:
                self.options[canonicalKey] = type(defaultValue)(v)

        return updatedNames

    @performancetiming.timeit("topology update")
    def updateTopologyAndConnectivity(self, model: FEModel, step, offerModelModifiers: bool = True) -> Change:
        """Run the topology update, then let every mesh-dependent consumer catch up on it, and say what changed.

        Two phases. First, the model modifiers plan and apply to a fixed point inside one topology
        window (:meth:`~edelweissfe.models.topologypipeline.TopologyPipeline.update`). Second, the pure
        readers of the settled model catch up, once, on the net change: the tie and contact surfaces,
        then the connectivity, which the model brings up to date itself
        (:meth:`~edelweissfe.models.femodel.FEModel.updateConnectivity`; the constraints of a finite
        element model, e.g. a contact search). They may not create or delete elements: the topology
        window is closed by then, so an attempt raises.

        One :class:`~edelweissfe.models.change.Change` comes out of it, the same for every solver:

        * :attr:`~edelweissfe.models.change.Change.MESH` if the model modifiers changed the mesh, or if a
          mesh-dependent consumer caught up (e.g. a tie or contact surface rebuilt on a mesh that changed in a way that
          concerns it). A consumer refreshes only when the topology version advanced, i.e. when nodes or elements did
          change, so its catching up is a change of the mesh whoever made it, and its changed records (e.g. the
          multi-point constraints of a tie) are sized by that mesh.
        * Otherwise the model's answer.

        Parameters
        ----------
        model
            The model tree.
        step
            The step being solved.
        offerModelModifiers
            False when an increment is retried after a cutback. The model modifiers decide once per
            converged state: the retry starts from the state they already decided on.

        Returns
        -------
        Change
            What changed; the solver decides from it how much of its discrete system to renew.
        """

        meshChanged = model.topology.update(step) if offerModelModifiers else False
        meshDependentsRefreshed = model.topology.refreshMeshDependents()
        change = model.updateConnectivity()

        if meshChanged or meshDependentsRefreshed:
            return Change.MESH
        return change

    def updateDiscreteSystem(self, model: FEModel, change: Change):
        """Bring the discrete system up to date with what changed: the one way a solver builds its discrete system, from
        the first to the last increment.

        * No discrete system yet, or a change of the numbering or of the mesh
          (:attr:`~edelweissfe.models.change.Change.NUMBERING`, :attr:`~edelweissfe.models.change.Change.MESH`): the
          model builds a new one (:meth:`~edelweissfe.models.femodel.FEModel.buildDiscreteSystem`), then
          :meth:`onNewNumbering` and :meth:`onNewLocations` follow. The old one is dropped before
          (:meth:`forgetDiscreteSystem`): never both at once.
        * A change of the locations (:attr:`~edelweissfe.models.change.Change.LOCATIONS`): the same numbering, new
          locations (:meth:`~edelweissfe.models.discretesystem.DiscreteSystem.withNewLocations`); only
          :meth:`onNewLocations` follows, so that nothing sized by the numbering is touched.
        * Else nothing.

        The change is the one :meth:`updateTopologyAndConnectivity` reports: the model classifies what its entities
        report, and the topology update adds a change of the mesh. The solver never tests whether an entity changed.

        Parameters
        ----------
        model
            The model tree.
        change
            What changed since the discrete system was built.
        """

        if self.system is not None and change == Change.NONE:
            return

        with performancetiming.timeit("discrete system rebuild"):
            if self.system is None or change >= Change.NUMBERING:
                self.forgetDiscreteSystem()
                with performancetiming.timeit("numbering and locations"):
                    self.system = model.buildDiscreteSystem()
                self.onNewNumbering()
            else:
                with performancetiming.timeit("numbering and locations"):
                    self.system = self.system.withNewLocations()
            self.onNewLocations()

    def forgetDiscreteSystem(self) -> None:
        """Drop the discrete system, and everything that belongs to it, before the next one is built; a solver that
        keeps more of it extends this."""

        self.system = None

    def onNewNumbering(self):
        """Everything that is sized by the numbering of the new discrete system, :attr:`system`, is built here, e.g. the
        vectors, which the state is read into from the model, the linear solver's view of the degrees of freedom, the
        layout of the convergence table. Called by :meth:`updateDiscreteSystem` when the discrete system was built
        anew, before :meth:`onNewLocations`. Never called for new locations on the same numbering: what is built here
        stays as it is.

        Raises
        ------
        NotImplementedError
            A solver that builds through :meth:`updateDiscreteSystem` implements this.
        """

        raise NotImplementedError("{:} implements onNewNumbering.".format(self.identification))

    def onNewLocations(self):
        """Everything that is derived from the locations of the new discrete system, :attr:`system`, is built here: the
        vectors are wrapped with them (:meth:`~edelweissfe.numerics.dofvector.DofVector.onLocations`, the values are
        kept), and the system matrix assembly, whose sparsity pattern the locations determine. Called by
        :meth:`updateDiscreteSystem` after every new discrete system, and after new locations on the same numbering.

        Raises
        ------
        NotImplementedError
            A solver that builds through :meth:`updateDiscreteSystem` implements this.
        """

        raise NotImplementedError("{:} implements onNewLocations.".format(self.identification))

    @staticmethod
    def collectLoads(stepActions: dict) -> list[LoadBase]:
        """The loads among a step's actions: every action that is a
        :class:`~edelweissfe.stepactions.base.loadbase.LoadBase`, whatever step action module it is defined in, so
        that a new kind of load needs no change to any solver.

        They come in the order in which their contributions are summed, module by module: the node forces, the
        distributed loads and the body forces first, then the loads of any other module, by the name of the module;
        within a module, in the order of their definition. A fixed order makes the residual the same sum in every run.

        Parameters
        ----------
        stepActions
            The step actions, by module.

        Returns
        -------
        list[LoadBase]
            The loads.
        """

        modulesSummedFirst = ("nodeforces", "distributedload", "bodyforce")
        modules = [module for module in modulesSummedFirst if module in stepActions]
        modules += sorted(module for module in stepActions if module not in modulesSummedFirst)

        return [action for module in modules for action in stepActions[module].values() if isinstance(action, LoadBase)]

    @performancetiming.timeit("assemble loads")
    def assembleLoads(
        self,
        loads: list[LoadBase],
        U_np: DofVector,
        dU: DofVector,
        P: DofVector,
        K: SystemMatrixBlocks | None,
        timeStep: TimeStep,
    ) -> tuple[DofVector, SystemMatrixBlocks]:
        """Assemble the residual contributions of all loads: each adds the negative of the external load it
        applies, and the tangent of that contribution, which is nonzero only for a load that follows the
        deformation; see :ref:`residual-and-tangent-convention`. Every load knows where it acts, see
        :meth:`~edelweissfe.stepactions.base.loadbase.LoadBase.assembleLoad`, so this only loops over them.

        Parameters
        ----------
        loads
            The loads, see :meth:`collectLoads`.
        U_np
            The current solution vector.
        dU
            The current solution increment.
        P
            The residual contribution vector to assemble into.
        K
            The system matrix to assemble into; None for an explicit solver, which needs no tangent.
        timeStep
            The current time step.

        Returns
        -------
        tuple[DofVector,SystemMatrixBlocks]
            - The updated residual contribution vector.
            - The updated system matrix.
        """

        for load in loads:
            load.assembleLoad(P, K, U_np, dU, timeStep, self.system.locations)

        if K is not None:
            # the block of the last load
            K.flush()

        return P, K

    @performancetiming.timeit("assemble constraints")
    def assembleConstraintForces(
        self,
        constraints: list,
        U_np: DofVector,
        dU: DofVector,
        P: DofVector,
        timeStep: TimeStep,
    ) -> DofVector:
        """Evaluate the constraints of the discrete system and add their residual contributions to the given vector.

        The explicit counterpart of
        :meth:`~edelweissfe.solvers.nonlinearimplicitstatic.NIST.assembleConstraints`, with the same
        sign: a constraint adds the negative of the force it exerts on the nodes, like an element its
        internal force, see :ref:`residual-and-tangent-convention`.

        No tangent is requested. An explicit increment solves no linear system, so a constraint's
        stiffness enters nothing, and
        :meth:`~edelweissfe.constraints.base.constraintbase.ConstraintBase.computeKernelsExplicit`
        is the entry point that evaluates the constraint forces directly without assembling a tangent.
        A constraint that could act *only* through its tangent would contribute nothing here; an explicit solver
        refuses it in its ``validateModelCapabilities``.

        The force buffer and the DOF indices of each constraint are kept in :attr:`_constraintForcePlans` (by
        constraint) and reused in every increment. They hold for the indices of the constraint they were made
        for: whenever the locations locate the constraint anew -- a rebuilt discrete system, or one with new
        locations --, the constraint has indices of its own, its plan is made anew, and the
        constraint is told so
        (:meth:`~edelweissfe.constraints.base.constraintbase.ConstraintBase.prepareExplicitEvaluation`).

        Parameters
        ----------
        constraints
            The constraints of the discrete system
            (:attr:`~edelweissfe.models.discretesystem.DiscreteSystem.constraints`).
        U_np
            The current solution vector.
        dU
            The current solution increment.
        P
            The residual contribution vector to be augmented.
        timeStep
            The current time step.

        Returns
        -------
        DofVector
            The augmented residual contribution vector.
        """

        # Per-constraint buffers and index plans, made once per discrete system.
        if self._constraintForcePlans is None:
            self._constraintForcePlans = {}
        PPlain = P.asPlainArray()

        for constraint in constraints:
            # A plan holds for the DOF indices it was made for: a constraint that is located anew has new ones.
            indices = self.system.locations[constraint]
            plan = self._constraintForcePlans.get(constraint)
            if plan is None or plan[1] is not indices:
                # np.add.at only if a DOF appears more than once (e.g. a slave node in its own master
                # facet), since fancy-indexed += would drop repeated contributions.
                plan = (np.zeros(constraint.nDof), indices, len(np.unique(indices)) != len(indices))
                self._constraintForcePlans[constraint] = plan
                constraint.prepareExplicitEvaluation()

            Pc, _, namesDofMoreThanOnce = plan

            # Reused, so it must be cleared: computeKernelsExplicit augments what it is handed.
            Pc[:] = 0.0

            constraint.computeKernelsExplicit(U_np[constraint], dU[constraint], Pc, timeStep)

            if namesDofMoreThanOnce:
                np.add.at(PPlain, indices, Pc)
            else:
                PPlain[indices] += Pc

        return P

    def getRestartData(self) -> dict[str, np.ndarray]:
        """The state this solver carries from one increment to the next, as it is; see
        :attr:`checkpointedState`.

        Returns
        -------
        dict[str, numpy.ndarray]
            The state; see :func:`~edelweissfe.utils.checkpointedstate.packState`.

        Raises
        ------
        RestartError
            If this solver does not declare its state, i.e. does not support restart.
        """

        return packState(self)

    def setRestartData(self, data: dict[str, np.ndarray]):
        """Restore the state :meth:`getRestartData` returned, so that the resumed step continues
        exactly where the uninterrupted run would have.

        Parameters
        ----------
        data
            The state.
        """

        unpackState(self, data)

    def applyOptionsOverride(self, fieldValues: dict) -> None:
        """Apply a partial override of this solver's own ``schema`` fields onto ``self.options``.

        The counterpart, on the solver side, of the name-based ``>>options`` override mechanism
        (``stepactions/options.py``): once that mechanism has resolved an ``>>options, name=X, ...``
        block to this solver instance and validated the present keys against ``type(self).schema``
        via :func:`~edelweissfe.utils.schema.coercePresentOptions`, it calls this method with the
        result to actually apply them.

        ``fieldValues`` is keyed by *schema field name* (e.g. ``rungeKuttaStages``), while
        ``self.options`` -- read throughout a step -- is keyed by the
        option's ``.inp``-facing spelling (e.g. ``"runge-kutta-stages"``), which are not always the
        same (a hyphenated name cannot be a Python identifier). The schema's ``optionName`` metadata
        is the one place that mapping is recorded, so it is consulted here rather than duplicated.

        Parameters
        ----------
        fieldValues
            Maps schema field name to its new, already-coerced value.
        """

        fieldsByName = {field.name: field for field in dataclasses.fields(self.schema)}
        for fieldName, value in fieldValues.items():
            optionName = fieldSchemaMeta(fieldsByName[fieldName]).optionName or fieldName
            self.journal.message("Updating option {:}={:}".format(optionName, value), self.identification)
            self.options[optionName] = value

    #: The status of the current increment, handed to the output managers (iterations, notes, ...),
    #: or None.
    incrementStatus = None

    #: The state this solver carries from one increment to the next, by attribute name and type; see
    #: :mod:`~edelweissfe.utils.checkpointedstate`. None for a solver that does not support restart:
    #: writing a checkpoint then refuses.
    checkpointedState: dict | None = None

    @abstractmethod
    def beginStep(self, step, model: FEModel, fieldOutputController, outputmanagers):
        """Start a step: set up what the step needs, and -- on a cold start only, see
        :meth:`~edelweissfe.timesteppers.base.timestepperbase.TimeStepperBase.isAtStepStart` -- apply
        the step actions' step-start parts and reset the state carried between increments. A resumed
        step continues from the state the checkpoint restored.

        The step's increment loop (:meth:`~edelweissfe.steps.base.stepbase.StepBase.solve`) then
        calls, per increment, :meth:`prepareIncrement`, :meth:`attemptIncrement` and
        :meth:`acceptIncrement`, and finally :meth:`endStep`.

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

    @abstractmethod
    def prepareIncrement(self, step, model: FEModel, isRetry: bool):
        """Bring the model up to date before an increment is proposed: the topology update, when due,
        and whatever the solver derives from the mesh. Runs before the time stepper proposes the
        increment, because it may change what is proposed (an explicit solver lowers its stable time
        increment after a refinement).

        Parameters
        ----------
        step
            The step being solved.
        model
            The model tree.
        isRetry
            True if the increment is retried after a cutback. The model modifiers decide once per
            accepted state: the retry starts from the state they already decided on.
        """

    @abstractmethod
    def attemptIncrement(self, step, model: FEModel, timeStep: TimeStep):
        """Solve the increment, without committing anything to the model.

        Parameters
        ----------
        step
            The step being solved.
        model
            The model tree.
        timeStep
            The increment proposed by the time stepper.

        Raises
        ------
        IncrementFailed
            If the increment cannot be solved; it is retried smaller.
        """

    @abstractmethod
    def acceptIncrement(self, step, model: FEModel, timeStep: TimeStep):
        """Commit the solved increment to the model, and keep the state the solver carries to the next
        increment. Called before the time stepper accepts it, so that the solver may still keep the
        next increment from growing.

        Parameters
        ----------
        step
            The step being solved.
        model
            The model tree.
        timeStep
            The solved increment.
        """

    def isOutputIncrement(self, timeStep: TimeStep) -> bool:
        """Whether field outputs, output managers and restart checkpoints are written after this
        increment. By default, after every increment.

        Parameters
        ----------
        timeStep
            The accepted increment.

        Returns
        -------
        bool
            True to write output.
        """

        return True

    @abstractmethod
    def endStep(self, step, model: FEModel):
        """Finish a step, however it ended.

        Parameters
        ----------
        step
            The step that was solved.
        model
            The model tree.
        """

    @abstractmethod
    def solveIncrement(self, *args):
        pass

    @performancetiming.timeit("dirichlet R")
    def applyDirichletToResidual(self, timeStep: TimeStep, R: DofVector, dirichlets: list[StepActionBase]):
        """Impose the Dirichlet BCs on the residual.

        For every constrained DOF we *overwrite* its residual entry with the
        value we want the linear solve to return for that DOF's increment.
        Together with :meth:`applyDirichletToStiffness` (which replaces the DOF's
        row of K by the identity and eliminates its column into R), the
        linearized system ``K ddU = R`` then reproduces exactly that increment for the DOF.

        Parameters
        ----------
        timeStep
            The current time step.
        R
            The residual vector of the global discrete system to be modified.
        dirichlets
            The list of dirichlet boundary conditions.

        Returns
        -------
        DofVector
            The modified residual vector.
        """
        for dirichlet in dirichlets:
            R[dirichlet.constrainedDofIndices] = dirichlet.getPrescribedIncrementOfConstrainedDofs(timeStep)

        return R

    @performancetiming.timeit("convergence check")
    def checkConvergence(
        self,
        R: DofVector,
        ddU: DofVector,
        dU: DofVector,
        F: DofVector,
        FConstraints: DofVector,
        iterationCounter: int,
        residualHistory: dict,
    ) -> tuple[bool, dict]:
        """Check the convergence, individually for each field and for the scalar variables, based on the current flux
        residual and the correction, using the convergence criterion of the solver (see
        :mod:`edelweissfe.solvers.base.convergencecriteria`). Is called by solveIncrement() to decide whether to
        continue iterating or stop.

        Parameters
        ----------
        R
            The current residual.
        ddU
            The current correction increment.
        dU
            The current solution increment.
        F
            The accumulated absolute element fluxes.
        FConstraints
            The accumulated absolute constraint fluxes.
        iterationCounter
            The current iteration number.
        residualHistory
            The convergence state of every field in each iteration checked so far in this increment; the state of
            this iteration is appended, for
            :meth:`~edelweissfe.solvers.nonlinearimplicitstatic.NIST.checkDivergingSolution`.

        Returns
        -------
        tuple[bool,dict]
            - True if converged.
            - The nodes with the largest residual, per field.

        Raises
        ------
        DivergingSolution
            If the residual contains a NaN: no further iteration can recover from it.
        """

        if np.isnan(R).any():
            raise DivergingSolution("Obtained NaN in the residual")

        tableColumns = []
        convergedAtAll = True
        nodesWithLargestResidual = {}

        spatialAveragedFluxes = self.convergenceCriterion.computeSpatialAveragedFluxes(
            F, FConstraints, self.system.locations
        )

        for field, fieldIndices in self.system.numbering.idcsOfFieldsInDofVector.items():
            fieldConvergence = self.convergenceCriterion.checkField(
                field,
                R[fieldIndices],
                ddU[fieldIndices] if ddU is not None else None,
                dU[fieldIndices],
                spatialAveragedFluxes[field],
                iterationCounter,
            )

            nodesWithLargestResidual[field] = self.system.numbering.getNodeForIndexInDofVector(
                range(len(R))[fieldIndices][fieldConvergence.indexOfLargestResidual]
            )

            residualHistory[field].append(fieldConvergence)

            tableColumns.append(fieldConvergence)
            convergedAtAll = (
                convergedAtAll and fieldConvergence.correctionConverged and fieldConvergence.fluxResidualConverged
            )

        if self.system.numbering.idcsOfScalarVariablesInDofVector:
            scalarVariableIndices = list(self.system.numbering.idcsOfScalarVariablesInDofVector.values())
            scalarVariablesConvergence = self.convergenceCriterion.checkScalarVariables(
                R[scalarVariableIndices],
                ddU[scalarVariableIndices] if ddU is not None else None,
                dU[scalarVariableIndices],
                iterationCounter,
            )

            tableColumns.append(scalarVariablesConvergence)
            convergedAtAll = (
                convergedAtAll
                and scalarVariablesConvergence.correctionConverged
                and scalarVariablesConvergence.fluxResidualConverged
            )

        summary = self.linSolver.lastSolveSummary if self.linSolver.reportsSolveSummary else None
        # ddU is None on the first iteration of an increment: no solve yet, so the column stays blank.
        thisIterationsSolve = summary if ddU is not None else None

        if not self.tableHeaderWritten:
            self.journal.preformatted(self.convergenceTable.headerLines(), self.identification, level=2)
            self.tableHeaderWritten = True

        self.journal.preformatted(
            [self.convergenceTable.row(tableColumns, thisIterationsSolve)], self.identification, level=2
        )

        if thisIterationsSolve is not None:
            self.journal.preformatted(
                [self.convergenceTable.linearSolveDetailLine(line) for line in thisIterationsSolve.detailLines],
                self.linSolver.identification,
                level=2,
            )

        return convergedAtAll, nodesWithLargestResidual

    @performancetiming.timeit("linear solve")
    def linearSolve(self, A: csr_matrix, b: DofVector) -> ndarray:
        """Solve the linear system of equations.

        Parameters
        ----------
        A
            The system matrix in compressed spare row format.
        b
            The right hand side.

        Returns
        -------
        ndarray
            The solution 'x'.
        """

        ddU = self.linSolver(A, b)

        if np.isnan(ddU).any():
            raise DivergingSolution("Obtained NaN in linear solve")

        return ddU

    @performancetiming.timeit("assemble stiffness CSR")
    def assembleStiffnessCSR(self, K: SystemMatrixAssembly) -> csr_matrix:
        """The assembled system matrix in CSR format, see
        :meth:`~edelweissfe.numerics.systemmatrixassembly.SystemMatrixAssembly.assembledMatrix`.

        Parameters
        ----------
        K
            The system matrix being assembled.

        Returns
        -------
        csr_matrix
            The system matrix in compressed sparse row format; the assembly's own, which the next assembly
            overwrites. Safe, since no solver keeps it across iterations, and the Dirichlet boundary conditions only
            overwrite values.
        """
        return K.assembledMatrix()

    def printResidualOutlierNodes(self, residualOutliers: dict):
        """Print which nodes have the largest residuals.

        Parameters
        ----------
        residualOutliers
            The dictionary containing the outlier nodes for every field.
        """
        self.journal.message(
            "Residual outliers:",
            self.identification,
            level=1,
        )
        for field, node in residualOutliers.items():
            self.journal.message(
                "|{:20}|node {:10}|".format(field, node.label),
                self.identification,
                level=2,
            )

    def applyStepActionsAtStepStart(self, model: FEModel, step):
        """Apply every step action's step-start part (initial conditions, material initialization,
        prescribed fields, ...).

        Only at the start of a step: a resumed step was checkpointed after the step's real start,
        and the state these set has evolved since -- applying them again would overwrite it.

        Parameters
        ----------
        model
            The model tree.
        step
            The step being started.
        """

        if not step.timeStepper.isAtStepStart():
            return
        for stepActionType in step.actions.values():
            for action in stepActionType.values():
                action.applyAtStepStart(model)

    def updateRigidBodies(self, model: FEModel, timeStep: TimeStep):
        """Refresh the kinematics of all rigid bodies in the model after a converged increment.

        A rigid body's surface (visualization) nodes are not degrees of freedom of their own; they
        are fully determined by the rigid body's reference point. This propagates the just-converged
        reference-point pose onto those surface nodes so that output managers write the transient
        geometry of the moving body and any consumer relying on the surface nodes' ``coordinates``
        (e.g. the fast-path AABB of :meth:`~edelweissfe.rigidbodies.discreterigidbody.DiscreteRigidBody.getAABB`)
        sees the current configuration.

        Every nonlinear solver must call this once per converged increment, in :meth:`acceptIncrement`.

        Parameters
        ----------
        model
            The model tree.
        timeStep
            The converged time step.
        """

        for rigidBody in model.rigidBodies.values():
            rigidBody.updateKinematics(timeStep)

    def applyStepActionsAtStepEnd(self, model: FEModel, stepActions: dict[str, StepActionBase]):
        """Called when all step actions should finish a step.

        Parameters
        ----------
        model
            The model tree.
        stepActions
            The dictionary of active step actions.
        """

        for stepActionType in stepActions.values():
            for action in stepActionType.values():
                action.applyAtStepEnd(model)

    def applyStepActionsAtIncrementStart(
        self, model: FEModel, timeStep: TimeStep, stepActions: dict[str, StepActionBase]
    ):
        """Called when all step actions should be applied at the start of a step.

        Parameters
        ----------
        model
            The model tree.
        increment
            The time increment.
        stepActions
            The dictionary of active step actions.
        """

        for stepActionType in stepActions.values():
            for action in stepActionType.values():
                action.applyAtIncrementStart(model, timeStep)

    def locateConstrainedDofs(self, dirichlets: list[StepActionBase]):
        """Determine, up front, which global DOFs each Dirichlet BC constrains.

        Called at the start of every increment, before its iterations: the numbering of the degrees of freedom may
        have changed since the last increment -- a rebuild of the discrete system after a change of the mesh or of the
        coupling, or a new active domain of a meshfree model, which locates a node set anew. The result
        is cached on each BC as :attr:`~DirichletBase.constrainedDofIndices`, together with the
        positions of their nodes in the BC's node set (:attr:`~DirichletBase.constrainedNodePositions`),
        so that the Newton loop can address the constrained DOFs directly, instead
        of recomputing the mapping on every residual update and every stiffness
        modification.

        Parameters
        ----------
        dirichlets
            The list of dirichlet boundary conditions active in this step.
        """
        for dirichlet in dirichlets:
            dirichlet.constrainedDofIndices, dirichlet.constrainedNodePositions = self._constrainedDofsOf(dirichlet)

    def _constrainedDofsOf(self, dirichlet: StepActionBase) -> tuple[np.ndarray, np.ndarray]:
        """Return the global DOF indices prescribed by a single Dirichlet BC, and the positions of their nodes in its
        node set.

        The numbering knows every DOF of ``field`` on ``nSet``, laid out node
        by node in a single flat array::

            [ node0: (u_x u_y u_z),  node1: (u_x u_y u_z),  ... ]

        A BC usually prescribes only some of the per-node components (given by
        ``dirichlet.components``, e.g. just u_x and u_z). So we view the flat
        array as one row per node, keep only the prescribed component columns,
        and flatten it back into a plain list of global DOF indices. The order
        stays node-major, matching the prescribed values of these nodes
        (:meth:`~DirichletBase.getPrescribedIncrementOfConstrainedDofs`).

        Only the nodes of the set that are part of the discrete system count, see
        :meth:`~edelweissfe.numerics.dofnumbering.DofNumbering.locateFieldOnNodeSet`: all nodes of the set in a
        finite element model, those in the active domain in a meshfree one.
        """
        dofsOfFieldOnNodeSet, nodePositions = self.system.numbering.locateFieldOnNodeSet(
            dirichlet.field, dirichlet.nSet
        )
        perNodeDofs = dofsOfFieldOnNodeSet.reshape((-1, dirichlet.fieldSize))

        return perNodeDofs[:, dirichlet.components].flatten(), nodePositions

    def buildMPCTransformation(self, model: FEModel, stepActions: dict = None):
        """Collect the linear dependency records from all multi-point constraints of the model
        and assemble the master-slave condensation operator for the current discrete system.
        Must be called whenever the discrete system is (re)built.

        Parameters
        ----------
        model
            The model tree.
        stepActions
            The step's actions, against which conflicting records are reconciled (see
            :meth:`_reconcileMPCDirichletConflicts`). Without them the records are used exactly as
            collected -- there is nothing to reconcile against.

        Returns
        -------
        MultiPointConstraintTransformation | None
            The assembled transformation, or None if the model has no multi-point constraints or
            none of them currently constrains a DOF.
        """

        if not model.multiPointConstraints:
            return None

        if not self.supportsMPC:
            raise NotImplementedError(
                f"Multi-point constraints (e.g. surface ties) are not supported by the {self.identification} solver."
            )

        records = self._collectMultiPointConstraintRecords(model)

        if stepActions is not None:
            records = self._reconcileMPCDirichletConflicts(records, stepActions)

        # Without a record T is the identity: skip the T^T K T + C product and the pattern-changing zero pruning.
        if not records:
            return None

        transformation = MultiPointConstraintTransformation(
            records,
            self.system.numbering.nDof,
            useThreadedCondensation=self.options.get("useThreadedMPCCondensation", False),
        )

        self.journal.message(
            "eliminating {:} slave DOF(s) via multi-point constraints".format(transformation.nEliminatedDof),
            self.identification,
            2,
        )

        return transformation

    def _prescribedDofValues(self, stepActions: dict) -> dict:
        """``{globalDofIndex: prescribed increment}`` over **all** Dirichlet BCs of the step.

        The union matters: a tie slave's masters routinely take the relevant component from a
        different boundary condition than the slave does (a node can sit in both a symmetry set,
        which prescribes one component, and an encastre set, which prescribes three). Evaluated
        per-DOF rather than per-node for the same reason -- a BC need not prescribe every component.
        """

        prescribed = {}
        for dirichlet in stepActions["dirichlet"].values():
            if not dirichlet.active:
                continue
            # the node set may have grown in place (adaptive refinement): let the BC catch up first
            dirichlet.reconcileIfSetChanged()
            indices, nodePositions = self._constrainedDofsOf(dirichlet)
            # the prescribed values of the nodes that are part of the discrete system, one row per node of the set
            values = np.asarray(dirichlet.delta)[nodePositions].flatten()
            if indices.shape != values.shape:
                raise ValueError(
                    "Dirichlet '{:}': {:} constrained DOF(s) but {:} prescribed value(s) -- the DOF "
                    "index layout and the delta layout must agree node-by-node.".format(
                        dirichlet.name, indices.size, values.size
                    )
                )
            for index, value in zip(indices, values):
                prescribed[int(index)] = float(value)
        return prescribed

    def _collectMultiPointConstraintRecords(self, model: FEModel) -> list:
        """Collect every multi-point constraint's records, keeping one claim per slave DOF.

        A DOF may be condensed out only once, so when several constraints ask for the same one, the
        first in model order keeps it. That is an invariant of the condensation operator, not of any
        constraint type -- resolving it here, where all records are in one place and every
        constraint has finished refreshing, is what makes the outcome independent of the order in
        which constraints happened to be *refreshed*.

        **Model order therefore carries meaning**, and one case depends on it: a hanging node lying
        on a tie's slave surface is claimed by both. The hanging-node constraint must win -- it has
        no stand-in, whereas the tie's equation for that node is still delivered by the node's coarse
        parents, which are themselves tie slaves. ``hAdaptivity`` registers its hanging-node
        constraint at the FRONT of ``model.multiPointConstraints`` for exactly this reason, and
        ``tests/test_mpc_slave_claim_arbitration.py`` pins it.
        """

        records = []
        claimed = set()
        dropped = {}

        for name, mpc in model.multiPointConstraints.items():
            for record in mpc.getMultiPointConstraints(self.system.numbering):
                if record[0] in claimed:
                    if not mpc.mayYieldSlaveToEarlierClaim:
                        raise TopologyError(
                            f"multi-point constraint '{name}' must keep every slave it declares, but DOF "
                            f"{record[0]} is already claimed by a constraint earlier in model order; such a "
                            "constraint has to be registered before any constraint that may claim its slaves"
                        )
                    dropped[name] = dropped.get(name, 0) + 1
                    continue
                claimed.add(record[0])
                records.append(record)

        for name, count in dropped.items():
            self.journal.message(
                "{:} slave DOF(s) of '{:}' are already claimed by an earlier multi-point constraint "
                "(e.g. hanging nodes); their redundant records were dropped".format(count, name),
                self.identification,
                2,
            )

        return records

    def _reconcileMPCDirichletConflicts(self, records: list, stepActions: dict) -> list:
        """Drop the multi-point-constraint records whose slave DOF is also Dirichlet-prescribed, so
        the boundary condition takes precedence -- and report what was dropped.

        A DOF cannot be both eliminated by a constraint and prescribed. The alternative -- rejecting
        the model -- forces the user to subtract the constraint's slave nodes from the boundary
        condition's node set by hand, offline: a snapshot that goes stale the moment the mesh, the
        constraint tolerance or an adaptive refinement changes, and which additionally blocks the
        propagation of that boundary condition to nodes created later, since refinement extends a
        node set only where the whole parent face already lies within it. Resolving in favour of the
        boundary condition is what Abaqus does with the same conflict.

        Records are classified, not silently dropped:

        * **redundant** -- the constraint equation already delivers the prescribed value (every
          master with a non-negligible weight is itself prescribed, and the weighted sum matches).
          Dropping it changes nothing; reported quietly.
        * **overridden** -- it does not. Dropping it *does* change the model, so this is reported
          loudly with the worst offender: it usually means the boundary condition's node set is
          incomplete, or the constraint is not the one the user thought it was.

        Masters are resolved transitively first (a master may itself be a slave), exactly as the
        transformation does before it enforces anything.

        Returns
        -------
        list
          The records to keep.
        """

        prescribed = self._prescribedDofValues(stepActions)
        if not prescribed:
            return records

        flattened = dict(_flattenChainedRecords(records))

        dropped = set()
        nRedundant = 0
        overridden = []
        for slaveDof, masters in flattened.items():
            if slaveDof not in prescribed:
                continue

            target = prescribed[slaveDof]
            # a scaled tolerance, never an exact comparison: clamped closest-point projections
            # routinely leave round-off-scale weights that are not exactly 0.0
            scale = max((abs(c) for _, c in masters), default=1.0)
            totalWeight = 0.0
            delivered = 0.0
            unresolvedWeight = 0.0
            for masterDof, coefficient in masters:
                if abs(coefficient) <= 1e-12 * scale:
                    continue
                totalWeight += abs(coefficient)
                if masterDof in prescribed:
                    delivered += coefficient * prescribed[masterDof]
                else:
                    unresolvedWeight += abs(coefficient)

            dropped.add(slaveDof)
            if unresolvedWeight <= 1e-12 * scale and abs(delivered - target) <= 1e-12 * max(1.0, abs(target)):
                nRedundant += 1
            else:
                weightFraction = unresolvedWeight / totalWeight if totalWeight > 0.0 else 1.0
                overridden.append((slaveDof, weightFraction))

        if not dropped:
            return records

        if nRedundant:
            self.journal.message(
                "reconciled {:} Dirichlet/constraint conflict(s) that were exactly redundant "
                "(the constraint already delivered the prescribed value)".format(nRedundant),
                self.identification,
                2,
            )
        if overridden:
            worstDof, worstWeight = max(overridden, key=lambda entry: entry[1])
            self.journal.message(
                "WARNING: {:} multi-point-constraint equation(s) conflicted with a Dirichlet boundary "
                "condition and were dropped -- a DOF cannot be both eliminated by a constraint and "
                "directly prescribed, so the boundary condition wins. This changes the model: those "
                "DOFs now get exactly their prescribed value instead of whatever the constraint (e.g. "
                "a tie or hanging-node link) would have computed for them. Worst case: DOF {:}, where "
                "{:.3e} of its constraint weight depends on masters that are themselves NOT prescribed "
                "-- meaning the constraint's own value would likely have differed. This usually means "
                "the boundary condition's node set was built independently of the mesh's tie/hanging-"
                "node topology and is missing some nodes -- check whether it should be extended, or "
                "whether these DOFs should be excluded from the constraint.".format(
                    len(overridden), worstDof, worstWeight
                ),
                self.identification,
                2,
            )

        return [record for record in records if record[0] not in dropped]

    def checkMPCDirichletConflicts(self, transformation, stepActions):
        """Raise if any Dirichlet boundary condition of the step prescribes a DOF that is a slave
        DOF of a multi-point constraint.

        Parameters
        ----------
        transformation
            The assembled MultiPointConstraintTransformation (may be None).
        stepActions
            The step's actions dictionary.
        """

        if transformation is None:
            return

        # A post-condition: _reconcileMPCDirichletConflicts has already removed every conflicting record.
        for dirichlet in stepActions["dirichlet"].values():
            constrainedDofIndices, _ = self._constrainedDofsOf(dirichlet)
            transformation.checkDirichletConflicts(constrainedDofIndices)
