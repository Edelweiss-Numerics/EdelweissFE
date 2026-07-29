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

from abc import ABC, abstractmethod

import numpy as np
from numpy import ndarray
from scipy.sparse import csr_matrix

import edelweissfe.utils.performancetiming as performancetiming
from edelweissfe.models.femodel import FEModel
from edelweissfe.numerics.dofmanager import DofVector, VIJSystemMatrix
from edelweissfe.numerics.mpctransformation import MultiPointConstraintTransformation
from edelweissfe.stepactions.base.stepactionbase import StepActionBase
from edelweissfe.timesteppers.timestep import TimeStep
from edelweissfe.utils.exceptions import DivergingSolution
from edelweissfe.utils.schema import OptionSchemaProvider


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

    #: Whether this solver polls model.modelModifiers (e.g. h-adaptivity) every increment.
    #: Subclasses that call modifier.updateModel(...) in their solveStep loop must set this
    #: to True; without it, a modifier silently never runs and the model never adapts.
    supportsModelModifiers = False

    #: The active multi-point-constraint (hanging node / tie) condensation, if any -- None
    #: whenever there are no multi-point constraints in the model. Lets
    #: applyDirichletToStiffness tell an MPC-transformed (fresh, disposable) system matrix
    #: apart from the assembler's own persistent, in-place-updated one: both implicit and
    #: explicit-dynamic solvers build one when needed (see NonlinearExplicitDynamic.solveStep),
    #: the distinction is about which matrix is in play, not about the solver family.
    mpcTransformation = None

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

        if model.modelModifiers and not self.supportsModelModifiers:
            raise NotImplementedError(
                f"Model modifiers (e.g. h-adaptivity) are not supported by the {self.identification} solver."
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
        """

        # Input keywords arrive case-folded (the parser lowercases option keys), while the option
        # names in SolverSpecificOptions are camelCase -- match them case-insensitively via their
        # canonical spelling. A >>options block carries the UNION of every solver's options (they all
        # register on the same 'options' keyword) plus routing/meta keys ('category', 'inputFile',
        # 'datalines'), so keys not belonging to this solver are silently skipped rather than rejected.
        canonicalByLower = {key.lower(): key for key in self.SolverSpecificOptions}
        for k, v in updatedOptions.items():
            canonicalKey = canonicalByLower.get(k.lower())
            if canonicalKey is None:
                if strict:
                    raise AttributeError("Invalid option {:} for {:}".format(k, self.identification))
                continue
            journal.message("Updating option {:}={:}".format(canonicalKey, v), self.identification)
            defaultValue = self.SolverSpecificOptions[canonicalKey]
            if isinstance(defaultValue, bool):
                # bool("False") is truthy, so parse the string explicitly rather than via bool(...)
                self.options[canonicalKey] = str(v).strip().lower() in ("true", "1", "yes", "on")
            else:
                self.options[canonicalKey] = type(defaultValue)(v)

    @abstractmethod
    def solveStep(self, *args):
        pass

    @abstractmethod
    def solveIncrement(self, *args):
        pass

    @performancetiming.timeit("dirichlet R")
    def applyDirichletToResidual(self, timeStep: TimeStep, R: DofVector, dirichlets: list[StepActionBase]):
        """Impose the Dirichlet BCs on the residual using the row-replacement method.

        For every constrained DOF we *overwrite* its residual entry with the
        value we want the linear solve to return for that DOF's increment.
        Together with :meth:`applyDirichletToStiffness` (which zeroes the DOF's
        row of K and puts 1.0 on the diagonal), the linearized system
        ``K ddU = R`` then reproduces exactly that increment for the DOF.

        Parameters
        ----------
        timeStep
            The current time step.
        R
            The residual vector of the global equation system to be modified.
        dirichlets
            The list of dirichlet boundary conditions.

        Returns
        -------
        DofVector
            The modified residual vector.
        """
        for dirichlet in dirichlets:
            R[dirichlet.constrainedDofIndices] = dirichlet.getPrescribedIncrement(timeStep).flatten()

        return R

    @performancetiming.timeit("convergence check")
    def checkConvergence(
        self,
        R: DofVector,
        ddU: DofVector,
        F: DofVector,
        iterationCounter: int,
        residualHistory: dict,
    ) -> tuple[bool, dict]:
        """Check the convergence, individually for each field,
        similar to Abaqus based on the current total flux residual and the field correction
        Is called by solveStep() to decide whether to continue iterating or stop.

        Parameters
        ----------
        R
            The current residual.
        ddU
            The current correction increment.
        F
            The accumulated fluxes.
        iterationCounter
            The current iteration number.
        residualHistory
            The previous residuals.

        Returns
        -------
        tuple[bool,dict]
            - True if converged.
            - The residual histories field wise.

        """

        iterationMessage = ""
        convergedAtAll = True
        nodesWithLargestResidual = {}

        spatialAveragedFluxes = self.computeSpatialAveragedFluxes(F)

        if iterationCounter < 15:  # standard tolerance set
            fluxResidualTolerances = self.fluxResidualTolerances
        else:  # alternative tolerance set
            fluxResidualTolerances = self.fluxResidualTolerancesAlt

        for field, fieldIndices in self.theDofManager.idcsOfFieldsInDofVector.items():
            fieldResidualAbs = np.abs(R[fieldIndices])

            indexOfMax = np.argmax(fieldResidualAbs)
            fluxResidual = fieldResidualAbs[indexOfMax]

            nodesWithLargestResidual[field] = self.theDofManager.getNodeForIndexInDofVector(indexOfMax)

            fieldCorrection = np.linalg.norm(ddU[fieldIndices], np.inf) if ddU is not None else 0.0

            convergedCorrection = fieldCorrection < self.fieldCorrectionTolerances[field]
            convergedFlux = fluxResidual <= max(fluxResidualTolerances[field] * spatialAveragedFluxes[field], 1e-7)

            previousFluxResidual, nGrew = residualHistory[field]
            if fluxResidual > previousFluxResidual:
                nGrew += 1
            residualHistory[field] = (fluxResidual, nGrew)

            iterationMessage += self.iterationMessageTemplate.format(
                fluxResidual,
                "✓" if convergedFlux else " ",
                fieldCorrection,
                "✓" if convergedCorrection else " ",
            )
            convergedAtAll = convergedAtAll and convergedCorrection and convergedFlux

        if self.theDofManager.idcsOfScalarVariablesInDofVector:
            residualScalarVariables = max(np.abs(R[list(self.theDofManager.idcsOfScalarVariablesInDofVector.values())]))
            correction = (
                np.linalg.norm(
                    ddU[list(self.theDofManager.idcsOfScalarVariablesInDofVector.values())],
                    np.inf,
                )
                if ddU is not None
                else 0.0
            )

            convergedCorrection = correction < self.fieldCorrectionTolerances["scalar variables"]
            convergedFlux = residualScalarVariables <= fluxResidualTolerances["scalar variables"]

            iterationMessage += self.iterationMessageTemplate.format(
                residualScalarVariables,
                "✓" if convergedFlux else " ",
                correction,
                "✓" if convergedCorrection else " ",
            )

            convergedAtAll = convergedAtAll and convergedCorrection and convergedFlux

        self.journal.message(iterationMessage, self.identification)

        return convergedAtAll, nodesWithLargestResidual

    @performancetiming.timeit("linear solve")
    def linearSolve(self, A: csr_matrix, b: DofVector) -> ndarray:
        """Solve the linear equation system.

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
    def assembleStiffnessCSR(self, K: VIJSystemMatrix) -> csr_matrix:
        """Construct a CSR matrix from VIJ format.

        Parameters
        ----------
        K
            The system matrix in VIJ format.
        Returns
        -------
        csr_matrix
            The system matrix in compressed sparse row format.
        """
        # In-place update: the returned matrix is the generator's internal CSR matrix.
        # This is safe since no solver retains it across iterations, and the subsequent
        # Dirichlet application only modifies values (the pattern is preserved), which
        # are fully overwritten again on the next update.
        KCsr = self.csrGenerator.updateInPlace(K)
        return KCsr

    def computeSpatialAveragedFluxes(self, F: DofVector) -> dict[str, float]:
        """Compute the spatial averaged flux for every field
        Is usually called by checkConvergence().

        Parameters
        ----------
        F
            The accumulated flux vector.

        Returns
        -------
        dict[str,float]
            A dictioary containg the spatial average fluxes for every field.
        """
        spatialAveragedFluxes = dict.fromkeys(self.theDofManager.idcsOfFieldsInDofVector, 0.0)
        for field, nDof in self.theDofManager.nAccumulatedNodalFluxesFieldwise.items():
            spatialAveragedFluxes[field] = max(
                1e-10,
                np.linalg.norm(F[self.theDofManager.idcsOfFieldsInDofVector[field]], 1) / nDof,
            )

        return spatialAveragedFluxes

    def extrapolateLastIncrement(
        self,
        extrapolation: str,
        timeStep: TimeStep,
        dU: DofVector,
        dirichlets: list,
        prevTimeStep: TimeStep,
        model,
    ) -> tuple[DofVector, bool]:
        """Depending on the current setting, extrapolate the solution of the last increment.

        Parameters
        ----------
        extrapolation
            The type of extrapolation.
        timeStep
            The current time step.
        dU
            The last solution increment.
        dirichlets
            The list of active dirichlet boundary conditions.
        lastIncrementSize
            The size of the last increment.

        Returns
        -------
        tuple[DofVector,bool]
            - The extrapolated solution increment.
            - True if an extrapolation was performed.
        """

        if extrapolation == "linear" and prevTimeStep and prevTimeStep.timeIncrement:
            dU *= timeStep.stepProgressIncrement / prevTimeStep.stepProgressIncrement
            dU = self.applyDirichletToResidual(timeStep, dU, dirichlets)
            isExtrapolatedIncrement = True
        else:
            isExtrapolatedIncrement = False
            dU[:] = 0.0

        return dU, isExtrapolatedIncrement

    def checkDivergingSolution(self, incrementResidualHistory: dict, maxGrowingIter: int) -> bool:
        """Check if the iterative solution scheme is diverging.

        Parameters
        ----------
        incrementResidualHistory
            The dictionary containing the residual history of all fields.
        maxGrowingIter
            The maximum allows number of growths of a residual during the iterative solution scheme.

        Returns
        -------
        bool
            True if solution is diverging.
        """
        for previousFluxResidual, nGrew in incrementResidualHistory.values():
            if nGrew > maxGrowingIter:
                return True
        return False

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

    def applyStepActionsAtStepStart(self, model: FEModel, stepActions: dict[str, StepActionBase]):
        """Called when all step actions should be appliet at the start a step.

        Parameters
        ----------
        model
            The model tree.
        stepActions
            The dictionary of active step actions.
        """

        for stepActionType in stepActions.values():
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

        Every nonlinear solver must call this once per converged increment. It lives on the base
        class so that solvers overriding :meth:`solveStep` (e.g. the parallel and arc-length
        variants) stay consistent with the serial implementation instead of silently omitting it.

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

        Called once when a step's boundary conditions are established. The result
        is cached on each BC as :attr:`~DirichletBase.constrainedDofIndices`, so
        that the Newton loop can address the constrained DOFs directly, instead
        of recomputing the mapping on every residual update and every stiffness
        modification.

        Parameters
        ----------
        dirichlets
            The list of dirichlet boundary conditions active in this step.
        """
        for dirichlet in dirichlets:
            dirichlet.constrainedDofIndices = self._constrainedDofsOf(dirichlet)

    def _constrainedDofsOf(self, dirichlet: StepActionBase) -> np.ndarray:
        """Return the global DOF indices prescribed by a single Dirichlet BC.

        The DofManager knows every DOF of ``field`` on ``nSet``, laid out node
        by node in a single flat array::

            [ node0: (u_x u_y u_z),  node1: (u_x u_y u_z),  ... ]

        A BC usually prescribes only some of the per-node components (given by
        ``dirichlet.components``, e.g. just u_x and u_z). So we view the flat
        array as one row per node, keep only the prescribed component columns,
        and flatten it back into a plain list of global DOF indices. The order
        stays node-major, matching ``getPrescribedIncrement().flatten()``.
        """
        dofsOfFieldOnNodeSet = self.theDofManager.idcsOfFieldsOnNodeSetsInDofVector[dirichlet.field][dirichlet.nSet]
        perNodeDofs = dofsOfFieldOnNodeSet.reshape((-1, dirichlet.fieldSize))

        return perNodeDofs[:, dirichlet.components].flatten()

    def buildMPCTransformation(self, model: FEModel):
        """Collect the linear dependency records from all multi-point constraints of the model
        and assemble the master-slave condensation operator for the current equation system.
        Must be called whenever the DofManager is (re)built.

        Parameters
        ----------
        model
            The model tree.

        Returns
        -------
        MultiPointConstraintTransformation | None
            The assembled transformation, or None if the model has no multi-point constraints.
        """

        if not model.multiPointConstraints:
            return None

        if not self.supportsMPC:
            raise NotImplementedError(
                f"Multi-point constraints (e.g. surface ties) are not supported by the {self.identification} solver."
            )

        records = [
            record
            for mpc in model.multiPointConstraints.values()
            for record in mpc.getMultiPointConstraints(self.theDofManager)
        ]

        transformation = MultiPointConstraintTransformation(records, self.theDofManager.nDof)

        self.journal.message(
            "eliminating {:} slave DOF(s) via multi-point constraints".format(transformation.nEliminatedDof),
            self.identification,
            0,
        )

        return transformation

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

        for dirichlet in stepActions["dirichlet"].values():
            transformation.checkDirichletConflicts(self._constrainedDofsOf(dirichlet))
