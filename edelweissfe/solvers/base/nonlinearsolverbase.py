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
from edelweissfe.stepactions.base.stepactionbase import StepActionBase
from edelweissfe.timesteppers.timestep import TimeStep
from edelweissfe.utils.exceptions import DivergingSolution


class NonlinearSolverBase(ABC):
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

    def __init__(self, jobInfo, journal, **kwargs):
        pass

    def _updateOptions(self, updatedOptions: dict, journal):
        """Update options of the solver using a string dict

        Parameters
        ----------
        updatedOptions
            The options dictionary.
        journal
            The journal module.
        """

        for k, v in updatedOptions.items():
            if k in self.SolverSpecificOptions:
                journal.message("Updating option {:}={:}".format(k, v), self.identification)
                self.options[k] = type(self.SolverSpecificOptions[k])(updatedOptions[k])
            else:
                raise AttributeError("Invalid option {:} for {:}".format(k, self.identification))

    @abstractmethod
    def solveStep(self, *args):
        pass

    @abstractmethod
    def solveIncrement(self, *args):
        pass

    @performancetiming.timeit("dirichlet R")
    def applyDirichlet(self, timeStep: TimeStep, R: DofVector, dirichlets: list[StepActionBase]):
        """Apply the dirichlet bcs on the residual vector
        Is called by solveStep() before solving the global equatuon system.

        Parameters
        ----------
        increment
            The increment.
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
            delta = dirichlet.getDelta(timeStep)
            R[self.findDirichletIndices(dirichlet)] = delta.flatten()

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
        KCsr = self.csrGenerator.updateCSR(K)
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
            dU = self.applyDirichlet(timeStep, dU, dirichlets)
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

    def findDirichletIndices(self, dirichlet):
        nSet = dirichlet.nSet
        field = dirichlet.field
        components = dirichlet.components

        fieldIndices = self.theDofManager.idcsOfFieldsOnNodeSetsInDofVector[field][nSet]

        return fieldIndices.reshape((len(nSet), -1))[:, components].flatten()
