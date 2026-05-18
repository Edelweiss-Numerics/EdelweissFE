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

import numpy as np
from scipy.sparse import csr_matrix

import edelweissfe.utils.performancetiming as performancetiming
from edelweissfe.config.linsolve import getDefaultLinSolver, getLinSolverByName
from edelweissfe.constraints.base.constraintbase import ConstraintBase
from edelweissfe.models.femodel import FEModel
from edelweissfe.numerics.csrgenerator import CSRGenerator
from edelweissfe.numerics.dofmanager import DofManager, DofVector, VIJSystemMatrix
from edelweissfe.outputmanagers.base.outputmanagerbase import OutputManagerBase
from edelweissfe.solvers.base.dirichlet import applyDirichletK
from edelweissfe.solvers.base.nonlinearsolverbase import NonlinearSolverBase
from edelweissfe.stepactions.base.stepactionbase import StepActionBase
from edelweissfe.stepactions.options import inputLanguage
from edelweissfe.timesteppers.timestep import TimeStep
from edelweissfe.utils.exceptions import (
    ConditionalStop,
    CutbackRequest,
    DivergingSolution,
    ReachedMaxIncrements,
    ReachedMaxIterations,
    ReachedMinIncrementSize,
    StepFailed,
)
from edelweissfe.utils.fieldoutput import FieldOutputController

kw = inputLanguage["step"].getModule("adaptive").getKeyword("options")
kw.addOptionalArg("defaultMaxIter", "", int, 10)
kw.addOptionalArg("defaultCriticalIter", "", int, 5)
kw.addOptionalArg("defaultMaxGrowingIter", "", int, 10)
kw.addOptionalArg("extrapolation", "", str, "linear")
kw.addOptionalArg("linsolver", "", str, "pardiso")
kw.addOptionalArg("linsolverConfigFile", "", str, "")


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

    SolverSpecificOptions = {
        "defaultMaxIter": 10,
        "defaultCriticalIter": 5,
        "defaultMaxGrowingIter": 10,
        "extrapolation": "linear",
        "linsolver": "pardiso",
        "linsolverConfigFile": "",
    }

    def __init__(self, jobInfo, journal, **kwargs):
        self.journal = journal

        self.fieldCorrectionTolerances = jobInfo["fieldCorrectionTolerance"]
        self.fluxResidualTolerances = jobInfo["fluxResidualTolerance"]
        self.fluxResidualTolerancesAlt = jobInfo["fluxResidualToleranceAlternative"]

        self.options = self.SolverSpecificOptions.copy()
        self._updateOptions(kwargs, journal)

    def solveStep(
        self,
        step,
        model: FEModel,
        fieldOutputController: FieldOutputController,
        outputmanagers: dict[str, OutputManagerBase],
    ) -> tuple[bool, FEModel]:
        """Public interface to solve for a step.

        Parameters
        ----------
        stepNumber
            The step number.
        step
            The dictionary containing the step definition.
        stepActions
            The dictionary containing all step actions.
        model
            The  model tree.
        fieldOutputController
            The field output controller.
        """

        self.journal.message("Creating monolithic equation system", self.identification, 0)
        self.theDofManager = DofManager(
            model.nodeFields.values(),
            model.scalarVariables.values(),
            model.elements.values(),
            model.constraints.values(),
            model.nodeSets.values(),
        )
        self.journal.message(
            "total size of eq. system: {:}".format(self.theDofManager.nDof),
            self.identification,
            0,
        )

        self.journal.printSeperationLine()

        presentVariableNames = list(self.theDofManager.idcsOfFieldsInDofVector.keys())

        if self.theDofManager.idcsOfScalarVariablesInDofVector:
            presentVariableNames += [
                "scalar variables",
            ]

        nVariables = len(presentVariableNames)
        self.iterationHeader = ("{:^25}" * nVariables).format(*presentVariableNames)
        self.iterationHeader2 = (" {:<10}  {:<10}  ").format("||R||∞", "||ddU||∞") * nVariables
        self.iterationMessageTemplate = "{:11.2e}{:1}{:11.2e}{:1} "

        U = self.theDofManager.constructDofVector()
        K = self.theDofManager.constructVIJSystemMatrix()

        self.csrGenerator = CSRGenerator(K)

        try:
            self._updateOptions(step.actions["options"]["NISTSolver"].options, self.journal)
        except KeyError:
            pass

        extrapolation = self.options["extrapolation"]
        linsolverOptions = self.options["linsolverConfigFile"]
        linsolverOptionDict = json.load(open(linsolverOptions, "r")) if linsolverOptions else ""
        self.linSolver = (
            getLinSolverByName(self.options["linsolver"], linsolverOptionDict)
            if "linsolver" in self.options
            else getDefaultLinSolver()
        )

        maxIter = step.maxIter
        criticalIter = step.criticalIter
        maxGrowingIter = step.maxGrowIter
        cutbackFactor = step.cutbackFactor

        U = self.theDofManager.constructDofVector()
        P = self.theDofManager.constructDofVector()
        dU = self.theDofManager.constructDofVector()

        for fieldName, field in model.nodeFields.items():
            U = self.theDofManager.writeNodeFieldToDofVector(U, field, "U")

        for variable in model.scalarVariables.values():
            U[self.theDofManager.idcsOfScalarVariablesInDofVector[variable]] = variable.value

        prevTimeStep = None

        self.applyStepActionsAtStepStart(model, step.actions)

        try:
            for timeStep in step.getTimeStep():
                statusInfoDict = {
                    "step": step.number,
                    "inc": timeStep.number,
                    "iters": None,
                    "converged": False,
                    "time inc": timeStep.timeIncrement,
                    "time end": timeStep.totalTime,
                    "notes": "",
                }

                self.journal.printSeperationLine()
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
                self.journal.message(self.iterationHeader, self.identification, level=2)
                self.journal.message(self.iterationHeader2, self.identification, level=2)

                try:
                    U, dU, P, iterationCounter, incrementResidualHistory = self.solveIncrement(
                        U,
                        dU,
                        P,
                        K,
                        step.actions,
                        model,
                        timeStep,
                        prevTimeStep,
                        extrapolation,
                        maxIter,
                        maxGrowingIter,
                    )

                except CutbackRequest as e:
                    self.journal.message(str(e), self.identification, 1)
                    step.discardAndChangeIncrement(max(e.cutbackSize, cutbackFactor))
                    prevTimeStep = None

                    statusInfoDict["iters"] = np.inf
                    statusInfoDict["notes"] = str(e)

                    for man in outputmanagers:
                        man.finalizeFailedIncrement(
                            statusInfoDict=statusInfoDict,
                        )

                except (ReachedMaxIterations, DivergingSolution) as e:
                    self.journal.message(str(e), self.identification, 1)
                    step.discardAndChangeIncrement(cutbackFactor)
                    prevTimeStep = None

                    statusInfoDict["iters"] = np.inf
                    statusInfoDict["notes"] = str(e)

                    for man in outputmanagers:
                        man.finalizeFailedIncrement(
                            statusInfoDict=statusInfoDict,
                        )

                else:
                    prevTimeStep = timeStep

                    if iterationCounter >= criticalIter:
                        step.preventIncrementIncrease()

                    # write results to nodes:
                    for fieldName, field in model.nodeFields.items():
                        self.theDofManager.writeDofVectorToNodeField(U, field, "U")
                        self.theDofManager.writeDofVectorToNodeField(P, field, "P")
                        self.theDofManager.writeDofVectorToNodeField(dU, field, "dU")

                    for variable in model.scalarVariables.values():
                        variable.value = U[self.theDofManager.idcsOfScalarVariablesInDofVector[variable]]

                    model.advanceToTime(timeStep.totalTime)

                    self.journal.message(
                        "Converged in {:} iteration(s)".format(iterationCounter),
                        self.identification,
                        1,
                    )

                    statusInfoDict["iters"] = iterationCounter
                    statusInfoDict["converged"] = True

                    fieldOutputController.finalizeIncrement()
                    for man in outputmanagers:
                        man.finalizeIncrement(
                            statusInfoDict=statusInfoDict,
                        )

        except (ReachedMaxIncrements, ReachedMinIncrementSize):
            self.journal.errorMessage("Incrementation failed", self.identification)
            raise StepFailed()

        except ConditionalStop:
            self.journal.message("Conditional Stop", self.identification)
            self.applyStepActionsAtStepEnd(model, step.actions)

        else:
            self.applyStepActionsAtStepEnd(model, step.actions)

        finally:
            prettyTable = performancetiming.makePrettyTable()
            self.journal.printPrettyTable(prettyTable, self.identification)
            performancetiming.reset()

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
        PExt = self.theDofManager.constructDofVector()
        U_np = self.theDofManager.constructDofVector()
        ddU = None

        dirichlets = stepActions["dirichlet"].values()
        nodeforces = stepActions["nodeforces"].values()
        distributedLoads = stepActions["distributedload"].values()
        bodyForces = stepActions["bodyforce"].values()

        self.applyStepActionsAtIncrementStart(model, timeStep, stepActions)

        dU, isExtrapolatedIncrement = self.extrapolateLastIncrement(
            extrapolation, timeStep, dU, dirichlets, prevTimeStep, model
        )

        while True:
            for geostatic in stepActions["geostatic"].values():
                geostatic.applyAtIterationStart()

            U_np[:] = U_n
            U_np += dU

            P[:] = K[:] = F[:] = PExt[:] = 0.0

            P, K, F = self.computeElements(elements, U_np, dU, P, K, F, timeStep)
            PExt, K = self.assembleLoads(nodeforces, distributedLoads, bodyForces, U_np, PExt, K, timeStep)
            PExt, K = self.assembleConstraints(constraints, U_np, dU, PExt, K, timeStep)

            R[:] = P
            R += PExt

            if iterationCounter == 0 and not isExtrapolatedIncrement and dirichlets:
                # first iteration? apply dirichlet bcs and unconditionally solve
                R = self.applyDirichlet(timeStep, R, dirichlets)
            else:
                # iteration cycle 1 or higher, time to check the convergence
                for dirichlet in dirichlets:
                    R[self.findDirichletIndices(dirichlet)] = 0.0

                converged, nodesWithLargestResidual = self.checkConvergence(
                    R, ddU, F, iterationCounter, incrementResidualHistory
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
            K_ = self.applyDirichletK(K_, dirichlets)

            ddU = self.linearSolve(K_, R)
            dU += ddU
            iterationCounter += 1

        return U_np, dU, P, iterationCounter, incrementResidualHistory

    @performancetiming.timeit("distributed loads")
    def computeDistributedLoads(
        self,
        distributedLoads: list[StepActionBase],
        U_np: DofVector,
        PExt: DofVector,
        K: VIJSystemMatrix,
        timeStep: TimeStep,
    ) -> tuple[DofVector, VIJSystemMatrix]:
        """Loop over all distributed loads acting on elements, and evaluate them.
        Assembles into the global external load vector and the system matrix.

        Parameters
        ----------
        distributedLoads
            The list of distributed loads.
        U_np
            The current solution vector.
        PExt
            The external load vector to be augmented.
        K
            The system matrix to be augmented.
        timeStep
            The current time step.

        Returns
        -------
        tuple[DofVector,VIJSystemMatrix]
            The augmented load vector and system matrix.
        """

        time = np.array([timeStep.stepTime, timeStep.totalTime])
        dT = timeStep.timeIncrement

        for dLoad in distributedLoads:
            load = dLoad.getCurrentLoad(timeStep)
            for faceID, elementSet in dLoad.surface.items():
                for el in elementSet:
                    Ke = K[el]
                    Pe = np.zeros(el.nDof)

                    el.computeDistributedLoad(dLoad.loadType, Pe, Ke, faceID, load, U_np[el], time, dT)

                    PExt[el] += Pe

        return PExt, K

    @performancetiming.timeit("body forces")
    def computeBodyForces(
        self,
        bodyForces: list[StepActionBase],
        U_np: DofVector,
        PExt: DofVector,
        K: VIJSystemMatrix,
        timeStep: TimeStep,
    ) -> tuple[DofVector, VIJSystemMatrix]:
        """Loop over all body forces loads acting on elements, and evaluate them.
        Assembles into the global external load vector and the system matrix.

        Parameters
        ----------
        distributedLoads
            The list of distributed loads.
        U_np
            The current solution vector.
        PExt
            The external load vector to be augmented.
        K
            The system matrix to be augmented.
        increment
            The increment.

        Returns
        -------
        tuple[DofVector,VIJSystemMatrix]
            The augmented load vector and system matrix.
        """

        time = np.array([timeStep.stepTime, timeStep.totalTime])
        dT = timeStep.timeIncrement

        for bForce in bodyForces:
            force = bForce.getCurrentLoad(timeStep)
            for el in bForce.elementSet:
                Pe = np.zeros(el.nDof)
                Ke = K[el]

                el.computeBodyForce(Pe, Ke, force, U_np[el], time, dT)

                PExt[el] += Pe

        return PExt, K

    @performancetiming.timeit("dirichlet K on CSR")
    def applyDirichletK(self, K: csr_matrix, dirichlets: list[StepActionBase]) -> csr_matrix:
        return applyDirichletK(self, K, dirichlets)

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
        Is is called by solveStep() in each iteration.

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

        time = np.array([timeStep.stepTime, timeStep.totalTime])
        dT = timeStep.timeIncrement

        for el in elements.values():
            Ke = K[el]
            Pe = np.zeros(el.nDof)

            el.computeYourself(Ke, Pe, U_np[el], dU[el], time, dT)

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
    ) -> tuple[DofVector, VIJSystemMatrix]:
        """Loop over all elements, and evaluate them.
        Is is called by solveStep() in each iteration.

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
        dT
            The time increment.
        time
            The step and total time.

        Returns
        -------
        tuple[DofVector,VIJSystemMatrix,DofVector]
            - The modified external load vector.
            - The modified system matrix.
        """

        for constraint in constraints.values():
            Kc = K[constraint].reshape(constraint.nDof, constraint.nDof, order="F")
            Pc = np.zeros(constraint.nDof)

            constraint.applyConstraint(U_np[constraint], dU[constraint], Pc, Kc, timeStep)

            # instead of PExt[constraint] += Pe, np.add.at allows for repeated indices
            np.add.at(PExt, PExt.entitiesInDofVector[constraint], Pc)

        return PExt, K

    @performancetiming.timeit("assemble loads")
    def assembleLoads(
        self,
        nodeForces: list[StepActionBase],
        distributedLoads: list[StepActionBase],
        bodyForces: list[StepActionBase],
        U_np: DofVector,
        PExt: DofVector,
        K: VIJSystemMatrix,
        timeStep: TimeStep,
    ) -> tuple[DofVector, VIJSystemMatrix]:
        """Assemble all loads into a right hand side vector.

        Parameters
        ----------
        nodeForces
            The list of concentrated (nodal) loads.
        distributedLoads
            The list of distributed (surface) loads.
        bodyForces
            The list of body (volumetric) loads.
        U_np
            The current solution vector.
        PExt
            The external load vector.
        K
            The system matrix.
        timeStep
            The current time step.

        Returns
        -------
        tuple[DofVector,VIJSystemMatrix]
            - The augmented external load vector.
            - The augmented system matrix.
        """
        for cLoad in nodeForces:
            PExt[
                self.theDofManager.idcsOfFieldsOnNodeSetsInDofVector[cLoad.field][cLoad.nodeSet]
            ] += cLoad.getCurrentLoad(timeStep).flatten()
        PExt, K = self.computeDistributedLoads(distributedLoads, U_np, PExt, K, timeStep)
        PExt, K = self.computeBodyForces(bodyForces, U_np, PExt, K, timeStep)

        return PExt, K
