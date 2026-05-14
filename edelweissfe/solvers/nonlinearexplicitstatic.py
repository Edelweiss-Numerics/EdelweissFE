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
#  Alexander Dummer alexander.dummer@uibk.ac.at
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

import json

import numpy as np

import edelweissfe.utils.performancetiming as performancetiming
from edelweissfe.config.linsolve import getLinSolverByName
from edelweissfe.config.timing import createTimingDict
from edelweissfe.models.femodel import FEModel
from edelweissfe.numerics.csrgenerator import CSRGenerator
from edelweissfe.numerics.dofmanager import DofManager, DofVector, VIJSystemMatrix
from edelweissfe.outputmanagers.base.outputmanagerbase import OutputManagerBase
from edelweissfe.solvers.nonlinearimplicitstatic import NIST
from edelweissfe.timesteppers.timestep import TimeStep
from edelweissfe.utils.exceptions import (
    ConditionalStop,
    CutbackRequest,
    ReachedMaxIncrements,
    ReachedMinIncrementSize,
    StepFailed,
)
from edelweissfe.utils.fieldoutput import FieldOutputController


def getRungeKuttaParameters(rungeKuttaStages: int) -> tuple[dict, dict, dict]:

    _alpha = {}
    _omega = {}
    _lambda = {}

    if rungeKuttaStages == 2:
        # parameters for stage 2
        _alpha[21] = 1

        # parameters for increment estimate
        _omega[1] = 0.5
        _omega[2] = 0.5

        # parameters for error control
        _lambda[1] = -0.5
        _lambda[2] = 0.5
    elif rungeKuttaStages == 3:
        # parameters for stage 2
        _alpha[21] = 2 / 3
        # parameters for stage 3
        _alpha[31] = 0
        _alpha[32] = 2 / 3

        # parameters for increment estimate
        _omega[1] = 2 / 8
        _omega[2] = 3 / 8
        _omega[3] = 3 / 8

        # parameters for error control
        _lambda[1] = 0
        _lambda[2] = -3 / 8
        _lambda[3] = 3 / 8
    elif rungeKuttaStages == 4:
        # parameters for stage 2
        _alpha[21] = -0.5
        # parameters for stage 3
        _alpha[31] = 0.6684895833
        _alpha[32] = -0.2434895833
        # parameters for stage 4
        _alpha[41] = -2.323685857
        _alpha[42] = 1.125483559
        _alpha[43] = 2.198202298

        # parameters for increment estimate
        _omega[1] = 0.03431372549
        _omega[2] = 0.02705627706
        _omega[3] = 0.7440130202
        _omega[4] = 0.1946169772

        # parameters for error control
        _lambda[1] = 0.03431372549
        _lambda[2] = -0.01262626262
        _lambda[3] = -0.0289338397
        _lambda[4] = 0.00724637679

    elif rungeKuttaStages == 6:
        # parameters for stage 2
        _alpha[21] = 0.2
        # parameters for stage 3
        _alpha[31] = 3 / 40
        _alpha[32] = 9 / 40
        # parameters for stage 4
        _alpha[41] = 3 / 10
        _alpha[42] = -9 / 10
        _alpha[43] = 6 / 5
        # parameters for stage 5
        _alpha[51] = -11 / 54
        _alpha[52] = 5 / 2
        _alpha[53] = -70 / 27
        _alpha[54] = 35 / 27
        # parameters for stage 6
        _alpha[61] = 1631 / 55296
        _alpha[62] = 175 / 512
        _alpha[63] = 575 / 13824
        _alpha[64] = 44275 / 110592
        _alpha[65] = 253 / 4096

        # parameters for increment estimate
        _omega[1] = 37 / 378
        _omega[2] = 0
        _omega[3] = 250 / 621
        _omega[4] = 125 / 594
        _omega[5] = 0
        _omega[6] = 512 / 1771

        # parameters for error control
        _lambda[1] = -277 / 64512
        _lambda[2] = 0
        _lambda[3] = 6925 / 370944
        _lambda[4] = -6925 / 202752
        _lambda[5] = -277 / 14336
        _lambda[6] = 277 / 7084

    else:
        raise NotImplementedError("runge-kutta-stages must be 2, 3, 4 or 6")

    return _alpha, _omega, _lambda


class NEST(NIST):
    """This is the Nonlinear Explicit STatic -- solver.

    Parameters
    ----------
    jobInfo
        A dictionary containing the job information.
    journal
        The journal instance for logging.
    """

    identification = "NESTSolver"

    SolverSpecificOptions = {
        "runge-kutta-stages": 2,
        "runge-kutta-error-tolerance": 1e-3,
        "runge-kutta-error-control": "on",
        "linsolver": "pardiso",
        "linsolverConfigFile": "",
    }

    def __init__(self, jobInfo, journal, **kwargs):
        self.journal = journal

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

        self.computationTimes = createTimingDict()

        _optionsUpdate = step.actions["options"].get("NESTSolver", {})
        self._updateOptions(_optionsUpdate, self.journal)

        # get parameters for runge kutta scheme
        self.rkAlpha, self.rkOmega, self.rkLambda = getRungeKuttaParameters(self.options.get("runge-kutta-stages", 2))
        self.rkStages = self.options.get("runge-kutta-stages", 2)

        self.tol = self.options.get("runge-kutta-error-tolerance", 1e-3)

        linsolverOptions = self.options["linsolverConfigFile"]
        linsolverOptionDict = {}
        if linsolverOptions:
            with open(linsolverOptions, "r") as f:
                linsolverOptionDict = json.load(f)
        self.linSolver = getLinSolverByName(self.options.get("linsolver", "default"), linsolverOptionDict)

        U = self.theDofManager.constructDofVector()
        P = self.theDofManager.constructDofVector()
        dU = self.theDofManager.constructDofVector()
        K = self.theDofManager.constructVIJSystemMatrix()
        self.csrGenerator = CSRGenerator(K)

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
                try:
                    U, dU, P, incScaleFactor = self.solveIncrement(
                        U,
                        dU,
                        P,
                        K,
                        step.actions,
                        model,
                        timeStep,
                        prevTimeStep,
                    )

                    step.changeIncrementSize(incScaleFactor)

                except CutbackRequest as e:
                    self.journal.message(str(e), self.identification, 1)
                    step.discardAndChangeIncrement(max(e.cutbackSize, 0.25))
                    prevTimeStep = None

                    statusInfoDict["notes"] = str(e)

                    for man in outputmanagers:
                        man.finalizeFailedIncrement(
                            statusInfoDict=statusInfoDict,
                            currentComputingTimes=self.computationTimes,
                        )

                else:
                    prevTimeStep = timeStep

                    # write results to nodes:
                    for fieldName, field in model.nodeFields.items():
                        self.theDofManager.writeDofVectorToNodeField(U, field, "U")
                        self.theDofManager.writeDofVectorToNodeField(P, field, "P")
                        self.theDofManager.writeDofVectorToNodeField(dU, field, "dU")

                    for variable in model.scalarVariables.values():
                        variable.value = U[self.theDofManager.idcsOfScalarVariablesInDofVector[variable]]

                    model.advanceToTime(timeStep.totalTime)

                    fieldOutputController.finalizeIncrement()
                    for man in outputmanagers:
                        man.finalizeIncrement(
                            currentComputingTimes=self.computationTimes,
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
    ) -> tuple[DofVector, DofVector, DofVector, float]:
        """Explicit Runge-Kutta scheme to solve for an increment.

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
        timeStep
            The current time step.
        prevTimeStep
            The previous time step.

        Returns
        -------
        tuple[DofVector,DofVector,DofVector,int,dict]
            A tuple containing
                - the new solution vector
                - the solution increment
                - the new reaction vector
                - the scale factor for the next increment size
        """

        elements = model.elements
        constraints = model.constraints

        R = self.theDofManager.constructDofVector()
        F = self.theDofManager.constructDofVector()
        PExt = self.theDofManager.constructDofVector()
        U_np = self.theDofManager.constructDofVector()
        dU_ = [self.theDofManager.constructDofVector() for i in range(self.rkStages)]
        err = self.theDofManager.constructDofVector()
        err[:] = 0.0

        dirichlets = stepActions["dirichlet"].values()
        nodeforces = stepActions["nodeforces"].values()
        distributedLoads = stepActions["distributedload"].values()
        bodyForces = stepActions["bodyforce"].values()

        self.applyStepActionsAtIncrementStart(model, timeStep, stepActions)

        for geostatic in stepActions["geostatic"].values():
            geostatic.applyAtIterationStart()

        dU[:] = 0.0

        # iterate over runge kutta stages
        for k in range(self.rkStages):
            U_np[:] = U_n
            dU_[k][:] = 0.0

            if k > 0:
                for j in range(k + 1 - 1):
                    a = (k + 1) * 10 + 1 + j
                    dU_[k] += self.rkAlpha[a] * dU_[j]

            U_np += dU_[k]

            P[:] = K[:] = F[:] = PExt[:] = 0.0

            P, K, F = self.computeElements(elements, U_np, dU_[k], P, K, F, timeStep)
            PExt, K = self.assembleLoads(nodeforces, distributedLoads, bodyForces, U_np, PExt, K, timeStep)
            PExt, K = self.assembleConstraints(constraints, U_np, dU_[k], PExt, K, timeStep)

            R[:] = P
            R += PExt

            R = self.applyDirichlet(timeStep, R, dirichlets)

            K_ = self.assembleStiffnessCSR(K)
            K_ = self.applyDirichletK(K_, dirichlets)

            # solve for increment
            dU_[k] = self.linearSolve(K_, R)

            # update solution increment
            dU += self.rkOmega[k + 1] * dU_[k]

            # update error measure
            err += self.rkLambda[k + 1] * dU_[k]

        # update solution
        U_np[:] = U_n
        U_np += dU

        normErr = np.linalg.norm(err) / (np.linalg.norm(U_np) + 1e-16)
        self.journal.message("Estimated error {:5.2e}".format(normErr), self.identification, 2)

        if normErr > self.tol and self.options.get("runge-kutta-error-control", "on") != "off":
            raise CutbackRequest("Estimated error too high!", np.sqrt(self.tol / normErr) * 0.9)

        # compute elements once more to get the final reaction
        P[:] = K[:] = F[:] = PExt[:] = 0.0
        P, K, F = self.computeElements(elements, U_np, dU, P, K, F, timeStep)

        # compute new increment size scale factor
        if self.options.get("runge-kutta-error-control", "on") == "off":
            beta = 1.0
        else:
            beta = max(0.1, min(np.sqrt(self.tol / (normErr + 1e-15)) * 0.9, 2))

        return U_np, dU, P, beta
