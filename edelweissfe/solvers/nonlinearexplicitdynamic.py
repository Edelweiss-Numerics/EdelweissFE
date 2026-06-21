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


from copy import deepcopy

import numpy as np

import edelweissfe.utils.performancetiming as performancetiming
from edelweissfe.models.femodel import FEModel
from edelweissfe.numerics.dofmanager import DofManager, DofVector, VIJSystemMatrix
from edelweissfe.outputmanagers.base.outputmanagerbase import OutputManagerBase
from edelweissfe.solvers.base.nonlinearsolverbase import NonlinearSolverBase
from edelweissfe.stepactions.base.stepactionbase import StepActionBase
from edelweissfe.timesteppers.timestep import TimeStep
from edelweissfe.utils.exceptions import (
    ConditionalStop,
    CutbackRequest,
    ReachedMaxIncrements,
    ReachedMinIncrementSize,
    StepFailed,
)
from edelweissfe.utils.fieldoutput import FieldOutputController


class NED(NonlinearSolverBase):
    """This is the Nonlinear Explicit Dynamic -- solver.

    Parameters
    ----------
    jobInfo
        A dictionary containing the job information.
    journal
        The journal instance for logging.
    """

    identification = "NEDSolver"

    NEDOptions = {
        "first-order-fields": [],
        "second-order-fields": [],
        "first-order-scheme": "forward-euler",
        "second-order-scheme": "central-difference",
        "courant-number": 0.8,
        "output-frequency": 1000,
    }

    def __init__(self, jobInfo, journal, **kwargs):
        self.journal = journal

        # Ensure mutable defaults (field lists) are isolated per solver instance.
        self.options = deepcopy(self.NEDOptions)
        self._updateOptions(kwargs, journal)
        self.ids_1st = None
        self.ids_2nd = None

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
            if k in self.NEDOptions:
                journal.message("Updating option {:}={:}".format(k, v), self.identification)
                if isinstance(self.NEDOptions[k], list):
                    for item in v.split(","):
                        self.options[k].append(item.strip())
                else:
                    self.options[k] = type(self.NEDOptions[k])(updatedOptions[k])
            else:
                raise AttributeError("Invalid option {:} for {:}".format(k, self.identification))

    def solveStep(
        self,
        step,
        model: FEModel,
        fieldOutputController: FieldOutputController,
        outputmanagers: dict[str, OutputManagerBase],
    ):
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

        if "NEDSolver" in step.actions["options"].keys():
            self._updateOptions(step.actions["options"]["NEDSolver"].options, self.journal)

        # initialize mass and damping matrices
        M = self.theDofManager.constructDofVector()  # initialize lumped mass matrix
        Minv = self.theDofManager.constructDofVector()  # initialize inverse lumped mass matrix

        U = self.theDofManager.constructDofVector()  # initialize displacement vector
        dU = self.theDofManager.constructDofVector()  # initialize displacement vector
        V = self.theDofManager.constructDofVector()  # initilize velocity vector
        P = self.theDofManager.constructDofVector()  # initialize reaction vector

        U_old = self.theDofManager.constructDofVector()  # initialize old displacement vector
        V_old = self.theDofManager.constructDofVector()  # initilize old velocity vector
        P_old = self.theDofManager.constructDofVector()  # initialize old reaction vector

        M[:] = 0.0
        for el in model.elements.values():
            Me = np.zeros(el.nDof)
            el.computeLumpedInertia(Me)
            M[el] += Me

        # compute inverses
        if np.any(M == 0.0):
            raise ValueError(
                "Zero mass found in mass vector. This can be caused by elements with zero density, or by elements with zero volume."
            )
        Minv[M != 0.0] = 1.0 / M[M != 0.0]

        # delete M to save memory
        del M

        for fieldName, field in model.nodeFields.items():
            U = self.theDofManager.writeNodeFieldToDofVector(U, field, "U")
            P = self.theDofManager.writeNodeFieldToDofVector(P, field, "P")

        for variable in model.scalarVariables.values():
            U[self.theDofManager.idcsOfScalarVariablesInDofVector[variable]] = variable.value

        prevTimeStep = None
        self.ids_1st = np.empty(0, dtype=int)
        self.ids_2nd = np.empty(0, dtype=int)

        self.applyStepActionsAtStepStart(model, step.actions)

        criticalTimeStep = self.options.get("courant-number") * self.getCriticalTimeStepForExplicitDynamics(model, U)
        self.journal.message(
            "Critical time step for explicit dynamics: {:e}".format(criticalTimeStep), self.identification, 1
        )

        # check if all fields are specified either in first-order-fields or second-order-fields
        isSpecified = {presentVariable: False for presentVariable in presentVariableNames}
        for fieldName in self.options["first-order-fields"] + self.options["second-order-fields"]:
            if fieldName not in presentVariableNames:
                raise ValueError(
                    "Field {:} specified in first-order-fields, but not present in model".format(fieldName)
                )
            if isSpecified[fieldName]:
                raise ValueError(
                    "Field {:} specified multiple times in first-order-fields and second-order-fields: {:}, {:}".format(
                        fieldName, self.options["first-order-fields"], self.options["second-order-fields"]
                    )
                )
            isSpecified[fieldName] = True

        # assign indices of fields to first-order and second-order update schemes
        for fieldName in self.options["first-order-fields"]:
            self.ids_1st = np.r_[self.ids_1st, self.theDofManager.idcsOfFieldsInDofVector[fieldName]]
        for fieldName in self.options["second-order-fields"]:
            self.ids_2nd = np.r_[self.ids_2nd, self.theDofManager.idcsOfFieldsInDofVector[fieldName]]

        try:
            for timeStep in step.getTimeStep(enforcedTimeIncrement=criticalTimeStep):
                # only print for increments matching the configured output-frequency
                if timeStep.number % self.options["output-frequency"] == 0:
                    self.journal.printSeperationLine()
                    self.journal.message(
                        "increment {:}: {:8e}, {:8e}; time {:10e} to {:10e}".format(
                            timeStep.number,
                            timeStep.stepProgressIncrement,
                            timeStep.stepProgress,
                            timeStep.totalTime - timeStep.timeIncrement,
                            timeStep.totalTime,
                        ),
                        self.identification,
                        level=1,
                    )
                U_old[:] = U
                V_old[:] = V
                P_old[:] = P
                dU[:] = 0.0
                try:
                    U, V, P = self.solveIncrement(
                        U,
                        dU,
                        V,
                        P,
                        Minv,
                        step.actions,
                        model,
                        timeStep,
                        prevTimeStep,
                    )

                except CutbackRequest as e:
                    self.journal.message(str(e), self.identification, 1)
                    cutback = getattr(e, "cutbackSize", 0.25)
                    step.discardAndChangeIncrement(max(cutback, 0.25))
                    prevTimeStep = None

                    for man in outputmanagers:
                        man.finalizeFailedIncrement(
                            statusInfoDict=None,
                        )
                    # reset to old state
                    U[:] = U_old
                    V[:] = V_old
                    P[:] = P_old
                else:
                    prevTimeStep = timeStep

                    for fieldName, field in model.nodeFields.items():
                        self.theDofManager.writeDofVectorToNodeField(U, field, "U")
                        self.theDofManager.writeDofVectorToNodeField(P, field, "P")

                    for variable in model.scalarVariables.values():
                        variable.value = U[self.theDofManager.idcsOfScalarVariablesInDofVector[variable]]

                    model.advanceToTime(timeStep.totalTime)

                    if timeStep.number % self.options["output-frequency"] == 0:
                        fieldOutputController.finalizeIncrement()
                        for man in outputmanagers:
                            man.finalizeIncrement(
                                statusInfoDict=None,
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
            performancetiming.times.clear()
            performancetiming.extractIncrementTimes._last_snapshot = None

    @performancetiming.timeit("increment")
    def solveIncrement(
        self,
        U_n: DofVector,
        dU: DofVector,
        V: DofVector,
        P: DofVector,
        Minv: DofVector,
        stepActions: list,
        model: FEModel,
        timeStep: TimeStep,
        prevTimeStep: TimeStep,
    ) -> tuple[DofVector, DofVector, DofVector]:
        """Standard explicit update scheme to solve for an increment.

        Parameters
        ----------
        Un
            The old solution vector.
        V
            The old velocity vector.
        P
            The old reaction vector.
        M
            The lumped mass matrix to be used.
        elements
            The dictionary containing all elements.
        stepActions
            The list of active step actions.
        model
            The model tree.
        timeStep
            The time step.
        prevTimeStep
            The previous time step.

        Returns
        -------
        tuple[DofVector,DofVector,DofVector,DofVector]
            A tuple containing
                - the new solution vector
                - the solution increment
                - the new velocity vector
                - the new reaction vector
        """

        elements = model.elements
        dirichlets = stepActions["dirichlet"].values()
        nodeforces = stepActions["nodeforces"].values()
        distributedLoads = stepActions["distributedload"].values()
        bodyForces = stepActions["bodyforce"].values()

        if timeStep.timeIncrement == 0.0:
            return U_n, V, P

        if prevTimeStep is None:

            prevTimeStep = TimeStep(
                timeStep.number,
                timeStep.stepProgressIncrement,
                timeStep.stepProgress,
                0.0,
                timeStep.stepTime,
                timeStep.totalTime - timeStep.timeIncrement,
            )

        # enforce dirichlet boundary conditions
        for dirichlet in dirichlets:
            P[self.findDirichletIndices(dirichlet)] = 0.0
            V[self.findDirichletIndices(dirichlet)] = dirichlet.getDelta(timeStep).flatten() / timeStep.timeIncrement

        if self.ids_1st is not None:
            V[self.ids_1st] = Minv[self.ids_1st] * P[self.ids_1st]
        if self.ids_2nd is not None:
            V[self.ids_2nd] += (
                Minv[self.ids_2nd] * P[self.ids_2nd] * 0.5 * (timeStep.timeIncrement + prevTimeStep.timeIncrement)
            )
        # update displacement increment vector
        np.multiply(V, timeStep.timeIncrement, out=dU)
        np.add(U_n, dU, out=U_n)

        self.applyStepActionsAtIncrementStart(model, timeStep, stepActions)

        for geostatic in stepActions["geostatic"].values():
            geostatic.applyAtIterationStart()

        P[:] = 0.0
        P, psi = self.computeElements(elements, U_n, dU, P, timeStep)
        P[:] = -P[:]
        P = self.assembleLoads(nodeforces, distributedLoads, bodyForces, U_n, P, timeStep)

        if timeStep.number % self.options["output-frequency"] == 0:
            Wint = psi
            Wkin = 0.5 * np.sum(1 / Minv * V**2)
            W = Wint + Wkin
            self.journal.message(
                "Internal energy: {:e} ({:.2f} %)".format(Wint, Wint / W * 100), self.identification, 2
            )
            self.journal.message(
                "Kinetic energy:  {:e} ({:.2f} %)".format(Wkin, Wkin / W * 100), self.identification, 2
            )

        return U_n, V, P

    @performancetiming.timeit("distributed loads")
    def computeDistributedLoads(
        self,
        distributedLoads: list[StepActionBase],
        U_np: DofVector,
        PExt: DofVector,
        timeStep: TimeStep,
    ) -> DofVector:
        """Loop over all distributed loads acting on elements, and evaluate them.
        Assembles into the global external load vector.

        Parameters
        ----------
        distributedLoads
            The list of distributed loads.
        U_np
            The current solution vector.
        PExt
            The external load vector to be augmented.
        timeStep
            The current time step.

        Returns
        -------
        DofVector
            The augmented load vector.
        """

        time = timeStep.totalTime
        dT = timeStep.timeIncrement

        for dLoad in distributedLoads:
            load = dLoad.getCurrentLoad(timeStep)
            for faceID, elementSet in dLoad.surface.items():
                for el in elementSet:
                    Pe = np.zeros(el.nDof)
                    Ke = np.zeros((el.nDof, el.nDof)).ravel()
                    el.computeDistributedLoad(dLoad.loadType, Pe, Ke, faceID, load, U_np[el], time, dT)

                    PExt[el] += Pe

        return PExt

    @performancetiming.timeit("body forces")
    def computeBodyForces(
        self,
        bodyForces: list[StepActionBase],
        U_np: DofVector,
        PExt: DofVector,
        timeStep: TimeStep,
    ) -> DofVector:
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
        increment
            The increment.

        Returns
        -------
        tuple[DofVector,VIJSystemMatrix]
            The augmented load vector and system matrix.
        """

        time = timeStep.totalTime
        dT = timeStep.timeIncrement

        for bForce in bodyForces:
            force = bForce.getCurrentLoad(timeStep)
            for el in bForce.elementSet:
                Pe = np.zeros(el.nDof)
                Ke = np.zeros((el.nDof, el.nDof)).ravel()

                el.computeBodyForce(Pe, Ke, force, U_np[el], time, dT)

                PExt[el] += Pe

        return PExt

    @performancetiming.timeit("elements")
    def computeElements(
        self,
        elements: list,
        U_np: DofVector,
        dU: DofVector,
        P: DofVector,
        timeStep: TimeStep,
    ) -> tuple[DofVector]:
        """Loop over all elements, and evalute them.
        Is is called by solveStep() in each iteration.

        Parameters
        ----------
        elements
            The list of finite elements.
        U_n
            The current solution vector.
        dU
            The  solution increment vector.
        P
            The reaction vector.
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
        P[:] = 0.0
        psi = 0.0
        for el in elements.values():
            Pe = np.zeros(el.nDof)
            el.computeKernelsExplicit(Pe, U_np[el], dU[el], time, dT)
            psi += el.computeInternalEnergy()
            P[el] += Pe

        return P, psi

    def assembleLoads(
        self,
        nodeForces: list[StepActionBase],
        distributedLoads: list[StepActionBase],
        bodyForces: list[StepActionBase],
        U_np: DofVector,
        PExt: DofVector,
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
        PExt = self.computeDistributedLoads(distributedLoads, U_np, PExt, timeStep)
        PExt = self.computeBodyForces(bodyForces, U_np, PExt, timeStep)

        return PExt

    def getCriticalTimeStepForExplicitDynamics(self, model: FEModel, U: DofVector) -> float:
        """Compute the critical time step for explicit dynamics.

        Parameters
        ----------
        model
            The model tree.

        Returns
        -------
        float
            The critical time step for explicit dynamics.
        """
        minTimeStep = np.inf

        for element in model.elements.values():
            elementTimeStep = np.inf
            elementTimeStep = element.computeCriticalTimeStepForExplicitDynamics(U[element])
            if elementTimeStep < minTimeStep:
                minTimeStep = elementTimeStep

        return minTimeStep
