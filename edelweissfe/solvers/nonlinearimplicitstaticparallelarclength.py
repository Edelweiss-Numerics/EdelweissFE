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

"""
(Parallel) Arc Length Solver, based on the proposed approach in Jirásek/Bažant 2001.
Replaces the NewtonRaphson scheme of the NISTParallel Solver.
"""

from dataclasses import dataclass

import numpy as np

import edelweissfe.utils.performancetiming as performancetiming
from edelweissfe.models.femodel import FEModel
from edelweissfe.numerics.dofmanager import DofVector, VIJSystemMatrix
from edelweissfe.numerics.parallelizationutilities import forEachSliceInParallel
from edelweissfe.outputmanagers.base.outputmanagerbase import OutputManagerBase
from edelweissfe.solvers.nonlinearimplicitstaticparallel import NISTParallel
from edelweissfe.stepactions.base.arclengthcontrollerbase import ArcLengthControllerBase
from edelweissfe.stepactions.base.stepactionbase import StepActionBase
from edelweissfe.timesteppers.timestep import TimeStep
from edelweissfe.utils.exceptions import (
    ConditionalStop,
    DivergingSolution,
    ReachedMaxIterations,
)
from edelweissfe.utils.fieldoutput import FieldOutputController
from edelweissfe.utils.math import createModelAccessibleFunction
from edelweissfe.utils.schema import schemaField


@dataclass(frozen=True)
class NISTPArcLengthSchema(NISTParallel.schema):
    """:class:`~edelweissfe.solvers.nonlinearimplicitstatic.NISTSchema`'s options, plus the two of the arc length
    method; like every option of this solver, they are ``*solver`` datalines, ``>>options`` and constructor keyword
    arguments alike.
    """

    arcLengthController: str | None = schemaField(
        description=(
            "The name of the step action of the step that controls the load parameter (an arc length controller, "
            "e.g. an indirectcontrol), or off: the step is solved without arc length control."
        ),
        dtype=str,
        default="off",
    )
    stopCondition: str | None = schemaField(
        description="A model accessible expression; the step ends once it is true (under arc length control).",
        dtype=str,
        default=None,
    )


class NISTPArcLength(NISTParallel):
    """The arc length method (Jirásek/Bažant 2001): a step's reference load is scaled by a load parameter
    :math:`\\lambda`, which is an unknown of every increment besides the solution, and which a controller step action
    determines, e.g. such that a crack opening grows steadily. Paths with snap-back or snap-through, which neither a
    prescribed load nor a prescribed displacement can follow, are traced this way.

    The controller is the step action of the step that the option ``arcLengthController`` names (see
    :class:`~edelweissfe.stepactions.base.arclengthcontrollerbase.ArcLengthControllerBase`); without one
    (``arcLengthController=off``), the step is solved as by
    :class:`~edelweissfe.solvers.nonlinearimplicitstaticparallel.NISTParallel`.

    Parameters
    ----------
    jobInfo
        A dictionary containing the job information.
    journal
        The journal instance for logging.
    """

    identification = "NISTPArcLength"

    supportsMPC = False

    #: Option schema for this solver, per OptionSchemaProvider.
    schema = NISTPArcLengthSchema

    SolverSpecificOptions = NISTParallel.SolverSpecificOptions | {"arcLengthController": "off", "stopCondition": None}

    #: Not restartable: the arc-length parameter is not checkpointed, so a resumed step would restart
    #: the load factor from zero.
    checkpointedState = None

    def __init__(self, jobInfo, journal, **kwargs):
        self.Lambda = 0.0
        self.dLambda = 0.0
        #: The controller of the current step, or None.
        self.arcLengthController = None

        return super().__init__(jobInfo, journal, **kwargs)

    def beginStep(
        self,
        step,
        model: FEModel,
        fieldOutputController: FieldOutputController,
        outputmanagers: dict[str, OutputManagerBase],
    ):
        """Take the controller of the step, and start the step; see
        :meth:`~edelweissfe.solvers.nonlinearimplicitstatic.NIST.beginStep`.

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

        self.validateModelCapabilities(model)

        self.arcLengthController = self.arcLengthControllerOfStep(step)
        if self.arcLengthController is not None:
            self.dLambda = 0.0
        else:
            self.journal.message("No ArcLengthController specified in current step", self.identification)
            self.dLambda = None

        stopCondition = self.options["stopCondition"]
        self.checkConditionalStop = (
            createModelAccessibleFunction(stopCondition, model=model, fieldOutputs=fieldOutputController.fieldOutputs)
            if stopCondition
            else lambda: False
        )

        if "arc length parameter" in model.additionalParameters:
            self.Lambda = model.additionalParameters["arc length parameter"]

        if self.arcLengthController is not None and self.options["extrapolation"] == "quadratic":
            raise NotImplementedError(
                "extrapolation=quadratic is not available under an arc length controller, which extrapolates the load "
                "parameter linearly only"
            )

        if self.arcLengthController is not None and (
            self.options["lineSearch"] or self.options["fallBackToQuasiNewton"]
        ):
            raise NotImplementedError(
                "lineSearch and fallBackToQuasiNewton are not available under an arc length controller, whose Newton "
                "iterations solve for the load parameter, too"
            )

        return super().beginStep(step, model, fieldOutputController, outputmanagers)

    def arcLengthControllerOfStep(self, step) -> ArcLengthControllerBase | None:
        """The arc length controller the option ``arcLengthController`` names among the step actions of the step.

        Parameters
        ----------
        step
            The step to be solved.

        Returns
        -------
        ArcLengthControllerBase | None
            The controller, or None for ``arcLengthController=off``.

        Raises
        ------
        ValueError
            If the step has no arc length controller of this name, or more than one.
        """

        name = self.options["arcLengthController"]
        if name == "off":
            return None

        controllers = [
            action
            for actions in step.actions.values()
            for action in actions.values()
            if isinstance(action, ArcLengthControllerBase)
        ]
        named = [controller for controller in controllers if controller.name == name]
        if len(named) != 1:
            raise ValueError(
                "arcLengthController={:}: the step has {:} arc length controller(s) of this name; its arc length "
                "controllers are: {:}".format(
                    name, len(named), ", ".join(controller.name for controller in controllers) or "none"
                )
            )
        return named[0]

    def solveIncrement(
        self,
        U_n: DofVector,
        dU: DofVector,
        P: DofVector,
        K: VIJSystemMatrix,
        stepActions: list,
        model,
        timeStep: TimeStep,
        prevTimeStep: TimeStep,
        extrapolation: str,
        maxIter: int,
        maxGrowingIter: int,
    ) -> tuple[DofVector, DofVector, DofVector, int, dict]:
        """Arc length method scheme to solve for an increment.

        Parameters
        ----------
        U_n
            The old solution vector.
        dU
            The old solution increment.
        P
            The old reaction vector.
        K
            The system matrix to be used.
        elements
            The dictionary containing all elements.
        activeStepActions
            The list of active step actions.
        constraints
            The dictionary containing all elements.
        timeStep
            The current time step.
        prevTimeStep
            The previous time step.
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

        if self.arcLengthController is None:
            return super().solveIncrement(
                U_n,
                dU,
                P,
                K,
                stepActions,
                model,
                timeStep,
                prevTimeStep,
                extrapolation,
                maxIter,
                maxGrowingIter,
            )

        if timeStep.number > 1 and self.checkConditionalStop():
            raise ConditionalStop

        iterationCounter = 0
        incrementResidualHistory = {field: [] for field in self.theDofManager.idcsOfFieldsInDofVector}

        dirichlets = stepActions["dirichlet"].values()
        loads = self.collectLoads(stepActions)

        with performancetiming.timeit("step actions at increment start"):
            # Find which global DOFs the Dirichlet BCs constrain, once up front.
            self.locateConstrainedDofs(dirichlets)

            for stepActionType in stepActions.values():
                for action in stepActionType.values():
                    action.applyAtIncrementStart(model, timeStep)

        R_ = np.tile(self.theDofManager.constructDofVector(), (2, 1)).T  # 2 RHSs
        R_0 = R_[:, 0]
        R_f = R_[:, 1]
        F = self.theDofManager.constructDofVector()  # accumulated Flux vector
        FConstraints = self.theDofManager.constructDofVector()  # accumulated constraint Flux vector

        PConstraints = self.theDofManager.constructDofVector()
        P_0 = self.theDofManager.constructDofVector()
        P_f = self.theDofManager.constructDofVector()
        K_f = self.theDofManager.constructVIJSystemMatrix()
        K_0 = self.theDofManager.constructVIJSystemMatrix()
        U_np = self.theDofManager.constructDofVector()
        ddU = None

        Lambda = self.Lambda
        dLambda = self.dLambda
        ddLambda = 0.0

        with performancetiming.timeit("predictor"):
            dU, isExtrapolatedIncrement, dLambda = self.extrapolateLastIncrement(
                extrapolation, timeStep, dU, dirichlets, prevTimeStep, model, dLambda
            )

        # referenceIncrement = incNumber, 1.0, 1.0, 0.0, 0.0, 0.0
        # zeroIncrement = incNumber, 0.0, 0.0, 0.0, 0.0, 0.0
        referenceTimeStep = TimeStep(timeStep.number, 1.0, 1.0, 0.0, 0.0, 0.0)
        zeroTimeStep = TimeStep(timeStep.number, 0.0, 0.0, 0.0, 0.0, 0.0)

        while True:
            for geostatic in stepActions["geostatic"].values():
                geostatic.applyAtIterationStart()

            with performancetiming.timeit("reset residual and tangent"):
                U_np[:] = U_n
                U_np += dU

                P[:] = F[:] = FConstraints[:] = PConstraints[:] = P_0[:] = P_f[:] = 0.0

                # the tangents hold the entries of all element matrices, far too many to zero on a single thread
                def zeroTangents(s):
                    K[s] = K_f[s] = K_0[s] = 0.0

                forEachSliceInParallel(zeroTangents, len(K))

            # Every element, load and constraint adds its residual contribution and the tangent of it; the loads
            # twice: once at the start of the step (the dead load P_0) and once at its end (the reference load).
            P, K, F = self.computeElements(model.elements, U_np, dU, P, K, F, timeStep)
            PConstraints, K = self.assembleConstraints(
                model.constraints, U_np, dU, PConstraints, K, timeStep, FConstraints
            )

            P_0, K_0 = self.assembleLoads(loads, U_np, dU, P_0, K_0, zeroTimeStep)
            P_f, K_f = self.assembleLoads(loads, U_np, dU, P_f, K_f, referenceTimeStep)

            P_f -= P_0  # and subtract the dead part, since we are only interested in the homogeneous linear part
            K_f -= K_0

            # The right-hand sides are the negative of the summed contributions: for the dead load and the reference
            # load scaled by the load parameter, R_0 = f_ext,0 + (Lambda + dLambda) f_ref - f_int, and for the
            # reference load alone, R_f = f_ref.
            R_0[:] = -(P_0 + (Lambda + dLambda) * P_f + P + PConstraints)
            R_f[:] = -P_f

            # add stiffness contribution
            K[:] += K_0
            K[:] += (Lambda + dLambda) * K_f

            # Dirichlets ..
            if isExtrapolatedIncrement and iterationCounter == 0:
                R_0 = self.applyDirichletToResidual(zeroTimeStep, R_0, dirichlets)
            else:
                modifiedTimeStep = TimeStep(timeStep.number, dLambda, Lambda + dLambda, 0.0, 0.0, 0.0)
                R_0 = self.applyDirichletToResidual(modifiedTimeStep, R_0, dirichlets)

            R_f = self.applyDirichletToResidual(referenceTimeStep, R_f, dirichlets)

            if iterationCounter > 0 or isExtrapolatedIncrement:
                converged, nodesWithLargestResidual = self.checkConvergence(
                    R_0, ddU, dU, F, FConstraints, iterationCounter, incrementResidualHistory
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
            # identity rows, and the columns eliminated into both right-hand sides R_0 and R_f
            K_ = self.applyDirichletToStiffness(K_, dirichlets, R_)

            # solve 2 eq. systems at once: ddU_0 = K⁻¹ R_0 and ddU_f = K⁻¹ R_f
            ddU_ = self.linearSolve(K_, R_)
            ddU_0, ddU_f = ddU_[:, 0], ddU_[:, 1]

            # compute the increment of the load parameter. Method depends on the employed arc length controller
            ddLambda = self.arcLengthController.computeDDLambda(dU, ddU_0, ddU_f, timeStep, self.theDofManager)

            # assemble total solution
            ddU = ddU_0 + ddLambda * ddU_f

            dU += ddU
            dLambda += ddLambda

            iterationCounter += 1

        self.Lambda += dLambda
        self.dLambda = dLambda

        self.arcLengthController.finishIncrement(U_n, dU, dLambda, timeStep, self.theDofManager)
        model.additionalParameters["additionalParameters"] = self.Lambda

        for stepActionType in stepActions.values():
            for action in stepActionType.values():
                action.applyAtIncrementStart(model, timeStep)

        self.journal.message(
            "Current load parameter: lambda={:6.2e}, last increment: dLambda={:6.2e}".format(self.Lambda, self.dLambda),
            self.identification,
            level=1,
        )

        return U_np, dU, P, iterationCounter, incrementResidualHistory

    def extrapolateLastIncrement(
        self,
        extrapolation: str,
        timeStep: TimeStep,
        dU: DofVector,
        dirichlets: list,
        prevTimeStep: TimeStep,
        model: FEModel,
        dLambda: float = None,
    ) -> tuple[DofVector, bool]:
        """Depending on the current setting, extrapolate the solution of the last increment.
        Also accounts for the arc length loading parameter.

        Parameters
        ----------
        extrapolation
            The type of extrapolation.
        timeStep
            The current time increment.
        dU
            The last solution increment.
        dirichlets
            The list of active dirichlet boundary conditions.
        prevTimeStep
            The previous TimeStep

        Returns
        -------
        tuple[DofVector,bool]
            - The extrapolated solution increment.
            - True if an extrapolation was performed.
        """

        # incNumber, incrementSize, stepProgress, dT, stepTime, totalTime = increment

        if dLambda is None:
            # arclength control is inactive
            return super().extrapolateLastIncrement(extrapolation, timeStep, dU, dirichlets, prevTimeStep, model)

        elif extrapolation == "linear" and prevTimeStep and prevTimeStep.stepProgressIncrement:
            dLambda = dLambda * (timeStep.stepProgressIncrement / prevTimeStep.stepProgressIncrement)
            dU, isExtrapolatedIncrement = super().extrapolateLastIncrement(
                extrapolation, timeStep, dU, {}, prevTimeStep, model
            )
        else:
            dLambda = 0.0
            dU[:] = 0.0
            isExtrapolatedIncrement = False

        return dU, isExtrapolatedIncrement, dLambda

    def applyStepActionsAtStepEnd(self, model, stepActions: dict[str, StepActionBase]):
        """Called when all step actions should finish a step.

        For the arc length controlled simulation,
        external loads at the end of a step depend on the arc length parameter.
        Hence, for subsequent steps, the new reference loads depend on this parameter.
        We communicate this to the step actions before the parent solver tells them that the
        step is finished.

        Parameters
        ----------
        U
            The solution vector.
        P
            The reaction vector.
        stepActions
            The list of active step actions.
        """

        if self.arcLengthController is not None:
            for load in self.collectLoads(stepActions):
                load.applyAtStepEnd(model, stepMagnitude=self.Lambda)

        return super().applyStepActionsAtStepEnd(model, stepActions)
