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
from edelweissfe.solvers.nonlinearimplicitstatic import QuasiNewtonFallback
from edelweissfe.solvers.nonlinearimplicitstaticparallel import NISTParallel
from edelweissfe.stepactions.base.arclengthcontrollerbase import ArcLengthControllerBase
from edelweissfe.stepactions.base.stepactionbase import StepActionBase
from edelweissfe.timesteppers.timestep import TimeStep
from edelweissfe.utils.exceptions import ConditionalStop, ReachedMaxIterations
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


@dataclass
class ArcLengthLoads:
    """The external load of an arc length increment, :math:`\\boldsymbol{f}_\\mathrm{ext} = \\boldsymbol{f}_0 +
    \\lambda \\, \\boldsymbol{f}_\\mathrm{ref}`: the residual contributions of the dead load (the loads at the start of
    the step, :math:`-\\boldsymbol{f}_0`) and of the reference load (their change over the step,
    :math:`-\\boldsymbol{f}_\\mathrm{ref}`), with their tangents, and the load parameter :math:`\\lambda` of the state
    being evaluated. Allocated per increment by :meth:`NISTPArcLength.solveIncrement`, assembled by
    :meth:`NISTPArcLength.assembleLoadsAtIncrementEnd`.
    """

    #: The residual contribution of the dead load.
    deadLoad: DofVector
    #: The residual contribution of the reference load.
    referenceLoad: DofVector
    #: The tangent of the dead load's contribution (nonzero only for loads that follow the deformation).
    deadLoadTangent: VIJSystemMatrix
    #: The tangent of the reference load's contribution.
    referenceLoadTangent: VIJSystemMatrix
    #: The load parameter at the end of the increment, of the state being evaluated.
    loadParameter: float = 0.0


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
        #: The load parameter reached by the last converged increment.
        self.Lambda = 0.0
        #: The increment of the load parameter in the last accepted increment, from which the predictor extrapolates.
        self.dLambda = None
        #: The increment of the load parameter in the accepted increment before, for a quadratic extrapolation.
        self.dLambdaBeforePrev = None
        #: The increment of the load parameter of the converged attempt, until the increment is accepted.
        self._dLambda = None
        #: The controller of the current step, or None.
        self.arcLengthController = None
        #: The loads of the increment being solved, None between increments.
        self.arcLengthLoads = None

        return super().__init__(jobInfo, journal, **kwargs)

    def beginStep(
        self,
        step,
        model: FEModel,
        fieldOutputController: FieldOutputController,
        outputmanagers: dict[str, OutputManagerBase],
    ):
        """Take the controller of the step, and the load parameter reached so far, and start the step; see
        :meth:`~edelweissfe.solvers.nonlinearimplicitstatic.NIST.beginStep`.

        The load parameter is part of the state of the model: every converged increment writes it to
        ``model.additionalParameters["arc length parameter"]``. So a step continues from the load parameter the steps
        before it reached.

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
        if self.arcLengthController is None:
            self.journal.message("No ArcLengthController specified in current step", self.identification)

        stopCondition = self.options["stopCondition"]
        self.checkConditionalStop = (
            createModelAccessibleFunction(stopCondition, model=model, fieldOutputs=fieldOutputController.fieldOutputs)
            if stopCondition
            else lambda: False
        )

        if "arc length parameter" in model.additionalParameters:
            self.Lambda = model.additionalParameters["arc length parameter"]

        # A step starts without a predictor state.
        self.dLambda = None
        self.dLambdaBeforePrev = None

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
        stepActions: dict,
        model: FEModel,
        timeStep: TimeStep,
        prevTimeStep: TimeStep,
        extrapolation: str,
        maxIter: int,
        maxGrowingIter: int,
    ) -> tuple[DofVector, DofVector, DofVector, int, dict]:
        """Newton's method for the solution increment and the increment of the load parameter together.

        Each iteration evaluates the model at the trial state (:meth:`assembleResidualAndTangent`, with the loads of
        :meth:`assembleLoadsAtIncrementEnd`), and solves the linearized system for two right-hand sides at once: the
        out-of-balance force :math:`R_0`, and the reference load :math:`R_f`. The controller combines the two
        corrections, :math:`\\Delta\\Delta U = \\Delta\\Delta U_0 + \\Delta\\Delta\\lambda \\, \\Delta\\Delta U_f`, and
        returns the correction of the load parameter :math:`\\Delta\\Delta\\lambda`. The prescribed degrees of freedom
        follow the load parameter, like the loads. The options of Newton's method apply to the pair
        :math:`(\\Delta U, \\Delta\\lambda)`: the prediction extrapolates both with the same weights, a line search
        scales both corrections with the same step length, and the quasi-Newton fallback keeps the tangent of the
        first iteration for both right-hand sides.

        Parameters
        ----------
        U_n
            The solution at the start of the increment.
        dU
            The solution increment, overwritten.
        P
            The internal forces (the reaction), overwritten.
        K
            The tangent in VIJ layout, overwritten.
        stepActions
            The active step actions.
        model
            The model tree.
        timeStep
            The time step.
        prevTimeStep
            The last accepted increment, from which the prediction extrapolates, or None.
        extrapolation
            The type of extrapolation.
        maxIter
            The maximum number of iterations.
        maxGrowingIter
            The maximum number of residual growths before the increment is cut back.

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

        with performancetiming.timeit("step actions at increment start"):
            # Find which global DOFs the Dirichlet BCs constrain, once up front.
            self.locateConstrainedDofs(dirichlets)

            self.applyStepActionsAtIncrementStart(model, timeStep, stepActions)

        # the two right-hand sides, solved at once: R_0 of the dead and the scaled reference load, R_f of the reference
        # load alone
        R_ = np.tile(self.theDofManager.constructDofVector(), (2, 1)).T
        R_0 = R_[:, 0]
        R_f = R_[:, 1]
        F = self.theDofManager.constructDofVector()
        FConstraints = self.theDofManager.constructDofVector()
        PLoadsAndConstraints = self.theDofManager.constructDofVector()
        U_np = self.theDofManager.constructDofVector()
        self.arcLengthLoads = ArcLengthLoads(
            self.theDofManager.constructDofVector(),
            self.theDofManager.constructDofVector(),
            self.theDofManager.constructVIJSystemMatrix(),
            self.theDofManager.constructVIJSystemMatrix(),
        )
        ddU = None

        Lambda = self.Lambda

        with performancetiming.timeit("predictor"):
            # The prescribed degrees of freedom follow the load parameter, so they are extrapolated with all others.
            dU, isExtrapolatedIncrement = self.extrapolateLastIncrement(
                extrapolation, timeStep, dU, (), prevTimeStep, model
            )
            dLambda = self.extrapolateLoadParameterIncrement(extrapolation, timeStep, prevTimeStep)

        referenceTimeStep = TimeStep(timeStep.number, 1.0, 1.0, 0.0, 0.0, 0.0)
        zeroTimeStep = TimeStep(timeStep.number, 0.0, 0.0, 0.0, 0.0, 0.0)

        quasiNewton = QuasiNewtonFallback(self.options["fallBackToQuasiNewton"])

        while True:
            self.arcLengthLoads.loadParameter = Lambda + dLambda
            P, K, F = self.assembleResidualAndTangent(
                U_n, dU, U_np, R_0, P, PLoadsAndConstraints, F, FConstraints, K, stepActions, model, timeStep
            )
            R_f[:] = -self.arcLengthLoads.referenceLoad

            # The prescribed degrees of freedom move with the load parameter: the reference solve carries their full
            # prescribed increment, and R_0 their increment up to the current load parameter.
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

                self.checkDivergingSolutionOrFallBack(
                    quasiNewton, incrementResidualHistory, maxGrowingIter, nodesWithLargestResidual
                )

                if iterationCounter == maxIter:
                    self.printResidualOutlierNodes(nodesWithLargestResidual)
                    raise ReachedMaxIterations("Reached max. iterations in current increment, cutting back")

            K_ = self.assembleSystemMatrix(K, quasiNewton)
            # identity rows, and the columns eliminated into both right-hand sides R_0 and R_f
            K_ = self.applyDirichletToStiffness(K_, dirichlets, R_)

            # solve 2 eq. systems at once: ddU_0 = K⁻¹ R_0 and ddU_f = K⁻¹ R_f
            ddU_ = self.linearSolve(K_, R_)
            ddU_0, ddU_f = ddU_[:, 0], ddU_[:, 1]

            # compute the increment of the load parameter. Method depends on the employed arc length controller
            ddLambda = self.arcLengthController.computeDDLambda(dU, ddU_0, ddU_f, timeStep, self.theDofManager)

            # assemble total solution
            ddU = ddU_0 + ddLambda * ddU_f

            if self.lineSearchIsDue(iterationCounter):

                def largestResidualAt(stepLength):
                    # the trial state along the correction of the pair: the solution and the load parameter
                    self.arcLengthLoads.loadParameter = Lambda + (dLambda + stepLength * ddLambda)
                    self.assembleResidualAndTangent(
                        U_n,
                        dU + stepLength * ddU,
                        U_np,
                        R_0,
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
                        R_0[dirichlet.constrainedDofIndices] = 0.0
                    return np.linalg.norm(R_0, np.inf)

                stepLength = self.quadraticLineSearch(largestResidualAt)
                ddU = stepLength * ddU
                ddLambda = stepLength * ddLambda

            dU += ddU
            dLambda += ddLambda

            iterationCounter += 1

        self.arcLengthLoads = None
        self.Lambda += dLambda
        self._dLambda = dLambda

        self.arcLengthController.finishIncrement(U_n, dU, dLambda, timeStep, self.theDofManager)
        model.additionalParameters["arc length parameter"] = self.Lambda

        for stepActionType in stepActions.values():
            for action in stepActionType.values():
                action.applyAtIncrementStart(model, timeStep)

        self.journal.message(
            "Current load parameter: lambda={:6.2e}, last increment: dLambda={:6.2e}".format(self.Lambda, dLambda),
            self.identification,
            level=1,
        )

        return U_np, dU, P, iterationCounter, incrementResidualHistory

    def assembleLoadsAtIncrementEnd(
        self,
        stepActions: dict,
        U_np: DofVector,
        dU: DofVector,
        P: DofVector,
        K: VIJSystemMatrix,
        timeStep: TimeStep,
    ) -> tuple[DofVector, VIJSystemMatrix]:
        """Assemble the loads at the end of the increment: under arc length control, the dead load plus the reference
        load scaled by the load parameter of the evaluated state, :math:`-(\\boldsymbol{f}_0 + \\lambda \\,
        \\boldsymbol{f}_\\mathrm{ref})`, with their tangents. The loads are evaluated at the start of the step (the dead
        load) and at its end, and their difference is the reference load (see :class:`ArcLengthLoads`). Without a
        controller, the loads of the step at the time of the increment, as in
        :meth:`~edelweissfe.solvers.nonlinearimplicitstatic.NIST.assembleLoadsAtIncrementEnd`.

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

        if self.arcLengthController is None:
            return super().assembleLoadsAtIncrementEnd(stepActions, U_np, dU, P, K, timeStep)

        arcLengthLoads = self.arcLengthLoads
        loads = self.collectLoads(stepActions)

        arcLengthLoads.deadLoad[:] = arcLengthLoads.referenceLoad[:] = 0.0

        # the tangents hold the entries of all element matrices, far too many to zero on a single thread
        def zeroTangents(s):
            arcLengthLoads.referenceLoadTangent[s] = arcLengthLoads.deadLoadTangent[s] = 0.0

        forEachSliceInParallel(zeroTangents, len(arcLengthLoads.deadLoadTangent))

        arcLengthLoads.deadLoad, arcLengthLoads.deadLoadTangent = self.assembleLoads(
            loads,
            U_np,
            dU,
            arcLengthLoads.deadLoad,
            arcLengthLoads.deadLoadTangent,
            TimeStep(timeStep.number, 0.0, 0.0, 0.0, 0.0, 0.0),
        )
        arcLengthLoads.referenceLoad, arcLengthLoads.referenceLoadTangent = self.assembleLoads(
            loads,
            U_np,
            dU,
            arcLengthLoads.referenceLoad,
            arcLengthLoads.referenceLoadTangent,
            TimeStep(timeStep.number, 1.0, 1.0, 0.0, 0.0, 0.0),
        )
        # the change of the loads over the step: the homogeneous, linear part
        arcLengthLoads.referenceLoad -= arcLengthLoads.deadLoad
        arcLengthLoads.referenceLoadTangent -= arcLengthLoads.deadLoadTangent

        P += arcLengthLoads.deadLoad
        P += arcLengthLoads.loadParameter * arcLengthLoads.referenceLoad

        K[:] += arcLengthLoads.deadLoadTangent
        K[:] += arcLengthLoads.loadParameter * arcLengthLoads.referenceLoadTangent

        return P, K

    def extrapolateLoadParameterIncrement(
        self, extrapolation: str, timeStep: TimeStep, prevTimeStep: TimeStep | None
    ) -> float:
        """Predict the increment of the load parameter, as the solution increment is predicted, see
        :meth:`~edelweissfe.solvers.nonlinearimplicitstatic.NIST.extrapolateLastIncrement`: from the same accepted
        increments, with the same weights (:meth:`~edelweissfe.solvers.nonlinearimplicitstatic.NIST.extrapolationWeights`),
        so that the prediction of the pair stays on one extrapolated path.

        Parameters
        ----------
        extrapolation
            The type of extrapolation.
        timeStep
            The new increment.
        prevTimeStep
            The last accepted increment, or None for no prediction.

        Returns
        -------
        float
            The predicted increment of the load parameter.
        """

        if not self.extrapolates(extrapolation, prevTimeStep):
            return 0.0

        weightOfLast, weightOfBeforeLast = self.extrapolationWeights(extrapolation, timeStep, prevTimeStep)
        dLambda = self.dLambda * weightOfLast
        if weightOfBeforeLast is not None:
            dLambda += weightOfBeforeLast * self.dLambdaBeforePrev

        return dLambda

    def acceptIncrement(self, step, model: FEModel, timeStep: TimeStep):
        """Accept the increment as :meth:`~edelweissfe.solvers.nonlinearimplicitstatic.NIST.acceptIncrement` does,
        and keep the increment of the load parameter for the predictor alongside the solution increment.

        Parameters
        ----------
        step
            The step being solved.
        model
            The model tree.
        timeStep
            The converged increment.
        """

        if self.arcLengthController is None:
            return super().acceptIncrement(step, model, timeStep)

        dLambdaOfLastIncrement = self.dLambda
        super().acceptIncrement(step, model, timeStep)
        # the increment before the last one is kept just as long as the solution's is
        self.dLambdaBeforePrev = dLambdaOfLastIncrement if self.timeStepBeforePrev is not None else None
        self.dLambda, self._dLambda = self._dLambda, None

    def applyStepActionsAtStepEnd(self, model, stepActions: dict[str, StepActionBase]):
        """Called when all step actions should finish a step.

        For the arc length controlled simulation,
        external loads at the end of a step depend on the arc length parameter.
        Hence, for subsequent steps, the new reference loads depend on this parameter.
        We communicate this to the step actions before the parent solver tells them that the
        step is finished.

        Parameters
        ----------
        model
            The model tree.
        stepActions
            The list of active step actions.
        """

        if self.arcLengthController is not None:
            for load in self.collectLoads(stepActions):
                load.applyAtStepEnd(model, stepMagnitude=self.Lambda)

        return super().applyStepActionsAtStepEnd(model, stepActions)
