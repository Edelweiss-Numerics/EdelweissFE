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
(Parallel) Arc length solver, based on the approach of Jirásek/Bažant 2001: the Newton iterations of the NISTParallel
solver, with the load parameter as an unknown of every increment besides the solution.
"""

from dataclasses import dataclass

import numpy as np
from scipy.sparse import csr_matrix

from edelweissfe.models.femodel import FEModel
from edelweissfe.numerics.dofmanager import DofVector, VIJSystemMatrix
from edelweissfe.numerics.parallelizationutilities import forEachSliceInParallel
from edelweissfe.outputmanagers.base.outputmanagerbase import OutputManagerBase
from edelweissfe.solvers.nonlinearimplicitstaticparallel import NISTParallel
from edelweissfe.stepactions.base.arclengthcontrollerbase import ArcLengthControllerBase
from edelweissfe.stepactions.base.stepactionbase import StepActionBase
from edelweissfe.timesteppers.timestep import TimeStep
from edelweissfe.utils.exceptions import ConditionalStop
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
        description=(
            "A model accessible expression; the step ends once it is true, checked after every accepted increment of "
            "nonzero size (under arc length control). It belongs to the controller it is declared with: an >>options "
            "block that names arcLengthController without a stopCondition returns to the stop condition of the *solver "
            "datalines (none by default), and one that sets a stopCondition without arc length control is refused."
        ),
        dtype=str,
        default=None,
    )


@dataclass
class ArcLengthIncrement:
    """What the arc length method adds to the Newton iterations of an increment: the load parameter :math:`\\lambda`,
    an unknown besides the solution, and the external load it scales, :math:`\\boldsymbol{f}_\\mathrm{ext} =
    \\boldsymbol{f}_0 + \\lambda \\, \\boldsymbol{f}_\\mathrm{ref}`, with the residual contributions of the dead load
    (the loads at the start of the step, :math:`-\\boldsymbol{f}_0`) and of the reference load (their change over the
    step, :math:`-\\boldsymbol{f}_\\mathrm{ref}`), and their tangents. Created for every attempt of an increment by
    :meth:`NISTPArcLength.solveIncrement`; the loads are assembled by :meth:`NISTPArcLength.assembleLoadsAtIncrementEnd`,
    which adds their tangents to the tangent of the increment.
    """

    #: The load parameter at the start of the increment, the one the last accepted increment reached.
    loadParameterAtStart: float
    #: The increment being solved.
    timeStep: TimeStep
    #: The whole step as one time step: at it, the Dirichlet boundary conditions give the change of their prescribed
    #: values over the step, which the load parameter scales like the reference load.
    referenceTimeStep: TimeStep
    #: The residual contribution of the dead load.
    deadLoad: DofVector
    #: The residual contribution of the reference load.
    referenceLoad: DofVector
    #: The tangent of the dead load's contribution (nonzero only for loads that follow the deformation).
    deadLoadTangent: VIJSystemMatrix
    #: The tangent of the reference load's contribution.
    referenceLoadTangent: VIJSystemMatrix
    #: The two right-hand sides of the linearized system, solved at once as its columns: the out-of-balance force
    #: :math:`\\boldsymbol{R}_0`, of the dead load and of the reference load scaled by the load parameter of the
    #: iterate, and the reference load alone, :math:`\\boldsymbol{R}_f`.
    rightHandSides: np.ndarray
    #: The correction of the solution :math:`\\boldsymbol{R}_f` causes, :math:`\\Delta\\Delta \\boldsymbol{U}_f`, of the
    #: last linear solve.
    referenceCorrection: np.ndarray | None = None
    #: The increment of the load parameter of the iterate, :math:`\\Delta\\lambda`.
    loadParameterIncrement: float = 0.0
    #: The load parameter of the state being evaluated: of the iterate, or of a trial state of a line search.
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

    The Newton iterations are those of :meth:`~edelweissfe.solvers.nonlinearimplicitstatic.NIST.solveIncrement`; the
    load parameter enters them through its hooks: the prediction (:meth:`extrapolateLastIncrement`), the loads of the
    evaluated state (:meth:`assembleLoadsAtIncrementEnd`), the first iterate (:meth:`firstIterateNeedsCorrection`), the
    linear solve for two right-hand sides (:meth:`solveForCorrection`), and the corrections of the pair of the solution
    and the load parameter (:meth:`setTrialState`, :meth:`applyCorrection`).

    Every arc length step scales its own reference load -- the change of its loads and prescribed displacements over
    the step -- from :math:`\\lambda = 0`, so it starts from the loads the steps before it ended with; at its end, every
    load keeps the value the step applied (see :meth:`applyStepActionsAtStepEnd`). The load parameter of the step being
    solved is published as ``model.additionalParameters["arc length parameter"]``, e.g. for a load-displacement curve.

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

    #: The state between increments: the load parameter the step has reached, and the predictor's state, NIST's plus
    #: the increments of the load parameter that belong to its accepted increments.
    checkpointedState = NISTParallel.checkpointedState | {"Lambda": float, "dLambda": float, "dLambdaBeforePrev": float}

    def __init__(self, jobInfo, journal, **kwargs):
        #: The load parameter of the current step, reached by its last converged increment.
        self.Lambda = 0.0
        #: The increment of the load parameter in the last accepted increment, from which the predictor extrapolates.
        self.dLambda = None
        #: The increment of the load parameter in the accepted increment before, for a quadratic extrapolation.
        self.dLambdaBeforePrev = None
        #: The increment of the load parameter of the converged attempt, until the increment is accepted.
        self._dLambda = None
        #: The controller of the current step, or None.
        self.arcLengthController = None
        #: What the arc length method adds to the iterations of the increment being solved; None between increments.
        self.arcLengthIncrement = None

        super().__init__(jobInfo, journal, **kwargs)

        #: The stop condition of the definition of the solver (its ``*solver`` datalines or constructor arguments), or
        #: None: the one of every step whose ``>>options`` name the controller without a stop condition of their own.
        self.stopConditionOfSolverDefinition = self.options["stopCondition"]

    def applyOptionsOverride(self, fieldValues: dict) -> None:
        """Apply an override of this solver's options, see
        :meth:`~edelweissfe.solvers.base.nonlinearsolverbase.NonlinearSolverBase.applyOptionsOverride`.

        A stop condition belongs to the arc length control it is declared with: an override that names the controller
        (``arcLengthController``, ``off`` too) without a ``stopCondition`` returns to the stop condition of the
        definition of the solver (:attr:`stopConditionOfSolverDefinition`), and drops the one of an earlier step.
        Otherwise a step would stop by the condition that ended the step before it -- which typically holds already --
        at its first check. An override that sets a ``stopCondition`` for a step without arc length control is refused
        before anything of it applies: it would never be checked. (The stop condition of the definition of the solver
        may stand without a controller: it applies to the steps that name one.)

        Parameters
        ----------
        fieldValues
            Maps schema field name to its new, already-coerced value.

        Raises
        ------
        ValueError
            If the override sets a stop condition for a step without arc length control.
        """

        stopCondition = fieldValues.get("stopCondition")
        if (
            stopCondition is not None
            and fieldValues.get("arcLengthController", self.options["arcLengthController"]) == "off"
        ):
            raise ValueError(
                "stopCondition={:}: a stop condition is checked under arc length control only, and the step has none "
                "(arcLengthController=off)".format(stopCondition)
            )

        if "arcLengthController" in fieldValues and "stopCondition" not in fieldValues:
            fieldValues = fieldValues | {"stopCondition": self.stopConditionOfSolverDefinition}

        super().applyOptionsOverride(fieldValues)

    def beginStep(
        self,
        step,
        model: FEModel,
        fieldOutputController: FieldOutputController,
        outputmanagers: dict[str, OutputManagerBase],
    ):
        """Take the controller of the step, and start the step at the load parameter zero; see
        :meth:`~edelweissfe.solvers.nonlinearimplicitstatic.NIST.beginStep`.

        The load parameter scales the change of the loads over this step, so it starts from zero: the external load is
        then the dead load, the loads the steps before ended with. A resumed step takes over the checkpointed load
        parameter instead.

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

        # A step starts from the load parameter zero and without a predictor state; a resumed step takes over the
        # checkpointed ones.
        self.Lambda = 0.0
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
        """Newton's method for the solution increment and the increment of the load parameter together: the iterations
        of :meth:`~edelweissfe.solvers.nonlinearimplicitstatic.NIST.solveIncrement`, into which the load parameter
        enters through the hooks of this class.

        Each iteration evaluates the model at the iterate (with the loads of :meth:`assembleLoadsAtIncrementEnd`), and
        solves the linearized system for two right-hand sides at once: the out-of-balance force :math:`R_0`, and the
        reference load :math:`R_f` (:meth:`solveForCorrection`). The controller combines the two corrections,
        :math:`\\Delta\\Delta U = \\Delta\\Delta U_0 + \\Delta\\Delta\\lambda \\, \\Delta\\Delta U_f`, and returns the
        correction of the load parameter :math:`\\Delta\\Delta\\lambda` (:meth:`applyCorrection`). The prescribed degrees
        of freedom follow the load parameter, like the loads: the reference solve carries their prescribed change over
        the step, so every correction of the load parameter moves them along, while the out-of-balance solve leaves
        them where they are -- they are no unknowns, and their rows are left out of the convergence check. The options
        of Newton's method apply to the pair :math:`(\\Delta U, \\Delta\\lambda)`: the prediction extrapolates both with
        the same weights (:meth:`extrapolateLastIncrement`), a line search scales the correction the out-of-balance
        force causes, which the controller completes at every step length, so that every trial state satisfies its
        equation (:meth:`setTrialState`), and the quasi-Newton fallback keeps the tangent of the first iteration for
        both right-hand sides.

        Before the iterations, the step's stop condition is checked, if an increment of nonzero size has been accepted
        before (see the option ``stopCondition``); once they have converged, the load parameter advances with the
        solution it belongs to.

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

        # The stop condition is checked after every accepted increment of nonzero size, i.e. before every increment once
        # the step has progressed: not after the zero increment, which leaves the step where it started.
        stepProgressAtIncrementStart = timeStep.stepProgress - timeStep.stepProgressIncrement
        if stepProgressAtIncrementStart > 0.0 and self.checkConditionalStop():
            raise ConditionalStop

        self.arcLengthIncrement = ArcLengthIncrement(
            loadParameterAtStart=self.Lambda,
            timeStep=timeStep,
            referenceTimeStep=TimeStep(timeStep.number, 1.0, 1.0, 0.0, 0.0, 0.0),
            deadLoad=self.theDofManager.constructDofVector(),
            referenceLoad=self.theDofManager.constructDofVector(),
            deadLoadTangent=self.theDofManager.constructVIJSystemMatrix(),
            referenceLoadTangent=self.theDofManager.constructVIJSystemMatrix(),
            rightHandSides=np.tile(self.theDofManager.constructDofVector(), (2, 1)).T,
        )
        try:
            U_np, dU, P, iterationCounter, incrementResidualHistory = super().solveIncrement(
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
            dLambda = self.arcLengthIncrement.loadParameterIncrement
        finally:
            self.arcLengthIncrement = None

        self.Lambda += dLambda
        self._dLambda = dLambda

        self.arcLengthController.finishIncrement(U_n, dU, dLambda, timeStep, self.theDofManager)
        # published for the outputs: the external load is the dead load plus the reference load times this
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

    def extrapolateLastIncrement(
        self,
        extrapolation: str,
        timeStep: TimeStep,
        dU: DofVector,
        dirichlets: list,
        prevTimeStep: TimeStep,
        model: FEModel,
    ) -> tuple[DofVector, bool]:
        """Predict the pair of the solution increment and the increment of the load parameter: the solution increment
        as :meth:`~edelweissfe.solvers.nonlinearimplicitstatic.NIST.extrapolateLastIncrement` predicts it, the increment
        of the load parameter with the same weights (:meth:`extrapolateLoadParameterIncrement`), and the prescribed
        degrees of freedom where the predicted load parameter puts them: their prescribed change over the step, scaled
        by it. Without a controller, as NIST.

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

        if self.arcLengthController is None:
            return super().extrapolateLastIncrement(extrapolation, timeStep, dU, dirichlets, prevTimeStep, model)

        increment = self.arcLengthIncrement

        dU, isExtrapolatedIncrement = super().extrapolateLastIncrement(
            extrapolation, timeStep, dU, (), prevTimeStep, model
        )
        dLambda = self.extrapolateLoadParameterIncrement(extrapolation, timeStep, prevTimeStep)
        for dirichlet in dirichlets:
            dU[dirichlet.constrainedDofIndices] = (
                dLambda * dirichlet.getPrescribedIncrement(increment.referenceTimeStep).flatten()
            )

        increment.loadParameterIncrement = dLambda
        increment.loadParameter = increment.loadParameterAtStart + dLambda

        return dU, isExtrapolatedIncrement

    def firstIterateNeedsCorrection(self, dirichlets: list) -> bool:
        """Under arc length control, the first iterate of an increment without a prediction always needs a correction:
        it is the converged state of the last increment, at the load parameter that increment reached, and the first
        correction moves the load parameter as the controller prescribes. Without a controller, as
        :meth:`~edelweissfe.solvers.nonlinearimplicitstatic.NIST.firstIterateNeedsCorrection`.

        Parameters
        ----------
        dirichlets
            The Dirichlet boundary conditions of the step.

        Returns
        -------
        bool
            True if the first iterate is corrected before the first convergence check.
        """

        if self.arcLengthController is None:
            return super().firstIterateNeedsCorrection(dirichlets)

        return True

    def solveForCorrection(self, K_: csr_matrix, R: DofVector, dirichlets: list) -> DofVector:
        """Solve the linearized system for two right-hand sides at once: the out-of-balance force :math:`R_0 = R`, and
        the reference load :math:`R_f`, for the corrections :math:`\\Delta\\Delta U_0` and :math:`\\Delta\\Delta U_f`.
        Without a controller, as :meth:`~edelweissfe.solvers.nonlinearimplicitstatic.NIST.solveForCorrection`.

        The prescribed degrees of freedom move with the load parameter, and only with it: the reference solve carries
        their prescribed change over the step, so a correction :math:`\\Delta\\Delta\\lambda` moves them by
        :math:`\\Delta\\Delta\\lambda` times that change, as it scales the reference load. The out-of-balance solve
        leaves them where they are, at the place the load parameter of the iterate puts them, in every iteration --
        also in the first iteration of an increment without a prediction, where NIST's would move them by their
        prescribed increment.

        The controller completes :math:`\\Delta\\Delta U_0` with :math:`\\Delta\\Delta U_f` when the iterate is moved
        (:meth:`setTrialState`, :meth:`applyCorrection`).

        Parameters
        ----------
        K_
            The system matrix of the iteration.
        R
            The out-of-balance force.
        dirichlets
            The Dirichlet boundary conditions of the step.

        Returns
        -------
        DofVector
            The correction the out-of-balance force causes, :math:`\\Delta\\Delta U_0`.
        """

        if self.arcLengthController is None:
            return super().solveForCorrection(K_, R, dirichlets)

        increment = self.arcLengthIncrement
        rightHandSides = increment.rightHandSides
        R_0, R_f = rightHandSides[:, 0], rightHandSides[:, 1]

        R_0[:] = R
        R_f[:] = -increment.referenceLoad
        for dirichlet in dirichlets:
            R_0[dirichlet.constrainedDofIndices] = 0.0
        R_f = self.applyDirichletToResidual(increment.referenceTimeStep, R_f, dirichlets)

        # identity rows, and the columns eliminated into both right-hand sides R_0 and R_f
        K_ = self.applyDirichletToStiffness(K_, dirichlets, rightHandSides)

        # solve 2 eq. systems at once: ddU_0 = K⁻¹ R_0 and ddU_f = K⁻¹ R_f
        ddU_ = self.linearSolve(K_, rightHandSides)
        increment.referenceCorrection = ddU_[:, 1]

        return ddU_[:, 0]

    def correctionAtStepLength(self, dU: DofVector, ddU_0: DofVector, stepLength: float) -> tuple[DofVector, float]:
        """The correction of the pair of the solution and the load parameter at a step length :math:`\\alpha`: the
        correction the out-of-balance force causes, scaled by :math:`\\alpha`, which the controller completes with the
        correction the reference load causes, :math:`\\Delta\\Delta U = \\alpha \\, \\Delta\\Delta U_0 +
        \\Delta\\Delta\\lambda \\, \\Delta\\Delta U_f`, by the correction of the load parameter
        :math:`\\Delta\\Delta\\lambda` its equation asks for. So the controller's equation holds at every step length,
        as for the full correction (:math:`\\alpha = 1`): a line search cannot leave it unsatisfied, also not from an
        iterate that does not satisfy it yet.

        Parameters
        ----------
        dU
            The solution increment of the iterate.
        ddU_0
            The correction the out-of-balance force causes, see :meth:`solveForCorrection`.
        stepLength
            The step length.

        Returns
        -------
        tuple[DofVector,float]
            - The correction of the solution.
            - The correction of the load parameter.
        """

        increment = self.arcLengthIncrement
        ddU_f = increment.referenceCorrection

        ddU_0 = stepLength * ddU_0
        ddLambda = self.arcLengthController.computeDDLambda(dU, ddU_0, ddU_f, increment.timeStep, self.theDofManager)

        return ddU_0 + ddLambda * ddU_f, ddLambda

    def setTrialState(self, dU: DofVector, ddU_0: DofVector, stepLength: float) -> DofVector:
        """Set the trial state of a line search at a step length along the correction of the pair: the solution
        increment and the load parameter, both moved by their corrections at that step length, which satisfy the
        controller's equation (see :meth:`correctionAtStepLength`); the iterate stays where it is. Without a
        controller, as :meth:`~edelweissfe.solvers.nonlinearimplicitstatic.NIST.setTrialState`.

        Parameters
        ----------
        dU
            The solution increment of the iterate.
        ddU_0
            The correction the out-of-balance force causes, see :meth:`solveForCorrection`.
        stepLength
            The step length.

        Returns
        -------
        DofVector
            The solution increment of the trial state.
        """

        if self.arcLengthController is None:
            return super().setTrialState(dU, ddU_0, stepLength)

        increment = self.arcLengthIncrement
        ddU, ddLambda = self.correctionAtStepLength(dU, ddU_0, stepLength)
        increment.loadParameter = increment.loadParameterAtStart + (increment.loadParameterIncrement + ddLambda)

        return dU + ddU

    def applyCorrection(self, dU: DofVector, ddU_0: DofVector, stepLength: float) -> DofVector:
        """Move the iterate, the pair of the solution increment and the increment of the load parameter, by their
        corrections at the step length a line search chose (1 without one), see :meth:`correctionAtStepLength`. Without
        a controller, as :meth:`~edelweissfe.solvers.nonlinearimplicitstatic.NIST.applyCorrection`.

        Parameters
        ----------
        dU
            The solution increment of the iterate, moved in place.
        ddU_0
            The correction the out-of-balance force causes, see :meth:`solveForCorrection`.
        stepLength
            The step length.

        Returns
        -------
        DofVector
            The correction the solution increment was moved by.
        """

        if self.arcLengthController is None:
            return super().applyCorrection(dU, ddU_0, stepLength)

        increment = self.arcLengthIncrement
        ddU, ddLambda = self.correctionAtStepLength(dU, ddU_0, stepLength)
        dU += ddU
        increment.loadParameterIncrement += ddLambda
        increment.loadParameter = increment.loadParameterAtStart + increment.loadParameterIncrement

        return ddU

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
        load) and at its end, and their difference is the reference load (see :class:`ArcLengthIncrement`). Without a
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

        increment = self.arcLengthIncrement
        loads = self.collectLoads(stepActions)

        increment.deadLoad[:] = increment.referenceLoad[:] = 0.0

        # the tangents hold the entries of all element matrices, far too many to zero on a single thread
        def zeroTangents(s):
            increment.referenceLoadTangent[s] = increment.deadLoadTangent[s] = 0.0

        forEachSliceInParallel(zeroTangents, len(increment.deadLoadTangent))

        increment.deadLoad, increment.deadLoadTangent = self.assembleLoads(
            loads,
            U_np,
            dU,
            increment.deadLoad,
            increment.deadLoadTangent,
            TimeStep(timeStep.number, 0.0, 0.0, 0.0, 0.0, 0.0),
        )
        increment.referenceLoad, increment.referenceLoadTangent = self.assembleLoads(
            loads,
            U_np,
            dU,
            increment.referenceLoad,
            increment.referenceLoadTangent,
            TimeStep(timeStep.number, 1.0, 1.0, 0.0, 0.0, 0.0),
        )
        # the change of the loads over the step: the homogeneous, linear part
        increment.referenceLoad -= increment.deadLoad
        increment.referenceLoadTangent -= increment.deadLoadTangent

        P += increment.deadLoad
        P += increment.loadParameter * increment.referenceLoad

        K[:] += increment.deadLoadTangent
        K[:] += increment.loadParameter * increment.referenceLoadTangent

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
        """Finish the step actions at the end of a step. Under arc length control, every load keeps the value the step
        applied at the load parameter it ended at, :math:`f(0) + \\lambda \\, (f(1) - f(0))` with its own amplitude
        (see :meth:`~edelweissfe.stepactions.base.loadbase.LoadBase.applyAtStepEnd`), before the other step actions
        finish as in any step: the next step starts from these loads.

        Parameters
        ----------
        model
            The model tree.
        stepActions
            The list of active step actions.
        """

        if self.arcLengthController is not None:
            for load in self.collectLoads(stepActions):
                load.applyAtStepEnd(model, loadParameter=self.Lambda)

        return super().applyStepActionsAtStepEnd(model, stepActions)
