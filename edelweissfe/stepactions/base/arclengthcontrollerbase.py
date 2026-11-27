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
"""The step actions that control the load parameter of the arc length method,
:class:`~edelweissfe.solvers.nonlinearimplicitstaticparallelarclength.NISTPArcLength`."""

from abc import abstractmethod
from dataclasses import dataclass

import numpy as np

from edelweissfe.models.discretesystem import DiscreteSystem
from edelweissfe.models.femodel import FEModel
from edelweissfe.numerics.dofvector import DofVector
from edelweissfe.stepactions.base.stepactionbase import StepActionBase
from edelweissfe.timesteppers.timestep import TimeStep


@dataclass(frozen=True)
class LinearizedConstraint:
    """The controller's equation, linearized at the iterate in the corrections of the solution and of the load
    parameter,

    .. math::

        \\boldsymbol{c} \\cdot \\Delta\\Delta \\boldsymbol{U} + d \\, \\Delta\\Delta\\lambda = r,

    the last row of the bordered system that
    :class:`~edelweissfe.solvers.nonlinearimplicitstaticparallelarclength.NISTPArcLength` solves with
    ``borderedSystem`` (see
    :meth:`~edelweissfe.stepactions.base.arclengthcontrollerbase.ArcLengthControllerBase.linearizedConstraint`).

    Attributes
    ----------
    weights
        :math:`\\boldsymbol{c}`, over the whole dof vector of the discrete system.
    loadParameterWeight
        :math:`d`.
    rightHandSide
        :math:`r`, what the equation lacks at the iterate.
    """

    weights: np.ndarray
    loadParameterWeight: float
    rightHandSide: float

    @classmethod
    def ofLoadParameter(cls, ddLambda: float, nDof: int) -> "LinearizedConstraint":
        """The equation :math:`\\Delta\\Delta\\lambda = \\Delta\\Delta\\lambda_0`, which fixes the correction of
        the load parameter, e.g. under load control.

        Parameters
        ----------
        ddLambda
            The correction of the load parameter, :math:`\\Delta\\Delta\\lambda_0`.
        nDof
            The size of the dof vector.
        """
        return cls(np.zeros(nDof), 1.0, ddLambda)


class ArcLengthControllerBase(StepActionBase):
    """The base of the step actions that control the load parameter of the arc length method,
    :class:`~edelweissfe.solvers.nonlinearimplicitstaticparallelarclength.NISTPArcLength`. The solver takes the
    controller of a step by its ``name``, from its option ``arcLengthController``.

    The arc length method scales the reference load of a step by a load parameter :math:`\\lambda`, which becomes an
    unknown of the increment besides the solution. The controller adds the equation that determines it, typically that a
    chosen combination of displacements -- a crack opening, say -- grows by a prescribed amount over the step. In every
    Newton iteration, the solver solves for the correction of the solution caused by the out-of-balance force,
    :math:`\\Delta\\Delta U_0`, and for the one caused by the reference load alone, :math:`\\Delta\\Delta U_f`. The
    controller returns the correction of the load parameter :math:`\\Delta\\Delta\\lambda` such that the corrected
    increment :math:`\\Delta U + \\Delta\\Delta U_0 + \\Delta\\Delta\\lambda \\, \\Delta\\Delta U_f` satisfies its
    equation. A line search asks it the same for every trial state, with :math:`\\Delta\\Delta U_0` scaled by the trial
    step length, so that no trial state leaves the equation unsatisfied; the controller is asked several times per
    iteration, and the answers must not change it.

    Every hook that needs the degrees of freedom receives the current discrete system: its numbering, and the
    locations of its elements and constraints, both renewed when the model changes.

    A controller whose equation is nonlinear in the solution, e.g. one that controls the dissipated energy, may
    override the optional hooks: :meth:`beginControl` and :meth:`endControl` (the step it controls),
    :meth:`setSystemState` (the state of the iterate, with :attr:`receivesSystemState`), :meth:`isConstraintSatisfied`
    (its equation in the convergence check, as it is only linearized in :meth:`computeDDLambda`) and
    :meth:`getExtrapolationFactor` (the predictor), with :attr:`extrapolatesAfterCutback`.

    With the solver option ``borderedSystem``, the solver does not split the correction into the two parts: it asks
    the controller for its equation linearized at the iterate as one row (:meth:`linearizedConstraint`), and solves
    for the corrections of the solution and of the load parameter together, as one system bordered by that row. A
    controller that can give its equation as such a row says so (:attr:`providesLinearizedConstraint`).
    """

    #: The name of this step action, by which the solver option ``arcLengthController`` selects it.
    name: str

    #: Whether the increment that follows a cutback is predicted from the last accepted increment as well, for a
    #: controller whose equation needs the direction of the path (a zero prediction may follow an unloading branch).
    extrapolatesAfterCutback = False

    #: Whether the controller receives the state of every iterate (:meth:`setSystemState`); the solver builds it only
    #: for a controller that does.
    receivesSystemState = False

    @abstractmethod
    def computeDDLambda(
        self, dU: DofVector, ddU_0: DofVector, ddU_f: DofVector, timeStep: TimeStep, system: DiscreteSystem
    ) -> float:
        """Compute the correction of the load parameter in the current Newton iteration.

        Parameters
        ----------
        dU
            The current increment of the solution.
        ddU_0
            The correction of the solution caused by the out-of-balance force, of the dead load and the reference load
            scaled by the current load parameter; for a trial state of a line search, scaled by its step length.
        ddU_f
            The correction of the solution caused by the reference load alone.
        timeStep
            The increment being solved: ``timeStep.stepProgress`` is the progress of the step at its end, and
            ``timeStep.stepProgressIncrement`` its size, so it starts at their difference.
        system
            The current discrete system.

        Returns
        -------
        float
            The correction of the load parameter.
        """

    @property
    def providesLinearizedConstraint(self) -> bool:
        """Whether the controller gives its equation as one linear row (:meth:`linearizedConstraint`), for the bordered
        system of the solver option ``borderedSystem``. By default, it does not."""
        return False

    def linearizedConstraint(self, dU: DofVector, timeStep: TimeStep, system: DiscreteSystem) -> LinearizedConstraint:
        """The controller's equation linearized at the iterate, the row that borders the linearized system (see
        :class:`LinearizedConstraint`): the corrections that satisfy it are those that :meth:`computeDDLambda` would
        give, :math:`\\boldsymbol{c} \\cdot (\\Delta\\Delta \\boldsymbol{U}_0 + \\Delta\\Delta\\lambda \\,
        \\Delta\\Delta \\boldsymbol{U}_f) + d \\, \\Delta\\Delta\\lambda = r`. Asked once per iteration, after
        :meth:`setSystemState`.

        Parameters
        ----------
        dU
            The current increment of the solution.
        timeStep
            The increment being solved.
        system
            The current discrete system.

        Returns
        -------
        LinearizedConstraint
            The row.
        """
        raise NotImplementedError(
            "{:} {:} gives no linearized equation for the bordered system".format(type(self).__name__, self.name)
        )

    @abstractmethod
    def finishIncrement(self, U: DofVector, dU: DofVector, dLambda: float, timeStep: TimeStep, system: DiscreteSystem):
        """Called once an increment has converged, e.g. to report or to store the controlled quantity it reached.

        Parameters
        ----------
        U
            The solution at the start of the increment.
        dU
            The converged increment of the solution.
        dLambda
            The converged increment of the load parameter.
        timeStep
            The converged time step.
        system
            The current discrete system.
        """

    def beginControl(self, model: FEModel):
        """Called when the solver starts (or resumes) a step this controller controls, before its first increment. By
        default, nothing is done.

        Parameters
        ----------
        model
            The model tree.
        """

    def endControl(self, model: FEModel):
        """Called when the solver ends a step this controller controls, whether it finished or failed. By default,
        nothing is done.

        Parameters
        ----------
        model
            The model tree.
        """

    def setSystemState(
        self, reaction: DofVector, referenceIncrement: DofVector, loadParameter: float, loadParameterIncrement: float
    ):
        """Receive the state of the iterate: called in every Newton iteration before :meth:`computeDDLambda`, and once
        an increment has converged, before :meth:`finishIncrement`. By default, nothing is kept.

        Parameters
        ----------
        reaction
            The internal forces of the elements and the constraints, i.e. the reactions at the prescribed degrees of
            freedom.
        referenceIncrement
            The change of the prescribed degrees of freedom over the step (zero elsewhere), conjugate to their
            reactions.
        loadParameter
            The load parameter at the start of the increment.
        loadParameterIncrement
            The increment of the load parameter of the iterate.
        """

    def isConstraintSatisfied(self, timeStep: TimeStep, dU: DofVector, system: DiscreteSystem) -> bool:
        """Whether the iterate satisfies the controller's equation: checked once the residual has converged. A linear
        equation holds after every correction, which is the default.

        Parameters
        ----------
        timeStep
            The increment being solved.
        dU
            The current increment of the solution.
        system
            The current discrete system.

        Returns
        -------
        bool
            True if the equation is satisfied.
        """

        return True

    def getExtrapolationFactor(self, timeStep: TimeStep, linearWeight: float) -> float:
        """The factor by which the last accepted increment, of the solution and of the load parameter, is scaled for
        the linear prediction of a new increment. By default the weight of the linear prediction without arc length
        control, the ratio of the sizes of the new and the last accepted increment.

        Parameters
        ----------
        timeStep
            The new increment.
        linearWeight
            The weight of the last accepted increment in the linear prediction of the solver, see
            :meth:`~edelweissfe.solvers.nonlinearimplicitstatic.NIST.extrapolationWeights`.

        Returns
        -------
        float
            The scale factor.
        """

        return linearWeight
