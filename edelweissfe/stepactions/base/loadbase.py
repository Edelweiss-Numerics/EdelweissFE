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

from abc import abstractmethod
from collections.abc import Callable

from edelweissfe.numerics.dofmanager import DofManager, DofVector, VIJSystemMatrix
from edelweissfe.stepactions.base.stepactionbase import StepActionBase
from edelweissfe.timesteppers.timestep import TimeStep


class LoadBase(StepActionBase):
    """A load: a step action that adds the residual contribution of the external load it applies -- the *negative*
    of that load, see :ref:`residual-and-tangent-convention` -- and the tangent of that contribution, which is
    nonzero only for a load that follows the deformation.

    Every load assembles its own contribution in :meth:`assembleLoad`, wherever it acts: a node force goes to the
    degrees of freedom of its nodes, a distributed load or a body force asks each of its elements for the integral
    over the loaded face or volume. A solver therefore only loops over the loads, see
    :meth:`~edelweissfe.solvers.base.nonlinearsolverbase.NonlinearSolverBase.assembleLoads`, which it finds among the
    step's actions by this base class, whatever step action module holds them (see
    :meth:`~edelweissfe.solvers.base.nonlinearsolverbase.NonlinearSolverBase.collectLoads`): a new kind of load -- one
    acting on the cells or particles of a meshfree model, say -- needs no change to any solver.

    Over a step, a load changes along its amplitude :math:`a(t)`, from its value at the start of the step,
    :math:`f(0)`, to :math:`f(1)` at the end; at the end of the step it keeps the value it reached, from which the next
    step starts (see :meth:`applyAtStepEnd`).
    """

    @abstractmethod
    def assembleLoad(
        self,
        P: DofVector,
        K: VIJSystemMatrix | None,
        U_np: DofVector,
        dU: DofVector,
        timeStep: TimeStep,
        dofManager: DofManager,
    ):
        """Add this load's residual contribution to ``P``, and its tangent to ``K``.

        Both are augmented, never overwritten.

        Parameters
        ----------
        P
            The global residual contribution vector to augment.
        K
            The global system matrix to augment, which hands out the block of an entity (an element, a cell, a
            particle) by ``K[entity]``; None for an explicit solver, which needs no tangent.
        U_np
            The current global solution.
        dU
            The global solution increment since the last converged state.
        timeStep
            The current time step.
        dofManager
            The DOF manager of the equation system, which locates the degrees of freedom the load acts on.
        """

    @abstractmethod
    def applyAtStepEnd(self, model, loadParameter: float | None = None):
        """Keep the value the load reached at the end of the step, from which the next step starts, and go idle until
        a later step declares a change again.

        A step that follows the amplitudes ends at :math:`f(1)`. An arc length step (see :ref:`arc-length-control`)
        applies :math:`f(0) + \\lambda \\, (f(1) - f(0))`, the change over the step scaled by its load parameter
        :math:`\\lambda`, and the load keeps exactly that: the factor of the change is
        :meth:`amplitudeAtStepEnd`.

        Parameters
        ----------
        model
            The current state of the model.
        loadParameter
            The load parameter an arc length step ended at; None for a step that follows the amplitudes.
        """

    @staticmethod
    def amplitudeAtStepEnd(amplitude: Callable[[float], float], loadParameter: float | None) -> float:
        """The value of a load's amplitude at the end of a step, by which the load keeps the change the step declared.

        A step that follows the amplitudes ends at the end of the amplitude, :math:`a(1)`. An arc length step scales the
        change of every load over the step, :math:`f(1) - f(0)`, by its load parameter :math:`\\lambda`, and so it ends
        on the straight line from the start to the end of the amplitude:

        .. math::

            a(0) + \\lambda \\, \\left( a(1) - a(0) \\right).

        With the default linear amplitude, this is the load parameter itself. A load the step holds at a constant
        amplitude, e.g. a self-weight declared with ``f(t)='1'``, is part of the dead load of the arc length step, and
        keeps its full value, whatever the load parameter.

        Parameters
        ----------
        amplitude
            The amplitude of the load over the step progress interval ``[0...1]``.
        loadParameter
            The load parameter an arc length step ended at; None for a step that follows the amplitudes.

        Returns
        -------
        float
            The amplitude at the end of the step.
        """

        if loadParameter is None:
            return amplitude(1.0)

        return amplitude(0.0) + loadParameter * (amplitude(1.0) - amplitude(0.0))
