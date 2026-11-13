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

from edelweissfe.numerics.dofmanager import DofManager, DofVector
from edelweissfe.stepactions.base.stepactionbase import StepActionBase
from edelweissfe.timesteppers.timestep import TimeStep


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
    """

    #: The name of this step action, by which the solver option ``arcLengthController`` selects it.
    name: str

    @abstractmethod
    def computeDDLambda(
        self,
        dU: DofVector,
        ddU_0: DofVector,
        ddU_f: DofVector,
        timeStep: TimeStep,
        dofManager: DofManager,
        dLambda: float,
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
        dofManager
            The dof manager of the current equation system.
        dLambda
            The current increment of the load parameter, for a controller whose equation involves the load parameter
            itself.

        Returns
        -------
        float
            The correction of the load parameter.
        """

    @abstractmethod
    def finishIncrement(self, U: DofVector, dU: DofVector, dLambda: float, timeStep: TimeStep, dofManager: DofManager):
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
        dofManager
            The dof manager of the current equation system.
        """
