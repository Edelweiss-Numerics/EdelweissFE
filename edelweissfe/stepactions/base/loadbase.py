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
    :meth:`~edelweissfe.solvers.base.nonlinearsolverbase.NonlinearSolverBase.assembleLoads`, and a new kind of
    load -- one acting on the cells or particles of a meshfree model, say -- needs no change to any solver.
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
