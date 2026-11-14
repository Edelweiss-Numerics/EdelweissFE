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
A mixin for constraints whose :meth:`computeKernels` can skip the tangent when handed ``K=None``.
"""

import numpy as np

from edelweissfe.timesteppers.timestep import TimeStep


class ForcesOnlyExplicitEvaluation:
    """Evaluate a constraint in an explicit solver without building a tangent.

    The base implementation of
    :meth:`~edelweissfe.constraints.base.constraintbase.ConstraintBase.computeKernelsExplicit`
    asks for a tangent container through the ordinary sparse-stiffness protocol and then discards it.
    For contact constraints that container is large -- one dense block per contact point, of order
    50 MB per increment on the anchor pry-out -- and filling it costs one outer product per point.

    A constraint that inherits this mixin (listed *before*
    :class:`~edelweissfe.constraints.base.constraintbase.ConstraintBase`) instead runs its one
    :meth:`computeKernels` loop with ``K=None``. One loop rather than two, so the physics cannot
    drift between the implicit and the explicit path. :meth:`computeKernels` must therefore skip
    every tangent contribution when ``K`` is None.
    """

    def computeKernelsExplicit(
        self,
        U_np: np.ndarray,
        dU: np.ndarray,
        P: np.ndarray,
        timeStep: TimeStep,
    ):
        """Add the constraint's residual contribution only, by calling :meth:`computeKernels` with ``K=None``.

        Parameters
        ----------
        U_np
            The current solution, restricted to this constraint's degrees of freedom.
        dU
            The current solution increment, likewise restricted.
        P
            The residual contribution to augment, with the sign described in
            :meth:`~edelweissfe.constraints.base.constraintbase.ConstraintBase.computeKernels`.
        timeStep
            The current time step.
        """

        self.computeKernels(U_np, dU, P, None, timeStep)
