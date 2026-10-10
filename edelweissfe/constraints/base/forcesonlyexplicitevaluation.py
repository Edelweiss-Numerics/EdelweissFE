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
A mixin for constraints whose :meth:`applyConstraint` can skip the tangent when handed ``K=None``.
"""

import numpy as np

from edelweissfe.timesteppers.timestep import TimeStep


class ForcesOnlyExplicitEvaluation:
    """Evaluate a constraint in an explicit solver without building a tangent.

    The base implementation of
    :meth:`~edelweissfe.constraints.base.constraintbase.ConstraintBase.applyConstraintExplicit`
    asks for a tangent container through the ordinary sparse-stiffness protocol and then discards it.
    For contact constraints that container is large -- one dense block per contact point, of order
    50 MB per increment on the anchor pry-out -- and filling it costs one outer product per point.

    A constraint that inherits this mixin (listed *before*
    :class:`~edelweissfe.constraints.base.constraintbase.ConstraintBase`) instead runs its one
    :meth:`applyConstraint` loop with ``K=None``. One loop rather than two, so the physics cannot
    drift between the implicit and the explicit path. :meth:`applyConstraint` must therefore skip
    every tangent contribution when ``K`` is None.
    """

    def applyConstraintExplicit(
        self,
        U_np: np.ndarray,
        dU: np.ndarray,
        PExt: np.ndarray,
        timeStep: TimeStep,
    ):
        """Evaluate the constraint forces only, by calling :meth:`applyConstraint` with ``K=None``.

        Parameters
        ----------
        U_np
            The current solution, restricted to this constraint's degrees of freedom.
        dU
            The current solution increment, likewise restricted.
        PExt
            The local residual / force vector to augment.
        timeStep
            The current time step.
        """

        self.applyConstraint(U_np, dU, PExt, None, timeStep)
