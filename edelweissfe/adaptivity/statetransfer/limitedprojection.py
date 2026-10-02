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

"""Bound-preserving, mean-preserving polynomial projection of quadrature-point state."""

import numpy as np

from edelweissfe.adaptivity.statetransfer.projection import PolynomialProjection


class LimitedPolynomialProjection(PolynomialProjection):
    """:class:`~edelweissfe.adaptivity.statetransfer.projection.PolynomialProjection` with a
    Barth-Jespersen-type slope limiter, applied per child element and per state column.

    The fitted polynomial :math:`p` is evaluated at the child quadrature points, which partly lie
    *outside* the parent's Gauss points (for an octree split of a :math:`2\\times2\\times2` rule at
    :math:`\\pm 0.789` versus :math:`\\pm 0.577`), so plain projection extrapolates and can over- or
    undershoot. The limiter keeps the child mean :math:`\\bar{p}` and scales the deviations,

    .. math::

        v_q = \\bar{p} + \\theta \\, (p_q - \\bar{p}), \\qquad
        \\theta = \\max \\{ \\theta \\in [0, 1] : \\min_Q v \\le v_q \\le \\max_Q v \\;\\forall q \\},

    with :math:`Q` the parent quadrature-point values. The result never leaves the parent's value
    range (no new extrema, so non-negative variables stay non-negative), :math:`\\theta = 1`
    recovers plain projection and :math:`\\theta = 0` the child mean. The child mean is the
    arithmetic mean over the child's quadrature points, i.e. the volume average for equal
    quadrature weights on an affine child. It equals the child's share of the parent integral as long
    as it lies inside the parent range (always the case for an octree split of a trilinear fit,
    whose octant averages are interpolated, not extrapolated); otherwise it is clipped to that range.
    """

    def _transferColumns(self, parentValues, parentRefCoords, childRefCoords, childInitValues, columns):
        parentColumns = parentValues[:, columns]
        projected = self._fitAndResample(parentColumns, parentRefCoords, childRefCoords)

        lower = parentColumns.min(axis=0)
        upper = parentColumns.max(axis=0)
        childMean = np.clip(projected.mean(axis=0), lower, upper)
        deviation = projected - childMean

        # largest admissible scaling per point: distance to the violated bound over the deviation
        with np.errstate(divide="ignore", invalid="ignore"):
            admissible = np.where(
                deviation > 0.0,
                (upper - childMean) / deviation,
                np.where(deviation < 0.0, (lower - childMean) / deviation, 1.0),
            )
        theta = np.clip(admissible.min(axis=0), 0.0, 1.0)

        return childMean + theta * deviation
