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
"""Penalty springs, and the bound of their stiffness that the explicit solver adds to the squared frequencies of the
degrees of freedom they couple, which lowers the critical time step.

A penalty constraint acts through springs. A spring :math:`p` measures a stretch :math:`g_p` -- the gap of a contact
point, the displacement of a held point from its prescribed value along one axis --; the vector with which it acts on
the degrees of freedom of the constraint is the gradient of the stretch, :math:`\\boldsymbol{a}_p =
\\partial g_p/\\partial \\boldsymbol{u}`, and its law gives it a force :math:`f_p` and a stiffness :math:`k_p =
\\mathrm{d}f_p/\\mathrm{d}g_p`. The residual contribution of the springs is :math:`\\sum_p f_p \\boldsymbol{a}_p`,
their stiffness :math:`\\sum_p k_p\\, \\boldsymbol{a}_p \\otimes \\boldsymbol{a}_p`.

The explicit dynamic solver ``NED`` needs the row sums of the absolute value of that stiffness, weighted by column
(:meth:`~edelweissfe.constraints.base.constraintbase.ConstraintBase.addBoundOfAbsoluteStiffnessTimes`). The absolute
value of a sum is at most the sum of the absolute values, so, with the weights :math:`\\boldsymbol{w}` of the columns,

.. math::

    b_i \\mathrel{+}= \\sum_p |k_p|\\, |a_{p,i}| \\, \\big(|\\boldsymbol{a}_p| \\cdot \\boldsymbol{w}\\big) :

one product of the absolute gradients with the weights per spring, and one distribution back, as the force is
distributed. The stiffness of a spring is symmetric, so this bounds the rows of its symmetric part as well, as the
contract asks. :func:`addBoundOfAbsoluteStiffnessOfSprings` is that formula, written once for every penalty constraint
of EdelweissFE and EdelweissMeshfree that bounds its springs instead of its tangent: one that has no tangent, or whose
tangent is costly, overrides ``addBoundOfAbsoluteStiffnessTimes`` with it. A constraint takes its springs for the bound
from the same code it takes them for its force, so that the force and the bound cannot drift apart.

The gradient of a spring moves nodes along the direction of the spring, :math:`a_{p,(n,j)} = C_{pn}\\, d_{pj}`: the
coefficient :math:`C_{pn}` with which the spring moves node :math:`n` -- e.g. the interpolation vector of a point of a
particle -- times the component :math:`j` of its direction (:class:`SpringGradientOnSharedNodes`).
"""

import numpy as np

from edelweissfe.utils.facetcontactgeometry import rowDot


class SpringGradientOnSharedNodes:
    """The part of the gradients of springs on nodes they share: :math:`a_{p,(n,j)} = C_{pn}\\, d_{pj}`, with a matrix
    :math:`C` of coefficients over the nodes. Only the absolute values enter the bound; they are kept.

    Parameters
    ----------
    interpolation
        :math:`C`, one row per spring, one column per node: a dense array or a SciPy sparse matrix.
    directions
        The direction of each spring, (nSprings, nComponents).
    dofsOfNodes
        Where the components of each node sit in the block of the constraint, (nNodes, nComponents): distinct
        degrees of freedom, each node once, so that the fancy indexing of :meth:`addAbsoluteTransposeTimes` drops no
        contribution (it keeps one of repeated indices). Not checked, since it is built anew in every increment.
    """

    def __init__(self, interpolation, directions: np.ndarray, dofsOfNodes: np.ndarray):
        self.absoluteInterpolation = abs(interpolation)
        self.absoluteDirections = np.abs(directions)
        self.dofsOfNodes = dofsOfNodes

    def absoluteTimes(self, weights: np.ndarray) -> np.ndarray:
        """:math:`|\\boldsymbol{a}_p| \\cdot \\boldsymbol{w}` of this part, per spring.

        Parameters
        ----------
        weights
            The weight of each degree of freedom of the block.

        Returns
        -------
        numpy.ndarray
            One value per spring.
        """

        return rowDot(self.absoluteDirections, self.absoluteInterpolation @ weights[self.dofsOfNodes])

    def addAbsoluteTransposeTimes(self, values: np.ndarray, bound: np.ndarray):
        """Distribute values of the springs with the absolute gradients of this part: :math:`b_i \\mathrel{+}= \\sum_p
        |a_{p,i}|\\, v_p`.

        Parameters
        ----------
        values
            One value per spring.
        bound
            The bound, one entry per degree of freedom of the block; augmented.
        """

        bound[self.dofsOfNodes] += self.absoluteInterpolation.T @ (values[:, None] * self.absoluteDirections)


def addBoundOfAbsoluteStiffnessOfSprings(
    bound: np.ndarray,
    weights: np.ndarray,
    stiffness: np.ndarray,
    gradient: list[SpringGradientOnSharedNodes],
):
    """Add the row sums of the absolute value of the stiffness of penalty springs, weighted by column,
    :math:`b_i \\mathrel{+}= \\sum_p |k_p|\\, |a_{p,i}| \\, (|\\boldsymbol{a}_p| \\cdot \\boldsymbol{w})` (see the
    module documentation).

    Parameters
    ----------
    bound
        The row sums, one per degree of freedom of the block of the constraint; augmented.
    weights
        The weight of each column, one per degree of freedom of the block.
    stiffness
        The stiffness :math:`k_p` of each spring.
    gradient
        The parts of the gradients of the springs, each over all springs.
    """

    weighted = sum(part.absoluteTimes(weights) for part in gradient)
    stiffnessTimesWeighted = np.abs(stiffness) * weighted
    for part in gradient:
        part.addAbsoluteTransposeTimes(stiffnessTimesWeighted, bound)
