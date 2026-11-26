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
point, the displacement of a tied point from rest or of a held point from its prescribed value along one axis --; the
vector with which it acts on the degrees of freedom of the constraint is the gradient of the stretch,
:math:`\\boldsymbol{a}_p = \\partial g_p/\\partial \\boldsymbol{u}`, and its law gives it a force :math:`f_p` and a
stiffness :math:`k_p = \\mathrm{d}f_p/\\mathrm{d}g_p`. The residual contribution of the springs is
:math:`\\sum_p f_p \\boldsymbol{a}_p`, their stiffness
:math:`\\sum_p k_p\\, \\boldsymbol{a}_p \\otimes \\boldsymbol{a}_p`.

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
from the same code it takes them for its force, so that the force and the bound cannot drift apart. A contact law gives
its springs at the paired contact points as :class:`PenaltySprings` (:mod:`~edelweissfe.constraints.base.penaltylaw`).

The gradient of a spring moves nodes along the direction of the spring, :math:`a_{p,(n,j)} = C_{pn}\\, d_{pj}`: the
coefficient :math:`C_{pn}` with which the spring moves node :math:`n` (an interpolation vector, or the negative shape
functions of a master face) times the component :math:`j` of its direction. It comes in parts, of two kinds:

* :class:`SpringGradientOnSharedNodes`: the springs share their nodes, and the coefficients are a matrix over them --
  the interpolation matrix of contact points
  (:meth:`~edelweissfe.constraints.base.contactpoints.ContactPoints.interpolationMatrixOfLocatedPoints`), the
  interpolation vectors of the points of a particle;
* :class:`SpringGradientOnOwnNodes`: every spring has nodes of its own in the block -- the parent face nodes of the
  master facet of a contact point, a slave node and its facet, the reference point of a rigid body.

**Over the next increment.** With ``follow-critical-time-step``, ``NED`` asks for the bound over the states the next
increment can reach (:class:`~edelweissfe.solvers.base.criticaltimestep.NextIncrement`). A spring that switches on
where its stretch turns negative -- a contact point that closes -- is in that bound if it is closed at its smallest
stretch within the increment. With the pairing and the normals held, as within an increment, the gap of a point is
affine in the solution, and so its smallest value within the increment is at a corner of the states the increment can
reach, :math:`g + \\min(0, \\Delta g_v) + \\min(0, \\Delta g_a)`, with the changes of the gap by the part of the
increment from the velocity and by the one from the acceleration (:func:`smallestGapsWithin`). A constraint that bounds
its springs takes them at these gaps, and sums over its springs: a point that could close within the increment has its
spring in the bound before it closes, whichever degrees of freedom the springs share, and a law whose stiffness grows
with the depth is stiffest there.
"""

from dataclasses import dataclass

import numpy as np

from edelweissfe.solvers.base.criticaltimestep import NextIncrement
from edelweissfe.timesteppers.timestep import TimeStep
from edelweissfe.utils.facetcontactgeometry import rowDot


@dataclass(frozen=True)
class PenaltySprings:
    """Penalty springs at located contact points, at most one per point, each along a unit direction: the spring of a
    contact along the normal at a closed point, or the spring of a tie along one axis. A law makes them
    (:meth:`~edelweissfe.constraints.base.penaltylaw.PenaltyLaw.springs`); the constraint takes from them its force,
    its tangent, the bound of its stiffness and its outputs."""

    #: The located points with a spring (indices into the located points).
    points: np.ndarray
    #: The direction of each spring, (len(points), nDim).
    directions: np.ndarray
    #: The force of each spring along its direction, :math:`f`.
    forces: np.ndarray
    #: The stiffness of each spring, :math:`k = \mathrm{d}f/\mathrm{d}g`.
    stiffness: np.ndarray
    #: The energy stored in each spring, :math:`\Pi` with :math:`\mathrm{d}\Pi/\mathrm{d}g = f`.
    energies: np.ndarray


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


class SpringGradientOnOwnNodes:
    """The part of the gradients of springs on nodes of their own: spring :math:`p` moves its nodes :math:`n` with
    :math:`a_{p,(n,j)} = N_{pn}\\, d_{pj}`, at the degrees of freedom ``dofs[p, n, j]`` of the block. Springs may name
    the same degrees of freedom (the reference point of a rigid body). Only the absolute values enter the bound; they
    are kept.

    Parameters
    ----------
    coefficients
        :math:`N`, (nSprings, nNodes).
    directions
        The direction of each spring, (nSprings, nComponents).
    dofs
        Where the components of the nodes of each spring sit in the block, (nSprings, nNodes, nComponents).
    """

    def __init__(self, coefficients: np.ndarray, directions: np.ndarray, dofs: np.ndarray):
        self.absoluteCoefficients = np.abs(coefficients)
        self.absoluteDirections = np.abs(directions)
        self.dofs = dofs

    def absoluteTimes(self, weights: np.ndarray) -> np.ndarray:
        """:math:`|\\boldsymbol{a}_p| \\cdot \\boldsymbol{w}` of this part, per spring; see
        :meth:`SpringGradientOnSharedNodes.absoluteTimes`."""

        return np.einsum("ai,aij,aj->a", self.absoluteCoefficients, weights[self.dofs], self.absoluteDirections)

    def addAbsoluteTransposeTimes(self, values: np.ndarray, bound: np.ndarray):
        """Distribute values of the springs with the absolute gradients of this part; see
        :meth:`SpringGradientOnSharedNodes.addAbsoluteTransposeTimes`."""

        np.add.at(
            bound,
            self.dofs,
            values[:, None, None] * self.absoluteCoefficients[:, :, None] * self.absoluteDirections[:, None, :],
        )


def addBoundOfAbsoluteStiffnessOfSprings(
    bound: np.ndarray,
    weights: np.ndarray,
    stiffness: np.ndarray,
    gradient: list[SpringGradientOnSharedNodes | SpringGradientOnOwnNodes],
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


def smallestGapsWithin(
    nextIncrement: NextIncrement, gapsAt, U_np: np.ndarray, dU: np.ndarray, timeStep: TimeStep
) -> np.ndarray:
    """The smallest gap each contact point may have within the next increment, at which a contact takes its springs
    for the bound over it (see the module documentation, "Over the next increment"). With the pairing and the normals
    held, a gap is affine in the solution, and changes over an increment of length :math:`s` by :math:`\\alpha\\,
    \\Delta g_v + \\beta\\, \\Delta g_a`, :math:`\\alpha, \\beta \\in [0, 1]`
    (:class:`~edelweissfe.solvers.base.criticaltimestep.NextIncrement`), with the changes :math:`\\Delta g_v` and
    :math:`\\Delta g_a` of the gap by the two parts of the increment at its largest length. Each part, where it closes
    the gap, closes it
    most at its full length, so within the increment, whatever its length and wherever the point closes in it, the gap
    is at least :math:`g + \\min(0, \\Delta g_v) + \\min(0, \\Delta g_a)`.

    Parameters
    ----------
    nextIncrement
        The next increment, restricted to the constraint.
    gapsAt
        The gaps of the contact points at a state, ``gapsAt(U_np, dU, timeStep)``.
    U_np
        The solution now, restricted to the constraint.
    dU
        The increment now, likewise.
    timeStep
        The time step now.

    Returns
    -------
    numpy.ndarray
        The smallest gaps, one per contact point.
    """

    gaps = gapsAt(U_np, dU, timeStep)
    move = nextIncrement.byTheVelocity
    byTheVelocity = gapsAt(U_np + move, dU + move, nextIncrement.timeStep) - gaps
    move = nextIncrement.byTheAcceleration
    byTheAcceleration = gapsAt(U_np + move, dU + move, timeStep) - gaps
    return gaps + np.minimum(byTheVelocity, 0.0) + np.minimum(byTheAcceleration, 0.0)
