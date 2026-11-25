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
"""The node-to-surface contact (:mod:`edelweissfe.constraints.nodetodeformablesurfacepenalty`): the bound of its
stiffness for ``NED``'s critical time step, from the springs of its force."""

import numpy as np
import pytest
from _hexa20cube import twoBlockModel

import edelweissfe.utils.inputfileparser  # noqa: F401 bootstrap input language
from edelweissfe.constraints.base.constraintbase import ConstraintBase
from edelweissfe.constraints.nodetodeformablesurfacepenalty import (
    Constraint as NodeToSurfaceContact,
)
from edelweissfe.constraints.nodetodeformablesurfacepenalty import (
    NodeToDeformableSurfacePenaltySchema,
)
from edelweissfe.journal.journal import Journal
from edelweissfe.solvers.base.criticaltimestep import NextIncrement
from edelweissfe.timesteppers.timestep import TimeStep


def _nodeToSurfaceContact(penetration=0.05, **options) -> NodeToSurfaceContact:
    """The slave nodes of the lower hexa20 cube's top face against the upper one's bottom face, searched once."""

    journal = Journal()
    model, slaveSurface, masterSurface = twoBlockModel(journal, penetration)
    configuration = NodeToDeformableSurfacePenaltySchema(penalty=1.0e4, searchDistance=1.0, **options)
    constraint = NodeToSurfaceContact(
        "theContact", model, slaveSurface, masterSurface, journal, configuration=configuration
    )
    constraint.updateConnectivity(model)
    return constraint


def _bounds(constraint, U: np.ndarray, dU: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """The bound of the constraint, and the row sums of the absolute value of its tangent (the default of
    :class:`~edelweissfe.constraints.base.constraintbase.ConstraintBase`), weighted by random columns."""

    dU = np.zeros_like(U) if dU is None else dU
    weights = np.random.default_rng(3).uniform(0.5, 2.0, constraint.nDof)
    bound, ofTheTangent = np.zeros(constraint.nDof), np.zeros(constraint.nDof)
    constraint.addBoundOfAbsoluteStiffnessTimes(U, dU, weights, bound, None)
    ConstraintBase.addBoundOfAbsoluteStiffnessTimes(constraint, U, dU, weights, ofTheTangent, None)
    assert ofTheTangent.max() > 0.0
    return bound, ofTheTangent


def _slaveMovedBy(constraint, motion: np.ndarray) -> np.ndarray:
    """A vector of the block that moves every slave node by ``motion`` and holds the nodes of the facets."""

    dU = np.zeros(constraint.nDof)
    for _, pIdcs, _ in constraint._slavesWithAFacet():
        dU[pIdcs] = motion
    return dU


@pytest.mark.parametrize("contactType", ["linear", "quadratic"])
def test_without_friction_and_with_small_sliding_the_bound_is_the_tangent(contactType):
    """Without friction and with small sliding, the tangent of a slave node is its one normal spring,
    :math:`k\\, \\boldsymbol{a}_p \\otimes \\boldsymbol{a}_p`, on its own copy of the nodes of its facet: the bound from
    the springs is the row sum of the absolute tangent itself."""

    constraint = _nodeToSurfaceContact(contactType=contactType, sliding="small")
    U = 1e-3 * np.random.default_rng(0).standard_normal(constraint.nDof)
    bound, ofTheTangent = _bounds(constraint, U)
    np.testing.assert_allclose(bound, ofTheTangent, rtol=1e-12, atol=1e-12 * ofTheTangent.max())


def test_a_sticking_node_has_springs_along_the_tangent_plane():
    """With friction, a sticking slave node adds a spring of the tangential penalty along every direction of the
    tangent plane: the bound holds the row sums of the absolute tangent, normal and tangential."""

    constraint = _nodeToSurfaceContact(sliding="small", mu=0.3, tangentPenalty=2.0e4)
    bound, ofTheTangent = _bounds(constraint, np.zeros(constraint.nDof))
    assert np.all(bound >= ofTheTangent * (1 - 1e-12))
    withoutFriction, _ = _bounds(_nodeToSurfaceContact(sliding="small"), np.zeros(constraint.nDof))
    assert np.all(bound >= withoutFriction) and np.any(bound > 1.5 * withoutFriction)


def test_the_bound_holds_the_tangent_of_a_slipping_node():
    """Slid along the surface within the increment, every slave node slips: its tangent couples the tangential to the
    normal force, :math:`\\mu k\\, ([c, -N^m] \\otimes s) \\otimes \\boldsymbol{a}_p`, and is unsymmetric. The bound
    holds what the default bound of the tangent gives: in every row the larger of the row sums of its absolute value and
    of their mean with its column sums."""

    constraint = _nodeToSurfaceContact(sliding="small", mu=0.3, tangentPenalty=2.0e4)
    dU = _slaveMovedBy(constraint, np.array([0.03, 0.0, 0.01]))
    bound, ofTheTangent = _bounds(constraint, dU, dU)
    tangentialForces = np.linalg.norm(constraint._tangentialForceCurrent, axis=1)
    np.testing.assert_allclose(tangentialForces, 0.3 * -constraint._normalForceCurrent, rtol=1e-12)  # all slip
    assert np.all(bound >= ofTheTangent * (1 - 1e-12))


@pytest.mark.parametrize("contactType", ["linear", "quadratic"])
def test_with_finite_sliding_the_bound_holds_the_geometric_stiffness(contactType):
    """With finite sliding the tangent adds the geometric stiffness, the normal force times the second derivative of
    the gap, :math:`f_n \\boldsymbol{H}`. At a penetration of 0.05 it is not negligible against the spring; the bound
    adds its absolute row sums and holds the tangent's, row by row."""

    constraint = _nodeToSurfaceContact(contactType=contactType, sliding="finite")
    bound, ofTheTangent = _bounds(constraint, np.zeros(constraint.nDof))
    assert np.all(bound >= ofTheTangent * (1 - 1e-12))
    assert np.any(bound > ofTheTangent * (1 + 1e-3))


@pytest.mark.parametrize("mu", [0.0, 0.3])
def test_over_the_next_increment_a_node_that_could_close_has_its_springs(mu):
    """The slave face lies 0.05 below the master face: open now, its bound is zero. Over a next increment in which
    the slave moves up by 0.1, from its velocity or from its acceleration, every slave node closes within it, and the
    bound over it is the one at its end, with the gradients of now (the same here, with small sliding); with friction
    the springs along the tangent plane come with it. Over one in which the slave moves up by 0.02, or down, no node
    closes (``NED``'s ``follow-critical-time-step``,
    :func:`~edelweissfe.constraints.base.penaltysprings.smallestGapsWithin`)."""

    constraint = _nodeToSurfaceContact(penetration=-0.05, sliding="small", mu=mu, tangentPenalty=2.0e4)
    zero = np.zeros(constraint.nDof)
    weights = np.random.default_rng(3).uniform(0.5, 2.0, constraint.nDof)
    timeStep = TimeStep(1, 0.1, 0.1, 0.1, 0.1, 0.1)

    def slaveMovedUpBy(motion: float) -> np.ndarray:
        return _slaveMovedBy(constraint, np.array([0.0, motion, 0.0]))

    def boundOver(nextIncrement: NextIncrement | None, U: np.ndarray = zero) -> np.ndarray:
        bound = np.zeros(constraint.nDof)
        constraint.addBoundOfAbsoluteStiffnessTimes(U, zero, weights, bound, timeStep, nextIncrement)
        return bound

    assert not boundOver(None).any()
    atTheEnd = boundOver(None, slaveMovedUpBy(0.1))
    assert atTheEnd[np.flatnonzero(slaveMovedUpBy(1.0))].min() > 0.0
    closing = boundOver(NextIncrement(slaveMovedUpBy(0.1), zero, timeStep))
    np.testing.assert_allclose(closing, atTheEnd, rtol=1e-12)
    accelerating = boundOver(NextIncrement(zero, slaveMovedUpBy(0.1), timeStep))
    np.testing.assert_allclose(accelerating, atTheEnd, rtol=1e-12)
    assert not boundOver(NextIncrement(slaveMovedUpBy(0.02), zero, timeStep)).any()
    assert not boundOver(NextIncrement(slaveMovedUpBy(-0.1), zero, timeStep)).any()
