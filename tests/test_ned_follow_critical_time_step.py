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
"""``NED`` with ``follow-critical-time-step`` and a contact that closes and opens again and again: a point mass on a
soft spring that hits a stiff rigid wall in every swing, beside a second mass at rest on a stiffer spring.

While the contact is open, the critical time step is the one of the stiffer spring, and the time increment is raised to
it; while it is closed, the one of the wall, shorter. The time increment drops before the mass reaches the wall, not one
increment after: the contact is bounded over the next increment, and the mass, whose gap is at most what it approaches
by in it with its velocity and its acceleration, has its spring in the bound before it closes
(:meth:`~edelweissfe.solvers.nonlinearexplicitdynamic.NED._estimateThePenaltyConstraints`). So:

* the contact closes and opens in every swing, and the time increment follows: long while open, short while closed,
* the energy of the mass, its two springs included, stays what it started with, to the error of the central
  difference, and the mass gets no deeper into the wall than its energy allows,
* the increment in which the mass reaches the wall is the short one,
* without the bound over the next increment, the mass reaches the wall in a long increment, gets far deeper into it
  than its energy allows, and comes back with many times the energy it had,
* lowered only, nothing is bounded over the next increment, and the increment stays at the one of the wall.

The motion that carries the energy is far below the critical time step, as in a model of many elements, whose highest
modes set the critical time step: each change of the increment changes the energy the central difference holds in a
mode by a fraction of the order of :math:`(\\omega \\Delta t)^2`.
"""

import numpy as np
import pytest

from edelweissfe.config.configurator import loadConfiguration
from edelweissfe.constraints.directionalspringpenalty import (
    Constraint as DirectionalSpringPenalty,
)
from edelweissfe.constraints.directionalspringpenalty import (
    DirectionalSpringPenaltySchema,
)
from edelweissfe.constraints.nodetorigidsurfacepenalty import (
    Constraint as NodeToRigidSurfacePenalty,
)
from edelweissfe.constraints.nodetorigidsurfacepenalty import (
    NodeToRigidSurfacePenaltySchema,
)
from edelweissfe.elements.pointmass import PointMass
from edelweissfe.journal.journal import Journal
from edelweissfe.models.femodel import FEModel
from edelweissfe.points.node import Node
from edelweissfe.sets.elementset import ElementSet
from edelweissfe.sets.nodeset import NodeSet
from edelweissfe.solvers.nonlinearexplicitdynamic import NED
from edelweissfe.steps.adaptivestepforexplicitsimulations import (
    AdaptiveStepForExplicitSimulations,
)
from edelweissfe.steps.stepmanager import StepActionCollection
from edelweissfe.utils.fieldoutput import FieldOutputController

#: The mass, the stiffness of its spring to the ground, the penalty of the wall, and where the wall is.
MASS, SPRING, WALL_PENALTY, WALL = 2.0, 50.0, 5000.0, 0.03
#: The stiffness of the spring of the second mass, which stays at rest and sets the critical time step while the wall is
#: open.
STIFFER_SPRING = 800.0
#: The velocity the mass starts with at rest position, towards the wall: it would swing twice as far as the wall is.
VELOCITY = 0.3
#: The energy of the mass.
ENERGY = 0.5 * MASS * VELOCITY**2
#: Below one half, so that the wall stops the mass over a few increments: near one, a penalty contact closes and opens
#: within about two increments, and the energy it gives back is far from the one it took.
COURANT_NUMBER = 0.4
#: The time increments with the wall open and closed: the courant number times the critical time step of the stiffer
#: spring, and of the mass on its spring and the wall.
OPEN_INCREMENT = COURANT_NUMBER * 2.0 * np.sqrt(MASS / STIFFER_SPRING)
CLOSED_INCREMENT = COURANT_NUMBER * 2.0 * np.sqrt(MASS / (SPRING + WALL_PENALTY))


def massOnASpringBeforeAWall() -> FEModel:
    """A point mass at the origin on a spring to the ground along x (``directionalSpringPenalty``), before a rigid wall
    at :data:`WALL` along x (``nodeToRigidSurfacePenalty``), which it reaches in every swing; beside it, a second point
    mass on a stiffer spring."""

    model = FEModel(3)
    for number in (1, 2):
        node = Node(number, np.array([0.0, float(number), 0.0]))
        model.nodes[number] = node
        model.elements[number] = PointMass(number, [node], model, MASS)
    model.elementSets["all"] = ElementSet("all", list(model.elements.values()))
    model.nodeSets["all"] = NodeSet("all", list(model.nodes.values()))
    model.nodeSets["mass"] = NodeSet("mass", [model.nodes[1]])
    model.nodeSets["secondMass"] = NodeSet("secondMass", [model.nodes[2]])
    for name, nodeSet, stiffness in (("spring", "mass", SPRING), ("stifferSpring", "secondMass", STIFFER_SPRING)):
        model.constraints[name] = DirectionalSpringPenalty(
            name,
            model,
            model.nodeSets[nodeSet],
            configuration=DirectionalSpringPenaltySchema(
                field="displacement", component=0, penalty=stiffness, nSet=nodeSet
            ),
        )
    model.constraints["wall"] = NodeToRigidSurfacePenalty(
        "wall",
        model,
        model.nodeSets["mass"],
        Journal(verbose=False),
        configuration=NodeToRigidSurfacePenaltySchema(
            field="displacement", component=0, penalty=WALL_PENALTY, nSet="mass", value=WALL, direction=1.0
        ),
    )
    return model


class _RecordingNED(NED):
    """``NED``, recording after every increment its time increment, the displacement and the velocity of the mass."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.history = []

    def acceptIncrement(self, step, model, timeStep):
        super().acceptIncrement(step, model, timeStep)
        displacement = model.nodeFields["displacement"]
        node = displacement.indexOfNode(model.nodes[1])
        self.history.append((timeStep.timeIncrement, displacement["U"][node, 0], displacement["V"][node, 0]))


def run(stepLength: float, follow: bool = True) -> np.ndarray:
    """One explicit step of the given length, which asks for the whole step as its increment, by default with
    ``follow-critical-time-step``; the mass starts at rest position with :data:`VELOCITY` towards the wall.

    Returns
    -------
    numpy.ndarray
        Per increment after the zero increment: the time increment and the displacement of the mass, and its energy
        (one less).
    """

    journal = Journal(verbose=False)
    model = massOnASpringBeforeAWall()
    model.prepareYourself(journal)
    displacement = model.nodeFields["displacement"]
    displacement["V"][displacement.indexOfNode(model.nodes[1]), 0] = VELOCITY

    fieldOutputController = FieldOutputController(model, journal)
    fieldOutputController.initializeJob()
    jobInfo = loadConfiguration(dict())
    solver = _RecordingNED(jobInfo, journal, **{"courant-number": COURANT_NUMBER, "follow-critical-time-step": follow})
    AdaptiveStepForExplicitSimulations(
        0,
        model,
        fieldOutputController,
        journal,
        jobInfo,
        solver,
        [],
        StepActionCollection(),
        stepLength=stepLength,
        maxNumInc=100000,
    ).solve()
    fieldOutputController.finalizeJob()

    timeIncrement, x, v = np.array(solver.history[1:]).T
    # The energy the central difference holds for the mass on its springs, with the velocities of the increments before
    # and after a displacement, :math:`\\tfrac{1}{2} m v_{n-1/2} v_{n+1/2} + \\tfrac{1}{2} k x_n^2`: the one it keeps
    # exactly on a spring at a constant increment.
    penetration = np.maximum(x[:-1] - WALL, 0.0)
    energy = 0.5 * MASS * v[:-1] * v[1:] + 0.5 * SPRING * x[:-1] ** 2 + 0.5 * WALL_PENALTY * penetration**2
    return timeIncrement, x, energy


#: Ten swings of the mass on its spring alone; it reaches the wall in every one.
STEP_LENGTH = 10 * 2.0 * np.pi * np.sqrt(MASS / SPRING)
#: How deep the mass gets into the wall with its energy: the depth :math:`g` at which both springs hold all of it,
#: :math:`\\tfrac{1}{2} k_s (a + g)^2 + \\tfrac{1}{2} k_w g^2 = E`.
DEEPEST = (
    -SPRING * WALL + np.sqrt((SPRING * WALL) ** 2 - (SPRING + WALL_PENALTY) * (SPRING * WALL**2 - 2.0 * ENERGY))
) / (SPRING + WALL_PENALTY)


def test_the_contact_closes_and_opens_in_every_swing_and_the_increment_follows():
    timeIncrement, x, _ = run(STEP_LENGTH)

    closed = x > WALL
    closings = np.count_nonzero(closed[1:] & ~closed[:-1])
    assert closings >= 9

    # long while the mass is far from the wall, short while it is in it (the last increment ends the step)
    np.testing.assert_allclose(timeIncrement[:-1].max(), OPEN_INCREMENT, rtol=1e-12)
    np.testing.assert_allclose(timeIncrement[:-1][closed[:-1]], CLOSED_INCREMENT, rtol=1e-12)


def energyInFreeFlight(timeIncrement: np.ndarray, x: np.ndarray, energy: np.ndarray) -> np.ndarray:
    """The energy of the mass where it is away from the wall at a constant increment, where the central difference
    keeps it exactly: what each contact left of it."""

    free = (x[:-2] <= WALL) & (x[1:-1] <= WALL) & (x[2:] <= WALL) & (timeIncrement[1:-1] == timeIncrement[2:])
    return energy[1:][free]


def test_the_energy_stays_and_the_mass_gets_no_deeper_into_the_wall_than_its_energy_allows():
    timeIncrement, x, energy = run(STEP_LENGTH)

    energyInFreeFlightNow = energyInFreeFlight(timeIncrement, x, energy)
    assert len(energyInFreeFlightNow) > 100
    np.testing.assert_allclose(energyInFreeFlightNow, ENERGY, rtol=0.06)
    assert x.max() - WALL < 1.05 * DEEPEST


def test_the_increment_in_which_the_mass_reaches_the_wall_is_the_short_one():
    timeIncrement, x, _ = run(STEP_LENGTH)

    reachesTheWall = (x[1:] > WALL) & (x[:-1] <= WALL)
    np.testing.assert_allclose(timeIncrement[1:][reachesTheWall], CLOSED_INCREMENT, rtol=1e-12)


def test_without_the_bound_over_the_next_increment_the_mass_reaches_the_wall_in_a_long_increment(monkeypatch):
    """The control: the contact bounded at the state after an increment only, as without ``follow-critical-time-step``,
    while the increment is raised to the open one."""

    monkeypatch.setattr(NED, "_theLargestNextIncrement", lambda self, step, timeStep: None)  # bound at the state
    timeIncrement, x, energy = run(STEP_LENGTH)

    reachesTheWall = (x[1:] > WALL) & (x[:-1] <= WALL)
    assert np.max(timeIncrement[1:][reachesTheWall]) == pytest.approx(OPEN_INCREMENT, rel=1e-12)
    assert x.max() - WALL > 2 * DEEPEST
    assert energyInFreeFlight(timeIncrement, x, energy).max() > 3 * ENERGY


def test_lowered_only_the_constraints_are_bounded_at_the_state_alone(monkeypatch):
    """Without ``follow-critical-time-step``, nothing is bounded over the next increment, as before the option
    existed."""

    def refused(self, step, timeStep):
        raise AssertionError("bounded over the next increment")

    monkeypatch.setattr(NED, "_theLargestNextIncrement", refused)
    timeIncrement, _, _ = run(STEP_LENGTH, follow=False)
    # lowered once, for good, when the mass first reaches the wall
    np.testing.assert_allclose(timeIncrement[-20:-1], CLOSED_INCREMENT, rtol=1e-12)
