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
"""``NED`` and the entities that carry their own inertia, on the smallest model that has them: point masses on springs.

An entity that carries its own inertia (:attr:`~edelweissfe.models.discretesystem.DiscreteSystem.entitiesWithOwnInertia`)
keeps its state in itself -- here, the displacement and the velocity of a point mass on a spring to the ground. ``NED``
hands it the increment of its node (``advanceEntitiesWithOwnInertia``), the model accepts the state and reconnects the
entities, and the entity returns, on its node, its internal force and its lumped momentum, its lumped mass and damping
and, when due, the bound of its stable time increment. The node holds the coefficients of one increment only. The RKPM
particles of EdelweissMeshfree are such entities; :class:`MassOnASpring` and :class:`SystemOfMassesOnSprings` are the
same contract in a few lines of Python, and the tests check ``NED``'s path for it in EdelweissFE itself:

* the calls of an increment come in the order of the module documentation of ``NED``,
* the motion starts from the velocity projected from the momentum, and the leapfrog then is the central difference
  recursion of the oscillator, to round-off; its period is the analytic one, to second order in the increment,
* the bound of the entities' stable time increment sets the increment of a step that asks for a larger one.
"""

import numpy as np
import pytest

from edelweissfe.config.configurator import loadConfiguration
from edelweissfe.elements.pointmass import PointMass
from edelweissfe.journal.journal import Journal
from edelweissfe.models.discretesystem import DiscreteSystem
from edelweissfe.models.femodel import FEModel
from edelweissfe.numerics.dofvector import DofVector
from edelweissfe.points.node import Node
from edelweissfe.sets.elementset import ElementSet
from edelweissfe.sets.nodeset import NodeSet
from edelweissfe.solvers.nonlinearexplicitdynamic import NED
from edelweissfe.steps.adaptivestepforexplicitsimulations import (
    AdaptiveStepForExplicitSimulations,
)
from edelweissfe.steps.stepmanager import StepActionCollection
from edelweissfe.utils.fieldoutput import FieldOutputController

MASS, STIFFNESS = 2.0, 50.0
#: The initial velocities of the masses, one per mass: each oscillates along x with the amplitude velocity / omega.
VELOCITIES = (0.3, -0.1)
OMEGA = np.sqrt(STIFFNESS / MASS)
#: The bound of the stable time increment of one mass on its spring, 2 / omega.
STABLE_TIME_INCREMENT = 2.0 / OMEGA


class MassOnASpring(PointMass):
    """A point mass on a spring to the ground that carries its displacement and its velocity in itself, as a particle
    carries its state. As a point mass, it sits on one node and lumps its mass there."""

    def __init__(self, number: int, node: Node, model: FEModel, velocity: float):
        super().__init__(number, [node], model, MASS)
        self.displacement = np.zeros(model.domainSize)
        self.velocity = np.array([velocity, 0.0])

    def advance(self, dU: np.ndarray, timeIncrement: float):
        """Take the increment of the node, and the velocity it was taken with; a zero-length increment (the one that
        starts a step) leaves the velocity alone."""

        self.displacement += dU
        if timeIncrement > 0.0:
            self.velocity = dU / timeIncrement


class SystemOfMassesOnSprings(DiscreteSystem):
    """The discrete system of a model whose entities are :class:`MassOnASpring` s: no finite element integrates a mass,
    every entity carries its own inertia. It records the calls ``NED`` makes, in ``model.calls``."""

    @property
    def elementsWithNodalInertia(self) -> list:
        return []

    @property
    def entitiesWithOwnInertia(self) -> list:
        return self.elements

    def dofsWithNodalInertia(self, fieldNames: list) -> np.ndarray:
        # every node is the node of a mass, and holds the coefficients of one increment, not a total solution
        return np.zeros(0, dtype=int)

    def advanceEntitiesWithOwnInertia(self, dU, timeStep, forEachChunk):
        self.model.calls.append("advance")

        def advance(masses):
            for mass in masses:
                mass.advance(dU[mass], timeStep.timeIncrement)

        forEachChunk(self.entitiesWithOwnInertia, advance)

    def assembleInternalForceAndMomentumOfEntitiesWithOwnInertia(self, forEachChunk):
        self.model.calls.append("force and momentum")
        # the residual contribution of the spring, the negative of the force it exerts on the node, with the sign of
        # every element and constraint of EdelweissFE
        return self._onTheNodes(lambda mass: STIFFNESS * mass.displacement), self._onTheNodes(
            lambda mass: mass.mass * mass.velocity
        )

    def assembleLumpedMassOfEntitiesWithOwnInertia(self, forEachChunk):
        self.model.calls.append("mass")
        return self._onTheNodes(lambda mass: np.full(mass.nDof, mass.mass))

    def assembleLumpedDampingOfEntitiesWithOwnInertia(self, forEachChunk):
        self.model.calls.append("damping")
        return self._onTheNodes(lambda mass: np.zeros(mass.nDof))

    def stableTimeIncrementOfEntitiesWithOwnInertia(self, lumpedMass, lumpedDamping, forEachChunk):
        self.model.calls.append("bound")
        # each mass on its spring alone: 2 / omega, from the mass the solver has just assembled
        return min(2.0 * np.sqrt(lumpedMass[mass][0] / STIFFNESS) for mass in self.entitiesWithOwnInertia)

    def _onTheNodes(self, valueOf) -> DofVector:
        vector = self.locations.constructDofVector()
        vector[:] = 0.0
        for mass in self.entitiesWithOwnInertia:
            vector[mass] = valueOf(mass)
        return vector


class ModelOfMassesOnSprings(FEModel):
    """A model of masses on springs, one per node, that answers with :class:`SystemOfMassesOnSprings`, and records
    what ``NED`` asks of it."""

    def __init__(self):
        super().__init__(2)
        self.calls = []
        for number, velocity in enumerate(VELOCITIES, start=1):
            node = Node(number, np.array([float(number), 0.0]))
            self.nodes[number] = node
            self.elements[number] = MassOnASpring(number, node, self, velocity)
        self.elementSets["all"] = ElementSet("all", list(self.elements.values()))
        self.nodeSets["all"] = NodeSet("all", list(self.nodes.values()))

    def buildDiscreteSystem(self) -> SystemOfMassesOnSprings:
        return SystemOfMassesOnSprings(self, super().buildDiscreteSystem().locations)

    def advanceToTime(self, time: float):
        self.calls.append("accept")
        return super().advanceToTime(time)

    def updateConnectivity(self):
        self.calls.append("reconnect")
        return super().updateConnectivity()


class _RecordingNED(NED):
    """``NED``, recording the increment and the displacement of every mass after each increment it accepts."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.history = []

    def acceptIncrement(self, step, model, timeStep):
        super().acceptIncrement(step, model, timeStep)
        self.history.append((timeStep.timeIncrement, [mass.displacement[0] for mass in model.elements.values()]))


def _run(stepLength: float, **options) -> tuple[ModelOfMassesOnSprings, _RecordingNED]:
    """Run one explicit step of the given length, which asks for the whole step as its increment."""

    journal = Journal(verbose=False)
    model = ModelOfMassesOnSprings()
    model.prepareYourself(journal)
    model.advanceToTime(0.0)
    model.calls.clear()

    fieldOutputController = FieldOutputController(model, journal)
    fieldOutputController.initializeJob()
    jobInfo = loadConfiguration(dict())
    solver = _RecordingNED(jobInfo, journal, **options)
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

    return model, solver


def test_the_calls_of_an_increment_come_in_the_order_of_the_module_documentation():
    """The step starts with a reconnection (the step-start topology update). Then, in every increment: advance the
    entities on the connectivity of the start of the increment; accept the state and reconnect; then, on the new
    connectivity, evaluate the force and the momentum, the mass, the damping, and the bound when it is due: in the zero
    increment, and here after every third increment."""

    model, solver = _run(10 * STABLE_TIME_INCREMENT, **{"stable-increment-frequency": 3})

    oneIncrement = ["advance", "accept", "reconnect", "force and momentum", "mass", "damping"]
    expected = ["reconnect"]
    for number in range(len(solver.history)):
        expected += oneIncrement + (["bound"] if number % 3 == 0 else [])
    assert model.calls[: len(expected)] == expected


def test_the_bound_of_the_entities_sets_the_increment_of_a_step_that_asks_for_more():
    """The step asks for its whole length; the increment taken is the bound of the entities times the courant
    number, from the first increment on (the last one ends the step)."""

    courantNumber = 0.5
    _, solver = _run(10 * STABLE_TIME_INCREMENT, **{"courant-number": courantNumber})

    timeIncrements = np.array([timeIncrement for timeIncrement, _ in solver.history])
    assert timeIncrements[0] == 0.0
    np.testing.assert_allclose(timeIncrements[1:-1], courantNumber * STABLE_TIME_INCREMENT, rtol=1e-14)


@pytest.mark.parametrize("courantNumber", [0.8, 0.05])
def test_the_motion_starts_from_the_projected_velocity_and_is_the_central_difference_recursion(courantNumber):
    """The masses start at rest at the origin, with the velocity of their state: ``NED`` projects it from their
    momentum, :math:`v_0 = p / m`, and the first increment moves them by :math:`v_0 \\Delta t` (the spring force is zero
    there). From then on the leapfrog is the central difference recursion
    :math:`x_{n+1} = 2 x_n - x_{n-1} - (\\omega \\Delta t)^2 x_n`, to round-off, at any increment below the bound."""

    _, solver = _run(40 * STABLE_TIME_INCREMENT, **{"courant-number": courantNumber})

    timeIncrement = courantNumber * STABLE_TIME_INCREMENT
    displacements = np.array([x for _, x in solver.history])[:-1]  # the last increment ends the step: shorter
    expected = np.zeros_like(displacements)
    expected[1] = timeIncrement * np.array(VELOCITIES)
    for n in range(1, len(expected) - 1):
        expected[n + 1] = 2 * expected[n] - expected[n - 1] - (OMEGA * timeIncrement) ** 2 * expected[n]

    np.testing.assert_allclose(displacements, expected, rtol=1e-12, atol=1e-14 * max(np.abs(VELOCITIES)) / OMEGA)


def test_the_period_is_the_one_of_the_oscillator_to_second_order_in_the_increment():
    """At a small increment, the masses swing with the amplitude :math:`v_0 / \\omega` and the period
    :math:`2 \\pi / \\omega` of the oscillator; the central difference shortens the period by
    :math:`(\\omega \\Delta t)^2 / 24`."""

    courantNumber = 0.05
    _, solver = _run(2 * np.pi / OMEGA, **{"courant-number": courantNumber})

    times = np.cumsum([timeIncrement for timeIncrement, _ in solver.history])
    displacements = np.array([x for _, x in solver.history])
    for i, velocity in enumerate(VELOCITIES):
        x = displacements[:, i] * np.sign(velocity)
        assert x.max() == pytest.approx(abs(velocity) / OMEGA, rel=5e-3)
        # the time at which the mass swings back through the origin, from above: half a period
        crossing = np.flatnonzero((x[:-1] > 0.0) & (x[1:] <= 0.0))[0]
        halfPeriod = times[crossing] + x[crossing] / (x[crossing] - x[crossing + 1]) * (
            times[crossing + 1] - times[crossing]
        )
        assert halfPeriod == pytest.approx(np.pi / OMEGA, rel=(OMEGA * courantNumber * STABLE_TIME_INCREMENT) ** 2 / 10)
