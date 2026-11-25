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
and, when its estimate is due, its critical time step. The node holds the coefficients of one increment only. The RKPM
particles of EdelweissMeshfree are such entities; :class:`MassOnASpring` and :class:`SystemOfMassesOnSprings` are the
same contract in a few lines of Python, and the tests check ``NED``'s path for it in EdelweissFE itself:

* the calls of an increment come in the order of the module documentation of ``NED``,
* the motion starts from the velocity projected from the momentum, and the leapfrog then is the central difference
  recursion of the oscillator, to round-off; its period is the analytic one, to second order in the increment,
* the critical time step of the entities sets the increment of a step that asks for a larger one, and, since it sets
  the increment, it is estimated in every increment,
* after a brief stiff event, the increment comes back with ``follow-critical-time-step`` and stays low without it, and
  the increment that follows the critical time step is at most the step's ``maxInc``, whatever its ``startInc``,
* a critical time step that is not a number is refused, whether the increment follows it or not, and without any
  bound the step keeps its own increment, also when it follows the critical time step,
* with ``carry-critical-time-step-weights``, the weights of the estimate are handed from one estimate to the next, zero
  at the first estimate of a step; without it, the estimate gets none; carried weights that give an estimate below the
  one they started from start anew, and the increment is not lowered by them,
* a field of the entities that gets neither an inertia nor a damping is refused, and so are a damping of the entities
  that is negative or not finite, and an inertia that is not finite,
* the rate of a first-order field of the entities (dashpots on springs) is published in every increment, also when the
  velocity is projected from the momentum in every increment,
* the step gives the entities zero Newmark parameters: their inertia is integrated at the nodes,
* a multi-point constraint whose slave is a degree of freedom of the entities is refused.
"""

from types import SimpleNamespace

import numpy as np
import pytest

from edelweissfe.config.configurator import loadConfiguration
from edelweissfe.constraints.base.multipointconstraintbase import (
    MultiPointConstraintBase,
)
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
#: The critical time step of one mass on its spring, 2 / omega.
CRITICAL_TIME_STEP = 2.0 / OMEGA
#: The coefficient of the dashpots of :class:`ModelOfDashpotsOnSprings`, and where each starts. The forward Euler update
#: of a dashpot on a spring has the critical time step 2 c / k.
DAMPING = 20.0
DISPLACEMENTS = (0.02, -0.05)
#: The brief stiff event of :class:`ModelOfMassesOnSpringsWithAStiffEvent`: when the springs are stiffer, and by how
#: much. Its ends lie between the start times of the increments taken at the courant number 0.5.
STIFF_EVENT_START, STIFF_EVENT_END = 10.3 * CRITICAL_TIME_STEP, 12.25 * CRITICAL_TIME_STEP
STIFF_EVENT_FACTOR = 25.0


class MassOnASpring(PointMass):
    """A point mass on a spring to the ground that carries its displacement and its velocity in itself, as a particle
    carries its state. As a point mass, it sits on one node and lumps its mass there."""

    def __init__(self, number: int, node: Node, model: FEModel, velocity: float, mass: float, damping: float):
        super().__init__(number, [node], model, mass)
        #: The coefficient of a dashpot to the ground, lumped to the node as the mass is.
        self.damping = damping
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
        stiffness = self.model.springStiffness()
        return self._onTheNodes(lambda mass: stiffness * mass.displacement), self._onTheNodes(
            lambda mass: mass.mass * mass.velocity
        )

    def assembleLumpedMassOfEntitiesWithOwnInertia(self, forEachChunk):
        self.model.calls.append("mass")
        return self._onTheNodes(lambda mass: np.full(mass.nDof, mass.mass))

    def assembleLumpedDampingOfEntitiesWithOwnInertia(self, forEachChunk):
        self.model.calls.append("damping")
        return self._onTheNodes(lambda mass: np.full(mass.nDof, mass.damping))

    def criticalTimeStepOfEntitiesWithOwnInertia(self, lumpedMass, lumpedDamping, forEachChunk, weightsOfTheBound=None):
        self.model.calls.append("estimate")
        # the weights it was handed, and one step of a stand-in refinement: every weight grows by one per estimate
        self.model.weightsOfTheBounds.append(None if weightsOfTheBound is None else np.array(weightsOfTheBound))
        if weightsOfTheBound is not None:
            weightsOfTheBound += 1.0
        stiffness = self.model.springStiffness()

        # each entity on its spring alone, from the mass and the damping the solver has just assembled: omega^2 = k / m
        # for a mass; for a dashpot without a mass, (k / c)^2, so that 2 / omega is its critical time step 2 c / k
        def squaredFrequencyOf(entity) -> float:
            mass, damping = lumpedMass[entity][0], lumpedDamping[entity][0]
            return stiffness / mass if mass > 0.0 else (stiffness / damping) ** 2

        entities = self.entitiesWithOwnInertia
        squaredFrequencies = np.array([squaredFrequencyOf(entity) for entity in entities])
        return SimpleNamespace(value=2.0 / np.sqrt(squaredFrequencies.max()))

    def _onTheNodes(self, valueOf) -> DofVector:
        vector = self.locations.constructDofVector()
        vector[:] = 0.0
        for mass in self.entitiesWithOwnInertia:
            vector[mass] = valueOf(mass)
        return vector


class ModelOfMassesOnSprings(FEModel):
    """A model of masses on springs, one per node, that answers with :class:`SystemOfMassesOnSprings`, and records
    what ``NED`` asks of it.

    Parameters
    ----------
    mass
        The mass of every point mass.
    damping
        The coefficient of the dashpot of every point mass to the ground; none by default.
    """

    def __init__(self, mass: float = MASS, damping: float = 0.0):
        super().__init__(2)
        self.calls = []
        #: The weights every estimate was handed (None without ``carry-critical-time-step-weights``).
        self.weightsOfTheBounds = []
        #: The Newmark parameters the solver gave last; those an implicit dynamic step would have left.
        self.newmarkParametersOfEntities = (0.25, 0.5)
        for number, velocity in enumerate(VELOCITIES, start=1):
            node = Node(number, np.array([float(number), 0.0]))
            self.nodes[number] = node
            self.elements[number] = MassOnASpring(number, node, self, velocity, mass, damping)
        self.elementSets["all"] = ElementSet("all", list(self.elements.values()))
        self.nodeSets["all"] = NodeSet("all", list(self.nodes.values()))

    def buildDiscreteSystem(self) -> SystemOfMassesOnSprings:
        return SystemOfMassesOnSprings(self, super().buildDiscreteSystem().locations)

    def springStiffness(self) -> float:
        """The stiffness of every spring now."""

        return STIFFNESS

    def setNewmarkParametersOfEntities(self, beta: float, gamma: float):
        self.newmarkParametersOfEntities = (beta, gamma)

    def advanceToTime(self, time: float):
        self.calls.append("accept")
        return super().advanceToTime(time)

    def updateConnectivity(self):
        self.calls.append("reconnect")
        return super().updateConnectivity()


class ModelOfDashpotsOnSprings(ModelOfMassesOnSprings):
    """Dashpots on springs: without a mass, the displacement is a first-order field, :math:`c \\dot{x} + k x = 0`, which
    ``NED`` integrates by forward Euler. Each dashpot starts displaced, and creeps back."""

    def __init__(self):
        super().__init__(mass=0.0, damping=DAMPING)
        for dashpot, displacement in zip(self.elements.values(), DISPLACEMENTS):
            dashpot.displacement[0] = displacement


class ModelOfMassesOnSpringsWithAStiffEvent(ModelOfMassesOnSprings):
    """Masses on springs that are stiffer for a while, from :data:`STIFF_EVENT_START` to :data:`STIFF_EVENT_END`, as a
    particle that is briefly compressed: the critical time step of the entities falls by the square root of
    :data:`STIFF_EVENT_FACTOR` meanwhile, and comes back after the event."""

    def springStiffness(self) -> float:
        stiffer = STIFF_EVENT_START <= self.time < STIFF_EVENT_END
        return STIFFNESS * (STIFF_EVENT_FACTOR if stiffer else 1.0)


class SystemOfMassesOnSpringsWithoutABound(SystemOfMassesOnSprings):
    """Masses on springs whose critical time step is not a number, as from a stiffness or a mass that is not
    finite."""

    def criticalTimeStepOfEntitiesWithOwnInertia(self, lumpedMass, lumpedDamping, forEachChunk, weightsOfTheBound=None):
        super().criticalTimeStepOfEntitiesWithOwnInertia(lumpedMass, lumpedDamping, forEachChunk, weightsOfTheBound)
        return SimpleNamespace(value=np.nan)


class ModelOfMassesOnSpringsWithoutABound(ModelOfMassesOnSprings):
    """Masses on springs whose critical time step is not a number."""

    def buildDiscreteSystem(self) -> SystemOfMassesOnSpringsWithoutABound:
        return SystemOfMassesOnSpringsWithoutABound(self, super().buildDiscreteSystem().locations)


class SystemOfMassesOnSpringsReportingNoCriticalTimeStep(SystemOfMassesOnSprings):
    """Masses on springs that report no critical time step (an infinite one), as entities whose stiffness gives none."""

    def criticalTimeStepOfEntitiesWithOwnInertia(self, lumpedMass, lumpedDamping, forEachChunk, weightsOfTheBound=None):
        super().criticalTimeStepOfEntitiesWithOwnInertia(lumpedMass, lumpedDamping, forEachChunk, weightsOfTheBound)
        return SimpleNamespace(value=np.inf)


class ModelOfMassesOnSpringsReportingNoCriticalTimeStep(ModelOfMassesOnSprings):
    """Masses on springs that report no critical time step."""

    def buildDiscreteSystem(self) -> SystemOfMassesOnSpringsReportingNoCriticalTimeStep:
        return SystemOfMassesOnSpringsReportingNoCriticalTimeStep(self, super().buildDiscreteSystem().locations)


class SystemWhoseCarriedWeightsFallBehind(SystemOfMassesOnSprings):
    """The masses on springs with a stand-in of carried weights that fall behind the stiffness: the estimate the weights
    start from (weights zero) is that of the springs, the next one (weights one) is larger, the one after it (weights
    two) below the start."""

    def criticalTimeStepOfEntitiesWithOwnInertia(self, lumpedMass, lumpedDamping, forEachChunk, weightsOfTheBound=None):
        factor = {0.0: 1.0, 1.0: 1.5, 2.0: 0.5}[float(np.max(weightsOfTheBound))]
        bound = super().criticalTimeStepOfEntitiesWithOwnInertia(
            lumpedMass, lumpedDamping, forEachChunk, weightsOfTheBound
        )
        return SimpleNamespace(value=factor * bound.value)


class ModelWhoseCarriedWeightsFallBehind(ModelOfMassesOnSprings):
    """The masses on springs, answering with :class:`SystemWhoseCarriedWeightsFallBehind`."""

    def buildDiscreteSystem(self) -> SystemWhoseCarriedWeightsFallBehind:
        return SystemWhoseCarriedWeightsFallBehind(self, super().buildDiscreteSystem().locations)


class TheSecondMassFollowsTheFirst(MultiPointConstraintBase):
    """A multi-point constraint that makes the displacement of the node of the second mass a slave of that of the
    first, :math:`u_2 = u_1`."""

    def __init__(self, name: str, model: FEModel):
        self.name = name
        self._nodes = (model.nodes[1], model.nodes[2])

    def getMultiPointConstraints(self, numbering) -> list[tuple[int, list[tuple[int, float]]]]:
        master, slave = (numbering.idcsOfFieldVariablesInDofVector[node.fields["displacement"]] for node in self._nodes)
        return [(slaveDof, [(masterDof, 1.0)]) for slaveDof, masterDof in zip(slave, master)]


class _RecordingNED(NED):
    """``NED``, recording the increment and the displacement of every mass after each increment it accepts, and the
    velocity the model holds at the nodes then (the entry ``V`` of the node field)."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.history = []
        self.publishedVelocities = []

    def acceptIncrement(self, step, model, timeStep):
        super().acceptIncrement(step, model, timeStep)
        self.history.append((timeStep.timeIncrement, [mass.displacement[0] for mass in model.elements.values()]))
        self.publishedVelocities.append(np.array(model.nodeFields["displacement"]["V"])[:, 0])


def _run(
    stepLength: float, model: ModelOfMassesOnSprings = None, stepOptions: dict = None, **options
) -> tuple[ModelOfMassesOnSprings, _RecordingNED]:
    """Run one explicit step of the given length, which asks for the whole step as its increment unless the options of
    the step say otherwise, on the given model (by default the masses on springs), with the options of the solver."""

    journal = Journal(verbose=False)
    model = model or ModelOfMassesOnSprings()
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
        **(stepOptions or {}),
    ).solve()
    fieldOutputController.finalizeJob()

    return model, solver


def test_the_calls_of_an_increment_come_in_the_order_of_the_module_documentation():
    """The step starts with a reconnection (the step-start topology update). Then, in every increment: advance the
    entities on the connectivity of the start of the increment; accept the state and reconnect; then, on the new
    connectivity, evaluate the force and the momentum, the mass, the damping, and the critical time step when its
    estimate is due: here in every increment, since the entities set the time increment, so that the margin of an
    estimate is used up after one increment
    (:class:`~edelweissfe.solvers.base.criticaltimestep.CriticalTimeStepOfEntitiesWithOwnInertia`)."""

    model, solver = _run(10 * CRITICAL_TIME_STEP)

    oneIncrement = ["advance", "accept", "reconnect", "force and momentum", "mass", "damping", "estimate"]
    expected = ["reconnect"]
    for _ in range(len(solver.history)):
        expected += oneIncrement
    assert model.calls[: len(expected)] == expected


def test_the_critical_time_step_of_the_entities_sets_the_increment_of_a_step_that_asks_for_more():
    """The step asks for its whole length; the increment taken is the critical time step of the entities times the
    courant number, from the first increment on (the last one ends the step)."""

    courantNumber = 0.5
    _, solver = _run(10 * CRITICAL_TIME_STEP, **{"courant-number": courantNumber})

    timeIncrements = np.array([timeIncrement for timeIncrement, _ in solver.history])
    assert timeIncrements[0] == 0.0
    np.testing.assert_allclose(timeIncrements[1:-1], courantNumber * CRITICAL_TIME_STEP, rtol=1e-14)


@pytest.mark.parametrize("follow", [False, True])
def test_after_a_brief_stiff_event_the_increment_comes_back_only_if_it_follows_the_critical_time_step(follow):
    """The springs are stiffer for a while, and the critical time step of the entities, estimated in every increment
    here, falls by a factor of five meanwhile. The increment that starts in the event is the critical time step of the
    stiff springs times the courant number. After the event, it is lowered only, and stays at the one of the event, by
    default; with ``follow-critical-time-step``, it comes back to the one before the event with the first estimate
    taken after it."""

    courantNumber = 0.5
    _, solver = _run(
        30 * CRITICAL_TIME_STEP,
        model=ModelOfMassesOnSpringsWithAStiffEvent(),
        **{
            "courant-number": courantNumber,
            "critical-time-step-decrease-per-increment": 1.0,
            "follow-critical-time-step": follow,
        },
    )

    timeIncrements = np.array([timeIncrement for timeIncrement, _ in solver.history])[1:-1]  # without the zero
    # increment, and the last one, which ends the step
    startTimes = np.cumsum(timeIncrements) - timeIncrements
    beforeTheEvent = startTimes < STIFF_EVENT_START
    inTheEvent = (startTimes >= STIFF_EVENT_START) & (startTimes < STIFF_EVENT_END)
    afterTheEvent = startTimes >= STIFF_EVENT_END
    assert beforeTheEvent.any() and inTheEvent.any() and afterTheEvent.any()

    soft = courantNumber * CRITICAL_TIME_STEP
    stiff = soft / np.sqrt(STIFF_EVENT_FACTOR)
    np.testing.assert_allclose(timeIncrements[beforeTheEvent], soft, rtol=1e-14)
    np.testing.assert_allclose(timeIncrements[inTheEvent], stiff, rtol=1e-14)
    np.testing.assert_allclose(timeIncrements[afterTheEvent], soft if follow else stiff, rtol=1e-14)


@pytest.mark.parametrize("follow", [False, True])
def test_a_critical_time_step_that_is_not_a_number_is_refused(follow):
    """A bound of the entities that is not a number fails every comparison: lowered only, it would be ignored silently;
    followed, it would become the time increment. Either way, ``NED`` refuses it when it takes the critical time
    step."""

    with pytest.raises(ValueError, match="the critical time step is not a number"):
        _run(
            10 * CRITICAL_TIME_STEP,
            model=ModelOfMassesOnSpringsWithoutABound(),
            **{"follow-critical-time-step": follow},
        )


@pytest.mark.parametrize(
    "follow, maxInc, expectedTimeIncrement",
    [(False, 0.05, 0.1), (True, 0.05, 0.5), (False, 1.0, 0.1), (True, 1.0, 0.8)],
)
def test_the_increment_that_follows_the_critical_time_step_is_at_most_the_steps_max_inc(
    follow, maxInc, expectedTimeIncrement
):
    """The step asks for a tenth of the critical time step to start with (``startInc``), and for at most half of it
    or ten times it (``maxInc``); the courant number is 0.8. Lowered only, the increment stays the step's own, the
    smaller; following the critical time step, it is the courant number times the critical time step, at most
    ``maxInc``: the small ``startInc`` is raised to it when ``maxInc`` allows it."""

    stepLength = 10 * CRITICAL_TIME_STEP
    _, solver = _run(
        stepLength,
        stepOptions={"startInc": 0.01, "maxInc": maxInc},
        **{"courant-number": 0.8, "follow-critical-time-step": follow},
    )

    timeIncrements = np.array([timeIncrement for timeIncrement, _ in solver.history])[1:-1]
    np.testing.assert_allclose(timeIncrements, expectedTimeIncrement * CRITICAL_TIME_STEP, rtol=1e-12)


@pytest.mark.parametrize("follow", [False, True])
def test_without_a_bound_the_step_keeps_its_own_increment(follow):
    """Neither the masses nor anything else reports a critical time step: it stays infinite, there is nothing to
    follow, and the step keeps its own increment (``startInc``), not its ``maxInc``."""

    stepLength = 10 * CRITICAL_TIME_STEP
    _, solver = _run(
        stepLength,
        model=ModelOfMassesOnSpringsReportingNoCriticalTimeStep(),
        stepOptions={"startInc": 0.01, "maxInc": 0.05},
        **{"follow-critical-time-step": follow},
    )

    timeIncrements = np.array([timeIncrement for timeIncrement, _ in solver.history])[1:-1]
    np.testing.assert_allclose(timeIncrements, 0.1 * CRITICAL_TIME_STEP, rtol=1e-12)


#: The options of a solver that estimates the entities in every increment.
IN_EVERY_INCREMENT = {"critical-time-step-decrease-per-increment": 1.0}


def test_without_carried_weights_the_estimate_gets_none():
    model, _ = _run(10 * CRITICAL_TIME_STEP, **IN_EVERY_INCREMENT)

    assert model.weightsOfTheBounds and all(weights is None for weights in model.weightsOfTheBounds)


def test_carried_weights_are_handed_from_estimate_to_estimate_starting_at_zero():
    """With ``carry-critical-time-step-weights``, every estimate gets the weights the last one left, a vector of the
    numbering; the first estimate of the step gets zeros: no weights yet."""

    model, _ = _run(10 * CRITICAL_TIME_STEP, **IN_EVERY_INCREMENT, **{"carry-critical-time-step-weights": True})

    assert len(model.weightsOfTheBounds) > 2
    for number, weights in enumerate(model.weightsOfTheBounds):
        np.testing.assert_array_equal(weights, np.full(len(weights), float(number)))


def test_carried_weights_that_give_an_estimate_below_their_start_start_anew():
    """The third estimate of the weights (weights two) is below the estimate they started from: ``NED`` starts them anew
    and estimates again from zero weights, so the weights handed cycle through zero, one, two, and the increment is
    never lowered below the start's."""

    courantNumber = 0.5
    model, solver = _run(
        10 * CRITICAL_TIME_STEP,
        ModelWhoseCarriedWeightsFallBehind(),
        **IN_EVERY_INCREMENT,
        **{"carry-critical-time-step-weights": True, "courant-number": courantNumber},
    )

    handed = [float(np.max(weights)) for weights in model.weightsOfTheBounds]
    assert len(handed) > 6
    assert handed == [float(number % 3) for number in range(len(handed))]
    timeIncrements = np.array([timeIncrement for timeIncrement, _ in solver.history])
    np.testing.assert_allclose(timeIncrements[1:-1], courantNumber * CRITICAL_TIME_STEP, rtol=1e-14)


@pytest.mark.parametrize("courantNumber", [0.8, 0.05])
def test_the_motion_starts_from_the_projected_velocity_and_is_the_central_difference_recursion(courantNumber):
    """The masses start at rest at the origin, with the velocity of their state: ``NED`` projects it from their
    momentum, :math:`v_0 = p / m`, and the first increment moves them by :math:`v_0 \\Delta t` (the spring force is zero
    there). From then on the leapfrog is the central difference recursion
    :math:`x_{n+1} = 2 x_n - x_{n-1} - (\\omega \\Delta t)^2 x_n`, to round-off, at any increment below the
    critical time step."""

    _, solver = _run(40 * CRITICAL_TIME_STEP, **{"courant-number": courantNumber})

    timeIncrement = courantNumber * CRITICAL_TIME_STEP
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
        assert halfPeriod == pytest.approx(np.pi / OMEGA, rel=(OMEGA * courantNumber * CRITICAL_TIME_STEP) ** 2 / 10)


def test_a_field_of_the_entities_without_inertia_and_damping_is_refused():
    """Point masses without a mass (as particles without a density) give the displacement neither an inertia nor a
    damping at any degree of freedom: there is nothing to integrate it with, and ``NED`` says so when it takes the
    operators of the entities, instead of leaving the field where it is for the whole step."""

    with pytest.raises(ValueError, match="displacement carries neither an inertia nor a damping"):
        _run(10 * CRITICAL_TIME_STEP, model=ModelOfMassesOnSprings(mass=0.0))


@pytest.mark.parametrize(
    "mass, damping, refusal",
    [
        (MASS, -1.0, "lumped damping is not a valid dissipation"),
        (MASS, np.nan, "lumped damping is not a valid dissipation"),
        (np.nan, 0.0, "lumped inertia is not finite"),
    ],
)
def test_an_invalid_damping_or_inertia_of_the_entities_is_refused(mass, damping, refusal):
    """The damping of the entities is checked as that of the elements is: a negative one would amplify a second-order
    field through the rate C/M, which no check of the divisor sees, and a coefficient that is not a number passes every
    comparison and would reach every increment."""

    with pytest.raises(ValueError, match=refusal):
        _run(10 * CRITICAL_TIME_STEP, model=ModelOfMassesOnSprings(mass=mass, damping=damping))


@pytest.mark.parametrize("projectVelocity", [False, True])
def test_the_rate_of_a_first_order_field_of_the_entities_is_published_in_every_increment(projectVelocity):
    """Dashpots on springs creep back by forward Euler, :math:`x_{n+1} = x_n (1 - k \\Delta t / c)`, and the velocity
    the model holds after an increment is the rate it was taken with. ``project-velocity-every-increment`` projects the
    velocity of the second-order fields from the momentum; a first-order field has no momentum, and keeps its rate."""

    _, solver = _run(
        5 * CRITICAL_TIME_STEP,
        model=ModelOfDashpotsOnSprings(),
        **{"project-velocity-every-increment": projectVelocity},
    )

    timeIncrements = np.array([timeIncrement for timeIncrement, _ in solver.history])
    displacements = np.array([x for _, x in solver.history])
    np.testing.assert_allclose(displacements[0], DISPLACEMENTS)
    np.testing.assert_allclose(
        displacements[1:],
        displacements[:-1] * (1.0 - STIFFNESS * timeIncrements[1:, np.newaxis] / DAMPING),
        rtol=1e-12,
    )

    rates = (displacements[1:] - displacements[:-1]) / timeIncrements[1:, np.newaxis]
    assert np.all(rates != 0.0)
    np.testing.assert_allclose(np.array(solver.publishedVelocities)[1:], rates, rtol=1e-12)


def test_the_step_gives_the_entities_zero_newmark_parameters():
    """An entity that can integrate its own inertia by the Newmark scheme keeps the parameters of the solver that gave
    them last, e.g. of an implicit dynamic step. ``NED`` integrates the inertia at the nodes, and gives zero at the
    start of its step: a tangent such an entity is asked for afterwards holds no inertia of a scheme that is not
    running."""

    model, _ = _run(CRITICAL_TIME_STEP)

    assert model.newmarkParametersOfEntities == (0.0, 0.0)


def test_a_multi_point_constraint_with_a_slave_among_the_degrees_of_freedom_of_the_entities_is_refused():
    """A slave of a multi-point constraint has no equation of motion of its own: the inertia the entities lump to it
    would be folded onto its masters, and the velocity and the kinetic energy published for it would be zero. ``NED``
    refuses it when it builds the numbering, and names the constraint."""

    model = ModelOfMassesOnSprings()
    model.multiPointConstraints["follow"] = TheSecondMassFollowsTheFirst("follow", model)

    with pytest.raises(
        NotImplementedError,
        match=r"multi-point constraint\(s\) 'follow' make 2 degree\(s\) of freedom of the entities that carry their own "
        r"inertia",
    ):
        _run(10 * CRITICAL_TIME_STEP, model=model)
