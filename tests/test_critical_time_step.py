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
"""The policies of the critical time step (:mod:`edelweissfe.solvers.base.criticaltimestep`): when the estimate of a
part is taken anew, and that the critical time step at worst, which the policy of the entities keeps between two
estimates, is never above the critical time step."""

from types import SimpleNamespace

import numpy as np
import pytest

from edelweissfe.solvers.base.criticaltimestep import (
    CriticalTimeStepEstimate,
    CriticalTimeStepOfEntitiesWithOwnInertia,
    CriticalTimeStepOfTheElements,
)

DELTA = 0.1


class _Element:
    """An element with a given critical time step."""

    def __init__(self, criticalTimeStep: float):
        self.criticalTimeStep = criticalTimeStep

    def computeCriticalTimeStepForExplicitDynamics(self, U) -> float:
        return self.criticalTimeStep


def _meshOfElements(*criticalTimeSteps, version=0):
    elements = [_Element(value) for value in criticalTimeSteps]
    return SimpleNamespace(
        model=SimpleNamespace(topology=SimpleNamespace(version=version)), elementsWithNodalInertia=elements
    )


def test_the_elements_are_estimated_once_per_mesh():
    system = _meshOfElements(3.0, 2.0, 5.0)
    U = {element: None for element in system.elementsWithNodalInertia}
    criticalTimeStep = CriticalTimeStepOfTheElements()

    assert criticalTimeStep.isDue(system)
    criticalTimeStep.estimate(system, U)
    assert criticalTimeStep.value == 2.0 and criticalTimeStep.numberOfEstimates == 1
    assert not criticalTimeStep.isDue(system), "the same mesh needs no new estimate"

    system.model.topology.version = 1
    assert criticalTimeStep.isDue(system), "a refined mesh does"

    criticalTimeStep.estimate(system, U)
    criticalTimeStep.reset()
    assert criticalTimeStep.isDue(system) and criticalTimeStep.value == np.inf, "and so does every new step"


def test_a_model_without_elements_has_no_critical_time_step_of_elements():
    system = _meshOfElements()
    criticalTimeStep = CriticalTimeStepOfTheElements()
    criticalTimeStep.estimate(system, {})
    assert criticalTimeStep.value == np.inf


def _rowSums(K: np.ndarray, M: np.ndarray) -> np.ndarray:
    """The absolute row sums of the mass-scaled stiffness."""

    s = np.zeros_like(M)
    s[M > 0] = 1 / np.sqrt(M[M > 0])
    return s * (np.abs(K) @ s)


def _gershgorinEstimate(K: np.ndarray, M: np.ndarray) -> float:
    """The critical time step of one second-order field from the largest absolute row sum of the mass-scaled stiffness:
    the estimate of the particles with a single field."""

    return 2 / np.sqrt(np.max(_rowSums(K, M)))


class _Entities:
    """A discrete system of entities with a fixed stiffness whose critical time step is :func:`_gershgorinEstimate`."""

    def __init__(self, K):
        self.K = K
        self.numbering = object()

    def criticalTimeStepOfEntitiesWithOwnInertia(self, M, C, forEachChunk, weightsOfTheEstimate=None):
        M = np.asarray(M)
        return CriticalTimeStepEstimate(_gershgorinEstimate(self.K, M))


def _entities(nDof=6, seed=0):
    rng = np.random.default_rng(seed)
    A = rng.normal(size=(nDof, nDof))
    return _Entities(A @ A.T + nDof * np.eye(nDof)), rng.uniform(1.0, 2.0, nDof)


def test_the_entities_are_estimated_in_the_first_evaluation_of_a_step_and_after_a_new_numbering():
    system, M = _entities()
    C = np.zeros_like(M)
    criticalTimeStep = CriticalTimeStepOfEntitiesWithOwnInertia(DELTA)

    assert criticalTimeStep.isDue(system, 0, M, C, 0.0)
    criticalTimeStep.estimate(system, M, C, None, 0)
    assert criticalTimeStep.value == _gershgorinEstimate(system.K, M)
    assert not criticalTimeStep.isDue(system, 0, M, C, 0.5 * criticalTimeStep.value)

    system.numbering = object()
    assert criticalTimeStep.isDue(system, 1, M, C, 0.5 * criticalTimeStep.value)

    criticalTimeStep.reset()
    assert criticalTimeStep.isDue(system, 0, M, C, 0.0)


def test_with_the_masses_held_the_entities_are_estimated_anew_when_the_assumed_decrease_uses_up_the_margin():
    """With a margin m of the estimate over the critical time step in use, the estimate is trusted for as many
    increments as (1 - delta)^n m >= 1 allows, and is due after the first one beyond."""

    system, M = _entities()
    C = np.zeros_like(M)
    criticalTimeStep = CriticalTimeStepOfEntitiesWithOwnInertia(DELTA)
    criticalTimeStep.estimate(system, M, C, None, 10)
    margin = 1.7
    inUse = criticalTimeStep.value / margin
    trusted = int(np.floor(np.log(margin) / -np.log(1 - DELTA)))

    due = [n for n in range(1, 20) if criticalTimeStep.isDue(system, 10 + n, M, C, inUse)]
    assert due[0] == trusted + 1
    assert criticalTimeStep.criticalTimeStepAtWorst(10 + trusted, M, C) == pytest.approx(
        criticalTimeStep.value * (1 - DELTA) ** trusted
    )


def test_a_fall_of_the_masses_lowers_the_critical_time_step_at_worst_by_its_square_root_and_makes_the_estimate_due():
    system, M = _entities()
    C = np.zeros_like(M)
    criticalTimeStep = CriticalTimeStepOfEntitiesWithOwnInertia(DELTA)
    criticalTimeStep.estimate(system, M, C, None, 0)
    inUse = 0.9 * criticalTimeStep.value

    fallen = M.copy()
    fallen[3] *= 0.5
    assert criticalTimeStep.criticalTimeStepAtWorst(0, fallen, C) == pytest.approx(
        criticalTimeStep.value * np.sqrt(0.5)
    )
    assert not criticalTimeStep.isDue(system, 0, M, C, inUse)
    assert criticalTimeStep.isDue(system, 0, fallen, C, inUse)

    risen = 2.0 * M
    assert (
        criticalTimeStep.criticalTimeStepAtWorst(0, risen, C) == criticalTimeStep.value
    ), "a heavier node earns no credit"


def test_after_any_change_of_the_masses_the_new_estimate_is_at_least_the_critical_time_step_at_worst(seed=1):
    """With the stiffness held, the estimate after any fall of the masses is never below the critical time step at
    worst that the policy keeps from the estimate before: the factor of the masses is exact, no assumption."""

    system, M = _entities(nDof=12, seed=seed)
    C = np.zeros_like(M)
    rng = np.random.default_rng(seed)
    for _ in range(200):
        criticalTimeStep = CriticalTimeStepOfEntitiesWithOwnInertia(DELTA)
        criticalTimeStep.estimate(system, M, C, None, 0)
        changed = M * rng.uniform(0.2, 1.5, M.size)
        atWorst = criticalTimeStep.criticalTimeStepAtWorst(0, changed, C)
        assert atWorst <= _gershgorinEstimate(system.K, changed) * (1 + 1e-12)


def test_a_node_that_gains_a_mass_makes_the_estimate_due_one_that_loses_it_does_not():
    system, M = _entities()
    M[2] = 0.0
    C = np.zeros_like(M)
    criticalTimeStep = CriticalTimeStepOfEntitiesWithOwnInertia(DELTA)
    criticalTimeStep.estimate(system, M, C, None, 0)
    inUse = 0.5 * criticalTimeStep.value

    gained = M.copy()
    gained[2] = 1e-3 * M.max()
    assert criticalTimeStep.isDue(system, 1, gained, C, inUse)

    lost = M.copy()
    lost[4] = 0.0
    assert not criticalTimeStep.isDue(system, 1, lost, C, inUse)


def test_the_damping_of_a_forward_euler_field_enters_linearly():
    system, M = _entities()
    M[:3] = 0.0
    C = np.zeros_like(M)
    C[:3] = 1.0
    criticalTimeStep = CriticalTimeStepOfEntitiesWithOwnInertia(DELTA)
    criticalTimeStep.estimate(system, M, C, None, 0)

    fallen = C.copy()
    fallen[1] = 0.25
    assert criticalTimeStep.criticalTimeStepAtWorst(0, M, fallen) == pytest.approx(0.25 * criticalTimeStep.value)

    gained = C.copy()
    gained[5] = 1.0
    gainedM = M.copy()
    gainedM[5] = 0.0
    assert criticalTimeStep.isDue(system, 0, gainedM, gained, 0.0), "a node turned first order is a new row"


@pytest.mark.parametrize("delta", [0.0, -0.1, 1.5])
def test_the_decrease_per_increment_lies_in_the_open_closed_unit_interval(delta):
    with pytest.raises(ValueError):
        CriticalTimeStepOfEntitiesWithOwnInertia(delta)


def test_a_decrease_of_one_estimates_in_every_increment():
    system, M = _entities()
    C = np.zeros_like(M)
    criticalTimeStep = CriticalTimeStepOfEntitiesWithOwnInertia(1.0)
    criticalTimeStep.estimate(system, M, C, None, 0)
    assert criticalTimeStep.isDue(system, 1, M, C, 1e-3 * criticalTimeStep.value)


class _EntitiesRecordingTheWeights(_Entities):
    """:class:`_Entities` that record the weights every estimate is handed, and refine them by a stand-in step: every
    weight grows by one per estimate."""

    def __init__(self, K):
        super().__init__(K)
        self.numbering = SimpleNamespace(nDof=len(K))
        self.handed = []

    def criticalTimeStepOfEntitiesWithOwnInertia(self, M, C, forEachChunk, weightsOfTheEstimate=None):
        self.handed.append(np.array(weightsOfTheEstimate))
        weightsOfTheEstimate += 1.0
        return super().criticalTimeStepOfEntitiesWithOwnInertia(M, C, forEachChunk)


def test_carried_weights_start_anew_with_a_new_numbering_and_with_every_step():
    """With carried weights, every estimate is handed the weights the last one left, a vector of the numbering; the
    first estimate on a new numbering, and the first one of a step, is handed zeros: the start."""

    system, M = _entities()
    system = _EntitiesRecordingTheWeights(system.K)
    C = np.zeros_like(M)
    criticalTimeStep = CriticalTimeStepOfEntitiesWithOwnInertia(DELTA, carryWeights=True)

    for increment in range(3):
        criticalTimeStep.estimate(system, M, C, None, increment)
    system.numbering = SimpleNamespace(nDof=len(M))
    criticalTimeStep.estimate(system, M, C, None, 3)
    criticalTimeStep.reset()
    criticalTimeStep.estimate(system, M, C, None, 0)

    assert [weights.shape for weights in system.handed] == [(len(M),)] * 5
    assert [float(np.max(weights)) for weights in system.handed] == [0.0, 1.0, 2.0, 0.0, 0.0]
    assert criticalTimeStep.value == _gershgorinEstimate(system.K, M)


def test_carried_weights_are_kept_when_they_beat_the_fresh_estimate_after_stiffening():
    """When the system stiffens so that the estimate with carried weights falls below the estimate they started from,
    the policy checks a fresh estimate from zero weights. If the carried weights still give a larger (tighter) estimate
    than the fresh one, they are kept rather than overwritten by the fresh weights; if the fresh estimate is larger,
    the fresh weights take over."""

    class _EntitiesWithWeightedEstimate:
        """Uses the carried weights as a Collatz-Wielandt vector on the mass-scaled stiffness and advances them by one
        power-iteration step."""

        def __init__(self, K):
            self.K = K
            self.numbering = SimpleNamespace(nDof=len(K))
            self.handed = []

        def criticalTimeStepOfEntitiesWithOwnInertia(self, M, C, forEachChunk, weightsOfTheEstimate=None):
            self.handed.append(np.array(weightsOfTheEstimate))
            M = np.asarray(M)
            s = 1.0 / np.sqrt(M)
            A = np.abs(s[:, None] * self.K * s[None, :])
            w = np.ones(len(M)) if not weightsOfTheEstimate.any() else weightsOfTheEstimate.copy()
            Aw = A @ w
            bound = float(np.max(Aw / w))
            weightsOfTheEstimate[:] = Aw / np.linalg.norm(Aw)
            return CriticalTimeStepEstimate(2.0 / np.sqrt(bound))

    rng = np.random.default_rng(7)
    A0 = rng.normal(size=(8, 8))
    K0 = A0 @ A0.T + np.diag([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 50.0])
    M = np.ones(8)
    C = np.zeros(8)
    system = _EntitiesWithWeightedEstimate(K0)
    criticalTimeStep = CriticalTimeStepOfEntitiesWithOwnInertia(DELTA, carryWeights=True)

    # Refine the weights over several increments on K0 so they converge well past the constant-weight bound.
    for increment in range(6):
        criticalTimeStep.estimate(system, M, C, None, increment)
    refinedValue = criticalTimeStep.value

    # Stiffen the system uniformly by 1.5x: the carried weights are still near the Perron vector, so they still beat
    # a fresh constant-weight estimate, even though the new estimate falls below the start value on K0.
    system.K = 1.5 * K0
    freshAtStiff = _gershgorinEstimate(system.K, M)
    criticalTimeStep.estimate(system, M, C, None, 6)
    keptValue = criticalTimeStep.value
    # Fresh estimate was tried (zeros) after the carried one fell below the start value, and lost to the carried one.
    assert not system.handed[-1].any()
    assert keptValue >= refinedValue / np.sqrt(1.5)
    assert keptValue > freshAtStiff

    # The next estimate must continue from the kept refined weights, not from the discarded fresh ones: the estimate
    # stays at or above keptValue (a discarded reset would drop back towards the 1-step constant-weight estimate).
    nHandedBefore = len(system.handed)
    criticalTimeStep.estimate(system, M, C, None, 7)
    assert len(system.handed) == nHandedBefore + 1, "no second reset needed once the baseline updated"
    assert criticalTimeStep.value >= keptValue * (1.0 - 1e-12)

    # Now rotate K so the old carried weights point at the wrong mode and lose to a fresh constant-weight estimate:
    # the fresh estimate must win, and its 1-step weights must take over for the next estimate.
    perm = np.roll(np.arange(8), 4)
    system.K = 1.5 * K0[np.ix_(perm, perm)]
    expectedFresh = _gershgorinEstimate(system.K, M)
    A_rot = np.abs(system.K)
    expectedFreshWeights = (A_rot @ np.ones(8)) / np.linalg.norm(A_rot @ np.ones(8))
    criticalTimeStep.estimate(system, M, C, None, 8)
    assert not system.handed[-1].any()
    assert criticalTimeStep.value == pytest.approx(expectedFresh, rel=1e-14)
    criticalTimeStep.estimate(system, M, C, None, 9)
    np.testing.assert_allclose(system.handed[-1], expectedFreshWeights, rtol=1e-14)
    assert criticalTimeStep.value > expectedFresh
