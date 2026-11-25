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

from edelweissfe.numerics.mpctransformation import MultiPointConstraintTransformation
from edelweissfe.solvers.base.criticaltimestep import (
    CriticalTimeStepEstimateFromTheBoundOfEachDof,
    CriticalTimeStepOfEntitiesWithOwnInertia,
    CriticalTimeStepOfTheElements,
    CriticalTimeStepWithThePenaltyConstraints,
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
        return CriticalTimeStepEstimateFromTheBoundOfEachDof(
            _gershgorinEstimate(self.K, M), np.arange(len(M)), _rowSums(self.K, M)
        )


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


class _ElementWithItsEigenvalue:
    """An element with a stiffness on some degrees of freedom and its own lumped mass there, whose critical time step is
    the exact one of the element alone, :math:`2/\\sqrt{\\lambda_\\mathrm{max}(\\boldsymbol{M}_e^{-1}\\boldsymbol{K}_e)}`:
    the element eigenvalue inequality."""

    def __init__(self, dofs, K, M):
        self.dofs, self.K, self.M = dofs, K, M
        s = 1 / np.sqrt(M)
        largest = np.max(np.linalg.eigvalsh(s[:, None] * K * s[None, :]))
        self.criticalTimeStep = 2 / np.sqrt(largest) if largest > 0 else np.inf

    def computeCriticalTimeStepForExplicitDynamics(self, U) -> float:
        return self.criticalTimeStep


def _randomModelWithPenaltySprings(rng, nDof=12, nElements=5, nSprings=4, stiffnessOfTheSprings=1.0):
    """Elements with random symmetric positive semi-definite stiffnesses and positive lumped masses on overlapping sets of
    degrees of freedom, and penalty springs :math:`k\\, w w^T` between random degrees of freedom: the assembled stiffness,
    the assembled mass, the elements, and the stiffness of the springs."""

    K, M = np.zeros((nDof, nDof)), np.zeros(nDof)
    elements = []
    for _ in range(nElements):
        dofs = rng.choice(nDof, size=4, replace=False)
        A = rng.normal(size=(4, 4))
        Ke, Me = A @ A.T, rng.uniform(0.5, 2.0, 4)
        K[np.ix_(dofs, dofs)] += Ke
        M[dofs] += Me
        elements.append(_ElementWithItsEigenvalue(dofs, Ke, Me))
    for dof in np.flatnonzero(M == 0.0):  # every degree of freedom has a mass: a point mass, no stiffness
        elements.append(_ElementWithItsEigenvalue(np.array([dof]), np.zeros((1, 1)), np.ones(1)))
        M[dof] = 1.0
    Kc = np.zeros((nDof, nDof))
    for _ in range(nSprings):
        w = np.zeros(nDof)
        dofs = rng.choice(nDof, size=3, replace=False)
        w[dofs] = rng.normal(size=3)
        Kc += stiffnessOfTheSprings * rng.uniform(0.5, 2.0) * np.outer(w, w)
    return K, M, elements, Kc


def _systemOf(elements, nDof):
    return SimpleNamespace(
        model=SimpleNamespace(topology=SimpleNamespace(version=0)),
        elementsWithNodalInertia=elements,
        numbering=SimpleNamespace(nDof=nDof),
        locations={element: element.dofs for element in elements},
    )


@pytest.mark.parametrize("stiffnessOfTheSprings", [0.01, 1.0, 100.0])
def test_the_penalty_springs_added_where_they_act_bound_the_exact_eigenvalue(stiffnessOfTheSprings, seed=3):
    """For elements and penalty springs, the critical time step of the degrees of freedom the springs couple -- the
    largest squared frequency of the elements at a degree of freedom plus the row sum of the springs' mass-scaled
    stiffness there -- is below the exact critical time step of the assembled system: the argument of the theory page,
    which lends the springs a share of the mass, checked on random systems, from soft springs that do not limit to
    stiff ones that do."""

    rng = np.random.default_rng(seed)
    elementsOnly, limiting = 0, 0
    for _ in range(300):
        K, M, elements, Kc = _randomModelWithPenaltySprings(rng, stiffnessOfTheSprings=stiffnessOfTheSprings)
        nDof = len(M)
        system = _systemOf(elements, nDof)
        s = 1 / np.sqrt(M)
        exact = 2 / np.sqrt(np.max(np.linalg.eigvalsh(s[:, None] * (K + Kc) * s[None, :])))

        ofTheElements = CriticalTimeStepOfTheElements()
        ofTheElements.estimate(system, {element: None for element in elements})
        ofTheEntities = CriticalTimeStepOfEntitiesWithOwnInertia(DELTA)
        ofTheConstraints = CriticalTimeStepWithThePenaltyConstraints()
        ofTheConstraints.estimate(system, s * (np.abs(Kc) @ s), ofTheElements, ofTheEntities)

        estimate = min(ofTheElements.value, ofTheConstraints.value)
        assert estimate <= exact * (1 + 1e-12)
        elementsOnly += ofTheElements.value <= exact * (1 + 1e-12)
        limiting += ofTheConstraints.value < ofTheElements.value
    assert elementsOnly < 300, "the springs raise the frequencies: the elements' estimate alone does not always hold"
    assert limiting > 0


def test_a_constraint_at_degrees_of_freedom_of_slow_elements_does_not_lower_the_critical_time_step():
    """The springs add to the frequencies of the degrees of freedom they couple, not to the largest frequency of the
    model: a spring between the nodes of soft elements leaves the critical time step of a stiff element elsewhere, and
    the estimate is the elements' own, bit for bit."""

    stiff = _ElementWithItsEigenvalue(np.array([0, 1]), np.array([[100.0, -100.0], [-100.0, 100.0]]), np.ones(2))
    soft = _ElementWithItsEigenvalue(np.array([2, 3]), np.array([[1.0, -1.0], [-1.0, 1.0]]), np.ones(2))
    system = _systemOf([stiff, soft], 4)
    ofTheElements = CriticalTimeStepOfTheElements()
    ofTheElements.estimate(system, {stiff: None, soft: None})

    rowSums = np.array([0.0, 0.0, 10.0, 10.0])
    ofTheConstraints = CriticalTimeStepWithThePenaltyConstraints()
    ofTheConstraints.estimate(system, rowSums, ofTheElements, CriticalTimeStepOfEntitiesWithOwnInertia(DELTA))
    assert ofTheConstraints.value == ofTheElements.value == stiff.criticalTimeStep

    rowSums[1] = 10.0
    ofTheConstraints.estimate(system, rowSums, ofTheElements, CriticalTimeStepOfEntitiesWithOwnInertia(DELTA))
    assert ofTheConstraints.value == pytest.approx(2 / np.sqrt((2 / stiff.criticalTimeStep) ** 2 + 10.0), rel=1e-14)


def test_the_entities_with_the_penalty_constraints_are_estimated_anew_when_their_combined_margin_may_be_used_up():
    """Where a penalty constraint couples the entities' degrees of freedom, the margin of their last estimate is the one
    with the constraint's row sums of now added: with the rows of a stiff constraint the estimate is due at once, with
    no rows as long as their own margin lasts."""

    system, M = _entities()
    C = np.zeros_like(M)
    criticalTimeStep = CriticalTimeStepOfEntitiesWithOwnInertia(DELTA)
    criticalTimeStep.estimate(system, M, C, None, 0)
    inUse = 0.5 * criticalTimeStep.value

    noRows = np.zeros_like(M)
    assert not criticalTimeStep.isDue(system, 1, M, C, inUse, noRows)

    stiffRows = noRows.copy()
    stiffRows[2] = 100 * (2 / inUse) ** 2
    assert criticalTimeStep.withThePenaltyConstraints(stiffRows) < inUse
    assert criticalTimeStep.isDue(system, 1, M, C, inUse, stiffRows)


def _rowSumsWithMultiPointConstraints(Kc: np.ndarray, M: np.ndarray, transformation) -> np.ndarray:
    """The row sums of the penalty springs as ``NED`` forms them with multi-point constraints: the column weights
    :math:`1/\\sqrt{M}` of the folded masses, a slave weighted by its masters, and the rows folded onto the masters."""

    scaling = np.zeros_like(M)
    scaling[M > 0] = 1 / np.sqrt(M[M > 0])
    weights = transformation.slaveWeightsOfAbsoluteCoefficients(scaling)
    return scaling * transformation.foldRowSumsOfAbsoluteCoefficients(np.abs(Kc) @ weights)


def _exactCriticalTimeStepWithMultiPointConstraints(K: np.ndarray, M: np.ndarray, records: list) -> float:
    """The exact critical time step of the condensed system: the largest eigenvalue of :math:`T^T K T` scaled by the
    folded masses, over the independent degrees of freedom; :math:`T` from the records of the slaves."""

    T = np.eye(len(M))
    for slave, masters in records:
        T[slave] = 0.0
        for master, coefficient in masters:
            T[slave, master] = coefficient
    independent = np.setdiff1d(np.arange(len(M)), [slave for slave, _ in records])
    s = 1 / np.sqrt(M[independent])
    condensed = (T.T @ K @ T)[np.ix_(independent, independent)]
    return 2 / np.sqrt(np.max(np.linalg.eigvalsh(s[:, None] * condensed * s[None, :])))


def _penaltyEstimateWithMultiPointConstraints(elements, lumpedMass, Kc, transformation) -> float:
    """The critical time step as ``NED`` forms it with multi-point constraints: the elements estimated with the
    condensation and their lumped mass before the fold, the rows of the penalty springs with the folded masses."""

    folded = lumpedMass.copy()
    transformation.foldLumpedMass(folded)
    system = _systemOf(elements, len(lumpedMass))
    ofTheElements = CriticalTimeStepOfTheElements()
    ofTheElements.estimate(system, {element: None for element in elements}, transformation, lumpedMass)
    ofTheConstraints = CriticalTimeStepWithThePenaltyConstraints()
    ofTheConstraints.estimate(
        system,
        _rowSumsWithMultiPointConstraints(Kc, folded, transformation),
        ofTheElements,
        CriticalTimeStepOfEntitiesWithOwnInertia(DELTA),
        transformation,
        lumpedMass,
    )
    return min(ofTheElements.value, ofTheConstraints.value)


def _folded(lumpedMass: np.ndarray, transformation) -> np.ndarray:
    folded = lumpedMass.copy()
    transformation.foldLumpedMass(folded)
    return folded


def test_an_element_at_a_slave_counts_at_its_master_where_a_penalty_acts():
    """Masters c, m and a slave s = m of a multi-point constraint; the element (c, m) of stiffness 1, the element (s, f)
    of stiffness 5, a penalty of 15 at s, all lumped masses 1. The slave's element acts on m through :math:`T^T K T`,
    and the penalty folded onto m adds 15/2 there: the bound at m is the largest squared frequency of the elements at m
    *and at its slave*, 10, plus 7.5 -- not 2 + 7.5, which, with the 10 of the element (s, f) as the largest, would
    estimate a critical time step 11 % above the exact one."""

    def element(dofs, k):
        return _ElementWithItsEigenvalue(np.array(dofs), k * np.array([[1.0, -1.0], [-1.0, 1.0]]), np.ones(2))

    c, m, s, f = range(4)
    elements = [element([c, m], 1.0), element([s, f], 5.0)]
    K = np.zeros((4, 4))
    for e in elements:
        K[np.ix_(e.dofs, e.dofs)] += e.K
    Kc = np.zeros((4, 4))
    Kc[s, s] = 15.0
    records = [(s, [(m, 1.0)])]
    transformation = MultiPointConstraintTransformation(records, 4)
    M = np.ones(4)

    exact = _exactCriticalTimeStepWithMultiPointConstraints(K + Kc, _folded(M, transformation), records)
    estimate = _penaltyEstimateWithMultiPointConstraints(elements, M, Kc, transformation)
    assert exact == pytest.approx(2 / np.sqrt(12.265), rel=1e-4)
    assert estimate == pytest.approx(2 / np.sqrt(10.0 + 7.5), rel=1e-14)
    assert estimate <= exact


def _randomRecords(rng, nDof: int, signs: str) -> list:
    """Three slaves of one to four masters each, with coefficients that sum to one: nonnegative, as an interpolation
    has, or of either sign, as the serendipity weights of the hanging nodes of a HEX20 mesh."""

    slaves = rng.choice(nDof, size=3, replace=False)
    independent = np.setdiff1d(np.arange(nDof), slaves)
    records = []
    for slave in slaves:
        masters = rng.choice(independent, size=rng.integers(1 if signs == "nonnegative" else 2, 5), replace=False)
        coefficients = rng.uniform(0.1 if signs == "nonnegative" else -0.3, 1.0, len(masters))
        records.append((int(slave), list(zip(masters.tolist(), (coefficients / coefficients.sum()).tolist()))))
    return records


@pytest.mark.parametrize("signs", ["nonnegative", "of either sign"])
def test_with_multi_point_constraints_the_penalty_springs_bound_the_exact_eigenvalue(signs, monkeypatch, seed=4):
    """Random elements and penalty springs on degrees of freedom of which some are slaves of multi-point constraints,
    with coefficients that sum to one: the estimate is below the exact critical time step of the condensed system,
    :math:`T^T (K + K^c) T` with the folded masses. With coefficients of either sign it is so only with the ratio of the
    mass the elements at a slave need to the mass folded
    (:meth:`~edelweissfe.numerics.mpctransformation.MultiPointConstraintTransformation.ratioOfTheMassLentToTheMassFolded`):
    without it -- the ratio held at one -- the estimate lies above the exact critical time step in some of the
    systems."""

    def estimates():
        rng = np.random.default_rng(seed)
        for _ in range(300):
            K, M, elements, Kc = _randomModelWithPenaltySprings(rng, stiffnessOfTheSprings=10.0)
            records = _randomRecords(rng, len(M), signs)
            transformation = MultiPointConstraintTransformation(records, len(M))
            folded = _folded(M, transformation)
            if np.any(folded[folded != 0.0] < 0.0):
                continue  # a lumped mass the explicit solver refuses
            exact = _exactCriticalTimeStepWithMultiPointConstraints(K + Kc, folded, records)
            yield _penaltyEstimateWithMultiPointConstraints(elements, M, Kc, transformation), exact, elements

    limiting = 0
    for estimate, exact, elements in estimates():
        assert estimate <= exact * (1 + 1e-12)
        limiting += estimate < min(e.criticalTimeStep for e in elements)
    assert limiting > 0, "the springs limit in some of the systems"

    if signs == "of either sign":
        monkeypatch.setattr(
            MultiPointConstraintTransformation,
            "ratioOfTheMassLentToTheMassFolded",
            lambda self, lumpedMass: np.ones(self.nDof),
        )
        assert any(estimate > exact * (1 + 1e-12) for estimate, exact, _ in estimates())


def _refinedSerendipityMesh(dimension: int):
    """Two unit HEX20 (or QUAD8) roots side by side, one of them refined 2:1, so that the hanging nodes on the shared
    face (edge) are tied to the coarse serendipity trace, with weights down to -1/4: the elements, with the stiffness of
    a linear elastic solid and the lumped mass of their consistent mass matrix scaled from its diagonal (positive, as
    the explicit solver needs), the assembled stiffness and lumped mass, and the records of the hanging nodes, one per
    slave degree of freedom."""

    from _adaptivemeshbuilder import AdaptiveMeshBuilder

    from edelweissfe.adaptivity.hex20shapefunctions import hex20_box_coords
    from edelweissfe.adaptivity.hex20topology import Hex20Topology
    from edelweissfe.adaptivity.quad8topology import Quad8Topology
    from edelweissfe.config.materiallibrary import getMaterialClass
    from edelweissfe.elements._quad2dnodeordering import quad8NodeXi
    from edelweissfe.elements.displacementelement.element import DisplacementElement
    from edelweissfe.points.node import Node

    if dimension == 3:
        builder, elementType = AdaptiveMeshBuilder(2, Hex20Topology()), "C3D20"
        builder.addRoot(hex20_box_coords(-1.0, 0.0, 0.0, 1.0, 0.0, 1.0))
        refined = builder.addRoot(hex20_box_coords(0.0, 1.0, 0.0, 1.0, 0.0, 1.0))
    else:
        builder, elementType = AdaptiveMeshBuilder(2, Quad8Topology()), "CPE8"

        def quad8Box(x0, x1):
            return np.array([x0, 0.0]) + (quad8NodeXi.astype(float) + 1) / 2 * np.array([x1 - x0, 1.0])

        builder.addRoot(quad8Box(-1.0, 0.0))
        refined = builder.addRoot(quad8Box(0.0, 1.0))
    mesh = builder.mesh
    mesh.refine(refined)

    labels = sorted({label for eid in mesh.active() for label in mesh.elements[eid]["conn"]})
    indexOf = {label: i for i, label in enumerate(labels)}
    nDof = dimension * len(labels)
    K, M, elements = np.zeros((nDof, nDof)), np.zeros(nDof), []
    for number, eid in enumerate(mesh.active()):
        element = DisplacementElement(elementType, number + 1)
        element.setNodes([Node(i + 1, x) for i, x in enumerate(np.asarray(mesh.elements[eid]["coords"], dtype=float))])
        element.setProperties(np.array([1.0]))
        element.setMaterial(getMaterialClass("linearelastic", "edelweiss")(np.array([1000.0, 0.3, 1.0])))
        element.initializeElement()
        Ke, consistentMass = np.zeros((element.nDof, element.nDof)), np.zeros((element.nDof, element.nDof))
        element.computeKernels(Ke, *np.zeros((3, element.nDof)), np.zeros(2), 1.0)
        element.computeConsistentMassMatrix(consistentMass)
        Me = np.diag(consistentMass) * consistentMass.sum() / np.trace(consistentMass)
        dofs = np.array(
            [dimension * indexOf[label] + j for label in mesh.elements[eid]["conn"] for j in range(dimension)]
        )
        K[np.ix_(dofs, dofs)] += Ke
        M[dofs] += Me
        elements.append(_ElementWithItsEigenvalue(dofs, Ke, Me))
    records = [
        (dimension * indexOf[slave] + j, [(dimension * indexOf[master] + j, weight) for master, weight in masters])
        for slave, masters in mesh.hanging_mpc_records().items()
        for j in range(dimension)
    ]
    return elements, K, M, records


@pytest.mark.parametrize("dimension", [2, 3], ids=["QUAD8", "HEX20"])
def test_with_the_hanging_nodes_of_serendipity_elements_the_estimate_bounds_the_exact_eigenvalue(dimension):
    """A refined HEX20 (QUAD8) mesh with hanging nodes, whose serendipity weights are negative at some masters: the
    estimate of the elements, alone and with a penalty spring at the hanging node of the largest absolute weights, is
    below the exact critical time step of the condensed system. The ratio of the mass the elements at the slaves need to
    the mass folded is above one at masters of the face, and lowers the elements' estimate below the smallest critical
    time step of an element."""

    elements, K, M, records = _refinedSerendipityMesh(dimension)
    assert min(weight for _, masters in records for _, weight in masters) < 0.0, "no negative serendipity weight"
    transformation = MultiPointConstraintTransformation(records, len(M))
    folded = _folded(M, transformation)
    assert transformation.ratioOfTheMassLentToTheMassFolded(M).max() > 1.0

    sumOfTheAbsoluteWeights = {slave: sum(abs(weight) for _, weight in masters) for slave, masters in records}
    slave = max(sumOfTheAbsoluteWeights, key=sumOfTheAbsoluteWeights.get)
    estimates = []
    for penalty in (0.0, 1e3, 1e5):
        Kc = np.zeros_like(K)
        Kc[slave, slave] = penalty
        exact = _exactCriticalTimeStepWithMultiPointConstraints(K + Kc, folded, records)
        estimates.append(_penaltyEstimateWithMultiPointConstraints(elements, M, Kc, transformation))
        assert estimates[-1] <= exact
    assert estimates[0] < min(element.criticalTimeStep for element in elements)
