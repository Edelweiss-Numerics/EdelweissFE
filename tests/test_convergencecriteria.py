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
"""The convergence criteria of the implicit solvers (:mod:`edelweissfe.solvers.base.convergencecriteria`) and the
counting of residual growths (:meth:`NonlinearSolverBase.checkDivergingSolution`).

Besides the behaviour of each rule, two equivalences are pinned, each against a replica of the code it replaces:

* the ``legacy`` criterion with EdelweissFE's defaults decides exactly as the original criterion of EdelweissFE
  (origin/next_v26.11 51b719b0) did, including its check of the scalar variables and its growth counting;
* the ``legacy`` criterion with EdelweissMeshfree's parameter set decides exactly as the implicit solvers of
  EdelweissMeshfree (origin/next_v26.11 bb1a090e) do: on random inputs, and on the decisions recorded from its example
  suite in ``golden/meshfree_convergence_decisions.jsonl``: complete increment attempts, one per line, each check
  with the exact inputs of the decision (iteration, field, largest flux residual, spatial average flux, largest
  increment, largest correction) and EdelweissMeshfree's verdicts (flux, correction).
"""

import json
from pathlib import Path

import numpy as np
import pytest

from edelweissfe.config import phenomena
from edelweissfe.config.configurator import loadConfiguration
from edelweissfe.journal.journal import Journal
from edelweissfe.solvers.base.convergencecriteria import (
    AbaqusConvergenceCriterion,
    FieldConvergence,
    LegacyConvergenceCriterion,
)
from edelweissfe.solvers.nonlinearimplicitstatic import NIST
from edelweissfe.utils.exceptions import DivergingSolution

_RECORDED_DECISIONS = Path(__file__).parent / "golden" / "meshfree_convergence_decisions.jsonl"


def legacy(**overrides) -> LegacyConvergenceCriterion:
    """The legacy criterion with EdelweissFE's defaults: the tolerances of the phenomena, the options of NIST."""

    parameters = dict(
        fluxResidualTolerances=phenomena.fluxResidualTolerance,
        fluxResidualTolerancesAlternative=phenomena.fluxResidualToleranceAlternative,
        absoluteFluxResidualTolerances=phenomena.absoluteFluxResidualTolerance,
        fieldCorrectionTolerances=phenomena.fieldCorrectionTolerance,
        relativeFieldCorrectionTolerances=phenomena.relativeFieldCorrectionTolerance,
        iterationsForAlternativeTolerance=NIST.SolverSpecificOptions["iterationsForAlternativeTolerance"],
        zeroFluxThreshold=NIST.SolverSpecificOptions["zeroFluxThreshold"],
        zeroIncrementThreshold=NIST.SolverSpecificOptions["zeroIncrementThreshold"],
        minimumIterations=NIST.SolverSpecificOptions["minimumIterations"],
    )
    return LegacyConvergenceCriterion(**(parameters | overrides))


def meshfree(
    fieldTolerances: dict, iterationsForAlternativeTolerance=5, zeroIncrementThreshold=1e-13, zeroFluxThreshold=1e-9
):
    """The legacy criterion with EdelweissMeshfree's parameter set; ``fieldTolerances`` per field, see
    :func:`meshfreeTolerances`."""

    def perField(key):
        return {field: tolerances[key] for field, tolerances in fieldTolerances.items()}

    return LegacyConvergenceCriterion(
        fluxResidualTolerances=perField("fluxTolRel"),
        fluxResidualTolerancesAlternative=perField("fluxTolRelAlt"),
        absoluteFluxResidualTolerances=perField("fluxTolAbs"),
        fieldCorrectionTolerances=perField("corrTolAbs"),
        relativeFieldCorrectionTolerances=perField("corrTolRel"),
        iterationsForAlternativeTolerance=iterationsForAlternativeTolerance,
        zeroFluxThreshold=zeroFluxThreshold,
        zeroIncrementThreshold=zeroIncrementThreshold,
        minimumIterations=1,
    )


def meshfreeTolerances(fields) -> dict:
    """EdelweissMeshfree's default tolerances (``NonlinearQuasistaticSolver.validOptions``) for the given fields."""
    defaults = dict(fluxTolRel=1e-4, fluxTolRelAlt=1e-3, fluxTolAbs=1e-14, corrTolRel=1e-3, corrTolAbs=1e-14)
    return {field: dict(defaults) for field in fields}


def abaqus(**parameters) -> AbaqusConvergenceCriterion:
    return AbaqusConvergenceCriterion(scalarVariableCriterion=legacy(), **parameters)


def nist(**options) -> NIST:
    """A NIST solver with EdelweissFE's default tolerances and the given options, as from the *solver datalines."""
    return NIST(loadConfiguration({}), Journal(verbose=False), **{key: str(value) for key, value in options.items()})


def growths(solver: NIST, fieldHistory: list) -> int:
    """The number of growths ``checkDivergingSolution`` counts in the history of one field: it is diverging for every
    allowed number of growths below that count."""
    return sum(
        solver.checkDivergingSolution({"field": fieldHistory}, allowed) for allowed in range(len(fieldHistory) + 1)
    )


# --- replicas of the code the legacy criterion replaces ----------------------------------------------------------------


def originalLegacyCheckField(field, R, ddU, spatialAveragedFlux, iterationCounter) -> FieldConvergence:
    """``LegacyConvergenceCriterion.checkField`` of EdelweissFE 51b719b0, with the default tolerances."""
    fluxResidualTolerances = (
        phenomena.fluxResidualTolerance if iterationCounter < 15 else phenomena.fluxResidualToleranceAlternative
    )
    indexOfMax = np.argmax(np.abs(R))
    fluxResidual = abs(R[indexOfMax])
    correction = np.linalg.norm(ddU, np.inf) if ddU is not None else 0.0
    return FieldConvergence(
        fluxResidual,
        fluxResidual <= max(fluxResidualTolerances[field] * spatialAveragedFlux, 1e-7),
        correction,
        correction < phenomena.fieldCorrectionTolerance[field],
        indexOfMax,
    )


def originalLegacySpatialAveragedFlux(F, nAccumulatedFluxes) -> float:
    """``LegacyConvergenceCriterion.computeSpatialAveragedFluxes`` of EdelweissFE 51b719b0, for one field."""
    return max(1e-10, np.linalg.norm(F, 1) / nAccumulatedFluxes)


def originalScalarVariablesCheck(R, ddU, iterationCounter) -> FieldConvergence:
    """The check of the scalar variables in ``NonlinearSolverBase.checkConvergence`` of EdelweissFE 51b719b0."""
    fluxResidualTolerances = (
        phenomena.fluxResidualTolerance if iterationCounter < 15 else phenomena.fluxResidualToleranceAlternative
    )
    scalarResiduals = np.abs(R)
    indexOfLargestScalarResidual = int(np.argmax(scalarResiduals))
    residualScalarVariables = scalarResiduals[indexOfLargestScalarResidual]
    correction = np.linalg.norm(ddU, np.inf) if ddU is not None else 0.0
    convergedCorrection = correction < phenomena.fieldCorrectionTolerance["scalar variables"]
    convergedFlux = residualScalarVariables <= fluxResidualTolerances["scalar variables"]
    return FieldConvergence(
        residualScalarVariables, convergedFlux, correction, convergedCorrection, indexOfLargestScalarResidual
    )


def originalGrowths(fluxResiduals) -> int:
    """The growth counting of ``checkConvergence`` of EdelweissFE 51b719b0: the history of a field starts at (0.0, 0)."""
    previousFluxResidual, nGrew = 0.0, 0
    for fluxResidual in fluxResiduals:
        if fluxResidual > previousFluxResidual:
            nGrew += 1
        previousFluxResidual = fluxResidual
    return nGrew


# --- replicas of EdelweissMeshfree's decisions (edelweissmeshfree/solvers/base/implicitbase.py, bb1a090e) -------------


def meshfreeDecision(R, ddU, dU, spatialAveragedFlux, iterations, tolerances, options) -> tuple[bool, bool]:
    """The verdict of ``_computeResiduals`` and ``_checkConvergence`` on one field: (converged flux, converged
    correction)."""
    fieldResidualAbs = np.abs(R)
    indexOfMax = np.argmax(fieldResidualAbs)
    fluxResidualAbsolute = fieldResidualAbs[indexOfMax]
    fluxResidualRelative = fluxResidualAbsolute / max(spatialAveragedFlux, 1e-16)
    maxIncrement = np.linalg.norm(dU, np.inf)
    correctionAbsolute = np.linalg.norm(ddU, np.inf) if ddU is not None else 0.0
    correctionRelative = correctionAbsolute / max(maxIncrement, 1e-16)

    useStrictFluxTolerances = iterations < options["iterations for alt. tolerances"]
    fluxTolRel = tolerances["fluxTolRel"] if useStrictFluxTolerances else tolerances["fluxTolRelAlt"]
    fluxTolAbs = tolerances["fluxTolAbs"]
    correctionTolRel = tolerances["corrTolRel"]
    correctionTolAbs = tolerances["corrTolAbs"]

    nonZeroIncrement = maxIncrement > options["zero increment threshhold"]
    convergedCorrection = correctionRelative < correctionTolRel if nonZeroIncrement else True
    convergedCorrection = convergedCorrection or correctionAbsolute < correctionTolAbs

    nonZeroFlux = spatialAveragedFlux > options["zero flux threshhold"]
    convergedFlux = fluxResidualRelative < fluxTolRel if nonZeroFlux else True
    convergedFlux = convergedFlux or fluxResidualAbsolute < fluxTolAbs

    if iterations == 0:
        convergedFlux = False

    return bool(convergedFlux), bool(convergedCorrection)


def meshfreeGrowths(history) -> int:
    """The growth counting of ``_checkDivergingSolution`` for one field; ``history`` holds (absolute flux residual,
    converged flux, converged correction) per checked iteration."""
    nGrew = 0
    for i in range(len(history) - 1):
        if history[i + 1][1] and history[i + 1][2]:  # don't count converged iterations
            continue
        if history[i + 1][0] > history[i][0]:
            nGrew += 1
    return nGrew


MESHFREE_OPTIONS = {
    "iterations for alt. tolerances": 5,
    "zero increment threshhold": 1e-13,
    "zero flux threshhold": 1e-9,
}


# --- random inputs around every threshold of both parameter sets ------------------------------------------------------


def randomInputs(seed: int, n: int):
    """``n`` random (R, ddU, dU, F, iteration): magnitudes from 1e-17 to 1e2 and exact zeros, so that every rule of
    both parameter sets decides some of them."""
    rng = np.random.default_rng(seed)

    def magnitudes(size, zeroProbability):
        values = 10.0 ** rng.uniform(-17.0, 2.0, size) * rng.choice([-1.0, 1.0], size)
        values[rng.random(size) < zeroProbability] = 0.0
        return values

    for _ in range(n):
        size = int(rng.integers(1, 5))
        R = magnitudes(size, 0.05)
        ddU = None if rng.random() < 0.15 else magnitudes(size, 0.05)
        dU = magnitudes(size, 0.15)
        F = np.abs(magnitudes(size, 0.15))
        yield R, ddU, dU, F, int(rng.integers(0, 21))


# --- the legacy criterion with EdelweissFE's defaults is the original criterion ------------------------------------


def test_legacy_defaults_decide_as_the_original_criterion():
    criterion = legacy()
    fields = [field for field in phenomena.fieldCorrectionTolerance if field != "scalar variables"]

    for number, (R, ddU, dU, F, iteration) in enumerate(randomInputs(seed=1, n=20000)):
        field = fields[number % len(fields)]
        nAccumulatedFluxes = 1 + number % 7

        class DofManager:
            idcsOfFieldsInDofVector = {field: slice(0, len(F))}
            nAccumulatedNodalFluxesFieldwise = {field: nAccumulatedFluxes}

        spatialAveragedFlux = criterion.computeSpatialAveragedFluxes(F, None, DofManager())[field]

        assert criterion.checkField(field, R, ddU, dU, spatialAveragedFlux, iteration) == originalLegacyCheckField(
            field, R, ddU, originalLegacySpatialAveragedFlux(F, nAccumulatedFluxes), iteration
        )


def test_legacy_defaults_decide_at_the_thresholds_as_the_original_criterion():
    criterion = legacy()
    tolerance = phenomena.fluxResidualTolerance["displacement"]
    for spatialAveragedFlux in (0.0, 1e-12, 1e-10, 1.0, 1e3, 1e6):
        for fluxResidual in (0.0, 1e-7, np.nextafter(1e-7, 1.0), tolerance * spatialAveragedFlux):
            for iteration in (14, 15):
                R = np.array([fluxResidual])
                ddU = np.array([phenomena.fieldCorrectionTolerance["displacement"]])
                original = originalLegacyCheckField("displacement", R, ddU, max(1e-10, spatialAveragedFlux), iteration)
                assert criterion.checkField("displacement", R, ddU, R, spatialAveragedFlux, iteration) == original


def test_legacy_defaults_check_the_scalar_variables_as_the_original_criterion():
    criterion = legacy()
    for R, ddU, dU, _, iteration in randomInputs(seed=2, n=5000):
        assert criterion.checkScalarVariables(R, ddU, dU, iteration) == originalScalarVariablesCheck(R, ddU, iteration)


def test_default_growth_counting_is_the_original_one():
    solver = nist()
    rng = np.random.default_rng(3)
    for _ in range(500):
        n = int(rng.integers(1, 12))
        fluxResiduals = 10.0 ** rng.uniform(-3, 3, n)
        fluxResiduals[rng.random(n) < 0.1] = 0.0
        repeated = rng.random(n) < 0.1
        fluxResiduals[1:][repeated[1:]] = fluxResiduals[:-1][repeated[1:]]
        converged = rng.random(n) < 0.3
        history = [FieldConvergence(r, bool(c), 0.0, bool(c), 0) for r, c in zip(fluxResiduals, converged)]

        assert growths(solver, history) == originalGrowths(fluxResiduals)


# --- the legacy criterion with EdelweissMeshfree's parameter set is EdelweissMeshfree's criterion ---------------------


def test_meshfree_parameter_set_decides_as_edelweissmeshfree_on_random_inputs():
    tolerances = meshfreeTolerances(["displacement"])
    criterion = meshfree(tolerances)

    for R, ddU, dU, F, iteration in randomInputs(seed=4, n=20000):
        spatialAveragedFlux = np.linalg.norm(F, 1) / len(F)
        verdict = criterion.checkField("displacement", R, ddU, dU, spatialAveragedFlux, iteration)

        assert (verdict.fluxResidualConverged, verdict.correctionConverged) == meshfreeDecision(
            R, ddU, dU, spatialAveragedFlux, iteration, tolerances["displacement"], MESHFREE_OPTIONS
        )


def test_meshfree_growth_counting_on_random_histories():
    solver = nist(countFirstResidualAsGrowth=False, countGrowthOfConvergedFields=False)
    rng = np.random.default_rng(5)
    for _ in range(500):
        n = int(rng.integers(1, 12))
        fluxResiduals = 10.0 ** rng.uniform(-3, 3, n)
        fluxResiduals[rng.random(n) < 0.1] = 0.0
        flux = rng.random(n) < 0.4
        correction = rng.random(n) < 0.6
        history = [FieldConvergence(r, bool(f), 0.0, bool(c), 0) for r, f, c in zip(fluxResiduals, flux, correction)]

        assert growths(solver, history) == meshfreeGrowths(list(zip(fluxResiduals, flux, correction)))


def recordedAttempts():
    return [json.loads(line) for line in _RECORDED_DECISIONS.read_text().splitlines()]


@pytest.mark.parametrize("attempt", recordedAttempts(), ids=lambda attempt: attempt["test"])
def test_meshfree_parameter_set_decides_as_edelweissmeshfree_on_recorded_decisions(attempt):
    criterion = meshfree(
        attempt["fieldTolerances"],
        attempt["iterationsForAlternativeTolerance"],
        attempt["zeroIncrementThreshold"],
        attempt["zeroFluxThreshold"],
    )
    options = {
        "iterations for alt. tolerances": attempt["iterationsForAlternativeTolerance"],
        "zero increment threshhold": attempt["zeroIncrementThreshold"],
        "zero flux threshhold": attempt["zeroFluxThreshold"],
    }
    solver = nist(countFirstResidualAsGrowth=False, countGrowthOfConvergedFields=False)

    feHistory = {field: [] for field in attempt["fieldTolerances"]}
    recordedHistory = {field: [] for field in attempt["fieldTolerances"]}
    for iteration, field, r, q, du, c, convergedFlux, convergedCorrection in attempt["checks"]:
        R, ddU, dU = np.array([r]), np.array([c]), np.array([du])
        verdict = criterion.checkField(field, R, ddU, dU, q, iteration)

        # the replica is faithful, and the criterion agrees with it
        assert meshfreeDecision(R, ddU, dU, q, iteration, attempt["fieldTolerances"][field], options) == (
            convergedFlux,
            convergedCorrection,
        )
        assert (verdict.fluxResidualConverged, verdict.correctionConverged) == (convergedFlux, convergedCorrection)

        feHistory[field].append(verdict)
        recordedHistory[field].append((r, convergedFlux, convergedCorrection))
        assert growths(solver, feHistory[field]) == meshfreeGrowths(recordedHistory[field])


# --- the rules of the legacy criterion -----------------------------------------------------------------------------


def test_legacy_flux_residual_relative_with_absolute_floor():
    criterion = legacy()
    R = np.array([0.0, -2e-7, 1e-8])

    # 1e-8 * 1.0 is below the absolute floor 1e-7, so 2e-7 is not converged, 1e-7 is
    assert not criterion.checkField("displacement", R, None, R, 1.0, 1).fluxResidualConverged
    assert criterion.checkField("displacement", R / 2, None, R, 1.0, 1).fluxResidualConverged
    # from iteration 15 on, the alternative tolerance 5e-3 applies
    assert criterion.checkField("displacement", 1e3 * R, None, R, 1.0, 15).fluxResidualConverged
    # ... or from the iteration given
    assert (
        legacy(iterationsForAlternativeTolerance=5)
        .checkField("displacement", 1e3 * R, None, R, 1.0, 5)
        .fluxResidualConverged
    )


def test_legacy_correction_is_absolute_by_default():
    criterion = legacy()
    R = np.zeros(2)

    converged = criterion.checkField("displacement", R, np.array([5e-8, 0.0]), np.array([1e-12, 0.0]), 1.0, 1)
    assert converged.correctionConverged
    assert converged.correction == 5e-8
    # a correction of half the increment is no help without a relative tolerance
    assert not criterion.checkField("displacement", R, np.array([0.5]), np.array([1.0]), 1.0, 1).correctionConverged


def test_relative_correction_tolerance():
    criterion = legacy(
        relativeFieldCorrectionTolerances=phenomena.relativeFieldCorrectionTolerance | {"displacement": 1e-3}
    )
    R = np.zeros(2)
    dU = np.array([1.0, -2.0])

    assert criterion.checkField("displacement", R, np.array([1.9e-3, 0.0]), dU, 1.0, 1).correctionConverged
    assert not criterion.checkField("displacement", R, np.array([2.1e-3, 0.0]), dU, 1.0, 1).correctionConverged


def test_zero_increment_rule():
    ddU = np.array([1.0])
    for threshold, converged in ((None, False), (1e-13, True)):
        criterion = legacy(zeroIncrementThreshold=threshold)
        assert (
            criterion.checkField("displacement", np.zeros(1), ddU, np.array([1e-14]), 1.0, 1).correctionConverged
            == converged
        )


def test_zero_flux_rule_and_minimum_iterations():
    R = np.array([1.0])
    for threshold, converged in ((None, False), (1e-9, True)):
        criterion = legacy(zeroFluxThreshold=threshold)
        assert criterion.checkField("nonlocal damage", R, None, R, 1e-12, 1).fluxResidualConverged == converged

    # before the first correction, a model at rest reports no flux: the minimum of one iteration keeps the zero flux rule
    # from accepting the unconverged prediction
    criterion = legacy(zeroFluxThreshold=1e-9, minimumIterations=1)
    assert not criterion.checkField("nonlocal damage", R, None, R, 0.0, 0).fluxResidualConverged
    assert criterion.checkField("nonlocal damage", R, None, R, 0.0, 1).fluxResidualConverged


def test_scalar_variables_are_checked_like_a_field_with_their_own_tolerances():
    tolerances = meshfreeTolerances(["displacement", "scalar variables"])
    tolerances["scalar variables"] |= {"fluxTolRel": 1e-6, "corrTolAbs": 1e-3}
    criterion = meshfree(tolerances)
    dU = np.array([10.0, 1.0])

    # compared with the tolerances directly, as if the average flux were one
    assert criterion.checkScalarVariables(np.array([0.5e-6, 0.0]), np.zeros(2), dU, 1).fluxResidualConverged
    assert not criterion.checkScalarVariables(np.array([2e-6, 0.0]), np.zeros(2), dU, 1).fluxResidualConverged
    # never before the minimum number of iterations
    assert not criterion.checkScalarVariables(np.zeros(2), None, dU, 0).fluxResidualConverged
    # the correction against the absolute and the relative tolerance
    assert criterion.checkScalarVariables(np.zeros(2), np.array([0.9e-3, 0.0]), dU, 1).correctionConverged
    assert criterion.checkScalarVariables(np.zeros(2), np.array([9e-3, 0.0]), dU, 1).correctionConverged
    assert not criterion.checkScalarVariables(np.zeros(2), np.array([11e-3, 0.0]), dU, 1).correctionConverged


def test_largest_residual_index():
    for criterion in (legacy(), abaqus()):
        assert (
            criterion.checkField(
                "displacement", np.array([1.0, -3.0, 2.0]), None, np.ones(3), 1.0, 1
            ).indexOfLargestResidual
            == 1
        )


# --- the options of the solver --------------------------------------------------------------------------------------


def test_options_make_the_criterion():
    solver = nist(
        iterationsForAlternativeTolerance=5, zeroFluxThreshold=1e-9, zeroIncrementThreshold=1e-13, minimumIterations=1
    )
    criterion = solver.createConvergenceCriterion()
    assert isinstance(criterion, LegacyConvergenceCriterion)
    assert (criterion.iterationsForAlternativeTolerance, criterion.zeroFluxThreshold) == (5, 1e-9)
    assert (criterion.zeroIncrementThreshold, criterion.minimumIterations) == (1e-13, 1)
    assert criterion.absoluteFluxResidualTolerances is solver.absoluteFluxResidualTolerances

    defaults = nist().createConvergenceCriterion()
    assert (defaults.zeroFluxThreshold, defaults.zeroIncrementThreshold, defaults.minimumIterations) == (None, None, 0)

    # Abaqus' criterion covers fields only: the scalar variables are checked by the legacy one
    abaqusCriterion = nist(convergenceCriterion="abaqus", minimumIterations=1).createConvergenceCriterion()
    assert isinstance(abaqusCriterion.scalarVariableCriterion, LegacyConvergenceCriterion)
    assert not abaqusCriterion.checkScalarVariables(np.zeros(1), None, np.zeros(1), 0).fluxResidualConverged


def test_a_nan_in_the_residual_is_a_diverging_solution():
    with pytest.raises(DivergingSolution):
        nist().checkConvergence(np.array([0.0, np.nan]), None, np.zeros(2), np.ones(2), np.zeros(2), 1, {})


def test_growth_counting_options():
    # the residual grows twice, once while the field is converged already
    history = [
        FieldConvergence(1.0, False, 0.0, False, 0),
        FieldConvergence(2.0, False, 0.0, False, 0),
        FieldConvergence(0.5, True, 0.0, True, 0),
        FieldConvergence(0.7, True, 0.0, True, 0),
    ]
    assert growths(nist(), history) == 3  # the first residual is a growth from zero
    assert growths(nist(countFirstResidualAsGrowth=False), history) == 2
    assert growths(nist(countGrowthOfConvergedFields=False), history) == 2
    assert growths(nist(countFirstResidualAsGrowth=False, countGrowthOfConvergedFields=False), history) == 1


# --- the criterion of Abaqus/Standard ---------------------------------------------------------------------------------


def test_abaqus_correction_is_relative_to_increment():
    criterion = abaqus()
    R = np.zeros(2)
    dU = np.array([1.0, -2.0])

    assert criterion.checkField("u", R + 1, np.array([0.019, 0.0]), dU, 1e3, 1).correctionConverged
    assert not criterion.checkField("u", R + 1, np.array([0.021, 0.0]), dU, 1e3, 1).correctionConverged
    # no correction computed yet: never converged by correction (unless linear)
    assert not criterion.checkField("u", R + 1, None, dU, 1e3, 1).correctionConverged


def test_abaqus_residual_tolerance_and_alternative():
    criterion = abaqus()
    ddU = np.array([0.0])
    dU = np.array([1.0])

    # first increment: the time average is the current spatial average
    assert criterion.checkField("u", np.array([4.9e-3]), ddU, dU, 1.0, 1).fluxResidualConverged
    assert not criterion.checkField("u", np.array([5.1e-3]), ddU, dU, 1.0, 8).fluxResidualConverged
    assert criterion.checkField("u", np.array([5.1e-3]), ddU, dU, 1.0, 9).fluxResidualConverged
    assert not criterion.checkField("u", np.array([2.1e-2]), ddU, dU, 1.0, 9).fluxResidualConverged


def test_abaqus_time_average_over_accepted_increments():
    criterion = abaqus()
    criterion.startStep(None)

    for spatialAveragedFlux in (1.0, 3.0):
        criterion.checkField("u", np.zeros(1), None, np.ones(1), spatialAveragedFlux, 1)
        criterion.acceptIncrement()

    # a rejected (not accepted) increment does not enter the average
    criterion.checkField("u", np.zeros(1), None, np.ones(1), 100.0, 1)

    assert criterion.timeAveragedFlux("u", 5.0) == pytest.approx((1.0 + 3.0 + 5.0) / 3)
    assert criterion.timeAveragedFluxesAtStepEnd() == {"u": pytest.approx(2.0)}

    # with the time average, a residual that is small compared to the past load is accepted even though the
    # current spatial average flux is small, e.g., after unloading
    converged = criterion.checkField("u", np.array([7e-3]), np.zeros(1), np.ones(1), 0.5, 1)
    assert converged.fluxResidualConverged


def test_abaqus_zero_flux_field():
    criterion = abaqus()
    criterion.checkField("nonlocal", np.zeros(1), None, np.ones(1), 1.0, 1)
    criterion.acceptIncrement()

    dU = np.array([1.0])
    # spatial average flux 1e-6 < 1e-5 * 1.0: no flux, so the residual is not checked, the correction against C_eps
    zeroFlux = criterion.checkField("nonlocal", np.array([1.0]), np.array([0.9e-3]), dU, 1e-6, 1)
    assert zeroFlux.fluxResidualConverged and zeroFlux.correctionConverged
    assert not criterion.checkField("nonlocal", np.array([1.0]), np.array([1.1e-3]), dU, 1e-6, 1).correctionConverged

    # a field which never carried flux and has no residual has nothing to iterate on
    fresh = abaqus()
    untouched = fresh.checkField("nonlocal", np.zeros(2), np.array([1e-20, 0.0]), np.array([1e-20, 0.0]), 0.0, 1)
    assert untouched.fluxResidualConverged and untouched.correctionConverged


def test_abaqus_linear_case_converges_without_correction():
    criterion = abaqus()
    converged = criterion.checkField("u", np.array([1e-9]), None, np.ones(1), 1.0, 0)
    assert converged.fluxResidualConverged and converged.correctionConverged


def test_abaqus_previous_step_is_zero_flux_reference():
    first = abaqus()
    first.checkField("u", np.zeros(1), None, np.ones(1), 10.0, 1)
    first.acceptIncrement()

    second = abaqus()
    second.startStep(first)
    # 1e-5 is below 1e-5 * 10 of the previous step: no flux
    assert second.checkField("u", np.array([1.0]), np.zeros(1), np.ones(1), 1e-5, 1).fluxResidualConverged
    # but the time average of the new step starts from scratch
    assert second.timeAveragedFlux("u", 2.0) == 2.0


def test_spatial_averaged_fluxes_include_constraints_only_for_abaqus():
    class DofManager:
        idcsOfFieldsInDofVector = {"u": slice(0, 3)}
        nAccumulatedNodalFluxesFieldwise = {"u": 4}

    F = np.array([1.0, 2.0, 1.0])
    FConstraints = np.array([0.0, 4.0, 0.0])

    assert legacy().computeSpatialAveragedFluxes(F, FConstraints, DofManager())["u"] == 1.0
    assert abaqus().computeSpatialAveragedFluxes(F, FConstraints, DofManager())["u"] == 2.0


def test_abaqus_initial_time_average_flux_for_a_field_without_flux_yet():
    # round-off flux and residual: against q~0 the field carries no flux
    criterion = abaqus()
    roundOff = criterion.checkField("nonlocal", np.array([3e-42]), np.array([2e-44]), np.array([1e-29]), 1e-42, 5)
    assert roundOff.fluxResidualConverged and roundOff.correctionConverged

    # a real flux, however small in absolute terms, still counts
    assert criterion.timeAveragedFlux("nonlocal", 1e-4) == pytest.approx(1e-4)

    # a field without flux in the whole step leaves no time average flux; the next step starts from q~0 again
    criterion.acceptIncrement()
    assert criterion.timeAveragedFluxesAtStepEnd() == {}
    following = abaqus()
    following.startStep(criterion)
    assert following.checkField(
        "nonlocal", np.array([3e-42]), np.array([2e-44]), np.array([1e-29]), 1e-42, 1
    ).fluxResidualConverged
