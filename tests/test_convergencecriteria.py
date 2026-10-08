import numpy as np
import pytest

from edelweissfe.solvers.base.convergencecriteria import (
    AbaqusConvergenceCriterion,
    LegacyConvergenceCriterion,
)


def legacy():
    return LegacyConvergenceCriterion({"displacement": 1e-8}, {"displacement": 5e-3}, {"displacement": 1e-7})


def test_legacy_flux_residual_relative_with_absolute_floor():
    criterion = legacy()
    R = np.array([0.0, -2e-7, 1e-8])

    # 1e-8 * 1.0 is below the absolute floor 1e-7, so 2e-7 is not converged, 1e-7 is
    assert not criterion.checkField("displacement", R, None, R, 1.0, 1).fluxResidualConverged
    assert criterion.checkField("displacement", R / 2, None, R, 1.0, 1).fluxResidualConverged
    # from iteration 15 on, the alternative tolerance 5e-3 applies
    assert criterion.checkField("displacement", 1e3 * R, None, R, 1.0, 15).fluxResidualConverged


def test_legacy_correction_is_absolute():
    criterion = legacy()
    R = np.zeros(2)

    converged = criterion.checkField("displacement", R, np.array([5e-8, 0.0]), np.array([1e-12, 0.0]), 1.0, 1)
    assert converged.correctionConverged
    assert converged.correction == 5e-8


def test_largest_residual_index():
    for criterion in (legacy(), AbaqusConvergenceCriterion()):
        assert (
            criterion.checkField(
                "displacement", np.array([1.0, -3.0, 2.0]), None, np.ones(3), 1.0, 1
            ).indexOfLargestResidual
            == 1
        )


def test_abaqus_correction_is_relative_to_increment():
    criterion = AbaqusConvergenceCriterion()
    R = np.zeros(2)
    dU = np.array([1.0, -2.0])

    assert criterion.checkField("u", R + 1, np.array([0.019, 0.0]), dU, 1e3, 1).correctionConverged
    assert not criterion.checkField("u", R + 1, np.array([0.021, 0.0]), dU, 1e3, 1).correctionConverged
    # no correction computed yet: never converged by correction (unless linear)
    assert not criterion.checkField("u", R + 1, None, dU, 1e3, 1).correctionConverged


def test_abaqus_residual_tolerance_and_alternative():
    criterion = AbaqusConvergenceCriterion()
    ddU = np.array([0.0])
    dU = np.array([1.0])

    # first increment: the time average is the current spatial average
    assert criterion.checkField("u", np.array([4.9e-3]), ddU, dU, 1.0, 1).fluxResidualConverged
    assert not criterion.checkField("u", np.array([5.1e-3]), ddU, dU, 1.0, 8).fluxResidualConverged
    assert criterion.checkField("u", np.array([5.1e-3]), ddU, dU, 1.0, 9).fluxResidualConverged
    assert not criterion.checkField("u", np.array([2.1e-2]), ddU, dU, 1.0, 9).fluxResidualConverged


def test_abaqus_time_average_over_accepted_increments():
    criterion = AbaqusConvergenceCriterion()
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
    criterion = AbaqusConvergenceCriterion()
    criterion.checkField("nonlocal", np.zeros(1), None, np.ones(1), 1.0, 1)
    criterion.acceptIncrement()

    dU = np.array([1.0])
    # spatial average flux 1e-6 < 1e-5 * 1.0: no flux, so the residual is not checked, the correction against C_eps
    zeroFlux = criterion.checkField("nonlocal", np.array([1.0]), np.array([0.9e-3]), dU, 1e-6, 1)
    assert zeroFlux.fluxResidualConverged and zeroFlux.correctionConverged
    assert not criterion.checkField("nonlocal", np.array([1.0]), np.array([1.1e-3]), dU, 1e-6, 1).correctionConverged

    # a field which never carried flux and has no residual has nothing to iterate on
    fresh = AbaqusConvergenceCriterion()
    untouched = fresh.checkField("nonlocal", np.zeros(2), np.array([1e-20, 0.0]), np.array([1e-20, 0.0]), 0.0, 1)
    assert untouched.fluxResidualConverged and untouched.correctionConverged


def test_abaqus_linear_case_converges_without_correction():
    criterion = AbaqusConvergenceCriterion()
    converged = criterion.checkField("u", np.array([1e-9]), None, np.ones(1), 1.0, 0)
    assert converged.fluxResidualConverged and converged.correctionConverged


def test_abaqus_previous_step_is_zero_flux_reference():
    first = AbaqusConvergenceCriterion()
    first.checkField("u", np.zeros(1), None, np.ones(1), 10.0, 1)
    first.acceptIncrement()

    second = AbaqusConvergenceCriterion()
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
    assert AbaqusConvergenceCriterion().computeSpatialAveragedFluxes(F, FConstraints, DofManager())["u"] == 2.0
