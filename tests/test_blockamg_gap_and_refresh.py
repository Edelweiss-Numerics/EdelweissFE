#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""BlockAMGSolver: the learned true/scaled residual gap and the refresh on a new increment.

1. The gap is the ratio of the true to the scaled relative residual of the same solution. It used to be
   divided by the tolerance GMRES was *given*; on a solve stopped at the iteration cap that inflated it by the
   unreached factor, so every following first pass asked for that much more accuracy.
2. ``refreshOnNewIncrement=False`` keeps the standing hierarchies across a residual jump (new increment);
   the default rebuilds them, as before.
3. A requested rebuild (setModel, a hot reload, a stale hierarchy) stands until a build succeeds.
4. A right-hand side that is not finite is not solved, and leaves the solver's state as it was.
"""

import numpy as np
import pytest
from _blockamgsystems import displacementDamageSystem

from edelweissfe.linsolve.blockamg import createSolver


def _solver(blocks, **options):
    # krylovOnScaledSystem=True pinned deliberately: the learned gap (_trueResidualGap) is only ever
    # updated on the scaled system (see BlockAMGSolver.__call__) -- on the unscaled one, the default
    # since the current defaults, the outer solve already converges the true residual itself, so there
    # is no gap to learn, and gapCompensatedTolerance has nothing to do.
    config = {
        "outerTol": 1e-10,
        "verbosity": "silent",
        "gapCompensatedTolerance": True,
        "krylovOnScaledSystem": True,
    }
    config.update(options)
    solver = createSolver(config)
    solver.setFieldStructure(blocks)
    return solver


def test_the_gap_is_learned_from_the_residual_actually_reached():
    A, b, blocks = displacementDamageSystem(12, coupling="one-way", rowScaling=True)
    # Two outer iterations cannot reach 1e-10: the solve stops at the cap. No continuations, so the returned
    # solution is the one the gap was learned from.
    solver = _solver(blocks, outerRestart=2, outerMaxiter=1, gapMaxFactor=1e6, trueResidualMaxContinuations=0)
    x = solver(A, b)

    dinv = 1.0 / np.sqrt(np.abs(A.diagonal()))
    r = b - A @ x
    trueRelative = np.linalg.norm(r) / np.linalg.norm(b)
    scaledRelative = np.linalg.norm(dinv * r) / np.linalg.norm(dinv * b)
    assert trueRelative > 1e-6, "the solve must stop at the cap for this test to mean anything"

    expected = min(max(0.5 * 1.0 + 0.5 * trueRelative / scaledRelative, 1.0), 1e6)
    assert solver._trueResidualGap == pytest.approx(expected, rel=1e-8)


@pytest.mark.parametrize("refresh", [True, False])
def test_a_new_increment_rebuilds_the_hierarchies_only_if_asked(refresh):
    A, b, blocks = displacementDamageSystem(12, coupling="one-way", rowScaling=True)
    solver = _solver(blocks, outerTol=1e-6, refreshOnNewIncrement=refresh, hierarchyStalenessFactor=1e9)
    solver(A, b)
    standing = solver._preconditioners

    # A residual jump far beyond residualGrowthFactor marks a new increment.
    solver(A, 1e3 * b)

    assert (solver._preconditioners is standing) is (not refresh)


def test_a_requested_rebuild_stands_until_a_build_succeeds(monkeypatch):
    A, b, blocks = displacementDamageSystem(12, coupling="one-way", rowScaling=True)
    solver = _solver(blocks, outerTol=1e-6)
    solver(A, b)
    standing = solver._preconditioners
    solver._refreshNext = True

    def failingBuild(*args):
        raise RuntimeError("build failed")

    monkeypatch.setattr(solver, "_buildFieldPreconditioners", failingBuild)
    with pytest.raises(RuntimeError, match="build failed"):
        solver(A, b)
    assert solver._refreshNext and solver._preconditioners is standing

    # the same layout and sparsity pattern: only the standing request rebuilds
    monkeypatch.undo()
    solver(A, b)
    assert not solver._refreshNext and solver._preconditioners is not standing


def test_a_right_hand_side_that_is_not_finite_is_not_solved():
    A, b, blocks = displacementDamageSystem(12, coupling="one-way", rowScaling=True)
    solver = _solver(blocks, outerTol=1e-6)
    solver(A, b)
    lastResidualNorm = solver._lastResidualNorm
    b[3] = np.nan
    assert np.isnan(solver(A, b)).all()
    assert solver._lastResidualNorm == lastResidualNorm


@pytest.mark.parametrize("entry", [np.nan, np.inf], ids=["NaN", "inf"])
def test_a_diagonal_that_is_not_finite_is_not_solved(entry):
    # Its equilibration is not finite either, and so are the translations' weights 1/scaling of the displacement
    # field's block hierarchy, which AMGCL's hierarchy build refuses with a RuntimeError the nonlinear solvers do
    # not catch: like a right-hand side that is not finite, it has to come back as a NaN correction (a cutback).
    A, b, blocks = displacementDamageSystem(12, coupling="one-way", rowScaling=True)
    diagonal = A.diagonal()
    diagonal[3] = entry
    A.setdiag(diagonal)
    solver = _solver(blocks, outerTol=1e-6)
    assert np.isnan(solver(A, b)).all()
    # nothing was built, and the solver's state is as it was
    assert solver._preconditioners is None
    assert solver._lastResidualNorm is None
