#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""BlockAMGSolver: the learned true/scaled residual gap and the refresh on a new increment.

1. The gap is the ratio of the true to the scaled relative residual of the same solution. It used to be
   divided by the tolerance GMRES was *given*; on a solve stopped at the iteration cap that inflated it by the
   unreached factor, so every following first pass asked for that much more accuracy.
2. ``refreshOnNewIncrement=False`` keeps the standing hierarchies across a residual jump (new increment);
   the default rebuilds them, as before.
"""

import numpy as np
import pytest
from _blockamgsystems import displacementDamageSystem

from edelweissfe.linsolve.blockamg import createSolver


def _system(n=12):
    """A one-way coupled displacement / damage system with displacement rows scaled over two orders of magnitude,
    so that the equilibration matters."""
    return displacementDamageSystem(n, coupling="one-way", rowScaling=True)


def _solver(blocks, **options):
    config = {"outerTol": 1e-10, "verbosity": "silent", "gapCompensatedTolerance": True}
    config.update(options)
    solver = createSolver(config)
    solver.setFieldStructure(blocks)
    return solver


def test_the_gap_is_learned_from_the_residual_actually_reached():
    A, b, blocks = _system()
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
    A, b, blocks = _system()
    solver = _solver(blocks, outerTol=1e-6, refreshOnNewIncrement=refresh, hierarchyStalenessFactor=1e9)
    solver(A, b)
    standing = solver._preconditioners

    # A residual jump far beyond residualGrowthFactor marks a new increment.
    solver(A, 1e3 * b)

    assert (solver._preconditioners is standing) is (not refresh)
