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
import scipy.sparse as sp

from edelweissfe.linsolve.base import FieldBlock
from edelweissfe.linsolve.blockamg import createSolver


def _system(n=12):
    """A coupled two-field system: a 3D displacement-like block (3 dofs per node of an n^3 grid Laplacian,
    rows scaled over two orders of magnitude so the equilibration matters) and a scalar block, weakly
    coupled one way."""
    lap1 = sp.diags([-1.0, 2.0, -1.0], [-1, 0, 1], shape=(n, n))
    eye = sp.identity(n)
    lap3 = sp.kron(sp.kron(lap1, eye), eye) + sp.kron(sp.kron(eye, lap1), eye) + sp.kron(sp.kron(eye, eye), lap1)
    nodes = n**3
    Kuu = sp.kron(lap3 + 0.01 * sp.identity(nodes), sp.identity(3))
    rowScale = sp.diags(np.logspace(0, 2, 3 * nodes))
    Kuu = (rowScale @ Kuu @ rowScale).tocsr()
    Kdd = (lap3 + sp.identity(nodes)).tocsr()
    Kud = sp.random(3 * nodes, nodes, density=1e-3, random_state=1) * 1e-2
    A = sp.bmat([[Kuu, Kud], [None, Kdd]], format="csr")
    blocks = [FieldBlock("displacement", 0, 3 * nodes, 3), FieldBlock("nonlocal damage", 3 * nodes, 4 * nodes, 1)]
    b = np.random.default_rng(0).standard_normal(A.shape[0])
    return A, b, blocks


def _solver(blocks, **options):
    config = {"outerTol": 1e-10, "verbosity": "silent", "gapCompensatedTolerance": True}
    config.update(options)
    solver = createSolver(config)
    solver.setFieldStructure(blocks)
    return solver


def test_the_gap_is_learned_from_the_residual_actually_reached():
    A, b, blocks = _system()
    # Two outer iterations cannot reach 1e-10: the solve stops at the cap.
    solver = _solver(blocks, outerRestart=2, outerMaxiter=1, gapMaxFactor=1e6)
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
