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
#  This library is free software; you can redistribute it and/or
#  modify it under the terms of the GNU Lesser General Public
#  License as published by the Free Software Foundation; either
#  version 2.1 of the License, or (at your option) any later version.
#
#  The full text of the license can be found in the file LICENSE.md at
#  the top level directory of EdelweissFE.
#  ---------------------------------------------------------------------
"""Correctness tests for the AMGCL-threaded T^T K T + C expression
(``MultiPointConstraintTransformation(..., useAmgclSpgemm=True)``) against the plain SciPy
expression it is an opt-in alternative to (default ``False``, pending a live gate).

Supersedes ``test_mpc_cached_condensation.py`` (removed): that file's cache-corruption/exact-zero-
scatter-map regressions were specific to the cached-pattern value scatter, which has been removed
entirely (measured slower than the plain expression, never shipped, superseded by this AMGCL path).
The general MPC-construction tests (chained records, zero slave DOFs) are ported here; the
scatter-map-specific ones have no equivalent in the AMGCL path, which has no cache to corrupt.
"""

import os

import numpy as np
import pytest
import scipy.sparse as sp

from edelweissfe.numerics.mpctransformation import MultiPointConstraintTransformation

# Every test here drives the AMGCL-threaded condensation path, which needs the compiled `amgcl`
# extension. setup.py builds that one as optional, so a SciPy-only install is supported and must
# skip these rather than fail them.
pytest.importorskip("edelweissfe.linsolve.amgcl.amgcl")


def _randomSparseK(n, density=0.02, seed=0):
    r = np.random.default_rng(seed)
    A = sp.random(n, n, density=density, format="csr", random_state=r, data_rvs=lambda k: r.standard_normal(k))
    A = (A + A.T) / 2.0 + sp.eye(n) * n  # diagonally dominant-ish, symmetric
    return A.tocsr()


def _maxRelDiff(A, B):
    diff = (A - B).tocsr()
    maxDiff = np.max(np.abs(diff.data)) if diff.nnz else 0.0
    scale = max(np.max(np.abs(A.data)) if A.nnz else 0.0, np.max(np.abs(B.data)) if B.nnz else 0.0, 1e-300)
    return maxDiff / scale


def test_amgcl_matches_plain_expression():
    n = 200
    records = [
        (5, [(0, 0.5), (1, 0.5)]),
        (6, [(1, 0.3), (2, 0.7)]),
        (7, [(0, 1.0)]),
        (8, [(6, 0.4), (3, 0.6)]),  # chained: master 6 is itself a slave
    ]
    K = _randomSparseK(n, seed=1)

    plain = MultiPointConstraintTransformation(records, n).transformSystemMatrix(K)
    amgcl = MultiPointConstraintTransformation(records, n, useAmgclSpgemm=True).transformSystemMatrix(K)
    assert _maxRelDiff(plain, amgcl) < 1e-9


def test_chained_records_are_flattened_to_independent_dofs():
    n = 200
    records = [
        (5, [(0, 0.5), (1, 0.5)]),
        (6, [(1, 0.3), (2, 0.7)]),
        (8, [(6, 0.4), (3, 0.6)]),  # chained: master 6 is itself a slave
    ]
    mpc = MultiPointConstraintTransformation(records, n, useAmgclSpgemm=True)

    idx8 = list(mpc.slaveDofIndices).index(8)
    wRow8 = mpc._W[idx8, :].toarray().flatten()
    nonzeroCols = np.nonzero(wRow8)[0]
    assert 6 not in nonzeroCols, "chained slave 6 should have been substituted, not left as a column"
    assert set(nonzeroCols) <= {1, 2, 3}, "unexpected columns after flattening"


def test_zero_slave_dofs_regression():
    """A system assembled before any hanging-node/tie constraint exists yet (T = identity, C = 0)
    must still condense to K unchanged, via either expression."""
    n = 200
    K = _randomSparseK(n, seed=6)

    plain = MultiPointConstraintTransformation([], n).transformSystemMatrix(K)
    amgcl = MultiPointConstraintTransformation([], n, useAmgclSpgemm=True).transformSystemMatrix(K)
    assert _maxRelDiff(plain, K) < 1e-9
    assert _maxRelDiff(amgcl, K) < 1e-9


def test_assert_exact_env_var_passes_on_a_correct_computation():
    n = 200
    records = [(5, [(0, 0.5), (1, 0.5)])]
    mpc = MultiPointConstraintTransformation(records, n, useAmgclSpgemm=True)
    K = _randomSparseK(n, seed=3)

    os.environ["EDELWEISS_MPC_ASSERT_EXACT"] = "1"
    try:
        mpc.transformSystemMatrix(K)  # correct implementation -- must not raise
    finally:
        del os.environ["EDELWEISS_MPC_ASSERT_EXACT"]


def test_default_uses_plain_expression():
    mpc = MultiPointConstraintTransformation([], 10)
    assert mpc._useAmgclSpgemm is False
