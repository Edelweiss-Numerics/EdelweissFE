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
"""Correctness tests for the threaded T^T K T + C condensation
(``MultiPointConstraintTransformation(..., useThreadedCondensation=True)``, the kernel of
:mod:`edelweissfe.numerics.mpccondensation`) against the plain SciPy expression it is an opt-in
alternative to (default ``False``).
"""

import os

import numpy as np
import pytest
import scipy.sparse as sp

from edelweissfe.numerics.mpctransformation import MultiPointConstraintTransformation

# setup.py builds every extension as optional, so a SciPy-only install is supported and must skip these
# rather than fail them.
mpccondensation = pytest.importorskip("edelweissfe.numerics.mpccondensation")


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


def _randomRecords(n, nSlaves, seed):
    """Random MPC records, incl. masters that are themselves slaves (chains, flattened by the transformation)."""
    r = np.random.default_rng(seed)
    slaves = r.choice(n, nSlaves, replace=False)
    # masters only from lower-numbered positions in this slave list or from independent DOFs: no cycles
    records = []
    for k, s in enumerate(slaves):
        candidates = np.setdiff1d(np.arange(n), slaves[k:])
        masters = r.choice(candidates, int(r.integers(1, 5)), replace=False)
        records.append((int(s), [(int(m), float(r.uniform(-1.0, 1.0))) for m in masters]))
    return records


def _assertIdentical(A, B):
    assert np.array_equal(A.indptr, B.indptr)
    assert np.array_equal(A.indices, B.indices)
    assert np.array_equal(A.data, B.data)


def test_threaded_matches_plain_expression():
    n = 200
    records = [
        (5, [(0, 0.5), (1, 0.5)]),
        (6, [(1, 0.3), (2, 0.7)]),
        (7, [(0, 1.0)]),
        (8, [(6, 0.4), (3, 0.6)]),  # chained: master 6 is itself a slave
    ]
    K = _randomSparseK(n, seed=1)

    plain = MultiPointConstraintTransformation(records, n).transformSystemMatrix(K)
    threaded = MultiPointConstraintTransformation(records, n, useThreadedCondensation=True).transformSystemMatrix(K)
    assert _maxRelDiff(plain, threaded) < 1e-14
    assert threaded.has_sorted_indices


@pytest.mark.parametrize("seed", range(5))
def test_threaded_matches_plain_expression_random_records(seed):
    n = 300
    K = _randomSparseK(n, density=0.05, seed=seed)
    records = _randomRecords(n, 60, seed)

    plain = MultiPointConstraintTransformation(records, n).transformSystemMatrix(K)
    threaded = MultiPointConstraintTransformation(records, n, useThreadedCondensation=True).transformSystemMatrix(K)
    assert _maxRelDiff(plain, threaded) < 1e-14


def test_result_independent_of_thread_count():
    n = 400
    K = _randomSparseK(n, density=0.05, seed=7)
    mpc = MultiPointConstraintTransformation(_randomRecords(n, 80, 7), n)
    condensation = mpccondensation.MPCCondensation(mpc._T, mpc._T.T.tocsr(), mpc._C)

    reference = condensation.condense(K, 1)
    for numThreads in (2, 3, 8):
        _assertIdentical(reference, condensation.condense(K, numThreads))


def test_unsorted_and_duplicate_entries_in_K():
    """K's rows need not be sorted, and duplicate entries are summed, as in the plain expression."""
    n = 150
    K = _randomSparseK(n, density=0.05, seed=4)
    r = np.random.default_rng(4)
    indices, data = K.indices.copy(), K.data.copy()
    for i in range(n):
        s, e = K.indptr[i], K.indptr[i + 1]
        p = r.permutation(e - s)
        indices[s:e], data[s:e] = indices[s:e][p], data[s:e][p]
    unsorted = sp.csr_matrix((data, indices, K.indptr.copy()), shape=K.shape)
    unsorted.has_sorted_indices = False
    # the same matrix with every entry of row 3 split into two halves
    s, e = K.indptr[3], K.indptr[4]
    rows = np.concatenate([np.repeat(np.arange(n), np.diff(K.indptr)), np.full(e - s, 3)])
    cols = np.concatenate([K.indices, K.indices[s:e]])
    vals = K.data.copy()
    vals[s:e] *= 0.5
    vals = np.concatenate([vals, K.data[s:e] * 0.5])
    order = np.argsort(rows, kind="stable")
    duplicates = sp.csr_matrix(
        (vals[order], cols[order], np.concatenate([[0], np.cumsum(np.bincount(rows, minlength=n))])), shape=K.shape
    )

    records = _randomRecords(n, 30, 4)
    plain = MultiPointConstraintTransformation(records, n).transformSystemMatrix(K)
    for variant in (unsorted, duplicates):
        threaded = MultiPointConstraintTransformation(records, n, useThreadedCondensation=True).transformSystemMatrix(
            variant
        )
        assert _maxRelDiff(plain, threaded) < 1e-14


def test_bitwise_equal_to_amgcl_products():
    """The kernel sums in the order of AMGCL's spgemm_saad applied as T^T (K T) + C, so the result is the
    same, bit for bit, as AMGCL's product()/sum() at up to 16 threads (AMGCL switches algorithm above)."""
    amgcl = pytest.importorskip("edelweissfe.linsolve.amgcl.amgcl")
    n = 300
    K = _randomSparseK(n, density=0.05, seed=9)
    mpc = MultiPointConstraintTransformation(_randomRecords(n, 60, 9), n)
    TTranspose = mpc._T.T.tocsr()

    helper = amgcl.PyAMGCLSpGEMM()
    viaAmgcl = helper.sum(1.0, helper.product(TTranspose, helper.product(K, mpc._T)), 1.0, mpc._C)
    viaAmgcl.sort_indices()
    _assertIdentical(viaAmgcl, mpccondensation.MPCCondensation(mpc._T, TTranspose, mpc._C).condense(K))


def test_explicit_zeros_are_kept():
    """The pattern is the structural one: an exactly cancelling entry stays stored (pruning is the solver's
    decision, see pruneCondensedMatrixZeros)."""
    n = 4
    K = sp.csr_matrix(np.array([[2.0, 1.0, -1.0, 0.0], [1.0, 2.0, 0.0, 0.0], [-1.0, 0.0, 2.0, 0.0], [0, 0, 0, 1.0]]))
    # u_2 = u_1: K[0,1] + K[0,2] = 0 is stored in the condensed row 0
    mpc = MultiPointConstraintTransformation([(2, [(1, 1.0)])], n, useThreadedCondensation=True)
    Kt = mpc.transformSystemMatrix(K)
    assert Kt[0, 1] == 0.0
    assert 1 in Kt.indices[Kt.indptr[0] : Kt.indptr[1]]


def test_chained_records_are_flattened_to_independent_dofs():
    n = 200
    records = [
        (5, [(0, 0.5), (1, 0.5)]),
        (6, [(1, 0.3), (2, 0.7)]),
        (8, [(6, 0.4), (3, 0.6)]),  # chained: master 6 is itself a slave
    ]
    mpc = MultiPointConstraintTransformation(records, n, useThreadedCondensation=True)

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
    threaded = MultiPointConstraintTransformation([], n, useThreadedCondensation=True).transformSystemMatrix(K)
    assert _maxRelDiff(plain, K) < 1e-9
    _assertIdentical(threaded, K)


def test_assert_exact_env_var_passes_on_a_correct_computation():
    n = 200
    records = [(5, [(0, 0.5), (1, 0.5)])]
    mpc = MultiPointConstraintTransformation(records, n, useThreadedCondensation=True)
    K = _randomSparseK(n, seed=3)

    os.environ["EDELWEISS_MPC_ASSERT_EXACT"] = "1"
    try:
        mpc.transformSystemMatrix(K)  # correct implementation -- must not raise
    finally:
        del os.environ["EDELWEISS_MPC_ASSERT_EXACT"]


def test_default_uses_plain_expression():
    mpc = MultiPointConstraintTransformation([], 10)
    assert mpc._useThreadedCondensation is False
