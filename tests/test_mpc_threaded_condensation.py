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
"""Correctness tests for the threaded, in-place T^T K T + C condensation
(``MultiPointConstraintTransformation(..., useThreadedCondensation=True)``, the kernel of
:mod:`edelweissfe.numerics.mpccondensation`): against the plain SciPy expression it is an opt-in
alternative to (default ``False``), and bitwise against a reference of its documented order of summation.
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


def _condensedThreaded(records, n, K):
    """The threaded condensation of (a copy of) K, given the fill-in it needs."""
    mpc = MultiPointConstraintTransformation(records, n, useThreadedCondensation=True)
    withFillIn = mpc.withCondensedPattern(K)
    return mpc.transformSystemMatrix(withFillIn.copy() if withFillIn is K else withFillIn)


def _referenceRows(mpc, K):
    """Row i of T^T (K T) + C as {column: value}, summed in the kernel's documented order: the inner sum over b in
    the stored order of row a of K, the outer sum over a in the stored order of row i of T^T, the first term of each
    sum assigned and the following ones added."""
    T, Tt, C = mpc._T, mpc._T.T.tocsr(), mpc._C

    def entries(A, i):
        return zip(A.indices[A.indptr[i] : A.indptr[i + 1]], A.data[A.indptr[i] : A.indptr[i + 1]])

    def add(row, c, v):
        row[c] = row[c] + v if c in row else v

    rows = []
    for i in range(K.shape[0]):
        row = {}
        tRow = list(entries(Tt, i))
        if len(tRow) == 1 and tRow[0][1] == 1.0:
            for b, kab in entries(K, tRow[0][0]):
                for c, tbc in entries(T, b):
                    add(row, c, kab * tbc)
        else:
            for a, w in tRow:
                kT = {}
                for b, kab in entries(K, a):
                    for c, tbc in entries(T, b):
                        add(kT, c, kab * tbc)
                for c, v in kT.items():
                    add(row, c, w * v)
        for c, v in entries(C, i):
            add(row, c, v)
        rows.append(row)
    return rows


@pytest.mark.parametrize("seed", range(3))
def test_bitwise_against_the_documented_order_of_summation(seed):
    n = 200
    K = _randomSparseK(n, density=0.05, seed=seed)
    records = _randomRecords(n, 40, seed)
    mpc = MultiPointConstraintTransformation(records, n, useThreadedCondensation=True)
    reference = _referenceRows(mpc, K)

    Kt = _condensedThreaded(records, n, K)
    for i in range(n):
        stored = dict(zip(Kt.indices[Kt.indptr[i] : Kt.indptr[i + 1]], Kt.data[Kt.indptr[i] : Kt.indptr[i + 1]]))
        assert set(reference[i]) <= set(stored)
        for c, v in stored.items():
            # bitwise, and entries outside the condensed pattern are stored zeros
            assert np.array_equal(v, reference[i].get(c, 0.0))


def test_condensed_in_place():
    """The threaded condensation overwrites K and returns it: the system matrix is held once."""
    n = 100
    records = [(5, [(0, 0.5), (1, 0.5)])]
    mpc = MultiPointConstraintTransformation(records, n, useThreadedCondensation=True)
    K = mpc.withCondensedPattern(_randomSparseK(n, seed=2))
    indices = K.indices.copy()
    assert mpc.transformSystemMatrix(K) is K
    assert np.array_equal(K.indices, indices)


def test_missing_fill_in_and_unsorted_rows_are_rejected():
    n = 100
    records = [(5, [(0, 0.5), (1, 0.5)])]
    mpc = MultiPointConstraintTransformation(records, n, useThreadedCondensation=True)
    K = _randomSparseK(n, density=0.05, seed=5)
    assert mpc.condensedPatternFillIn(K).nnz > 0
    with pytest.raises(RuntimeError, match="misses entries"):
        mpc.transformSystemMatrix(K.copy())

    K = mpc.withCondensedPattern(K)
    s, e = K.indptr[0], K.indptr[1]
    K.indices[s:e] = K.indices[s:e][::-1].copy()
    K.data[s:e] = K.data[s:e][::-1].copy()
    with pytest.raises(RuntimeError, match="not sorted"):
        mpc.transformSystemMatrix(K)


def test_plain_expression_needs_no_fill_in():
    n = 100
    mpc = MultiPointConstraintTransformation([(5, [(0, 0.5), (1, 0.5)])], n)
    K = _randomSparseK(n, seed=8)
    assert mpc.condensedPatternFillIn(K).nnz == 0
    assert mpc.withCondensedPattern(K) is K


def test_fill_in_added_to_the_csr_generator_pattern():
    """The CSR generator assembles the fill-in as stored zeros, and its matrix condenses in place."""
    from types import SimpleNamespace

    csrgeneratorv2 = pytest.importorskip("edelweissfe.numerics.csrgeneratorv2")
    n = 150
    K = _randomSparseK(n, density=0.05, seed=9).tocoo()
    # every entry split into two COO pairs, as overlapping elements give
    rows = np.concatenate([K.row, K.row]).astype(np.intc)
    columns = np.concatenate([K.col, K.col]).astype(np.intc)
    V = np.concatenate([0.25 * K.data, 0.75 * K.data])
    generator = csrgeneratorv2.CSRGenerator(SimpleNamespace(I=rows, J=columns, nDof=n))
    reference = generator.updateCSR(V)

    records = _randomRecords(n, 30, 9)
    mpc = MultiPointConstraintTransformation(records, n, useThreadedCondensation=True)
    fillIn = mpc.condensedPatternFillIn(generator.csrMatrix)
    generator.addToPattern(fillIn)
    assert generator.nnz == reference.nnz + fillIn.nnz
    assert mpc.condensedPatternFillIn(generator.csrMatrix).nnz == 0

    for _ in range(2):  # the values of each update are fresh, the fill-in assembled as zeros
        assembled = generator.updateInPlace(V)
        assert (assembled - reference).nnz == 0
        assert assembled.nnz == reference.nnz + fillIn.nnz
        Kt = mpc.transformSystemMatrix(assembled)
        assert _maxRelDiff(Kt, MultiPointConstraintTransformation(records, n).transformSystemMatrix(reference)) < 1e-14

    with pytest.raises(ValueError, match="already in the pattern"):
        generator.addToPattern(fillIn)


def test_a_matrix_obtained_before_add_to_pattern_keeps_its_pattern():
    from types import SimpleNamespace

    csrgeneratorv2 = pytest.importorskip("edelweissfe.numerics.csrgeneratorv2")
    n = 150
    K = _randomSparseK(n, density=0.05, seed=10).tocoo()
    generator = csrgeneratorv2.CSRGenerator(SimpleNamespace(I=K.row.astype(np.intc), J=K.col.astype(np.intc), nDof=n))
    old = generator.updateInPlace(K.data)
    reference = old.copy()
    mpc = MultiPointConstraintTransformation(_randomRecords(n, 30, 10), n, useThreadedCondensation=True)
    generator.addToPattern(mpc.condensedPatternFillIn(old))
    del generator
    _assertIdentical(old, reference)


def test_a_pattern_array_obtained_before_add_to_pattern_stays_valid_without_its_matrix():
    from types import SimpleNamespace

    csrgeneratorv2 = pytest.importorskip("edelweissfe.numerics.csrgeneratorv2")
    n = 150
    K = _randomSparseK(n, density=0.05, seed=11).tocoo()
    generator = csrgeneratorv2.CSRGenerator(SimpleNamespace(I=K.row.astype(np.intc), J=K.col.astype(np.intc), nDof=n))
    old = generator.updateInPlace(K.data)
    indptr, indices = old.indptr, old.indices
    referenceIndptr, referenceIndices = indptr.copy(), indices.copy()
    mpc = MultiPointConstraintTransformation(_randomRecords(n, 30, 11), n, useThreadedCondensation=True)
    generator.addToPattern(mpc.condensedPatternFillIn(old))
    del old, generator
    np.testing.assert_array_equal(indptr, referenceIndptr)
    np.testing.assert_array_equal(indices, referenceIndices)


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
    threaded = _condensedThreaded(records, n, K)
    assert _maxRelDiff(plain, threaded) < 1e-14
    assert threaded.has_sorted_indices


@pytest.mark.parametrize("seed", range(5))
def test_threaded_matches_plain_expression_random_records(seed):
    n = 300
    K = _randomSparseK(n, density=0.05, seed=seed)
    records = _randomRecords(n, 60, seed)

    plain = MultiPointConstraintTransformation(records, n).transformSystemMatrix(K)
    threaded = _condensedThreaded(records, n, K)
    assert _maxRelDiff(plain, threaded) < 1e-14


def test_result_independent_of_thread_count():
    n = 400
    mpc = MultiPointConstraintTransformation(_randomRecords(n, 80, 7), n, useThreadedCondensation=True)
    K = mpc.withCondensedPattern(_randomSparseK(n, density=0.05, seed=7))
    condensation = mpccondensation.MPCCondensation(mpc._T, mpc._T.T.tocsr(), mpc._C)

    reference = K.copy()
    condensation.condenseInPlace(reference, 1)
    for numThreads in (2, 3, 8):
        condensed = K.copy()
        condensation.condenseInPlace(condensed, numThreads)
        _assertIdentical(reference, condensed)


def test_repeated_condensation_reuses_the_kernel():
    """The kernel is built once per transformation; later calls (every Newton iteration) give the same result."""
    n = 300
    mpc = MultiPointConstraintTransformation(_randomRecords(n, 60, 11), n, useThreadedCondensation=True)
    K = mpc.withCondensedPattern(_randomSparseK(n, density=0.05, seed=11))
    first = mpc.transformSystemMatrix(K.copy())
    kernel = mpc._threadedCondensation
    _assertIdentical(first, mpc.transformSystemMatrix(K.copy()))
    assert mpc._threadedCondensation is kernel


@pytest.mark.parametrize("matrixFormat", ["csc", "bsr", "coo"])
def test_non_csr_matrices_are_rejected(matrixFormat):
    """A CSC or BSR matrix also has an indptr, and would be read as a different matrix."""
    n = 50
    mpc = MultiPointConstraintTransformation([(5, [(0, 0.5), (1, 0.5)])], n)
    condensation = mpccondensation.MPCCondensation(mpc._T, mpc._T.T.tocsr(), mpc._C)
    with pytest.raises(TypeError, match="must be CSR"):
        condensation.condenseInPlace(_randomSparseK(n, seed=12).asformat(matrixFormat))
    with pytest.raises(TypeError, match="must be CSR"):
        mpccondensation.MPCCondensation(mpc._T.tocsc(), mpc._T.T.tocsr(), mpc._C)


def test_64_bit_indices_are_rejected():
    """K is condensed through its own arrays, so they are checked, not converted."""
    n = 50
    mpc = MultiPointConstraintTransformation([(5, [(0, 0.5), (1, 0.5)])], n, useThreadedCondensation=True)
    K = mpc.withCondensedPattern(_randomSparseK(n, seed=13))
    K.indptr, K.indices = K.indptr.astype(np.int64), K.indices.astype(np.int64)
    with pytest.raises(TypeError, match="32-bit indices"):
        mpc.condensedPatternFillIn(K)
    with pytest.raises(TypeError, match="32-bit indices"):
        mpc.transformSystemMatrix(K)


def test_explicit_zeros_are_kept():
    """The pattern is the structural one: an exactly cancelling entry stays stored (pruning is the solver's
    decision, see pruneCondensedMatrixZeros)."""
    n = 4
    K = sp.csr_matrix(np.array([[2.0, 1.0, -1.0, 0.0], [1.0, 2.0, 0.0, 0.0], [-1.0, 0.0, 2.0, 0.0], [0, 0, 0, 1.0]]))
    # u_2 = u_1: K[0,1] + K[0,2] = 0 is stored in the condensed row 0
    Kt = _condensedThreaded([(2, [(1, 1.0)])], n, K)
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
    threaded = _condensedThreaded([], n, K)
    assert _maxRelDiff(plain, K) < 1e-9
    _assertIdentical(threaded, K)


def test_assert_exact_env_var_passes_on_a_correct_computation():
    n = 200
    records = [(5, [(0, 0.5), (1, 0.5)])]
    mpc = MultiPointConstraintTransformation(records, n, useThreadedCondensation=True)
    K = mpc.withCondensedPattern(_randomSparseK(n, seed=3))

    os.environ["EDELWEISS_MPC_ASSERT_EXACT"] = "1"
    try:
        mpc.transformSystemMatrix(K)  # correct implementation -- must not raise
    finally:
        del os.environ["EDELWEISS_MPC_ASSERT_EXACT"]


def test_default_uses_plain_expression():
    mpc = MultiPointConstraintTransformation([], 10)
    assert mpc._useThreadedCondensation is False
