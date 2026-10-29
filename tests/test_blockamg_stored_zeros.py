#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""The blockamg linear solver on a system constrained by the solvers' own Dirichlet elimination, which keeps the eliminated
entries as explicitly stored zeros: the solve must be the one of the same system without them -- bitwise on one
thread; with several, AMGCL's own power iterations sum in thread order, so even two identical solves differ in
the last bits -- while the hierarchy-reuse pattern gate still sees the input's own nonzero count.
"""

import numpy as np
import scipy.sparse as sp
from _linsolvesystems import DirichletStub, displacementDamageSystem

from edelweissfe.config.linsolve import getLinSolverByName
from edelweissfe.solvers.base.dirichlet import applyDirichletToStiffness


def _system(n=14):
    K, b, blocks = displacementDamageSystem(n, coupling="symmetric")
    K.sort_indices()
    dofs = np.arange(0, 3 * n * n)  # the first face of the displacement field
    b[dofs] = 0.0
    K = applyDirichletToStiffness(K, [DirichletStub(dofs)], b)
    return K, b, blocks


def _solve(A, b, blocks):
    solver = getLinSolverByName("blockamg", {"tolerance": 1e-8, "verbosity": "silent"})
    solver.setFieldStructure(blocks)
    return solver(A, b), solver


def test_stored_zeros_change_nothing_but_the_pattern_gate_input():
    withZeros, b, blocks = _system()
    assert np.any(withZeros.data == 0.0), "the Dirichlet elimination must leave stored zeros for this test"
    withoutZeros = withZeros.copy()
    withoutZeros.eliminate_zeros()

    x, solver = _solve(withZeros, b, blocks)
    y, reference = _solve(withoutZeros, b, blocks)
    assert np.linalg.norm(x - y) <= 1e-12 * np.linalg.norm(y)
    assert solver.lastSolveSummary.iters == reference.lastSolveSummary.iters
    assert solver.lastNnz == withZeros.nnz


def test_a_non_canonical_matrix_is_solved_as_its_canonical_form():
    A, b, blocks = displacementDamageSystem(10, coupling="symmetric")
    # a duplicate of a_00 appended to row 0 (unsorted, too), and every seventh row's columns reversed
    indptr = A.indptr.copy()
    indptr[1:] += 1
    indices = np.insert(A.indices, A.indptr[1], 0)
    data = np.insert(A.data, A.indptr[1], 0.5 * A[0, 0])
    for i in range(7, A.shape[0], 7):
        indices[indptr[i] : indptr[i + 1]] = indices[indptr[i] : indptr[i + 1]][::-1]
        data[indptr[i] : indptr[i + 1]] = data[indptr[i] : indptr[i + 1]][::-1]
    nonCanonical = sp.csr_matrix((data, indices, indptr), shape=A.shape)
    assert not nonCanonical.has_canonical_format
    canonical = A.tolil()
    canonical[0, 0] += 0.5 * A[0, 0]
    canonical = canonical.tocsr()
    before = (nonCanonical.indptr.copy(), nonCanonical.indices.copy(), nonCanonical.data.copy())

    x, solver = _solve(nonCanonical, b, blocks)
    assert np.linalg.norm(b - canonical @ x) <= 1.0001e-8 * np.linalg.norm(b)
    # the caller's matrix is left as it was
    for array, original in zip((nonCanonical.indptr, nonCanonical.indices, nonCanonical.data), before):
        assert np.array_equal(array, original)
