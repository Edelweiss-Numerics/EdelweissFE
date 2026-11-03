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
"""Tests for blockamg's equilibrated, field-split system (``EquilibratedSystemT`` in
``amgcl-wrapper.hpp``, :class:`~edelweissfe.linsolve.amgcl.amgcl.PyAMGCLEquilibratedSystem`): the
one-pass C++ build must reproduce the SciPy construction it replaced -- ``As = D A D`` scaled in place,
row/column slices for the diagonal and coupling blocks -- bit for bit.
"""

import numpy as np
import pytest
import scipy.sparse as sp

pytest.importorskip("edelweissfe.linsolve.amgcl.amgcl")

from edelweissfe.linsolve.amgcl.amgcl import PyAMGCLEquilibratedSystem  # noqa: E402

#: Three fields, the last one not starting where the second one ends (a DOF range in no field).
_FIELDS = [(0, 120), (120, 170), (175, 230)]


@pytest.fixture(scope="module")
def system():
    rng = np.random.default_rng(3)
    n = 230
    A = sp.random(n, n, density=0.05, random_state=rng, format="csr") + sp.diags(rng.uniform(1.0, 5.0, n))
    A = A.tocsr()
    A.sort_indices()
    dinv = 1.0 / np.sqrt(np.abs(A.diagonal()))
    # the former construction in blockamg.py
    As = A.copy()
    As.data *= dinv[As.indices]
    As.data *= np.repeat(dinv, np.diff(As.indptr))
    equilibrated = PyAMGCLEquilibratedSystem()
    equilibrated.build(A, dinv, _FIELDS)
    return A, As, dinv, equilibrated


def test_the_absolute_diagonal_is_scipys(system):
    A, As, dinv, equilibrated = system
    assert np.array_equal(PyAMGCLEquilibratedSystem.absoluteDiagonal(A), np.abs(A.diagonal()))


def test_the_operator_is_the_scipy_one_bit_for_bit(system):
    # Bitwise equality holds because the C++ row sum uses the same operands in the same order as
    # SciPy's CSR product, with no fused multiply-add rounding difference between the two. That is
    # only guaranteed for an extension built without FMA-contracting architecture flags (setup.py's
    # default, with no -march/-mfma); a user build with such flags on an FMA-capable CPU can contract
    # the row sum's multiply-adds and round differently from SciPy's separate multiply and add.
    A, As, dinv, equilibrated = system
    x = np.random.default_rng(4).standard_normal(A.shape[0])
    rhs = np.random.default_rng(5).standard_normal(A.shape[0])
    # identical entries in identical order: every row sum comes out bitwise the same
    assert np.array_equal(equilibrated.matvec(x), As @ x)
    assert np.array_equal(equilibrated.residual(rhs, x), rhs - As @ x)


def test_the_diagonal_blocks_are_the_scipy_slices(system):
    A, As, dinv, equilibrated = system
    for field, (start, stop) in enumerate(_FIELDS):
        block = equilibrated.diagonalBlock(field)
        reference = As[start:stop, :][:, start:stop].tocsr()
        assert block.shape == reference.shape
        assert np.array_equal(block.indptr, reference.indptr)
        assert np.array_equal(block.indices, reference.indices)
        assert np.array_equal(block.data, reference.data)


def test_the_coupling_blocks_are_the_scipy_slices(system):
    A, As, dinv, equilibrated = system
    rng = np.random.default_rng(6)
    for field, (start, stop) in enumerate(_FIELDS):
        for otherField, (otherStart, otherStop) in enumerate(_FIELDS):
            if field == otherField:
                continue
            x = rng.standard_normal(otherStop - otherStart)
            reference = As[start:stop, :][:, otherStart:otherStop].tocsr() @ x
            assert np.array_equal(equilibrated.couplingMatvec(field, otherField, x), reference)


def test_overlapping_fields_raise(system):
    A, As, dinv, equilibrated = system
    with pytest.raises(RuntimeError, match="overlap"):
        PyAMGCLEquilibratedSystem().build(A, dinv, [(0, 100), (90, 200)])


def test_explicitly_stored_zeros_are_dropped_without_changing_any_product(system):
    A, As, dinv, equilibrated = system
    # the Dirichlet elimination's form: zeroed entries stay stored
    withZeros = A.copy()
    withZeros.data[::7] = 0.0
    reference = withZeros.copy()
    reference.eliminate_zeros()
    a, b = PyAMGCLEquilibratedSystem(), PyAMGCLEquilibratedSystem()
    a.build(withZeros, dinv, _FIELDS)
    b.build(reference, dinv, _FIELDS)
    x = np.random.default_rng(8).standard_normal(A.shape[0])
    assert np.array_equal(a.matvec(x), b.matvec(x))
    for field in range(len(_FIELDS)):
        block = a.diagonalBlock(field)
        assert not np.any(block.data == 0.0)
        assert np.array_equal(block.indices, b.diagonalBlock(field).indices)


def test_a_rejected_build_keeps_the_previous_system(system):
    A, As, dinv, equilibrated = system
    x = np.random.default_rng(5).standard_normal(A.shape[0])
    before = equilibrated.matvec(x)
    with pytest.raises(RuntimeError, match="overlap"):
        equilibrated.build(A, dinv, [(0, 100), (90, 200)])
    assert np.array_equal(equilibrated.matvec(x), before)


def test_a_vector_of_the_wrong_length_raises(system):
    A, As, dinv, equilibrated = system
    with pytest.raises(ValueError, match="length"):
        equilibrated.matvec(np.zeros(A.shape[0] + 1))
    with pytest.raises(ValueError, match="length"):
        equilibrated.residual(np.zeros(A.shape[0]), np.zeros(A.shape[0] - 3))


def test_a_failing_preconditioner_is_called_once_and_its_exception_raised(system):
    from edelweissfe.linsolve.amgcl.amgcl import PyAMGCLKrylovSolver

    A, As, dinv, equilibrated = system
    calls = []

    def failing(r):
        calls.append(1)
        raise RuntimeError("preconditioner failed")

    solver = PyAMGCLKrylovSolver(A.shape[0], "lgmres", {"M": 10, "K": 2})
    with pytest.raises(RuntimeError, match="preconditioner failed"):
        solver.solve(equilibrated, np.ones(A.shape[0]), failing, 1e-10, 50)
    assert len(calls) == 1

    # the failure ended the AMGCL call at once; the instance solves again
    x = solver.solve(equilibrated, np.ones(A.shape[0]), lambda r: r, 1e-8, 400)
    assert np.isfinite(x).all() and 0 < solver.lastIterations


def test_a_diagonal_block_product_is_the_scipy_one_bit_for_bit(system):
    A, As, dinv, equilibrated = system
    x = np.random.default_rng(6).standard_normal(A.shape[0])
    for field, (start, stop) in enumerate(_FIELDS):
        block = As[start:stop, start:stop]
        assert np.array_equal(equilibrated.diagonalBlockMatvec(field, x[start:stop]), block @ x[start:stop])


def test_the_outer_solver_rejects_a_right_hand_side_of_the_wrong_length(system):
    from edelweissfe.linsolve.amgcl.amgcl import PyAMGCLKrylovSolver

    A, As, dinv, equilibrated = system
    solver = PyAMGCLKrylovSolver(A.shape[0], "lgmres", {"M": 10})
    with pytest.raises(ValueError, match="length"):
        solver.solve(equilibrated, np.ones(A.shape[0] - 1), lambda r: r, 1e-8, 20)


def test_the_unscaled_operator_is_scipys_bit_for_bit(system):
    A, As, dinv, equilibrated = system
    unscaled = PyAMGCLEquilibratedSystem()
    unscaled.build(A, dinv, _FIELDS, unscaled=True)
    x = np.random.default_rng(7).standard_normal(A.shape[0])
    rhs = np.random.default_rng(8).standard_normal(A.shape[0])
    assert np.array_equal(unscaled.matvec(x, unscaled=True), A @ x)
    assert np.array_equal(unscaled.residual(rhs, x, unscaled=True), rhs - A @ x)
    # only the operator built for is stored
    with pytest.raises(RuntimeError, match="not kept"):
        equilibrated.matvec(x, unscaled=True)
    with pytest.raises(RuntimeError, match="not kept"):
        unscaled.matvec(x)


def test_the_blocks_are_the_same_whichever_operator_is_stored(system):
    # with A stored, the entries of As the field split reads are formed from A's by the expression As is assembled
    # with: bit for bit the same blocks
    A, As, dinv, equilibrated = system
    unscaled = PyAMGCLEquilibratedSystem()
    unscaled.build(A, dinv, _FIELDS, unscaled=True)
    rng = np.random.default_rng(10)
    for f, (start, stop) in enumerate(_FIELDS):
        expected, actual = equilibrated.diagonalBlock(f), unscaled.diagonalBlock(f)
        assert np.array_equal(expected.indptr, actual.indptr) and np.array_equal(expected.indices, actual.indices)
        assert np.array_equal(expected.data, actual.data)
        x = rng.standard_normal(stop - start)
        assert np.array_equal(unscaled.diagonalBlockMatvec(f, x), equilibrated.diagonalBlockMatvec(f, x))
        for g, (gStart, gStop) in enumerate(_FIELDS):
            if g != f:
                v = rng.standard_normal(gStop - gStart)
                assert np.array_equal(unscaled.couplingMatvec(f, g, v), equilibrated.couplingMatvec(f, g, v))


def test_a_tiny_right_hand_side_is_solved_like_any_other(system):
    # AMGCL's lgmres returns x = 0 for ||rhs|| < 4.4e-16; the wrapper solves for rhs scaled by a power of two,
    # which is exact, so the solution is the scaled one bit for bit
    from edelweissfe.linsolve.amgcl.amgcl import PyAMGCLKrylovSolver

    A, As, dinv, equilibrated = system
    solver = PyAMGCLKrylovSolver(A.shape[0], "lgmres", {"M": 30})
    rhs = np.random.default_rng(9).standard_normal(A.shape[0])
    x = solver.solve(equilibrated, rhs, lambda r: r, 1e-8, 200)
    tiny = solver.solve(equilibrated, np.ldexp(rhs, -80), lambda r: r, 1e-8, 200)
    assert np.any(x != 0.0) and np.array_equal(tiny, np.ldexp(x, -80))
