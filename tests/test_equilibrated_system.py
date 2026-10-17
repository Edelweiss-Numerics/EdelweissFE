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
