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
#  This file is part of EdelweissFE.
#
#  This library is free software; you can redistribute it and/or
#  modify it under the terms of the GNU Lesser General Public
#  License as published by the Free Software Foundation; either
#  version 2.1 of the License, or (at your option) any later version.
#
#  The full text of the license can be found in the file LICENSE.md at
#  the top level directory of EdelweissFE.
#  ---------------------------------------------------------------------
"""Tests for the symmetric Dirichlet elimination in applyDirichletToStiffness.

Given the right-hand side(s), the constrained DOFs are eliminated from the columns as well as the
rows: same solution as row replacement alone, a symmetric K stays symmetric, the sparsity pattern is
kept, and the right-hand side is updated in place whatever its memory layout -- the arc-length solver
passes its two right-hand sides as the columns of a transposed array.
"""

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve

from edelweissfe.solvers.base.dirichlet import applyDirichletToStiffness


class _Dirichlet:
    def __init__(self, indices):
        self.constrainedDofIndices = np.asarray(indices, dtype=np.int64)


def _system(n=30, seed=0):
    """A symmetric positive definite 1D Laplacian-plus-mass stiffness, in CSR with int32 indices."""
    rng = np.random.default_rng(seed)
    off = -0.5 - 0.5 * rng.random(n - 1)
    K = sp.diags([off, 2.0 + rng.random(n), off], [-1, 0, 1], format="csr")
    K.indices = K.indices.astype(np.int32)
    K.indptr = K.indptr.astype(np.int32)
    return K, rng


def _prescribe(R, constrained, values):
    R = R.copy()
    R[constrained] = values
    return R


dirichlets = [_Dirichlet([0, 7]), _Dirichlet([8, 29])]
constrained = np.array([0, 7, 8, 29])


def test_symmetric_elimination_gives_the_row_replacement_solution():
    K, rng = _system()
    R = _prescribe(rng.standard_normal(30), constrained, rng.standard_normal(4))
    reference = spsolve(applyDirichletToStiffness(K.copy(), dirichlets).tocsc(), R)

    R_ = R.copy()
    K_ = applyDirichletToStiffness(K.copy(), dirichlets, R_)
    np.testing.assert_allclose(spsolve(K_.tocsc(), R_), reference, rtol=1e-12, atol=1e-14)
    np.testing.assert_array_equal(R_[constrained], R[constrained])


def test_symmetric_elimination_keeps_k_symmetric_and_its_pattern():
    K, rng = _system()
    R = _prescribe(rng.standard_normal(30), constrained, rng.standard_normal(4))
    rowsOnly = applyDirichletToStiffness(K.copy(), dirichlets)
    assert abs(rowsOnly - rowsOnly.T).max() > 0.0

    K_ = applyDirichletToStiffness(K.copy(), dirichlets, R.copy())
    assert abs(K_ - K_.T).max() == 0.0
    np.testing.assert_array_equal(K_.indices, K.indices)
    np.testing.assert_array_equal(K_.indptr, K.indptr)


def test_two_right_hand_sides_as_columns_of_a_transposed_array_are_updated_in_place():
    K, rng = _system()
    # the arc-length solver's layout: R_ = np.tile(vector, (2, 1)).T, its columns R_0 and R_f
    R_ = np.tile(np.zeros(30), (2, 1)).T
    R_0, R_f = R_[:, 0], R_[:, 1]
    R_0[:] = _prescribe(rng.standard_normal(30), constrained, rng.standard_normal(4))
    R_f[:] = _prescribe(rng.standard_normal(30), constrained, rng.standard_normal(4))
    references = [spsolve(applyDirichletToStiffness(K.copy(), dirichlets).tocsc(), R_[:, k].copy()) for k in (0, 1)]

    K_ = applyDirichletToStiffness(K.copy(), dirichlets, R_)
    for k in (0, 1):
        np.testing.assert_allclose(spsolve(K_.tocsc(), R_[:, k]), references[k], rtol=1e-12, atol=1e-14)
    assert R_0.base is R_.base and R_f.base is R_.base  # the views the solver keeps using


def test_zero_prescribed_increments_leave_the_right_hand_side_unchanged():
    K, rng = _system()
    R = _prescribe(rng.standard_normal(30), constrained, 0.0)
    R_ = R.copy()
    applyDirichletToStiffness(K.copy(), dirichlets, R_)
    np.testing.assert_array_equal(R_, R)


def test_without_dirichlets_nothing_changes():
    K, rng = _system()
    R = rng.standard_normal(30)
    R_ = R.copy()
    K_ = applyDirichletToStiffness(K.copy(), [], R_)
    np.testing.assert_array_equal(K_.data, K.data)
    np.testing.assert_array_equal(R_, R)
