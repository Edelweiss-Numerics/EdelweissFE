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
"""The implicit multi-point-constraint condensation fills the slave rows and columns of
:math:`T^T K T` with the identity and recovers the slaves after the solve
(:meth:`~edelweissfe.numerics.mpctransformation.MultiPointConstraintTransformation.recoverSlaveSolution`),
instead of writing the raw constraint equation :math:`u_s - \\sum_a N_a u_{m_a}` into the slave rows.

Pinned here: the solution is identical to the raw-constraint-row formulation (also with a
Dirichlet-prescribed master and a nonzero drift), the condensed operator is symmetric whenever K is,
and a small-valued K (a consistent mass matrix) stays well scaled under the symmetric diagonal
equilibration the iterative solvers apply.
"""

import numpy as np
import pytest
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve

from edelweissfe.numerics.mpctransformation import MultiPointConstraintTransformation

# hanging-node-like records: edge midpoints (0.5/0.5), a face center (4 x 0.25), an unequal
# interpolation, and a chain (master 9 is itself a slave). Master 0 is prescribed below.
_RECORDS = [
    (9, [(0, 0.5), (1, 0.5)]),
    (10, [(1, 0.25), (2, 0.25), (3, 0.25), (4, 0.25)]),
    (11, [(5, 0.3), (6, 0.7)]),
    (12, [(9, 0.5), (7, 0.5)]),
]
_PRESCRIBED_MASTER = 0


class _Dirichlet:
    def __init__(self, indices):
        self.constrainedDofIndices = np.asarray(indices, dtype=np.int64)


def _randomSPD(n, seed=0):
    """Symmetric and strictly diagonally dominant with a positive diagonal, hence SPD."""
    rng = np.random.default_rng(seed)
    B = sp.random(n, n, density=0.3, format="csr", random_state=rng, data_rvs=lambda k: rng.standard_normal(k))
    B = (B + B.T).tolil()
    B.setdiag(0.0)
    B = B.tocsr()
    return (B + sp.diags(np.asarray(abs(B).sum(axis=1)).ravel() + 1.0)).tocsr()


def _int32(K):
    K = K.tocsr()
    K.indices = K.indices.astype(np.int32)
    K.indptr = K.indptr.astype(np.int32)
    return K


def _rawConstraintRowOperator(mpc, K):
    """The former formulation: T^T K T plus the constraint equation in the slave rows,
    C_ss = 1, C_sm = -N_a."""
    nSlaves = len(mpc.slaveDofIndices)
    S = sp.csr_matrix((np.ones(nSlaves), (np.arange(nSlaves), mpc.slaveDofIndices)), shape=(nSlaves, mpc.nDof))
    E = sp.csr_matrix((np.ones(nSlaves), (mpc.slaveDofIndices, np.arange(nSlaves))), shape=(mpc.nDof, nSlaves))
    C = E @ (S - mpc._W)
    return (mpc._T.T @ K @ mpc._T + C).tocsr()


def _symmetricEquilibration(A):
    D = sp.diags(1.0 / np.sqrt(np.abs(A.diagonal())))
    return (D @ A @ D).tocsr()


def test_slave_rows_and_columns_are_the_identity():
    n = 20
    mpc = MultiPointConstraintTransformation(_RECORDS, n)
    Kt = mpc.transformSystemMatrix(_randomSPD(n)).toarray()

    s = mpc.slaveDofIndices
    np.testing.assert_array_equal(Kt[s][:, s], np.eye(len(s)))
    independent = np.setdiff1d(np.arange(n), s)
    assert not Kt[s][:, independent].any()
    assert not Kt[independent][:, s].any()


def test_solution_identical_to_raw_constraint_rows_with_prescribed_master():
    dirichletModule = pytest.importorskip("edelweissfe.solvers.base.dirichlet")
    n = 20
    mpc = MultiPointConstraintTransformation(_RECORDS, n)
    K = _randomSPD(n, seed=1)
    rng = np.random.default_rng(2)
    R = rng.standard_normal(n)
    dU = rng.standard_normal(n)  # deliberately inconsistent: nonzero drift at the slaves
    prescribedIncrement = 0.37
    dirichlets = [_Dirichlet([_PRESCRIBED_MASTER])]

    def solve(Kt):
        Rt = mpc.transformResidual(R, dU)
        Rt[_PRESCRIBED_MASTER] = prescribedIncrement
        Kt = dirichletModule.applyDirichletToStiffness(_int32(Kt), dirichlets, Rt)
        return spsolve(Kt.tocsc(), Rt)

    reference = solve(_rawConstraintRowOperator(mpc, K))
    x = mpc.recoverSlaveSolution(solve(mpc.transformSystemMatrix(K)))

    np.testing.assert_allclose(x, reference, rtol=1e-12, atol=1e-12)
    assert x[_PRESCRIBED_MASTER] == pytest.approx(prescribedIncrement, abs=1e-14)
    # the corrected increment satisfies every constraint exactly, drift included
    U = dU + x
    np.testing.assert_allclose(U[mpc.slaveDofIndices], mpc._W @ U, rtol=1e-12, atol=1e-12)


def test_recovery_handles_several_right_hand_sides():
    n = 20
    mpc = MultiPointConstraintTransformation(_RECORDS, n)
    X = np.random.default_rng(3).standard_normal((n, 2))
    expected = X.copy()
    for k in range(2):
        expected[mpc.slaveDofIndices, k] += mpc._W @ X[:, k]
    np.testing.assert_allclose(mpc.recoverSlaveSolution(X), expected, rtol=1e-14)


def test_condensed_operator_is_symmetric():
    n = 20
    mpc = MultiPointConstraintTransformation(_RECORDS, n)
    Kt = mpc.transformSystemMatrix(_randomSPD(n, seed=4))
    asymmetry = abs(Kt - Kt.T).max()
    assert asymmetry <= 1e-12 * abs(Kt).max()
    # the former raw-constraint-row operator was not
    Kraw = _rawConstraintRowOperator(mpc, _randomSPD(n, seed=4))
    assert abs(Kraw - Kraw.T).max() > 0.1


def test_mass_scaled_operator_stays_well_scaled_under_equilibration():
    """A consistent mass matrix has a diagonal of ~1e-6: after D A D with D = |diag|^(-1/2), the
    former slave-row couplings N_a would grow to ~N_a / sqrt(1e-6) ~ 1e3 -- the identity slave rows
    carry none, and an SPD operator equilibrates to entries bounded by 1."""
    n = 20
    mpc = MultiPointConstraintTransformation(_RECORDS, n)
    K = _randomSPD(n, seed=5)
    M = (K * (1e-6 / K.diagonal().max())).tocsr()  # diagonal <= 1e-6

    Mhat = _symmetricEquilibration(mpc.transformSystemMatrix(M))
    assert abs(Mhat).max() <= 1.0 + 1e-12
    np.testing.assert_allclose(Mhat.diagonal(), 1.0, rtol=1e-12)

    MhatRaw = _symmetricEquilibration(_rawConstraintRowOperator(mpc, M))
    assert abs(MhatRaw).max() > 50.0
