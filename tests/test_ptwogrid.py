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
"""Tests for the C++ p-two-grid preconditioner (:mod:`edelweissfe.linsolve.blockamg.ptwogrid`,
``PTwoGridPreconditionerT`` in ``amgcl-wrapper.hpp``).

The system is a small Hex20-like lattice: corner nodes on a cubic grid, one exclusive midside node on
every grid edge, a vector-Laplacian-like 3x3-coupled operator with a mild non-symmetric part, fully
clamped nodes on one face and a single constrained component on another (partially constrained
nodes), symmetric Dirichlet elimination and diagonal equilibration -- the shape of the block blockamg
hands to the preconditioner. The reference is the former pure-Python cycle on the explicitly
extracted free submatrix, with the fine spectral radius taken from the C++ build.
"""

import numpy as np
import pytest
import scipy.sparse as sp
from scipy.spatial import cKDTree

pytest.importorskip("edelweissfe.linsolve.amgcl.amgcl")

from edelweissfe.linsolve.amgcl.amgcl import PyAMGCLSolver  # noqa: E402
from edelweissfe.linsolve.base import FieldBlock  # noqa: E402
from edelweissfe.linsolve.blockamg.ptwogrid import (  # noqa: E402
    _DEFAULT_COARSE_PRECOND,
    PTwoGridPreconditioner,
)
from edelweissfe.linsolve.nullspace import rigidBodyNullspace  # noqa: E402


def _lattice(nCornersPerAxis: int, seed: int = 0):
    """Corner and midside nodes of a cubic lattice, in a shuffled node order."""
    axis = np.arange(nCornersPerAxis, dtype=float)
    corners = np.array(np.meshgrid(axis, axis, axis, indexing="ij")).reshape(3, -1).T
    cornerIndex = {tuple(c): i for i, c in enumerate(corners)}
    midsides, endpoints = [], []
    for i, corner in enumerate(corners):
        for direction in range(3):
            neighbour = corner.copy()
            neighbour[direction] += 1.0
            j = cornerIndex.get(tuple(neighbour))
            if j is not None:
                midsides.append(0.5 * (corner + neighbour))
                endpoints.append((i, j))
    coords = np.vstack([corners, midsides])
    isCorner = np.r_[np.ones(len(corners), dtype=bool), np.zeros(len(midsides), dtype=bool)]
    edgeEndpoints = np.vstack([-np.ones((len(corners), 2), dtype=int), np.array(endpoints)])

    order = np.random.default_rng(seed).permutation(len(coords))
    newIndexOfOld = np.empty_like(order)
    newIndexOfOld[order] = np.arange(len(order))
    edgeEndpoints = edgeEndpoints[order]
    edgeEndpoints[~isCorner[order]] = newIndexOfOld[edgeEndpoints[~isCorner[order]]]
    return coords[order], isCorner[order], edgeEndpoints


def _system(coords, dirichletDofs, seed: int = 1):
    """The equilibrated, Dirichlet-eliminated block and its equilibration vector."""
    nNodes = len(coords)
    pairs = cKDTree(coords).query_pairs(1.0 + 1e-9, output_type="ndarray")
    W = sp.coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(nNodes, nNodes))
    W = (W + W.T).tocsr()
    laplacian = sp.diags(np.asarray(W.sum(axis=1)).ravel() + 0.05) - W
    coupling = np.array([[2.0, 0.3, 0.1], [0.3, 1.5, 0.2], [0.1, 0.2, 1.8]])
    skew = np.random.default_rng(seed).uniform(-1.0, 1.0, (3, 3))
    A = sp.kron(laplacian, coupling) + 0.02 * sp.kron(sp.triu(W, 1) - sp.tril(W, -1), skew)
    A = A.tolil()
    A[dirichletDofs, :] = 0.0
    A[:, dirichletDofs] = 0.0
    A[dirichletDofs, dirichletDofs] = 1.0
    A = A.tocsr()
    A.eliminate_zeros()
    dinv = 1.0 / np.sqrt(np.abs(A.diagonal()))
    As = sp.diags(dinv) @ A @ sp.diags(dinv)
    As = As.tocsr()
    As.sort_indices()
    return As, dinv


def _dirichletDofs(coords):
    """All components on the x = 0 face, only the z component on the rest of the y = 0 face."""
    clamped = np.nonzero(coords[:, 0] == 0.0)[0]
    sliding = np.nonzero((coords[:, 1] == 0.0) & (coords[:, 0] > 0.0))[0]
    return np.sort(np.r_[(3 * clamped[:, None] + np.arange(3)).ravel(), 3 * sliding + 2])


@pytest.fixture(scope="module")
def problem():
    coords, isCorner, edgeEndpoints = _lattice(5)
    As, dinv = _system(coords, _dirichletDofs(coords))
    r = np.random.default_rng(2).standard_normal(As.shape[0])
    return As, dinv, coords, isCorner, edgeEndpoints, r


def _referenceCycle(As, dinv, coords, isCorner, edgeEndpoints, spectralRadius, nuPre, nuPost, degree, r):
    """The former pure-Python cycle, on the extracted free submatrix."""
    n, nNodes = As.shape[0], len(isCorner)
    nDim = n // nNodes
    isDirichlet = np.diff(As.indptr) == 1
    free = np.nonzero(~isDirichlet)[0]
    A_free = As[free, :][:, free].tocsr()

    cornerNodes = np.nonzero(isCorner)[0]
    cornerIndex = -np.ones(nNodes, dtype=int)
    cornerIndex[cornerNodes] = np.arange(len(cornerNodes))
    rows, cols, vals = [], [], []
    for node in range(nNodes):
        if isCorner[node]:
            rows, cols, vals = rows + [node], cols + [cornerIndex[node]], vals + [1.0]
        else:
            a, b = edgeEndpoints[node]
            rows, cols, vals = rows + [node, node], cols + [cornerIndex[a], cornerIndex[b]], vals + [0.5, 0.5]
    P_node = sp.csr_matrix((vals, (rows, cols)), shape=(nNodes, len(cornerNodes)))
    P_dof = sp.kron(P_node, sp.identity(nDim), format="csr")
    cornerDofs = np.repeat(cornerNodes, nDim) * nDim + np.tile(np.arange(nDim), len(cornerNodes))
    freeCoarse = np.nonzero(~isDirichlet[cornerDofs])[0]
    P_free = P_dof[free, :][:, freeCoarse].tocsr()

    coarse = PyAMGCLSolver({"precond": dict(_DEFAULT_COARSE_PRECOND), "backendBlockSize": 1})
    coarseBlock = FieldBlock("coarse", 0, len(cornerNodes) * nDim, nDim)
    coarse.set_nullspace(rigidBodyNullspace(coarseBlock, coords[cornerNodes], dinv[cornerDofs])[freeCoarse, :])
    coarse.build((P_free.T @ A_free @ P_free).tocsr())

    # amgcl::relaxation::chebyshev with its float lower/higher parameters
    lo = spectralRadius * float(np.float32(0.01))
    hi = spectralRadius * float(np.float32(1.1))
    d, c = 0.5 * (hi + lo), 0.5 * (hi - lo)

    def smooth(x, b):
        p = np.zeros_like(x)
        alpha = 0.0
        for k in range(degree):
            residual = b - A_free @ x
            if k == 0:
                alpha, beta = 1.0 / d, 0.0
            elif k == 1:
                alpha = 2 * d * (1.0 / (2 * d * d - c * c))
                beta = alpha * d - 1.0
            else:
                alpha = 1.0 / (d - 0.25 * alpha * c * c)
                beta = alpha * d - 1.0
            p = alpha * residual + beta * p
            x = x + p
        return x

    rFree = r[free]
    xFree = np.zeros_like(rFree)
    for _ in range(nuPre):
        xFree = smooth(xFree, rFree)
    xFree = xFree + P_free @ coarse.applyPreconditioner(P_free.T @ (rFree - A_free @ xFree))
    for _ in range(nuPost):
        xFree = smooth(xFree, rFree)
    x = r.copy()
    x[free] = xFree
    return x


def _relativeDifference(x, y):
    return np.linalg.norm(x - y) / np.linalg.norm(y)


@pytest.mark.parametrize("nuPre, nuPost, degree", [(1, 1, 5), (2, 2, 8), (0, 1, 16), (1, 0, 8), (0, 2, 4)])
def test_the_scalar_cycle_matches_the_free_submatrix_reference(problem, nuPre, nuPost, degree):
    As, dinv, coords, isCorner, edgeEndpoints, r = problem
    preconditioner = PTwoGridPreconditioner(isCorner, edgeEndpoints, nuPre=nuPre, nuPost=nuPost, fineDegree=degree)
    preconditioner.build(As, dinv, coords=coords)
    x = preconditioner.applyPreconditioner(r)
    reference = _referenceCycle(
        As, dinv, coords, isCorner, edgeEndpoints, preconditioner.fineSpectralRadius, nuPre, nuPost, degree, r
    )
    assert _relativeDifference(x, reference) < 1e-12


def test_the_spectral_radius_estimate_is_the_free_submatrix_one(problem):
    As, dinv, coords, isCorner, edgeEndpoints, r = problem
    preconditioner = PTwoGridPreconditioner(isCorner, edgeEndpoints, finePowerIters=2000)
    preconditioner.build(As, dinv, coords=coords)
    free = np.nonzero(np.diff(As.indptr) != 1)[0]
    A_free = As[free, :][:, free].toarray()
    exact = np.max(np.abs(np.linalg.eigvals(A_free)))
    assert abs(preconditioner.fineSpectralRadius - exact) < 1e-3 * exact


def test_dirichlet_rows_pass_through(problem):
    As, dinv, coords, isCorner, edgeEndpoints, r = problem
    preconditioner = PTwoGridPreconditioner(isCorner, edgeEndpoints)
    preconditioner.build(As, dinv, coords=coords)
    x = preconditioner.applyPreconditioner(r)
    isDirichlet = np.diff(As.indptr) == 1
    assert np.array_equal(x[isDirichlet], r[isDirichlet])


def test_block3_storage_matches_scalar_storage(problem):
    As, dinv, coords, isCorner, edgeEndpoints, r = problem
    scalar = PTwoGridPreconditioner(isCorner, edgeEndpoints, nu=2, fineDegree=8)
    scalar.build(As, dinv, coords=coords)
    block3 = PTwoGridPreconditioner(isCorner, edgeEndpoints, nu=2, fineDegree=8, fineBlockSize=3)
    block3.build(As, dinv, coords=coords)
    assert abs(block3.fineSpectralRadius - scalar.fineSpectralRadius) < 1e-12 * scalar.fineSpectralRadius
    assert _relativeDifference(block3.applyPreconditioner(r), scalar.applyPreconditioner(r)) < 1e-12


def test_a_reused_spectral_radius_reproduces_the_estimated_one(problem):
    As, dinv, coords, isCorner, edgeEndpoints, r = problem
    estimated = PTwoGridPreconditioner(isCorner, edgeEndpoints)
    estimated.build(As, dinv, coords=coords)
    again = PTwoGridPreconditioner(isCorner, edgeEndpoints)
    again.build(As, dinv, coords=coords)
    # deterministic: the same matrix gives the same estimate, bit for bit, whatever the thread count
    assert again.fineSpectralRadius == estimated.fineSpectralRadius

    reused = PTwoGridPreconditioner(isCorner, edgeEndpoints, fineSpectralRadius=estimated.fineSpectralRadius)
    reused.build(As, dinv, coords=coords)
    assert reused.fineSpectralRadius == estimated.fineSpectralRadius
    assert reused.timings()["setup: fine spectral radius"] < estimated.timings()["setup: fine spectral radius"]
    # the coarse AMGCL hierarchy is rebuilt, with its own (thread-order dependent) power iteration
    assert _relativeDifference(reused.applyPreconditioner(r), estimated.applyPreconditioner(r)) < 1e-12


def _reusing(estimated, isCorner, edgeEndpoints):
    return PTwoGridPreconditioner(
        isCorner,
        edgeEndpoints,
        fineSpectralRadius=estimated.fineSpectralRadius,
        fineDominantVector=estimated.fineDominantVector,
    )


def test_the_recheck_keeps_a_reused_spectral_radius_on_an_unchanged_operator(problem):
    As, dinv, coords, isCorner, edgeEndpoints, r = problem
    estimated = PTwoGridPreconditioner(isCorner, edgeEndpoints)
    estimated.build(As, dinv, coords=coords)
    assert estimated.fineSpectralRadiusSource == "estimated"
    reused = _reusing(estimated, isCorner, edgeEndpoints)
    reused.build(As, dinv, coords=coords)
    # the continued power iteration still creeps up a little, below the re-check's 1% tolerance
    assert reused.fineSpectralRadiusSource == "supplied"
    assert reused.fineSpectralRadius == estimated.fineSpectralRadius
    assert _relativeDifference(reused.applyPreconditioner(r), estimated.applyPreconditioner(r)) < 1e-12


def test_the_recheck_raises_a_reused_spectral_radius_within_the_margin(problem):
    As, dinv, coords, isCorner, edgeEndpoints, r = problem
    estimated = PTwoGridPreconditioner(isCorner, edgeEndpoints)
    estimated.build(As, dinv, coords=coords)
    reused = _reusing(estimated, isCorner, edgeEndpoints)
    reused.build((1.05 * As).tocsr(), dinv, coords=coords)
    assert reused.fineSpectralRadiusSource == "raised"
    assert abs(reused.fineSpectralRadius - 1.05 * estimated.fineSpectralRadius) < 1e-4 * estimated.fineSpectralRadius


def test_the_recheck_re_estimates_a_stiffened_operator_in_full(problem):
    As, dinv, coords, isCorner, edgeEndpoints, r = problem
    estimated = PTwoGridPreconditioner(isCorner, edgeEndpoints)
    estimated.build(As, dinv, coords=coords)
    stiffened = (1.5 * As).tocsr()
    fresh = PTwoGridPreconditioner(isCorner, edgeEndpoints)
    fresh.build(stiffened, dinv, coords=coords)
    reused = _reusing(estimated, isCorner, edgeEndpoints)
    reused.build(stiffened, dinv, coords=coords)
    assert reused.fineSpectralRadiusSource == "re-estimated"
    assert reused.fineSpectralRadius == fresh.fineSpectralRadius


def test_the_preconditioner_accelerates_gmres(problem):
    As, dinv, coords, isCorner, edgeEndpoints, r = problem
    from scipy.sparse.linalg import LinearOperator, gmres

    preconditioner = PTwoGridPreconditioner(isCorner, edgeEndpoints, fineBlockSize=3)
    preconditioner.build(As, dinv, coords=coords)
    M = LinearOperator(As.shape, matvec=preconditioner.applyPreconditioner)
    counts = {"preconditioned": 0, "plain": 0}
    for key, precond in (("preconditioned", M), ("plain", None)):

        def count(_, key=key):
            counts[key] += 1

        x, _ = gmres(As, r, M=precond, rtol=1e-6, restart=400, maxiter=1, callback=count, callback_type="pr_norm")
        assert np.linalg.norm(r - As @ x) < 1e-5 * np.linalg.norm(r)
    assert counts["preconditioned"] < counts["plain"] / 3


@pytest.mark.parametrize("fineBlockSize", [1, 3])
def test_a_float_fine_operator_stays_close_to_the_double_one(problem, fineBlockSize):
    As, dinv, coords, isCorner, edgeEndpoints, r = problem
    double = PTwoGridPreconditioner(isCorner, edgeEndpoints, nu=2, fineDegree=8, fineBlockSize=fineBlockSize)
    double.build(As, dinv, coords=coords)
    single = PTwoGridPreconditioner(
        isCorner, edgeEndpoints, nu=2, fineDegree=8, fineBlockSize=fineBlockSize, fineValueType="float"
    )
    single.build(As, dinv, coords=coords)
    # the spectral radius comes from the double operator in both
    assert single.fineSpectralRadius == double.fineSpectralRadius
    # only the stored entries are rounded (relative 6e-8); vectors and sums stay double
    assert _relativeDifference(single.applyPreconditioner(r), double.applyPreconditioner(r)) < 1e-5
    isDirichlet = np.diff(As.indptr) == 1
    assert np.array_equal(single.applyPreconditioner(r)[isDirichlet], r[isDirichlet])


def test_a_float_fine_operator_still_accelerates_gmres(problem):
    As, dinv, coords, isCorner, edgeEndpoints, r = problem
    from scipy.sparse.linalg import LinearOperator, gmres

    counts = {}
    for valueType in ("double", "float"):
        preconditioner = PTwoGridPreconditioner(isCorner, edgeEndpoints, fineBlockSize=3, fineValueType=valueType)
        preconditioner.build(As, dinv, coords=coords)
        M = LinearOperator(As.shape, matvec=preconditioner.applyPreconditioner)
        history = []
        x, _ = gmres(As, r, M=M, rtol=1e-6, restart=400, maxiter=1, callback=history.append, callback_type="pr_norm")
        assert np.linalg.norm(r - As @ x) < 1e-5 * np.linalg.norm(r)
        counts[valueType] = len(history)
    assert counts["float"] <= counts["double"] + 1


def test_a_float_coarse_level_stays_close_to_the_double_one(problem):
    As, dinv, coords, isCorner, edgeEndpoints, r = problem
    double = PTwoGridPreconditioner(isCorner, edgeEndpoints, fineBlockSize=3)
    double.build(As, dinv, coords=coords)
    single = PTwoGridPreconditioner(isCorner, edgeEndpoints, fineBlockSize=3, coarseValueType="float")
    single.build(As, dinv, coords=coords)
    # the coarse hierarchy itself is built in float, its own smoother's spectral radius included
    assert _relativeDifference(single.applyPreconditioner(r), double.applyPreconditioner(r)) < 1e-3


class _DirichletStub:
    """What ``applyDirichletToStiffness`` reads from a boundary condition."""

    def __init__(self, dofs):
        self.constrainedDofIndices = np.asarray(dofs, dtype=np.int64)


def _systemWithStoredZeros(coords, dirichletDofs):
    """The same block as :func:`_system`, but constrained by the solvers' own Dirichlet elimination,
    which keeps the eliminated entries as explicitly stored zeros, and equilibrated in place."""
    from edelweissfe.solvers.base.dirichlet import applyDirichletToStiffness

    K, _ = _system(coords, [])
    K = applyDirichletToStiffness(K, [_DirichletStub(dirichletDofs)], np.zeros(K.shape[0]))
    dinv = 1.0 / np.sqrt(np.abs(K.diagonal()))
    As = K.copy()
    As.data *= dinv[As.indices]
    As.data *= np.repeat(dinv, np.diff(As.indptr))
    return As, dinv


def test_dirichlet_rows_with_stored_zeros_are_recognized():
    coords, isCorner, edgeEndpoints = _lattice(5)
    dofs = _dirichletDofs(coords)
    withZeros, dinv = _systemWithStoredZeros(coords, dofs)
    assert np.any(withZeros.data == 0.0), "the Dirichlet elimination must leave stored zeros for this test"
    withoutZeros = withZeros.copy()
    withoutZeros.eliminate_zeros()

    from edelweissfe.linsolve.amgcl.amgcl import PyAMGCLPTwoGrid

    isDirichlet = PyAMGCLPTwoGrid.dirichletRows(withZeros)
    assert np.array_equal(np.nonzero(isDirichlet)[0], dofs)

    r = np.random.default_rng(2).standard_normal(withZeros.shape[0])
    results = []
    for A in (withZeros, withoutZeros):
        preconditioner = PTwoGridPreconditioner(isCorner, edgeEndpoints, fineBlockSize=3)
        preconditioner.build(A, dinv, coords=coords)
        results.append(preconditioner.applyPreconditioner(r))
    assert np.array_equal(results[0][dofs], r[dofs])
    assert _relativeDifference(results[0], results[1]) < 1e-12


def test_an_orphan_midside_raises():
    coords, isCorner, edgeEndpoints = _lattice(4)
    # an interior midside whose two edge-endpoint corners are fully clamped, while it stays free
    interior = np.all((coords > 0.0) & (coords < 3.0), axis=1)
    orphan = np.nonzero(~isCorner & interior)[0][0]
    clampedCorners = edgeEndpoints[orphan]
    dofs = np.sort((3 * clampedCorners[:, None] + np.arange(3)).ravel())
    As, dinv = _system(coords, dofs)
    preconditioner = PTwoGridPreconditioner(isCorner, edgeEndpoints)
    with pytest.raises(RuntimeError, match="orphan"):
        preconditioner.build(As, dinv, coords=coords)


def test_an_unsupported_block_size_or_value_type_raises():
    coords, isCorner, edgeEndpoints = _lattice(3)
    with pytest.raises(ValueError):
        PTwoGridPreconditioner(isCorner, edgeEndpoints, fineBlockSize=2)
    with pytest.raises(ValueError):
        PTwoGridPreconditioner(isCorner, edgeEndpoints, fineValueType="half")
    with pytest.raises(ValueError):
        PTwoGridPreconditioner(isCorner, edgeEndpoints, coarseValueType="half")
