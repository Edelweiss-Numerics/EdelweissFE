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
"""Tests for the C++ corner-node two-level cycle preconditioner
(:mod:`edelweissfe.linsolve.composition.cornernodetwolevelcycle`, ``CornerNodeTwoLevelCycleT`` in
``amgcl-wrapper.hpp``).

The system is a small Hex20-like lattice: corner nodes on a cubic grid, one exclusive midside node on
every grid edge, a vector-Laplacian-like 3x3-coupled operator with a mild non-symmetric part, fully
clamped nodes on one face and a single constrained component on another (partially constrained
nodes), symmetric Dirichlet elimination and diagonal scaling -- the shape of the block a ``DiagonalScaling`` hands to
the preconditioner. The coarse solve is exact (a sparse LU of the coarse operator), so that the cycle is compared, not
a coarse hierarchy. The reference is a pure-Python cycle on the explicitly extracted free submatrix, with the fine
spectral radius taken from the C++ build.
"""

import numpy as np
import pytest
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from _linsolvesystems import DirichletStub
from _p1lattice import (
    dirichletDofs,
    fineChebyshevTree,
    lattice,
    latticeSystem,
    orphanSystem,
)

pytest.importorskip("edelweissfe.linsolve.amgcl.amgcl")

from edelweissfe.linsolve.composition.cornernodetwolevelcycle import (  # noqa: E402
    CornerNodeTwoLevelCycle,
)


@pytest.fixture(scope="module")
def problem():
    coords, isCorner, edgeEndpoints = lattice(5)
    As, scaling = latticeSystem(coords, dirichletDofs(coords))
    r = np.random.default_rng(2).standard_normal(As.shape[0])
    return As, scaling, coords, isCorner, edgeEndpoints, r


class ExactCoarseSolve:
    """The exact coarse solve, :math:`A_c^{-1} r_c` by a sparse LU, counting its applications."""

    def __init__(self):
        self.solve = None
        self.applications = 0

    def setUp(self, A_c):
        self.solve = spla.factorized(A_c.tocsc()) if A_c.shape[0] > 0 else None

    def __call__(self, r_c):
        self.applications += 1
        return self.solve(r_c)


def _built(As, scaling, isCorner, edgeEndpoints, fineTree=None, **options):
    """The corner-node two-level cycle with the fine smoother ``fineTree`` (default :func:`fineChebyshevTree`) and an
    exact coarse solve, built for ``As``; returns it, its coarse solve and the coarse operator."""
    coarse = ExactCoarseSolve()
    preconditioner = CornerNodeTwoLevelCycle(
        isCorner, edgeEndpoints, fineTree if fineTree is not None else fineChebyshevTree(), **options
    )
    preconditioner.setCoarseSolve(coarse)
    A_c = preconditioner.build(As, scaling)
    coarse.setUp(A_c)
    return preconditioner, coarse, A_c


def _geometricProlongation(isCorner, edgeEndpoints, nDim):
    """P_geo: a corner node keeps its value, a midside node takes the mean of its two edge-endpoint corners."""
    nNodes = len(isCorner)
    cornerIndex = np.cumsum(isCorner) - 1
    rows, cols, vals = [], [], []
    for node in range(nNodes):
        corners = [node] if isCorner[node] else edgeEndpoints[node]
        for corner in corners:
            rows, cols, vals = rows + [node], cols + [cornerIndex[corner]], vals + [1.0 / len(corners)]
    P_node = sp.csr_matrix((vals, (rows, cols)), shape=(nNodes, int(np.count_nonzero(isCorner))))
    return sp.kron(P_node, sp.identity(nDim), format="csr")


def _cornerDofs(isCorner, nDim):
    cornerNodes = np.nonzero(isCorner)[0]
    return (cornerNodes[:, None] * nDim + np.arange(nDim)).ravel()


def _referenceCycle(As, scaling, isCorner, edgeEndpoints, spectralRadius, nuPre, nuPost, degree, r):
    """A pure-Python cycle on the extracted free submatrix, with P = D^-1 P_geo D_c and an exact coarse solve."""
    n, nNodes = As.shape[0], len(isCorner)
    nDim = n // nNodes
    isDirichlet = np.diff(As.indptr) == 1
    free = np.nonzero(~isDirichlet)[0]
    A_free = As[free, :][:, free].tocsr()

    cornerDofs = _cornerDofs(isCorner, nDim)
    P = sp.diags(1.0 / scaling) @ _geometricProlongation(isCorner, edgeEndpoints, nDim) @ sp.diags(scaling[cornerDofs])
    freeCoarse = np.nonzero(~isDirichlet[cornerDofs])[0]
    P_free = P.tocsr()[free, :][:, freeCoarse].tocsr()
    coarseSolve = spla.factorized((P_free.T @ A_free @ P_free).tocsc())

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
    xFree = xFree + P_free @ coarseSolve(P_free.T @ (rFree - A_free @ xFree))
    for _ in range(nuPost):
        xFree = smooth(xFree, rFree)
    x = r.copy()
    x[free] = xFree
    return x


def _relativeDifference(x, y):
    return np.linalg.norm(x - y) / np.linalg.norm(y)


@pytest.mark.parametrize("nuPre, nuPost, degree", [(1, 1, 5), (2, 2, 8), (0, 1, 16), (1, 0, 8), (0, 2, 4)])
def test_the_scalar_cycle_matches_the_free_submatrix_reference(problem, nuPre, nuPost, degree):
    As, scaling, coords, isCorner, edgeEndpoints, r = problem
    preconditioner, coarse, _ = _built(As, scaling, isCorner, edgeEndpoints, fineChebyshevTree(nuPre, nuPost, degree))
    x = preconditioner.applyPreconditioner(r)
    reference = _referenceCycle(
        As, scaling, isCorner, edgeEndpoints, preconditioner.fineSpectralRadius, nuPre, nuPost, degree, r
    )
    assert _relativeDifference(x, reference) < 1e-12
    # the coarse solve is called back once per cycle
    assert coarse.applications == 1


def test_the_prolongation_reproduces_the_scaled_translations(problem):
    """P = D^-1 P_geo D_c interpolates D^-1 t exactly at the free midside dofs whose two corners are free; the plain
    weights 1/2, 1/2 do not, wherever the scaling varies along an edge."""
    As, scaling, coords, isCorner, edgeEndpoints, r = problem
    preconditioner, _, _ = _built(As, scaling, isCorner, edgeEndpoints)
    P = preconditioner.prolongationMatrix()
    cornerDofs = _cornerDofs(isCorner, 3)
    isDirichlet = np.diff(As.indptr) == 1
    nodeOf, componentOf = np.arange(As.shape[0]) // 3, np.arange(As.shape[0]) % 3
    endpointsFree = np.ones(As.shape[0], dtype=bool)
    midside = ~isCorner[nodeOf]
    for k in range(2):
        endpointDofs = 3 * edgeEndpoints[nodeOf[midside], k] + componentOf[midside]
        endpointsFree[np.nonzero(midside)[0]] &= ~isDirichlet[endpointDofs]
    checked = ~isDirichlet & endpointsFree
    for component in range(3):
        translation = (componentOf == component).astype(float)
        scaledTranslation = translation / scaling
        interpolated = P @ scaledTranslation[cornerDofs]
        assert np.allclose(interpolated[checked], scaledTranslation[checked], rtol=1e-14, atol=0.0)
    plain = _geometricProlongation(isCorner, edgeEndpoints, 3)
    scaledTranslation = (componentOf == 0) / scaling
    assert not np.allclose((plain @ scaledTranslation[cornerDofs])[checked], scaledTranslation[checked], rtol=1e-3)


def test_the_coarse_operator_is_the_scaled_p1_operator(problem):
    """A_c = D_c P_geo^T A P_geo D_c on the free corner dofs, with an identity row for each Dirichlet corner dof."""
    As, scaling, coords, isCorner, edgeEndpoints, r = problem
    _, _, A_c = _built(As, scaling, isCorner, edgeEndpoints)
    isDirichlet = np.diff(As.indptr) == 1
    cornerDofs = _cornerDofs(isCorner, 3)
    A = sp.diags(1.0 / scaling) @ As @ sp.diags(1.0 / scaling)
    free = sp.diags((~isDirichlet).astype(float))
    P_geo = free @ _geometricProlongation(isCorner, edgeEndpoints, 3) @ sp.diags((~isDirichlet[cornerDofs]) * 1.0)
    D_c = sp.diags(scaling[cornerDofs])
    expected = (D_c @ P_geo.T @ A @ P_geo @ D_c).toarray()
    coarseDirichlet = isDirichlet[cornerDofs]
    expected[np.ix_(coarseDirichlet, coarseDirichlet)] = np.diag(np.ones(np.count_nonzero(coarseDirichlet)))
    assert A_c.shape == (len(cornerDofs), len(cornerDofs))
    assert np.abs(A_c.toarray() - expected).max() < 1e-13 * np.abs(expected).max()


def test_the_spectral_radius_estimate_is_the_free_submatrix_one(problem):
    As, scaling, coords, isCorner, edgeEndpoints, r = problem
    preconditioner, _, _ = _built(As, scaling, isCorner, edgeEndpoints, fineChebyshevTree(powerIters=2000))
    free = np.nonzero(np.diff(As.indptr) != 1)[0]
    A_free = As[free, :][:, free].toarray()
    exact = np.max(np.abs(np.linalg.eigvals(A_free)))
    assert abs(preconditioner.fineSpectralRadius - exact) < 1e-3 * exact


def test_dirichlet_rows_pass_through(problem):
    As, scaling, coords, isCorner, edgeEndpoints, r = problem
    preconditioner, _, _ = _built(As, scaling, isCorner, edgeEndpoints)
    x = preconditioner.applyPreconditioner(r)
    isDirichlet = np.diff(As.indptr) == 1
    assert np.array_equal(x[isDirichlet], r[isDirichlet])


def test_block3_storage_matches_scalar_storage(problem):
    As, scaling, coords, isCorner, edgeEndpoints, r = problem
    scalar, _, _ = _built(As, scaling, isCorner, edgeEndpoints, fineChebyshevTree(2, 2, 8))
    block3, _, _ = _built(As, scaling, isCorner, edgeEndpoints, fineChebyshevTree(2, 2, 8), backendBlockSize=3)
    assert abs(block3.fineSpectralRadius - scalar.fineSpectralRadius) < 1e-12 * scalar.fineSpectralRadius
    assert _relativeDifference(block3.applyPreconditioner(r), scalar.applyPreconditioner(r)) < 1e-12


def test_block2_storage_of_a_two_component_field_matches_scalar_storage():
    coords, isCorner, edgeEndpoints = lattice(5)
    clamped = np.nonzero(coords[:, 0] == 0.0)[0]
    As, scaling = latticeSystem(coords, np.sort((2 * clamped[:, None] + np.arange(2)).ravel()), components=2)
    r = np.random.default_rng(3).standard_normal(As.shape[0])
    scalar, _, _ = _built(As, scaling, isCorner, edgeEndpoints, fineChebyshevTree(2, 2, 8))
    block2, _, _ = _built(As, scaling, isCorner, edgeEndpoints, fineChebyshevTree(2, 2, 8), backendBlockSize=2)
    assert abs(block2.fineSpectralRadius - scalar.fineSpectralRadius) < 1e-12 * scalar.fineSpectralRadius
    assert _relativeDifference(block2.applyPreconditioner(r), scalar.applyPreconditioner(r)) < 1e-12


def test_a_reused_spectral_radius_reproduces_the_estimated_one(problem):
    As, scaling, coords, isCorner, edgeEndpoints, r = problem
    estimated, _, _ = _built(As, scaling, isCorner, edgeEndpoints)
    again, _, _ = _built(As, scaling, isCorner, edgeEndpoints)
    # deterministic: the same matrix gives the same estimate, bit for bit, whatever the thread count
    assert again.fineSpectralRadius == estimated.fineSpectralRadius

    reused, _, _ = _built(As, scaling, isCorner, edgeEndpoints, fineSpectralRadius=estimated.fineSpectralRadius)
    assert reused.fineSpectralRadius == estimated.fineSpectralRadius
    assert _relativeDifference(reused.applyPreconditioner(r), estimated.applyPreconditioner(r)) < 1e-12


def _reusing(estimated, As, scaling, isCorner, edgeEndpoints):
    reused, _, _ = _built(
        As,
        scaling,
        isCorner,
        edgeEndpoints,
        fineSpectralRadius=estimated.fineSpectralRadius,
        fineDominantVector=estimated.fineDominantVector,
    )
    return reused


def test_the_recheck_keeps_a_reused_spectral_radius_on_an_unchanged_operator(problem):
    As, scaling, coords, isCorner, edgeEndpoints, r = problem
    estimated, _, _ = _built(As, scaling, isCorner, edgeEndpoints)
    assert estimated.fineSpectralRadiusSource == "estimated"
    reused = _reusing(estimated, As, scaling, isCorner, edgeEndpoints)
    # the continued power iteration still creeps up a little, below the re-check's 1% tolerance
    assert reused.fineSpectralRadiusSource == "supplied"
    assert reused.fineSpectralRadius == estimated.fineSpectralRadius
    assert _relativeDifference(reused.applyPreconditioner(r), estimated.applyPreconditioner(r)) < 1e-12


def test_the_recheck_raises_a_reused_spectral_radius_within_the_margin(problem):
    As, scaling, coords, isCorner, edgeEndpoints, r = problem
    estimated, _, _ = _built(As, scaling, isCorner, edgeEndpoints)
    reused = _reusing(estimated, (1.05 * As).tocsr(), scaling, isCorner, edgeEndpoints)
    assert reused.fineSpectralRadiusSource == "raised"
    assert abs(reused.fineSpectralRadius - 1.05 * estimated.fineSpectralRadius) < 1e-4 * estimated.fineSpectralRadius


def test_the_recheck_re_estimates_a_stiffened_operator_in_full(problem):
    As, scaling, coords, isCorner, edgeEndpoints, r = problem
    estimated, _, _ = _built(As, scaling, isCorner, edgeEndpoints)
    stiffened = (1.5 * As).tocsr()
    fresh, _, _ = _built(stiffened, scaling, isCorner, edgeEndpoints)
    reused = _reusing(estimated, stiffened, scaling, isCorner, edgeEndpoints)
    assert reused.fineSpectralRadiusSource == "re-estimated"
    assert reused.fineSpectralRadius == fresh.fineSpectralRadius


def _gmresIterations(As, r, applyPreconditioner):
    from scipy.sparse.linalg import LinearOperator, gmres

    M = None if applyPreconditioner is None else LinearOperator(As.shape, matvec=applyPreconditioner)
    history = []
    x, _ = gmres(As, r, M=M, rtol=1e-6, restart=400, maxiter=1, callback=history.append, callback_type="pr_norm")
    assert np.linalg.norm(r - As @ x) < 1e-5 * np.linalg.norm(r)
    return len(history)


def test_the_preconditioner_accelerates_gmres(problem):
    As, scaling, coords, isCorner, edgeEndpoints, r = problem
    preconditioner, _, _ = _built(As, scaling, isCorner, edgeEndpoints, backendBlockSize=3)
    assert _gmresIterations(As, r, preconditioner.applyPreconditioner) < _gmresIterations(As, r, None) / 3


@pytest.mark.parametrize("fineBlockSize", [1, 3])
def test_a_float_fine_operator_stays_close_to_the_double_one(problem, fineBlockSize):
    As, scaling, coords, isCorner, edgeEndpoints, r = problem
    double, _, _ = _built(
        As, scaling, isCorner, edgeEndpoints, fineChebyshevTree(2, 2, 8), backendBlockSize=fineBlockSize
    )
    single, _, _ = _built(
        As,
        scaling,
        isCorner,
        edgeEndpoints,
        fineChebyshevTree(2, 2, 8),
        backendBlockSize=fineBlockSize,
        backendPrecision="float",
    )
    # the spectral radius comes from the double operator in both
    assert single.fineSpectralRadius == double.fineSpectralRadius
    # only the stored entries are rounded (relative 6e-8); vectors and sums stay double
    assert _relativeDifference(single.applyPreconditioner(r), double.applyPreconditioner(r)) < 1e-5
    isDirichlet = np.diff(As.indptr) == 1
    assert np.array_equal(single.applyPreconditioner(r)[isDirichlet], r[isDirichlet])


def test_a_float_fine_operator_still_accelerates_gmres(problem):
    As, scaling, coords, isCorner, edgeEndpoints, r = problem
    counts = {}
    for valueType in ("double", "float"):
        preconditioner, _, _ = _built(
            As, scaling, isCorner, edgeEndpoints, backendBlockSize=3, backendPrecision=valueType
        )
        counts[valueType] = _gmresIterations(As, r, preconditioner.applyPreconditioner)
    assert counts["float"] <= counts["double"] + 1


def test_an_exception_of_the_coarse_solve_is_raised_as_it_is(problem):
    As, scaling, coords, isCorner, edgeEndpoints, r = problem

    def failingCoarseSolve(r_c):
        raise ArithmeticError("coarse solve failed")

    preconditioner = CornerNodeTwoLevelCycle(isCorner, edgeEndpoints, fineChebyshevTree())
    preconditioner.setCoarseSolve(failingCoarseSolve)
    preconditioner.build(As, scaling)
    with pytest.raises(ArithmeticError, match="coarse solve failed"):
        preconditioner.applyPreconditioner(r)


def test_the_coarse_solve_set_last_is_the_one_called(problem):
    """A coarse solve set after the build replaces the one set before: the C++ cycle reads it at every application, so
    the replaced one (and its context, freed) is never called."""
    As, scaling, coords, isCorner, edgeEndpoints, r = problem
    first = ExactCoarseSolve()
    preconditioner = CornerNodeTwoLevelCycle(isCorner, edgeEndpoints, fineChebyshevTree())
    preconditioner.setCoarseSolve(first)
    A_c = preconditioner.build(As, scaling)
    first.setUp(A_c)
    second = ExactCoarseSolve()
    second.setUp(A_c)
    preconditioner.setCoarseSolve(second)
    del first
    z = preconditioner.applyPreconditioner(r)
    assert second.applications == 1
    assert np.all(np.isfinite(z))


def test_a_coarse_solve_may_be_set_after_the_build_but_must_be_set_before_the_first_application(problem):
    As, scaling, coords, isCorner, edgeEndpoints, r = problem
    preconditioner = CornerNodeTwoLevelCycle(isCorner, edgeEndpoints, fineChebyshevTree())
    A_c = preconditioner.build(As, scaling)
    with pytest.raises(RuntimeError, match="no coarse solve"):
        preconditioner.applyPreconditioner(r)
    coarse = ExactCoarseSolve()
    coarse.setUp(A_c)
    preconditioner.setCoarseSolve(coarse)
    assert np.all(np.isfinite(preconditioner.applyPreconditioner(r)))


def _systemWithStoredZeros(coords, constrainedDofs):
    """The same block as :func:`latticeSystem`, but constrained by the solvers' own Dirichlet elimination,
    which keeps the eliminated entries as explicitly stored zeros, and scaled in place."""
    from edelweissfe.solvers.base.dirichlet import applyDirichletToStiffness

    K, _ = latticeSystem(coords, [])
    K = applyDirichletToStiffness(K, [DirichletStub(constrainedDofs)], np.zeros(K.shape[0]))
    scaling = 1.0 / np.sqrt(np.abs(K.diagonal()))
    As = K.copy()
    As.data *= scaling[As.indices]
    As.data *= np.repeat(scaling, np.diff(As.indptr))
    return As, scaling


def test_dirichlet_rows_with_stored_zeros_are_recognized():
    coords, isCorner, edgeEndpoints = lattice(5)
    dofs = dirichletDofs(coords)
    withZeros, scaling = _systemWithStoredZeros(coords, dofs)
    assert np.any(withZeros.data == 0.0), "the Dirichlet elimination must leave stored zeros for this test"
    withoutZeros = withZeros.copy()
    withoutZeros.eliminate_zeros()

    from edelweissfe.linsolve.amgcl.amgcl import PyAMGCLCornerNodeTwoLevelCycle

    isDirichlet = PyAMGCLCornerNodeTwoLevelCycle.dirichletRows(withZeros)
    assert np.array_equal(np.nonzero(isDirichlet)[0], dofs)

    r = np.random.default_rng(2).standard_normal(withZeros.shape[0])
    results = []
    for A in (withZeros, withoutZeros):
        preconditioner, _, _ = _built(A, scaling, isCorner, edgeEndpoints, backendBlockSize=3)
        results.append(preconditioner.applyPreconditioner(r))
    assert np.array_equal(results[0][dofs], r[dofs])
    assert _relativeDifference(results[0], results[1]) < 1e-12


def test_a_free_midside_dof_between_two_constrained_corners_makes_its_node_a_coarse_node():
    """Issue #223: interpolated from its corners, the free dof would get an empty row of P; its node becomes a coarse
    node instead, and P still reproduces the scaled translations D^-1 t, at that dof too."""
    As, scaling, coords, isCorner, edgeEndpoints, orphan = orphanSystem()
    isDirichlet = np.diff(As.indptr) == 1
    assert not isDirichlet[3 * orphan + 2] and isDirichlet[3 * edgeEndpoints[orphan] + 2].all()
    preconditioner, coarse, A_c = _built(As, scaling, isCorner, edgeEndpoints)
    coarseNodes = preconditioner.coarseNodes
    assert np.array_equal(coarseNodes, isCorner | (np.arange(len(isCorner)) == orphan))
    coarseDofs = _cornerDofs(coarseNodes, 3)
    assert A_c.shape == (len(coarseDofs), len(coarseDofs)) == (preconditioner.nCoarse,) * 2
    # the coarse dof of the orphan's z component is free, those of its constrained components are identity rows
    orphanCoarseDofs = 3 * np.count_nonzero(coarseNodes[:orphan]) + np.arange(3)
    identityRows = np.diff(A_c.indptr) == 1
    assert np.array_equal(identityRows[orphanCoarseDofs], isDirichlet[3 * orphan + np.arange(3)])
    P = preconditioner.prolongationMatrix()
    componentOf = np.arange(As.shape[0]) % 3
    free = ~isDirichlet
    assert P[3 * orphan + 2].nnz == 1
    for component in range(3):
        scaledTranslation = (componentOf == component) / scaling
        interpolated = P @ scaledTranslation[coarseDofs]
        nodeOf = np.arange(As.shape[0]) // 3
        midside = ~coarseNodes[nodeOf]
        endpointsFree = np.ones(As.shape[0], dtype=bool)
        for k in range(2):
            endpointsFree[midside] &= ~isDirichlet[3 * edgeEndpoints[nodeOf[midside], k] + componentOf[midside]]
        checked = free & endpointsFree
        assert checked[3 * orphan + 2]
        assert np.allclose(interpolated[checked], scaledTranslation[checked], rtol=1e-14, atol=0.0)
    r = np.random.default_rng(5).standard_normal(As.shape[0])
    assert _gmresIterations(As, r, preconditioner.applyPreconditioner) < _gmresIterations(As, r, None) / 3


def _unbuilt(isCorner, edgeEndpoints, fineTree=None, **options):
    preconditioner = CornerNodeTwoLevelCycle(
        isCorner, edgeEndpoints, fineTree if fineTree is not None else fineChebyshevTree(), **options
    )
    preconditioner.setCoarseSolve(ExactCoarseSolve())
    return preconditioner


def test_an_unsupported_block_size_or_value_type_raises():
    coords, isCorner, edgeEndpoints = lattice(3)
    with pytest.raises(ValueError):
        _unbuilt(isCorner, edgeEndpoints, backendBlockSize=4)
    with pytest.raises(ValueError):
        _unbuilt(isCorner, edgeEndpoints, backendBlockSize=2, backendPrecision="float")
    with pytest.raises(ValueError):
        _unbuilt(isCorner, edgeEndpoints, backendPrecision="half")


def test_an_invalid_smoother_raises_before_any_build():
    coords, isCorner, edgeEndpoints = lattice(3)
    with pytest.raises(ValueError, match="relax.degree"):
        _unbuilt(isCorner, edgeEndpoints, fineChebyshevTree(degree=0))
    with pytest.raises(ValueError, match="npre and npost"):
        _unbuilt(isCorner, edgeEndpoints, fineChebyshevTree(npre=-1))
    with pytest.raises(ValueError, match="relax.type must be chebyshev"):
        _unbuilt(isCorner, edgeEndpoints, {"relax": {"type": "spai0"}})
    # the C++ class checks them itself, in its constructor, in the spelling of the rigid-body coarse level
    from edelweissfe.linsolve.amgcl.amgcl import PyAMGCLCornerNodeTwoLevelCycle

    for amgclTree, parameters in (
        (fineChebyshevTree(degree=0), {}),
        (fineChebyshevTree(npost=-1), {}),
        (fineChebyshevTree(powerIters=0), {}),
        (fineChebyshevTree() | {"relax": fineChebyshevTree()["relax"] | {"higher": 0.9}}, {}),
        (fineChebyshevTree() | {"relax": fineChebyshevTree()["relax"] | {"lower": 0.0}}, {}),
        (fineChebyshevTree(), {"backendBlockSize": 4}),
    ):
        with pytest.raises(RuntimeError):
            PyAMGCLCornerNodeTwoLevelCycle({"amgcl": amgclTree, **parameters})


def test_an_unknown_parameter_raises_naming_the_valid_ones():
    from edelweissfe.linsolve.amgcl.amgcl import PyAMGCLCornerNodeTwoLevelCycle

    with pytest.raises(RuntimeError, match='unknown parameter "coarseHierarchyPrecision"; valid: amgcl, '):
        PyAMGCLCornerNodeTwoLevelCycle({"amgcl": fineChebyshevTree(), "coarseHierarchyPrecision": "double"})


def test_the_coarse_operator_is_held_by_the_coarse_solve_not_by_the_cycle(problem):
    """The corner-node cycle's coarse solve is a callable (a component of the composed solver), so the cycle returns
    A_c from build() and holds none to inspect."""
    As, scaling, coords, isCorner, edgeEndpoints, r = problem
    preconditioner, _, _ = _built(As, scaling, isCorner, edgeEndpoints)
    with pytest.raises(RuntimeError, match="holds the coarse operator"):
        preconditioner._preconditioner.coarseOperator()


def test_a_fully_constrained_field_is_the_identity():
    coords, isCorner, edgeEndpoints = lattice(3)
    n = 3 * len(coords)
    preconditioner, coarse, A_c = _built(sp.identity(n, format="csr"), np.ones(n), isCorner, edgeEndpoints)
    assert A_c.shape == (0, 0)
    r = np.random.default_rng(4).standard_normal(n)
    assert np.array_equal(preconditioner.applyPreconditioner(r), r)
    assert coarse.applications == 0
    assert "identity" in preconditioner.report()


def test_an_empty_field_builds_and_applies():
    preconditioner, _, _ = _built(
        sp.csr_matrix((0, 0)), np.zeros(0), np.zeros(0, dtype=bool), np.zeros((0, 2), dtype=int)
    )
    assert preconditioner.applyPreconditioner(np.zeros(0)).shape == (0,)


def test_a_zero_block_raises_instead_of_dividing_by_zero():
    coords, isCorner, edgeEndpoints = lattice(3)
    n = 3 * len(coords)
    with pytest.raises(RuntimeError, match="no nonzero diagonal"):
        _built(sp.csr_matrix((n, n)), np.ones(n), isCorner, edgeEndpoints)


def test_a_failed_build_leaves_nothing_to_apply():
    coords, isCorner, edgeEndpoints = lattice(3)
    As, scaling = latticeSystem(coords, dirichletDofs(coords))
    # the fine spectral radius is checked after the prolongation and the Galerkin product are built: a supplied
    # infinite one fails the build that late
    preconditioner = _unbuilt(isCorner, edgeEndpoints, fineSpectralRadius=np.inf)
    with pytest.raises(RuntimeError, match="spectral radius"):
        preconditioner.build(As, scaling)
    with pytest.raises(RuntimeError, match="no preconditioner built"):
        preconditioner.applyPreconditioner(np.ones(As.shape[0]))


def test_dirichlet_rows_are_solved_exactly_whatever_their_diagonal(problem):
    As, scaling, coords, isCorner, edgeEndpoints, r = problem
    dofs = dirichletDofs(coords)
    withDiagonalTwo = As.tolil()
    withDiagonalTwo[dofs, dofs] = 2.0
    preconditioner, _, _ = _built(withDiagonalTwo.tocsr(), scaling, isCorner, edgeEndpoints)
    assert np.array_equal(preconditioner.applyPreconditioner(r)[dofs], r[dofs] / 2.0)
