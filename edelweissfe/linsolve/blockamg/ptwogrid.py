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
"""p-multigrid (Galerkin P1 corner-node) preconditioner for one field's diagonal block inside
:class:`~edelweissfe.linsolve.blockamg.blockamg.BlockAMGSolver`'s per-field sweep.

Precondition the quadratic serendipity operator through a low-order P1 operator: :math:`\\nu`
Chebyshev sweeps on the field's own (equilibrated) block, restrict the residual through :math:`P^T`,
one AMGCL V-cycle on the Galerkin-projected :math:`A_1 = P^T A P`, prolong through :math:`P`,
:math:`\\nu` more Chebyshev sweeps. :func:`build` and :meth:`PTwoGridPreconditioner.applyPreconditioner`
match :class:`~edelweissfe.linsolve.amgcl.amgcl.PyAMGCLSolver`'s ``build``/``applyPreconditioner``
call shape closely enough to slot into the same per-field sweep, so
:class:`~edelweissfe.linsolve.blockamg.blockamg.BlockAMGSolver` needs no change beyond choosing
which class to build (see ``blockamg.py``'s ``p1Maps`` option).

This is an **opt-in, experimental variant of the field-split AMG preconditioner**, not the shipped
default. The underlying two-grid algorithm converges (fewer outer GMRES iterations than the
single-level default is common), but it carries two structural costs the single-level default does
not: a fixed per-solve setup cost (the Galerkin projection ``P^T A P`` and the coarse hierarchy build
happen fresh every solve, since the operator's sparsity pattern generally changes every Newton
iteration on a model with contact/tie constraints, leaving nothing stable to cache), and a
near-null-space handling gap -- the coarse solve can be given the same rigid-body near-null-space
treatment the single-level default uses (restricted to the corner-node subset), but the fine-level
Chebyshev smoother, unlike a full recursive AMG hierarchy, has no coarsening step of its own and
receives no near-null-space information at all. Whether the iteration-count win outweighs these
costs is problem-dependent; it has been measured to lose to the single-level default overall on at
least one real reference model in the regime tested, so it should not be assumed to help without
checking on the model at hand.

**Dirichlet handling is load-bearing, do not simplify away.** A field's diagonal block still carries
its Dirichlet identity rows (production applies Dirichlet elimination upstream of blockamg). Both
the fine smoothing and the Galerkin projection must operate on the *free* submatrix with Dirichlet
rows/columns removed entirely, not merely masked in place -- masking in place for the fine smoother
alone was tried and diverged outright (17-80x residual growth). The mechanism is not a genuine
non-symmetry in the physical operator: a raw, Dirichlet-row-included diagonal block looks roughly
50% non-symmetric by a simple ``‖B-Bᵀ‖/‖B‖`` measure, but that figure collapses to ~0.6% once the
Dirichlet rows are properly removed (rather than merely zeroed) -- a storage artifact of how a
Dirichlet-eliminated row is represented in a sparse matrix, not physics. A free midside node whose
edge-endpoint corner is Dirichlet-constrained keeps only the surviving ½-weight on its free endpoint
-- no renormalization (the constrained corner contributes exactly zero to a homogeneous Newton
correction, which the dropped weight already encodes).
"""

import numpy as np
import scipy.sparse as sp

from edelweissfe.linsolve.base import FieldBlock
from edelweissfe.linsolve.nullspace import rigidBodyNullspace, translationNullspace

#: Coarse-level Chebyshev degree/sweep-count configuration, from a config sweep across degree and
#: npre/npost on a real coupled system: halving the Chebyshev degree (8 -> 4) at npre=2/npost=2
#: matches a higher-degree hierarchy's outer-iteration count almost exactly while cutting the coarse
#: apply's own cost by ~7% -- npre=1/npost=1 variants were cheaper per call but pushed outer GMRES
#: iterations up enough to be a wash or a regression (a weaker coarse correction is not free). A
#: direct PARDISO factorization of A1 was also tried and rejected: exact (residual ~1e-14) but no
#: cheaper per call (~70ms, matching this config) -- A1's ~71 nnz/row at 70k free coarse DOF fills in
#: heavily under a general (unsymmetric) LU, so an exact triangular solve costs about as much as the
#: approximate V-cycle here.
#: power_iters=300 (was 50): the coarse solve uses the identical AMGCL Chebyshev relaxation code path
#: as blockamg's own single-level default, just on the coarse P1 operator instead of the full field
#: block, and is subject to the identical thread-count-dependent spectral-radius-estimate issue (see
#: blockamg.py's own comment on ``power_iters`` for the mechanism) -- bumped for the same reason and
#: by the same amount, not independently re-tuned at this exact value for the coarse operator's own
#: (much smaller) size.
_DEFAULT_COARSE_PRECOND = {
    "coarsening": {"type": "smoothed_aggregation", "aggr": {"eps_strong": 0.01}},
    "relax": {"type": "chebyshev", "degree": 4, "power_iters": 300, "lower": 0.01},
    "npre": 2,
    "npost": 2,
}
#: R2's winning fine-smoother configuration: nu=1 sweeps, Chebyshev degree 5.
_DEFAULT_NU = 1
_DEFAULT_FINE_DEGREE = 5


def _buildChebyshevSmoother(A, degree, powerIters=300, lower=0.01, higher=1.1):
    """The fine smoother, backed by AMGCL's own OpenMP-threaded ``runtime::relaxation::wrapper``
    instead of a serial scipy/numpy polynomial -- a serial fine smoother was measured costing 81%+ of
    this preconditioner's own apply time on a real reference model, so it must run OpenMP-threaded
    like every other AMGCL kernel here to be worth using at all. The spectral radius (power
    iteration; a short/badly-under-converged estimate can make Chebyshev smoothing diverge outright)
    is computed inside AMGCL's own chebyshev constructor via its own algorithm, so it no longer needs
    a separate Python-side pass. ``higher`` defaults to 1.1, a safety margin above the estimated
    spectral radius -- AMGCL's own chebyshev defaults ``higher`` to 1.0, which would silently retune
    the fine smoother relative to what has been validated here.

    Also returns ``residual``, computed on this same object's cached backend matrix: a plain
    ``A @ x`` scipy CSR matvec is not OpenMP-threaded regardless of ``OMP_NUM_THREADS`` (scipy sparse
    matvec is single-threaded C code), and was measured costing ~30ms/call on a real ~190k-DOF free
    displacement block -- silently charged to ``coarseSeconds`` in
    :meth:`PTwoGridPreconditioner.applyPreconditioner`, since that is the timing block it shared,
    even though it has nothing to do with the coarse level."""
    from edelweissfe.linsolve.amgcl.amgcl import PyAMGCLRelaxationSmoother

    smoother = PyAMGCLRelaxationSmoother(
        {"type": "chebyshev", "degree": degree, "power_iters": powerIters, "lower": lower, "higher": higher}
    )
    smoother.build(A)

    def smooth(x, rhs, sweeps):
        for _ in range(sweeps):
            smoother.applyStep(x, rhs)
        return x

    return smooth, smoother.residual


def buildNodeLevelP(isCorner: np.ndarray, edgeEndpoints: np.ndarray) -> sp.csr_matrix:
    """The node-level P1 restriction operator: identity on corners, ½/½ on each exclusive midside
    from its two edge-endpoint corners (:func:`edelweissfe.numerics.p1topology.buildP1Map`'s output, in its own node order).
    """
    nNodes = len(isCorner)
    cornerNodeRows = np.nonzero(isCorner)[0]
    nCorners = len(cornerNodeRows)
    cornerLocalIdx = -np.ones(nNodes, dtype=int)
    cornerLocalIdx[cornerNodeRows] = np.arange(nCorners)

    rows, cols, vals = [], [], []
    for node in range(nNodes):
        if isCorner[node]:
            rows.append(node)
            cols.append(cornerLocalIdx[node])
            vals.append(1.0)
        else:
            a, b = edgeEndpoints[node]
            rows += [node, node]
            cols += [cornerLocalIdx[a], cornerLocalIdx[b]]
            vals += [0.5, 0.5]
    return sp.csr_matrix((vals, (rows, cols)), shape=(nNodes, nCorners))


class PTwoGridPreconditioner:
    """A p-two-grid preconditioner for one vector field's diagonal block, built once per solve
    (mirroring :class:`~edelweissfe.linsolve.amgcl.amgcl.PyAMGCLSolver`'s ``build`` /
    ``applyPreconditioner`` life cycle) and applied many times within the outer GMRES.

    Parameters
    ----------
    isCorner, edgeEndpoints
        This field's P1 topology map (:func:`edelweissfe.numerics.p1topology.buildP1Map`), in the
        field's own node order.
    nu
        Fine Chebyshev sweeps before *and* after the coarse-grid correction.
    fineDegree
        Fine Chebyshev polynomial degree.
    coarsePrecond
        AMGCL parameter tree for the coarse-level (:math:`A_1`) solve.
    useCoarseNullspace
        Whether to give the coarse AMGCL solver a rigid-body near null-space on the free-corner
        space -- measurably helps here, unlike on the full quadratic block, since the coarse level's
        own aggregation is what actually needs it. Built as the full rigid-body basis (translations +
        rotations) when :meth:`build` is given node coordinates, translations alone otherwise --
        mirroring :class:`~edelweissfe.linsolve.blockamg.blockamg.BlockAMGSolver`'s own
        ``useRigidBodyNullspace`` fallback behaviour for the full field, since this is the same
        construction restricted to the corner-node subset.
    """

    def __init__(
        self,
        isCorner: np.ndarray,
        edgeEndpoints: np.ndarray,
        nu: int = _DEFAULT_NU,
        fineDegree: int = _DEFAULT_FINE_DEGREE,
        coarsePrecond: dict = None,
        useCoarseNullspace: bool = True,
        pNode: "sp.csr_matrix | None" = None,
    ):
        self._isCorner = isCorner
        # ``pNode`` lets the caller supply an already-built restriction operator. It is a pure
        # function of (isCorner, edgeEndpoints) -- both fixed for a given mesh -- while this class is
        # reconstructed on every hierarchy refresh, i.e. potentially every Newton iteration on a
        # churning contact model. Rebuilding it there would repeat the same O(nNodes) Python loop for
        # an identical result; see BlockAMGSolver's own cache.
        self._P_node = buildNodeLevelP(isCorner, edgeEndpoints) if pNode is None else pNode
        self._nu = nu
        self._fineDegree = fineDegree
        self._coarsePrecond = dict(coarsePrecond) if coarsePrecond is not None else dict(_DEFAULT_COARSE_PRECOND)
        self._useCoarseNullspace = useCoarseNullspace
        # cumulative wall time, fine (Chebyshev smoother) vs. coarse (AMGCL, OpenMP-threaded) --
        # kept measured, not assumed, since the fine/coarse cost split determines whether a serial
        # fine smoother is acceptable or must be threaded (see _buildChebyshevSmoother above).
        self.fineSeconds = 0.0
        self.coarseSeconds = 0.0
        self.applyCalls = 0

    @property
    def pNode(self):
        """The node-level P1 restriction operator this preconditioner uses.

        Exposed so a caller that reconstructs this object per hierarchy refresh can cache the
        operator and hand it back via the ``pNode`` constructor argument -- it depends only on the
        mesh topology, not on the matrix.
        """
        return self._P_node

    def build(self, A: sp.csr_matrix, dinv: np.ndarray, coords: np.ndarray = None) -> None:
        """Build the free submatrix, the restricted ``P``, the Galerkin coarse operator, its AMGCL
        hierarchy, and the fine Chebyshev smoother -- everything :meth:`applyPreconditioner` needs.

        Parameters
        ----------
        A
            This field's equilibrated diagonal block (``As[slice, slice]``), still carrying its
            Dirichlet identity rows.
        dinv
            This field's own slice of the global equilibration vector (``dinv[block.start:block.stop]``),
            needed to scale the coarse near-null-space consistently with ``A``'s own scaling
            (:func:`~edelweissfe.linsolve.nullspace.translationNullspace`'s convention).
        coords
            This field's node coordinates, node-major, or ``None``. When given and
            ``useCoarseNullspace`` is set, the coarse solve's near null-space is the full rigid-body
            basis (translations + rotations) on the corner-node subset instead of translations alone.
        """
        from edelweissfe.linsolve.amgcl.amgcl import PyAMGCLSolver

        A = A.tocsr()
        n = A.shape[0]
        nNodes = len(self._isCorner)
        if n % nNodes != 0:
            raise ValueError(
                "ptwogrid: block size {:} is not a multiple of the topology map's node count {:} -- "
                "the map does not match this field.".format(n, nNodes)
            )
        nDim = n // nNodes
        P_dof = sp.kron(self._P_node, sp.identity(nDim), format="csr")

        dirichletMaskBool = np.diff(A.indptr) == 1
        self._dirichletRows = np.nonzero(dirichletMaskBool)[0]
        self._freeRows = np.nonzero(~dirichletMaskBool)[0]

        A_free = A[self._freeRows, :][:, self._freeRows].tocsr()

        cornerNodeRows = np.nonzero(self._isCorner)[0]
        nCorners = len(cornerNodeRows)
        fullDofRowsForCorners = np.repeat(cornerNodeRows, nDim) * nDim + np.tile(np.arange(nDim), nCorners)
        coarseColIsFree = ~dirichletMaskBool[fullDofRowsForCorners]
        freeCoarseCols = np.nonzero(coarseColIsFree)[0]

        # restrict P to free fine rows x free coarse columns -- slicing alone implements the "no
        # renormalization" interpolation rule (a free midside's 1/2 weight to a Dirichlet corner is
        # simply dropped, not redistributed).
        P_free = P_dof[self._freeRows, :][:, freeCoarseCols].tocsr()
        rowSums = np.asarray(np.abs(P_free).sum(axis=1)).flatten()
        orphanRows = np.nonzero(rowSums == 0)[0]
        if len(orphanRows):
            raise AssertionError(
                "ptwogrid: {:} free midside row(s) of the restricted P are entirely zero -- both "
                "edge-endpoint corners are Dirichlet-constrained, an orphan the topology map "
                "disagrees with the Dirichlet data on. First bad row: {:}.".format(len(orphanRows), orphanRows[0])
            )

        self._As_free = A_free
        self._P_free = P_free

        A1_free = (P_free.T @ A_free @ P_free).tocsr()

        coarseSolver = PyAMGCLSolver({"precond": self._coarsePrecond, "backendBlockSize": 1})
        if self._useCoarseNullspace:
            # Build the full (unrestricted) coarse-DOF near null-space first, then restrict to the
            # free coarse columns -- same two-step shape as the rest of blockamg's null-space
            # construction, just applied to the corner-node subset's own synthetic block instead of
            # the full field.
            coarseBlock = FieldBlock("coarse", 0, nCorners * nDim, nDim)
            fullCornerDinv = dinv[fullDofRowsForCorners]
            if coords is not None:
                cornerCoords = np.asarray(coords, dtype=float)[cornerNodeRows]
                fullNullspace = rigidBodyNullspace(coarseBlock, cornerCoords, fullCornerDinv)
            else:
                fullNullspace = translationNullspace(coarseBlock, fullCornerDinv)
            coarseSolver.set_nullspace(fullNullspace[freeCoarseCols, :])
        coarseSolver.build(A1_free)
        self._coarseSolver = coarseSolver

        self._smooth, self._residual = _buildChebyshevSmoother(A_free, self._fineDegree)

    def applyPreconditioner(self, r: np.ndarray) -> np.ndarray:
        """One two-grid V-cycle: pre-smooth, coarse-grid correction, post-smooth, on the free
        submatrix; Dirichlet rows pass through unchanged (``A[i, i] = 1`` there by construction, so
        the exact local solve is ``x[i] = r[i]``)."""
        import time

        self.applyCalls += 1
        rFree = r[self._freeRows]
        xFree = np.zeros_like(rFree)

        t0 = time.perf_counter()
        self._smooth(xFree, rFree, self._nu)
        self.fineSeconds += time.perf_counter() - t0

        t0 = time.perf_counter()
        res = self._residual(rFree, xFree)
        resCoarse = self._P_free.T @ res
        corrCoarse = self._coarseSolver.applyPreconditioner(resCoarse)
        xFree = xFree + self._P_free @ corrCoarse
        self.coarseSeconds += time.perf_counter() - t0

        t0 = time.perf_counter()
        self._smooth(xFree, rFree, self._nu)
        self.fineSeconds += time.perf_counter() - t0

        x = np.empty_like(r)
        x[self._freeRows] = xFree
        x[self._dirichletRows] = r[self._dirichletRows]
        return x

    def report(self) -> str:
        """The coarse level's AMGCL hierarchy report (context/diagnostics only, not a convergence gate)."""
        return self._coarseSolver.report()
