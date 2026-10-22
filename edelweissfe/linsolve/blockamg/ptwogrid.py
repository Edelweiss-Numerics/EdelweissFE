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

The whole cycle -- and every stage of its build -- runs in C++ (``PTwoGridPreconditionerT`` in
``edelweissfe/linsolve/amgcl/amgcl-wrapper.hpp``, wrapped as
:class:`~edelweissfe.linsolve.amgcl.amgcl.PyAMGCLPTwoGrid`), OpenMP-threaded throughout; this module
only chooses the parameters and builds the coarse near null-space, which is cheap.

This is an **opt-in variant of the field-split AMG preconditioner**, not the default. It is
rebuilt exactly when blockamg's single-level hierarchies are, and its rebuild is cheaper than theirs;
the coarse solve always runs on the plain scalar backend, so it gets the same rigid-body
near-null-space treatment the single-level default gives a scalar-backend field (restricted to the
corner-node subset, independent of the field's own single-level backend -- a block backend, the
default for 2D/3D fields, takes the translations at most), while the fine-level Chebyshev smoother
needs none. Measured
against blockamg's default single-level configuration with the same outer settings, it was
slower on one reference model and level within noise on another -- worth measuring on a model
with expensive hierarchy rebuilds, not assumed to help.

**Dirichlet handling is load-bearing, do not simplify away.** A field's diagonal block still carries
its Dirichlet identity rows (production applies Dirichlet elimination upstream of blockamg), and
the identity rows of multi-point-constraint slaves, which are treated alike. Both
the fine smoothing and the Galerkin projection must act on the *free* DOFs only, with the Dirichlet
DOFs decoupled entirely, not merely masked in place -- masking in place for the fine smoother
alone was tried and diverged outright. The mechanism is not a genuine
non-symmetry in the physical operator: a raw, Dirichlet-row-included diagonal block looks substantially
non-symmetric by a simple ``‖B-Bᵀ‖/‖B‖`` measure, but that figure collapses to a small fraction of a
percent once the Dirichlet rows are properly removed (rather than merely zeroed) -- a storage artifact of how a
Dirichlet-eliminated row is represented in a sparse matrix, not physics. A free midside node whose
edge-endpoint corner is Dirichlet-constrained keeps only the surviving ½-weight on its free endpoint
-- no renormalization (the constrained corner contributes exactly zero to a homogeneous Newton
correction, which the dropped weight already encodes).

The C++ implementation does not extract the free submatrix, it decouples the Dirichlet DOFs exactly
inside the full block layout instead: the fine right-hand side and iterate are kept zero on them (a
Dirichlet row holds only its diagonal, so every residual and every Chebyshev update stays zero
there, whatever the columns hold), and ``P`` has no rows or columns for them. That is the
free-submatrix cycle up to floating-point summation order, and it is what allows a node-wise 3x3
block storage of the fine operator (``fineBlockSize``) on partially constrained nodes.
"""

import numpy as np

from edelweissfe.linsolve.base import FieldBlock
from edelweissfe.linsolve.nullspace import rigidBodyNullspace, translationNullspace
from edelweissfe.utils import performancetiming

#: Coarse-level Chebyshev degree/sweep-count configuration, from a config sweep across degree and
#: npre/npost on a real coupled system: halving the Chebyshev degree (8 -> 4) at npre=2/npost=2
#: matches a higher-degree hierarchy's outer-iteration count almost exactly while cutting the coarse
#: apply's own cost -- npre=1/npost=1 variants were cheaper per call but pushed outer GMRES
#: iterations up enough to be a wash or a regression (a weaker coarse correction is not free). A
#: direct PARDISO factorization of A1 was also tried and rejected: exact (residual ~1e-14) but no
#: cheaper per call -- A1 fills in heavily under a general (unsymmetric) LU at the problem sizes
#: tested, so an exact triangular solve costs about as much as the approximate V-cycle here.
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
#: The fine smoother's defaults, the best of a sweep over sweep counts and degrees on a coupled system: one sweep
#: of a degree-5 Chebyshev polynomial.
_DEFAULT_NU = 1
_DEFAULT_FINE_DEGREE = 5
#: The fine Chebyshev smoother's spectral bounds and power iteration: ``higher`` = 1.1 is a safety
#: margin above the estimated spectral radius (AMGCL's own chebyshev defaults it to 1.0, which would
#: silently retune the validated smoother); a short, badly under-converged power iteration can make
#: Chebyshev smoothing diverge outright, hence 300 iterations.
_DEFAULT_FINE_LOWER = 0.01
_DEFAULT_FINE_HIGHER = 1.1
_DEFAULT_FINE_POWER_ITERS = 300
#: Power iterations of the re-check of a reused fine spectral radius, from the previous build's dominant
#: vector -- a few sparse products instead of the full estimate's 300.
_DEFAULT_FINE_RECHECK_ITERS = 15


class PTwoGridPreconditioner:
    """A p-two-grid preconditioner for one vector field's diagonal block, built once per hierarchy
    refresh (mirroring :class:`~edelweissfe.linsolve.amgcl.amgcl.PyAMGCLSolver`'s ``build`` /
    ``applyPreconditioner`` life cycle) and applied many times within the outer GMRES.

    Parameters
    ----------
    isCorner, edgeEndpoints
        This field's P1 topology map (:func:`edelweissfe.numerics.p1topology.buildP1Map`), in the
        field's own node order.
    nu
        Fine Chebyshev sweeps before *and* after the coarse-grid correction.
    nuPre, nuPost
        Separate pre- and post-smoothing sweep counts, each ``None`` (default) for ``nu``. With
        ``nuPre = 0`` the cycle starts with the coarse-grid correction of the right-hand side itself
        (the residual of x = 0) and ends with ``nuPost`` sweeps: one smoother application fewer per
        cycle than a symmetric one, at the price of a nonsymmetric preconditioner -- of no concern to
        the right-preconditioned GMRES around it, which needs a fixed linear operator, not a symmetric
        one. Ending on the smoother is the right way round for that one sweep: it damps the
        high-frequency error the prolongated correction leaves on the fine level; starting with it
        (``nuPre = 1, nuPost = 0``) leaves that error in the result.
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
    fineBlockSize
        Storage of the fine operator for the smoother and residual sparse matrix-vector products:
        ``1`` (scalar entries) or ``3`` (node-wise 3x3 blocks, a 3D field only) -- roughly half the
        memory traffic per product, the dominant cost of the apply. Both give the same cycle up to
        floating-point summation order.
    fineSpectralRadius, fineDominantVector
        The fine operator's spectral radius for the Chebyshev bounds, or ``None`` to estimate it by
        ``finePowerIters`` power iterations from a deterministic start vector -- the single most
        expensive stage of :meth:`build`. A caller rebuilding on a slowly changing operator can hand
        back a previous build's :attr:`fineSpectralRadius` and :attr:`fineDominantVector` (see
        blockamg's ``p1FineSpectralRadiusReuse``). The radius is then not reused blindly but
        re-checked by ``fineRecheckIters`` power iterations continued from that vector: a re-check
        estimate more than 1% above the supplied radius replaces it, and one beyond the ``fineHigher`` safety
        margin -- the operator stiffened more than the smoother's bounds cover -- triggers the full
        estimate. A radius without a vector is used as supplied.
    fineValueType
        ``"double"`` (default) or ``"float"``: the value type the fine operator is stored in for the
        smoother and the residual -- fine smoothing dominates an outer iteration's cost, which is
        bound by the bytes per entry read. Vectors and every accumulation stay double (a float
        matrix times a double vector, summed in double), and the spectral-radius estimate and its
        re-check run on the double operator before it is rounded, so float only rounds the stored
        entries (relative 6e-8).
    coarseValueType
        ``"double"`` (default) or ``"float"``: the value type of the whole coarse AMGCL hierarchy
        (AMGCL's own single-precision backend, so its setup -- including its Chebyshev smoother's power
        iteration -- runs in float too); the coarse right-hand side and correction are converted at
        the boundary.
    fineLower, fineHigher, finePowerIters, fineRecheckIters
        The fine Chebyshev spectral bounds relative to the spectral radius, the power iteration
        count of its full estimate, and that of the re-check of a reused one.
    """

    def __init__(
        self,
        isCorner: np.ndarray,
        edgeEndpoints: np.ndarray,
        nu: int = _DEFAULT_NU,
        nuPre: int = None,
        nuPost: int = None,
        fineDegree: int = _DEFAULT_FINE_DEGREE,
        coarsePrecond: dict = None,
        useCoarseNullspace: bool = True,
        fineBlockSize: int = 1,
        fineValueType: str = "double",
        coarseValueType: str = "double",
        fineSpectralRadius: float = None,
        fineDominantVector: np.ndarray = None,
        fineLower: float = _DEFAULT_FINE_LOWER,
        fineHigher: float = _DEFAULT_FINE_HIGHER,
        finePowerIters: int = _DEFAULT_FINE_POWER_ITERS,
        fineRecheckIters: int = _DEFAULT_FINE_RECHECK_ITERS,
    ):
        self._isCorner = np.ascontiguousarray(isCorner, dtype=np.int8)
        self._edgeEndpoints = np.ascontiguousarray(edgeEndpoints, dtype=np.int32)
        self._useCoarseNullspace = useCoarseNullspace
        self._fineSpectralRadius = fineSpectralRadius
        self._fineDominantVector = fineDominantVector
        self._params = {
            "nuPre": int(nu if nuPre is None else nuPre),
            "nuPost": int(nu if nuPost is None else nuPost),
            "fineDegree": int(fineDegree),
            "fineLower": float(fineLower),
            "fineHigher": float(fineHigher),
            "finePowerIters": int(finePowerIters),
            "fineRecheckIters": int(fineRecheckIters),
            "fineBlockSize": int(fineBlockSize),
            "fineValueType": fineValueType,
            "coarseValueType": coarseValueType,
            "coarse": dict(coarsePrecond) if coarsePrecond is not None else dict(_DEFAULT_COARSE_PRECOND),
        }
        from edelweissfe.linsolve.amgcl.amgcl import PyAMGCLPTwoGrid

        # The C++ class parses and validates every parameter in its constructor -- the one place they are
        # checked -- so a bad value fails here, before anything is built.
        try:
            self._preconditioner = PyAMGCLPTwoGrid(self._params)
        except RuntimeError as error:
            raise ValueError(str(error)) from error

    @property
    def fineSpectralRadius(self) -> float:
        """The fine spectral radius the smoother was built with (estimated, or the one supplied).

        Exposed so a caller that reconstructs this object per hierarchy refresh can hand it back via
        the ``fineSpectralRadius`` constructor argument instead of paying for the estimate again.
        """
        return self._preconditioner.fineSpectralRadius

    @property
    def fineDominantVector(self) -> np.ndarray:
        """The estimate of the fine operator's dominant eigenvector, to hand back together with
        :attr:`fineSpectralRadius` for the re-check of a reused radius."""
        return self._preconditioner.fineDominantVector

    @property
    def fineSpectralRadiusSource(self) -> str:
        """``"estimated"``, ``"supplied"``, ``"raised"`` or ``"re-estimated"`` -- see the
        ``fineSpectralRadius`` parameter."""
        return self._preconditioner.fineSpectralRadiusSource

    def build(self, A, scaling: np.ndarray, coords: np.ndarray = None) -> None:
        """Build the restricted ``P``, the Galerkin coarse operator, its AMGCL hierarchy, and the fine
        Chebyshev smoother -- everything :meth:`applyPreconditioner` needs, all in C++.

        Parameters
        ----------
        A
            This field's equilibrated diagonal block (``As[slice, slice]``, scipy CSR), still carrying
            its Dirichlet identity rows.
        scaling
            This field's own slice of the global equilibration vector (``scaling[block.start:block.stop]``),
            needed to scale the coarse near-null-space consistently with ``A``'s own scaling
            (:func:`~edelweissfe.linsolve.nullspace.translationNullspace`'s convention).
        coords
            This field's node coordinates, node-major, or ``None``. When given and
            ``useCoarseNullspace`` is set, the coarse solve's near null-space is the full rigid-body
            basis (translations + rotations) on the corner-node subset instead of translations alone.
        """
        from edelweissfe.linsolve.amgcl.amgcl import PyAMGCLPTwoGrid

        A = A.tocsr()
        n = A.shape[0]
        nNodes = len(self._isCorner)
        if (nNodes == 0) != (n == 0) or (nNodes > 0 and n % nNodes != 0):
            raise ValueError(
                "ptwogrid: block size {:} is not a multiple of the topology map's node count {:} -- "
                "the map does not match this field.".format(n, nNodes)
            )
        nDim = n // nNodes if nNodes > 0 else 0

        with performancetiming.timeit("coarse nullspace"):
            nullspace = None
            if self._useCoarseNullspace and n > 0:
                # Build the full (unrestricted) coarse-DOF near null-space first, then restrict to the
                # free coarse DOFs -- same two-step shape as the rest of blockamg's null-space
                # construction, just applied to the corner-node subset's own synthetic block instead of
                # the full field. The coarse DOFs are the corners' DOFs in node order; the Dirichlet rows
                # are the C++ build's own (a nonzero diagonal and no nonzero off-diagonal entry).
                isDirichlet = PyAMGCLPTwoGrid.dirichletRows(A)
                cornerNodeRows = np.nonzero(self._isCorner)[0]
                nCorners = len(cornerNodeRows)
                fullDofRowsForCorners = np.repeat(cornerNodeRows, nDim) * nDim + np.tile(np.arange(nDim), nCorners)
                freeCoarseCols = np.nonzero(~isDirichlet[fullDofRowsForCorners])[0]
                coarseBlock = FieldBlock("coarse", 0, nCorners * nDim, nDim)
                cornerScaling = scaling[fullDofRowsForCorners]
                if coords is not None:
                    cornerCoords = np.asarray(coords, dtype=float)[cornerNodeRows]
                    fullNullspace = rigidBodyNullspace(coarseBlock, cornerCoords, cornerScaling)
                else:
                    fullNullspace = translationNullspace(coarseBlock, cornerScaling)
                nullspace = fullNullspace[freeCoarseCols, :]

        with performancetiming.timeit("C++ build"):
            # a build that throws leaves the C++ object unbuilt: applyPreconditioner then raises
            self._preconditioner.build(
                A,
                self._isCorner,
                self._edgeEndpoints,
                nullspace,
                self._fineSpectralRadius if self._fineSpectralRadius is not None else 0.0,
                self._fineDominantVector,
            )

    def applyPreconditioner(self, r: np.ndarray) -> np.ndarray:
        """One two-grid V-cycle: pre-smooth, coarse-grid correction, post-smooth, on the free DOFs;
        Dirichlet rows are solved exactly, ``x[i] = r[i] / A[i, i]`` (``r[i]`` itself for the unit diagonal of
        an equilibrated block)."""
        return self._preconditioner.applyPreconditioner(r)

    def timings(self) -> dict:
        """Wall time [s] of the last build's stages and the cumulative apply (fine vs. coarse) --
        kept measured, not assumed, since the split decides where the next optimization goes."""
        return self._preconditioner.timings()

    def report(self) -> str:
        """The coarse level's AMGCL hierarchy report (context/diagnostics only, not a convergence gate)."""
        return self._preconditioner.report()


def parameterRejection(**parameters) -> "str | None":
    """Why these p-two-grid settings -- :class:`PTwoGridPreconditioner` keywords -- cannot be used, or ``None``.
    The C++ class validates them in its constructor, the one place they are checked; this asks it, without a
    topology map."""
    try:
        PTwoGridPreconditioner(np.zeros(0, dtype=bool), np.zeros((0, 2), dtype=int), **parameters)
    except (TypeError, ValueError) as error:
        return str(error)
    return None
