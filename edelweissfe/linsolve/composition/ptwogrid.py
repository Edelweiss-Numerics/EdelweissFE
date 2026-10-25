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
"""p-multigrid (Galerkin P1 corner-node) preconditioner for one field's diagonal block: the cycle of
``TwoLevel(coarseSpace=P1Interpolation())`` (:class:`~edelweissfe.linsolve.composition.twolevel.TwoLevel`).

Precondition the quadratic serendipity operator through a low-order P1 operator: :math:`\\nu`
Chebyshev sweeps on the field's own (equilibrated) block, restrict the residual through :math:`P^T`,
one AMGCL V-cycle on the Galerkin-projected :math:`A_1 = P^T A P`, prolong through :math:`P`,
:math:`\\nu` more Chebyshev sweeps. :func:`build` and :meth:`PTwoGridPreconditioner.applyPreconditioner`
match :class:`~edelweissfe.linsolve.amgcl.amgcl.PyAMGCLSolver`'s ``build``/``applyPreconditioner``
call shape.

The whole cycle -- and every stage of its build -- runs in C++ (``PTwoGridPreconditionerT`` in
``edelweissfe/linsolve/amgcl/amgcl-wrapper.hpp``, wrapped as
:class:`~edelweissfe.linsolve.amgcl.amgcl.PyAMGCLPTwoGrid`), OpenMP-threaded throughout; this module
only chooses the parameters and builds the coarse near null-space, which is cheap.

This is an **opt-in variant of the field-split AMG preconditioner**, not the default. It is
rebuilt exactly when single-level hierarchies are, and its rebuild is cheaper than theirs;
the coarse solve always runs on the plain scalar backend, so it gets the same rigid-body
near-null-space treatment the single-level default gives a scalar-backend field (restricted to the
corner-node subset, independent of the field's own single-level backend -- a block backend, the
default for 2D/3D fields, takes the translations at most), while the fine-level Chebyshev smoother
needs none. Measured
against the default single-level configuration with the same outer settings, it was
slower on one reference model and level within noise on another -- worth measuring on a model
with expensive hierarchy rebuilds, not assumed to help.

**Dirichlet handling is load-bearing, do not simplify away.** A field's diagonal block still carries
its Dirichlet identity rows (production applies Dirichlet elimination upstream of the linear solver), and
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
from edelweissfe.linsolve.composition.twolevelcycle import TwoLevelPreconditioner
from edelweissfe.linsolve.nullspace import rigidBodyNullspace, translationNullspace
from edelweissfe.utils import performancetiming

#: The AMGCL tree of the coarse-level (P1) solve, the default of ``coarsePrecond``: from a sweep over the Chebyshev
#: degree and the sweeps on a coupled system, degree 4 with two sweeps before and after matched a higher degree in
#: outer iterations at a lower cost per application, while one sweep each raised the outer iterations. An exact
#: PARDISO factorization of the P1 operator was no cheaper per application (it fills in heavily). ``power_iters`` as
#: in the field trees (see :func:`~edelweissfe.linsolve.composition.amg.defaultFieldTree`).
_DEFAULT_COARSE_PRECOND = {
    "coarsening": {"type": "smoothed_aggregation", "aggr": {"eps_strong": 0.01}},
    "relax": {"type": "chebyshev", "degree": 4, "power_iters": 300, "lower": 0.01},
    "npre": 2,
    "npost": 2,
}


class PTwoGridPreconditioner(TwoLevelPreconditioner):
    """A p-two-grid preconditioner for one vector field's diagonal block, a two-level preconditioner
    (:class:`~edelweissfe.linsolve.composition.twolevelcycle.TwoLevelPreconditioner`) whose coarse level is the P1 corner nodes.

    Parameters
    ----------
    isCorner, edgeEndpoints
        This field's P1 topology map (:func:`edelweissfe.numerics.p1topology.buildP1Map`), in the
        field's own node order.
    amgclTree, backendBlockSize, backendPrecision, coarseHierarchyPrecision
        The parameters of the cycle, see
        :meth:`~edelweissfe.linsolve.composition.twolevelcycle.TwoLevelPreconditioner.cycleParameters`: the fine
        Chebyshev smoother is the ``relax``, ``npre`` and ``npost`` of the field's AMGCL tree. With ``npre`` 0 the cycle
        starts with the coarse-grid correction of the right-hand side itself and ends with ``npost`` sweeps: one
        smoother application fewer per cycle than a symmetric one, at the price of a nonsymmetric preconditioner -- of
        no concern to the right-preconditioned GMRES around it, which needs a fixed linear operator, not a symmetric
        one. Ending on the smoother is the right way round for that one sweep: it damps the high-frequency error the
        prolongated correction leaves on the fine level.
    coarsePrecond
        AMGCL parameter tree for the coarse-level (:math:`A_1`) solve.
    useCoarseNullspace
        Whether to give the coarse AMGCL solver a rigid-body near null-space on the free-corner
        space -- measurably helps here, unlike on the full quadratic block, since the coarse level's
        own aggregation is what actually needs it. Built as the full rigid-body basis (translations +
        rotations) when :meth:`build` is given node coordinates, translations alone otherwise --
        mirroring the near null-space of :class:`~edelweissfe.linsolve.composition.amg.AMG` on the full
        field, since this is the same construction restricted to the corner-node subset.
    fineSpectralRadius, fineDominantVector
        A previous build's fine spectral radius and dominant vector to start from (``fineSpectralRadiusReuse`` of
        :class:`~edelweissfe.linsolve.composition.twolevel.TwoLevel`), or ``None`` to estimate the radius in full, see
        :class:`~edelweissfe.linsolve.composition.twolevelcycle.TwoLevelPreconditioner`.
    """

    def __init__(
        self,
        isCorner: np.ndarray,
        edgeEndpoints: np.ndarray,
        amgclTree: dict,
        backendBlockSize: int = 1,
        backendPrecision: str = "double",
        coarseHierarchyPrecision: str = "double",
        coarsePrecond: dict = None,
        useCoarseNullspace: bool = True,
        fineSpectralRadius: float = None,
        fineDominantVector: np.ndarray = None,
    ):
        self._isCorner = np.ascontiguousarray(isCorner, dtype=np.int8)
        self._edgeEndpoints = np.ascontiguousarray(edgeEndpoints, dtype=np.int32)
        self._useCoarseNullspace = useCoarseNullspace
        from edelweissfe.linsolve.amgcl.amgcl import PyAMGCLPTwoGrid

        super().__init__(
            PyAMGCLPTwoGrid,
            {
                **self.cycleParameters(amgclTree, backendBlockSize, backendPrecision, coarseHierarchyPrecision),
                "coarse": dict(coarsePrecond) if coarsePrecond is not None else dict(_DEFAULT_COARSE_PRECOND),
            },
            fineSpectralRadius,
            fineDominantVector,
        )

    @classmethod
    def parameterRejection(cls, **parameters) -> "str | None":
        """Why these settings -- keywords of this class -- cannot be used, or ``None``, asked without a topology
        map."""
        return super().parameterRejection(np.zeros(0, dtype=bool), np.zeros((0, 2), dtype=int), **parameters)

    def build(self, A, scaling: np.ndarray, coordinates: "np.ndarray | None") -> None:
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
        coordinates
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
                # free coarse DOFs -- same two-step shape as the rest of the null-space
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
                if coordinates is not None:
                    cornerCoords = np.asarray(coordinates, dtype=float)[cornerNodeRows]
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
                self._suppliedFineSpectralRadius(),
                self._fineDominantVector,
            )
