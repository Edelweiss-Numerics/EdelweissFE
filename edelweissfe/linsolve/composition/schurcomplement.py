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
"""Condensing hard dofs exactly: :class:`SchurComplement`, a block LU factorization around a child, and the
condensation it is built from, :class:`SchurComplementCondensation`.

Some dofs make a system hard for AMG: the dofs of embedded bars tied to their host elements by bond springs
(:math:`u_{bar} \\approx \\sum_i N_i u_i`, which an aggregation coarse space cannot represent), or the dofs of a
softening zone. We call these dofs :math:`b` (the condensed dofs: "bars") and all others :math:`k` (the kept, or
remaining, dofs). Ordered this way, the system reads

.. math::

    \\begin{bmatrix} A_{bb} & A_{bk} \\\\ A_{kb} & A_{kk} \\end{bmatrix}
    \\begin{bmatrix} x_b \\\\ x_k \\end{bmatrix} =
    \\begin{bmatrix} r_b \\\\ r_k \\end{bmatrix}.

The block :math:`A_{bb}` is factorized exactly (a direct solver). Eliminating :math:`x_b` leaves the remaining dofs
with the Schur complement :math:`S = A_{kk} - A_{kb} A_{bb}^{-1} A_{bk}` -- for bars, the concrete stiffened by the
bars as seen through their bond -- on which the child of :class:`SchurComplement` is built (e.g. a field-split AMG). Gaussian elimination by
blocks is the block LU factorization of :math:`A`, and its inverse is a preconditioner of the outer Krylov solve:

.. math::

    y_b = A_{bb}^{-1} r_b, \\quad z_k = M_{\\tilde S}^{-1} (r_k - A_{kb} y_b), \\quad
    z_b = A_{bb}^{-1} (r_b - A_{bk} z_k),

with :math:`M_{\\tilde S}^{-1}` the child (``preconditionS``) on an approximation :math:`\\tilde S` of
:math:`S`. With :math:`\\tilde S = S` and an exact sweep it is :math:`A^{-1}`.

It solves with :math:`A_{bb}` twice. The default (``blockFactors = "U"``) inverts only the block upper triangular
factor :math:`U = \\begin{bmatrix} A_{bb} & A_{bk} \\\\ 0 & S \\end{bmatrix}` of
:math:`A = \\begin{bmatrix} I & 0 \\\\ A_{kb} A_{bb}^{-1} & I \\end{bmatrix} U`, with one solve:

.. math::

    z_k = M_{\\tilde S}^{-1} r_k, \\quad z_b = A_{bb}^{-1} (r_b - A_{bk} z_k):

the remaining dofs first, then the condensed ones follow them (for bars: the bars follow the concrete). Its last step
solves the rows of :math:`b` exactly, so after the first outer iteration the residual of these rows is zero, and the
outer Krylov solve iterates on the rows of :math:`k` alone, with the operator :math:`S \\tilde S^{-1}`: the same
operator as with the full block LU factorization. With exact :math:`S`, GMRES converges in two iterations instead of
one; with :math:`\\tilde S` and AMG, the outer iterations hardly grow, and every one saves a solve with
:math:`A_{bb}`.

:math:`S` is dense wherever :math:`A_{bb}^{-1}` is, e.g. along a whole bar. Its entries decay quickly away from
the diagonal, though (a bar on its bond is a beam on an elastic foundation: a load is felt only a few host elements
away), so :math:`\\tilde S` keeps only the significant ones:

- a banded component of :math:`A_{bb}` (a bar, ordered along the bar) is cut into overlapping windows of nodes
  along it; for the dofs of a window's core, :math:`A_{bb}^{-1}` is replaced by the inverse of the window's block
  (restricted additive Schwarz); a short component is a single, exact window,
- a component that is not banded (a zone of a 2D or 3D mesh, e.g. a softening zone) gets its exact correction from
  a factorization of its own block, in batches of right-hand sides,
- entries :math:`|\\tilde S_{ij}| < \\tau \\sqrt{|\\tilde S_{ii} \\tilde S_{jj}|}` are dropped (``dropTolerance``).

Which dofs are condensed is decided by markers, the same ones that mark elements for adaptive refinement
(:mod:`edelweissfe.adaptivity.marking`): :math:`b` holds all dofs of the *interior nodes* of the marked elements,
i.e., of the nodes that have at least one element, all of them marked::

    bar node:   bar, bond                -> all marked -> condensed
    host node:  bond, concrete, concrete -> not all    -> kept

For embedded bars, mark the bar and the bond elements. For a softening zone, mark the elements of a ``perElement``
fieldOutput above a threshold, with a ``halo`` so that the zone's own nodes become interior. Only the dofs of nodes
can be condensed, and always all fields of a node together.
"""

from typing import NamedTuple

import numpy as np
import scipy.sparse as sp
from scipy.linalg import solve_banded
from scipy.sparse.csgraph import connected_components, reverse_cuthill_mckee

import edelweissfe.utils.performancetiming as performancetiming
from edelweissfe.adaptivity.marking import RefineableElements
from edelweissfe.linsolve.base import FieldBlock
from edelweissfe.linsolve.composition.amg import isSignificant
from edelweissfe.linsolve.composition.base import (
    Description,
    Preconditioner,
    SetUpContext,
    Step,
    requirePreconditioner,
)
from edelweissfe.linsolve.composition.direct import DirectSolver
from edelweissfe.linsolve.composition.systemmatrix import SystemMatrix
from edelweissfe.numerics.parallelizationutilities import (
    getNumberOfThreads,
    getThreadPool,
)

#: A component of :math:`A_{bb}` is a chain (a bar) if its bandwidth, in the reverse Cuthill-McKee ordering, is at
#: most a window's core divided by this: a chain couples each node only to a few neighbours along it, while a zone of
#: a 2D or 3D mesh couples it across the zone's width.
CHAIN_BANDWIDTH_DIVISOR = 4

#: The number of right-hand sides per batch of the exact correction of a zone (:meth:`_exactCorrection`) ...
EXACT_CORRECTION_BATCH = 256
#: ... at most, and fewer when the zones are so large that a batch would exceed this many entries.
EXACT_CORRECTION_MAX_ENTRIES = 2**25

#: The correction of every window, and of every batch of a zone, is thinned against the diagonal of :math:`A_{kk}` at
#: this fraction of the drop tolerance before the parts are summed, which bounds their memory; the final drop is
#: against the diagonal of :math:`\tilde S` itself. A fraction well below 1 keeps what the final drop keeps, also an
#: entry to which several parts contribute.
THINNING_FRACTION = 0.1

#: A window's block is solved in LAPACK band storage while its band (lower + upper + 1) holds at most this fraction of
#: its columns (:func:`_solveTransposed`): a window along a bar has a band of a few dofs, so the banded solve costs a
#: small fraction of the dense one, while for a short window, whose band reaches across it, the band storage no longer
#: pays.
BAND_STORAGE_MAX_FRACTION = 0.25

#: The split of :math:`A` into its blocks and the drop of the small entries of :math:`\tilde S` work on blocks of about
#: this many entries in parallel threads, so that the temporaries of each block stay small. The threads write into
#: arrays of the calling thread: in the free-threaded interpreter, an array made by a worker thread and dropped by
#: another is freed only when that worker runs again.
PARALLEL_BLOCK_ENTRIES = 2**20


class SchurComplementCondensation:
    """The condensed dofs :math:`b`, the factorization of :math:`A_{bb}`, the approximate Schur complement
    :math:`\\tilde S` of all other dofs :math:`k`, and the preconditioner; see the module documentation.

    :class:`SchurComplement` uses it in three steps: :meth:`update` on every solve (select :math:`b`, split :math:`A`;
    the Schur complement then factorizes :math:`A_{bb}` with its ``solveA_bb``), :meth:`approximateSchurComplement`
    whenever it rebuilds its child (built on :math:`\\tilde S` instead of :math:`A`), and :func:`blockLU` in every outer
    iteration.

    Parameters
    ----------
    markers
        The markers whose marked elements' interior nodes are condensed.
    zoneSolver
        The :class:`~edelweissfe.linsolve.composition.direct.DirectSolver` of the zones corrected exactly (see
        :meth:`_exactCorrection`): set up on their block, and applied to batches of right-hand sides. Default: SuperLU.
    approximation
        ``"local"``: :math:`\\tilde S` as described in the module documentation; ``"none"``:
        :math:`\\tilde S = A_{kk}` (the coupling through :math:`b` is ignored).
    dropTolerance
        The relative size :math:`\\tau` below which entries of :math:`\\tilde S` are dropped.
    windowNodes, haloNodes
        The nodes of a window's core along a chain, and of its halo on either side (``None``: as many as the core).
    """

    def __init__(
        self,
        markers: list,
        zoneSolver: DirectSolver = None,
        approximation: str = "local",
        dropTolerance: float = 1e-4,
        windowNodes: int = 50,
        haloNodes: "int | None" = None,
    ):
        if approximation not in ("local", "none"):
            raise ValueError("SchurComplement: approximation must be 'local' or 'none', got {!r}".format(approximation))
        self._markers = markers
        self._zoneSolver = zoneSolver if zoneSolver is not None else DirectSolver("superlu")
        self._approximation = approximation
        self._dropTolerance = dropTolerance
        self._windowNodes = windowNodes
        self._haloNodes = windowNodes if haloNodes is None else haloNodes

        # the model, its elements as the candidates of the markers (with the elements at each node), and the dofs of
        # each node, by node label; see setModel
        self._model = None
        self._candidateElements = None
        self._dofsOfNode = None
        #: The condensed dofs b and all other (remaining) dofs k, sorted.
        self.condensedDofs = np.empty(0, dtype=np.int64)
        self.remainingDofs = None
        # the node (label) of each condensed dof, for the windows along the chains
        self._nodeOfCondensedDof = np.empty(0, dtype=np.int64)
        #: Whether the markers selected every dof at the last :meth:`update`, which leaves nothing for the field
        #: split: then nothing is condensed (a direct solver is the better choice for such a system).
        self.selectedAllDofs = False
        #: The number of diagonal entries of the last :math:`\\tilde S` that are not positive (see
        #: :meth:`approximateSchurComplement`).
        self.nonPositiveDiagonalEntries = 0
        # where the entries of the blocks of A are in A.data, found once per sparsity pattern of A and choice of b,
        # see _split
        self._splitCache = None
        # the last selection of the condensed dofs, and the markers' stamps it was made with, see
        # _selectCondensedDofs
        self._selection = self._selectionStamp = None
        self.forget()

    def forget(self) -> None:
        """Drop what was formed from the last matrix: its blocks :math:`A_{bb}`, :math:`A_{bk}`, :math:`A_{kb}`, and
        what :meth:`approximateSchurComplement` formed from them (the factorization of the zones). What is reused for
        the next matrix stays: the selection of the condensed dofs, and where the entries of the blocks are in a matrix
        of the same pattern."""
        self.A_bb = self.A_bk = self.A_kb = None
        self._zoneSolver.forget()

    # ----- the condensed dofs -----

    def setModel(self, model, dofManager) -> None:
        """Take the model and its dofs: check that what the markers read exists, and find the dofs of each node."""
        for marker in self._markers:
            marker.checkReferences(model)
        self._model = model
        self._candidateElements = RefineableElements(list(model.elements.values()))
        indices = dofManager.idcsOfFieldVariablesInDofVector
        self._dofsOfNode = {}
        for element in model.elements.values():
            for node in element.nodes:
                dofs = [np.atleast_1d(indices[fv]) for fv in node.fields.values() if fv in indices]
                if dofs:
                    self._dofsOfNode[node.label] = np.concatenate(dofs)
        self.condensedDofs = np.empty(0, dtype=np.int64)
        self._splitCache = None
        self._selection = self._selectionStamp = None

    def _selectCondensedDofs(self) -> tuple:
        """All dofs of the interior nodes of the marked elements, sorted: a node is interior if it has at least one
        element and all of its elements are marked. A marker that has nothing to decide on yet (a fieldOutput
        before the first converged increment) marks nothing.

        The selection is kept, and reused while every marker's
        :meth:`~edelweissfe.adaptivity.marking.MarkerBase.selectionStamp` stays the same: a marker of sets changes
        only with the model (:meth:`setModel` drops the selection), one of a fieldOutput when the fieldOutput is
        updated (after a converged increment), and a marker that cannot tell (stamp ``None``) marks again on every
        call. So a fieldOutput marker with a halo is not re-evaluated on every Newton iteration.

        Returns
        -------
        tuple
            ``(dofs, nodeOfDof)``: the condensed dofs and the label of the node of each.
        """
        if self._model is None:
            raise RuntimeError(
                "schurComplement: the condensed dofs are chosen on the model, but setModel was not called"
            )
        stamp = tuple(marker.selectionStamp(self._model) for marker in self._markers)
        if self._selection is not None and None not in stamp and stamp == self._selectionStamp:
            return self._selection
        self._selection, self._selectionStamp = self._markedInteriorDofs(), stamp
        return self._selection

    def _markedInteriorDofs(self) -> tuple:
        """The selection of :meth:`_selectCondensedDofs`, made anew."""
        marked = set()
        for marker in self._markers:
            if marker.isReady(self._model):
                marked |= marker.mark(self._model, self._candidateElements, None)
        elementsAtNode = self._candidateElements.elementsAtNode()
        markedNodes = {node.label for element in marked for node in element.nodes}
        interiorNodes = sorted(
            label
            for label in markedNodes
            if label in self._dofsOfNode and all(element in marked for element in elementsAtNode[label])
        )
        if not interiorNodes:
            return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
        dofs = np.concatenate([self._dofsOfNode[label] for label in interiorNodes]).astype(np.int64)
        nodeOfDof = np.repeat(interiorNodes, [self._dofsOfNode[label].size for label in interiorNodes])
        order = np.argsort(dofs)
        return dofs[order], nodeOfDof[order]

    def hasCondensedDofs(self) -> bool:
        """Whether anything is condensed (after the last :meth:`update`)."""
        return self.condensedDofs.size > 0

    # ----- every solve -----

    def update(self, A: sp.csr_matrix) -> None:
        """Select the condensed dofs and cut the blocks :math:`A_{bb}`, :math:`A_{bk}`, :math:`A_{kb}` out of ``A``.

        Parameters
        ----------
        A
            The system matrix, CSR in canonical format.
        """
        # the blocks of the last matrix go before those of this one are cut out
        self.forget()
        with performancetiming.timeit("schur dof selection"):
            dofs, nodeOfDof = self._selectCondensedDofs()
        selectedAllDofs = dofs.size == A.shape[0]
        if selectedAllDofs:
            dofs, nodeOfDof = np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
        if np.array_equal(dofs, self.condensedDofs):
            remainingDofs, splitCache = self.remainingDofs, self._splitCache
        else:
            isCondensed = np.zeros(A.shape[0], dtype=bool)
            isCondensed[dofs] = True
            remainingDofs, splitCache = np.flatnonzero(~isCondensed), None
        A_bb = A_bk = A_kb = None
        if dofs.size > 0:
            with performancetiming.timeit("schur split"):
                splitCache = _splitCacheFor(A, splitCache)
                A_bb, A_bk, A_kb = _split(A, ("bb", "bk", "kb"), dofs, remainingDofs, splitCache)
        self.selectedAllDofs = selectedAllDofs
        self.condensedDofs, self._nodeOfCondensedDof, self.remainingDofs = dofs, nodeOfDof, remainingDofs
        self._splitCache = splitCache
        self.A_bb, self.A_bk, self.A_kb = A_bb, A_bk, A_kb

    @property
    def approximation(self) -> str:
        """How :math:`\\tilde S` is formed: ``"local"`` or ``"none"`` (:math:`\\tilde S = A_{kk}`)."""
        return self._approximation

    # ----- the remaining dofs, on which the child is built -----

    def remainingFieldBlocks(self, blocks: list) -> list:
        """The field blocks of the remaining dofs: every field without its condensed dofs, in the same order (and
        contiguous, as the remaining dofs keep their order). A field whose dofs are all condensed (e.g. the rotations
        of embedded beams) has no block."""
        remainingBlocks, start = [], 0
        for block in blocks:
            n = int(np.count_nonzero(self.remainingNodes(block))) * block.dimension
            if n == 0:
                continue
            remainingBlocks.append(FieldBlock(block.name, start, start + n, block.dimension))
            start += n
        return remainingBlocks

    def remainingNodes(self, block: FieldBlock) -> np.ndarray:
        """Which nodes of the field block ``block`` (of the whole system) remain, are not condensed: a boolean mask
        over the field's nodes, in its dof order. Condensation takes whole nodes, and a field's dofs are node-major."""
        inBlock = self.condensedDofs[(self.condensedDofs >= block.start) & (self.condensedDofs < block.stop)]
        remains = np.ones((block.stop - block.start) // block.dimension, dtype=bool)
        remains[(inBlock - block.start) // block.dimension] = False
        return remains

    # ----- the approximate Schur complement -----

    def approximateSchurComplement(self, A: sp.csr_matrix) -> sp.csr_matrix:
        """:math:`\\tilde S`, see the module documentation. Also counts its diagonal entries that are not positive
        (:attr:`nonPositiveDiagonalEntries`): for a symmetric positive definite :math:`A` there are none, so any
        means that the condensed zone did not take all the softening.

        Parameters
        ----------
        A
            The system matrix of the last :meth:`update`.
        """
        if self._approximation == "none":
            (S,) = _split(A, ("kk",), self.condensedDofs, self.remainingDofs, self._splitCache)
        else:
            S = self._localSchurComplement(A)
        self.nonPositiveDiagonalEntries = int(np.count_nonzero(S.diagonal() <= 0.0))
        return S

    def _localSchurComplement(self, A: sp.csr_matrix) -> sp.csr_matrix:
        """:math:`A_{kk}` minus the corrections of the windows and of the zones, with small entries dropped.

        Each large matrix is formed only once the one before it is no longer needed: the sum :math:`C` of the
        corrections (thinned against the diagonal of :math:`A_{kk}`, which is that of :math:`A` at k), then
        :math:`A_{kk} - C`, then :math:`\\tilde S` by the drop of its small entries."""
        tau = self._dropTolerance
        sqrtAbsDiagonalOfA_kk = np.sqrt(np.abs(A.diagonal()[self.remainingDofs]))
        corrections = self._corrections(sqrtAbsDiagonalOfA_kk, THINNING_FRACTION * tau)
        with performancetiming.timeit("schur assembly"):
            (A_kk,) = _split(A, ("kk",), self.condensedDofs, self.remainingDofs, self._splitCache)
            S = A_kk - corrections
            del A_kk, corrections  # before the drop, which forms S~ beside S
            return _dropSmallEntries(S, tau)

    def _corrections(self, sqrtAbsDiagonalOfA_kk: np.ndarray, thinningTolerance: float) -> sp.csr_matrix:
        """The sum :math:`C` of the corrections of all windows and zones, each thinned (see
        :data:`THINNING_FRACTION`), as a sparse matrix over the remaining dofs k."""
        windows, zoneDofs = self._windowsAndZones()
        A_kb_byColumns = self.A_kb.tocsc()
        parts = []
        with performancetiming.timeit("schur windows"):
            pool = getThreadPool(getNumberOfThreads())
            for part in pool.map(
                lambda window: self._windowCorrection(window, A_kb_byColumns, sqrtAbsDiagonalOfA_kk, thinningTolerance),
                windows,
            ):
                if part is not None:
                    parts.append(part)
        with performancetiming.timeit("schur exact correction"):
            parts += self._exactCorrection(zoneDofs, sqrtAbsDiagonalOfA_kk, thinningTolerance)
        n = self.remainingDofs.size
        with performancetiming.timeit("schur sum of corrections"):
            if not parts:
                return sp.csr_matrix((n, n))
            rows, cols, values = (np.concatenate(v) for v in zip(*parts))
            return sp.csr_matrix((values, (rows, cols)), shape=(n, n))

    def _windowsAndZones(self) -> tuple:
        """The windows along the chains of :math:`A_{bb}`, and the dofs of its zones: of every connected component of
        :math:`A_{bb}`, in the reverse Cuthill-McKee order (along the bar, for a bar, see
        :func:`_componentsInBandOrder`), a banded one (see :data:`CHAIN_BANDWIDTH_DIVISOR`) is a chain, cut into
        overlapping windows (:func:`_cutIntoWindows`), and any other is a zone (of a 2D or 3D mesh), corrected exactly
        (see :meth:`_exactCorrection`).

        Returns
        -------
        tuple
            ``(windows, zoneDofs)``: the :class:`Window` s (the cores partition the dofs of the chains), and the dofs of
            the zones (positions in b).
        """
        windows, zones = [], []
        for dofs, bandwidth in _componentsInBandOrder(self.A_bb):
            core, halo = self._coreAndHalo(dofs)
            if bandwidth > core // CHAIN_BANDWIDTH_DIVISOR:
                zones.append(dofs)
            else:
                windows += _cutIntoWindows(dofs, core, halo)
        zoneDofs = np.sort(np.concatenate(zones)) if zones else np.empty(0, dtype=np.int64)
        return windows, zoneDofs

    def _coreAndHalo(self, dofs: np.ndarray) -> tuple:
        """The core and the halo of the windows along the chain of the dofs ``dofs``, in dofs: ``windowNodes`` and
        ``haloNodes`` nodes (all nodes of a chain carry the same dofs)."""
        dofsPerNode = max(1, round(dofs.size / np.unique(self._nodeOfCondensedDof[dofs]).size))
        return self._windowNodes * dofsPerNode, self._haloNodes * dofsPerNode

    def _windowCorrection(
        self,
        window: "Window",
        A_kb_byColumns: sp.csc_matrix,
        sqrtAbsDiagonalOfA_kk: np.ndarray,
        thinningTolerance: float,
    ):
        """The correction :math:`A_{kb} A_{bb}^{-1} A_{bk}` of one window's core, in which :math:`A_{bb}^{-1}` is
        replaced by the inverse of the window's block :math:`A_{ww}`. Only the core rows of that inverse are needed,
        :math:`(A_{ww}^{-1})_{core,:}`, far fewer right-hand sides than there are remaining dofs coupled to the window:

        .. math::

            \\text{correction}_{ij} = (A_{kb})_{i,core} \\, (A_{ww}^{-1})_{core,:} \\, (A_{bk})_{window,j}

        for the remaining dofs i coupled to the core and j coupled to the window. ``A_kb_byColumns`` is
        :math:`A_{kb}` in CSC format, from which the columns of the core are cut.

        Returns
        -------
        tuple | None
            ``(rows, cols, values)`` of the significant entries, as positions in k, or ``None``.
        """
        A_bk_window = self.A_bk[window.dofs]
        coupledToWindow = np.unique(A_bk_window.indices)
        A_kb_core = A_kb_byColumns[:, window.dofs[window.core]]
        coupledToCore = np.unique(A_kb_core.indices)
        if len(coupledToWindow) == 0 or len(coupledToCore) == 0:
            return None
        A_ww = self.A_bb[window.dofs][:, window.dofs]
        # (A_ww^-1)_{core,:}: the transposed solution of A_ww^T X = I[:, core], the core columns of the identity
        inverseOfA_ww_coreRows = _solveTransposed(A_ww.tocoo(), np.eye(len(window.dofs))[:, window.core]).T
        correction = A_kb_core[coupledToCore] @ inverseOfA_ww_coreRows @ A_bk_window[:, coupledToWindow]
        return _significantEntries(correction, coupledToCore, coupledToWindow, sqrtAbsDiagonalOfA_kk, thinningTolerance)

    def _exactCorrection(
        self, zoneDofs: np.ndarray, sqrtAbsDiagonalOfA_kk: np.ndarray, thinningTolerance: float
    ) -> list:
        """The exact correction :math:`A_{kz} A_{zz}^{-1} A_{zk}` of the zones (dofs ``zoneDofs`` of b): their block
        :math:`A_{zz}` of :math:`A_{bb}` (block diagonal over the components, so its inverse is the zones' part of
        :math:`A_{bb}^{-1}`) is factorized once, and solved for the remaining dofs coupled to the zones in batches of
        at most :data:`EXACT_CORRECTION_BATCH` right-hand sides.

        Returns
        -------
        list
            ``(rows, cols, values)`` of the significant entries of every batch, as positions in k.
        """
        if zoneDofs.size == 0:
            return []
        A_zz = self.A_bb[zoneDofs][:, zoneDofs].tocsr()
        self._zoneSolver.setUp(SystemMatrix(A_zz), SetUpContext(fields=(FieldBlock("zones", 0, zoneDofs.size, 1),)))
        A_bk_zones = self.A_bk[zoneDofs].tocsc()
        coupledColumns = np.flatnonzero(np.diff(A_bk_zones.indptr))
        A_kb_zones = self.A_kb[:, zoneDofs].tocsr()
        coupledRows = np.flatnonzero(np.diff(A_kb_zones.indptr))
        A_kb_zones = A_kb_zones[coupledRows]
        batch = max(1, min(EXACT_CORRECTION_BATCH, EXACT_CORRECTION_MAX_ENTRIES // zoneDofs.size))
        parts = []
        for start in range(0, len(coupledColumns), batch):
            columns = coupledColumns[start : start + batch]
            rightHandSides = A_bk_zones[:, columns].toarray()
            X = np.asarray(self._zoneSolver(rightHandSides)).reshape(rightHandSides.shape)
            correction = A_kb_zones @ X
            parts.append(
                _significantEntries(correction, coupledRows, columns, sqrtAbsDiagonalOfA_kk, thinningTolerance)
            )
        return parts


class Window(NamedTuple):
    """A window along a chain of :math:`A_{bb}`, see :meth:`SchurComplementCondensation._windowsAndZones`."""

    #: The dofs of the window, positions in b: its core and the halos on either side, in order along the chain.
    dofs: np.ndarray
    #: The positions of its core within ``dofs``.
    core: np.ndarray


def _componentsInBandOrder(A_bb: sp.csr_matrix) -> list:
    """The connected components of :math:`A_{bb}`: per component, its dofs (positions in b) in the reverse
    Cuthill-McKee order -- along the bar, for a bar -- and its bandwidth in that order."""
    order = reverse_cuthill_mckee(A_bb, symmetric_mode=True)
    nComponents, componentOf = connected_components(A_bb, directed=False)
    # the dofs component by component, each component in the reverse Cuthill-McKee order
    alongComponents = order[np.argsort(componentOf[order], kind="stable")]
    componentSizes = np.bincount(componentOf, minlength=nComponents)
    rank = np.empty_like(order)
    rank[order] = np.arange(order.size)
    coo = A_bb.tocoo()
    bandwidth = np.zeros(nComponents, dtype=np.int64)
    np.maximum.at(bandwidth, componentOf[coo.row], np.abs(rank[coo.row] - rank[coo.col]))
    return list(zip(np.split(alongComponents, np.cumsum(componentSizes)[:-1]), bandwidth))


def _cutIntoWindows(dofs: np.ndarray, core: int, halo: int) -> list:
    """The windows along a chain of the dofs ``dofs`` (in order along it): cores of ``core`` dofs, each with a halo of
    ``halo`` dofs on either side; a chain no longer than one core with both halos is a single, exact window."""
    n = dofs.size
    if n <= core + 2 * halo:
        return [Window(dofs, np.arange(n))]
    windows = []
    for start in range(0, n, core):
        stop = min(start + core, n)
        low, high = max(0, start - halo), min(n, stop + halo)
        windows.append(Window(dofs[low:high], np.arange(start - low, stop - low)))
    return windows


def _splitCacheFor(A: sp.csr_matrix, cache: "dict | None") -> dict:
    """The cache of :func:`_split` for ``A``: ``cache`` if it was made for ``A``'s sparsity pattern, else an empty one
    for it (made, too, when ``cache`` is ``None``: a new choice of the condensed dofs)."""
    if cache is not None and np.array_equal(cache["indptr"], A.indptr) and np.array_equal(cache["indices"], A.indices):
        return cache
    return {"indptr": A.indptr.copy(), "indices": A.indices.copy(), "blocks": {}}


def _split(A: sp.csr_matrix, which: tuple, condensedDofs: np.ndarray, remainingDofs: np.ndarray, cache: dict) -> list:
    """The blocks ``which`` (of ``"bb"``, ``"bk"``, ``"kb"``, ``"kk"``) of ``A`` for the condensed dofs b and the
    remaining dofs k, as CSR matrices.

    The simple way is fancy indexing, ``A[b][:, k]``, which copies and searches the large matrix on every solve.
    But the sparsity pattern of ``A`` stays the same over many solves, so this finds once per pattern (and choice of
    b) where the entries of a block are in ``A.data`` (:func:`_entriesOfBlock`), and keeps that in ``cache`` (see
    :func:`_splitCacheFor`). Every solve then only gathers the values and the column indices of these entries, for
    blocks of them in parallel threads (:func:`_gatheredInParallel`)."""
    dofsOf = {"b": condensedDofs, "k": remainingDofs}
    blocks = []
    for name in which:
        if name not in cache["blocks"]:
            cache["blocks"][name] = _entriesOfBlock(A, dofsOf[name[0]], dofsOf[name[1]])
        indexInData, positionInCols, indptr, shape = cache["blocks"][name]
        data = _gatheredInParallel(lambda index: A.data[index], indexInData, A.data.dtype)
        indices = _gatheredInParallel(lambda index: positionInCols[A.indices[index]], indexInData, indptr.dtype)
        blocks.append(sp.csr_matrix((data, indices, indptr), shape=shape))
    return blocks


def _entriesOfBlock(A: sp.csr_matrix, rows: np.ndarray, cols: np.ndarray) -> tuple:
    """Where the entries of the block ``A[rows][:, cols]`` are in ``A.data``, for sorted ``rows`` and ``cols`` and a
    canonical ``A``: the entries of the rows ``rows`` in the columns ``cols``, which keep their order in ``A.data``.

    Returns
    -------
    tuple
        ``(indexInData, positionInCols, indptr, shape)``: the index in ``A.data`` of every entry of the block, in the
        order of the block's CSR format, the position of every column of ``A`` in ``cols`` (the block's column index
        of an entry in that column), and the block's row pointers and shape. The indices are int32 while ``A`` has
        fewer than 2^31 entries.
    """
    n = A.shape[0]
    indexType = np.int32 if A.nnz <= np.iinfo(np.int32).max else np.int64
    isRow, isCol = np.zeros(n, dtype=bool), np.zeros(n, dtype=bool)
    isRow[rows], isCol[cols] = True, True
    inBlock = np.repeat(isRow, np.diff(A.indptr))
    inBlock &= isCol[A.indices]
    indexInData = np.flatnonzero(inBlock).astype(indexType)
    # the entries of the block up to its row r are those before the end of A's row r
    rowEnds = A.indptr[rows + 1].astype(indexType)  # else searchsorted converts indexInData to their type
    indptr = np.concatenate([[0], np.searchsorted(indexInData, rowEnds)]).astype(indexType)
    positionInCols = np.full(n, -1, dtype=indexType)
    positionInCols[cols] = np.arange(cols.size)
    return indexInData, positionInCols, indptr, (rows.size, cols.size)


def _gatheredInParallel(gather, index: np.ndarray, dtype) -> np.ndarray:
    """``gather(index)``, an array of ``index``'s size, computed for blocks of :data:`PARALLEL_BLOCK_ENTRIES` entries
    of ``index`` in parallel threads. Blocks also keep the temporaries small: numpy indexes with an int32 index by
    an int64 copy of it."""
    gathered = np.empty(index.size, dtype=dtype)

    def gatherBlock(start):
        gathered[start : start + PARALLEL_BLOCK_ENTRIES] = gather(index[start : start + PARALLEL_BLOCK_ENTRIES])

    list(getThreadPool(getNumberOfThreads()).map(gatherBlock, range(0, index.size, PARALLEL_BLOCK_ENTRIES)))
    return gathered


def _significantEntries(
    correction: np.ndarray, rows: np.ndarray, cols: np.ndarray, sqrtAbsDiagonal: np.ndarray, tolerance: float
) -> tuple:
    """The entries of a dense correction block (rows ``rows``, columns ``cols`` of k) that pass the relative drop test
    at ``tolerance`` (:func:`~edelweissfe.linsolve.composition.amg.isSignificant`, against the diagonal of
    :math:`A_{kk}`), as ``(rows, cols, values)``."""
    significant = isSignificant(correction, rows[:, None], cols[None, :], sqrtAbsDiagonal, tolerance)
    i, j = np.nonzero(significant)
    return rows[i], cols[j], correction[i, j]


def _dropSmallEntries(S: sp.csr_matrix, tau: float) -> sp.csr_matrix:
    """``S`` without the entries that fail the relative drop test :math:`|S_{ij}| \\geq \\tau \\sqrt{|S_{ii} S_{jj}|}`
    (:func:`~edelweissfe.linsolve.composition.amg.isSignificant`) and without stored zeros.

    The simple form is ``S.data[~significant] = 0; S.eliminate_zeros()``. For the hundreds of millions of entries of a
    large model, this computes it for blocks of rows (of about :data:`PARALLEL_BLOCK_ENTRIES` entries) in parallel
    threads instead, in two passes: the first marks the kept entries (one byte per entry) and counts them per row, the
    second copies them into the result, allocated once their number is known."""
    sqrtAbsDiagonal = np.sqrt(np.abs(S.diagonal()))
    rowBlocks = _rowBlocks(S)
    indexType = S.indices.dtype  # also of the result's indptr, else scipy converts it
    keep, keptPerRow = np.empty(S.nnz, dtype=bool), np.empty(S.shape[0], dtype=indexType)

    def markKeptEntries(rowBlock):
        low, high = rowBlock
        entries = slice(S.indptr[low], S.indptr[high])
        rowOfEntry = np.repeat(np.arange(low, high), np.diff(S.indptr[low : high + 1]))
        keepHere = isSignificant(S.data[entries], rowOfEntry, S.indices[entries], sqrtAbsDiagonal, tau)
        keepHere &= S.data[entries] != 0.0
        keep[entries] = keepHere
        keptPerRow[low:high] = np.bincount(rowOfEntry[keepHere] - low, minlength=high - low)

    pool = getThreadPool(getNumberOfThreads())
    list(pool.map(markKeptEntries, rowBlocks))
    indptr = np.zeros(S.shape[0] + 1, dtype=indexType)
    np.cumsum(keptPerRow, out=indptr[1:])
    data, indices = np.empty(indptr[-1], dtype=S.data.dtype), np.empty(indptr[-1], dtype=indexType)

    def copyKeptEntries(rowBlock):
        low, high = rowBlock
        entries, kept = slice(S.indptr[low], S.indptr[high]), slice(indptr[low], indptr[high])
        np.compress(keep[entries], S.data[entries], out=data[kept])
        np.compress(keep[entries], S.indices[entries], out=indices[kept])

    list(pool.map(copyKeptEntries, rowBlocks))
    return sp.csr_matrix((data, indices, indptr), shape=S.shape)


def _rowBlocks(S: sp.csr_matrix) -> list:
    """Consecutive ranges ``(low, high)`` of the rows of ``S`` that cover all of them, each with about
    :data:`PARALLEL_BLOCK_ENTRIES` entries."""
    nBlocks = max(1, int(np.ceil(S.nnz / PARALLEL_BLOCK_ENTRIES)))
    firstRows = np.searchsorted(S.indptr, np.linspace(0, S.nnz, nBlocks + 1)[1:-1])
    bounds = np.unique(np.concatenate([[0], firstRows, [S.shape[0]]]))
    return list(zip(bounds[:-1], bounds[1:]))


def _solveTransposed(A: sp.coo_matrix, B: np.ndarray) -> np.ndarray:
    """:math:`A^{-T} B` for a square sparse ``A``: by a banded LU factorization if ``A`` is banded (a window along
    a bar), else by a dense one (see :data:`BAND_STORAGE_MAX_FRACTION`)."""
    n = A.shape[0]
    lower = int(max(0, (A.row - A.col).max(initial=0)))
    upper = int(max(0, (A.col - A.row).max(initial=0)))
    if lower + upper + 1 > int(BAND_STORAGE_MAX_FRACTION * n):
        return np.linalg.solve(A.toarray().T, B)
    # A^T in LAPACK band storage: (A^T)_ij = A_ji at [upper(A^T) + i - j, j], and upper(A^T) = lower(A)
    banded = np.zeros((lower + upper + 1, n))
    banded[lower + A.col - A.row, A.row] = A.data
    return solve_banded((upper, lower), banded, B, check_finite=False)


def blockLU(r: np.ndarray, condensation, preconditionS, solveA_bb, blockFactors: str) -> np.ndarray:
    """One application of the block LU preconditioner, see the module documentation:

    .. math::

        y_b = A_{bb}^{-1} r_b, \\qquad z_k = M_{\\tilde S}^{-1} (r_k - A_{kb} y_b), \\qquad
        z_b = A_{bb}^{-1} (r_b - A_{bk} z_k)

    or, with ``blockFactors="U"``, :math:`z_k = M_{\\tilde S}^{-1} r_k` (the remaining dofs first, then the condensed
    ones follow them: the bars follow the concrete). The application of :class:`SchurComplement`.

    Parameters
    ----------
    r
        The vector to precondition (all dofs).
    condensation
        The :class:`SchurComplementCondensation`: the dofs :math:`b, k` and the blocks :math:`A_{bk}, A_{kb}`.
    preconditionS
        :math:`M_{\\tilde S}^{-1}`, applied to a vector of the remaining dofs :math:`k`.
    solveA_bb
        :math:`A_{bb}^{-1}`, applied to a vector of the condensed dofs :math:`b`.
    blockFactors
        ``"U"`` or ``"LU"``, see the module documentation.
    """
    A_bk, A_kb = condensation.A_bk, condensation.A_kb
    b, k = condensation.condensedDofs, condensation.remainingDofs
    r_b, r_k = r[b], r[k]
    if blockFactors == "LU":
        y_b = solveA_bb(r_b)
        z_k = preconditionS(r_k - A_kb @ y_b)
    else:
        z_k = preconditionS(r_k)
    z_b = solveA_bb(r_b - A_bk @ z_k)
    z = np.empty_like(r)
    z[b], z[k] = z_b, z_k
    return z


def _compactCount(count: int) -> str:
    """A count with three significant digits and a suffix: ``233398138`` is ``233M``, ``240084`` is ``240k``."""
    for factor, suffix in ((1e9, "G"), (1e6, "M"), (1e3, "k")):
        if count >= factor:
            return "{:.3g}{:}".format(count / factor, suffix)
    return str(count)


class SchurComplement(Preconditioner):
    """The block LU preconditioner around the Schur complement of the condensed dofs; see the module documentation.

    Every solve selects the condensed dofs again and sets up ``solveA_bb`` on :math:`A_{bb}` (factorizes it);
    :math:`\\tilde S` and the child are rebuilt when a rebuild is asked for, and when the condensed dofs change. If
    :math:`A_{bb}` cannot be factorized (a bond that lost all stiffness), that solve does without the condensation: the
    child is built on the whole :math:`A`.

    Parameters
    ----------
    condense
        The markers whose marked elements' interior nodes are condensed (e.g. an element set marker).
    preconditionS
        :math:`M_{\\tilde S}^{-1}`, built on :math:`\\tilde S`.
    solveA_bb
        The exact solve with :math:`A_{bb}`, a :class:`~edelweissfe.linsolve.composition.direct.DirectSolver`
        (default: SuperLU). The zones corrected exactly in :math:`\\tilde S` are factorized by a second one of the same
        backend and options.
    approximation, dropTolerance, windowNodes, haloNodes
        How :math:`\\tilde S` is formed, see :class:`SchurComplementCondensation`.
    blockFactors
        ``"U"`` or ``"LU"``.
    """

    def __init__(
        self,
        condense: list = (),
        preconditionS: Preconditioner = None,
        solveA_bb: DirectSolver = None,
        approximation: str = "local",
        dropTolerance: float = 1e-4,
        windowNodes: int = 50,
        haloNodes: int = None,
        blockFactors: str = "U",
    ):
        if preconditionS is None:
            raise ValueError("SchurComplement: preconditionS, the preconditioner of the Schur complement, is needed")
        requirePreconditioner("SchurComplement", "preconditionS", preconditionS)
        if solveA_bb is not None and not isinstance(solveA_bb, DirectSolver):
            raise TypeError(
                "SchurComplement: solveA_bb must be a DirectSolver, the exact solve with A_bb, got {!r}".format(
                    solveA_bb
                )
            )
        if blockFactors not in ("U", "LU"):
            raise ValueError("SchurComplement: blockFactors must be 'U' or 'LU', got {!r}".format(blockFactors))
        if not list(condense):
            raise ValueError(
                "SchurComplement: nothing to condense -- give the markers of the condensed elements (condense=[...])"
            )
        self.preconditionS = preconditionS
        self.solveA_bb = solveA_bb if solveA_bb is not None else DirectSolver("superlu")
        self.blockFactors = blockFactors
        self.condensation = SchurComplementCondensation(
            list(condense),
            DirectSolver(self.solveA_bb.backend, **self.solveA_bb.options),
            approximation=approximation,
            dropTolerance=dropTolerance,
            windowNodes=windowNodes,
            haloNodes=haloNodes,
        )
        # the messages shown once per model
        self._messagesShown = set()
        # after the condensation: what this component builds includes the blocks it cuts out (see _forget)
        super().__init__()

    def _forget(self) -> None:
        self.condensation.forget()
        #: Whether the last set-up condenses (else the child is built on the whole matrix).
        self.condensing = False
        #: :math:`\tilde S`, on which the child is built.
        self.S_tilde = None
        # the condensed dofs the child was built for (None: built on the whole matrix, or not built)
        self._builtForDofs = None

    def setModel(self, model, dofManager) -> None:
        self.condensation.setModel(model, dofManager)
        self._builtForDofs = None
        self._messagesShown = set()
        super().setModel(model, dofManager)

    def _setUp(self, A, context: SetUpContext) -> None:
        self._condense(A, context, rebuild=True, change=None)

    def _update(self, A, change) -> None:
        self._condense(A, change.context, rebuild=change.requiresRebuild, change=change)

    def _condense(self, A, context: SetUpContext, rebuild: bool, change) -> None:
        """Select the condensed dofs and factorize A_bb, then set up or update the child on S~ (or, without condensed
        dofs, on A)."""
        condensation = self.condensation
        condensation.update(A.matrix())
        condensing = condensation.hasCondensedDofs()
        try:
            if condensing:
                A_bb = condensation.A_bb
                with performancetiming.timeit("schur factorization"):
                    self.solveA_bb.setUp(
                        SystemMatrix(A_bb), SetUpContext(fields=(FieldBlock("condensed", 0, A_bb.shape[0], 1),))
                    )
        except np.linalg.LinAlgError as error:
            # e.g. a softening zone or a bond that lost all stiffness. The condensed dofs come from the last converged
            # increment, so a cutback would meet the same block again: this solve does without.
            self._warn(context, "the condensed block A_bb cannot be factorized ({:}); solved without it".format(error))
            condensing = False
        if condensation.selectedAllDofs:
            self._warnOnce(context, "the markers select every dof; nothing is condensed")
        elif condensing and condensation.condensedDofs.size > A.shape[0] // 2:
            self._warnOnce(
                context,
                "the markers select more than half of the dofs: the solve is mostly a direct one, with the cost of the "
                "preconditioner of S̃ on top",
            )
        rebuild = rebuild or change is None
        dofs, dofsChanged = self._condensedDofsOf(condensing)
        if rebuild or dofsChanged:
            # the child is built anew, on a new S~ or on A: the old S~ and the child's old state go first
            self.S_tilde = None
            self.preconditionS.forget()
        if condensing and (rebuild or dofsChanged):
            try:
                with performancetiming.timeit("schur complement"):
                    self.S_tilde = SystemMatrix(condensation.approximateSchurComplement(A.matrix()))
            except np.linalg.LinAlgError as error:
                # e.g. a singular window block of A_bb, which factorizes as a whole: like a failed factorization, this
                # solve does without the condensation
                self._warn(
                    context, "the approximate Schur complement cannot be formed ({:}); solved without it".format(error)
                )
                condensing = False
                dofs, dofsChanged = self._condensedDofsOf(condensing)
            else:
                self.statistics.sizes.update({"condensed dofs": dofs.size, "S̃ nnz": self.S_tilde.nnz})
                if condensation.nonPositiveDiagonalEntries:
                    self._warn(
                        context,
                        "the approximate Schur complement has {:} non-positive diagonal entries: the condensed dofs do "
                        "not hold all of the softening, and the preconditioner of the remaining dofs may "
                        "struggle".format(condensation.nonPositiveDiagonalEntries),
                    )
        self.condensing = condensing

        if not condensing:
            childMatrix, childContext = A, context
        else:
            remainingFields = condensation.remainingFieldBlocks(list(context.fields))
            childMatrix = self.S_tilde
            remainingNodes = {block.name: condensation.remainingNodes(block) for block in context.fields}
            childContext = context.restrictedTo(remainingFields, condensation.remainingDofs, nodesOf=remainingNodes.get)
        if change is None:
            self.preconditionS.setUp(childMatrix, childContext)
        else:
            childChange = change.withContext(childContext)
            if dofsChanged:
                childChange = childChange.withRebuild("condensed dofs")
            self.preconditionS.update(childMatrix, childChange)
        self._builtForDofs = dofs

    def _condensedDofsOf(self, condensing: bool) -> tuple:
        """The condensed dofs of this solve (``None`` without condensation), and whether they differ from those the child
        was built for."""
        dofs = self.condensation.condensedDofs if condensing else None
        changed = (dofs is None) != (self._builtForDofs is None) or (
            dofs is not None and not np.array_equal(dofs, self._builtForDofs)
        )
        return dofs, changed

    def finishSolve(self) -> None:
        # S~ is kept for the next solve (its child may be reused), but what the solve formed from it is not
        if self.S_tilde is not None:
            self.S_tilde.releaseFormedMatrices()
        super().finishSolve()

    def builtMatrices(self) -> tuple:
        if not self.condensing:
            return ()
        return (("A_bb", self.condensation.A_bb), ("S̃", self.S_tilde.A))

    def _warn(self, context: SetUpContext, message: str) -> None:
        context.message("warning", "⚠ SchurComplement: " + message)

    def _warnOnce(self, context: SetUpContext, message: str) -> None:
        """A warning shown only the first time for the current model."""
        if message not in self._messagesShown:
            self._messagesShown.add(message)
            self._warn(context, message)

    def condensedDofs(self) -> "np.ndarray | None":
        return self.condensation.condensedDofs if self.condensing else None

    def setUpNotes(self) -> tuple:
        if not self.condensing:
            return super().setUpNotes()
        return (
            "{:} dofs condensed".format(_compactCount(self.condensation.condensedDofs.size)),
            "S̃ {:} nnz".format(_compactCount(self.S_tilde.nnz)),
        ) + super().setUpNotes()

    def _apply(self, r: np.ndarray) -> np.ndarray:
        if not self.condensing:
            return self.preconditionS(r)
        return blockLU(r, self.condensation, self.preconditionS, self.solveA_bb, self.blockFactors)

    def children(self) -> tuple:
        return (("solveA_bb", self.solveA_bb), ("preconditionS", self.preconditionS))

    def describe(self) -> Description:
        if self.blockFactors == "LU":
            steps = (
                Step("y_b = A_bb⁻¹ r_b", "solveA_bb"),
                Step("z_k = M_S̃⁻¹ (r_k − A_kb y_b)", "preconditionS"),
                Step("z_b = A_bb⁻¹ (r_b − A_bk z_k)", "solveA_bb"),
            )
        else:
            steps = (
                Step("z_k = M_S̃⁻¹ r_k", "preconditionS"),
                Step("z_b = A_bb⁻¹ (r_b − A_bk z_k)", "solveA_bb"),
            )
        return Description(
            name="SchurComplement",
            formula="BlockLU[{solveA_bb}, {preconditionS}]",
            latex="\\mathrm{BlockLU}\\left[{solveA_bb},\\, {preconditionS}\\right]",
            parameters={
                "blockFactors": self.blockFactors,
                "S̃": "A_kk" if self.condensation.approximation == "none" else "A_kk − A_kb A_bb⁻¹ A_bk, local",
            },
            steps=steps,
        )
