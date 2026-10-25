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

from collections.abc import Callable
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


class SchurComplementCondensation:
    """The condensed dofs :math:`b`, the factorization of :math:`A_{bb}`, the approximate Schur complement
    :math:`\\tilde S` of all other dofs :math:`k`, and the preconditioner; see the module documentation.

    :class:`SchurComplement` uses it in three steps: :meth:`update` on every solve (select :math:`b`, split :math:`A`,
    factorize :math:`A_{bb}`), :meth:`approximateSchurComplement` whenever it rebuilds its child (built on
    :math:`\\tilde S` instead of :math:`A`), and :func:`blockLU` in every outer iteration.

    Parameters
    ----------
    markers
        The markers whose marked elements' interior nodes are condensed.
    createDirectSolver
        Creates a linear solver that factorizes once for several right-hand sides
        (:meth:`~edelweissfe.linsolve.base.DirectSolverBackend.factorize`,
        :meth:`~edelweissfe.linsolve.base.DirectSolverBackend.solveFactorized`): one for :math:`A_{bb}`, one for the zones
        corrected exactly.
    approximation
        ``"local"``: :math:`\\tilde S` as described in the module documentation; ``"none"``:
        :math:`\\tilde S = A_{kk}` (the coupling through :math:`b` is ignored).
    dropTolerance
        The relative size :math:`\\tau` below which entries of :math:`\\tilde S` are dropped.
    windowNodes, haloNodes
        The nodes of a window's core along a chain, and of its halo on either side (``None``: as many as the core).
    blockFactors
        The block factors of :math:`A` the preconditioner inverts: ``"U"``, the block upper triangular one (one solve
        with :math:`A_{bb}`), or ``"LU"``, both (two solves), see the module documentation.
    """

    def __init__(
        self,
        markers: list,
        createDirectSolver: Callable,
        approximation: str = "local",
        dropTolerance: float = 1e-4,
        windowNodes: int = 50,
        haloNodes: "int | None" = None,
        blockFactors: str = "U",
    ):
        self._markers = markers
        self._directSolver = createDirectSolver()
        self._zoneSolver = createDirectSolver()
        self._approximation = approximation
        self._dropTolerance = dropTolerance
        self._windowNodes = windowNodes
        self._haloNodes = windowNodes if haloNodes is None else haloNodes
        self._blockFactors = blockFactors

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
        self.A_bb = self.A_bk = self.A_kb = None
        # A_kb by columns, for cutting out the columns of the windows, see approximateSchurComplement
        self._A_kb_byColumns = None
        # the last selection of the condensed dofs, and the markers' stamps it was made with, see
        # _selectCondensedDofs
        self._selection = self._selectionStamp = None

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
        """Select the condensed dofs, cut the blocks :math:`A_{bb}`, :math:`A_{bk}`, :math:`A_{kb}` out of ``A`` and
        factorize :math:`A_{bb}`.

        Parameters
        ----------
        A
            The system matrix, CSR in canonical format.

        Raises
        ------
        numpy.linalg.LinAlgError
            If :math:`A_{bb}` cannot be factorized (e.g. a softening zone or a bond that lost all stiffness). The
            update is then not applied: the condensation keeps the state of the last successful one.
        """
        with performancetiming.timeit("schur dof selection"):
            dofs, nodeOfDof = self._selectCondensedDofs()
        selectedAllDofs = dofs.size == A.shape[0]
        if selectedAllDofs:
            dofs, nodeOfDof = np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
        # The new state is computed into locals and assigned only once A_bb is factorized: a failed update leaves the
        # condensation as the last successful one left it, so that a solve without the condensation can take over.
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
            with performancetiming.timeit("schur factorization"):
                self._directSolver.factorize(A_bb)
        self.selectedAllDofs = selectedAllDofs
        self.condensedDofs, self._nodeOfCondensedDof, self.remainingDofs = dofs, nodeOfDof, remainingDofs
        self._splitCache = splitCache
        self.A_bb, self.A_bk, self.A_kb = A_bb, A_bk, A_kb

    @property
    def blockFactors(self) -> str:
        """The block factors of :math:`A` the preconditioner inverts: ``"U"`` or ``"LU"``."""
        return self._blockFactors

    def solveA_bb(self, r_b: np.ndarray) -> np.ndarray:
        """:math:`A_{bb}^{-1} r_b` with the factorization of the last :meth:`update`. One application of the block
        LU preconditioner is :func:`blockLU`."""
        return self._directSolver.solveFactorized(r_b)

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
        (A_kk,) = _split(A, ("kk",), self.condensedDofs, self.remainingDofs, self._splitCache)
        if self._approximation == "none":
            S = A_kk
        else:
            S = self._localSchurComplement(A_kk)
        self.nonPositiveDiagonalEntries = int(np.count_nonzero(S.diagonal() <= 0.0))
        return S

    def _localSchurComplement(self, A_kk: sp.csr_matrix) -> sp.csr_matrix:
        """:math:`A_{kk}` minus the corrections of the windows and of the zones, with small entries dropped."""
        tau = self._dropTolerance
        sqrtAbsDiagonalOfA_kk = np.sqrt(np.abs(A_kk.diagonal()))
        thinningTolerance = THINNING_FRACTION * tau

        windows, zoneDofs = self._windowsAndZones()
        self._A_kb_byColumns = self.A_kb.tocsc()
        parts = []
        with performancetiming.timeit("schur windows"):
            pool = getThreadPool(getNumberOfThreads())
            for part in pool.map(
                lambda window: self._windowCorrection(window, sqrtAbsDiagonalOfA_kk, thinningTolerance), windows
            ):
                if part is not None:
                    parts.append(part)
        with performancetiming.timeit("schur exact correction"):
            parts += self._exactCorrection(zoneDofs, sqrtAbsDiagonalOfA_kk, thinningTolerance)

        with performancetiming.timeit("schur assembly"):
            if parts:
                rows, cols, values = (np.concatenate(v) for v in zip(*parts))
                S = (A_kk - sp.csr_matrix((values, (rows, cols)), shape=A_kk.shape)).tocsr()
            else:
                S = A_kk.copy()
            return _dropSmallEntries(S, tau)

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

    def _windowCorrection(self, window: "Window", sqrtAbsDiagonalOfA_kk: np.ndarray, thinningTolerance: float):
        """The correction :math:`A_{kb} A_{bb}^{-1} A_{bk}` of one window's core, in which :math:`A_{bb}^{-1}` is
        replaced by the inverse of the window's block :math:`A_{ww}`. Only the core rows of that inverse are needed,
        :math:`(A_{ww}^{-1})_{core,:}`, far fewer right-hand sides than there are remaining dofs coupled to the window:

        .. math::

            \\text{correction}_{ij} = (A_{kb})_{i,core} \\, (A_{ww}^{-1})_{core,:} \\, (A_{bk})_{window,j}

        for the remaining dofs i coupled to the core and j coupled to the window.

        Returns
        -------
        tuple | None
            ``(rows, cols, values)`` of the significant entries, as positions in k, or ``None``.
        """
        A_bk_window = self.A_bk[window.dofs]
        coupledToWindow = np.unique(A_bk_window.indices)
        A_kb_core = self._A_kb_byColumns[:, window.dofs[window.core]]
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
        self._zoneSolver.factorize(self.A_bb[zoneDofs][:, zoneDofs].tocsr())
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
            X = np.asarray(self._zoneSolver.solveFactorized(rightHandSides)).reshape(rightHandSides.shape)
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
    b), for every entry of a block, the index of that entry in ``A.data``: it slices a matrix with ``A``'s pattern
    whose values are those indices, and keeps them in ``cache`` (see :func:`_splitCacheFor`). Every solve then only
    gathers ``A.data`` at these indices."""
    dofsOf = {"b": condensedDofs, "k": remainingDofs}
    blocks = []
    for name in which:
        if name not in cache["blocks"]:
            # index + 1: a zero value could be dropped by the slicing
            entryIndexInData = sp.csr_matrix(
                (np.arange(1, A.nnz + 1, dtype=np.float64), A.indices, A.indptr), shape=A.shape
            )
            block = entryIndexInData[dofsOf[name[0]]][:, dofsOf[name[1]]].tocsr()
            block.sort_indices()
            cache["blocks"][name] = (block.data.astype(np.int64) - 1, block.indices, block.indptr, block.shape)
        indexInData, indices, indptr, shape = cache["blocks"][name]
        blocks.append(sp.csr_matrix((A.data[indexInData], indices, indptr), shape=shape))
    return blocks


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

    The simple form is ``S.data[~significant] = 0; S.eliminate_zeros()``. For the tens of millions of entries of a large
    model, this computes it for blocks of rows in parallel threads instead."""
    sqrtAbsDiagonal = np.sqrt(np.abs(S.diagonal()))

    def keptEntries(rowRange):
        low, high = rowRange
        entries = slice(S.indptr[low], S.indptr[high])
        rowOfEntry = np.repeat(np.arange(low, high), np.diff(S.indptr[low : high + 1]))
        keep = isSignificant(S.data[entries], rowOfEntry, S.indices[entries], sqrtAbsDiagonal, tau)
        keep &= S.data[entries] != 0.0
        return (
            S.data[entries][keep],
            S.indices[entries][keep],
            np.bincount(rowOfEntry[keep] - low, minlength=high - low),
        )

    nThreads = getNumberOfThreads()
    bounds = np.linspace(0, S.shape[0], nThreads + 1).astype(np.int64)
    parts = list(getThreadPool(nThreads).map(keptEntries, zip(bounds[:-1], bounds[1:])))
    data, indices, keptPerRow = (np.concatenate(v) for v in zip(*parts))
    return sp.csr_matrix((data, indices, np.concatenate([[0], np.cumsum(keptPerRow)])), shape=S.shape)


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


def restrictP1Map(isCorner: np.ndarray, edgeEndpoints: np.ndarray, remains: np.ndarray) -> tuple:
    """A field's P1 map (see :func:`~edelweissfe.numerics.p1topology.buildP1Map`) restricted to its remaining
    nodes ``remains``: the rows are renumbered, and a midside node whose edge endpoint is condensed becomes a corner.
    That is always safe: the P1 restriction takes a corner as it is, so a promoted node only enlarges the coarse space
    of the p-two-grid preconditioner, it never makes it wrong.

    Returns
    -------
    tuple
        ``(isCorner, edgeEndpoints)`` of the remaining nodes.
    """
    newRow = np.cumsum(remains) - 1
    isCorner, edgeEndpoints = isCorner[remains].copy(), edgeEndpoints[remains].copy()
    isMidside = ~isCorner
    endpointRemains = np.ones_like(isMidside)
    endpointRemains[isMidside] = remains[edgeEndpoints[isMidside]].all(axis=1)
    becomesCorner = isMidside & ~endpointRemains
    isCorner[becomesCorner] = True
    edgeEndpoints[becomesCorner] = -1
    stillMidside = ~isCorner
    edgeEndpoints[stillMidside] = newRow[edgeEndpoints[stillMidside]]
    return isCorner, edgeEndpoints


def blockLU(r: np.ndarray, condensation, preconditionS, solveA_bb=None) -> np.ndarray:
    """One application of the block LU preconditioner, see the module documentation:

    .. math::

        y_b = A_{bb}^{-1} r_b, \\qquad z_k = M_{\\tilde S}^{-1} (r_k - A_{kb} y_b), \\qquad
        z_b = A_{bb}^{-1} (r_b - A_{bk} z_k)

    or, with ``blockFactors="U"``, :math:`z_k = M_{\\tilde S}^{-1} r_k` (the remaining dofs first, then the condensed
    ones follow them: the bars follow the concrete). The ``__call__`` of :class:`SchurComplement`.

    Parameters
    ----------
    r
        The vector to precondition (all dofs).
    condensation
        The :class:`SchurComplementCondensation`: the dofs
        :math:`b, k`, the blocks :math:`A_{bk}, A_{kb}` and the block factors.
    preconditionS
        :math:`M_{\\tilde S}^{-1}`, applied to a vector of the remaining dofs :math:`k`.
    solveA_bb
        :math:`A_{bb}^{-1}`, default the condensation's own factorization.
    """
    solveA_bb = condensation.solveA_bb if solveA_bb is None else solveA_bb
    A_bk, A_kb = condensation.A_bk, condensation.A_kb
    b, k = condensation.condensedDofs, condensation.remainingDofs
    r_b, r_k = r[b], r[k]
    if condensation.blockFactors == "LU":
        y_b = solveA_bb(r_b)
        z_k = preconditionS(r_k - A_kb @ y_b)
    else:
        z_k = preconditionS(r_k)
    z_b = solveA_bb(r_b - A_bk @ z_k)
    z = np.empty_like(r)
    z[b], z[k] = z_b, z_k
    return z


class SchurComplement(Preconditioner):
    """The block LU preconditioner around the Schur complement of the condensed dofs; see the module documentation.

    Every solve selects the condensed dofs again and factorizes :math:`A_{bb}`; :math:`\\tilde S` and the child are
    rebuilt when a rebuild is asked for, and when the condensed dofs change. If :math:`A_{bb}` cannot be factorized
    (a bond that lost all stiffness), that solve does without the condensation: the child is built on the whole
    :math:`A`.

    Parameters
    ----------
    condense
        The markers whose marked elements' interior nodes are condensed (e.g. an element set marker).
    preconditionS
        :math:`M_{\\tilde S}^{-1}`, built on :math:`\\tilde S`.
    solveA_bb
        The exact solve with :math:`A_{bb}` (default: SuperLU).
    approximation, dropTolerance, windowNodes, haloNodes
        How :math:`\\tilde S` is formed, see
        :class:`SchurComplementCondensation`.
    blockFactors
        ``"U"`` or ``"LU"``.
    condensation
        A condensation to use instead of one created from ``condense`` (e.g. of given dofs, without a model). It must
        create its direct solvers through ``solveA_bb.newSolver``.
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
        condensation: SchurComplementCondensation = None,
    ):
        if preconditionS is None:
            raise ValueError("SchurComplement: preconditionS, the preconditioner of the Schur complement, is needed")
        requirePreconditioner("SchurComplement", "preconditionS", preconditionS)
        if solveA_bb is not None:
            requirePreconditioner("SchurComplement", "solveA_bb", solveA_bb)
        if blockFactors not in ("U", "LU"):
            raise ValueError("SchurComplement: blockFactors must be 'U' or 'LU', got {!r}".format(blockFactors))
        if condensation is None and not list(condense):
            raise ValueError(
                "SchurComplement: nothing to condense -- give the markers of the condensed elements (condense=[...]) "
                "or a condensation"
            )
        self.preconditionS = preconditionS
        self.solveA_bb = solveA_bb if solveA_bb is not None else DirectSolver("superlu")
        self.blockFactors = blockFactors
        # The condensation factorizes A_bb with the first solver it creates, which becomes solveA_bb's own (see
        # DirectSolver.newSolver); the second one is for the zones it corrects exactly.
        self.condensation = (
            condensation
            if condensation is not None
            else SchurComplementCondensation(
                list(condense),
                self.solveA_bb.newSolver,
                approximation=approximation,
                dropTolerance=dropTolerance,
                windowNodes=windowNodes,
                haloNodes=haloNodes,
                blockFactors=blockFactors,
            )
        )
        # a given condensation brings its own block factors
        self.blockFactors = self.condensation.blockFactors
        self.condensing = False
        self.S_tilde = None
        self._builtForDofs = None
        # the messages shown once per model
        self._messagesShown = set()

    def setModel(self, model, dofManager) -> None:
        self.condensation.setModel(model, dofManager)
        self._builtForDofs = None
        self._messagesShown = set()
        super().setModel(model, dofManager)

    def setUp(self, A, context: SetUpContext) -> None:
        self._condense(A, context, rebuild=True, change=None)

    def update(self, A, change) -> None:
        self._condense(A, change.context, rebuild=change.requiresRebuild, change=change)

    def _condense(self, A, context: SetUpContext, rebuild: bool, change) -> None:
        """Select the condensed dofs and factorize A_bb, then set up or update the child on S~ (or, without condensed
        dofs, on A)."""
        condensation = self.condensation
        try:
            condensation.update(A.matrix())
            condensing = condensation.hasCondensedDofs()
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
                context.message(
                    "info",
                    "SchurComplement: {:} dofs condensed; the approximate Schur complement of the {:} remaining dofs "
                    "has {:} nonzeros".format(dofs.size, A.shape[0] - dofs.size, self.S_tilde.nnz),
                )
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
        self._builtForDofs = dofs
        if change is None:
            self.preconditionS.setUp(childMatrix, childContext)
        else:
            childChange = change.withContext(childContext)
            if dofsChanged:
                childChange = childChange.withRebuild("condensed dofs")
            self.preconditionS.update(childMatrix, childChange)

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

    def __call__(self, r: np.ndarray) -> np.ndarray:
        if not self.condensing:
            return self.preconditionS(r)
        return blockLU(r, self.condensation, self.preconditionS, self.solveA_bb)

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
            parameters={"blockFactors": self.blockFactors, "S̃": "A_kk − A_kb A_bb⁻¹ A_bk, local"},
            children=(("solveA_bb", self.solveA_bb), ("preconditionS", self.preconditionS)),
            steps=steps,
        )
