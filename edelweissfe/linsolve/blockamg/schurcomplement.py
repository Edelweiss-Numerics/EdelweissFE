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

"""Schur complement condensation of selected dofs, an option of ``blockamg`` (``schurComplement``).

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
bars as seen through their bond -- on which ``blockamg`` builds its field-split AMG as usual. Gaussian elimination by
blocks is the block LU factorization of :math:`A`, and its inverse is a preconditioner of the outer Krylov solve:

.. math::

    y_b = A_{bb}^{-1} r_b, \\quad z_k = M_{\\tilde S}^{-1} (r_k - A_{kb} y_b), \\quad
    z_b = A_{bb}^{-1} (r_b - A_{bk} z_k),

with :math:`M_{\\tilde S}^{-1}` the field-split sweep of ``blockamg`` on an approximation :math:`\\tilde S` of
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

import numbers
from collections.abc import Callable, Mapping
from typing import NamedTuple

import numpy as np
import scipy.sparse as sp
from scipy.linalg import solve_banded
from scipy.sparse.csgraph import connected_components, reverse_cuthill_mckee

import edelweissfe.utils.performancetiming as performancetiming
from edelweissfe.adaptivity.marking import RefineableElements
from edelweissfe.linsolve.base import FieldBlock
from edelweissfe.linsolve.blockamg.sparsity import SparsityPattern, isSignificant
from edelweissfe.numerics.parallelizationutilities import (
    getNumberOfThreads,
    getThreadPool,
)

#: The configuration keys of ``schurComplement`` and their defaults (``markers`` is required).
DEFAULT_OPTIONS = {
    "markers": None,
    "approximation": "local",
    "dropTolerance": 1e-4,
    "windowNodes": 50,
    "haloNodes": None,
    "directSolver": {"linsolver": "superlu"},
    "blockFactors": "U",
}

#: A component of :math:`A_{bb}` is a chain (a bar) if its bandwidth, in the reverse Cuthill-McKee ordering, is at
#: most a window's core divided by this: a chain couples each node only to a few neighbours along it, while a zone of
#: a 2D or 3D mesh couples it across the zone's width.
CHAIN_BANDWIDTH_DIVISOR = 4

#: The number of right-hand sides per batch of the exact correction of a zone (:meth:`_exactCorrection`) ...
EXACT_CORRECTION_BATCH = 256
#: ... at most, and fewer when the zones are so large that a batch would exceed this many entries.
EXACT_CORRECTION_MAX_ENTRIES = 2**25


def optionsRejection(options) -> "str | None":
    """Why ``schurComplement`` options cannot be used, or ``None``. Checks what can be checked without the model.

    Parameters
    ----------
    options
        The ``schurComplement`` section of the ``blockamg`` configuration.
    """
    if not isinstance(options, Mapping):
        return "must be a JSON object"
    unknown = sorted(set(options) - set(DEFAULT_OPTIONS))
    if unknown:
        return "unknown key(s) {:}; known are {:}".format(unknown, sorted(DEFAULT_OPTIONS))
    markers = options.get("markers")
    if not isinstance(markers, list) or not markers or not all(isinstance(m, Mapping) for m in markers):
        return "'markers' must be a non-empty list of marker objects, each with a 'type'"
    if any("initialOnly" in m for m in markers):
        return "'initialOnly' has no meaning for markers of schurComplement: they are evaluated on every solve"
    if options.get("approximation", "local") not in ("local", "none"):
        return "'approximation' must be 'local' or 'none'"
    dropTolerance = options.get("dropTolerance", DEFAULT_OPTIONS["dropTolerance"])
    if not _isReal(dropTolerance) or not 0.0 <= dropTolerance < 1.0:
        return "'dropTolerance' must be a number of at least 0 and less than 1 (else the diagonal would be dropped)"
    windowNodes, haloNodes = options.get("windowNodes", DEFAULT_OPTIONS["windowNodes"]), options.get("haloNodes")
    if not _isInteger(windowNodes) or windowNodes < 1:
        return "'windowNodes' must be an integer of at least 1"
    if haloNodes is not None and (not _isInteger(haloNodes) or haloNodes < 0):
        return "'haloNodes' must be an integer, not negative (or null for windowNodes)"
    if options.get("blockFactors", "U") not in ("U", "LU"):
        return "'blockFactors' must be 'U' or 'LU'"
    directSolver = options.get("directSolver", DEFAULT_OPTIONS["directSolver"])
    if not isinstance(directSolver, Mapping) or "linsolver" not in directSolver:
        return "'directSolver' must be an object with a 'linsolver' name (and optional 'options')"
    return _directSolverRejection(directSolver)


def _isInteger(value) -> bool:
    """Whether ``value`` is an integer (a JSON integer), not a bool, a float or a string."""
    return isinstance(value, numbers.Integral) and not isinstance(value, bool)


def _isReal(value) -> bool:
    """Whether ``value`` is a real number (a JSON number), not a bool or a string."""
    return isinstance(value, numbers.Real) and not isinstance(value, bool)


def _directSolverRejection(directSolver: Mapping) -> "str | None":
    """Why the ``directSolver`` of ``schurComplement`` cannot be used, or ``None``: it must be a registered linear
    solver that factorizes once for several right-hand sides, i.e. whose class overrides both
    :meth:`~edelweissfe.linsolve.base.LinearSolver.factorize` and
    :meth:`~edelweissfe.linsolve.base.LinearSolver.solveFactorized` of the base class, which only raise."""
    from edelweissfe.config.linsolve import getLinSolverByName
    from edelweissfe.linsolve.base import LinearSolver

    name = directSolver["linsolver"]
    try:
        solver = getLinSolverByName(name, dict(directSolver.get("options", {})))
    except (ImportError, LookupError, ValueError, TypeError) as error:
        return "'directSolver' {!r} cannot be created: {:}".format(name, error)
    solverClass = type(solver)
    if solverClass.factorize is LinearSolver.factorize or solverClass.solveFactorized is LinearSolver.solveFactorized:
        return (
            "'directSolver' {!r} does not factorize once for several right-hand sides (factorize and solveFactorized), "
            "which the condensed block needs: use superlu or pardiso".format(name)
        )
    return None


def createCondensation(options) -> "SchurComplementCondensation":
    """The condensation of the ``schurComplement`` options (see :func:`optionsRejection`).

    The markers are created through the marker registry (:mod:`edelweissfe.config.markerlibrary`) from exactly the
    options of a ``>>marker`` line, the direct solvers through the linear solver registry (SuperLU by default: its
    back substitution, once or twice per outer iteration, is cheap for the chains of bars).
    """
    from edelweissfe.config.linsolve import getLinSolverByName
    from edelweissfe.config.markerlibrary import getMarkerClass

    reason = optionsRejection(options)
    if reason is not None:
        raise ValueError("blockamg: schurComplement: " + reason)
    options = DEFAULT_OPTIONS | dict(options)

    markers = []
    for markerOptions in options["markers"]:
        markerOptions = dict(markerOptions)
        markerType = markerOptions.pop("type", None)
        if markerType is None:
            raise ValueError("blockamg: schurComplement: every marker needs a 'type'")
        markers.append(getMarkerClass(markerType).fromOptions(markerOptions))

    def createDirectSolver():
        return getLinSolverByName(
            options["directSolver"]["linsolver"], dict(options["directSolver"].get("options", {}))
        )

    windowNodes = int(options["windowNodes"])
    return SchurComplementCondensation(
        markers,
        createDirectSolver,
        approximation=options["approximation"],
        dropTolerance=float(options["dropTolerance"]),
        windowNodes=windowNodes,
        haloNodes=windowNodes if options["haloNodes"] is None else int(options["haloNodes"]),
        blockFactors=options["blockFactors"],
    )


class SchurComplementCondensation:
    """The condensed dofs :math:`b`, the factorization of :math:`A_{bb}`, the approximate Schur complement
    :math:`\\tilde S` of all other dofs :math:`k`, and the preconditioner; see the module documentation.

    ``blockamg`` uses it in three steps: :meth:`update` on every solve (select :math:`b`, split :math:`A`,
    factorize :math:`A_{bb}`), :meth:`approximateSchurComplement` whenever it rebuilds its hierarchies (they are
    built on :math:`\\tilde S` instead of :math:`A`), and :meth:`precondition` in every outer iteration.

    Parameters
    ----------
    markers
        The markers whose marked elements' interior nodes are condensed.
    createDirectSolver
        Creates a linear solver that factorizes once for several right-hand sides
        (:meth:`~edelweissfe.linsolve.base.LinearSolver.factorize`,
        :meth:`~edelweissfe.linsolve.base.LinearSolver.solveFactorized`): one for :math:`A_{bb}`, one for the zones
        corrected exactly.
    approximation
        ``"local"``: :math:`\\tilde S` as described in the module documentation; ``"none"``:
        :math:`\\tilde S = A_{kk}` (the coupling through :math:`b` is ignored).
    dropTolerance
        The relative size :math:`\\tau` below which entries of :math:`\\tilde S` are dropped.
    windowNodes, haloNodes
        The nodes of a window's core along a chain, and of its halo on either side.
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
        haloNodes: int = 50,
        blockFactors: str = "U",
    ):
        self._markers = markers
        self._directSolver = createDirectSolver()
        self._zoneSolver = createDirectSolver()
        self._approximation = approximation
        self._dropTolerance = dropTolerance
        self._windowNodes = windowNodes
        self._haloNodes = haloNodes
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
        # per field: which of its nodes remain (are not condensed), see remainingFieldBlocks
        self._remainingNodes = {}
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
        blocks = (None, None, None)
        if dofs.size > 0:
            with performancetiming.timeit("schur split"):
                splitCache = _splitCacheFor(A, splitCache)
                blocks = _split(A, ("bb", "bk", "kb"), dofs, remainingDofs, splitCache)
            with performancetiming.timeit("schur factorization"):
                self._directSolver.factorize(blocks[0])
        self.selectedAllDofs = selectedAllDofs
        self.condensedDofs, self._nodeOfCondensedDof, self.remainingDofs = dofs, nodeOfDof, remainingDofs
        self._splitCache = splitCache
        self.A_bb, self.A_bk, self.A_kb = blocks

    def precondition(self, r: np.ndarray, preconditionRemaining: Callable) -> np.ndarray:
        """One application of the preconditioner (the inverse of the block factors ``blockFactors`` of :math:`A`),
        see the module documentation.

        Parameters
        ----------
        r
            The vector to precondition (all dofs).
        preconditionRemaining
            :math:`M_{\\tilde S}^{-1}`, applied to a vector of the remaining dofs k.

        Returns
        -------
        np.ndarray
            :math:`z \\approx A^{-1} r`.
        """
        b, k = self.condensedDofs, self.remainingDofs
        solveCondensed = self._directSolver.solveFactorized
        z = np.empty_like(r)
        # the remaining dofs first: the field-split sweep on S~; with "LU", for the residual that the condensed dofs,
        # solved alone, leave on them
        if self._blockFactors == "LU":
            z[k] = preconditionRemaining(r[k] - self.A_kb @ solveCondensed(r[b]))
        else:
            z[k] = preconditionRemaining(r[k])
        # then the condensed dofs follow them (the bars follow the concrete)
        z[b] = solveCondensed(r[b] - self.A_bk @ z[k])
        return z

    # ----- the remaining dofs, on which blockamg builds its field split -----

    def remainingFieldBlocks(self, blocks: list) -> list:
        """The field blocks of the remaining dofs: every field without its condensed dofs, in the same order (and
        contiguous, as the remaining dofs keep their order). A field whose dofs are all condensed (e.g. the rotations
        of embedded beams) has no block. Also records which nodes of every field remain, see
        :meth:`remainingNodes`."""
        remainingBlocks, start = [], 0
        self._remainingNodes = {}
        for block in blocks:
            inBlock = self.condensedDofs[(self.condensedDofs >= block.start) & (self.condensedDofs < block.stop)]
            n = (block.stop - block.start) - inBlock.size
            if n == 0:
                continue
            remainingBlocks.append(FieldBlock(block.name, start, start + n, block.dimension))
            start += n
            # condensation takes whole nodes, and a field's dofs are node-major
            remains = np.ones((block.stop - block.start) // block.dimension, dtype=bool)
            remains[(inBlock - block.start) // block.dimension] = False
            self._remainingNodes[block.name] = remains
        return remainingBlocks

    def remainingNodes(self, fieldName: str) -> "np.ndarray | None":
        """Which nodes of the field ``fieldName`` remain, as a boolean mask over the field's nodes (in the field's
        dof order), or ``None`` if no block of that name remains."""
        return self._remainingNodes.get(fieldName)

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
        sqrtDiagonalOfA_kk = np.sqrt(np.abs(A_kk.diagonal()))
        # The correction of every window or batch is thinned against A_kk's diagonal at a tenth of the drop tolerance
        # before the parts are summed, which bounds the memory; the final drop is against the diagonal of S~ itself.
        thinningTolerance = 0.1 * tau

        windows, zoneDofs = self._windowsAndZones()
        self._A_kb_byColumns = self.A_kb.tocsc()
        parts = []
        with performancetiming.timeit("schur windows"):
            pool = getThreadPool(getNumberOfThreads())
            for part in pool.map(
                lambda window: self._windowCorrection(window, sqrtDiagonalOfA_kk, thinningTolerance), windows
            ):
                if part is not None:
                    parts.append(part)
        with performancetiming.timeit("schur exact correction"):
            parts += self._exactCorrection(zoneDofs, sqrtDiagonalOfA_kk, thinningTolerance)

        with performancetiming.timeit("schur assembly"):
            if parts:
                rows, cols, values = (np.concatenate(v) for v in zip(*parts))
                S = (A_kk - sp.csr_matrix((values, (rows, cols)), shape=A_kk.shape)).tocsr()
            else:
                S = A_kk.copy()
            return _dropSmallEntries(S, tau)

    def _windowsAndZones(self) -> tuple:
        """The windows along the chains of :math:`A_{bb}`, and the dofs of its zones.

        Each connected component of :math:`A_{bb}` is ordered by the reverse Cuthill-McKee ordering (along the bar,
        for a bar). A banded component (see :data:`CHAIN_BANDWIDTH_DIVISOR`) is a chain: it is cut into cores of
        ``windowNodes`` nodes, each with a halo of ``haloNodes`` nodes on either side; one no longer than a core with
        both halos is a single, exact window. Any other component is a zone (of a 2D or 3D mesh), corrected exactly,
        see :meth:`_exactCorrection`.

        Returns
        -------
        tuple
            ``(windows, zoneDofs)``: ``(dofs of the window, positions of its core in the window)`` for every window
            (the cores partition the dofs of the chains), and the dofs of the zones (positions in b).
        """
        A_bb = self.A_bb
        order = reverse_cuthill_mckee(A_bb, symmetric_mode=True)
        nComponents, componentOf = connected_components(A_bb, directed=False)
        # the dofs component by component, each component in the reverse Cuthill-McKee order
        alongComponents = order[np.argsort(componentOf[order], kind="stable")]
        componentSizes = np.bincount(componentOf, minlength=nComponents)
        # the bandwidth of each component in that order
        rank = np.empty_like(order)
        rank[order] = np.arange(order.size)
        coo = A_bb.tocoo()
        bandwidth = np.zeros(nComponents, dtype=np.int64)
        np.maximum.at(bandwidth, componentOf[coo.row], np.abs(rank[coo.row] - rank[coo.col]))

        windows, zoneDofs = [], []
        for component, dofs in enumerate(np.split(alongComponents, np.cumsum(componentSizes)[:-1])):
            # windows count nodes; all nodes of a chain carry the same dofs
            dofsPerNode = max(1, round(dofs.size / np.unique(self._nodeOfCondensedDof[dofs]).size))
            core, halo = self._windowNodes * dofsPerNode, self._haloNodes * dofsPerNode
            n = dofs.size
            if bandwidth[component] > core // CHAIN_BANDWIDTH_DIVISOR:
                zoneDofs.append(dofs)
            elif n <= core + 2 * halo:
                windows.append((dofs, np.arange(n)))
            else:
                for start in range(0, n, core):
                    stop = min(start + core, n)
                    low, high = max(0, start - halo), min(n, stop + halo)
                    windows.append((dofs[low:high], np.arange(start - low, stop - low)))
        zoneDofs = np.sort(np.concatenate(zoneDofs)) if zoneDofs else np.empty(0, dtype=np.int64)
        return windows, zoneDofs

    def _windowCorrection(self, window: tuple, sqrtDiagonalOfA_kk: np.ndarray, thinningTolerance: float):
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
        dofs, core = window
        A_bk_window = self.A_bk[dofs]
        coupledToWindow = np.unique(A_bk_window.indices)
        A_kb_core = self._A_kb_byColumns[:, dofs[core]]
        coupledToCore = np.unique(A_kb_core.indices)
        if len(coupledToWindow) == 0 or len(coupledToCore) == 0:
            return None
        # the core rows of A_ww^-1 = the transposed solution of A_ww^T X = I[:, core]
        unitColumns = np.zeros((len(dofs), len(core)))
        unitColumns[core, np.arange(len(core))] = 1.0
        inverseRowsOfCore = _solveTransposed(self.A_bb[dofs][:, dofs].tocoo(), unitColumns).T
        # the product above; transposed only because SciPy's sparse factor (A_bk) must multiply from the left
        coreTimesInverse = A_kb_core[coupledToCore] @ inverseRowsOfCore
        correction = (A_bk_window[:, coupledToWindow].T @ coreTimesInverse.T).T
        return _significantEntries(correction, coupledToCore, coupledToWindow, sqrtDiagonalOfA_kk, thinningTolerance)

    def _exactCorrection(self, zoneDofs: np.ndarray, sqrtDiagonalOfA_kk: np.ndarray, thinningTolerance: float) -> list:
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
            parts.append(_significantEntries(correction, coupledRows, columns, sqrtDiagonalOfA_kk, thinningTolerance))
        return parts


class _BlockGather(NamedTuple):
    """Where the entries of one block of ``A`` (e.g. :math:`A_{bk}`) are in ``A.data``, and the block's CSR structure:
    the block of every matrix of the same sparsity pattern is ``csr_matrix((A.data[indexInData], indices, indptr),
    shape)``."""

    indexInData: np.ndarray
    indices: np.ndarray
    indptr: np.ndarray
    shape: tuple


class _SplitCache(NamedTuple):
    """The gathers of the blocks :func:`_split` cut out of the matrices of one sparsity pattern, by block name, for one
    choice of the condensed dofs."""

    pattern: SparsityPattern
    blocks: dict


def _splitCacheFor(A: sp.csr_matrix, cache: "_SplitCache | None") -> _SplitCache:
    """The cache of :func:`_split` for ``A``: ``cache`` if it was made for ``A``'s sparsity pattern, else an empty one
    for it (made, too, when ``cache`` is ``None``: a new choice of the condensed dofs)."""
    if cache is not None and cache.pattern.matches(A):
        return cache
    return _SplitCache(SparsityPattern.of(A), {})


def _split(
    A: sp.csr_matrix, which: tuple, condensedDofs: np.ndarray, remainingDofs: np.ndarray, cache: _SplitCache
) -> list:
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
        rowDofs, colDofs = dofsOf[name[0]], dofsOf[name[1]]
        if name not in cache.blocks:
            # index + 1: a zero value could be dropped by the slicing
            entryIndexInData = sp.csr_matrix(
                (np.arange(1, A.nnz + 1, dtype=np.float64), A.indices, A.indptr), shape=A.shape
            )
            block = entryIndexInData[rowDofs][:, colDofs].tocsr()
            block.sort_indices()
            cache.blocks[name] = _BlockGather(block.data.astype(np.int64) - 1, block.indices, block.indptr, block.shape)
        gather = cache.blocks[name]
        blocks.append(sp.csr_matrix((A.data[gather.indexInData], gather.indices, gather.indptr), shape=gather.shape))
    return blocks


def _significantEntries(
    correction: np.ndarray, rows: np.ndarray, cols: np.ndarray, sqrtAbsDiagonal: np.ndarray, tolerance: float
) -> tuple:
    """The entries of a dense correction block (rows ``rows``, columns ``cols`` of k) that pass the relative drop test
    at ``tolerance`` (:func:`~edelweissfe.linsolve.blockamg.sparsity.isSignificant`, against the diagonal of
    :math:`A_{kk}`), as ``(rows, cols, values)``."""
    significant = isSignificant(correction, rows[:, None], cols[None, :], sqrtAbsDiagonal, tolerance)
    i, j = np.nonzero(significant)
    return rows[i], cols[j], correction[i, j]


def _dropSmallEntries(S: sp.csr_matrix, tau: float) -> sp.csr_matrix:
    """``S`` without the entries that fail the relative drop test :math:`|S_{ij}| \\geq \\tau \\sqrt{|S_{ii} S_{jj}|}`
    (:func:`~edelweissfe.linsolve.blockamg.sparsity.isSignificant`) and without stored zeros.

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
    a bar), else by a dense one.

    A window along a bar has a bandwidth of a few dofs, so the banded solve costs a small fraction of the dense one. If
    the band holds more than a quarter of the columns (a short window), the band storage no longer pays."""
    n = A.shape[0]
    lower = int(max(0, (A.row - A.col).max(initial=0)))
    upper = int(max(0, (A.col - A.row).max(initial=0)))
    if lower + upper + 1 > n // 4:
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
