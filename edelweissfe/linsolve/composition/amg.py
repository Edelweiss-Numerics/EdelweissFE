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
"""Algebraic multigrid on one field: :class:`AMG`, one AMGCL hierarchy, applied as one cycle per application.

Smoothed aggregation builds coarse levels from a *near null-space*: the error components a smoother such as
Chebyshev does not reduce, because the operator has (almost) no stiffness in their direction. For a displacement
field those are the rigid-body modes -- 3 translations and 3 rotations in 3D; for a scalar field the constant. In
the variables of a scaled block :math:`\\hat A_{ii} = D_i A_{ii} D_i` they are :math:`D_i^{-1}` times the physical
modes (see :mod:`~edelweissfe.linsolve.nullspace`).

AMGCL stores a 3D displacement field's hierarchy in node-wise 3x3 blocks by default -- faster per cycle, but a
block hierarchy takes the 3 translations as its near null-space at most. The rotations then need a coarse level
of their own, see :class:`~edelweissfe.linsolve.composition.twolevel.TwoLevel`.
"""

import copy

import numpy as np
import scipy.sparse as sp

import edelweissfe.utils.performancetiming as performancetiming
from edelweissfe.linsolve.composition.base import (
    Description,
    Preconditioner,
    SetUpContext,
)
from edelweissfe.linsolve.nullspace import (
    carriesRigidBodyModes,
    rigidBodyNullspace,
    translationNullspace,
)

#: The coarsening and the smoother every vector field's default tree shares, whatever its dimension (why each value,
#: and the measurements behind them: "Why the defaults are what they are" in
#: ``doc/source/documentation/fieldsplitmeasurements.rst``):
#:
#: - ``estimate_spectral_radius``: a sharper weight of the prolongation smoother of smoothed aggregation than the
#:   Gershgorin bound;
#: - Chebyshev of degree 16 after the coarse correction only (``npre`` 0, ``npost`` 1): for the same number of
#:   preconditioner applications, one high-degree post-smoothing sweep beat a symmetric V(1,1) cycle of low degree.
#:
#: Both spectral radii -- of the smoother and of the prolongation smoothing -- are estimated by ``spectralRadiusSteps``
#: Arnoldi steps (default 10) from a start vector seeded per row, the same at any thread count
#: (``edelweissfe/linsolve/amgcl/amgcl-spectral-radius.hpp``, "The spectral radius estimate" in
#: ``fieldsplitmeasurements.rst``); the trees leave the default.
_VECTOR_FIELD_TREE = {
    "coarsening": {
        "type": "smoothed_aggregation",
        "aggr": {"eps_strong": 0.01},
        "estimate_spectral_radius": True,
    },
    "relax": {"type": "chebyshev", "degree": 16, "lower": 0.01, "higher": 1.1},
    "npre": 0,
    "npost": 1,
}
_SCALAR_FIELD_TREE = {
    "coarsening": {"type": "smoothed_aggregation"},
    "relax": {"type": "chebyshev"},
}
#: The smoother of the corner-node two-level cycle, ``TwoLevel(coarseSpace=CornerNodeInterpolation())``: Chebyshev of
#: degree 8 on the interval from 0.03 to 1.1 times the spectral radius, on the quadratic field and on the rigid-body
#: coarse level of its default coarse solve (:func:`defaultCornerNodeTwoLevelTree`). A sweep of degree 8 costs half the
#: matrix products of one of degree 16, and the corner-node two-level cycle needs fewer than twice the outer iterations
#: with it. Measured ("The smoother of the corner-node two-level cycle" in
#: ``doc/source/documentation/fieldsplitmeasurements.rst``): 8 % less time on 14 captured systems, the same or less
#: on every one. ``AMG()`` and the rigid-body coarse level alone keep degree 16 (:data:`_VECTOR_FIELD_TREE`):
#: there, degree 8 gained nothing.
_CORNER_NODE_TWO_LEVEL_SMOOTHER = {"type": "chebyshev", "degree": 8, "lower": 0.03, "higher": 1.1}


def defaultFieldTree(dimension: int) -> dict:
    """The default AMGCL preconditioner tree of a field, from its nodal dimension: :data:`_VECTOR_FIELD_TREE` for a
    vector field, a Chebyshev-smoothed smoothed aggregation with AMGCL's defaults for a scalar field (e.g.
    ``nonlocal damage``), but the smoother's margin ``higher`` 1.1 above the estimated spectral radius, the default of
    EdelweissFE's Chebyshev smoother (see ``amgcl-chebyshev.hpp``). A copy: the trees are nested, and a caller may adapt the one it gets. The backend the tree is
    built on (block size, precision) is chosen apart from it, see :func:`backendOf`."""
    return copy.deepcopy(_VECTOR_FIELD_TREE if dimension > 1 else _SCALAR_FIELD_TREE)


def defaultCornerNodeTwoLevelTree() -> dict:
    """The default AMGCL tree of the corner-node two-level cycle on a vector field,
    ``TwoLevel(coarseSpace=CornerNodeInterpolation())``, and of the rigid-body coarse level of its default coarse solve
    (a scalar field keeps :func:`defaultFieldTree`): the default tree of a vector field
    (:data:`_VECTOR_FIELD_TREE`) with the smoother of the corner-node two-level cycle
    (:data:`_CORNER_NODE_TWO_LEVEL_SMOOTHER`). The corner-node two-level cycle reads only the smoother of its tree
    (``relax``, ``npre``, ``npost``), the rigid-body coarse level the whole tree. A copy, as :func:`defaultFieldTree`.
    """
    tree = defaultFieldTree(3)
    tree["relax"] = dict(_CORNER_NODE_TWO_LEVEL_SMOOTHER)
    return tree


#: The near null-spaces :class:`AMG` builds its hierarchy from, see :func:`backendOf`.
NEAR_NULLSPACES = ("rigidBodyModes", "translations", None)


def checkBackend(owner: str, backendBlockSize, backendPrecision, nearNullspace="rigidBodyModes") -> None:
    """Raise, naming ``owner``, for a backend AMGCL does not have, at construction (``None`` is the default of the
    field, see :func:`backendOf`): ``backendBlockSize`` 1 (scalar entries), 2 or 3 (node-wise blocks),
    ``backendPrecision`` ``"double"`` or ``"float"`` (2x2 blocks only in double), ``nearNullspace`` one of
    :data:`NEAR_NULLSPACES`."""
    if backendBlockSize not in (None, 1, 2, 3):
        raise ValueError("{:}: backendBlockSize must be 1, 2 or 3, got {!r}".format(owner, backendBlockSize))
    if backendPrecision not in (None, "double", "float"):
        raise ValueError("{:}: backendPrecision must be 'double' or 'float', got {!r}".format(owner, backendPrecision))
    if backendBlockSize == 2 and backendPrecision == "float":
        raise ValueError("{:}: 2x2 blocks are stored in double only, got backendPrecision 'float'".format(owner))
    if not (nearNullspace is None or (isinstance(nearNullspace, str) and nearNullspace in NEAR_NULLSPACES)):
        raise ValueError(
            "{:}: nearNullspace must be 'rigidBodyModes', 'translations' or None, got {!r}".format(owner, nearNullspace)
        )


def backendOf(
    owner: str, field, amgclTree: dict, backendBlockSize=None, backendPrecision=None, nearNullspace="rigidBodyModes"
):
    """The AMGCL backend a one-field component builds on for the field (:class:`AMG`,
    :class:`~edelweissfe.linsolve.composition.twolevel.TwoLevel`): what it was given, the field's default for what it
    was not -- the one place of those defaults.

    - ``backendBlockSize``: the field's dimension for a 2D or 3D field (node-wise blocks: less index and value traffic
      per product than scalar entries), else 1;
    - ``backendPrecision``: ``"float"`` on 3x3 blocks, which keeps the setup in double and stores only the matrices the
      cycle reads in float (measured faster at the same iterations), else ``"double"`` -- on the scalar backend float
      would build the whole hierarchy in single precision, its spectral-radius estimates included;
    - ``nearNullspace``, what smoothed aggregation starts from, in the variables of the scaled block (divided by
      :math:`D`, see :mod:`~edelweissfe.linsolve.nullspace`): ``"rigidBodyModes"`` (the default) the field's
      rigid-body modes as far as the backend and the field hold them -- translations and rotations of a displacement
      field on the scalar backend (:func:`~edelweissfe.linsolve.nullspace.carriesRigidBodyModes`), the translations on
      a block backend (which takes the translations at most) and for any other vector field; ``"translations"`` the
      translations :math:`D^{-1} t` (:func:`~edelweissfe.linsolve.nullspace.translationNullspace`); ``None`` none
      given, AMGCL's own: constant blocks on a block backend, which are :math:`D`-weighted translations, the constant
      on the scalar backend. A scalar field always takes AMGCL's constant.

    Raises, naming ``owner`` and the field, for blocks that do not match the field, and for a near null-space on a
    block backend without smoothed aggregation.

    Returns
    -------
    tuple
        ``(backendBlockSize, backendPrecision, nearNullspace)``, the near null-space as built: ``"rigidBodyModes"``,
        ``"translations"`` or ``None``.
    """
    blockSize = (
        backendBlockSize if backendBlockSize is not None else (field.dimension if field.dimension in (2, 3) else 1)
    )
    precision = backendPrecision if backendPrecision is not None else ("float" if blockSize == 3 else "double")
    if field.dimension == 1 or nearNullspace is None:
        nearNullspace = None
    elif nearNullspace == "rigidBodyModes" and (blockSize > 1 or not carriesRigidBodyModes(field)):
        nearNullspace = "translations"
    if blockSize > 1 and blockSize != field.dimension:
        raise ValueError(
            "{:} on the field '{:}': backendBlockSize {:} does not match the field's dimension {:}".format(
                owner, field.name, blockSize, field.dimension
            )
        )
    if blockSize == 2 and precision == "float":
        raise ValueError("{:} on the field '{:}': 2x2 blocks are stored in double only".format(owner, field.name))
    # AMGCL's defaults: class amg, coarsening.type smoothed_aggregation
    coarsening = amgclTree.get("coarsening", {})
    coarseningType = coarsening.get("type", "smoothed_aggregation") if isinstance(coarsening, dict) else None
    onBlocks = blockSize > 1 and nearNullspace is not None
    if onBlocks and (amgclTree.get("class", "amg") != "amg" or coarseningType != "smoothed_aggregation"):
        raise ValueError(
            "{:} on the field '{:}': the near null-space on a block backend (nearNullspace, the translations by "
            "default) needs smoothed aggregation (coarsening.type); set nearNullspace=None for another "
            "coarsening".format(owner, field.name)
        )
    return blockSize, precision, nearNullspace


def isSignificant(values, rows, cols, sqrtAbsDiagonal: np.ndarray, tau: float) -> np.ndarray:
    """Whether the entries ``values`` at the positions ``(rows, cols)`` pass the relative drop test of AMG's strength
    of connection,

    .. math::

        \\lvert a_{ij} \\rvert \\geq \\tau \\sqrt{\\lvert a_{ii} a_{jj} \\rvert},

    with ``sqrtAbsDiagonal`` holding :math:`\\sqrt{\\lvert a_{ii} \\rvert}`. The one definition of the test: the
    filtered matrix of :func:`droppedSmallEntries` and the drop and thinning of the approximate Schur complement
    (:mod:`~edelweissfe.linsolve.composition.schurcomplement`) use it. ``rows`` and ``cols`` broadcast against
    ``values``: the row and the column of every stored entry of a sparse matrix, or ``rows[:, None]`` and
    ``cols[None, :]`` for a dense block.

    Returns
    -------
    np.ndarray
        The keep mask, of the shape of ``values``.
    """
    return np.abs(values) >= tau * sqrtAbsDiagonal[rows] * sqrtAbsDiagonal[cols]


def droppedSmallEntries(A: sp.csr_matrix, dropTolerance: float, lumpDroppedEntries: bool = False) -> sp.csr_matrix:
    """The *filtered* matrix :math:`\\tilde A` of ``A`` that a preconditioner is built on: the off-diagonal entries
    that fail the relative drop test (:func:`isSignificant`) at :math:`\\tau` = ``dropTolerance`` dropped. The diagonal
    is always kept, so no row is emptied. Shared by :class:`AMG` and
    :class:`~edelweissfe.linsolve.composition.twolevel.TwoLevel`.

    Why: some operators (e.g. of RKPM) store many entries per row whose median off-diagonal is orders of magnitude
    below the diagonal -- numerically far sparser than structurally. Smoothed aggregation then weighs every one of
    them per row, and the Galerkin products carry them to every coarse level. Only the preconditioner is built on
    :math:`\\tilde A`; the Krylov solve still applies :math:`A`, so the converged solution is that of :math:`A`, and
    only the number of iterations to reach it changes. The criterion is scale-invariant and symmetric in
    :math:`(i, j)`: a symmetric matrix stays symmetric, as smoothed aggregation and the Chebyshev smoother assume.

    Parameters
    ----------
    A
        The matrix.
    dropTolerance
        :math:`\\tau`.
    lumpDroppedEntries
        Add the dropped entries of each row onto its diagonal instead of discarding them: the row sums are kept,
        :math:`\\tilde A \\mathbf 1 = A \\mathbf 1`, so the constant near null-space vector survives (the "filtered
        matrix" of the AMG literature). For elasticity, whose off-diagonals are mostly negative, this weakens the
        diagonal dominance; whether that helps depends on the operator.
    """
    coo = A.tocoo()
    sqrtAbsDiagonal = np.sqrt(np.abs(A.diagonal()))
    keep = (coo.row == coo.col) | isSignificant(coo.data, coo.row, coo.col, sqrtAbsDiagonal, dropTolerance)
    filtered = sp.coo_matrix((coo.data[keep], (coo.row[keep], coo.col[keep])), shape=A.shape).tocsr()
    if lumpDroppedEntries:
        droppedRowSums = np.bincount(coo.row[~keep], weights=coo.data[~keep], minlength=A.shape[0])
        filtered = (filtered + sp.diags(droppedRowSums, format="csr", dtype=float)).tocsr()
    return filtered


def preconditionerMatrix(A, dropTolerance: float, lumpDroppedEntries: bool, statistics) -> sp.csr_matrix:
    """The matrix a one-field preconditioner (:class:`AMG`,
    :class:`~edelweissfe.linsolve.composition.twolevel.TwoLevel`) is built on: :math:`A` itself, or with
    ``dropTolerance > 0`` its filtered copy (:func:`droppedSmallEntries`), with the stored entries before and after
    recorded in the component's ``statistics``."""
    matrix = A.matrix()
    if dropTolerance <= 0.0:
        return matrix
    with performancetiming.timeit("diagonal blocks"):
        filtered = droppedSmallEntries(matrix, dropTolerance, lumpDroppedEntries)
    statistics.sizes.update({"nnz": matrix.nnz, "nnz kept": filtered.nnz})
    return filtered


def fieldTreeOf(owner: str, amgclTree: "dict | None", field, defaultTree) -> dict:
    """The AMGCL tree a one-field component builds on: ``amgclTree``, or the component's default for the field,
    ``defaultTree(field)`` (built only then); raises, naming ``owner`` and the field, if ``amgclTree`` is not a
    tree."""
    if amgclTree is None:
        return defaultTree(field)
    if not isinstance(amgclTree, dict):
        raise ValueError(
            "{:} on the field '{:}': its amgclTree must be an AMGCL preconditioner tree (a dict), got {!r}".format(
                owner, field.name, amgclTree
            )
        )
    return amgclTree


def buildFieldHierarchy(block, amgclTree: dict, backend: tuple, scaling, coords, diagonalBlock):
    """One AMGCL hierarchy for a field's scaled diagonal block :math:`\\hat A_{ii}` (``diagonalBlock``, a callable
    that returns it, so that it is cut out only now), with the field's near null-space.

    The near null-space is the one :func:`backendOf` resolved: the rigid-body modes, with ``coords`` (else the
    translations), the translations, or none given (AMGCL's constant blocks, or its constant).

    Parameters
    ----------
    block
        The field's :class:`~edelweissfe.linsolve.base.FieldBlock`.
    amgclTree
        The field's AMGCL tree, forwarded verbatim.
    backend
        ``(backendBlockSize, backendPrecision, nearNullspace)``, see :func:`backendOf`.
    scaling
        The field's scaling :math:`D_i`.
    coords
        The field's node coordinates for its rigid-body rotations, or ``None``.
    diagonalBlock
        ``() -> scipy CSR``, the scaled diagonal block.

    Returns
    -------
    PyAMGCLSolver
        The hierarchy, applied by its ``applyPreconditioner``.
    """
    from edelweissfe.linsolve.amgcl.amgcl import PyAMGCLSolver

    backendBlockSize, backendPrecision, nearNullspace = backend
    hierarchy = PyAMGCLSolver(
        {
            "precond": amgclTree,
            "backendPrecision": backendPrecision,
            "backendBlockSize": backendBlockSize,
        }
    )
    if nearNullspace is not None:
        with performancetiming.timeit("nullspace construction"):
            if nearNullspace == "rigidBodyModes" and coords is not None:
                nullspace = rigidBodyNullspace(block, coords, scaling)
            else:
                nullspace = translationNullspace(block, scaling)
        hierarchy.set_nullspace(nullspace)
    matrix = diagonalBlock()
    with performancetiming.timeit("{:} AMG build".format(block.name)):
        hierarchy.build(matrix)
    return hierarchy


class AMG(Preconditioner):
    """One AMGCL hierarchy on the matrix of a field, applied as one cycle: :math:`z = \\mathrm{AMG}(A)^{-1} r`.

    Reused while only the values of the matrix change: the hierarchy is built again only if a rebuild is asked for
    (see :class:`~edelweissfe.linsolve.composition.base.MatrixChange`).

    Parameters
    ----------
    amgclTree
        The AMGCL parameter tree, forwarded verbatim; ``None``: the default tree of the field's dimension
        (:func:`defaultFieldTree`).
    backendBlockSize, backendPrecision
        The AMGCL backend: scalar entries (1) or node-wise blocks (2, 3), stored in ``"double"`` or ``"float"``;
        ``None``: the field's default, see :func:`backendOf`.
    nearNullspace
        What smoothed aggregation starts from: ``"rigidBodyModes"`` (the default; the rotations of a displacement field
        on the scalar backend need node coordinates, a block backend takes the translations), ``"translations"``, or
        ``None`` (AMGCL's own); see :func:`backendOf`.
    dropTolerance, lumpDroppedEntries
        Build the hierarchy on the filtered matrix, see :func:`droppedSmallEntries`; ``0`` (the default) builds it
        on the matrix itself.
    """

    def __init__(
        self,
        amgclTree: dict = None,
        backendBlockSize: int = None,
        backendPrecision: str = None,
        nearNullspace: "str | None" = "rigidBodyModes",
        dropTolerance: float = 0.0,
        lumpDroppedEntries: bool = False,
    ):
        if not dropTolerance >= 0.0:
            raise ValueError("AMG: dropTolerance must not be negative, got {!r}".format(dropTolerance))
        checkBackend("AMG", backendBlockSize, backendPrecision, nearNullspace)
        super().__init__()
        self.amgclTree = amgclTree
        self.backendBlockSize = backendBlockSize
        self.backendPrecision = backendPrecision
        self.nearNullspace = nearNullspace
        self.dropTolerance = dropTolerance
        self.lumpDroppedEntries = lumpDroppedEntries

    def _forget(self) -> None:
        self.hierarchy = None
        self._field = None
        # the near null-space the hierarchy was built with, as far as the backend and the field hold the one asked for
        self._builtNearNullspace = None

    def _setUp(self, A, context: SetUpContext) -> None:
        field = context.onlyField("AMG")
        self._field = field
        amgclTree = fieldTreeOf("AMG", self.amgclTree, field, lambda field: defaultFieldTree(field.dimension))
        backend = backendOf("AMG", field, amgclTree, self.backendBlockSize, self.backendPrecision, self.nearNullspace)
        coords = context.coordinates(field.name) if backend[2] == "rigidBodyModes" else None
        # the rotations need the node coordinates; without them, the hierarchy takes the translations
        self._builtNearNullspace = "translations" if backend[2] == "rigidBodyModes" and coords is None else backend[2]
        self.hierarchy = buildFieldHierarchy(
            field,
            amgclTree,
            backend,
            context.scalingOf(slice(None)),
            coords,
            lambda: preconditionerMatrix(A, self.dropTolerance, self.lumpDroppedEntries, self.statistics),
        )
        self.statistics.sizes["rows"] = A.shape[0]

    def _update(self, A, change) -> None:
        if change.requiresRebuild or self.hierarchy is None:
            self.setUp(A, change.context)

    def _apply(self, r):
        return self.hierarchy.applyPreconditioner(r)

    def describe(self) -> Description:
        backend = {
            "backendBlockSize": self.backendBlockSize,
            "backendPrecision": self.backendPrecision,
        }
        parameters = {"amgclTree": "default" if self.amgclTree is None else "given"}
        parameters.update({key: value for key, value in backend.items() if value is not None})
        # the near null-space as built, once it is, where it differs from the one asked for (a block backend takes the
        # translations at most); before, the one asked for, where it is not the default
        nearNullspace = self._builtNearNullspace if self._field is not None else self.nearNullspace
        if nearNullspace != self.nearNullspace or self.nearNullspace != "rigidBodyModes":
            parameters["nearNullspace"] = nearNullspace
        if self.dropTolerance > 0.0:
            parameters["dropTolerance"] = self.dropTolerance
            parameters["lumpDroppedEntries"] = self.lumpDroppedEntries
        name = self._field.name if self._field is not None else "A"
        return Description(
            name="AMG",
            formula="AMG(" + name + ")⁻¹",
            latex="\\mathrm{AMG}(A_{\\text{" + name + "}})^{-1}",
            parameters=parameters,
        )
