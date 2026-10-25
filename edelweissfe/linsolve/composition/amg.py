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
from edelweissfe.linsolve.nullspace import rigidBodyNullspace, translationNullspace

# "backendPrecision" and "backendBlockSize" are not AMGCL parameters -- they select the AMGCL
# wrapper's own backend value type ("double" or "float" for backendPrecision; 1 (scalar), 2, or 3 for a
# block-valued backend operating on B x B nodal blocks for backendBlockSize). Omitted from a field
# tree they fall back to "double" and 1; the default trees (defaultFieldTree) set both per dimension.
# buildFieldHierarchy pops both, and "blockTranslationNullspace" (see below), out of whichever precond
# dict applies (default or a field tree of a script) before forwarding the rest verbatim as the AMGCL JSON
# parameter tree.
#
# backendPrecision defaults to "double" when omitted from a field tree -- not what the shipped
# field defaults use: defaultFieldTree's own 3D tree sets it to "float" explicitly (see below). On
# the scalar backend "float" is AMGCL's builtin<float>,
# which builds the whole hierarchy in single precision; measured across a set of dumped systems of a
# real reference model, it inflated outer GMRES iterations and only won on wall-clock part of the
# time, netting only a small aggregate gain -- plausibly because the Chebyshev smoother's own
# power-iteration spectral-radius estimate then runs in float32 too. With backendBlockSize 3, "float"
# instead sets the hierarchy up in double and only stores the level matrices the cycle reads in float
# (see builtin_float_matrices in amgcl-wrapper.hpp): measured with outer iteration counts identical to
# the double hierarchy's on a captured damage-onset sequence, and faster.
#
# backendBlockSize defaults to the field's own nodal dimension (see defaultFieldTree below): 3 for
# a 3D vector field, 2 for a 2D one, 1 (scalar) otherwise. Setting it > 1 gives up the rotational
# rigid-body modes (see buildFieldHierarchy): a block hierarchy takes the translations as its
# near-null-space at most, and AMGCL's own identity blocks are constants per component in the equilibrated
# variables, i.e. D-weighted translations rather than the translations D^-1 t of the scaled block (nullspace.py).
# "blockTranslationNullspace": true, part of the 2D and 3D default trees, gives a block hierarchy those
# translations instead (amgcl-block-translations.hpp); false falls back to the identity blocks. Inside this
# block Gauss-Seidel preconditioner, 3x3 blocks stored in float with a smoothed-aggregation spectral-radius
# estimate and the translations are nevertheless the measured winner on the reference models (less index and
# value traffic per cycle than the scalar backend, fewer outer iterations than the identity blocks) -- the
# code default.
#
# All three are set per field in its tree, e.g.
# AMG(**{**defaultFieldTree(3), "backendBlockSize": 1, "backendPrecision": "double"}) to put a
# 3D field on the scalar backend, in double (see above), which takes the full rigid-body modes, rotations
# included (blockTranslationNullspace has nothing to add there, see takesBlockTranslations).
# power_iters 300: AMGCL's default (0) bounds the spectral radius by Gershgorin's theorem, pessimistically; a
# power iteration is tighter but starts from a vector seeded per OpenMP thread, so it depends on the thread count.
# The former 50 iterations left it badly under-converged on hard operators at some thread counts (needing far
# more outer iterations than an otherwise identical solve), 300 do not (see the Chebyshev section of the theory page).
#
# The coarsening/relaxation recipe every vector field shares, whatever its dimension -- the measured
# winner on the reference models (see the theory page's "why the defaults are what they are"):
# estimate_spectral_radius/power_iters 20 sharpens smoothed aggregation's own prolongation-smoother
# weight; degree 16 with npre 0, npost 1 (one high-degree post-smoothing sweep, no pre-smoothing) beats
# the former symmetric V(1,1) cycle at degree 5 for the same number of preconditioner applications.
_VECTOR_PRECOND_BASE = {
    "coarsening": {
        "type": "smoothed_aggregation",
        "aggr": {"eps_strong": 0.01},
        "estimate_spectral_radius": True,
        "power_iters": 20,
    },
    "relax": {"type": "chebyshev", "degree": 16, "power_iters": 300, "lower": 0.01, "higher": 1.1},
    "npre": 0,
    "npost": 1,
}
_DEFAULT_SCALAR_PRECOND = {
    "backendPrecision": "double",
    "backendBlockSize": 1,
    "coarsening": {"type": "smoothed_aggregation"},
    "relax": {"type": "chebyshev"},
}


def defaultFieldTree(dimension: int) -> dict:
    """The default AMGCL preconditioner tree for one field, chosen from its nodal dimension alone --
    the per-field hierarchy build has nothing else to go on before the first solve.

    A 3D vector field (displacement in 3D, dimension 3) gets the measured winner: its hierarchy is
    stored in node-wise 3x3 blocks, in float -- set up in double, only the level matrices the cycle
    reads are rounded (see the ``backendPrecision`` comment above), which was measured faster per
    outer iteration (less index and value traffic) at identical iteration counts. A 2D vector field
    (dimension 2) gets 2x2 blocks in double instead: this AMGCL wrapper has no float 2x2 backend to
    fall back to. A vector field of any other dimension, which was never part of the measurement, gets
    the same smoother on the plain scalar backend -- the only backend that takes the full rigid-body
    near null-space (see :func:`buildFieldHierarchy`), so it keeps the rigid-body path. A
    scalar field (dimension 1, e.g. ``nonlocal damage``) gets the scalar tree.

    Both block trees build smoothed aggregation from the translations ``D^-1 t`` of the field's equilibrated
    block (``blockTranslationNullspace``) instead of AMGCL's identity blocks, which are ``D``-weighted
    translations: measured live on the reference models, faster on all but one and level within noise on
    that one, with the same physical results (see the theory page).
    """
    # deep copies: the trees are nested, and a caller may adapt the one it gets
    if dimension == 3:
        return copy.deepcopy(
            {
                **_VECTOR_PRECOND_BASE,
                "backendPrecision": "float",
                "backendBlockSize": 3,
                "blockTranslationNullspace": True,
            }
        )
    if dimension == 2:
        return copy.deepcopy(
            {
                **_VECTOR_PRECOND_BASE,
                "backendPrecision": "double",
                "backendBlockSize": 2,
                "blockTranslationNullspace": True,
            }
        )
    if dimension > 1:
        return copy.deepcopy({**_VECTOR_PRECOND_BASE, "backendPrecision": "double", "backendBlockSize": 1})
    return copy.deepcopy(_DEFAULT_SCALAR_PRECOND)


#: The keys of a field tree that are not AMGCL parameters but select how the field's hierarchy is built: the backend (see the comment above the default trees) and the translations of a block backend
#: (``blockTranslationNullspace``).
_FIELD_TREE_KEYS = ("backendPrecision", "backendBlockSize", "blockTranslationNullspace")


def amgclTreeOf(fieldTree: dict) -> dict:
    """A field's tree without the keys that select the backend (:data:`_FIELD_TREE_KEYS`): the AMGCL tree, forwarded
    verbatim (to AMGCL's hierarchy, or to the rigid-body coarse level, which reads it as AMGCL would)."""
    return {key: value for key, value in fieldTree.items() if key not in _FIELD_TREE_KEYS}


def takesBlockTranslations(fieldTree: dict) -> bool:
    """Whether a field's tree builds its block hierarchy from the translations ``D^-1 t`` of the scaled block
    (``blockTranslationNullspace``) -- the one hierarchy where the key changes the coarse space. The scalar backend
    (``backendBlockSize`` 1) takes the rigid-body modes, the translations included, whatever the key says. So a
    default tree stays valid, and means what it says, with the scalar backend set on top."""
    return fieldTree.get("blockTranslationNullspace", False) is True and fieldTree.get("backendBlockSize", 1) > 1


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
    """The matrix a one-field preconditioner (:class:`AMG`, :class:`~edelweissfe.linsolve.composition.twolevel.TwoLevel`)
    is built on: :math:`A` itself, or with ``dropTolerance > 0`` its filtered copy (:func:`droppedSmallEntries`),
    with the stored entries before and after recorded in the component's ``statistics``."""
    matrix = A.matrix()
    if dropTolerance <= 0.0:
        return matrix
    with performancetiming.timeit("diagonal blocks"):
        filtered = droppedSmallEntries(matrix, dropTolerance, lumpDroppedEntries)
    statistics.sizes.update({"nnz": matrix.nnz, "nnz kept": filtered.nnz})
    return filtered


def fieldTreeRejection(tree, dimension: "int | None" = None) -> "str | None":
    """Why a field tree (an AMGCL tree plus ``backendPrecision``, ``backendBlockSize``, ``blockTranslationNullspace``)
    cannot build a hierarchy, or ``None``: what the build would otherwise fail on later, or (for a key AMGCL does not
    know) only warn about on stderr. ``dimension``: the field's nodal dimension, ``None`` if not known yet. Shared by
    :class:`AMG` and :class:`~edelweissfe.linsolve.composition.twolevel.TwoLevel`."""
    if not isinstance(tree, dict):
        return "must be an AMGCL preconditioner tree (a JSON object)"
    precision = tree.get("backendPrecision", "double")
    blockSize = tree.get("backendBlockSize", 1)
    if precision not in ("double", "float"):
        return "backendPrecision must be 'double' or 'float'"
    if blockSize not in (1, 2, 3) or (blockSize == 2 and precision == "float"):
        return "backendBlockSize must be 1, 2 or 3 (2 only in double)"
    if "hierarchyOnUnscaledBlock" in tree:
        if tree["hierarchyOnUnscaledBlock"] is False:
            return "hierarchyOnUnscaledBlock was removed; false was its default and did nothing, so drop the key"
        return (
            "hierarchyOnUnscaledBlock was removed. For a vector field on a block backend, its coarse space is "
            'blockTranslationNullspace: set "blockTranslationNullspace": true in this tree (a tree replaces the '
            'default tree, it does not inherit that key), and to keep its Jacobi-scaled smoother put "scale": true '
            'inside the existing relax dict, next to "type": "chebyshev". A scalar field has no replacement: its '
            "hierarchy is built on the scaled block"
        )
    if not isinstance(tree.get("blockTranslationNullspace", False), bool):
        return "blockTranslationNullspace must be true or false"
    # AMGCL's defaults: class amg, coarsening.type smoothed_aggregation
    coarsening = tree.get("coarsening", {})
    coarseningType = coarsening.get("type", "smoothed_aggregation") if isinstance(coarsening, dict) else None
    if takesBlockTranslations(tree) and (tree.get("class", "amg") != "amg" or coarseningType != "smoothed_aggregation"):
        return (
            "blockTranslationNullspace needs smoothed aggregation (coarsening.type); it is true in the default 2D/3D "
            "trees, set it to false for another coarsening"
        )
    if blockSize > 1 and dimension is not None and dimension != blockSize:
        return "backendBlockSize {:} does not match the field's dimension {:}".format(blockSize, dimension)
    return None


def checkedFieldTree(owner: str, tree: "dict | None", field) -> dict:
    """The tree a one-field component builds on: ``tree``, or the default tree of the field's dimension; raises,
    naming ``owner`` and the field, if it cannot build a hierarchy (:func:`fieldTreeRejection`)."""
    if tree is None:
        return defaultFieldTree(field.dimension)
    reason = fieldTreeRejection(tree, field.dimension)
    if reason is not None:
        raise ValueError("{:} on the field '{:}': its tree: {:}".format(owner, field.name, reason))
    return tree


def buildFieldHierarchy(block, fieldTree: dict, scaling, coords, diagonalBlock):
    """One AMGCL hierarchy for a field's scaled diagonal block :math:`\\hat A_{ii}` (``diagonalBlock``, a callable
    that returns it, so that it is cut out only now), with the field's near null-space.

    A vector field on the scalar backend takes its rigid-body modes (the translations alone without
    ``coords``), on a block backend the translations (``blockTranslationNullspace``, else AMGCL's constant blocks);
    a scalar field takes AMGCL's default constant.

    Parameters
    ----------
    block
        The field's :class:`~edelweissfe.linsolve.base.FieldBlock`.
    fieldTree
        The field's tree: AMGCL parameters plus ``backendPrecision``, ``backendBlockSize``,
        ``blockTranslationNullspace``.
    scaling
        The field's scaling :math:`D_i`.
    coords
        The field's node coordinates, or ``None``.
    diagonalBlock
        ``() -> scipy CSR``, the scaled diagonal block.

    Returns
    -------
    PyAMGCLSolver
        The hierarchy, applied by its ``applyPreconditioner``.
    """
    from edelweissfe.linsolve.amgcl.amgcl import PyAMGCLSolver

    backendBlockSize = fieldTree.get("backendBlockSize", 1)
    # The hierarchy is only ever applied as a preconditioner: "preonly" keeps AMGCL from allocating the work vectors
    # of an iterative solver that would never run.
    hierarchy = PyAMGCLSolver(
        {
            "precond": amgclTreeOf(fieldTree),
            "solver": {"type": "preonly"},
            "backendPrecision": fieldTree.get("backendPrecision", "double"),
            "backendBlockSize": backendBlockSize,
        }
    )
    if block.dimension > 1 and (backendBlockSize == 1 or takesBlockTranslations(fieldTree)):
        with performancetiming.timeit("nullspace construction"):
            if coords is not None:
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
    tree
        The AMGCL parameter tree, plus ``backendPrecision`` (``"double"``/``"float"``), ``backendBlockSize``
        (1, 2, 3) and ``blockTranslationNullspace``. ``None``: the default tree of the field's dimension
        (:func:`defaultFieldTree`).
    rigidBodyNullspace
        Whether a vector field on the scalar backend takes its rotations too (needs node coordinates).
    dropTolerance, lumpDroppedEntries
        Build the hierarchy on the filtered matrix, see :func:`droppedSmallEntries`; ``0`` (the default) builds it
        on the matrix itself.
    """

    def __init__(
        self,
        tree: dict = None,
        rigidBodyNullspace: bool = True,
        dropTolerance: float = 0.0,
        lumpDroppedEntries: bool = False,
    ):
        if not dropTolerance >= 0.0:
            raise ValueError("AMG: dropTolerance must not be negative, got {!r}".format(dropTolerance))
        self.tree = tree
        self.rigidBodyNullspace = rigidBodyNullspace
        self.dropTolerance = dropTolerance
        self.lumpDroppedEntries = lumpDroppedEntries
        self.hierarchy = None
        self._field = None

    def setUp(self, A, context: SetUpContext) -> None:
        field = context.onlyField("AMG")
        self._field = field
        tree = checkedFieldTree("AMG", self.tree, field)
        onScalarBackend = tree.get("backendBlockSize", 1) == 1
        coords = (
            context.coordinates(field.name)
            if field.dimension > 1 and onScalarBackend and self.rigidBodyNullspace
            else None
        )
        self.hierarchy = buildFieldHierarchy(
            field,
            tree,
            context.scalingOf(slice(None)),
            coords,
            lambda: preconditionerMatrix(A, self.dropTolerance, self.lumpDroppedEntries, self.statistics),
        )
        self.statistics.sizes["rows"] = A.shape[0]

    def update(self, A, change) -> None:
        if change.requiresRebuild or self.hierarchy is None:
            self.setUp(A, change.context)

    def __call__(self, r):
        return self.hierarchy.applyPreconditioner(r)

    def describe(self) -> Description:
        tree = self.tree if self.tree is not None else {}
        parameters = {"tree": "default" if self.tree is None else "given"}
        for key in ("backendBlockSize", "backendPrecision"):
            if key in tree:
                parameters[key] = tree[key]
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
