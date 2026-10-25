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
of their own, see :class:`~edelweissfe.linsolve.composition.twogrid.TwoGrid`.
"""

import numpy as np
import scipy.sparse as sp

import edelweissfe.utils.performancetiming as performancetiming
from edelweissfe.linsolve.composition.base import (
    Description,
    Preconditioner,
    SetUpContext,
)
from edelweissfe.linsolve.nullspace import rigidBodyNullspace, translationNullspace


def defaultFieldTree(dimension: int) -> dict:
    """The default AMGCL tree of a field of this nodal dimension: blockamg's measured defaults (see
    :func:`edelweissfe.linsolve.blockamg.blockamg._defaultFieldPrecond`)."""
    from edelweissfe.linsolve.blockamg.blockamg import _defaultFieldPrecond

    return _defaultFieldPrecond(dimension)


def amgclTreeOf(fieldTree: dict) -> dict:
    """The AMGCL parameters of a field tree, without the keys that select the backend."""
    from edelweissfe.linsolve.blockamg.blockamg import _amgclTree

    return _amgclTree(fieldTree)


def droppedSmallEntries(A: sp.csr_matrix, dropTolerance: float, lumpDroppedEntries: bool = False) -> sp.csr_matrix:
    """The *filtered* matrix :math:`\\tilde A` of ``A`` that a preconditioner is built on: the off-diagonal entries
    with :math:`\\lvert a_{ij} \\rvert < \\tau \\sqrt{\\lvert a_{ii} a_{jj} \\rvert}` dropped,
    :math:`\\tau` = ``dropTolerance``. The diagonal is always kept, so no row is emptied. Shared by :class:`AMG`,
    :class:`~edelweissfe.linsolve.composition.twogrid.TwoGrid` and
    :class:`~edelweissfe.linsolve.blockamg.blockamg.BlockAMGSolver` (``hierarchyDropTol``).

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
    diagonal = np.abs(A.diagonal())
    coo = A.tocoo()
    scale = np.sqrt(diagonal[coo.row] * diagonal[coo.col])
    keep = (coo.row == coo.col) | (np.abs(coo.data) >= dropTolerance * scale)
    filtered = sp.coo_matrix((coo.data[keep], (coo.row[keep], coo.col[keep])), shape=A.shape).tocsr()
    if lumpDroppedEntries:
        droppedRowSums = np.bincount(coo.row[~keep], weights=coo.data[~keep], minlength=A.shape[0])
        filtered = (filtered + sp.diags(droppedRowSums, format="csr", dtype=float)).tocsr()
    return filtered


def preconditionerMatrix(A, dropTolerance: float, lumpDroppedEntries: bool, statistics) -> sp.csr_matrix:
    """The matrix a one-field preconditioner (:class:`AMG`, :class:`~edelweissfe.linsolve.composition.twogrid.TwoGrid`)
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
    :class:`AMG`, :class:`~edelweissfe.linsolve.composition.twogrid.TwoGrid` and ``fieldPreconds`` of
    :class:`~edelweissfe.linsolve.blockamg.blockamg.BlockAMGSolver`."""
    from edelweissfe.linsolve.blockamg.blockamg import _takesBlockTranslations

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
    if _takesBlockTranslations(tree) and (
        tree.get("class", "amg") != "amg" or coarseningType != "smoothed_aggregation"
    ):
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
    a scalar field takes AMGCL's default constant. Shared by
    :class:`~edelweissfe.linsolve.blockamg.blockamg.BlockAMGSolver` and :class:`AMG`.

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
    from edelweissfe.linsolve.blockamg.blockamg import _takesBlockTranslations

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
    if block.dimension > 1 and (backendBlockSize == 1 or _takesBlockTranslations(fieldTree)):
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
