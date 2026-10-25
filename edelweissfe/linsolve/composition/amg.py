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
    """

    def __init__(self, tree: dict = None, rigidBodyNullspace: bool = True):
        self.tree = tree
        self.rigidBodyNullspace = rigidBodyNullspace
        self.hierarchy = None
        self._field = None

    def setUp(self, A, context: SetUpContext) -> None:
        (field,) = context.fields
        self._field = field
        tree = self.tree if self.tree is not None else defaultFieldTree(field.dimension)
        onScalarBackend = tree.get("backendBlockSize", 1) == 1
        coords = (
            context.coordinates(field.name)
            if field.dimension > 1 and onScalarBackend and self.rigidBodyNullspace
            else None
        )
        self.hierarchy = buildFieldHierarchy(field, tree, context.scalingOf(slice(None)), coords, lambda: A.A)
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
        name = self._field.name if self._field is not None else "A"
        return Description(
            name="AMG",
            formula="AMG(" + name + ")⁻¹",
            latex="\\mathrm{AMG}(A_{\\text{" + name + "}})^{-1}",
            parameters=parameters,
        )
