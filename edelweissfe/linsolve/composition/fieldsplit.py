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
"""A field split: :class:`FieldSplit`, one preconditioner per field, combined by a block Gauss--Seidel sweep.

Split the matrix by fields, :math:`A = L + B + U` with the block diagonal :math:`B = \\operatorname{blockdiag}(A_{ii})`
and the strictly lower and upper block parts :math:`L` and :math:`U`. One forward block Gauss--Seidel sweep is

.. math::

    M^{-1} = (\\tilde B + L)^{-1}, \\qquad \\tilde B = \\operatorname{blockdiag}(M_i),

with :math:`M_i^{-1}` the preconditioner of field :math:`i` (e.g. one AMG cycle on :math:`A_{ii}`). Field by
field, in residual-correction form:

.. math::

    z_i \\leftarrow z_i + M_i^{-1} \\Big( r_i - \\sum_{j} A_{ij} z_j \\Big)

where on a first visit :math:`z_i = 0` and only the fields already visited contribute to the sum. A coupled
system of fields that are each AMG-friendly -- elasticity for a displacement field, a Helmholtz-like operator for a
nonlocal damage field -- is preconditioned well this way, although one AMG hierarchy for the monolithic system is
not.

The sweep reads the couplings :math:`A_{ij}` only as products :math:`A_{ij} z_j`, from the field blocks formed in
one threaded pass (:meth:`~edelweissfe.linsolve.composition.systemmatrix.SystemMatrix.fieldSplit`).
"""

from collections.abc import Callable

import numpy as np

from edelweissfe.linsolve.composition.base import (
    Description,
    Preconditioner,
    SetUpContext,
    Step,
)
from edelweissfe.linsolve.composition.systemmatrix import FieldSplitBlocks, SystemMatrix


def blockGaussSeidel(
    blocks: FieldSplitBlocks, fieldPreconditioners: list, r: np.ndarray, sweeps: int = 1, symmetric: bool = False
) -> np.ndarray:
    """Block Gauss--Seidel sweeps over the fields: :math:`z = M^{-1} r`, see the module documentation.

    Parameters
    ----------
    blocks
        The field blocks :math:`A_{ij}` (their products with vectors).
    fieldPreconditioners
        :math:`M_i^{-1}` per field, as callables ``r_i -> z_i``.
    r
        The vector to precondition.
    sweeps
        Forward sweeps per application; a revisited field improves its :math:`z_i`.
    symmetric
        If True, each forward sweep is followed by a backward one, starting one field before the last (the last
        field was just updated with everything else current).

    Returns
    -------
    np.ndarray
        :math:`z`.
    """
    fields = blocks.slices
    nFields = len(fields)
    z = [np.zeros(field.stop - field.start) for field in fields]
    visited = [False] * nFields

    def sweep(order):
        for i in order:
            r_i = r[fields[i]].copy()
            for j in range(nFields):
                if j != i and visited[j]:
                    r_i -= blocks.couplingTimes(i, j, z[j])
            if visited[i]:
                r_i -= blocks.diagonalBlockTimes(i, z[i])
                z[i] = z[i] + fieldPreconditioners[i](r_i)
            else:
                z[i] = fieldPreconditioners[i](r_i)
                visited[i] = True

    for _ in range(sweeps):
        sweep(range(nFields))
        if symmetric:
            sweep(range(nFields - 2, -1, -1))
    return np.concatenate(z)


class FieldBlockMatrix(SystemMatrix):
    """The diagonal block :math:`A_{ii}` of a field split as a
    :class:`~edelweissfe.linsolve.composition.systemmatrix.SystemMatrix`, cut out only when it is first used: a
    child that reuses its preconditioner never needs it."""

    def __init__(self, blocks: FieldSplitBlocks, i: int):
        self._blocks = blocks
        self._i = i
        self._block = None
        self.scaling = None
        self._unscaled = None
        self._krylovOperator = None

    @property
    def A(self):
        if self._block is None:
            self._block = self._blocks.diagonalBlock(self._i)
        return self._block


class FieldSplit(Preconditioner):
    """One preconditioner per field, combined by block Gauss--Seidel sweeps; see the module documentation.

    Parameters
    ----------
    fields
        The preconditioner of each field, by field name, e.g. ``{"displacement": AMG(), "nonlocal damage": AMG()}``.
    otherFields
        Creates the preconditioner of a field not named in ``fields``: ``(fieldBlock, context) -> Preconditioner``.
        Called once per field name, at the first set-up. Without it, every field must be named.
    sweeps
        Forward sweeps per application.
    symmetric
        Whether each forward sweep is followed by a backward one.
    """

    def __init__(self, fields: dict = None, otherFields: Callable = None, sweeps: int = 1, symmetric: bool = False):
        self.fieldPreconditioners = dict(fields or {})
        self.otherFields = otherFields
        self.sweeps = sweeps
        self.symmetric = symmetric
        self.blocks = None
        self._order = []

    def _preconditionerOf(self, field, context: SetUpContext) -> Preconditioner:
        if field.name not in self.fieldPreconditioners:
            if self.otherFields is None:
                raise ValueError("FieldSplit: no preconditioner for the field '{:}'".format(field.name))
            self.fieldPreconditioners[field.name] = self.otherFields(field, context)
        return self.fieldPreconditioners[field.name]

    def setUp(self, A, context: SetUpContext) -> None:
        self.blocks = A.fieldSplit([(field.start, field.stop) for field in context.fields])
        self._order = [self._preconditionerOf(field, context) for field in context.fields]
        for i, field in enumerate(context.fields):
            self._order[i].setUp(FieldBlockMatrix(self.blocks, i), context.restrictedToField(field))

    def update(self, A, change) -> None:
        if change.requiresRebuild or self.blocks is None:
            self.setUp(A, change.context)
            return
        # new values, the same fields: the couplings of the sweep always come from the current matrix
        self.blocks = A.fieldSplit(self.blocks.bounds)
        for i, field in enumerate(change.context.fields):
            fieldChange = change.withContext(change.context.restrictedToField(field))
            self._order[i].update(FieldBlockMatrix(self.blocks, i), fieldChange)

    def __call__(self, r: np.ndarray) -> np.ndarray:
        return blockGaussSeidel(self.blocks, self._order, r, self.sweeps, self.symmetric)

    def describe(self) -> Description:
        children = tuple(self.fieldPreconditioners.items())
        roles = ", ".join("{" + name + "}" for name, _ in children) or "Mᵢ⁻¹ per field"
        sweep = "symmetric Gauss–Seidel" if self.symmetric else "Gauss–Seidel"
        return Description(
            name="FieldSplit",
            formula="GS[" + roles + "]",
            latex="\\mathrm{GS}\\left[" + roles + "\\right]",
            parameters={"sweep": sweep, "sweeps": self.sweeps},
            children=children,
            steps=tuple(Step("zᵢ += Mᵢ⁻¹ (rᵢ − Σⱼ Aᵢⱼ zⱼ),  i = " + name, name) for name, _ in children),
        )
