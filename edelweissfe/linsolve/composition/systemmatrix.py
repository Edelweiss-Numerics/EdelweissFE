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
"""The matrix handed down a composed solver: :class:`SystemMatrix`.

A component gets its matrix as a :class:`SystemMatrix`, not as a bare SciPy matrix, for one reason: speed. A
diagonal scaling :math:`\\hat A = D A D` followed by a split into field blocks :math:`\\hat A_{ij}` would cost a
chain of serial SciPy passes over :math:`A` -- a scaled copy, then a row and column slice per block. Instead,
:meth:`SystemMatrix.scaled` only records :math:`D`, and :meth:`SystemMatrix.fieldSplit` forms :math:`\\hat A` and
all its blocks in one threaded native pass
(:class:`~edelweissfe.linsolve.amgcl.amgcl.PyAMGCLEquilibratedSystem`). The same pass also keeps :math:`A` itself
as the operator of an outer Krylov solve (:meth:`SystemMatrix.krylovOperator`), so a field split under a diagonal
scaling under a Krylov solver reads :math:`A` exactly once per solve.

A leaf that needs the matrix itself (a direct solver, the Schur complement) asks for :meth:`SystemMatrix.matrix`.
"""

import numpy as np
import scipy.sparse as sp

import edelweissfe.utils.performancetiming as performancetiming


def canonicalMatrix(A) -> sp.csr_matrix:
    """``A`` as CSR with sorted columns and no duplicate entries: AMGCL's block backends read the columns as sorted,
    and no backend sums duplicate entries. A canonical copy if needed, leaving the caller's matrix as it is."""
    A = A.tocsr()
    if not A.has_canonical_format:
        A = A.copy()
        A.sum_duplicates()
    return A


class FieldSplitBlocks:
    """The field blocks :math:`\\hat A_{ij}` of a (scaled) matrix, formed in one threaded pass: the diagonal blocks,
    and products with the coupling blocks, as a block Gauss--Seidel sweep needs them.

    Parameters
    ----------
    native
        The :class:`~edelweissfe.linsolve.amgcl.amgcl.PyAMGCLEquilibratedSystem` that holds them.
    bounds
        The ``(start, stop)`` dof range of each field.
    """

    def __init__(self, native, bounds: list):
        self._native = native
        self.bounds = bounds
        self.slices = [slice(start, stop) for start, stop in bounds]

    def diagonalBlock(self, i: int) -> sp.csr_matrix:
        """:math:`\\hat A_{ii}` as a new SciPy CSR matrix."""
        with performancetiming.timeit("diagonal blocks"):
            return self._native.diagonalBlock(i)

    def diagonalBlockTimes(self, i: int, x: np.ndarray) -> np.ndarray:
        """:math:`\\hat A_{ii} x`."""
        return self._native.diagonalBlockMatvec(i, x)

    def couplingTimes(self, i: int, j: int, x: np.ndarray) -> np.ndarray:
        """:math:`\\hat A_{ij} x`, for :math:`i \\neq j`."""
        return self._native.couplingMatvec(i, j, x)


class SystemMatrix:
    """A matrix :math:`A`, as handed to a component: optionally with a diagonal scaling, :math:`\\hat A = D A D`.

    See the module documentation: :math:`\\hat A` is formed on demand, together with its field blocks, in one
    threaded pass.

    Parameters
    ----------
    A
        The matrix (SciPy sparse, square). Made canonical (CSR, sorted columns, no duplicates) if needed.
    scaling
        :math:`D` as a vector, or ``None`` for :math:`D = I`.
    """

    def __init__(self, A, scaling: "np.ndarray | None" = None, _unscaled: "SystemMatrix | None" = None):
        self.A = canonicalMatrix(A)
        self.scaling = scaling
        # the unscaled matrix this one is a scaled view of: a field split of the view keeps A for its Krylov operator
        self._unscaled = _unscaled
        self._krylovOperator = None

    @property
    def shape(self) -> tuple:
        return self.A.shape

    @property
    def nnz(self) -> int:
        """The stored entries of :math:`A` (a proxy for its sparsity pattern)."""
        return self.A.nnz

    def scaled(self, D: np.ndarray) -> "SystemMatrix":
        """:math:`D \\hat A D`, without forming it (see :meth:`fieldSplit`)."""
        scaling = D if self.scaling is None else self.scaling * D
        return SystemMatrix(self.A, scaling, _unscaled=self if self.scaling is None else self._unscaled)

    def absoluteDiagonal(self) -> np.ndarray:
        """:math:`\\lvert \\hat a_{ii} \\rvert`, threaded."""
        from edelweissfe.linsolve.amgcl.amgcl import PyAMGCLEquilibratedSystem

        diagonal = PyAMGCLEquilibratedSystem.absoluteDiagonal(self.A)
        return diagonal if self.scaling is None else diagonal * self.scaling**2

    def matrix(self) -> sp.csr_matrix:
        """:math:`\\hat A` itself, formed with SciPy (for a leaf that needs the matrix, e.g. a direct solver)."""
        if self.scaling is None:
            return self.A
        D = sp.diags(self.scaling)
        return (D @ self.A @ D).tocsr()

    def __matmul__(self, x: np.ndarray) -> np.ndarray:
        """:math:`\\hat A x`."""
        if self.scaling is None:
            return self.A @ x
        return self.scaling * (self.A @ (self.scaling * x))

    def fieldSplit(self, bounds: list) -> FieldSplitBlocks:
        """The field blocks :math:`\\hat A_{ij}` for the field ranges ``bounds``, formed with :math:`\\hat A` in one
        threaded pass. That pass also keeps :math:`A` as the operator of a Krylov solve on the unscaled matrix
        this one is a view of (:meth:`krylovOperator`)."""
        from edelweissfe.linsolve.amgcl.amgcl import PyAMGCLEquilibratedSystem

        with performancetiming.timeit("equilibrated system"):
            native = PyAMGCLEquilibratedSystem()
            scaling = np.ones(self.A.shape[0]) if self.scaling is None else self.scaling
            native.build(self.A, scaling, bounds, unscaled=True)
        unscaled = self if self.scaling is None else self._unscaled
        if unscaled is not None:
            unscaled._krylovOperator = native
        return FieldSplitBlocks(native, bounds)

    def krylovOperator(self):
        """:math:`A` as the operator of AMGCL's LGMRES (unscaled: a Krylov solve on :math:`D A D` is
        ``DiagonalScaling(LGMRES(...))``, not offered by this prototype). Kept from a field split of this matrix or
        of a scaled view of it if there was one, else built (threaded)."""
        if self.scaling is not None:
            raise NotImplementedError("a Krylov solve on a diagonally scaled matrix is not offered yet")
        if self._krylovOperator is None:
            from edelweissfe.linsolve.amgcl.amgcl import PyAMGCLEquilibratedSystem

            with performancetiming.timeit("equilibrated system"):
                self._krylovOperator = PyAMGCLEquilibratedSystem()
                self._krylovOperator.buildOperator(self.A)
        return self._krylovOperator
