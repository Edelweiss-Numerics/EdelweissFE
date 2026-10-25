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
        # whether a Krylov solve runs on this matrix (see useAsKrylovOperator)
        self._isKrylovOperator = False
        # the field split of this matrix, (bounds, blocks), formed at the first request (see fieldSplit)
        self._fieldSplit = None

    @property
    def shape(self) -> tuple:
        return self.A.shape

    def useAsKrylovOperator(self) -> None:
        """Say that a Krylov solve runs on this matrix: a field split formed afterwards keeps :math:`\\hat A` itself as
        the operator of that solve, in its one threaded pass (see :meth:`krylovOperator`)."""
        self._isKrylovOperator = True

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
        """:math:`\\hat A` itself, for a component built on the whole matrix (AMG, a direct solver): :math:`A`, or
        :math:`D A D` formed in the threaded pass of a split into one block (:meth:`fieldSplit`), which also keeps
        :math:`A` as the Krylov operator."""
        if self.scaling is None:
            return self.A
        return self.fieldSplit([(0, self.A.shape[0])]).diagonalBlock(0)

    def __matmul__(self, x: np.ndarray) -> np.ndarray:
        """:math:`\\hat A x`."""
        if self.scaling is None:
            return self.A @ x
        return self.scaling * (self.A @ (self.scaling * x))

    def fieldSplit(self, bounds: list) -> FieldSplitBlocks:
        """The field blocks :math:`\\hat A_{ij}` for the field ranges ``bounds``, formed with :math:`\\hat A` in one
        threaded pass at the first request and kept with this matrix: every component handed this matrix (the
        corrections of a combination, each on its field) gets the same blocks. That pass also keeps the operator of the
        Krylov solve (:meth:`krylovOperator`): :math:`D A D` if the solve runs on this scaled matrix
        (:meth:`useAsKrylovOperator`), else :math:`A`, for a solve on the unscaled matrix this one is a view of."""
        bounds = [tuple(bound) for bound in bounds]
        if self._fieldSplit is not None and self._fieldSplit[0] == bounds:
            return self._fieldSplit[1]
        from edelweissfe.linsolve.amgcl.amgcl import PyAMGCLEquilibratedSystem

        # a split for other bounds goes before the new one is formed: never two copies at once
        self._fieldSplit = None
        # The pass keeps one operator besides the blocks: D A D for a Krylov solve on this (scaled) matrix, else A, for
        # a Krylov solve on the unscaled matrix this one is a view of.
        keepsScaled = self.scaling is not None and self._isKrylovOperator
        with performancetiming.timeit("equilibrated system"):
            native = PyAMGCLEquilibratedSystem()
            scaling = np.ones(self.A.shape[0]) if self.scaling is None else self.scaling
            native.build(self.A, scaling, bounds, unscaled=not keepsScaled)
        if keepsScaled:
            self._krylovOperator = native
        else:
            unscaled = self if self.scaling is None else self._unscaled
            if unscaled is not None:
                unscaled._krylovOperator = native
        self._fieldSplit = (bounds, FieldSplitBlocks(native, bounds))
        return self._fieldSplit[1]

    def releaseFormedMatrices(self) -> None:
        """Drop what was formed from this matrix in a solve, the native field split and the operator of a Krylov solve
        (formed again when next asked for), and keep only :math:`A`: for a matrix a component keeps between solves
        (the :math:`\\tilde S` of a Schur complement)."""
        self._fieldSplit = None
        self._krylovOperator = None

    def krylovOperator(self):
        """:math:`\\hat A` as the operator of AMGCL's Krylov solvers: :math:`A` itself (applied with
        ``onUnscaledOperator``), or :math:`D A D` for a Krylov solve on a scaled matrix,
        ``DiagonalScaling(LGMRES(...))``. Kept from the field split of the matrix if there was one (see
        :meth:`useAsKrylovOperator`), else built in a threaded pass."""
        if self._krylovOperator is None:
            if self.scaling is not None:
                # D A D comes only from a split pass; one of a single block holds nothing else
                self.useAsKrylovOperator()
                self.fieldSplit([(0, self.A.shape[0])])
            else:
                from edelweissfe.linsolve.amgcl.amgcl import PyAMGCLEquilibratedSystem

                with performancetiming.timeit("equilibrated system"):
                    self._krylovOperator = PyAMGCLEquilibratedSystem()
                    self._krylovOperator.buildOperator(self.A)
        return self._krylovOperator


class FieldBlockMatrix(SystemMatrix):
    """The diagonal block :math:`\\hat A_{ii}` of a field split as a :class:`SystemMatrix`, cut out only when it is
    first used: a child that reuses its preconditioner never needs it, and the split itself is formed only when the
    block or a product with it is first asked for.

    Parameters
    ----------
    blocksOf
        ``() -> FieldSplitBlocks``, the field split of the whole matrix.
    i
        The index of the field.
    """

    def __init__(self, blocksOf, i: int):
        self._blocksOf = blocksOf
        self._i = i
        self._block = None
        self.scaling = None
        self._unscaled = None
        self._krylovOperator = None
        self._isKrylovOperator = False
        self._fieldSplit = None

    @property
    def A(self):
        if self._block is None:
            self._block = self._blocksOf().diagonalBlock(self._i)
        return self._block

    @property
    def shape(self) -> tuple:
        start, stop = self._blocksOf().bounds[self._i] if self._block is None else (0, self._block.shape[0])
        return (stop - start, stop - start)

    def fieldSplit(self, bounds: list) -> FieldSplitBlocks:
        """The block as a split of one field: its products come from the split of the whole matrix, not from a copy."""
        start, stop = self._blocksOf().bounds[self._i]
        if [tuple(bound) for bound in bounds] != [(0, stop - start)]:
            return super().fieldSplit(bounds)
        return _OneFieldOf(self._blocksOf(), self._i)


class _OneFieldOf(FieldSplitBlocks):
    """Field ``i`` of a field split, as a split of one field (see :meth:`FieldBlockMatrix.fieldSplit`)."""

    def __init__(self, blocks: FieldSplitBlocks, i: int):
        start, stop = blocks.bounds[i]
        super().__init__(blocks._native, [(0, stop - start)])
        self._i = i

    def diagonalBlock(self, i: int) -> sp.csr_matrix:
        return super().diagonalBlock(self._i)

    def diagonalBlockTimes(self, i: int, x: np.ndarray) -> np.ndarray:
        return super().diagonalBlockTimes(self._i, x)

    def couplingTimes(self, i: int, j: int, x: np.ndarray) -> np.ndarray:
        raise ValueError("a split of one field has no coupling blocks")
