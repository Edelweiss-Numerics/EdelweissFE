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
#  This library is free software; you can redistribute it and/or
#  modify it under the terms of the GNU Lesser General Public
#  License as published by the Free Software Foundation; either
#  version 2.1 of the License, or (at your option) any later version.
#
#  The full text of the license can be found in the file LICENSE.md at
#  the top level directory of EdelweissFE.
#  ---------------------------------------------------------------------
"""The sparse-matrix tools that :class:`~edelweissfe.linsolve.blockamg.blockamg.BlockAMGSolver` and its Schur complement
condensation (:mod:`~edelweissfe.linsolve.blockamg.schurcomplement`) share: the relative drop test of AMG's strength of
connection, and the sparsity pattern of a matrix, which tells whether what was found for one matrix holds for the next.
"""

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp


def isSignificant(values, rows, cols, sqrtAbsDiagonal: np.ndarray, tau: float) -> np.ndarray:
    """Whether the entries ``values`` at the positions ``(rows, cols)`` pass the relative drop test of AMG's strength
    of connection,

    .. math::

        |a_{ij}| \\geq \\tau \\sqrt{|a_{ii} a_{jj}|},

    with ``sqrtAbsDiagonal`` holding :math:`\\sqrt{|a_{ii}|}`. ``rows`` and ``cols`` broadcast against ``values``: the
    row and the column of every stored entry of a sparse matrix, or ``rows[:, None]`` and ``cols[None, :]`` for a dense
    block.

    Returns
    -------
    np.ndarray
        The keep mask, of the shape of ``values``.
    """
    return np.abs(values) >= tau * sqrtAbsDiagonal[rows] * sqrtAbsDiagonal[cols]


@dataclass(frozen=True, eq=False)
class SparsityPattern:
    """The sparsity pattern of a CSR matrix in canonical format. What was derived from the pattern of one matrix -- the
    positions of the entries of its blocks, an operator built on it -- holds for every matrix that :meth:`matches` the
    pattern: only the values need refreshing. Compared exactly, never by a proxy such as the number of entries."""

    #: The row pointers of the matrix.
    indptr: np.ndarray
    #: The column indices of the matrix.
    indices: np.ndarray

    @classmethod
    def of(cls, A: sp.csr_matrix) -> "SparsityPattern":
        """The pattern of ``A``, copied: the caller may change ``A``'s arrays in place later."""
        return cls(A.indptr.copy(), A.indices.copy())

    def matches(self, A: sp.csr_matrix) -> bool:
        """Whether ``A`` has this pattern."""
        return np.array_equal(self.indptr, A.indptr) and np.array_equal(self.indices, A.indices)
