#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#  ---------------------------------------------------------------------
#
#  _____    _      _              _         _____ _____
# | ____|__| | ___| |_      _____(_)___ ___|  ___| ____|
# |  _| / _` |/ _ \ \ \ /\ / / _ \ / __/ __| |_  |  _|
# | |__| (_| |  __/ |\ V  V /  __/ \__ \__ \  _| | |___
# |_____\__, _|\___|_| \_/\_/ \___|_|___/___/_|   |_____|
#
#
#  Unit of Strength of Materials and Structural Analysis
#  University of Innsbruck,
#  2017 - today
#
#  Matthias Neuner matthias.neuner@uibk.ac.at
#  ALexander Dummer alexander.dummer@uibk.ac.at
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

import numpy as np

cimport cython

from collections.abc import Iterable

from scipy.sparse import csr_matrix


@cython.boundscheck(False)
@cython.wraparound(False)
def applyDirichletToStiffness(K: csr_matrix, dirichlets: Iterable, rhs=None) -> csr_matrix:
    """Impose the Dirichlet BCs on the global stiffness matrix.

    Every constrained DOF (row) gets its whole row set to zero and 1.0 on the diagonal. Solving
    K ddU = R then returns ddU[row] = R[row], i.e. the value that
    :func:`applyDirichletToResidual` wrote into R for that DOF.

    If the right-hand side(s) ``rhs`` are given, the constrained DOFs are eliminated from the
    columns as well (symmetric elimination), which keeps a symmetric K symmetric: for every
    unconstrained row r and constrained column c, ``K[r, c] * rhs[c]`` -- the prescribed value
    times its coupling -- is moved to ``rhs[r]`` and ``K[r, c]`` is set to zero. The solution is
    exactly the same as with row replacement alone; iterative solvers, and AMG preconditioners in
    particular, need the symmetric operator (row replacement alone keeps the full column coupling of
    every constrained DOF, which equilibration then amplifies by orders of magnitude). Without
    ``rhs``, only the rows are replaced -- still correct, just not symmetric.

    The constrained DOF indices are precomputed once per step (by the solver's
    ``locateConstrainedDofs``) and cached on each boundary condition, so here we
    just read ``dirichlet.constrainedDofIndices``.

    Cythonized version for speed!
    http://stackoverflux.com/questions/12129948/scipy-sparse-set-row-to-zeros

    Parameters
    ----------
    K: scipy.sparse.csr_matrix
        The system matrix.
    dirichlets: list
        The list of dirichlet boundary conditions.
    rhs
        Optional right-hand side(s) of ``K ddU = rhs`` -- one vector of shape ``(n,)`` or several as
        the columns of an ``(n, k)`` array, in any memory layout -- whose constrained rows already
        hold the prescribed values. Modified in place.

    Returns
    -------
    scipy.sparse.csr_matrix
        The modified system matrix.
    """
    if len(dirichlets) == 0:
        return K

    # gather the (precomputed) constrained DOF indices of all dirichlet bcs
    all_indices = []
    for d in dirichlets:
        all_indices.append(d.constrainedDofIndices)

    cdef long[::1] dirichletIndices = np.concatenate(all_indices).astype(np.int64)

    cdef int i, j, k, row, col
    cdef int [::1] indices = K.indices
    cdef int [::1] indptr = K.indptr
    cdef double[::1] data = K.data
    cdef int n_indices = dirichletIndices.shape[0]
    cdef int n = K.shape[0]

    if rhs is None:
        for i in range(n_indices):
            row = dirichletIndices[i]

            # Access the range for this specific row once
            for j in range(indptr[row], indptr[row + 1]):
                if indices[j] == row:
                    data[j] = 1.0  # Diagonal
                else:
                    data[j] = 0.0  # Off-diagonal

        # NOTE: the zeroed off-diagonals are deliberately kept as explicitly stored zeros.
        # This preserves the sparsity pattern, so the assembled CSR matrix can be updated
        # in place across iterations and direct solvers can reuse their symbolic
        # factorization. (Direct solvers handle stored zeros without issues.)
        return K

    # a strided 2D view of the right-hand side(s): writes go to the caller's own memory, whatever its
    # layout (e.g. the arc-length solver's two right-hand sides are the columns of a transposed array)
    rhsArray = np.asarray(rhs)
    cdef double[:, :] b = rhsArray[:, None] if rhsArray.ndim == 1 else rhsArray
    cdef int nRhs = b.shape[1]

    isConstrainedArray = np.zeros(n, dtype=np.uint8)
    isConstrainedArray[dirichletIndices] = 1
    cdef unsigned char[::1] isConstrained = isConstrainedArray

    for row in range(n):
        if isConstrained[row]:
            for j in range(indptr[row], indptr[row + 1]):
                if indices[j] == row:
                    data[j] = 1.0  # Diagonal
                else:
                    data[j] = 0.0  # Off-diagonal
        else:
            for j in range(indptr[row], indptr[row + 1]):
                col = indices[j]
                if isConstrained[col] and data[j] != 0.0:
                    # move the prescribed value's coupling to the right-hand side
                    for k in range(nRhs):
                        b[row, k] -= data[j] * b[col, k]
                    data[j] = 0.0

    # NOTE: as above, the zeroed entries (rows and columns) stay stored -- the pattern is unchanged.
    return K
