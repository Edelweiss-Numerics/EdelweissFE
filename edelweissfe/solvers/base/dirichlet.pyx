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
def applyDirichletToStiffness(K: csr_matrix, dirichlets: Iterable) -> csr_matrix:
    """Impose the Dirichlet BCs on the global stiffness matrix (row-replacement method).

    For every constrained DOF (row) we set the whole row to zero and put 1.0 on
    the diagonal. Solving  K ddU = R  then simply returns  ddU[row] = R[row],
    i.e. the value that :func:`applyDirichletToResidual` wrote into R for that DOF.

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

    cdef int i, j, row
    cdef int [::1] indices = K.indices
    cdef int [::1] indptr = K.indptr
    cdef double[::1] data = K.data
    cdef int n_indices = dirichletIndices.shape[0]

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
