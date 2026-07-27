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
#  Alexander Dummer alexander.dummer@uibk.ac.at
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
# Created on Fri Feb  9 20:38:16 2018

# @author: matthias

"""
This module provides an interface to the PARDISO solver provided by the Intel Math Kernel Library (MKL).
"""

import numpy as np

cimport numpy as np


cdef extern from "mkl.h" nogil:
    void pardiso(void*, int*,  int*, int*,
                 int*,  int*, void*, int*,
                 int*,  int*,  int*, int*,
                 int*, void*, void*, int*)

    void pardisoinit(void*, int*, int*)


def setParameter(iparm, idx: int, value: int):
    """
    Set a parameter of the PARDISO solver.

    Parameters
    ----------
    iparm : int[:]
        The parameter array.
    idx : int
        The index of the parameter.
    value : int
        The value of the parameter.
    """
    iparm[idx] = value

    return iparm


cdef class PardisoSolver:
    """
    Stateful interface to the MKL PARDISO solver.

    The reordering and symbolic factorization (PARDISO phase 11) depend only on the
    sparsity pattern of the system matrix, so it is in principle safe to compute it
    once and reuse it for every subsequent solve with the same pattern (each solve
    then only performs the numerical factorization, phase 22, and back substitution,
    phase 33). A change of the sparsity pattern is detected automatically and
    triggers a re-analysis.

    However, this reuse has been observed to silently produce numerically wrong
    results (with PARDISO reporting ``error == 0``, so the usual NaN-based failure
    check does not catch it) for some coupled-DOF problems where the numerically
    relevant pivot structure shifts substantially between solves even though the
    sparsity pattern itself does not change. Reuse is therefore **disabled by
    default**; pass ``reuseSymbolicFactorization=True`` to opt in once this has been
    verified safe for the problem at hand (e.g. by comparing against one-shot
    solves on the actual matrix sequence). With reuse disabled, this class behaves
    like the free function :func:`pardisoSolve`, just as a reusable object.

    The number of threads is controlled by MKL via the usual environment variables
    (``OMP_NUM_THREADS`` / ``MKL_NUM_THREADS``).
    """

    cdef long pt[64]      # internal solver memory pointer
    cdef int iparm[64]    # parameters for pardiso
    cdef int mtype        # real and unsymmetric matrix
    cdef int maxfct
    cdef int mnum
    cdef int msglvl
    cdef int rows
    cdef bint ptIsActive  # pt may hold PARDISO-internal allocations (phase -1 required)
    cdef bint hasSymbolicFactorization
    cdef bint reuseSymbolicFactorization

    # the pattern arrays of the currently analyzed matrix (0-based, for change detection)
    cdef object currentIndices
    cdef object currentIndptr
    # persistent 1-based copies handed to pardiso (fortran indexing)
    cdef int[::1] indicesFortran
    cdef int[::1] indptrFortran

    def __cinit__(self, reuseSymbolicFactorization=False):
        cdef int i

        self.mtype = 11
        self.maxfct = 1
        self.mnum = 1
        self.msglvl = 0
        self.ptIsActive = False
        self.hasSymbolicFactorization = False
        self.reuseSymbolicFactorization = reuseSymbolicFactorization
        self.currentIndices = None
        self.currentIndptr = None

        # PARDISO requires pt to be all zeros before the first call
        for i in range(64):
            self.pt[i] = 0
            self.iparm[i] = 0

    def __dealloc__(self):
        self._releaseMemory()

    cdef void _releaseMemory(self):
        """Release all internal PARDISO memory (phase -1)."""
        cdef int phase = -1
        cdef int error = 0
        cdef int nRhs = 1
        cdef int idum = 0
        cdef double ddum = 0

        if not self.ptIsActive:
            return

        pardiso(self.pt, &self.maxfct, &self.mnum, &self.mtype, &phase,
                &self.rows, &ddum, &idum, &idum, &idum, &nRhs,
                &self.iparm[0], &self.msglvl, &ddum, &ddum, &error)

        self.ptIsActive = False
        self.hasSymbolicFactorization = False
        self.currentIndices = None
        self.currentIndptr = None

    cdef bint _hasSamePattern(self, A):
        """Check if the sparsity pattern of A matches the analyzed one."""
        if not self.hasSymbolicFactorization:
            return False

        indices = A.indices
        indptr = A.indptr

        # fast path: in-place assembly reuses the identical pattern arrays
        if indices is self.currentIndices and indptr is self.currentIndptr:
            return True

        if (
            A.shape[0] == self.rows
            and np.array_equal(indptr, self.currentIndptr)
            and np.array_equal(indices, self.currentIndices)
        ):
            # same pattern in new arrays; adopt them for future identity checks
            self.currentIndices = indices
            self.currentIndptr = indptr
            return True

        return False

    cdef int _analyze(self, A) except -1:
        """Run reordering and symbolic factorization (phase 11) for the pattern of A."""
        cdef int phase = 11
        cdef int error = 0
        cdef int nRhs = 1
        cdef int idum = 0
        cdef double ddum = 0

        self._releaseMemory()

        if A.nnz > np.iinfo(np.intc).max:
            raise ValueError(
                "matrix has {:} nonzeros, exceeding the 32-bit PARDISO interface".format(A.nnz)
            )
        if A.shape[0] > np.iinfo(np.intc).max:
            raise ValueError(
                "matrix has {:} rows, exceeding the 32-bit PARDISO interface".format(A.shape[0])
            )

        self.rows = A.shape[0]
        # pardiso uses fortran 1-based indexing; scipy may use int64 index arrays for
        # large matrices, so cast explicitly to the 32-bit interface type
        self.indicesFortran = np.ascontiguousarray(A.indices + 1, dtype=np.intc)
        self.indptrFortran = np.ascontiguousarray(A.indptr + 1, dtype=np.intc)

        pardisoinit(self.pt, &self.mtype, &self.iparm[0])
        self.ptIsActive = True

        cdef double[::1] data = A.data

        pardiso(self.pt, &self.maxfct, &self.mnum, &self.mtype, &phase,
                &self.rows, &data[0], &self.indptrFortran[0], &self.indicesFortran[0], &idum, &nRhs,
                &self.iparm[0], &self.msglvl, &ddum, &ddum, &error)

        if error != 0:
            self._releaseMemory()
            raise RuntimeError("PARDISO analysis failed with error code {:}".format(error))

        self.hasSymbolicFactorization = True
        self.currentIndices = A.indices
        self.currentIndptr = A.indptr

        return 0

    def __call__(self, A, b):
        """
        Solve a linear system of equations.

        Parameters
        ----------
        A : csr_matrix
            The system matrix.
        b : ndarray
            The right-hand side vector (or matrix for multiple right-hand sides).

        Returns
        -------
        ndarray
            The solution vector.
        """

        # if reuse is disabled, always re-analyze; _hasSamePattern is not even
        # evaluated in that case (see the class docstring for why reuse is opt-in)
        if not self.reuseSymbolicFactorization or not self._hasSamePattern(A):
            self._analyze(A)

        cdef double[::1] data = A.data

        # prepare rhs and solution
        cdef double[::1, :] b_ = np.asfortranarray(b.reshape((self.rows, -1)))
        cdef int nRhs = b_.shape[1]
        cdef double[::1, :] x = np.zeros_like(b_, order="F")

        cdef int phase
        cdef int error = 0
        cdef int idum = 0
        cdef double ddum = 0

        # numerical factorization
        phase = 22
        pardiso(self.pt, &self.maxfct, &self.mnum, &self.mtype, &phase,
                &self.rows, &data[0], &self.indptrFortran[0], &self.indicesFortran[0], &idum, &nRhs,
                &self.iparm[0], &self.msglvl, &ddum, &ddum, &error)

        if error == 0:
            # back substitution and iterative refinement
            phase = 33
            pardiso(self.pt, &self.maxfct, &self.mnum, &self.mtype, &phase,
                    &self.rows, &data[0], &self.indptrFortran[0], &self.indicesFortran[0], &idum, &nRhs,
                    &self.iparm[0], &self.msglvl, &b_[0, 0], &x[0, 0], &error)

        if error != 0:
            # signal failure via NaNs; the nonlinear solvers translate this into a cutback
            np.asarray(x).fill(np.nan)

        return np.reshape(x, b.shape)

    def invalidate(self):
        """
        Force a fresh reordering and symbolic factorization (PARDISO phase 11) on the
        next solve, regardless of what the array-identity / ``array_equal`` pattern
        check in :meth:`_hasSamePattern` would otherwise conclude.

        Call this whenever the caller knows the sparsity pattern may have changed
        through a channel the automatic detection might not reliably catch — e.g. a
        solver that rebuilds its CSR generator whenever the active domain changes,
        which can happen more often than once per analysis step (unlike EdelweissFE's
        own static-mesh usage, where a fresh instance is constructed once per step and
        the pattern is never actually re-checked against a real change).

        No-op when ``reuseSymbolicFactorization`` is False, since every solve already
        re-analyzes unconditionally in that case.
        """
        self.hasSymbolicFactorization = False


def pardisoSolve(A, b):
    """
    Solve a linear system of equations using the Intel MKL PARDISO solver.

    One-shot convenience wrapper around :class:`PardisoSolver`; for repeated solves
    with an identical sparsity pattern, use a persistent :class:`PardisoSolver`
    instance to reuse the symbolic factorization.

    Parameters
    ----------
    A : csr_matrix
        The system matrix.
    b : ndarray
        The right-hand side vector.

    Returns
    -------
    ndarray
        The solution vector.
    """

    return PardisoSolver()(A, b)
