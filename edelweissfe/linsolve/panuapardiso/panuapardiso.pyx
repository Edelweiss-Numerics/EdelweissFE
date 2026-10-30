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

"""
The commercial Panua PARDISO solver: :class:`PanuaPardisoFactorization` reorders and factorizes a matrix once (phases 11
and 22) and solves for as many right-hand sides as needed (phase 33).
It is only available, if the PARDISO solver is installed on the system.
You can get the binaries from https://panua.ch/. A license is required to use the solver.
"""

import os

import numpy as np
cimport numpy as np


cdef extern from "pardiso.h" nogil:
    void pardiso(void*, int*, int *, int*, int *, int *,
                 void*, int*, int *, int*, int *, int *,
                 int*, void*, void*, int*, double*)

    void pardisoinit(void*, int*, int*, int*, double*, int*)


cdef class PanuaPardisoFactorization:
    """The Panua PARDISO factorization of the last :meth:`factorize`, for :meth:`solve`. Owns PARDISO's internal memory,
    which it releases (phase -1) when it factorizes again or goes."""

    cdef void* pt[64]  # internal solver memory pointer
    cdef int iparm[64]  # parameters for pardiso
    cdef double dparm[64]  # parameters for pardiso
    cdef int mtype  # real and unsymmetric matrix
    cdef int maxfct  # maximum number of numerical factorizations
    cdef int mnum  # which factorization to use
    cdef int msglvl  # print statistical information
    cdef int rows
    cdef bint ptIsActive
    # the matrix the factorization belongs to, in PARDISO's 1-based indexing: phase 33 is handed its arrays again
    cdef double[::1] data
    cdef int[::1] indices
    cdef int[::1] indptr

    def __cinit__(self):
        self.mtype = 11
        self.maxfct = 1
        self.mnum = 1
        self.msglvl = 0
        self.rows = 0
        self.ptIsActive = False

    def __dealloc__(self):
        self._releaseMemory()

    cdef void _releaseMemory(self):
        """Release PARDISO's internal memory (phase -1)."""
        cdef int phase = -1
        cdef int error = 0
        cdef int nRhs = 1
        cdef int idum = 0
        cdef double ddum = 0
        if not self.ptIsActive:
            return
        pardiso(self.pt, &self.maxfct, &self.mnum, &self.mtype, &phase,
                &self.rows, &ddum, &idum, &idum, &idum, &nRhs,
                self.iparm, &self.msglvl, &ddum, &ddum, &error, self.dparm)
        self.ptIsActive = False

    cdef void _run(self, int phase, int nRhs, double* b, double* x, int* error):
        """One PARDISO phase on the stored matrix."""
        cdef int idum = 0
        pardiso(self.pt, &self.maxfct, &self.mnum, &self.mtype, &phase,
                &self.rows, &self.data[0], &self.indptr[0], &self.indices[0], &idum, &nRhs,
                self.iparm, &self.msglvl, b, x, error, self.dparm)

    def release(self):
        """Release PARDISO's internal memory and the matrix, the factorization (nothing to release before the
        first)."""
        self._releaseMemory()
        self.data = None
        self.indices = None
        self.indptr = None

    def factorize(self, A):
        """Reorder and factorize the square sparse matrix ``A`` (CSR): phases 11 and 22.

        Raises
        ------
        numpy.linalg.LinAlgError
            If PARDISO reports an error (its error code in the message).
        """
        cdef int solver = 0  # use sparse direct solver
        cdef int error = 0
        cdef double ddum = 0
        self.release()
        A = A.tocsr()
        self.rows = A.shape[0]
        # a copy: phase 33 reads the values again (iterative refinement), which must be those it factorized
        self.data = np.array(A.data, dtype=np.float64, copy=True)
        self.indices = np.ascontiguousarray(A.indices + 1, dtype=np.intc)  # pardiso uses fortran 1-based indexing
        self.indptr = np.ascontiguousarray(A.indptr + 1, dtype=np.intc)
        if self.rows == 0:
            return

        os.environ["PARDISOLICMESSAGE"] = "1"  # don't print license message
        pardisoinit(self.pt, &self.mtype, &solver, self.iparm, self.dparm, &error)
        if error != 0:
            raise np.linalg.LinAlgError("Panua PARDISO: initialization failed with error code {:}".format(error))
        self.ptIsActive = True
        # custom parameters: the defaults (iparm[0] = 0), and the number of threads
        self.iparm[0] = 0
        omp_num_threads = os.environ.get("OMP_NUM_THREADS")
        self.iparm[2] = int(omp_num_threads) if omp_num_threads is not None else -1

        for phase in (11, 22):  # reordering and symbolic factorization, numerical factorization
            self._run(phase, 1, &ddum, &ddum, &error)
            if error != 0:
                self._releaseMemory()
                raise np.linalg.LinAlgError(
                    "Panua PARDISO: phase {:} failed with error code {:}".format(phase, error)
                )

    def solve(self, b):
        """``A^-1 b`` with the factorization of the last :meth:`factorize` (phase 33), for one right-hand side (a
        vector) or several (the columns of a matrix); ``b`` is not changed."""
        cdef int error = 0
        cdef double[::1, :] b_ = np.asfortranarray(np.array(b, dtype=np.float64, copy=True).reshape(self.rows, -1))
        cdef double[::1, :] x = np.zeros((self.rows, b_.shape[1]), dtype=np.float64, order="F")
        cdef int nRhs = b_.shape[1]
        if self.rows > 0 and nRhs > 0:
            if not self.ptIsActive:
                raise RuntimeError("Panua PARDISO: solve() before a successful factorize()")
            self._run(33, nRhs, &b_[0, 0], &x[0, 0], &error)
            if error != 0:
                raise np.linalg.LinAlgError("Panua PARDISO: the solve failed with error code {:}".format(error))
        return np.asarray(x).reshape(np.shape(b))
