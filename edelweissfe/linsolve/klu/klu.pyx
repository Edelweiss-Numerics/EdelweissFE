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

"""
SuiteSparse's KLU, a sparse LU factorization: :class:`KLUFactorization` factorizes a matrix once (``klu_analyze``,
``klu_factor``) and solves for as many right-hand sides as needed (``klu_solve``).
"""

import numpy as np

cimport numpy as np


cdef extern from "klu.h" nogil:
    ctypedef struct klu_common:
        int status
    ctypedef struct klu_symbolic:
        pass
    ctypedef struct klu_numeric:
        pass
    int KLU_SINGULAR
    int klu_defaults(klu_common* common)
    klu_symbolic* klu_analyze(int n, int* Ap, int* Ai, klu_common* common)
    klu_numeric* klu_factor(int* Ap, int* Ai, double* Ax, klu_symbolic* symbolic, klu_common* common)
    int klu_solve(klu_symbolic* symbolic, klu_numeric* numeric, int ldim, int nrhs, double* B, klu_common* common)
    int klu_free_symbolic(klu_symbolic** symbolic, klu_common* common)
    int klu_free_numeric(klu_numeric** numeric, klu_common* common)


cdef class KLUFactorization:
    """The KLU factorization of the last :meth:`factorize`, for :meth:`solve`. Owns KLU's symbolic and numeric objects,
    which it frees when it factorizes again or goes."""

    cdef klu_common common
    cdef klu_symbolic* symbolic
    cdef klu_numeric* numeric
    cdef int n

    def __cinit__(self):
        klu_defaults(&self.common)
        self.symbolic = NULL
        self.numeric = NULL
        self.n = 0

    def __dealloc__(self):
        self._free()

    cdef void _free(self):
        if self.numeric != NULL:
            klu_free_numeric(&self.numeric, &self.common)
        if self.symbolic != NULL:
            klu_free_symbolic(&self.symbolic, &self.common)

    def release(self):
        """Free KLU's symbolic and numeric objects, the factorization (nothing to free before the first)."""
        self._free()
        self.n = 0

    def factorize(self, A):
        """Factorize the square sparse matrix ``A``: its ordering (``klu_analyze``) and its LU factors
        (``klu_factor``).

        Raises
        ------
        numpy.linalg.LinAlgError
            If ``A`` is singular, or KLU fails otherwise (``Common.status`` in the message).
        """
        # KLU reads the matrix by columns
        csc = A.tocsc()
        cdef int[::1] indptr = np.ascontiguousarray(csc.indptr, dtype=np.int32)
        cdef int[::1] indices = np.ascontiguousarray(csc.indices, dtype=np.int32)
        cdef double[::1] data = np.ascontiguousarray(csc.data, dtype=np.float64)
        cdef int n = csc.shape[0]
        self.release()
        if n == 0:
            return
        self.symbolic = klu_analyze(n, &indptr[0], &indices[0], &self.common)
        if self.symbolic == NULL:
            raise np.linalg.LinAlgError("KLU: the analysis failed (status {:})".format(self.common.status))
        self.numeric = klu_factor(&indptr[0], &indices[0], &data[0], self.symbolic, &self.common)
        if self.numeric == NULL:
            status = self.common.status
            self._free()
            if status == KLU_SINGULAR:
                raise np.linalg.LinAlgError("KLU: the matrix is singular")
            raise np.linalg.LinAlgError("KLU: the factorization failed (status {:})".format(status))
        self.n = n

    def solve(self, b):
        """``A^-1 b`` with the factorization of the last :meth:`factorize`, for one right-hand side (a vector) or
        several (the columns of a matrix); ``b`` is not changed."""
        shape = np.shape(b)
        # KLU overwrites the right-hand sides with the solutions, column by column
        cdef double[::1, :] x = np.asfortranarray(np.array(b, dtype=np.float64, copy=True).reshape(self.n, -1))
        cdef int nRhs = x.shape[1]
        if self.n > 0 and nRhs > 0:
            if self.numeric == NULL:
                raise RuntimeError("KLU: solve() before a successful factorize()")
            klu_solve(self.symbolic, self.numeric, self.n, nRhs, &x[0, 0], &self.common)
        return np.asarray(x).reshape(shape)
