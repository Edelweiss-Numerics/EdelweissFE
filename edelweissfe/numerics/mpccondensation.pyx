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

import numpy as np
from scipy.sparse import csr_matrix

cimport numpy as np
from libc.stdint cimport int32_t, int64_t


cdef extern from "_mpccondensation.h":
    cdef cppclass MPCCondensationCore nogil:
        MPCCondensationCore(int n, const int32_t* Tp, const int32_t* Ti, const double* Tv,
                            const int32_t* Ttp, const int32_t* Tti, const double* Ttv,
                            const int32_t* Cp, const int32_t* Ci, const double* Cv) except +

        int64_t countFillIn(const int32_t* Kp, const int32_t* Ki, int64_t* fillPtr, int numThreads) except +
        void writeFillIn(const int32_t* Kp, const int32_t* Ki, const int64_t* fillPtr, int32_t* fillIdx,
                         int numThreads) except +
        void condenseInPlace(const int32_t* Kp, const int32_t* Ki, double* Kv, int numThreads) except +


def _csrArrays32(A, name):
    """The CSR arrays of ``A`` as contiguous int32/float64 arrays (copied only if their type differs)."""
    # CSC and BSR matrices have an indptr too, and would be read as a different matrix
    if A.format != "csr":
        raise TypeError("MPC condensation: {:} must be CSR, not {:}.".format(name, A.format))
    if A.nnz > np.iinfo(np.int32).max:
        raise OverflowError("MPC condensation: {:} has more than 2^31 - 1 entries.".format(name))
    return (
        np.ascontiguousarray(A.indptr, dtype=np.int32),
        np.ascontiguousarray(A.indices, dtype=np.int32),
        np.ascontiguousarray(A.data, dtype=np.float64),
    )


cdef class MPCCondensation:
    """The multi-point-constraint condensation :math:`\\tilde{K} = T^T K \\, T + C` of a system matrix, computed
    in place -- the values of :math:`\\tilde{K}` overwrite those of :math:`K`, in the pattern of :math:`K` -- in one
    OpenMP-threaded pass over the rows that change.

    The pattern of :math:`K` must hold the structural pattern of :math:`T^T K T + C`. Its rows gain entries where a
    master collects the couplings of its slaves; :meth:`condensedPatternFillIn` returns them, to be added to the
    pattern once (as stored zeros) before the first :meth:`condenseInPlace`. Entries of :math:`K` that the condensed
    matrix does not have (the slave columns, and the slave rows apart from their diagonal) become stored zeros.

    Each changed row is built by one thread, which sums the rows of :math:`K` that row of :math:`T^T` selects, each
    multiplied into :math:`T`. The sums are taken in the fixed order of evaluating :math:`T^T (K T) + C` row by row,
    so the result does not depend on the number of threads. Rows that no constraint touches are left as they are.

    Parameters
    ----------
    T
        The square ``nDof x nDof`` transformation matrix (CSR).
    TTranspose
        :math:`T^T`, as CSR.
    C
        The matrix added to the product (CSR), the identity on the slave DOFs.
    """

    cdef MPCCondensationCore* core
    cdef int n
    # the core borrows these arrays; they are kept alive here
    cdef object _T, _TTranspose, _C

    def __dealloc__(self):
        if self.core != NULL:
            del self.core

    def __init__(self, T, TTranspose, C):
        n = T.shape[0]
        for A, name in ((T, "T"), (TTranspose, "T^T"), (C, "C")):
            if A.shape != (n, n):
                raise ValueError("MPC condensation: {:} has shape {:}, expected {:}.".format(name, A.shape, (n, n)))

        self.n = n
        self._T = _csrArrays32(T, "T")
        self._TTranspose = _csrArrays32(TTranspose, "T^T")
        self._C = _csrArrays32(C, "C")

        cdef int32_t[::1] Tp = self._T[0], Ti = self._T[1]
        cdef double[::1] Tv = self._T[2]
        cdef int32_t[::1] Ttp = self._TTranspose[0], Tti = self._TTranspose[1]
        cdef double[::1] Ttv = self._TTranspose[2]
        cdef int32_t[::1] Cp = self._C[0], Ci = self._C[1]
        cdef double[::1] Cv = self._C[2]

        # empty index/value arrays have no first element; any valid pointer will do, nothing is read
        cdef int32_t emptyIdx = 0
        cdef double emptyVal = 0.0
        self.core = new MPCCondensationCore(
            n,
            &Tp[0], &Ti[0] if Ti.shape[0] else &emptyIdx, &Tv[0] if Tv.shape[0] else &emptyVal,
            &Ttp[0], &Tti[0] if Tti.shape[0] else &emptyIdx, &Ttv[0] if Ttv.shape[0] else &emptyVal,
            &Cp[0], &Ci[0] if Ci.shape[0] else &emptyIdx, &Cv[0] if Cv.shape[0] else &emptyVal,
        )

    def _checkMatrix(self, K):
        """K is read and written through its own arrays, never a converted copy: their types must fit."""
        if K.format != "csr":
            raise TypeError("MPC condensation: K must be CSR, not {:}.".format(K.format))
        if K.shape != (self.n, self.n):
            raise ValueError("MPC condensation: K has shape {:}, expected {:}.".format(K.shape, (self.n, self.n)))
        if K.indptr.dtype != np.int32 or K.indices.dtype != np.int32:
            raise TypeError("MPC condensation: K must have 32-bit indices, not {:}.".format(K.indices.dtype))
        if K.data.dtype != np.float64 or not K.data.flags.c_contiguous:
            raise TypeError("MPC condensation: K must have contiguous float64 values, not {:}.".format(K.data.dtype))

    def condensedPatternFillIn(self, K, int numThreads=0) -> csr_matrix:
        """The entries of the structural pattern of :math:`T^T K T + C` that the pattern of :math:`K` lacks.

        Parameters
        ----------
        K
            The system matrix (CSR, ``nDof x nDof``, 32-bit indices); only its pattern is read.
        numThreads
            The number of OpenMP threads; 0 for the OpenMP default (``OMP_NUM_THREADS``).

        Returns
        -------
        csr_matrix
            The missing entries, as stored zeros, with sorted column indices.
        """
        self._checkMatrix(K)
        cdef int32_t[::1] Kp = K.indptr, Ki = K.indices
        cdef int32_t emptyIdx = 0
        cdef const int32_t* KiPtr = &Ki[0] if Ki.shape[0] else &emptyIdx

        fillPtr_ = np.empty(self.n + 1, dtype=np.int64)
        cdef int64_t[::1] fillPtr = fillPtr_
        cdef int64_t nFill
        with nogil:
            nFill = self.core.countFillIn(&Kp[0], KiPtr, &fillPtr[0], numThreads)

        if nFill > np.iinfo(np.int32).max:
            raise OverflowError("MPC condensation: the fill-in has more than 2^31 - 1 entries.")
        fillIdx_ = np.empty(max(nFill, 1), dtype=np.int32)
        cdef int32_t[::1] fillIdx = fillIdx_
        with nogil:
            self.core.writeFillIn(&Kp[0], KiPtr, &fillPtr[0], &fillIdx[0], numThreads)

        fillIn = csr_matrix(
            (np.zeros(nFill), fillIdx_[:nFill], fillPtr_.astype(np.int32)), shape=(self.n, self.n)
        )
        fillIn.has_canonical_format = True
        return fillIn

    def condenseInPlace(self, K, int numThreads=0):
        """Overwrite the values of :math:`K` with those of :math:`T^T K T + C`, in the pattern of :math:`K`.

        Parameters
        ----------
        K
            The system matrix (CSR, ``nDof x nDof``, 32-bit indices, float64 values), with sorted rows without
            duplicates and a pattern that holds the one of :math:`T^T K T + C` (see :meth:`condensedPatternFillIn`).
            Its values are overwritten; they are undefined if this raises.
        numThreads
            The number of OpenMP threads; 0 for the OpenMP default (``OMP_NUM_THREADS``).
        """
        self._checkMatrix(K)
        cdef int32_t[::1] Kp = K.indptr, Ki = K.indices
        cdef double[::1] Kv = K.data
        cdef int32_t emptyIdx = 0
        cdef double emptyVal = 0.0
        cdef const int32_t* KiPtr = &Ki[0] if Ki.shape[0] else &emptyIdx
        cdef double* KvPtr = &Kv[0] if Kv.shape[0] else &emptyVal
        with nogil:
            self.core.condenseInPlace(&Kp[0], KiPtr, KvPtr, numThreads)
