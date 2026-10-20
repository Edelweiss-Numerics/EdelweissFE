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
from libc.stdint cimport int32_t, int64_t, uint8_t


cdef extern from "_mpccondensation.h":
    cdef cppclass MPCCondensationCore nogil:
        MPCCondensationCore(int n, const int32_t* Tp, const int32_t* Ti, const double* Tv,
                            const int32_t* Ttp, const int32_t* Tti, const double* Ttv,
                            const int32_t* Cp, const int32_t* Ci, const double* Cv) except +

        int64_t countPattern(const int32_t* Kp, const int32_t* Ki, int32_t* outPtr, uint8_t* rowIsCopy,
                             int numThreads) except +
        void fill(const int32_t* Kp, const int32_t* Ki, const double* Kv, const int32_t* outPtr,
                  const uint8_t* rowIsCopy, int32_t* outIdx, double* outVal, int numThreads) except +


def _csrArrays32(A, name):
    """The CSR arrays of ``A`` as contiguous int32/float64 arrays (copied only if their type differs)."""
    if A.nnz > np.iinfo(np.int32).max:
        raise OverflowError("MPC condensation: {:} has more than 2^31 - 1 entries.".format(name))
    return (
        np.ascontiguousarray(A.indptr, dtype=np.int32),
        np.ascontiguousarray(A.indices, dtype=np.int32),
        np.ascontiguousarray(A.data, dtype=np.float64),
    )


cdef class MPCCondensation:
    """The multi-point-constraint condensation :math:`\\tilde{K} = T^T K \\, T + C` of a system matrix, computed
    in one OpenMP-threaded pass over the rows of the result, with no intermediate matrix.

    Each row of the result is built by one thread, which sums the rows of :math:`K` that row of
    :math:`T^T` selects, each multiplied into :math:`T`. The sums are taken in the same order as AMGCL's
    single-threaded-per-row product (``spgemm_saad``) evaluates :math:`T^T (K T) + C`, and the result does not
    depend on the number of threads. Rows of :math:`K` that no constraint touches are copied unchanged. The
    column indices of the result are sorted; explicit zeros are kept (the pattern is the structural pattern of
    :math:`T^T K T + C`).

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

    def condense(self, K, int numThreads=0) -> csr_matrix:
        """Compute :math:`T^T K \\, T + C`.

        Parameters
        ----------
        K
            The system matrix (CSR, ``nDof x nDof``). Its rows need not be sorted, duplicates are summed.
        numThreads
            The number of OpenMP threads; 0 for the OpenMP default (``OMP_NUM_THREADS``).

        Returns
        -------
        csr_matrix
            The condensed matrix, with sorted column indices.
        """
        if K.shape != (self.n, self.n):
            raise ValueError("MPC condensation: K has shape {:}, expected {:}.".format(K.shape, (self.n, self.n)))

        Kp_, Ki_, Kv_ = _csrArrays32(K, "K")
        cdef int32_t[::1] Kp = Kp_, Ki = Ki_
        cdef double[::1] Kv = Kv_
        cdef int32_t emptyIdx = 0
        cdef double emptyVal = 0.0
        cdef const int32_t* KiPtr = &Ki[0] if Ki.shape[0] else &emptyIdx
        cdef const double* KvPtr = &Kv[0] if Kv.shape[0] else &emptyVal

        outPtr_ = np.empty(self.n + 1, dtype=np.int32)
        rowIsCopy_ = np.empty(max(self.n, 1), dtype=np.uint8)
        cdef int32_t[::1] outPtr = outPtr_
        cdef uint8_t[::1] rowIsCopy = rowIsCopy_
        cdef int64_t nnz

        with nogil:
            nnz = self.core.countPattern(&Kp[0], KiPtr, &outPtr[0], &rowIsCopy[0], numThreads)

        outIdx_ = np.empty(max(nnz, 1), dtype=np.int32)
        outVal_ = np.empty(max(nnz, 1), dtype=np.float64)
        cdef int32_t[::1] outIdx = outIdx_
        cdef double[::1] outVal = outVal_

        with nogil:
            self.core.fill(&Kp[0], KiPtr, KvPtr, &outPtr[0], &rowIsCopy[0], &outIdx[0], &outVal[0], numThreads)

        Kt = csr_matrix((outVal_[:nnz], outIdx_[:nnz], outPtr_), shape=(self.n, self.n))
        Kt.has_canonical_format = True
        return Kt
