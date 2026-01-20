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
from libcpp.vector cimport vector


cdef extern from "_csrcore.h":
    cdef cppclass CSRCore nogil:
        CSRCore(const int* I, const int* J, long n_pairs, int n_dof) except +

        vector[int] indptr
        vector[int] indices
        int nnz
        int nDof

        void update(const double* V_data, double* csr_data) nogil

cdef class CSRGenerator:
    """
    CSRGenerator class to create and manage a CSR matrix from COO format.

    This class utilizes a C++ core for efficient conversion and updating of the CSR matrix.

    Parameters
    ----------
    systemMatrix : object
        An object containing COO format data with attributes I, J, and nDof.
    """

    cdef CSRCore* core
    cdef public object csrMatrix
    cdef double[:] data_view
    cdef long nCooPairs  # Kept as long (int64)

    def __dealloc__(self):
        if self.core != NULL:
            del self.core

    def __init__(self, systemMatrix):
        cdef int[::1] I = systemMatrix.I  # noqa
        cdef int[::1] J = systemMatrix.J

        self.nCooPairs = len(I)  # Length is still 64-bit capable

        cdef int nDof = int(systemMatrix.nDof)

        # 1. Run C++ Core
        with nogil:
            self.core = new CSRCore(&I[0], &J[0], self.nCooPairs, nDof)

        cdef int* ptr_indptr = self.core.indptr.data()
        cdef int* ptr_indices = self.core.indices.data()

        cdef int nnz = self.core.nnz

        cdef int[::1] view_indptr = <int[:nDof+1]> ptr_indptr
        cdef int[::1] view_indices = <int[:nnz]> ptr_indices

        cdef np.ndarray nd_indptr = np.asarray(view_indptr)
        cdef np.ndarray nd_indices = np.asarray(view_indices)

        cdef np.ndarray[double, ndim=1] data = np.zeros(nnz, dtype=np.double)
        self.csrMatrix = csr_matrix((data, nd_indices, nd_indptr), shape=(nDof, nDof))

        self.csrMatrix._parent = self

        self.data_view = self.csrMatrix.data

    def updateCSR(self, double[:] V):
        """
        Update the values of the CSR matrix based on the input vector V.

        Parameters
        ----------
        V : double[:]
            Input vector used to update the CSR matrix values.

        Returns
        -------
        csr_matrix
            The updated CSR matrix.
        """

        cdef double* d_ptr = &self.data_view[0]
        cdef double* v_ptr = &V[0]

        with nogil:
            self.core.update(v_ptr, d_ptr)

        return self.csrMatrix.copy()
