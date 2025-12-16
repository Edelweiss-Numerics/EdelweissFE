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
        CSRCore(long* I, long* J, long n_pairs, long n_dof) except +
        vector[int] indptr
        vector[int] indices
        long nnz
        long nDof

        void update( const double* V_data, double* csr_data ) nogil

cdef class CSRGenerator:
    """
    CSRGenerator class to create and manage a CSR matrix from COO format.

    Parameters
    ----------
    systemMatrix : object
        An object containing COO format data with attributes I, J, and nDof.
    """

    cdef CSRCore* core
    cdef public object csrMatrix
    cdef double[:] data_view
    cdef int nCooPairs

    def __dealloc__(self):
        if self.core != NULL:
            del self.core

    def __init__(self, systemMatrix):
        cdef long[::1] I = np.ascontiguousarray(systemMatrix.I.astype(np.int64))
        cdef long[::1] J = np.ascontiguousarray(systemMatrix.J.astype(np.int64))
        self.nCooPairs = len(I)
        cdef long nDof = systemMatrix.nDof

        # 1. Run C++ Core
        with nogil:
            self.core = new CSRCore(&I[0], &J[0], self.nCooPairs, nDof)

        # 2. Direct Zero-Copy Access
        # Cython allows us to call .data() on the vector attribute directly
        cdef int* ptr_indptr = self.core.indptr.data()
        cdef int* ptr_indices = self.core.indices.data()
        # cdef int* ptr_map = self.core.map_to_csr.data()

        # 3. Cast to Memoryviews
        cdef long nnz = self.core.nnz

        # We tell Cython: "This pointer is an array of size X"
        cdef int[::1] view_indptr = <int[:nDof+1]> ptr_indptr
        cdef int[::1] view_indices = <int[:nnz]> ptr_indices

        # 4. Create NumPy arrays (Shared Memory)
        cdef np.ndarray nd_indptr = np.asarray(view_indptr)
        cdef np.ndarray nd_indices = np.asarray(view_indices)

        # 5. Create Scipy Object
        cdef np.ndarray[double, ndim=1] data = np.zeros(nnz, dtype=np.double)
        self.csrMatrix = csr_matrix((data, nd_indices, nd_indptr), shape=(nDof, nDof))

        # 6. Safety: Keep 'self' alive as long as 'csrMatrix' is alive
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

        # Pointers to raw data
        cdef double* d_ptr = &self.data_view[0]
        cdef double* v_ptr = &V[0]

        # Release GIL:
        # C++ std::execution will spawn its own thread pool (e.g. TBB).
        with nogil:
            self.core.update(v_ptr, d_ptr )

        return self.csrMatrix
