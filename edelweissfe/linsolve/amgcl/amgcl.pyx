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

import json

import numpy as np
import scipy.sparse

cimport numpy as np


cdef class PyAMGCLSolver:
    cdef LinearSolver* solver

    def __cinit__(self, dict params=None):
        """
        Initialize with parameters only.
        Example params:
        {
            "solver": {"type": "bicgstab", "tol": 1e-6},
            "precond": {"relaxation": {"type": "ilu0"}}
        }
        """
        if params is None:
            # Default to Robust Blackbox (BiCGStab + ILU0)
            params = {
                "solver": {"type": "bicgstab", "tol": 1e-6},
                "precond": {
                    "coarsening": {"type": "smoothed_aggregation"},
                    "relaxation": {"type": "ilu0"}
                }
            }

        # Convert dict to JSON string for C++
        cdef bytes json_bytes = json.dumps(params).encode("utf-8")
        cdef const char* c_json = json_bytes
        self.solver = new LinearSolver(c_json)

    def __dealloc__(self):
        if self.solver != NULL:
            del self.solver

    def solve(self, object A, object rhs):
        """
        A: scipy.sparse.csr_matrix
        rhs: array-like, will be converted to 1D float64 (C-contiguous)
        """
        if not scipy.sparse.isspmatrix_csr(A):
            A = A.tocsr()

        cdef np.ndarray[np.float64_t, ndim=1, mode="c"] rhs_arr = np.asarray(rhs, dtype=np.float64, order="C")
        if rhs_arr.ndim != 1:
            raise ValueError("rhs must be a 1D array-like")

        cdef int n = A.shape[0]
        if rhs_arr.shape[0] != n:
            raise ValueError(f"Dimension mismatch: Matrix {n}x{n}, RHS {rhs_arr.shape[0]}")

        cdef np.ndarray[np.int32_t, ndim=1, mode="c"] indptr = A.indptr.astype(np.int32, copy=False)
        cdef np.ndarray[np.int32_t, ndim=1, mode="c"] indices = A.indices.astype(np.int32, copy=False)
        cdef np.ndarray[np.float64_t, ndim=1, mode="c"] data = A.data.astype(np.float64, copy=False)

        # Ensure contiguous (astype might return non-contiguous if copy=False was possible)
        if not indptr.flags["C_CONTIGUOUS"]:
            indptr = np.ascontiguousarray(indptr)
        if not indices.flags["C_CONTIGUOUS"]:
            indices = np.ascontiguousarray(indices)
        if not data.flags["C_CONTIGUOUS"]:
            data = np.ascontiguousarray(data)

        # 3. Prepare Solution Array
        cdef np.ndarray[np.float64_t, ndim=1, mode="c"] x = np.zeros(n, dtype=np.float64)

        cdef int iters = 0
        cdef double error = 0.0

        cdef int[::1] indptr_ = indptr
        cdef int[::1] indices_ = indices
        cdef double[::1] data_ = data
        cdef double[::1] rhs_ = rhs_arr
        cdef double[::1] x_ = x

        self.solver.solve(
                n,
                &indptr_[0],
                &indices_[0],
                &data_[0],
                &rhs_[0],
                &x_[0],
                iters,
                error
            )

        return x
