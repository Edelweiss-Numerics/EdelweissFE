import json
import time

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

    def solve(self, object A, np.ndarray[np.float64_t, ndim=1, mode="c"] rhs):
        """
        A: scipy.sparse.csr_matrix
        rhs: numpy.ndarray (1D)
        """
        # 1. Validate and Enforce CSR Format
        if not scipy.sparse.isspmatrix_csr(A):
            # If not CSR, convert it (might incur copy overhead)
            A = A.tocsr()

        cdef int n = A.shape[0]
        if rhs.shape[0] != n:
            raise ValueError(f"Dimension mismatch: Matrix {n}x{n}, RHS {rhs.shape[0]}")

        # 2. Extract and Cast Arrays
        # AMGCL expects C-int (int32 usually). SciPy indptr can be int64 for huge matrices.
        # We enforce int32 here for safety with the C++ 'int*' signature.
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
        cdef double[::1] rhs_ = rhs
        cdef double[::1] x_ = x

        tic = time.time()

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
        toc = time.time()
        print(f"AMGCL solve time: {toc - tic:.6f} seconds, iters: {iters}, error: {error:.2e}")

        return x
