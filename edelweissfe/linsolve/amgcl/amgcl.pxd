# amgcl_wrapper.pxd
# cython: language_level=3

cdef extern from "amgcl-wrapper.hpp":
    cdef cppclass LinearSolver:
        LinearSolver(const char* json_params) except +
        void solve(int n,
                   const int* ptr,
                   const int* col,
                   const double* val,
                   const double* rhs,
                   double* x,
                   int& iters,
                   double& error) except +
