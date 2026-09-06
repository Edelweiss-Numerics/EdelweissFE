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

from libcpp.string cimport string


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
        void set_nullspace(const double* B, int rows, int cols) except +
        void build(int n, const int* ptr, const int* col, const double* val) except +
        void applyPreconditioner(int n, const double* rhs, double* x) except +
        string report() except +

    # Same interface, backed by amgcl::backend::builtin<float> -- half the memory traffic in the
    # smoother apply, at the cost of matrix/hierarchy precision. rhs/x stay double at this
    # boundary too; the narrowing/widening happens inside the C++ wrapper.
    cdef cppclass LinearSolverFloat:
        LinearSolverFloat(const char* json_params) except +
        void solve(int n,
                   const int* ptr,
                   const int* col,
                   const float* val,
                   const double* rhs,
                   double* x,
                   int& iters,
                   double& error) except +
        void set_nullspace(const double* B, int rows, int cols) except +
        void build(int n, const int* ptr, const int* col, const float* val) except +
        void applyPreconditioner(int n, const double* rhs, double* x) except +
        string report() except +

    # Block-valued backends: the matrix stays a plain scalar CSR at this boundary (val is
    # double, same layout as LinearSolver) -- the wrapper adapts it to node-major B x B blocks
    # internally via amgcl::adapter::block_matrix. set_nullspace() always throws (AMGCL's own
    # tentative-prolongation nullspace path is unimplemented for block value types).
    cdef cppclass LinearSolverBlock2:
        LinearSolverBlock2(const char* json_params) except +
        void solve(int n,
                   const int* ptr,
                   const int* col,
                   const double* val,
                   const double* rhs,
                   double* x,
                   int& iters,
                   double& error) except +
        void set_nullspace(const double* B, int rows, int cols) except +
        void build(int n, const int* ptr, const int* col, const double* val) except +
        void applyPreconditioner(int n, const double* rhs, double* x) except +
        string report() except +

    cdef cppclass LinearSolverBlock3:
        LinearSolverBlock3(const char* json_params) except +
        void solve(int n,
                   const int* ptr,
                   const int* col,
                   const double* val,
                   const double* rhs,
                   double* x,
                   int& iters,
                   double& error) except +
        void set_nullspace(const double* B, int rows, int cols) except +
        void build(int n, const int* ptr, const int* col, const double* val) except +
        void applyPreconditioner(int n, const double* rhs, double* x) except +
        string report() except +

    # A standalone, OpenMP-threaded relaxation smoother -- e.g. the p-two-grid
    # preconditioner's fine sweep, one level below a full AMG hierarchy. Its own constructor
    # parameter tree is flat ({"type": "chebyshev", ...}), not nested under "precond.relax".
    cdef cppclass RelaxationSmoother:
        RelaxationSmoother(const char* json_params) except +
        void build(int n, const int* ptr, const int* col, const double* val) except +
        void applyStep(int n, const double* rhs, double* x) except +
        void residual(int n, const double* rhs, const double* x, double* r) except +

    # A plain OpenMP-threaded matvec/residual, no smoother -- the outer GMRES operator.
    cdef cppclass ThreadedMatrix:
        ThreadedMatrix() except +
        void build(int n, const int* ptr, const int* col, const double* val) except +
        void matvec(int n, const double* x, double* y) except +
        void residual(int n, const double* rhs, const double* x, double* r) except +
        void buildRect(int nrows, int ncols, const int* ptr, const int* col, const double* val) except +
        void matvecRect(const double* x, double* y) except +

    # Callback signature bridging AMGCL's native lgmres Krylov loop back into a Python-level
    # preconditioner -- see amgcl-wrapper.hpp's own comment on PyPrecondApplyFn/
    # PyLGMRESPrecondT for why this is a bare function pointer + opaque context rather than a
    # Cython-overridden C++ virtual method.
    ctypedef void (*PyPrecondApplyFn)(void* ctx, int n, const double* rhs, double* x)

    # AMGCL's own native outer Krylov solve, in place of scipy.sparse.linalg.gmres. Persists
    # its recycled/augmented Krylov vectors across solve() calls on the same instance whenever
    # always_reset is set to false in the constructor's params -- see LGMRESOuterSolverT's own
    # comment for why this class is meant to be built once per BlockAMGSolver, not once per solve.
    cdef cppclass LGMRESOuterSolver:
        LGMRESOuterSolver(int n, const char* json_params) except +
        void solve(int n,
                   const int* ptr,
                   const int* col,
                   const double* val,
                   const double* rhs,
                   double* x,
                   double tol,
                   int maxiter,
                   PyPrecondApplyFn applyFn,
                   void* ctx,
                   int& iters,
                   double& error,
                   bint resetOnce) except +

    # OpenMP-threaded sparse-matrix product/sum, wired into
    # edelweissfe/numerics/mpctransformation.py's T^T K T + C condensation as an opt-in alternative
    # (useAmgclSpgemm/useAmgclMPCCondensation) to the plain expression's scipy CSR (single-threaded)
    # routines. Square matrices only -- see SpGEMMHelperT's own comment for why.
    cdef cppclass SpGEMMHelper:
        SpGEMMHelper() except +
        void product(int aN, const int* aPtr, const int* aCol, const double* aVal,
                     int bN, const int* bPtr, const int* bCol, const double* bVal) except +
        void sum(double alpha, int aN, const int* aPtr, const int* aCol, const double* aVal,
                 double beta, int bN, const int* bPtr, const int* bCol, const double* bVal) except +
        int resultNRows() except +
        int resultNCols() except +
        int resultNnz() except +
        void copyResult(int* ptrOut, int* colOut, double* valOut) except +
