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
    # The interface every AMGCL solver backend implements (scalar or B x B blocks, double or float):
    # the matrix values and rhs/x are always double at this boundary, each backend converts
    # internally -- see AMGCLSolverInterface. set_nullspace() throws on a block backend (AMGCL's own
    # tentative-prolongation nullspace path is unimplemented for block value types).
    cdef cppclass AMGCLSolverInterface:
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

    # The backend for a block size (1, 2 or 3) and value type (float: builtin<float> for scalars,
    # 3x3 blocks set up in double and stored in float), owned by the caller.
    AMGCLSolverInterface* makeAMGCLSolver(const char* json_params, int blockSize, bint isFloat) except +

    # blockamg's equilibrated, field-split system As = D A D: the outer Krylov operator, the
    # coupling blocks between fields and, on demand, the diagonal blocks -- built in one threaded
    # pass, see EquilibratedSystemT's own comment.
    cdef cppclass EquilibratedSystem:
        EquilibratedSystem() except +

        @staticmethod
        void absoluteDiagonal(int n, const int* ptr, const int* col, const double* val, double* out) except +
        void build(int n, const int* ptr, const int* col, const double* val, const double* dinv,
                   int nFields, const int* fieldBounds) except +
        int diagonalBlockSize(int f) except +
        int diagonalBlockNnz(int f) except +
        void fillDiagonalBlock(int f, int* ptr, int* col, double* val) except +
        void matvec(const double* x, double* y) except +
        void residual(const double* rhs, const double* x, double* r) except +
        void couplingMatvec(int f, int g, const double* x, double* y) except +

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
        void solve(const EquilibratedSystem& system,
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

    # The p-two-grid (P1) preconditioner of edelweissfe/linsolve/blockamg/ptwogrid.py, built and
    # applied entirely in C++ (OpenMP) -- see PTwoGridPreconditionerT's own comment.
    cdef cppclass PTwoGridPreconditioner:
        PTwoGridPreconditioner(const char* json_params) except +
        void build(int n, const int* ptr, const int* col, const double* val,
                   int nNodes, const signed char* isCorner, const int* edgeEndpoints,
                   const double* nullspace, int nullspaceRows, int nullspaceCols,
                   double spectralRadius, const double* dominantVector, int dominantVectorSize) except +
        void apply(int n, const double* r, double* x) except +
        double fineSpectralRadius()
        int fineSpectralRadiusSource()
        int fineDominantVectorSize()
        void copyFineDominantVector(double* out)
        int nCoarseFree()
        string report() except +
        double setupDirichletAndPSeconds_
        double setupGalerkinSeconds_
        double setupCoarseSeconds_
        double setupFineOperatorSeconds_
        double setupSpectralRadiusSeconds_
        double applyFineSeconds_
        double applyCoarseSeconds_
