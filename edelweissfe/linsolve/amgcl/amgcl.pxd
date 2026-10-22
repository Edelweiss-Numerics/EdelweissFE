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
    # The interface of every AMGCL solver backend (scalar or block, double or float); values are double
    # at this boundary. On a block backend set_nullspace() takes the translations only.
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

    # blockamg's equilibrated, field-split system As = D A D; see EquilibratedSystemT.
    cdef cppclass EquilibratedSystem:
        EquilibratedSystem() except +

        @staticmethod
        void absoluteDiagonal(int n, const int* ptr, const int* col, const double* val, double* out) except +
        void build(int n, const int* ptr, const int* col, const double* val, const double* scaling,
                   int nFields, const int* fieldBounds, bint unscaled) except +
        int diagonalBlockSize(int f) except +
        int diagonalBlockNnz(int f) except +
        void fillDiagonalBlock(int f, int* ptr, int* col, double* val) except +
        void matvec(const double* x, double* y, bint unscaled) except +
        void residual(const double* rhs, const double* x, double* r, bint unscaled) except +
        void couplingMatvec(int f, int g, const double* x, double* y) except +
        void diagonalBlockMatvec(int f, const double* x, double* y) except +

    # Callback from AMGCL's native lgmres loop into a Python-level preconditioner; see PyLGMRESPrecondT.
    ctypedef int (*PyPrecondApplyFn)(void* ctx, int n, const double* rhs, double* x)

    # AMGCL's native outer Krylov solve; one instance per system size; see LGMRESOuterSolverT.
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
                   bint onUnscaledOperator) except +

    # The p-two-grid (P1) preconditioner of blockamg/ptwogrid.py, in C++; see PTwoGridPreconditionerT.
    cdef cppclass PTwoGridPreconditioner:
        PTwoGridPreconditioner(const char* json_params) except +

        @staticmethod
        void dirichletRowMask(int n, const int* ptr, const int* col, const double* val, char* isDirichlet) except +
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
