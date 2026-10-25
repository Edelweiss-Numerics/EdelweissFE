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

from libc.stddef cimport ptrdiff_t
from libcpp.pair cimport pair
from libcpp.string cimport string
from libcpp.vector cimport vector


cdef extern from "amgcl-wrapper.hpp":
    # The interface every AMGCL solver backend implements (scalar or B x B blocks, double or float):
    # the matrix values and rhs/x are always double at this boundary, each backend converts
    # internally -- see AMGCLSolverInterface. On a block backend set_nullspace() takes the translations
    # only, one diagonal block per node (see amgcl-block-translations.hpp).
    cdef cppclass AMGCLSolverInterface:
        void set_nullspace(const double* B, int rows, int cols) except +
        void build(int n, const int* ptr, const int* col, const double* val) except +
        void applyPreconditioner(int n, const double* rhs, double* x) except +
        string report() except +

    # The backend for a block size (1, 2 or 3) and value type (float: builtin<float> for scalars,
    # 3x3 blocks set up in double and stored in float), owned by the caller.
    AMGCLSolverInterface* makeAMGCLSolver(const char* json_params, int blockSize, bint isFloat) except +

    # The diagonally scaled, field-split system As = D A D of a composed solve: the outer Krylov operator, the
    # coupling blocks between fields and, on demand, the diagonal blocks -- built in one threaded
    # pass, see EquilibratedSystemT's own comment.
    cdef cppclass EquilibratedSystem:
        EquilibratedSystem() except +

        @staticmethod
        void absoluteDiagonal(int n, const int* ptr, const int* col, const double* val, double* out) except +
        void build(int n, const int* ptr, const int* col, const double* val, const double* scaling,
                   int nFields, const int* fieldBounds, bint unscaled) except +
        void buildOperator(int n, const int* ptr, const int* col, const double* val) except +
        int diagonalBlockSize(int f) except +
        int diagonalBlockNnz(int f) except +
        void fillDiagonalBlock(int f, int* ptr, int* col, double* val) except +
        void matvec(const double* x, double* y, bint unscaled) except +
        void residual(const double* rhs, const double* x, double* r, bint unscaled) except +
        void couplingMatvec(int f, int g, const double* x, double* y) except +
        void diagonalBlockMatvec(int f, const double* x, double* y) except +

    # Callback signature bridging AMGCL's native lgmres Krylov loop back into a Python-level
    # preconditioner -- see amgcl-wrapper.hpp's own comment on PyPrecondApplyFn/
    # PyKrylovPrecondT for why this is a bare function pointer + opaque context rather than a
    # Cython-overridden C++ virtual method.
    ctypedef int (*PyPrecondApplyFn)(void* ctx, int n, const double* rhs, double* x)

    # AMGCL's own native outer Krylov solve (lgmres or fgmres), in place of scipy.sparse.linalg.gmres; one instance
    # per system size, no subspace carried across calls -- see OuterKrylovSolverT.
    cdef cppclass OuterKrylovSolver:
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

    OuterKrylovSolver* makeOuterKrylovSolver(const string& method, int n, const char* json_params) except +

    # The arrays of a scalar CSR matrix, an operator of a two-level cycle as Python inspects it.
    cdef cppclass ScalarCsr:
        vector[int] ptr
        vector[int] col
        vector[double] val

    # The two-level cycle the p-two-grid and the rigid-body coarse level share: the cycle itself, the fine spectral
    # radius, P and the coarse operator, the timings and the report -- see TwoLevelPreconditioner's own comment. Arrays
    # come back as C++ vectors, which Cython copies (_copied in amgcl.pyx).
    cdef cppclass TwoLevelPreconditioner:
        void apply(int n, const double* r, double* x) except +
        double fineSpectralRadius()
        string fineSpectralRadiusSource()
        const vector[double]& fineDominantVector()
        int nCoarse()
        ScalarCsr prolongationCsr() except +
        ScalarCsr coarseOperatorCsr() except +
        string report() except +
        vector[pair[string, double]] setupSeconds_
        double applyFineSeconds_
        double applyCoarseSeconds_

    # The p-two-grid (P1) preconditioner of edelweissfe/linsolve/composition/ptwogrid.py, built and
    # applied entirely in C++ (OpenMP) -- see PTwoGridPreconditionerT's own comment.
    cdef cppclass PTwoGridPreconditioner(TwoLevelPreconditioner):
        PTwoGridPreconditioner(const char* json_params) except +

        @staticmethod
        void dirichletRowMask(int n, const int* ptr, const int* col, const double* val, char* isDirichlet) except +
        void build(int n, const int* ptr, const int* col, const double* val,
                   int nNodes, const signed char* isCorner, const int* edgeEndpoints,
                   const double* nullspace, int nullspaceRows, int nullspaceCols,
                   double spectralRadius, const double* dominantVector, int dominantVectorSize) except +

    # The rigid-body coarse level preconditioner of edelweissfe/linsolve/composition/rigidbodycoarselevel.py, built and
    # applied entirely in C++ (OpenMP) on the near null-space Python hands it -- see
    # RigidBodyCoarseLevelPreconditioner's own comment, and amgcl-rigid-body-coarse-level.hpp for its steps.
    cdef cppclass RigidBodyCoarseLevelPreconditioner(TwoLevelPreconditioner):
        void build(int n, const int* ptr, const int* col, const double* val,
                   const double* nullspace, int nullspaceRows, int nullspaceCols,
                   double spectralRadius, const double* dominantVector, int dominantVectorSize) except +
        int blockSize()
        int modesPerAggregate()
        int nAggregates()
        int droppedModes()
        double prolongationSmoothingWeight()
        int coarseNullspaceColumns()
        const vector[ptrdiff_t]& aggregateOfNode()
        const vector[double]& coarseNullspace()

    # The rigid-body coarse level preconditioner for the field's backendBlockSize (2 or 3), owned by the caller.
    RigidBodyCoarseLevelPreconditioner* makeRigidBodyCoarseLevelPreconditioner(const char* json_params) except +
