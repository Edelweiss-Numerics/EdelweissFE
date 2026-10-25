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
from libc.stddef cimport ptrdiff_t
from libcpp.pair cimport pair
from libcpp.string cimport string
from libcpp.vector cimport vector


def _csrArrays(object A, bint sortIndices=False):
    """``A`` as a CSR matrix (converted if needed; with ``sortIndices``, with the columns of every row in increasing
    order, as the two-level builds read them) and its arrays as C-contiguous int32 row pointers, int32 column indices
    and float64 values -- the form every C++ entry point of this module takes, and the one place their int32 range is
    checked."""
    if not scipy.sparse.isspmatrix_csr(A):
        A = A.tocsr()
    if sortIndices and not A.has_sorted_indices:
        A = A.sorted_indices()
    int32Max = np.iinfo(np.int32).max
    if max(A.shape) > int32Max or A.nnz > int32Max:
        raise OverflowError("a {:}x{:} CSR matrix with {:} entries exceeds the int32 range of the AMGCL wrapper".format(
            A.shape[0], A.shape[1], A.nnz))
    return (
        A,
        np.ascontiguousarray(A.indptr, dtype=np.int32),
        np.ascontiguousarray(A.indices, dtype=np.int32),
        np.ascontiguousarray(A.data, dtype=np.float64),
    )


def _vector(object x, int n, str name):
    """``x`` as a C-contiguous float64 vector of length ``n``. The C++ side reads exactly ``n`` entries without
    bounds checks, so a wrong length (e.g. a vector of the mesh before adaptive refinement) must raise here."""
    cdef np.ndarray[np.float64_t, ndim=1, mode="c"] v = np.ascontiguousarray(x, dtype=np.float64).reshape(-1)
    if v.shape[0] != n:
        raise ValueError("{:} has length {:}, expected {:}".format(name, v.shape[0], n))
    return v


cdef inline const int* _intData(const int[::1] values):
    """The first entry's address, or NULL for an empty array -- bounds checking is off in this module, so an
    empty array must not be indexed."""
    return &values[0] if values.shape[0] > 0 else NULL


cdef inline const double* _constDoubleData(const double[::1] values):
    """See :func:`_intData`."""
    return &values[0] if values.shape[0] > 0 else NULL


cdef inline double* _doubleData(double[::1] values):
    """See :func:`_intData`."""
    return &values[0] if values.shape[0] > 0 else NULL


cdef inline const signed char* _byteData(const signed char[::1] values):
    """See :func:`_intData`."""
    return &values[0] if values.shape[0] > 0 else NULL


cdef inline char* _writableByteData(signed char[::1] values):
    """See :func:`_intData` -- for an output array the C++ side writes."""
    return <char*> &values[0] if values.shape[0] > 0 else NULL


cdef class PyAMGCLSolver:
    cdef AMGCLSolverInterface* solver
    cdef readonly bint isFloat
    """True if this instance runs a mixed-precision hierarchy: builtin<float> for the scalar backend, or
    3x3 blocks set up in double but stored in float (``LinearSolverBlock3Float``) for block size 3."""
    cdef readonly int blockSize
    """1 (scalar, default), 2, or 3 -- the AMGCL backend's block size."""
    cdef readonly int n
    """The size of the matrix of the last :meth:`build`, -1 before the first."""

    def __cinit__(self, dict params=None):
        """
        Initialize with parameters only.

        The AMG smoother key is ``relax``, not ``relaxation``; AMGCL ignores unknown keys with only a
        warning on stderr, so a misspelling here silently runs the default smoother instead of the
        requested one.

        ``backendPrecision``: ``"double"`` (default) or ``"float"``. Selects the AMGCL backend's value
        type -- ``"float"`` halves the memory traffic of the hierarchy build and, more importantly, of
        every :meth:`applyPreconditioner` call (the dominant cost on large coupled solves). Not an
        AMGCL parameter itself, so it is popped out before the rest of ``params`` is forwarded as JSON.
        The near null-space (:meth:`set_nullspace`) always stays double regardless -- AMGCL's own
        ``coarsening::nullspace_params::B`` is hardcoded double, independent of the backend.

        ``backendBlockSize``: ``1`` (default, scalar), ``2``, or ``3``. Selects a block-valued backend
        operating on node-major B x B nodal blocks instead of scalar entries -- shrinks the
        CSR index traffic by ~B² and, with ``backendPrecision == "double"`` (block size 2 or 3), lets
        block-aware smoothers (block-ILU0, block-GS) invert each node's coupling exactly: both run on
        plain ``amgcl::backend::builtin<BlockType>``, the same backend the scalar solver uses, just over
        B x B blocks instead of scalar entries. The matrix still arrives as a plain scalar CSR; the
        wrapper adapts it internally. With ``backendPrecision == "float"``, block size 3
        (``LinearSolverBlock3Float``) sets the hierarchy up in double -- coarsening, Galerkin products
        and every Chebyshev spectral-radius estimate -- and stores only the level matrices the cycle
        reads in float, with double vectors and double accumulation; on *that* backend block-GS is
        unsupported (AMGCL rejects it at construction, since the float-stored level matrices do not
        provide a row iterator) and block-ILU0/ILUk/ILUT/ILUP fall back to AMGCL's generic approximate
        triangular solve instead of the exact one plain ``builtin`` gets. Block size 2 with float raises.
        On a block backend :meth:`set_nullspace` takes the B translations only (the rotations are carried by the
        scalar backend), see there.

        The hierarchy is AMGCL's preconditioner of the parameters under ``"precond"``, e.g.
        {"precond": {"coarsening": {"type": "smoothed_aggregation"}, "relax": {"type": "chebyshev"}}};
        it is only ever applied as a preconditioner (the Krylov solve around it is the composition's LGMRES or
        FGMRES).
        """
        if params is None:
            params = {"precond": {"coarsening": {"type": "smoothed_aggregation"}, "relax": {"type": "chebyshev"}}}
        else:
            params = dict(params)

        backendPrecision = params.pop("backendPrecision", "double")
        if backendPrecision not in ("double", "float"):
            raise ValueError(f"backendPrecision must be 'double' or 'float', got {backendPrecision!r}")
        self.isFloat = backendPrecision == "float"

        blockSize = params.pop("backendBlockSize", 1)
        if blockSize not in (1, 2, 3):
            raise ValueError(f"backendBlockSize must be 1, 2, or 3, got {blockSize!r}")
        self.blockSize = blockSize
        if self.blockSize == 2 and self.isFloat:
            raise NotImplementedError(
                "backendBlockSize 2 combined with backendPrecision='float' is not implemented."
            )

        self.n = -1

        # Convert dict to JSON string for C++
        cdef bytes json_bytes = json.dumps(params).encode("utf-8")
        cdef const char* c_json = json_bytes
        self.solver = makeAMGCLSolver(c_json, self.blockSize, self.isFloat)

    def __dealloc__(self):
        if self.solver != NULL:
            del self.solver

    def set_nullspace(self, object B):
        """
        Supply near null-space vectors for smoothed-aggregation coarsening.

        AMGCL's smoothed aggregation defaults to a single constant near null-space vector, which is
        wrong for an elasticity operator -- there the near null-space is the rigid body modes. AMGCL
        takes these vectors as a raw pointer in its property tree, so they cannot be passed through the
        JSON parameter string like every other option; this is the separate entry point for them.

        B: an (n, cols) array-like; column j is the j-th near null-space vector (float64). Copied into
        the C++ solver, so the array need not be kept alive. Must be called before :meth:`build`.
        Passing an (n, 0) array clears any previously set null-space. Always double, regardless of
        ``backendPrecision`` -- see the class docstring.

        A block backend (``blockSize > 1``) takes the B translations only: an (n, B) array with one diagonal
        B x B block of positive weights per node, e.g. the translations ``D^-1 t`` of an equilibrated block
        ``D A D`` (:func:`~edelweissfe.linsolve.nullspace.translationNullspace`). Its next hierarchy is then
        smoothed aggregation with a tentative prolongation built from them instead of AMGCL's identity blocks
        (``amgcl-block-translations.hpp``), so the parameters must ask for smoothed aggregation. Anything else
        -- the rotations, an entry off the diagonal blocks, a weight that is not positive -- raises
        ``RuntimeError``; the rotations are carried by the scalar backend.
        """
        cdef np.ndarray[np.float64_t, ndim=2, mode="c"] B_arr = np.ascontiguousarray(B, dtype=np.float64)
        cdef int rows = B_arr.shape[0]
        cdef int cols = B_arr.shape[1]
        cdef double[:, ::1] B_view
        if cols == 0:
            self.solver.set_nullspace(<const double*> 0, rows, 0)
            return
        B_view = B_arr
        self.solver.set_nullspace(&B_view[0, 0], rows, cols)

    def build(self, object A):
        """
        Build the AMG hierarchy for A once, for repeated preconditioner application via
        :meth:`applyPreconditioner`.

        A: scipy.sparse.csr_matrix, passed as double; a float backend rounds its values inside the C++
        wrapper (build() runs once per solve, not once per outer Krylov iteration), and a block
        backend (``blockSize > 1``) adapts the scalar CSR to node-major blocks there.
        """
        A, indptr, indices, data = _csrArrays(A)
        self.solver.build(A.shape[0], _intData(indptr), _intData(indices), _constDoubleData(data))
        self.n = A.shape[0]

    def applyPreconditioner(self, object rhs):
        """
        Apply one AMG cycle of the hierarchy built by :meth:`build` to rhs: returns M^-1 rhs.

        rhs: array-like, converted to 1D float64 (C-contiguous). Always double at this boundary --
        any backend-specific narrowing/widening or block reinterpretation happens inside the C++
        wrapper, not here, since this is called once per outer Krylov iteration.
        """
        rhsArray = _vector(rhs, self.n, "rhs")
        cdef np.ndarray[np.float64_t, ndim=1, mode="c"] x = np.zeros(self.n, dtype=np.float64)
        self.solver.applyPreconditioner(self.n, _constDoubleData(rhsArray), _doubleData(x))
        return x

    def report(self):
        """
        Human-readable AMGCL hierarchy report (levels, operator complexity, coarse size) for the most
        recently built hierarchy.

        Raises RuntimeError if nothing has been built yet.
        """
        return self.solver.report().decode("utf-8")


cdef class PyAMGCLEquilibratedSystem:
    """The diagonally scaled, field-split system ``As = D A D`` (``D = diag(scaling)``) of a composed solve
    (:mod:`~edelweissfe.linsolve.composition.systemmatrix`), built in one
    OpenMP-threaded pass over ``A``: the outer Krylov operator (:meth:`matvec`, :meth:`residual`, and
    the operator :meth:`PyAMGCLKrylovSolver.solve` iterates on), the coupling blocks between fields
    (:meth:`couplingMatvec`) and, on demand, a field's diagonal block (:meth:`diagonalBlock`).

    Replaces a chain of serial SciPy operations -- a full copy of ``A`` scaled in place, a threaded
    copy of that for the operator, row and column slices for every coupling and diagonal block --
    measured as a non-trivial share of every solve on a large coupled system. The operator is bitwise
    the former one (see ``EquilibratedSystemT`` in ``amgcl-wrapper.hpp``).
    """
    cdef EquilibratedSystem* system
    cdef readonly int n
    """The system size."""

    def __cinit__(self):
        self.system = new EquilibratedSystem()
        self.n = 0

    def __dealloc__(self):
        if self.system != NULL:
            del self.system

    @staticmethod
    def absoluteDiagonal(object A):
        """``abs(A.diagonal())`` of a scipy.sparse CSR matrix, threaded."""
        A, indptr, indices, data = _csrArrays(A)
        cdef int n = A.shape[0]
        cdef np.ndarray[np.float64_t, ndim=1, mode="c"] out = np.empty(n, dtype=np.float64)
        EquilibratedSystem.absoluteDiagonal(n, _intData(indptr), _intData(indices), _constDoubleData(data),
                                            _doubleData(out))
        return out

    def build(self, object A, object scaling, object fieldBounds, bint unscaled=False):
        """
        A: scipy.sparse csr_matrix (square); scaling: the equilibration, length n; fieldBounds: a
        sequence of ``(start, stop)`` DOF ranges, one per field, disjoint. unscaled: store A itself instead
        of ``As``, for :meth:`matvec`/:meth:`residual` with ``unscaled`` and an outer solve on it; the blocks of
        ``As`` are the same either way.
        """
        if A.shape[0] != A.shape[1]:
            raise ValueError("A must be square, got shape {:}".format(A.shape))
        A, indptr, indices, data = _csrArrays(A)
        cdef int n = A.shape[0]
        scalingArray = _vector(scaling, n, "scaling")
        bounds = np.ascontiguousarray(np.asarray(fieldBounds, dtype=np.int32).reshape(-1))
        if bounds.shape[0] % 2:
            raise ValueError("fieldBounds must be (start, stop) pairs, got {:} numbers".format(bounds.shape[0]))
        self.system.build(n, _intData(indptr), _intData(indices), _constDoubleData(data),
                          _constDoubleData(scalingArray), bounds.shape[0] // 2, _intData(bounds), unscaled)
        self.n = n

    def buildOperator(self, object A):
        """Keep ``A`` itself (scipy.sparse csr_matrix, square, its stored zeros too) as the operator of an outer solve
        on the unscaled system, without fields or blocks: for :meth:`matvec`/:meth:`residual` with ``unscaled`` and the
        outer LGMRES."""
        if A.shape[0] != A.shape[1]:
            raise ValueError("A must be square, got shape {:}".format(A.shape))
        A, indptr, indices, data = _csrArrays(A)
        self.system.buildOperator(A.shape[0], _intData(indptr), _intData(indices), _constDoubleData(data))
        self.n = A.shape[0]

    def diagonalBlock(self, int field):
        """``As[field, field]`` as a new scipy.sparse csr_matrix (sorted columns if ``A``'s were), whichever
        operator :meth:`build` stored."""
        cdef int size = self.system.diagonalBlockSize(field)
        cdef int nnz = self.system.diagonalBlockNnz(field)
        cdef np.ndarray[np.int32_t, ndim=1, mode="c"] indptr = np.empty(size + 1, dtype=np.int32)
        cdef np.ndarray[np.int32_t, ndim=1, mode="c"] indices = np.empty(max(nnz, 1), dtype=np.int32)
        cdef np.ndarray[np.float64_t, ndim=1, mode="c"] data = np.empty(max(nnz, 1), dtype=np.float64)
        cdef int[::1] indptr_ = indptr
        cdef int[::1] indices_ = indices
        cdef double[::1] data_ = data
        self.system.fillDiagonalBlock(field, &indptr_[0], &indices_[0], &data_[0])
        return scipy.sparse.csr_matrix((data[:nnz], indices[:nnz], indptr), shape=(size, size), copy=False)

    def matvec(self, object x, bint unscaled=False):
        """Returns ``As @ x`` -- or with ``unscaled``, ``A @ x`` -- as a new float64 array: the operator
        :meth:`build` stored."""
        xArray = _vector(x, self.n, "x")
        cdef np.ndarray[np.float64_t, ndim=1, mode="c"] y = np.empty(self.n, dtype=np.float64)
        self.system.matvec(_constDoubleData(xArray), _doubleData(y), unscaled)
        return y

    def residual(self, object rhs, object x, bint unscaled=False):
        """Returns ``rhs - As @ x`` -- or with ``unscaled``, ``rhs - A @ x`` -- as a new float64 array."""
        rhsArray, xArray = _vector(rhs, self.n, "rhs"), _vector(x, self.n, "x")
        cdef np.ndarray[np.float64_t, ndim=1, mode="c"] r = np.empty(self.n, dtype=np.float64)
        self.system.residual(_constDoubleData(rhsArray), _constDoubleData(xArray), _doubleData(r), unscaled)
        return r

    def diagonalBlockMatvec(self, int field, object x):
        """Returns ``As[field, field] @ x`` as a new float64 array of this field's size."""
        cdef np.ndarray[np.float64_t, ndim=1, mode="c"] x_arr = np.ascontiguousarray(x, dtype=np.float64)
        if x_arr.shape[0] != self.system.diagonalBlockSize(field):
            raise ValueError("x has the wrong length for field {:}".format(field))
        cdef np.ndarray[np.float64_t, ndim=1, mode="c"] y = np.empty(x_arr.shape[0], dtype=np.float64)
        if x_arr.shape[0] == 0:
            return y
        cdef double[::1] x_ = x_arr
        cdef double[::1] y_ = y
        self.system.diagonalBlockMatvec(field, &x_[0], &y_[0])
        return y

    def couplingMatvec(self, int field, int otherField, object x):
        """Returns ``As[field, otherField] @ x`` (``x`` of the other field's size) as a new float64
        array of this field's size."""
        cdef np.ndarray[np.float64_t, ndim=1, mode="c"] x_arr = np.ascontiguousarray(x, dtype=np.float64)
        if x_arr.shape[0] != self.system.diagonalBlockSize(otherField):
            raise ValueError("x has the wrong length for field {:}".format(otherField))
        cdef np.ndarray[np.float64_t, ndim=1, mode="c"] y = np.empty(
            self.system.diagonalBlockSize(field), dtype=np.float64
        )
        cdef double[::1] x_ = x_arr
        cdef double[::1] y_ = y
        if x_arr.shape[0] == 0 or y.shape[0] == 0:
            y[:] = 0.0
            return y
        self.system.couplingMatvec(field, otherField, &x_[0], &y_[0])
        return y


cdef int _krylovPrecondApplyTrampoline(void* ctx, int n, const double* rhs, double* x) noexcept:
    """The C-callable trampoline bridging amgcl::solver::lgmres's (or fgmres's) per-``apply()`` Precond callback
    back into Python -- passed to :class:`PyAMGCLKrylovSolver` as a bare function pointer (see
    ``amgcl-wrapper.hpp``'s ``PyPrecondApplyFn``/``PyKrylovPrecondT``).

    ``ctx`` is the calling :class:`PyAMGCLKrylovSolver` instance itself, cast to ``void*`` and back --
    Cython's own documented pattern for passing a Python object through an opaque C callback context.
    No manual ``PyObject*`` reference counting is needed: ``ctx`` only ever points at ``self`` from the
    *same*, still-on-stack :meth:`PyAMGCLKrylovSolver.solve` call that constructed it (see there), so
    its lifetime is strictly nested inside this function's own call and never outlives it.

    Declared ``noexcept`` and *not* ``nogil``: every call into this function happens nested inside
    :meth:`PyAMGCLKrylovSolver.solve`'s own C++ call, made from a plain ``def`` method that never
    releases the GIL (matching every other AMGCL wrapper method in this module -- the GIL only blocks
    *other* Python threads from running during the solve; AMGCL's own OpenMP worker threads never touch
    a Python object, so they are unaffected either way). Not releasing the GIL around the C++ call was
    a deliberate choice, not an oversight: doing so would require re-acquiring it here (``with gil:``)
    on every single preconditioner application -- the hottest inner loop this whole class exists to
    speed up -- for a benefit (letting unrelated Python threads run during one linear solve) this
    codebase's actual threading model does not use.

    A Python exception raised by the preconditioner callable cannot propagate through this C
    function-pointer boundary, so it is caught here and stashed on the calling instance instead, and the
    function returns 1 (0 on success). The C++ side then throws, which ends AMGCL's lgmres at once, and
    :meth:`PyAMGCLKrylovSolver.solve` re-raises the stashed exception in place of the C++ one.
    """
    cdef PyAMGCLKrylovSolver self = <PyAMGCLKrylovSolver> ctx
    cdef Py_ssize_t i
    # rhs arrives as a `const double*` (it is AMGCL's own internal Arnoldi-vector storage, never meant
    # to be written through) -- copied element-by-element into a fresh, ordinary (non-const) numpy
    # array rather than wrapped in a zero-copy memoryview, which would need casting away the pointer's
    # constness. n is the system size, so this is one pass over a vector per preconditioner
    # application -- small against the application itself (a block Gauss-Seidel sweep of AMG cycles),
    # which reads every matrix entry at least once.
    cdef np.ndarray[np.float64_t, ndim=1, mode="c"] rhsArr
    cdef double[::1] xView
    try:
        # inside the try: a MemoryError here must be stashed and re-raised like any other failure, not
        # swallowed by this noexcept function, which would leave x stale and the solve "successful"
        rhsArr = np.empty(n, dtype=np.float64)
        for i in range(n):
            rhsArr[i] = rhs[i]
        result = self._precondCallable(rhsArr)
        xView = np.ascontiguousarray(result, dtype=np.float64)
        if xView.shape[0] != n:
            raise ValueError(
                "PyAMGCLKrylovSolver: preconditioner callable returned a vector of length {:} for an "
                "input of length {:}".format(xView.shape[0], n)
            )
        for i in range(n):
            x[i] = xView[i]
    except BaseException as exc:
        self._pendingException = exc
        return 1
    return 0


cdef class PyAMGCLKrylovSolver:
    """AMGCL's own native ``amgcl::solver::lgmres`` (or ``fgmres``, by ``method``) as the outer Krylov solve, in
    place of ``scipy.sparse.linalg.gmres`` -- see ``amgcl-wrapper.hpp``'s ``OuterKrylovSolverT`` for
    the full motivation and design reasoning (this class is a thin Cython shell around it).

    Bridges the preconditioner of a composed Krylov solver
    (:class:`~edelweissfe.linsolve.composition.krylov.KrylovSolver`, a Python callable, e.g. a block Gauss-Seidel
    sweep over per-field AMGCL hierarchies) to AMGCL's native ``Precond`` interface via
    :func:`_krylovPrecondApplyTrampoline`, called once per Arnoldi vector.

    One instance per system size (AMGCL preallocates every scratch vector for it), reused across solves
    with per-call tolerances. Every call starts from an empty augmentation subspace (``always_reset`` is
    forced): carrying the augmentation vectors across calls was measured to buy nothing and to turn a
    struggling solve's subspace into much more expensive or NaN subsequent ones, so it is not offered.
    """
    cdef OuterKrylovSolver* solver
    cdef readonly int n
    """The fixed problem size this instance was constructed for."""
    cdef readonly int lastIterations
    """The iteration count AMGCL's lgmres reported for the most recent solve, -1 before the first."""
    cdef readonly double lastError
    """The relative residual ``||rhs - Op x|| / ||rhs||`` AMGCL's lgmres reached in the most recent solve,
    recomputed at its exit, of the operator it iterated on: the equilibrated system, so not the true
    residual, or with ``onUnscaledOperator`` the unscaled one. lgmres's own convergence test is this
    value below ``tol``. NaN before the first solve."""
    cdef object _precondCallable
    cdef object _pendingException

    def __cinit__(self, int n, str method="lgmres", dict params=None):
        """
        n
            The fixed problem size this instance is built for.
        method
            ``"lgmres"`` or ``"fgmres"`` (flexible GMRES, for a preconditioner that changes between iterations).
        params
            AMGCL's own ``lgmres::params`` field names, forwarded verbatim as JSON (``M``, ``K``,
            ``maxiter``, ``tol``, ``abstol``, ``ns_search``, ``verbose``; ``always_reset`` is forced to true
            and ``pside`` to right preconditioning), or those of ``fgmres::params`` (the same without ``K``,
            ``always_reset`` and ``pside``)
            -- unlike :class:`PyAMGCLSolver`, there is no ``precond``/``solver`` nesting here, since
            this wraps a single ``amgcl::solver::lgmres`` object directly, not ``amgcl::make_solver``.
            ``maxiter``/``tol`` given here only set the *initial* defaults -- :meth:`solve` overrides
            both on every call (see ``amgcl-wrapper.hpp``'s ``OuterKrylovSolverT`` for why).
        """
        if params is None:
            params = {}
        self.n = n
        self.lastIterations = -1
        self.lastError = float("nan")
        self._precondCallable = None
        self._pendingException = None

        cdef bytes json_bytes = json.dumps(params).encode("utf-8")
        cdef const char* c_json = json_bytes
        self.solver = makeOuterKrylovSolver(method.encode("utf-8"), n, c_json)

    def __dealloc__(self):
        if self.solver != NULL:
            del self.solver

    def solve(self, PyAMGCLEquilibratedSystem system, object rhs, object applyPreconditioner, double tol,
              int maxiter, object x0=None, bint onUnscaledOperator=False):
        """
        system
            This solve's equilibrated full coupled operator -- or, with ``onUnscaledOperator``, the
            unscaled one it stored (``build(..., unscaled=True)``).
        rhs
            array-like, converted to 1D float64 (C-contiguous).
        applyPreconditioner
            A Python callable, ``residual (1D float64 array) -> correction (1D float64 array)`` --
            the shape of a composed preconditioner's ``__call__``. Called once per Arnoldi vector via
            :func:`_krylovPrecondApplyTrampoline`; an exception it raises ends the AMGCL call at once and is re-raised
            here.
        tol, maxiter
            This call's relative residual tolerance and maximum total Arnoldi iterations (across every
            internal restart -- AMGCL's own ``maxiter`` semantics differ from scipy's restart x maxiter,
            the ``maxIterations`` of the Krylov component), set on the underlying ``amgcl::solver::lgmres`` object's own
            ``prm`` for this call.
        x0
            Optional initial guess (warm start), e.g. a true-residual continuation of the Krylov component.
            Defaults to zero.

        AMGCL's lgmres returns ``x = 0`` at once for ``||rhs|| < 2 eps`` (about 4.4e-16), whatever the units of
        the system. So the solve runs on ``rhs`` (and ``x0``) scaled by a power of two to a norm in [0.5, 1), and
        its solution is scaled back: exact in floating point, so the iterates are those of the unscaled solve.
        """
        cdef int n = system.n
        if n != self.n:
            raise ValueError(
                "PyAMGCLKrylovSolver: matrix size {:} does not match the size {:} this instance was "
                "constructed for -- construct a fresh instance on any field-structure/size "
                "change.".format(n, self.n)
            )

        cdef np.ndarray[np.float64_t, ndim=1, mode="c"] rhs_arr = _vector(rhs, n, "rhs")

        cdef np.ndarray[np.float64_t, ndim=1, mode="c"] x
        if x0 is None:
            x = np.zeros(n, dtype=np.float64)
        else:
            x = _vector(x0, n, "x0").copy()

        cdef int exponent = np.frexp(np.linalg.norm(rhs_arr))[1]
        rhs_arr = np.ldexp(rhs_arr, -exponent)
        x = np.ldexp(x, -exponent)

        cdef double[::1] rhs_ = rhs_arr
        cdef double[::1] x_ = x

        cdef int iters = 0
        cdef double error = 0.0

        # self is passed through as the callback's opaque context, cast to void* and back in
        # _krylovPrecondApplyTrampoline -- see that function's own docstring for why this needs no
        # manual PyObject* reference counting (self is kept alive by this very call's own stack frame
        # for as long as the cast pointer is in use).
        self._precondCallable = applyPreconditioner
        self._pendingException = None
        cdef void* ctx = <void*> self
        try:
            self.solver.solve(
                system.system[0], &rhs_[0], &x_[0],
                tol, maxiter, _krylovPrecondApplyTrampoline, ctx, iters, error, onUnscaledOperator
            )
        except RuntimeError:
            # the C++ side's report of a failed callback: raise the callback's own exception instead
            if self._pendingException is None:
                raise
        finally:
            self._precondCallable = None

        if self._pendingException is not None:
            exc = self._pendingException
            self._pendingException = None
            raise exc

        self.lastIterations = iters
        self.lastError = error
        return np.ldexp(x, exponent)


ctypedef fused _ArrayEntry:
    int
    double
    ptrdiff_t


cdef np.ndarray _copied(const vector[_ArrayEntry]& values):
    """A C++ vector as a new numpy array -- the one way arrays come back from C++ in this module: a copy, since the
    C++ object may change it or go."""
    if values.size() == 0:
        if _ArrayEntry is int:
            return np.empty(0, dtype=np.int32)
        elif _ArrayEntry is double:
            return np.empty(0, dtype=np.float64)
        else:
            return np.empty(0, dtype=np.intp)
    return np.asarray(<_ArrayEntry[:values.size()]> <_ArrayEntry*> values.data()).copy()


cdef object _scipyCsr(const ScalarCsr& csr, int nCols):
    """A ScalarCsr of the C++ side as a scipy.sparse csr_matrix of nCols columns."""
    return scipy.sparse.csr_matrix(
        (_copied(csr.val), _copied(csr.col), _copied(csr.ptr)), shape=(csr.ptr.size() - 1, nCols)
    )


cdef class _PyAMGCLTwoLevel:
    """What the two-level preconditioners share (``TwoLevelPreconditioner`` in ``amgcl-wrapper.hpp``): the cycle, the
    fine spectral radius it was built with, the timings and the report. A derived class creates the C++ object and
    points :attr:`twoLevel` at it; this class owns it (the C++ base class has a virtual destructor)."""
    cdef TwoLevelPreconditioner* twoLevel

    def __dealloc__(self):
        if self.twoLevel != NULL:
            del self.twoLevel

    def applyPreconditioner(self, object r):
        """One two-level cycle: returns M^-1 r as a new float64 array (Dirichlet rows solved exactly)."""
        rArray = np.ascontiguousarray(r, dtype=np.float64).reshape(-1)
        cdef np.ndarray[np.float64_t, ndim=1, mode="c"] x = np.empty(rArray.shape[0], dtype=np.float64)
        # the C++ apply checks the length against the built size
        self.twoLevel.apply(rArray.shape[0], _constDoubleData(rArray), _doubleData(x))
        return x

    @property
    def fineSpectralRadius(self):
        """The fine operator's spectral radius the Chebyshev smoother was built with -- estimated, or
        the one handed to ``build``."""
        return self.twoLevel.fineSpectralRadius()

    @property
    def fineDominantVector(self):
        """The power iteration's last iterate, the estimate of the fine operator's dominant
        eigenvector -- hand it back to the next ``build`` with :attr:`fineSpectralRadius`. Empty
        if the radius was used as supplied, without a re-check."""
        return _copied(self.twoLevel.fineDominantVector())

    @property
    def fineSpectralRadiusSource(self):
        """How the last build arrived at its fine spectral radius: ``"estimated"`` in full,
        ``"supplied"`` and kept, ``"raised"`` by the re-check, or ``"re-estimated"`` in full because
        the re-check found the operator stiffened beyond the smoother's margin."""
        return self.twoLevel.fineSpectralRadiusSource().decode("utf-8")

    @property
    def nCoarse(self):
        """The number of coarse DOFs, the rows of the Galerkin coarse operator (0 without a coarse level)."""
        return self.twoLevel.nCoarse()

    def timings(self):
        """Wall time [s] of the stages of the last build (``"setup: <stage>"``) and the cumulative wall time of the
        apply (fine smoothing incl. the residual, coarse correction incl. restriction and prolongation)."""
        cdef pair[string, double] stage
        result = {}
        for stage in self.twoLevel.setupSeconds_:
            result["setup: " + stage.first.decode("utf-8")] = stage.second
        result["apply: fine"] = self.twoLevel.applyFineSeconds_
        result["apply: coarse"] = self.twoLevel.applyCoarseSeconds_
        return result

    def report(self):
        """The report of the coarse level and its AMGCL hierarchy."""
        return self.twoLevel.report().decode("utf-8")

    def prolongationMatrix(self):
        """The prolongation P of the last build as :meth:`applyPreconditioner` uses it (stored in the precision of the
        fine level, widened to double): a scipy.sparse csr_matrix of n rows and :attr:`nCoarse` columns, for
        inspection."""
        return _scipyCsr(self.twoLevel.prolongationCsr(), self.twoLevel.nCoarse())

    def coarseOperator(self):
        """The Galerkin coarse operator P^T A P of the last build as its AMGCL hierarchy holds it: a scipy.sparse
        csr_matrix, for inspection."""
        return _scipyCsr(self.twoLevel.coarseOperatorCsr(), self.twoLevel.nCoarse())


cdef class PyAMGCLPTwoGrid(_PyAMGCLTwoLevel):
    """The p-two-grid (P1) preconditioner of :mod:`edelweissfe.linsolve.composition.ptwogrid`, built and
    applied entirely in C++: Dirichlet detection, the restricted prolongation ``P``, the Galerkin
    coarse operator ``P^T A P``, the coarse AMGCL hierarchy, the fine Chebyshev smoother and the
    whole two-level cycle, all OpenMP-threaded -- no Python runs inside :meth:`applyPreconditioner`.
    See ``PTwoGridPreconditionerT`` in ``amgcl-wrapper.hpp`` for the Dirichlet scheme and the
    fine spectral-radius estimate.
    """
    cdef PTwoGridPreconditioner* preconditioner

    def __cinit__(self, dict params):
        """
        params: {"amgcl": {<the field's AMGCL tree: relax (Chebyshev), npre, npost>}, "backendBlockSize": 1,
        "backendPrecision": "double", "coarseHierarchyPrecision": "double", "coarse": {<the coarse level's AMGCL
        tree>}}, the spelling of the rigid-body coarse level too (``TwoLevelCycleParameters`` in
        ``amgcl-wrapper.hpp``). Validated here, before any build.
        """
        cdef bytes json_bytes = json.dumps(params).encode("utf-8")
        cdef const char* c_json = json_bytes
        self.preconditioner = new PTwoGridPreconditioner(c_json)
        self.twoLevel = self.preconditioner

    @staticmethod
    def dirichletRows(object A):
        """The rows of a scipy.sparse CSR matrix this preconditioner treats as Dirichlet rows: a nonzero
        diagonal and no nonzero off-diagonal entry (explicitly stored zeros, which the Dirichlet
        elimination keeps, do not count) -- the identity slave rows of the multi-point-constraint
        condensation included. Returns a bool array."""
        A, indptr, indices, data = _csrArrays(A)
        mask = np.zeros(A.shape[0], dtype=np.int8)
        PTwoGridPreconditioner.dirichletRowMask(A.shape[0], _intData(indptr), _intData(indices), _constDoubleData(data),
                                                _writableByteData(mask))
        return mask.astype(bool)

    def build(self, object A, object isCorner, object edgeEndpoints, object nullspace=None,
              double spectralRadius=0.0, object dominantVector=None):
        """
        Build the preconditioner for A (scipy.sparse csr_matrix, the field's equilibrated diagonal
        block with its Dirichlet identity rows).

        isCorner: (nNodes,) bool/int8; edgeEndpoints: (nNodes, 2) int, read for midside nodes only.
        nullspace: (nFreeCoarseDofs, k) float64 coarse near null-space, or None.
        spectralRadius: > 0 to start from this fine spectral radius instead of estimating it in full.
        dominantVector: with spectralRadius, the previous build's :attr:`fineDominantVector`; the
        radius is then re-checked by a few power iterations from it (``fineRecheckIters``) and
        raised, or re-estimated in full, if the operator has stiffened.
        """
        A, indptr, indices, data = _csrArrays(A, sortIndices=True)
        corner = np.ascontiguousarray(isCorner, dtype=np.int8)
        endpoints = np.ascontiguousarray(np.asarray(edgeEndpoints, dtype=np.int32).reshape(-1))
        cdef int nNodes = corner.shape[0]
        if endpoints.shape[0] != 2 * nNodes or (nNodes > 0 and np.ndim(edgeEndpoints) != 2):
            raise ValueError("edgeEndpoints must have shape (nNodes, 2)")
        B = np.zeros((0, 0)) if nullspace is None else np.ascontiguousarray(nullspace, dtype=np.float64)
        if B.ndim != 2:
            raise ValueError("the coarse near null-space must be a 2D array")
        dominant = np.zeros(0) if dominantVector is None else np.ascontiguousarray(dominantVector, dtype=np.float64)
        self.preconditioner.build(A.shape[0], _intData(indptr), _intData(indices), _constDoubleData(data),
                                  nNodes, _byteData(corner), _intData(endpoints),
                                  _constDoubleData(B.reshape(-1)), B.shape[0], B.shape[1],
                                  spectralRadius, _constDoubleData(dominant), dominant.shape[0])


cdef class PyAMGCLRigidBodyCoarseLevel(_PyAMGCLTwoLevel):
    """The rigid-body coarse level preconditioner of :mod:`edelweissfe.linsolve.composition.rigidbodycoarselevel`, built
    and applied entirely in C++ (``RigidBodyCoarseLevelPreconditionerT`` in ``amgcl-wrapper.hpp``, its steps in
    ``amgcl-rigid-body-coarse-level.hpp``) on the near null-space handed to :meth:`build`: the aggregates, the tentative
    prolongation, the smoothed prolongation, the Galerkin coarse operator, the AMGCL hierarchy below it, the fine
    Chebyshev smoother and the cycle, all OpenMP-threaded -- no Python runs inside :meth:`applyPreconditioner`.
    """
    cdef RigidBodyCoarseLevelPreconditioner* preconditioner

    def __cinit__(self, dict params):
        """
        params: {"backendBlockSize": 3, "backendPrecision": "float", "prolongation": "smoothed",
        "coarseHierarchyPrecision": "double", "amgcl": {<the field's AMGCL tree>}}, see
        ``RigidBodyCoarseLevelPreconditioner`` in ``amgcl-wrapper.hpp``. Validated here, before any build.
        """
        cdef bytes json_bytes = json.dumps(params).encode("utf-8")
        cdef const char* c_json = json_bytes
        self.preconditioner = makeRigidBodyCoarseLevelPreconditioner(c_json)
        self.twoLevel = self.preconditioner

    def build(self, object A, object nullspace, double spectralRadius=0.0, object dominantVector=None):
        """
        Build the preconditioner for A (scipy.sparse csr_matrix, the field's equilibrated diagonal block D A D with
        its Dirichlet identity rows, node-major with ``backendBlockSize`` DOFs per node).

        nullspace: (n, M) float64, the near null-space of A whose vectors span the coarse space of every aggregate,
        already divided by the equilibration D: M = ``backendBlockSize`` (the translations) or ``backendBlockSize *
        (backendBlockSize + 1) / 2`` (all rigid-body modes).
        spectralRadius: > 0 to start from this fine spectral radius instead of estimating it in full.
        dominantVector: with spectralRadius, the previous build's :attr:`fineDominantVector`, to re-check it from.
        """
        A, indptr, indices, data = _csrArrays(A, sortIndices=True)
        B = np.ascontiguousarray(nullspace, dtype=np.float64)
        if B.ndim != 2:
            raise ValueError("the near null-space must be an (n, M) array, got the shape {:}".format(B.shape))
        dominant = np.zeros(0) if dominantVector is None else np.ascontiguousarray(dominantVector, dtype=np.float64)
        # the C++ build checks the shape of the near null-space against the block
        self.preconditioner.build(A.shape[0], _intData(indptr), _intData(indices), _constDoubleData(data),
                                  _constDoubleData(B.reshape(-1)), B.shape[0], B.shape[1],
                                  spectralRadius, _constDoubleData(dominant), dominant.shape[0])

    @property
    def nAggregates(self):
        """The number of aggregates of the last build."""
        return self.preconditioner.nAggregates()

    @property
    def modesPerAggregate(self):
        """The coarse DOFs per aggregate of the last build, M, one per column of its near null-space: 6 in 3D and 3
        in 2D for the rigid-body modes."""
        return self.preconditioner.modesPerAggregate()

    @property
    def droppedModes(self):
        """The number of modes the last build dropped as linearly dependent (e.g. the rotations of a single-node
        aggregate)."""
        return self.preconditioner.droppedModes()

    @property
    def prolongationSmoothingWeight(self):
        """The damping weight omega of the prolongation smoothing of the last build (0 for a tentative one)."""
        return self.preconditioner.prolongationSmoothingWeight()

    @property
    def aggregates(self):
        """The aggregate of every node of the last build, ``< 0`` for a node in none."""
        return _copied(self.preconditioner.aggregateOfNode())

    def coarseNullspace(self):
        """The near null-space of the AMGCL hierarchy below, the coefficients of the near null-space in the coarse
        basis (per aggregate the R factor of its QR factorization): (:attr:`nCoarse`, M) float64."""
        return _copied(self.preconditioner.coarseNullspace()).reshape(
            self.preconditioner.nCoarse(), self.preconditioner.coarseNullspaceColumns()
        )
