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


class AliasedCSRMatrix(csr_matrix):
    """
    The csr_matrix returned by :meth:`CSRGenerator.updateInPlace`.

    Its ``data``/``indices``/``indptr`` buffers are owned by the generator's C++ core,
    which rewrites ``data`` in place on every subsequent ``updateInPlace`` call via a
    gather/scatter map fixed once at construction time. Structurally mutating this
    matrix (``eliminate_zeros()``, ``prune()``, ``sum_duplicates()``, ``resize()``, or
    reassigning ``.data``/``.indices``/``.indptr`` to arrays of a different length)
    would silently desynchronize that fixed map from the (now compacted/reshaped)
    matrix -- every subsequent update would then write numerically correct values into
    the wrong ``(row, col)`` slots, with no error or warning. See GitHub issue #72.

    This subclass turns that into a loud failure instead of silent corruption. If you
    need to reduce/reshape the pattern for a one-off use, copy the matrix first --
    ``.copy()`` (here and via :meth:`CSRGenerator.updateCSR`) always returns a plain,
    unrestricted ``csr_matrix``.

    The guard only applies once :class:`CSRGenerator` marks construction complete
    (``_locked = True``): scipy's own ``csr_matrix.__init__`` calls ``self.prune()``
    internally as part of its format check, which must go through unhindered.
    """

    _locked = False

    _MUTATION_ERROR = (
        "Refusing to call {:}() on a CSRGenerator.updateInPlace() matrix: this is the "
        "generator's own aliased, reused buffer, and structurally mutating it would "
        "silently corrupt every subsequent update (see GitHub issue #72). Call "
        ".copy() first if you need an independently mutable snapshot."
    )

    def eliminate_zeros(self):
        if self._locked:
            raise RuntimeError(self._MUTATION_ERROR.format("eliminate_zeros"))
        super().eliminate_zeros()

    def prune(self):
        if self._locked:
            raise RuntimeError(self._MUTATION_ERROR.format("prune"))
        super().prune()

    def sum_duplicates(self):
        if self._locked:
            raise RuntimeError(self._MUTATION_ERROR.format("sum_duplicates"))
        super().sum_duplicates()

    def resize(self, *shape):
        if self._locked:
            raise RuntimeError(self._MUTATION_ERROR.format("resize"))
        super().resize(*shape)

    def __setattr__(self, name, value):
        if self._locked and name in ("data", "indices", "indptr"):
            existing = getattr(self, name, None)
            if existing is not None and len(value) != len(existing):
                raise RuntimeError(
                    "Refusing to reassign '{:}' to an array of a different length on a "
                    "CSRGenerator.updateInPlace() matrix (see AliasedCSRMatrix docstring, "
                    "GitHub issue #72). Call .copy() first if you need an independently "
                    "mutable snapshot.".format(name)
                )
        super().__setattr__(name, value)

    def copy(self):
        # Always hand back a plain, unrestricted csr_matrix: once copied, the arrays
        # are independent of the generator's buffer and safe to mutate freely.
        c = super().copy()
        return csr_matrix((c.data, c.indices, c.indptr), shape=c.shape)


cdef extern from "_csrcore.h":
    cdef cppclass CSRCore nogil:
        CSRCore(const int* I, const int* J, long n_pairs, int n_dof, bint patternOnly) except +

        vector[int] indptr
        vector[int] indices
        int nnz
        int nDof

        void update(const double* V_data, double* csr_data) nogil
        void releaseGatherMap()
        bint gatherMapReleased
        long memoryBytes()

    cdef cppclass CSRDirectAssembler nogil:
        CSRDirectAssembler(const int* indptr, const int* indices, int nnz, int nDof,
                           const int* I, const int* J, long nPairs, int nThreads) except +

        void assembleFromVIJ(const double* V, const int* I, double* csr_data) nogil
        void registerEntities(const long* mapStarts, const int* nDofs, int nEntities, const int* I)
        void beginAssembly() nogil
        void scatterBlock(int tid, int entity, const double* block) nogil
        void reduce(double* csr_data) nogil
        void setNumBuffers(int n) except +
        long memoryBytes()
        int nBuffers

cdef class CSRGenerator:
    """
    CSRGenerator class to create and manage a CSR matrix from COO format.

    This class utilizes a C++ core for efficient conversion and updating of the CSR matrix.

    Parameters
    ----------
    systemMatrix : object
        An object containing COO format data with attributes I, J, and nDof.
    patternOnly : bool
        Build only the CSR pattern (``indptr``/``indices``), not the gather map. Halves the sort
        array and, more importantly, removes the 32-bit limit on the pair count -- ``gather_sources``
        and ``assembly_ptr`` are int32 indices into the COO list, so a full build refuses more than
        INT32_MAX pairs. :meth:`updateInPlace` and :meth:`updateCSR` then raise, exactly as after
        :meth:`releaseGatherMap`. Use it when the pattern is being borrowed by a
        :class:`DirectCSRAssembler` and nothing will gather.
    """

    cdef CSRCore* core
    cdef public object csrMatrix
    cdef double[:] data_view
    cdef long nCooPairs  # Kept as long (int64)

    def __dealloc__(self):
        if self.core != NULL:
            del self.core

    def __init__(self, systemMatrix, bint patternOnly=False):
        # Ensure int32 dtype regardless of the source array's dtype.
        # dofmanager.py already produces np.intc arrays, but we guard here
        # in case CSRGenerator is called from outside the standard path.
        cdef int[::1] I = np.asarray(systemMatrix.I, dtype=np.intc)  # noqa
        cdef int[::1] J = np.asarray(systemMatrix.J, dtype=np.intc)

        self.nCooPairs = len(I)  # Length is still 64-bit capable

        cdef int nDof = int(systemMatrix.nDof)

        # 1. Run C++ Core
        with nogil:
            self.core = new CSRCore(&I[0], &J[0], self.nCooPairs, nDof, patternOnly)

        cdef int* ptr_indptr = self.core.indptr.data()
        cdef int* ptr_indices = self.core.indices.data()

        cdef int nnz = self.core.nnz

        cdef int[::1] view_indptr = <int[:nDof+1]> ptr_indptr
        cdef int[::1] view_indices = <int[:nnz]> ptr_indices

        cdef np.ndarray nd_indptr = np.asarray(view_indptr)
        cdef np.ndarray nd_indices = np.asarray(view_indices)

        cdef np.ndarray[double, ndim=1] data = np.zeros(nnz, dtype=np.double)
        self.csrMatrix = AliasedCSRMatrix((data, nd_indices, nd_indptr), shape=(nDof, nDof))

        # Keep this CSRGenerator object alive as long as csrMatrix is referenced.
        # _parent is a SciPy-internal attribute — it exists in all supported
        # versions but is undocumented; callers should not hold csrMatrix
        # independently of its CSRGenerator.
        self.csrMatrix._parent = self

        self.data_view = self.csrMatrix.data

        # Construction (including scipy's own internal format check/prune) is done;
        # from here on, structural mutation of this specific matrix is unsafe -- see
        # AliasedCSRMatrix's docstring.
        self.csrMatrix._locked = True

    @property
    def memoryBytes(self):
        """Bytes held by the CSR pattern, the gather map and the CSR data array.

        The counterpart to :attr:`DirectCSRAssembler.memoryBytes`, on the same basis -- what the
        object owns. The dominant term is the gather map's ``gather_sources``, one int32 per COO
        pair, which is what the direct path replaces with a uint16 offset per pair.

        Neither figure counts the VIJ value array or the ``I``/``J`` index arrays: those belong to
        the DofManager, and ``I``/``J`` are needed by both paths.
        """
        return self.core.memoryBytes() + 8 * <long> self.core.nnz

    @property
    def nnz(self):
        """Number of stored entries in the CSR pattern."""
        return self.core.nnz

    @property
    def gatherMapReleased(self):
        """True once :meth:`releaseGatherMap` has been called and this generator can no longer gather."""
        return bool(self.core.gatherMapReleased)

    def releaseGatherMap(self):
        """Give back the gather map, keeping the CSR pattern -- and give up the ability to gather.

        ``gather_sources`` is one int32 per COO pair, so the map is by far the largest thing this
        object holds (6.69 GiB at 43,350 DOF against 0.80 GiB for the pattern and the data array).
        A solver assembling directly into CSR borrows only the pattern, so holding the map would
        cancel most of what the direct path saves.

        After this call :meth:`updateInPlace` and :meth:`updateCSR` raise. That is deliberate: the
        alternative is a ``nogil`` loop reading freed vectors, and a matrix that is quietly wrong is
        worse than one that is loudly unavailable. :attr:`csrMatrix` and the pattern stay valid and
        usable, and :attr:`memoryBytes` drops accordingly.
        """
        self.core.releaseGatherMap()

    def updateInPlace(self, double[:] V):
        """
        Update the values of the CSR matrix in-place based on the input vector V.

        Returns the internal CSR matrix directly (no copy). The caller must not
        retain the returned object across subsequent calls to ``updateInPlace``
        or ``updateCSR``, as the underlying data will be overwritten. It also must not
        be structurally mutated (``eliminate_zeros()``, ``prune()``, ``resize()``, a
        differently-shaped ``.data``/``.indices``/``.indptr`` reassignment, ...) --
        the returned :class:`AliasedCSRMatrix` raises rather than allowing this
        silently; see its docstring and GitHub issue #72.

        Parameters
        ----------
        V : double[:]
            Input vector used to update the CSR matrix values.

        Returns
        -------
        AliasedCSRMatrix
            A live view of the internal CSR matrix (not a copy).
        """

        if self.core.gatherMapReleased:
            raise RuntimeError(
                "CSRGenerator.updateInPlace on a generator with no gather map: it was either built "
                "with patternOnly=True or gave the map back via releaseGatherMap(), and can only "
                "supply the CSR pattern now. Assemble through the DirectCSRAssembler that borrowed "
                "the pattern, or build a generator that keeps its map."
            )

        cdef double* d_ptr = &self.data_view[0]
        cdef double* v_ptr = &V[0]

        with nogil:
            self.core.update(v_ptr, d_ptr)

        return self.csrMatrix

    def updateCSR(self, double[:] V):
        """
        Update the values of the CSR matrix and return an independent copy.

        Use ``updateInPlace`` instead when the caller does not need to retain
        the matrix across subsequent assembly steps, to avoid the allocation
        cost of copying.

        Parameters
        ----------
        V : double[:]
            Input vector used to update the CSR matrix values.

        Returns
        -------
        csr_matrix
            An independent copy of the updated CSR matrix.
        """

        self.updateInPlace(V)
        return self.csrMatrix.copy()


cdef class DirectCSRAssembler:
    """Scatters entity blocks straight into a CSR data array, with no VIJ staging array.

    The alternative to :class:`CSRGenerator`'s stage-then-gather. It borrows an existing pattern
    rather than deriving its own, so there is exactly one definition of the CSR pattern and the two
    paths cannot drift apart -- and :meth:`assembleFromVIJ` exists so the addressing can be checked
    against ``CSRGenerator.updateCSR`` on real models rather than argued about.

    Reduction defaults to one private CSR copy per thread -- no synchronisation and a fixed summation
    order, so results are bit-reproducible, at ``nThreads * nnz * 8`` bytes. :meth:`setNumBuffers`
    trades that memory for atomics; see the header's design note for why the earlier "atomics are
    1.57-2.13x slower" figure does not settle the question.

    Parameters
    ----------
    generator
        A :class:`CSRGenerator` whose pattern is borrowed. It must outlive this object.
    systemMatrix
        The VIJ system matrix supplying the ``I``/``J`` index arrays the map is built from.
    numThreads
        Threads used for the map build, the scatter and the reduction.
    """

    cdef CSRDirectAssembler* asm_
    cdef object _generator          # kept alive: the pattern is borrowed, not owned
    cdef object _I
    cdef public object csrMatrix
    cdef double[:] data_view
    cdef long nCooPairs

    def __dealloc__(self):
        if self.asm_ != NULL:
            del self.asm_

    def __init__(self, generator, systemMatrix, int numThreads=1):
        cdef int[::1] I = np.asarray(systemMatrix.I, dtype=np.intc)  # noqa
        cdef int[::1] J = np.asarray(systemMatrix.J, dtype=np.intc)
        self._generator = generator
        self._I = np.asarray(I)
        self.nCooPairs = len(I)

        self.csrMatrix = generator.csrMatrix
        self.data_view = self.csrMatrix.data

        cdef int[::1] indptr = np.asarray(self.csrMatrix.indptr, dtype=np.intc)
        cdef int[::1] indices = np.asarray(self.csrMatrix.indices, dtype=np.intc)
        cdef int nnz = indices.shape[0]
        cdef int nDof = int(systemMatrix.nDof)

        with nogil:
            self.asm_ = new CSRDirectAssembler(&indptr[0], &indices[0], nnz, nDof,
                                               &I[0], &J[0], self.nCooPairs, numThreads)

    @property
    def memoryBytes(self):
        """Bytes held by the offset map plus the private buffers."""
        return self.asm_.memoryBytes()

    @property
    def numBuffers(self):
        """How many private CSR copies are currently held. Equal to ``numThreads`` unless changed."""
        return self.asm_.nBuffers

    def setNumBuffers(self, int n):
        """Set how many private CSR copies to keep, reallocating them.

        ``n == numThreads`` is the default: one copy per thread, no synchronisation, fixed summation
        order and therefore bit-reproducible results. Any smaller ``n`` makes threads share a copy
        and synchronise the scatter with atomics, saving ``(numThreads - n) * nnz * 8`` bytes at the
        cost of reproducibility -- ``n == 1`` is the fully atomic case. Values are clamped to
        ``[1, numThreads]``.

        Not to be called during an assembly: it releases and reallocates the buffers.
        """
        self.asm_.setNumBuffers(n)

    def assembleFromVIJ(self, double[::1] V):
        """Scatter a VIJ value array into CSR through the map, and return the CSR matrix.

        Equivalence path for validation and measurement. The fused assembly hands each entity's
        block in directly and never builds ``V``.
        """
        cdef int[::1] I = self._I
        cdef double[:] out = self.data_view
        with nogil:
            self.asm_.assembleFromVIJ(&V[0], &I[0], &out[0])
        return self.csrMatrix

    def registerEntities(self, long[::1] mapStarts, int[::1] nDofs):
        """Give each entity its slice of the offset map, once per connectivity change.

        ``mapStarts[e]`` is entity e's offset into the VIJ ordering -- the same value the DofManager
        already records in ``idcsOfHigherOrderEntitiesInVIJ`` -- and ``nDofs[e]`` its local DOF count.
        """
        cdef int[::1] I = self._I
        self.asm_.registerEntities(&mapStarts[0], &nDofs[0], mapStarts.shape[0], &I[0])

    @property
    def corePointer(self):
        """Address of the underlying ``CSRDirectAssembler``, for a threaded caller in another extension.

        EdelweissMeshfree's particle kernel calls ``scatterBlock`` from inside its ``prange``, which
        cannot go through Python. It re-declares the C++ class (from ``edelweissfe.numerics.get_include()``)
        and casts this address back, so the scatter -- including the column-major traversal and its
        measured loop order -- exists in exactly one place.

        The assembler must outlive any caller holding this.
        """
        return <size_t> self.asm_

    def beginAssembly(self):
        """Zero the private buffers. Call once before the entity loop."""
        with nogil:
            self.asm_.beginAssembly()

    def scatterBlock(self, int tid, int entity, double[::1] block):
        """Scatter one entity's dense block from a scratch buffer into thread ``tid``'s private CSR.

        Race-free by construction. Intended to be called from inside a threaded entity loop; this
        Python-level binding exists for testing and for the constraint contributions, which are
        assembled sequentially and account for under 2% of runtime.
        """
        with nogil:
            self.asm_.scatterBlock(tid, entity, &block[0])

    def reduce(self):
        """Sum the private buffers into the CSR matrix and return it."""
        cdef double[:] out = self.data_view
        with nogil:
            self.asm_.reduce(&out[0])
        return self.csrMatrix
