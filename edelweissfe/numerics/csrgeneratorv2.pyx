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
from cpython.ref cimport Py_INCREF
from libc.stdint cimport int32_t, int64_t, uint8_t
from libc.string cimport memcpy
from libcpp.vector cimport vector

np.import_array()


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
        # A plain csr_matrix: the copied arrays are independent of the generator's buffer.
        c = super().copy()
        return csr_matrix((c.data, c.indices, c.indptr), shape=c.shape)


cdef extern from "_csrcore.h":
    cdef cppclass SystemMatrixEntities nogil:
        int64_t nEntities
        const int64_t* dofStart
        const int32_t* dofs
        const uint8_t* couplesAllItsDofs
        const int64_t* localEntryStart
        const int32_t* localRows
        const int32_t* localColumns

    cdef cppclass CSRPattern nogil:
        vector[int] indptr
        vector[int] indices
        int nnz
        int nDof

        void addEntries(const int64_t* extraPtr, const int32_t* extraIdx, vector[int]& retiredIndptr,
                        vector[int]& retiredIndices) except +

        @staticmethod
        CSRPattern ofEntities(const SystemMatrixEntities& entities, int nDof, int nThreads) except +

    cdef cppclass CSRCore(CSRPattern) nogil:
        CSRCore(const int* I, const int* J, int64_t n_pairs, int n_dof) except +

        void update(const double* V_data, double* csr_data) nogil
        void releaseGatherMap()
        void addEntries(const int64_t* extraPtr, const int32_t* extraIdx, vector[int]& retiredIndptr,
                        vector[int]& retiredIndices) except +
        int64_t memoryBytes()

    cdef cppclass CSRDirectAssembler(CSRPattern) nogil:
        CSRDirectAssembler(const SystemMatrixEntities& entities, int nDof, int nThreads) except +

        vector[int64_t] entryStart

        void mapEntities(const SystemMatrixEntities& entities, double* data) except +
        void releaseMap()
        void beginAssembly() nogil
        void scatterBlock(int tid, int entity, const double* block) nogil
        int64_t memoryBytes()

cdef SystemMatrixEntities _borrowedTable(
    const int64_t[::1] dofStart,
    const int32_t[::1] dofs,
    const uint8_t[::1] couplesAllItsDofs,
    const int64_t[::1] localEntryStart,
    const int32_t[::1] localRows,
    const int32_t[::1] localColumns,
):
    """The C++ view of a :class:`~edelweissfe.numerics.systemmatrixcontributor.SystemMatrixEntities` table; the arrays
    must outlive it."""
    cdef int64_t nEntities = dofStart.shape[0] - 1
    if (
        nEntities < 0
        or couplesAllItsDofs.shape[0] != nEntities
        or localEntryStart.shape[0] != nEntities + 1
        or dofStart[nEntities] != dofs.shape[0]
        or localEntryStart[nEntities] != localRows.shape[0]
        or localRows.shape[0] != localColumns.shape[0]
    ):
        raise ValueError("SystemMatrixEntities: the arrays of the table do not fit together.")
    cdef SystemMatrixEntities entities
    entities.nEntities = nEntities
    entities.dofStart = &dofStart[0]
    entities.dofs = &dofs[0] if dofs.shape[0] > 0 else NULL
    entities.couplesAllItsDofs = &couplesAllItsDofs[0] if nEntities > 0 else NULL
    entities.localEntryStart = &localEntryStart[0]
    entities.localRows = &localRows[0] if localRows.shape[0] > 0 else NULL
    entities.localColumns = &localColumns[0] if localColumns.shape[0] > 0 else NULL
    return entities


cdef tuple _patternArrays(CSRPattern& pattern):
    """Copies of ``indptr`` and ``indices`` of a C++ pattern, as NumPy arrays."""
    cdef np.ndarray[int, ndim=1] indptr = np.empty(pattern.nDof + 1, dtype=np.intc)
    cdef np.ndarray[int, ndim=1] indices = np.empty(pattern.nnz, dtype=np.intc)
    memcpy(&indptr[0], pattern.indptr.data(), (pattern.nDof + 1) * sizeof(int))
    if pattern.nnz > 0:
        memcpy(&indices[0], pattern.indices.data(), pattern.nnz * sizeof(int))
    return indptr, indices


def csrPatternOfEntities(entities, int nDof, int numThreads=1):
    """The CSR pattern of the entities' blocks, derived row by row from their DOF lists, as the direct assembly derives
    it (``CSRPattern::ofEntities`` in ``_csrcore.h``): each row is the union of the columns the entities couple it
    with. It is the same pattern as the one :class:`CSRGenerator` derives from the VIJ pairs of the same entities by
    sort-and-sum, derived independently.

    Parameters
    ----------
    entities
        The :class:`~edelweissfe.numerics.systemmatrixcontributor.SystemMatrixEntities` table.
    nDof
        The number of rows and columns.
    numThreads
        The number of threads; the pattern does not depend on it.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray]
        ``indptr`` and ``indices``.
    """
    cdef SystemMatrixEntities table = _borrowedTable(
        entities.dofStart,
        entities.dofs,
        entities.couplesAllItsDofs,
        entities.localEntryStart,
        entities.localRows,
        entities.localColumns,
    )
    cdef CSRPattern pattern
    with nogil:
        pattern = CSRPattern.ofEntities(table, nDof, numThreads)
    return _patternArrays(pattern)


cdef class _CSRCoreOwner:
    """Owns the C++ core of a :class:`CSRGenerator`, its CSR pattern and its gather map, and frees it when it goes.

    The generator holds it, and so does the owner of the core's current pattern (:class:`_CSRPattern`), the base
    object of the pattern arrays of the generator's matrix. So the pattern lives as long as the generator or the
    matrix, whichever goes last, the gather map only as long as the generator. Neither of the two refers to the
    other: a reference cycle would leave both to the cyclic garbage collector, which does not see the size of the core.
    """

    cdef CSRCore* core

    def __dealloc__(self):
        if self.core != NULL:
            del self.core

    @property
    def memoryBytes(self):
        """Bytes the core still holds: its pattern, and its gather map while the generator lives."""
        return self.core.memoryBytes()


cdef class _DirectCSRAssemblerOwner:
    """Owns the C++ core of a :class:`DirectCSRAssembler`, its CSR pattern and its map, and frees it when it goes;
    held like :class:`_CSRCoreOwner`: the pattern lives as long as the assembler or its matrix, the map only as long as
    the assembler."""

    cdef CSRDirectAssembler* assembler

    def __dealloc__(self):
        if self.assembler != NULL:
            del self.assembler

    @property
    def memoryBytes(self):
        """Bytes the core still holds: its pattern, and its map while the assembler lives."""
        return self.assembler.memoryBytes()


cdef class _CSRPattern:
    """The NumPy base object of one pattern's arrays (``indptr``, ``indices``): they stay valid as long as any of
    them is kept, with or without its matrix.

    While this pattern is the current one of a C++ core, its memory is the core's, kept alive by ``coreOwner``. When
    entries are added to the pattern, the old vectors are moved here (their memory stays where it is) and
    ``coreOwner`` is dropped.
    """

    cdef object coreOwner
    cdef vector[int] retiredIndptr
    cdef vector[int] retiredIndices

    @property
    def memoryBytesKeptAlive(self):
        """Bytes that the arrays of this pattern keep alive: all that the core still holds while this is its current
        pattern, else the retired arrays alone."""
        if self.coreOwner is not None:
            return self.coreOwner.memoryBytes
        return <int64_t> (self.retiredIndptr.size() + self.retiredIndices.size()) * sizeof(int)


cdef np.ndarray _intArrayOwnedBy(_CSRPattern owner, int* data, np.npy_intp n):
    """The ``n`` ints at ``data``, memory of the pattern of ``owner``, as a NumPy array whose base is ``owner``."""
    if n == 0:
        # an empty vector of the core may have no memory at all
        return np.empty(0, dtype=np.intc)
    cdef np.ndarray array = np.PyArray_SimpleNewFromData(1, &n, np.NPY_INT, data)
    # PyArray_SetBaseObject takes over a reference
    Py_INCREF(owner)
    np.PyArray_SetBaseObject(array, owner)
    return array


cdef tuple _matrixOnPattern(object coreOwner, CSRPattern* pattern):
    """A matrix with zero values whose pattern arrays are the memory of ``pattern``, which ``coreOwner`` owns; and the
    :class:`_CSRPattern` that keeps it alive.

    Structural mutation of the matrix is forbidden once it is built (see :class:`AliasedCSRMatrix`).
    """
    cdef _CSRPattern holder = _CSRPattern()
    holder.coreOwner = coreOwner
    cdef np.ndarray nd_indptr = _intArrayOwnedBy(holder, pattern.indptr.data(), pattern.nDof + 1)
    cdef np.ndarray nd_indices = _intArrayOwnedBy(holder, pattern.indices.data(), pattern.nnz)

    cdef np.ndarray[double, ndim=1] data = np.zeros(pattern.nnz, dtype=np.double)
    matrix = AliasedCSRMatrix((data, nd_indices, nd_indptr), shape=(pattern.nDof, pattern.nDof))

    # Construction (incl. scipy's internal prune) is done; forbid structural mutation from now on.
    matrix._locked = True
    return holder, matrix


cdef tuple _entriesToAdd(entries, tuple shape):
    """The pattern of ``entries`` as the arrays ``CSRPattern::addEntries`` takes, or None if there is nothing to add."""
    if entries.shape != shape:
        raise ValueError("Entries to add to the pattern: shape {:} differs from {:}.".format(entries.shape, shape))
    if entries.nnz == 0:
        return None
    return np.ascontiguousarray(entries.indptr, dtype=np.int64), np.ascontiguousarray(entries.indices, dtype=np.int32)


cdef class CSRGenerator:
    """
    CSRGenerator class to create and manage a CSR matrix from COO format.

    This class utilizes a C++ core for efficient conversion and updating of the CSR matrix.

    Parameters
    ----------
    I
        The row of every COO pair.
    J
        The column of every COO pair.
    nDof
        The number of rows and columns.

    The gather map indexes the COO pairs with 32-bit integers, so there may be at most ``INT32_MAX`` of them.
    """

    # the owner of the core, the owner of its current pattern (the base of csrMatrix's pattern arrays), and the core
    # itself, for the calls below
    cdef _CSRCoreOwner coreOwner
    cdef _CSRPattern pattern
    cdef CSRCore* core
    cdef public object csrMatrix
    cdef double[:] data_view
    cdef int64_t nCooPairs

    def __dealloc__(self):
        # Only this generator gathers: its gather map goes with it, also while the matrix keeps the pattern alive.
        # A cyclic collection may have cleared coreOwner (to None) first, and freed the core with it.
        if self.coreOwner is not None:
            self.core.releaseGatherMap()

    def __init__(self, I, J, int nDof):  # noqa: E741
        # Ensure int32 dtype regardless of the source array's dtype.
        cdef int[::1] rows = np.asarray(I, dtype=np.intc)
        cdef int[::1] columns = np.asarray(J, dtype=np.intc)

        self.nCooPairs = len(rows)  # Length is still 64-bit capable

        # 1. Run C++ Core
        cdef CSRCore* core
        with nogil:
            core = new CSRCore(&rows[0], &columns[0], self.nCooPairs, nDof)
        self.coreOwner = _CSRCoreOwner()
        self.coreOwner.core = core
        self.core = core

        self._wrapPattern()

    cdef _wrapPattern(self):
        """(Re)create :attr:`csrMatrix` as a view of the core's pattern, with a zeroed data array."""
        # the pattern arrays point into the core and keep it alive (see _CSRPattern); the values are NumPy's own
        self.pattern, self.csrMatrix = _matrixOnPattern(self.coreOwner, self.core)
        self.data_view = self.csrMatrix.data

    def addToPattern(self, entries):
        """Add entries to the CSR pattern that no COO pair contributes to; they are assembled as stored zeros.

        Makes room for values written into :attr:`csrMatrix` after the assembly, such as the
        multi-point-constraint condensation's (see
        :meth:`~edelweissfe.numerics.mpctransformation.MultiPointConstraintTransformation.condensedPatternFillIn`).

        :attr:`csrMatrix` is replaced by a new matrix with the extended pattern (and zero values). A matrix or
        pattern array obtained before keeps the old pattern and stays valid, but the generator no longer updates it.

        Parameters
        ----------
        entries
            The entries to add (CSR, ``nDof x nDof``), with sorted rows and none of them already in the pattern; only
            the pattern is read.
        """
        cdef int64_t[::1] extraPtr
        cdef int32_t[::1] extraIdx
        toAdd = _entriesToAdd(entries, self.csrMatrix.shape)
        if toAdd is None:
            return
        extraPtr, extraIdx = toAdd
        # the old pattern moves to the owner its arrays already have, and no longer needs the core
        self.core.addEntries(&extraPtr[0], &extraIdx[0], self.pattern.retiredIndptr, self.pattern.retiredIndices)
        self.pattern.coreOwner = None
        self._wrapPattern()

    @property
    def memoryBytes(self):
        """Bytes held by the CSR pattern, the gather map and the CSR data array.

        The counterpart to :attr:`DirectCSRAssembler.memoryBytes`, on the same basis -- what the
        object owns. The dominant term is the gather map's ``gather_sources``, one int32 per COO
        pair, which is what the direct path replaces with a uint16 offset per pair (of a dense block in rows short
        enough for it; any other pair keeps an int32 position).

        The VIJ value array and the ``I``/``J`` index arrays are not counted: the generator does not own them.
        """
        return self.core.memoryBytes() + 8 * <int64_t> self.core.nnz

    @property
    def nnz(self):
        """Number of stored entries in the CSR pattern."""
        return self.core.nnz

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
    """Scatters entity blocks straight into a CSR data array, with no VIJ staging array and no I/J pairs.

    The alternative to :class:`CSRGenerator`'s stage-then-gather. It derives its own CSR pattern from the entities, row
    by row (:func:`csrPatternOfEntities`), adds the entries ``patternFillIn`` asks for, and then maps every value of
    every block to its CSR entry. The two derivations of the pattern are independent, and tested against each other.

    It assembles into the CSR data array of its own matrix, :attr:`csrMatrix`: all threads scatter into that one array,
    each addition atomic. The order in which the threads add to an entry depends on their timing, so the values agree
    with a serial assembly to round-off, not bit for bit.

    Parameters
    ----------
    entities
        The :class:`~edelweissfe.numerics.systemmatrixcontributor.SystemMatrixEntities` table; entity ``e`` is number
        ``e`` in :meth:`scatterBlock`.
    nDof
        The number of rows and columns.
    numThreads
        Threads used for the pattern, the map and the zeroing, and how many may scatter at the same time
        (``0 <= tid < numThreads`` in :meth:`scatterBlock`).
    patternFillIn
        Called with the pattern (as :attr:`csrMatrix`, zero values) before the map is built; returns the entries to add
        to it (CSR, sorted rows, none already in the pattern), assembled as stored zeros -- e.g.
        :meth:`~edelweissfe.numerics.mpctransformation.MultiPointConstraintTransformation.condensedPatternFillIn`.
    """

    cdef _DirectCSRAssemblerOwner owner
    cdef _CSRPattern pattern
    cdef CSRDirectAssembler* asm_
    cdef public object csrMatrix

    def __dealloc__(self):
        # Only this assembler scatters: its map goes with it, also while the matrix, or a linear solver that keeps the
        # pattern arrays to compare the next pattern with, keeps the pattern alive. A cyclic collection may have
        # cleared owner (to None) first, and freed the core with it.
        if self.owner is not None:
            self.asm_.releaseMap()

    def __init__(self, entities, int nDof, int numThreads=1, patternFillIn=None):
        cdef SystemMatrixEntities table = _borrowedTable(
            entities.dofStart,
            entities.dofs,
            entities.couplesAllItsDofs,
            entities.localEntryStart,
            entities.localRows,
            entities.localColumns,
        )
        cdef CSRDirectAssembler* assembler
        with nogil:
            assembler = new CSRDirectAssembler(table, nDof, numThreads)
        self.owner = _DirectCSRAssemblerOwner()
        self.owner.assembler = assembler
        self.asm_ = assembler
        self.pattern, self.csrMatrix = _matrixOnPattern(self.owner, assembler)

        cdef int64_t[::1] extraPtr
        cdef int32_t[::1] extraIdx
        toAdd = None if patternFillIn is None else _entriesToAdd(patternFillIn(self.csrMatrix), self.csrMatrix.shape)
        if toAdd is not None:
            extraPtr, extraIdx = toAdd
            assembler.addEntries(&extraPtr[0], &extraIdx[0], self.pattern.retiredIndptr, self.pattern.retiredIndices)
            self.pattern.coreOwner = None
            self.pattern, self.csrMatrix = _matrixOnPattern(self.owner, assembler)

        cdef double[::1] data = self.csrMatrix.data
        cdef double* dataPointer = &data[0] if data.shape[0] > 0 else NULL
        with nogil:
            assembler.mapEntities(table, dataPointer)

    @property
    def entryStart(self):
        """Where the values of the block of each entity begin, counted over all entities in their order, and in the
        last place the number of all values: the block of entity ``e`` has ``entryStart[e + 1] - entryStart[e]``
        values (int64, a copy)."""
        cdef vector[int64_t]* starts = &self.asm_.entryStart
        cdef np.ndarray[int64_t, ndim=1] result = np.empty(starts.size(), dtype=np.int64)
        memcpy(&result[0], starts.data(), starts.size() * sizeof(int64_t))
        return result

    @property
    def memoryBytes(self):
        """Bytes held by the pattern, the offset map and the entity tables; the CSR data array is not counted."""
        return self.asm_.memoryBytes()

    @property
    def corePointer(self):
        """Address of the underlying ``CSRDirectAssembler``, for a threaded caller in another extension.

        A compiled entity loop that scatters from inside its own ``prange`` cannot go through Python. It re-declares
        the C++ class (from ``edelweissfe.numerics.get_include()``) and casts this address back, so the scatter --
        including the column-major traversal and its loop order -- exists in exactly one place.

        The assembler must outlive any caller holding this.
        """
        return <size_t> self.asm_

    def beginAssembly(self):
        """Zero the values of :attr:`csrMatrix`. Call once before the entity loop."""
        with nogil:
            self.asm_.beginAssembly()

    def scatterBlock(self, int tid, int entity, double[::1] block):
        """Add one entity's block -- dense or listed, as the entity declared it -- from a scratch buffer to
        :attr:`csrMatrix`.

        Every addition is atomic, so any number of threads scatter at the same time, as long as no two concurrent
        callers pass the same ``tid`` (``0 <= tid < numThreads``), which selects the caller's own row scratch.
        Releases the GIL, so the workers of a threaded entity loop scatter side by side; see
        :class:`~edelweissfe.numerics.systemmatrixassembly.DirectCSRSystemMatrixAssembly`.
        """
        if block.shape[0] == 0:
            return
        with nogil:
            self.asm_.scatterBlock(tid, entity, &block[0])
