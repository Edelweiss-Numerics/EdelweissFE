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
        CSRCore(const int* I, const int* J, long n_pairs, int n_dof) except +

        vector[int] indptr
        vector[int] indices
        int nnz
        int nDof

        void update(const double* V_data, double* csr_data) nogil

cdef class CSRGenerator:
    """
    CSRGenerator class to create and manage a CSR matrix from COO format.

    This class utilizes a C++ core for efficient conversion and updating of the CSR matrix.

    Parameters
    ----------
    systemMatrix : object
        An object containing COO format data with attributes I, J, and nDof.
    """

    cdef CSRCore* core
    cdef public object csrMatrix
    cdef double[:] data_view
    cdef long nCooPairs  # Kept as long (int64)

    def __dealloc__(self):
        if self.core != NULL:
            del self.core

    def __init__(self, systemMatrix):
        # Ensure int32 dtype regardless of the source array's dtype.
        # dofmanager.py already produces np.intc arrays, but we guard here
        # in case CSRGenerator is called from outside the standard path.
        cdef int[::1] I = np.asarray(systemMatrix.I, dtype=np.intc)  # noqa
        cdef int[::1] J = np.asarray(systemMatrix.J, dtype=np.intc)

        self.nCooPairs = len(I)  # Length is still 64-bit capable

        cdef int nDof = int(systemMatrix.nDof)

        # 1. Run C++ Core
        with nogil:
            self.core = new CSRCore(&I[0], &J[0], self.nCooPairs, nDof)

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
