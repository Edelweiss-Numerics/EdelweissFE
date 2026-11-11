#!/usr/bin/env python3
# -*- coding: utf-8 -*-
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
"""The assembly of the system matrix: where every element, constraint and load writes its block of the tangent, and
how the blocks become the global matrix in CSR format.

An implicit solver assembles the tangent from the blocks of all entities: ``Ke = K[element]`` hands out the array the
element writes its block into, and :meth:`SystemMatrixAssembly.assembledMatrix` sums all blocks into the global
matrix. Two ways of summing are implemented behind this one interface, so that neither the solvers nor the entities
know which is in use (see :ref:`system-matrix-assembly`):

:class:`VIJSystemMatrixAssembly`
    Every block has its own slice of one long value array (VIJ, or COO, layout), and the duplicates are gathered into
    CSR afterwards.
:class:`DirectCSRSystemMatrixAssembly`
    Every block is scattered into CSR right after it was computed; no value array exists.
"""

from abc import ABC, abstractmethod
from functools import partial
from typing import Callable

import numpy as np
from scipy.sparse import csr_matrix

import edelweissfe.utils.performancetiming as performancetiming
from edelweissfe.numerics.csrgeneratorv2 import (
    AliasedCSRMatrix,
    CSRGenerator,
    DirectCSRAssembler,
)
from edelweissfe.numerics.dofmanager import DofManager
from edelweissfe.numerics.parallelizationutilities import (
    forEachSliceInParallel,
    mapChunksInParallel,
)
from edelweissfe.numerics.systemmatrixcontributor import (
    SystemMatrixEntities,
    countEntriesOfBlock,
)
from edelweissfe.numerics.vijsystemmatrix import VIJSystemMatrix

#: Rows per block in which the positions of the entries of a matrix in the sparsity pattern are searched (see
#: :meth:`SystemMatrixAssembly.copyOntoSparsityPattern`): the temporaries of a block stay small.
_ROWS_PER_SEARCH_BLOCK = 1 << 14


def _addScaledValues(systemValues: np.ndarray, values: np.ndarray, factor: float):
    """``systemValues += factor * values``, in slices on the thread pool, each slice with a temporary of its own size
    only: the term of :meth:`SystemMatrixAssembly.addMatrix`.

    Parameters
    ----------
    systemValues
        The values of the system matrix, added to in place.
    values
        The values of the matrix added, one per entry of the same pattern.
    factor
        The factor they enter with.
    """

    def addSlice(s: slice):
        systemValues[s] += factor * values[s]

    forEachSliceInParallel(addSlice, len(systemValues))


def _addScaledValuesAt(systemValues: np.ndarray, positions: np.ndarray, values: np.ndarray, factor: float):
    """``systemValues[positions] += factor * values``, every position once: the term of
    :meth:`SystemMatrixAssembly.addLumpedMatrix`.

    Parameters
    ----------
    systemValues
        The values of the system matrix, added to in place.
    positions
        The positions of the entries the values enter, distinct.
    values
        The values.
    factor
        The factor they enter with.
    """

    systemValues[positions] += factor * values


class SystemMatrixBlocks(ABC):
    """Hands out the blocks of the system matrix: ``K[entity]`` returns the array an entity writes its block of the
    tangent into, shaped by the entity (see
    :meth:`~edelweissfe.numerics.systemmatrixcontributor.SystemMatrixContributorBase.shapeBlock`). A block that is
    handed out must be committed to the matrix with :meth:`flush`; handing out the next block commits the previous
    one, so whoever loops over entities flushes once after the loop.

    A block is either the entity's own place in the matrix, which already holds what was assembled for the entity
    before, or a zeroed scratch array. So an entity adds its block to the array it is given or overwrites it, and
    never reads it.
    """

    @abstractmethod
    def __getitem__(self, entity) -> np.ndarray:
        """The array ``entity`` writes its block into.

        Parameters
        ----------
        entity
            An element, a constraint, or any other entity that contributes a block to the system matrix.

        Returns
        -------
        numpy.ndarray
            The block, shaped by the entity.
        """

    @abstractmethod
    def flush(self):
        """Commit the block handed out last to the matrix, if any."""


class _ScratchSystemMatrixBlocks(SystemMatrixBlocks):
    """Blocks handed out as one zeroed scratch array, reused for every entity, and committed elsewhere: the block
    handed out last is pending until the next one is handed out or :meth:`flush` is called, which commits it
    (:meth:`_commit`)."""

    def __init__(self):
        self._scratch = np.zeros(0)
        self._pendingEntity = None
        self._pendingValues = None

    @abstractmethod
    def numberOfEntriesInBlock(self, entity) -> int:
        """The number of values of the block of ``entity``, from the stored slice bounds.

        Parameters
        ----------
        entity
            An entity of the assembly.

        Returns
        -------
        int
            The number of values.
        """

    @abstractmethod
    def _commit(self, entity, values: np.ndarray):
        """Commit the values of the block of ``entity``.

        Parameters
        ----------
        entity
            The entity.
        values
            The values of its block, flat, in the order of the block.
        """

    def __getitem__(self, entity) -> np.ndarray:
        self.flush()

        size = self.numberOfEntriesInBlock(entity)
        if size > self._scratch.shape[0]:
            self._scratch = np.zeros(size)
        values = self._scratch[:size]
        values[:] = 0.0

        self._pendingEntity = entity
        self._pendingValues = values
        return entity.shapeBlock(values)

    def flush(self):
        if self._pendingEntity is None:
            return
        self._commit(self._pendingEntity, self._pendingValues)
        self.dropPendingBlock()

    def dropPendingBlock(self):
        """Forget the block handed out last, without committing it."""
        self._pendingEntity = None
        self._pendingValues = None


class ScaledSystemMatrixBlocks(_ScratchSystemMatrixBlocks):
    """Blocks that enter another :class:`SystemMatrixBlocks` scaled by a factor: every block is handed out as a zeroed
    scratch array, and is added, times the factor, to the entity's block of the target when it is committed.

    The arc length method assembles the tangents of its dead and its reference load this way into the tangent of the
    increment, see :class:`~edelweissfe.solvers.nonlinearimplicitstaticparallelarclength.NISTPArcLength`. Serial
    only: one scratch array serves all entities.

    Parameters
    ----------
    target
        The assembly the scaled blocks enter.
    factor
        The factor.
    """

    def __init__(self, target: "SystemMatrixAssembly", factor: float):
        super().__init__()
        self._target = target
        self._factor = factor

    def numberOfEntriesInBlock(self, entity) -> int:
        return self._target.numberOfEntriesInBlock(entity)

    def _commit(self, entity, values: np.ndarray):
        targetBlock = self._target[entity]
        targetBlock += self._factor * entity.shapeBlock(values)
        self._target.flush()


class SystemMatrixAssembly(SystemMatrixBlocks):
    """The system matrix of an implicit solver while its entities assemble it, and the CSR matrix it becomes.

    One assembly per Newton iteration: :meth:`beginAssembly`, then every entity writes its block (``K[entity]``, or
    from a worker thread, ``K.blocksOfWorker(worker)[entity]``), then :meth:`assembledMatrix` returns the global
    matrix. It is built per equation system, together with its CSR pattern, see
    :meth:`~edelweissfe.solvers.nonlinearimplicitstatic.NIST.createSystemMatrixAssembly`.

    ``K[entity]`` hands out the blocks of worker 0, for the entity loops in the calling thread.

    A term that is not the block of an entity, but a whole matrix on the sparsity pattern of the system matrix, enters
    during the assembly too: :meth:`addMatrix` and :meth:`addLumpedMatrix`. A dynamic solver forms its effective
    tangent this way, ``K_eff = K + M / (beta dT^2) + gamma C / (beta dT)``::

        K.addMatrix(M, massFactor)
        K.addLumpedMatrix(lumpedDamping, dampingFactor)

    with the mass ``M`` kept on the sparsity pattern of the system matrix (:meth:`copyOntoSparsityPattern`).

    Parameters
    ----------
    numberOfWorkers
        How many threads may write blocks at the same time, each through its own
        :meth:`blocksOfWorker`.
    """

    def __init__(self, numberOfWorkers: int):
        #: How many threads may write blocks at the same time.
        self.numberOfWorkers = numberOfWorkers

    @abstractmethod
    def beginAssembly(self):
        """Start a new assembly: drop everything assembled before, and any block that was handed out but not
        committed (an assembly that was abandoned must not leak into the next)."""

    @abstractmethod
    def blocksOfWorker(self, worker: int) -> SystemMatrixBlocks:
        """The blocks worker thread ``worker`` writes into, while other workers write into theirs.

        Parameters
        ----------
        worker
            The number of the worker, ``0 <= worker <`` :attr:`numberOfWorkers`.

        Returns
        -------
        SystemMatrixBlocks
            The blocks of this worker.
        """

    @abstractmethod
    def numberOfEntriesInBlock(self, entity) -> int:
        """The number of values of the block of ``entity``, as the assembly stored it when it was built.

        Parameters
        ----------
        entity
            An element, a constraint, or any other entity that contributes a block to the system matrix.

        Returns
        -------
        int
            The number of values.
        """

    @property
    @abstractmethod
    def csrMatrix(self) -> csr_matrix:
        """The CSR matrix the assembly assembles into, the one :meth:`assembledMatrix` returns: its own buffer, which
        the next assembly overwrites, and whose pattern must not change."""

    @abstractmethod
    def assembledMatrix(self) -> csr_matrix:
        """Commit every block, and return the global matrix in CSR format.

        The matrix is the assembly's own: the next assembly overwrites its values. Its values may be changed (the
        Dirichlet boundary conditions, the multi-point-constraint condensation), its pattern must not.

        Returns
        -------
        csr_matrix
            The system matrix.
        """

    @abstractmethod
    def _addDuringAssembly(self, addToValues: Callable[[np.ndarray], None]):
        """Let ``addToValues`` add a term to the values of :attr:`csrMatrix`, where the assembly sums the blocks into
        them: at once, if they hold the sum from :meth:`beginAssembly` on, or in :meth:`assembledMatrix`, if they are
        written only there.

        Parameters
        ----------
        addToValues
            Called with the value array of :attr:`csrMatrix`, to which it adds the term in place.
        """

    def addMatrix(self, matrix: csr_matrix, factor: float):
        """Add ``factor * matrix`` to the system matrix, during the assembly: like the block of an entity, after
        :meth:`beginAssembly` and before :meth:`assembledMatrix`. E.g., the mass of a dynamic solver to its effective
        tangent, ``K_eff = K + M / (beta dT^2) + ...``.

        ``matrix`` is on the sparsity pattern of the system matrix (:meth:`copyOntoSparsityPattern`), so its values are
        added entry by entry, in slices on the thread pool, without a temporary matrix. Its values must not change
        before :meth:`assembledMatrix`: an assembly that writes the values of the system matrix only there adds them
        there.

        Parameters
        ----------
        matrix
            The matrix, on the sparsity pattern of the system matrix.
        factor
            The factor it enters with.

        Raises
        ------
        ValueError
            If ``matrix`` is not on the sparsity pattern of the system matrix.
        """

        pattern = self.csrMatrix
        if matrix.indptr is not pattern.indptr or matrix.indices is not pattern.indices:
            raise ValueError(
                "The matrix to add is not on the sparsity pattern of the system matrix; copy it onto the pattern first "
                "(copyOntoSparsityPattern)."
            )

        self._addDuringAssembly(partial(_addScaledValues, values=matrix.data, factor=factor))

    def addLumpedMatrix(self, lumpedMatrix: np.ndarray, factor: float):
        """Add ``factor`` times a lumped matrix -- a diagonal matrix, kept as one value per degree of freedom -- to the
        system matrix, during the assembly: like the block of an entity, after :meth:`beginAssembly` and before
        :meth:`assembledMatrix`. E.g., the lumped damping of a dynamic solver to its effective tangent,
        ``K_eff = ... + gamma C / (beta dT)``.

        Every nonzero value enters the diagonal entry of its degree of freedom, which the sparsity pattern must have.
        The values are taken when this is called.

        Parameters
        ----------
        lumpedMatrix
            The lumped matrix, one value per degree of freedom.
        factor
            The factor it enters with.

        Raises
        ------
        ValueError
            If ``lumpedMatrix`` does not hold one value per degree of freedom, or if a nonzero value lies on a degree of
            freedom without a diagonal entry in the sparsity pattern.
        """

        lumpedMatrix = np.asarray(lumpedMatrix)
        if lumpedMatrix.shape != (self.csrMatrix.shape[0],):
            raise ValueError(
                "A lumped matrix holds one value per degree of freedom, {:}; got an array of shape {:}.".format(
                    self.csrMatrix.shape[0], lumpedMatrix.shape
                )
            )

        dofs = np.flatnonzero(lumpedMatrix)
        positions = self._positionsOfEntries(dofs, dofs)
        withoutDiagonal = dofs[positions < 0]
        if withoutDiagonal.size:
            raise ValueError(
                "The lumped matrix has values on {:} degrees of freedom (first: {:}) that have no diagonal entry in the "
                "sparsity pattern of the system matrix: no entity couples them.".format(
                    withoutDiagonal.size, int(withoutDiagonal[0])
                )
            )

        self._addDuringAssembly(
            partial(_addScaledValuesAt, positions=positions, values=lumpedMatrix[dofs], factor=factor)
        )

    def copyOntoSparsityPattern(self, matrix: csr_matrix) -> csr_matrix:
        """A copy of ``matrix`` on the sparsity pattern of the system matrix, the form :meth:`addMatrix` takes: a value
        array of its own, one value per entry of the pattern (zero where ``matrix`` has no entry), and the pattern
        arrays of the system matrix, shared, not copied. E.g., the mass a dynamic solver assembled through this
        assembly (:meth:`assembledMatrix`, whose values the next assembly overwrites), or the mass of the equation
        system before a rebuild, moved onto the pattern of the new one.

        The copy refuses to be changed structurally (see :class:`~edelweissfe.numerics.csrgeneratorv2.AliasedCSRMatrix`):
        its pattern is the one of the system matrix. Its values may be changed in place. It keeps the pattern alive as
        long as it lives.

        Parameters
        ----------
        matrix
            The matrix, on this pattern, on a part of it, or on any pattern whose entries outside this one are zero.

        Returns
        -------
        csr_matrix
            The copy.

        Raises
        ------
        ValueError
            If ``matrix`` does not have the shape of the system matrix, or has a nonzero entry outside the sparsity
            pattern.
        """

        pattern = self.csrMatrix
        if matrix.shape != pattern.shape:
            raise ValueError(
                "A matrix of shape {:} cannot be copied onto the sparsity pattern of the system matrix, of shape "
                "{:}.".format(matrix.shape, pattern.shape)
            )

        if matrix.indptr is pattern.indptr and matrix.indices is pattern.indices:
            values = np.array(matrix.data, dtype=float)
        else:
            values = np.zeros(pattern.nnz)

            def copyRows(rows: slice):
                entries = slice(matrix.indptr[rows.start], matrix.indptr[rows.stop])
                rowOfEntry = np.repeat(
                    np.arange(rows.start, rows.stop), np.diff(matrix.indptr[rows.start : rows.stop + 1])
                )
                positions = self._positionsOfEntries(rowOfEntry, matrix.indices[entries])
                entryValues = matrix.data[entries]
                outside = positions < 0
                if np.any(entryValues[outside] != 0.0):
                    first = np.flatnonzero(outside & (entryValues != 0.0))[0]
                    raise ValueError(
                        "The matrix has a nonzero entry ({:}, {:}) outside the sparsity pattern of the system "
                        "matrix.".format(int(rowOfEntry[first]), int(matrix.indices[entries][first]))
                    )
                # the rows of a block are the block's alone, so the blocks add to separate values; a duplicate entry
                # is summed
                np.add.at(values, positions[~outside], entryValues[~outside])

            forEachSliceInParallel(copyRows, pattern.shape[0], _ROWS_PER_SEARCH_BLOCK)

        # Built empty and given the arrays afterwards: the constructor might copy the pattern arrays (SciPy's prune).
        copy = AliasedCSRMatrix(pattern.shape)
        copy.indptr = pattern.indptr
        copy.indices = pattern.indices
        copy.data = values
        copy.lock()
        return copy

    def _positionsOfEntries(self, rows: np.ndarray, columns: np.ndarray) -> np.ndarray:
        """Where the entries ``(rows[k], columns[k])`` are in the value array of the sparsity pattern: in row ``r``,
        the columns are the sorted ``indices[indptr[r] : indptr[r + 1]]``, so each column is found by bisection there,
        all entries at once.

        Parameters
        ----------
        rows
            The rows of the entries.
        columns
            Their columns.

        Returns
        -------
        np.ndarray
            The position of every entry in the value array, or -1 for an entry that is not in the pattern.
        """

        indptr, indices = self.csrMatrix.indptr, self.csrMatrix.indices
        rows = np.asarray(rows, dtype=np.int64)
        columns = np.asarray(columns, dtype=np.int64)
        if indices.size == 0:
            return np.full(rows.shape, -1, dtype=np.int64)

        rowEnd = indptr[rows + 1].astype(np.int64)
        # The first position in the row whose column is not smaller than the one searched for, in [lower, upper].
        lower = indptr[rows].astype(np.int64)
        upper = rowEnd.copy()
        searching = lower < upper
        while np.any(searching):
            middle = (lower + upper) // 2
            # a finished search reads any valid position, and does not move
            columnInTheMiddle = indices[np.where(searching, middle, 0)]
            goRight = searching & (columnInTheMiddle < columns)
            goLeft = searching & ~goRight
            lower[goRight] = middle[goRight] + 1
            upper[goLeft] = middle[goLeft]
            searching = lower < upper

        found = (lower < rowEnd) & (indices[np.minimum(lower, indices.size - 1)] == columns)
        return np.where(found, lower, -1)

    def scaledBy(self, factor: float) -> ScaledSystemMatrixBlocks:
        """Blocks that enter this matrix scaled by ``factor``, see :class:`ScaledSystemMatrixBlocks`.

        Parameters
        ----------
        factor
            The factor.

        Returns
        -------
        ScaledSystemMatrixBlocks
            The scaled blocks.
        """

        return ScaledSystemMatrixBlocks(self, factor)

    def __getitem__(self, entity) -> np.ndarray:
        return self.blocksOfWorker(0)[entity]

    def flush(self):
        self.blocksOfWorker(0).flush()


class VIJSystemMatrixAssembly(SystemMatrixAssembly):
    """Stage, then gather: every block is the entity's own slice of one value array in VIJ (COO) layout, which is
    gathered into CSR, summing the duplicates, once all blocks are written.

    The textbook way: every value ``V[k]`` of a block has its global row ``I[k]`` and column ``J[k]``, the entity's
    local row and column (:meth:`~edelweissfe.numerics.systemmatrixcontributor.SystemMatrixContributorBase.localEntriesOfBlock`)
    looked up in its global DOF list, and the CSR matrix is the sum of all ``(I[k], J[k], V[k])``.

    The blocks of all workers are slices of the same array, and no two entities share a slice, so the workers need
    nothing of their own. An entity that is handed its block more than once in an assembly (an element, then a body
    load on it) adds to what is there.

    Parameters
    ----------
    dofManager
        The DOF manager of the equation system: the global DOF list of every entity.
    numberOfWorkers
        How many threads may write blocks at the same time.
    patternFillIn
        Called with the CSR pattern (as a matrix of zeros) once it is derived; returns the entries no entity
        contributes that the pattern must have as well (e.g.
        :meth:`~edelweissfe.numerics.mpctransformation.MultiPointConstraintTransformation.condensedPatternFillIn`).
    """

    def __init__(self, dofManager: DofManager, numberOfWorkers: int, patternFillIn: Callable | None = None):
        super().__init__(numberOfWorkers)

        idcsOfEntity = dofManager.idcsOfHigherOrderEntitiesInDofVector

        # The entities follow one another in V, I and J, so where each one starts is known before any of them is
        # located; each then fills only its own part of I and J, and they do so in chunks on the thread pool.
        #: The slice of V, I and J that belongs to every entity.
        self.vijSliceOfEntity = {}
        nEntries = 0
        for entity in idcsOfEntity:
            nEntriesOfEntity = countEntriesOfBlock(entity)
            self.vijSliceOfEntity[entity] = slice(nEntries, nEntries + nEntriesOfEntity)
            nEntries += nEntriesOfEntity

        #: The global row of every value.
        self.I = np.empty(nEntries, dtype=np.intc)  # noqa: E741
        #: The global column of every value.
        self.J = np.empty(nEntries, dtype=np.intc)
        mapChunksInParallel(list(idcsOfEntity.items()), self._locateEntriesInSystem)

        #: The CSR pattern, derived from I and J by sort-and-sum, and the gather map.
        self.csrGenerator = CSRGenerator(self.I, self.J, dofManager.nDof)
        if patternFillIn is not None:
            self.csrGenerator.addToPattern(patternFillIn(self.csrGenerator.csrMatrix))
        #: The VIJ value array: the blocks of all entities, each in its slice.
        self.vijArray = self.constructVIJSystemMatrix()
        #: The terms added during the current assembly (:meth:`addMatrix`, :meth:`addLumpedMatrix`), added to the
        #: values of the CSR matrix after the gather, which overwrites them.
        self._termsAddedAfterGather = []

    def _locateEntriesInSystem(self, entitiesWithIndices: list):
        """The local-to-global step: the global row and column of every value of the blocks of some entities.

        Parameters
        ----------
        entitiesWithIndices
            The entities, each with its global DOF list.
        """
        for entity, idcs in entitiesWithIndices:
            localRows, localColumns = entity.localEntriesOfBlock()
            vijSlice = self.vijSliceOfEntity[entity]
            self.I[vijSlice] = idcs[localRows]
            self.J[vijSlice] = idcs[localColumns]

    def constructVIJSystemMatrix(self) -> VIJSystemMatrix:
        """A value array in the layout of this assembly, zeroed, for another matrix assembled from the blocks of the
        same entities; :attr:`csrGenerator` gathers it into CSR.

        Returns
        -------
        VIJSystemMatrix
            The value array.
        """
        return VIJSystemMatrix(len(self.I), self.vijSliceOfEntity)

    def beginAssembly(self):
        # the value array holds the entries of all blocks, far too many to zero on a single thread
        forEachSliceInParallel(lambda s: self.vijArray[s].fill(0.0), len(self.vijArray))
        self._termsAddedAfterGather.clear()

    def _addDuringAssembly(self, addToValues: Callable[[np.ndarray], None]):
        # the gather overwrites the values of the CSR matrix: the term is added after it
        self._termsAddedAfterGather.append(addToValues)

    def blocksOfWorker(self, worker: int) -> SystemMatrixBlocks:
        return self

    def numberOfEntriesInBlock(self, entity) -> int:
        vijSlice = self.vijSliceOfEntity[entity]
        return vijSlice.stop - vijSlice.start

    def __getitem__(self, entity) -> np.ndarray:
        return self.vijArray[entity]

    def flush(self):
        # the value array is the matrix being assembled: a written block is in it already
        pass

    @property
    def csrMatrix(self) -> csr_matrix:
        return self.csrGenerator.csrMatrix

    def assembledMatrix(self) -> csr_matrix:
        with performancetiming.timeit("gather VIJ to CSR"):
            matrix = self.csrGenerator.updateInPlace(self.vijArray)
        # after every gather anew, so that the matrix is the same however often it is asked for
        for addToValues in self._termsAddedAfterGather:
            addToValues(matrix.data)
        return matrix


class _DirectCSRBlocksOfWorker(_ScratchSystemMatrixBlocks):
    """The blocks one worker of a :class:`DirectCSRSystemMatrixAssembly` writes: one zeroed scratch array, reused for
    every entity, added to the CSR values when the block is committed.

    It holds what it needs of the assembly, not the assembly itself: a reference cycle would keep the whole assembly
    alive until the cyclic garbage collector runs, also across a rebuild of the equation system.

    Parameters
    ----------
    assembler
        The scatter.
    entityIds
        The number of every entity in the assembler.
    entryStart
        Where the values of the block of every entity begin, see :attr:`DirectCSRSystemMatrixAssembly.entryStart`.
    worker
        The number of the worker, which selects its scratch in the scatter.
    """

    def __init__(self, assembler: DirectCSRAssembler, entityIds: dict, entryStart: np.ndarray, worker: int):
        super().__init__()
        self._assembler = assembler
        self._entityIds = entityIds
        self._entryStart = entryStart
        self._worker = worker

    def numberOfEntriesInBlock(self, entity) -> int:
        i = self._entityIds[entity]
        return int(self._entryStart[i + 1] - self._entryStart[i])

    def _commit(self, entity, values: np.ndarray):
        self._assembler.scatterBlock(self._worker, self._entityIds[entity], values)


class DirectCSRSystemMatrixAssembly(SystemMatrixAssembly):
    """Scatter directly: every block is scattered into CSR as soon as it is committed, through a map that knows the
    CSR position of every value of every block (:class:`~edelweissfe.numerics.csrgeneratorv2.DirectCSRAssembler`).
    The CSR pattern is derived from the entities' DOF lists; neither a value array nor index arrays of the VIJ layout
    exist.

    Every worker writes its blocks into a scratch array of its own, and all workers add them to the one CSR matrix,
    each addition atomic. The order in which the workers add to an entry depends on their timing, so with more than
    one worker the matrix agrees from run to run to round-off, not bit for bit.

    Every block is a zeroed scratch array: an entity that is handed its block more than once in an assembly (an
    element, then a body load on it) has each one scattered on its own. Entities that list the entries of their block
    themselves (constraints that couple only some of their degrees of freedom, see
    :meth:`~edelweissfe.numerics.systemmatrixcontributor.SystemMatrixContributorBase.localEntriesOfBlock`) are
    scattered entry by entry.

    Parameters
    ----------
    dofManager
        The DOF manager of the equation system: the global DOF list of every entity.
    numberOfWorkers
        How many threads may write blocks at the same time; one scratch array each.
    patternFillIn
        As for :class:`VIJSystemMatrixAssembly`.
    """

    def __init__(self, dofManager: DofManager, numberOfWorkers: int, patternFillIn: Callable | None = None):
        super().__init__(numberOfWorkers)

        dofsOfEntity = dofManager.idcsOfHigherOrderEntitiesInDofVector
        #: The scatter, for a compiled entity loop that scatters from its own threads.
        self.assembler = DirectCSRAssembler(
            SystemMatrixEntities.ofEntities(dofsOfEntity), dofManager.nDof, numberOfWorkers, patternFillIn
        )
        #: Where the values of the block of every entity begin, counted over all entities in their order, and in the
        #: last place the number of all values: the block of entity ``i`` has ``entryStart[i + 1] - entryStart[i]``
        #: values.
        self.entryStart = self.assembler.entryStart
        #: The number of every entity in :attr:`assembler`.
        self.entityIds = {entity: i for i, entity in enumerate(dofsOfEntity)}
        self._workers = [
            _DirectCSRBlocksOfWorker(self.assembler, self.entityIds, self.entryStart, w) for w in range(numberOfWorkers)
        ]

    def beginAssembly(self):
        for worker in self._workers:
            worker.dropPendingBlock()
        self.assembler.beginAssembly()

    def blocksOfWorker(self, worker: int) -> SystemMatrixBlocks:
        return self._workers[worker]

    def numberOfEntriesInBlock(self, entity) -> int:
        return self._workers[0].numberOfEntriesInBlock(entity)

    @property
    def csrMatrix(self) -> csr_matrix:
        return self.assembler.csrMatrix

    def _addDuringAssembly(self, addToValues: Callable[[np.ndarray], None]):
        # the values of the CSR matrix hold the sum from beginAssembly on: the term is added at once
        addToValues(self.assembler.csrMatrix.data)

    def assembledMatrix(self) -> csr_matrix:
        for worker in self._workers:
            worker.flush()
        return self.assembler.csrMatrix
