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

import numpy as np
from scipy.sparse import csr_matrix

import edelweissfe.utils.performancetiming as performancetiming
from edelweissfe.numerics.csrgeneratorv2 import CSRGenerator, DirectCSRAssembler
from edelweissfe.numerics.dofmanager import DofManager
from edelweissfe.numerics.parallelizationutilities import forEachSliceInParallel


class SystemMatrixBlocks(ABC):
    """Hands out the blocks of the system matrix: ``K[entity]`` returns the array an entity writes its block of the
    tangent into, shaped by the entity (see
    :meth:`~edelweissfe.numerics.vijentitybase.VIJEntityBase.shapeVIJContribution`). A block that is handed out must
    be committed to the matrix with :meth:`flush`; handing out the next block commits the previous one, so whoever
    loops over entities flushes once after the loop.

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
            An element, a constraint, or any other entity of the DOF manager's VIJ pattern.

        Returns
        -------
        numpy.ndarray
            The block, shaped by the entity.
        """

    @abstractmethod
    def flush(self):
        """Commit the block handed out last to the matrix, if any."""


class ScaledSystemMatrixBlocks(SystemMatrixBlocks):
    """Blocks that enter another :class:`SystemMatrixBlocks` scaled by a factor: every block is handed out as a zeroed
    scratch array, and is added, times the factor, to the entity's block of the target when it is committed.

    The arc length method assembles the tangents of its dead and its reference load this way into the tangent of the
    increment, see :class:`~edelweissfe.solvers.nonlinearimplicitstaticparallelarclength.NISTPArcLength`. Serial
    only: one scratch array serves all entities.

    Parameters
    ----------
    target
        The blocks the scaled blocks enter.
    factor
        The factor.
    """

    def __init__(self, target: SystemMatrixBlocks, factor: float):
        self._target = target
        self._factor = factor
        self._scratch = np.zeros(0)
        self._pendingEntity = None
        self._pendingBlock = None

    def __getitem__(self, entity) -> np.ndarray:
        self.flush()

        size = entity.getVIJContributionSize()
        if size > self._scratch.shape[0]:
            self._scratch = np.zeros(size)
        self._scratch[:size] = 0.0

        self._pendingEntity = entity
        self._pendingBlock = entity.shapeVIJContribution(self._scratch[:size])
        return self._pendingBlock

    def flush(self):
        if self._pendingEntity is None:
            return
        targetBlock = self._target[self._pendingEntity]
        targetBlock += self._factor * self._pendingBlock
        self._target.flush()
        self._pendingEntity = None
        self._pendingBlock = None


class SystemMatrixAssembly(SystemMatrixBlocks):
    """The system matrix of an implicit solver while its entities assemble it, and the CSR matrix it becomes.

    One assembly per Newton iteration: :meth:`beginAssembly`, then every entity writes its block (``K[entity]``, or
    from a worker thread, ``K.blocksOfWorker(worker)[entity]``), then :meth:`assembledMatrix` returns the global
    matrix. It is built per equation system, after its CSR pattern is complete, see
    :meth:`~edelweissfe.solvers.nonlinearimplicitstatic.NIST.createSystemMatrixAssembly`.

    ``K[entity]`` hands out the blocks of worker 0, for the entity loops in the calling thread.

    Parameters
    ----------
    numberOfWorkers
        How many threads may write blocks at the same time, each through its own
        :meth:`blocksOfWorker`.
    """

    #: Whether the CSR generator of the pattern must keep its gather map for this assembly (see
    #: :class:`~edelweissfe.numerics.csrgeneratorv2.CSRGenerator`).
    needsGatherMap = True

    #: The VIJ value array, for code that works on it directly (e.g., the mass term of a dynamic solver, or a compiled
    #: entity loop); None on a path without one.
    vijArray = None

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
    def assembledMatrix(self) -> csr_matrix:
        """Commit every block, and return the global matrix in CSR format.

        The matrix is the assembly's own: the next assembly overwrites its values. Its values may be changed (the
        Dirichlet boundary conditions, the multi-point-constraint condensation), its pattern must not.

        Returns
        -------
        csr_matrix
            The system matrix.
        """

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

    The blocks of all workers are slices of the same array, and no two entities share a slice, so the workers need
    nothing of their own. An entity that is handed its block more than once in an assembly (an element, then a body
    load on it) adds to what is there.

    Parameters
    ----------
    csrGenerator
        The CSR pattern and the gather map.
    dofManager
        The DOF manager of the equation system, with its VIJ pattern.
    numberOfWorkers
        How many threads may write blocks at the same time.
    """

    needsGatherMap = True

    def __init__(self, csrGenerator: CSRGenerator, dofManager: DofManager, numberOfWorkers: int):
        super().__init__(numberOfWorkers)
        self.vijArray = dofManager.constructVIJSystemMatrix()
        self._csrGenerator = csrGenerator

    def beginAssembly(self):
        # the value array holds the entries of all blocks, far too many to zero on a single thread
        forEachSliceInParallel(lambda s: self.vijArray[s].fill(0.0), len(self.vijArray))

    def blocksOfWorker(self, worker: int) -> SystemMatrixBlocks:
        return self

    def __getitem__(self, entity) -> np.ndarray:
        return self.vijArray[entity]

    def flush(self):
        # the value array is the matrix being assembled: a written block is in it already
        pass

    @performancetiming.timeit("gather VIJ to CSR")
    def assembledMatrix(self) -> csr_matrix:
        return self._csrGenerator.updateInPlace(self.vijArray)


class _DirectCSRBlocksOfWorker(SystemMatrixBlocks):
    """The blocks one worker of a :class:`DirectCSRSystemMatrixAssembly` writes: one zeroed scratch array, reused for
    every entity, added to the CSR values when the block is committed.

    Parameters
    ----------
    assembler
        The scatter.
    entityIds
        The number of every entity in the assembler.
    worker
        The number of the worker, which selects its scratch in the scatter.
    """

    def __init__(self, assembler: DirectCSRAssembler, entityIds: dict, worker: int):
        self._assembler = assembler
        self._entityIds = entityIds
        self._worker = worker
        self._scratch = np.zeros(0)
        self._pendingEntity = None
        self._pendingBlock = None

    def __getitem__(self, entity) -> np.ndarray:
        self.flush()

        size = entity.getVIJContributionSize()
        if size > self._scratch.shape[0]:
            self._scratch = np.zeros(size)
        block = self._scratch[:size]
        block[:] = 0.0

        self._pendingEntity = entity
        self._pendingBlock = block
        return entity.shapeVIJContribution(block)

    def flush(self):
        if self._pendingEntity is None:
            return
        self._assembler.scatterBlock(self._worker, self._entityIds[self._pendingEntity], self._pendingBlock)
        self._pendingEntity = None
        self._pendingBlock = None

    def dropPendingBlock(self):
        """Forget the block handed out last, without committing it."""
        self._pendingEntity = None
        self._pendingBlock = None


class DirectCSRSystemMatrixAssembly(SystemMatrixAssembly):
    """Scatter directly: every block is scattered into CSR as soon as it is committed, through a map that knows the
    CSR position of every entry of the VIJ pattern
    (:class:`~edelweissfe.numerics.csrgeneratorv2.DirectCSRAssembler`). No value array of the VIJ pattern's length
    exists.

    Every worker writes its blocks into a scratch array of its own, and all workers add them to the one CSR matrix,
    each addition atomic. The order in which the workers add to an entry depends on their timing, so with more than
    one worker the matrix agrees from run to run to round-off, not bit for bit.

    Every block is a zeroed scratch array: an entity that is handed its block more than once in an assembly (an
    element, then a body load on it) has each one scattered on its own. Entities with a sparse VIJ pattern of their
    own (constraints that couple only some of their degrees of freedom) are scattered entry by entry.

    Parameters
    ----------
    csrGenerator
        The CSR pattern, complete (with the entries the multi-point-constraint condensation adds); its gather map is
        not needed.
    dofManager
        The DOF manager of the equation system, with its VIJ pattern.
    numberOfWorkers
        How many threads may write blocks at the same time; one scratch array each.
    """

    needsGatherMap = False

    def __init__(self, csrGenerator: CSRGenerator, dofManager: DofManager, numberOfWorkers: int):
        super().__init__(numberOfWorkers)

        entities = list(dofManager.idcsOfHigherOrderEntitiesInVIJ)
        #: The scatter, for a compiled entity loop that scatters from its own threads.
        self.assembler = DirectCSRAssembler(csrGenerator, dofManager, numberOfWorkers)
        self.assembler.registerEntities(
            np.fromiter(dofManager.idcsOfHigherOrderEntitiesInVIJ.values(), dtype=np.int64, count=len(entities)),
            np.fromiter((entity.nDof for entity in entities), dtype=np.intc, count=len(entities)),
            np.fromiter((entity.getVIJContributionSize() for entity in entities), dtype=np.int64, count=len(entities)),
        )
        #: The number of every entity in :attr:`assembler`.
        self.entityIds = {entity: i for i, entity in enumerate(entities)}
        self._workers = [_DirectCSRBlocksOfWorker(self.assembler, self.entityIds, w) for w in range(numberOfWorkers)]

    def beginAssembly(self):
        for worker in self._workers:
            worker.dropPendingBlock()
        self.assembler.beginAssembly()

    def blocksOfWorker(self, worker: int) -> SystemMatrixBlocks:
        return self._workers[worker]

    def assembledMatrix(self) -> csr_matrix:
        for worker in self._workers:
            worker.flush()
        return self.assembler.csrMatrix
