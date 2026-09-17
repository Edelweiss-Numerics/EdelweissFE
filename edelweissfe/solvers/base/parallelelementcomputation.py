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


from typing import NamedTuple

import numpy as np

from edelweissfe.numerics.dofmanager import DofVector, VIJSystemMatrix
from edelweissfe.numerics.parallelizationutilities import (
    chunked_iterable,
    getNumberOfThreads,
    getThreadPool,
    isFreeThreadingSupported,
)
from edelweissfe.timesteppers.timestep import TimeStep

# The Marmot element extension is optional (a pure-Python build has none); without it every
# chunk takes the per-element path.
try:
    from edelweissfe.elements.marmotelement.element import (
        MarmotElementWrapper,
        computeKernelsExplicitForChunk,
    )
except ImportError:  # pragma: no cover - depends on the build
    MarmotElementWrapper = None
    computeKernelsExplicitForChunk = None


def computeElementsInParallel(
    elements: dict, Un1: DofVector, dU: DofVector, P: DofVector, K: VIJSystemMatrix, F: DofVector, timeStep: TimeStep
) -> tuple[DofVector, VIJSystemMatrix, DofVector]:
    """
    Compute the elements in parallel for quasi-static anlysis.

    Parameters
    ----------
    elements : dict
        The elements to compute.
    Un1 : DofVector
        The displacement vector.
    dU : DofVector
        The displacement increment vector.
    P : DofVector
        The internal force vector.
    K : VIJSystemMatrix
        The stiffness matrix.
    F : DofVector
        The flux vector.
    timeStep : TimeStep
        The time step.

    Returns
    -------
    P : DofVector
        The internal force vector.
    K : VIJSystemMatrix
        The stiffness matrix.
    F : DofVector
        The flux vector.
    """

    scatter_P = (
        P.createScatterVector()
    )  # make a scatter vector; which gives 1) contiguous memory access and 2) thread safety

    time = timeStep.totalTime
    dT = timeStep.timeIncrement

    # Process a CHUNK of elements per task, not just one, to keep the per-task
    # dispatch overhead negligible compared to the actual element computation.
    def computeElementsWorker(elementChunk):
        for element in elementChunk:
            Pe = scatter_P[element]
            Ue = Un1[element]
            dUe = dU[element]
            Ke = K[element]
            element.computeKernels(Ke, Pe, Ue, dUe, time, dT)

    numThreads = getNumberOfThreads() if isFreeThreadingSupported() else 1

    if numThreads == 1:
        # avoid ThreadPoolExecutor/task dispatch overhead when there is nothing to parallelize
        computeElementsWorker(elements.values())
    else:
        chunkSize = max(1, len(elements) // (numThreads * 4))
        chunks = chunked_iterable(elements.values(), chunkSize)

        executor = getThreadPool(numThreads)
        list(executor.map(computeElementsWorker, chunks))

    scatter_P.assembleInto(P)
    scatter_P.assembleInto(F, absolute=True)

    return P, K, F


#: Chunks per worker thread for the explicit loop. Four was right when a chunk paid a Python call
#: per element and the elements cost the same: fewer, larger chunks meant less overhead. With the
#: kernels of a chunk in one Cython call the overhead per chunk is negligible, and the elements are
#: NOT equally expensive -- return mapping at yielding and damaging quadrature points makes the
#: elements at the damage front several times dearer than the elastic bulk, and they are spatially
#: clustered, hence clustered in element order. Contiguous chunks of 420 elements were then so
#: unequal that the map waited on its heaviest chunk (32 threads: 43 ms, flat from 16 threads on);
#: 16 chunks per thread balance that dynamically through the executor queue (30 ms). Measured on the
#: anchor pry-out at 53 605 elements; 32 made no further difference.
_chunksPerThread = 16


class _PlannedChunk(NamedTuple):
    """One chunk of the explicit element loop, with everything precomputed that does not change
    between increments; see :func:`_chunkedPlan`."""

    elements: tuple
    #: Concatenated global DOF indices of the chunk's elements; one fancy-index gathers the chunk.
    flatIndices: np.ndarray
    #: ``offsets[i]:offsets[i+1]`` is element ``i``'s slice of the gathered buffers.
    offsets: np.ndarray
    #: Start of the chunk's contiguous range in the scatter buffer, or None if its elements' slots
    #: are not contiguous there (then each element's force is written to its own slot).
    scatterStart: int | None
    scatterEnd: int
    #: Whether every element is a MarmotElementWrapper, so the chunk's kernels run in one Cython call.
    allMarmot: bool


#: Single-entry cache of the per-chunk plan; see :func:`_chunkedPlan`. One entry suffices because
#: a solver works on one element set at a time, and holding references to the mappings it was
#: built for keeps them alive, so identity comparison against them is sound (a freed dict could
#: otherwise have its id reused by a different one).
_planCache = None


def _chunkedPlan(elements: dict, entitiesInDofVector: dict, scatterOffsetMap: dict, chunkSize: int) -> list:
    """Build, or reuse, the plan of the explicit element loop for one chunking of the elements.

    Each chunk gets the concatenation of its elements' DOF indices, the offsets at which each
    element's slice begins, the range its elements occupy in the scatter buffer, and whether all of
    them are Marmot elements -- so a worker gathers the whole chunk with one fancy-index, evaluates
    all its kernels in one Cython call against a buffer of its own, and hands the result to the
    scatter buffer in one slice assignment.

    The plan is only valid for the element set and DOF layout it was built from. h-adaptivity
    rebuilds the DofManager on every topology change, which produces a fresh
    ``idcsOfHigherOrderEntitiesInDofVector`` dict and a fresh scatter layout, so identity of those
    mappings is what detects a stale plan. The element count, the element dict and the chunk size
    are compared as well: those would catch a rebuild that somehow preserved the mapping objects,
    and a stale plan here would silently gather the wrong degrees of freedom rather than fail.

    Parameters
    ----------
    elements
        The elements to compute, in the order they will be chunked.
    entitiesInDofVector
        The entity-to-DOF-index mapping the plan is built against.
    scatterOffsetMap
        The entity-to-``(offset, size)`` mapping of the scatter buffer.
    chunkSize
        Number of elements per chunk.

    Returns
    -------
    list
        One :class:`_PlannedChunk` per chunk.
    """

    global _planCache
    cached = _planCache
    if (
        cached is not None
        and cached[0] is entitiesInDofVector
        and cached[1] is scatterOffsetMap
        and cached[2] == chunkSize
        and cached[3] == len(elements)
        and cached[4] is elements
    ):
        return cached[5]

    plan = []
    for chunk in chunked_iterable(elements.values(), chunkSize):
        indicesPerElement = [entitiesInDofVector[element] for element in chunk]
        offsets = np.zeros(len(chunk) + 1, dtype=np.intp)
        np.cumsum([len(indices) for indices in indicesPerElement], out=offsets[1:])

        scatterStart = scatterOffsetMap[chunk[0]][0]
        scatterEnd = scatterStart
        contiguous = True
        for element, indices in zip(chunk, indicesPerElement):
            offset, size = scatterOffsetMap[element]
            if offset != scatterEnd or size != len(indices):
                contiguous = False
                break
            scatterEnd += size

        allMarmot = computeKernelsExplicitForChunk is not None and all(
            isinstance(element, MarmotElementWrapper) for element in chunk
        )
        plan.append(
            _PlannedChunk(
                chunk,
                np.concatenate(indicesPerElement),
                offsets,
                scatterStart if contiguous else None,
                scatterEnd,
                allMarmot,
            )
        )

    _planCache = (entitiesInDofVector, scatterOffsetMap, chunkSize, len(elements), elements, plan)
    return plan


def computeElementsInParallelForExplicit(
    elements: dict, Un1: DofVector, dU: DofVector, P: DofVector, timeStep: TimeStep
) -> tuple[DofVector, float]:
    """Evaluate the explicit element kernels across the available threads.

    Every chunk of elements works on buffers of its own: the gathered solution and increment, and
    the force buffer its elements write to, which is handed to the shared scatter buffer in one
    slice assignment at the end. Nothing shared is touched per element -- the earlier per-element
    views into the one shared scatter buffer, a Python ``__getitem__`` on a shared object each,
    were what kept 32 threads from getting past a third of their throughput under free threading.
    Where a chunk consists of Marmot elements, its kernels run in one Cython call through raw
    pointers (:func:`~edelweissfe.elements.marmotelement.element.computeKernelsExplicitForChunk`);
    any other chunk takes the per-element path. Both call the same kernels on the same memory, so
    the forces are bit-identical to the per-element loop's.

    Parameters
    ----------
    elements
        The elements to compute.
    Un1
        The solution vector.
    dU
        The solution increment vector.
    P
        The internal force vector, assembled into.
    timeStep
        The time step.

    Returns
    -------
    tuple[DofVector, float]
        The assembled force vector and the summed internal energy.
    """

    scatter_P = P.createScatterVector()
    scatterPlain = scatter_P.view(np.ndarray)
    time = timeStep.totalTime
    dT = timeStep.timeIncrement

    # Both vectors come from DofManager.constructDofVector, which hands every DofVector the same
    # idcsOfHigherOrderEntitiesInDofVector object -- so one index plan serves both. Assert it
    # rather than assume it: gathering dU through indices built for a different layout would
    # produce wrong forces silently.
    if dU.entitiesInDofVector is not Un1.entitiesInDofVector:
        raise ValueError(
            "The solution and increment vectors carry different entity mappings, so a shared "
            "gather plan cannot be used for both."
        )

    Un1_plain = Un1.asPlainArray()
    dU_plain = dU.asPlainArray()

    def compute_chunk(planned: _PlannedChunk) -> float:
        gatheredU = Un1_plain[planned.flatIndices]
        gatheredDU = dU_plain[planned.flatIndices]
        offsets = planned.offsets

        if planned.scatterStart is None:
            # Slots not contiguous: each element writes into its own slot of the scatter buffer.
            chunk_psi = 0.0
            for position, element in enumerate(planned.elements):
                begin = offsets[position]
                end = offsets[position + 1]
                element.computeKernelsExplicit(
                    scatter_P[element], gatheredU[begin:end], gatheredDU[begin:end], time, dT
                )
                chunk_psi += element.computeInternalEnergy()
            return chunk_psi

        Pe = np.zeros(planned.scatterEnd - planned.scatterStart)
        if planned.allMarmot:
            chunk_psi = computeKernelsExplicitForChunk(planned.elements, Pe, gatheredU, gatheredDU, offsets, time, dT)
        else:
            chunk_psi = 0.0
            for position, element in enumerate(planned.elements):
                begin = offsets[position]
                end = offsets[position + 1]
                element.computeKernelsExplicit(Pe[begin:end], gatheredU[begin:end], gatheredDU[begin:end], time, dT)
                chunk_psi += element.computeInternalEnergy()
        scatterPlain[planned.scatterStart : planned.scatterEnd] = Pe
        return chunk_psi

    numThreads = getNumberOfThreads() if isFreeThreadingSupported() else 1

    if numThreads > 1:
        chunk_size = max(1, len(elements) // (numThreads * _chunksPerThread))
    else:
        chunk_size = min(len(elements), 4000)

    plan = _chunkedPlan(elements, Un1.entitiesInDofVector, scatter_P.offsetMap, chunk_size)

    if numThreads == 1:
        psi_total = sum(compute_chunk(plannedChunk) for plannedChunk in plan)
    else:
        executor = getThreadPool(numThreads)
        psi_total = sum(executor.map(compute_chunk, plan))

    scatter_P.assembleInto(P)
    return P, psi_total
