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


import numpy as np

from edelweissfe.numerics.dofmanager import DofVector, VIJSystemMatrix
from edelweissfe.numerics.parallelizationutilities import (
    chunked_iterable,
    getNumberOfWorkerThreads,
    getThreadPool,
)
from edelweissfe.timesteppers.timestep import TimeStep


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

    # One chunk of elements per task, to keep the dispatch overhead negligible.
    def computeElementsWorker(elementChunk):
        for element in elementChunk:
            Pe = scatter_P[element]
            Ue = Un1[element]
            dUe = dU[element]
            Ke = K[element]
            element.computeKernels(Ke, Pe, Ue, dUe, time, dT)

    numThreads = getNumberOfWorkerThreads()

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


#: How many chunks of elements each thread is handed in the explicit element loop, so that the
#: executor can balance unequal chunks dynamically; see :func:`computeElementsInParallelForExplicit`.
_chunksPerThread = 16


#: Single-entry cache of the per-chunk gather plan; see :func:`_chunkedGatherPlan`. It holds
#: references to the mappings it was built for, which keeps identity comparison against them sound.
_gatherPlanCache = None


def _chunkedGatherPlan(elements: dict, entitiesInDofVector: dict, scatterOffsetMap: dict, chunkSize: int) -> list:
    """Build, or reuse, the flat gather index plan for one chunking of the elements.

    Each chunk gets the concatenation of its elements' DOF indices, the offsets at which each
    element's slice begins, and the positions its elements occupy in the scatter buffer -- so a
    worker can gather the whole chunk with one fancy-index, hand out views into the gathered
    buffers, and write the chunk's forces back with one fancy-index.

    The plan is only valid for the element set and DOF layout it was built from. h-adaptivity
    rebuilds the DofManager on every topology change, which produces a fresh
    ``idcsOfHigherOrderEntitiesInDofVector`` dict and a fresh scatter layout, so identity of those
    mappings is what detects a stale plan. The element count and chunk size are compared as well:
    those would catch a rebuild that somehow preserved the mapping objects, and a stale plan here
    would silently gather the wrong degrees of freedom rather than fail.

    Parameters
    ----------
    elements
        The elements to compute, in the order they will be chunked.
    entitiesInDofVector
        The entity-to-DOF-index mapping the plan is built against.
    scatterOffsetMap
        The entity-to-``(offset, size)`` layout of the scatter buffer the forces are written to.
    chunkSize
        Number of elements per chunk.

    Returns
    -------
    list
        One ``(chunkElements, flatIndices, offsets, scatterIndices)`` tuple per chunk.
    """

    global _gatherPlanCache

    # Read the global once: under PYTHON_GIL=0 another thread may replace it between check and use.
    # The tuple is replaced atomically, so one read is a consistent snapshot.
    cached = _gatherPlanCache

    if (
        cached is not None
        and cached[0] is entitiesInDofVector
        and cached[1] == chunkSize
        and cached[2] == len(elements)
        # The plan stores positions into the scatter layout, so it is compared in its own right.
        and cached[5] is scatterOffsetMap
        # The plan stores the chunked elements, so an equal-length different collection must miss.
        and cached[4] is elements
    ):
        return cached[3]

    plan = []
    for chunk in chunked_iterable(elements.values(), chunkSize):
        indicesPerElement = [entitiesInDofVector[element] for element in chunk]
        offsets = np.zeros(len(chunk) + 1, dtype=np.intp)
        np.cumsum([len(indices) for indices in indicesPerElement], out=offsets[1:])

        # The chunk's slots in the scatter buffer, in the order of its own force buffer.
        scatterSlots = [scatterOffsetMap[element] for element in chunk]
        scatterIndices = np.concatenate([np.arange(offset, offset + size) for offset, size in scatterSlots])

        plan.append((chunk, np.concatenate(indicesPerElement), offsets, scatterIndices))

    _gatherPlanCache = (entitiesInDofVector, chunkSize, len(elements), plan, elements, scatterOffsetMap)
    return plan


def computeElementsInParallelForExplicit(
    elements: dict, Un1: DofVector, dU: DofVector, P: DofVector, timeStep: TimeStep
) -> tuple[DofVector, float]:
    """Evaluate the explicit element kernels across the available threads.

    Every chunk of elements works on buffers of its own: the gathered solution and increment, and
    a force buffer its elements write into, which reaches the shared scatter buffer in one indexed
    assignment once the chunk is done. Nothing shared is touched per element, which is what lets
    the loop scale with the GIL disabled; see :func:`_chunkedGatherPlan`.

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
        The assembled internal force vector, and the summed internal energy.
    """

    scatter_P = P.createScatterVector()
    time = timeStep.totalTime
    dT = timeStep.timeIncrement

    # One index plan serves both vectors only if they share the same entity mapping.
    if dU.entitiesInDofVector is not Un1.entitiesInDofVector:
        raise ValueError(
            "The solution and increment vectors carry different entity mappings, so a shared "
            "gather plan cannot be used for both."
        )

    # Plain ndarray views, so the gather bypasses the DofVector entity lookup.
    Un1_plain = Un1.asPlainArray()
    dU_plain = dU.asPlainArray()

    def compute_chunk(plannedChunk) -> float:
        chunkElements, flatIndices, offsets, scatterIndices = plannedChunk

        # One gather per chunk; the elements take views into these buffers.
        gatheredU = Un1_plain[flatIndices]
        gatheredDU = dU_plain[flatIndices]

        # The chunk's own force buffer: per-element views into the shared scatter buffer would make
        # all threads contend on its reference count under PYTHON_GIL=0.
        Pe = np.zeros(offsets[-1])

        chunk_psi = 0.0
        for position, element in enumerate(chunkElements):
            begin = offsets[position]
            end = offsets[position + 1]

            element.computeKernelsExplicit(Pe[begin:end], gatheredU[begin:end], gatheredDU[begin:end], time, dT)
            chunk_psi += element.computeInternalEnergy()

        # Each element owns its slot in the scatter buffer, so this assigns rather than accumulates.
        scatter_P[scatterIndices] = Pe

        return chunk_psi

    numThreads = getNumberOfWorkerThreads()

    # Many small chunks per thread for load balancing: inelastic elements (e.g. at a damage front) are
    # more expensive and adjacent in element order.
    chunk_size = (
        max(1, len(elements) // (numThreads * _chunksPerThread)) if numThreads > 1 else min(len(elements), 4000)
    )
    plan = _chunkedGatherPlan(elements, Un1.entitiesInDofVector, scatter_P.offsetMap, chunk_size)

    if numThreads == 1:
        # avoid ThreadPoolExecutor/task dispatch overhead when there is nothing to parallelize
        psi_total = sum(compute_chunk(plannedChunk) for plannedChunk in plan)
    else:
        executor = getThreadPool(numThreads)
        # map returns the chunk_psi from each worker
        psi_total = sum(executor.map(compute_chunk, plan))

    scatter_P.assembleInto(P)

    return P, psi_total
