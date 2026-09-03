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

import itertools

import numpy as np

from edelweissfe.numerics.dofmanager import DofVector, VIJSystemMatrix
from edelweissfe.numerics.parallelizationutilities import (
    getNumberOfThreads,
    getThreadPool,
    isFreeThreadingSupported,
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


def chunked_iterable(iterable, size):
    """Yield successive n-sized chunks from an iterable."""
    it = iter(iterable)
    while True:
        chunk = tuple(itertools.islice(it, size))
        if not chunk:
            break
        yield chunk


#: Single-entry cache of the per-chunk gather plan; see :func:`_chunkedGatherPlan`. One entry
#: suffices because a solver works on one element set at a time, and holding a reference to the
#: entity mapping it was built for keeps that mapping alive, so identity comparison against it is
#: sound (a freed dict could otherwise have its id reused by a different one).
_gatherPlanCache = None


def _chunkedGatherPlan(elements: dict, entitiesInDofVector: dict, chunkSize: int) -> list:
    """Build, or reuse, the flat gather index plan for one chunking of the elements.

    Each chunk gets the concatenation of its elements' DOF indices, plus the offsets at which
    each element's slice begins, so a worker can gather the whole chunk with one fancy-index and
    then hand out views.

    The plan is only valid for the element set and DOF layout it was built from. h-adaptivity
    rebuilds the DofManager on every topology change, which produces a fresh
    ``idcsOfHigherOrderEntitiesInDofVector`` dict, so identity of that mapping is what detects a
    stale plan. The element count and chunk size are compared as well: those would catch a
    rebuild that somehow preserved the mapping object, and a stale plan here would silently
    gather the wrong degrees of freedom rather than fail.

    Parameters
    ----------
    elements
        The elements to compute, in the order they will be chunked.
    entitiesInDofVector
        The entity-to-DOF-index mapping the plan is built against.
    chunkSize
        Number of elements per chunk.

    Returns
    -------
    list
        One ``(chunkElements, flatIndices, offsets)`` tuple per chunk.
    """

    global _gatherPlanCache

    if (
        _gatherPlanCache is not None
        and _gatherPlanCache[0] is entitiesInDofVector
        and _gatherPlanCache[1] == chunkSize
        and _gatherPlanCache[2] == len(elements)
    ):
        return _gatherPlanCache[3]

    plan = []
    for chunk in chunked_iterable(elements.values(), chunkSize):
        indicesPerElement = [entitiesInDofVector[element] for element in chunk]
        offsets = np.zeros(len(chunk) + 1, dtype=np.intp)
        np.cumsum([len(indices) for indices in indicesPerElement], out=offsets[1:])
        plan.append((chunk, np.concatenate(indicesPerElement), offsets))

    _gatherPlanCache = (entitiesInDofVector, chunkSize, len(elements), plan)
    return plan


def computeElementsInParallelForExplicit(
    elements: dict, Un1: DofVector, dU: DofVector, P: DofVector, timeStep: TimeStep
) -> tuple[DofVector, float]:

    scatter_P = P.createScatterVector()
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

    # Plain ndarray aliases: the gather below indexes them directly, bypassing the DofVector
    # entity lookup entirely for the hot path.
    Un1_plain = Un1.view(np.ndarray)
    dU_plain = dU.view(np.ndarray)

    def compute_chunk(plannedChunk) -> float:
        chunkElements, flatIndices, offsets = plannedChunk

        # One gather per chunk rather than two per element. The elements then take views into
        # these buffers, which allocate nothing.
        gatheredU = Un1_plain[flatIndices]
        gatheredDU = dU_plain[flatIndices]

        chunk_psi = 0.0
        for position, element in enumerate(chunkElements):
            begin = offsets[position]
            end = offsets[position + 1]

            element.computeKernelsExplicit(scatter_P[element], gatheredU[begin:end], gatheredDU[begin:end], time, dT)
            chunk_psi += element.computeInternalEnergy()

        return chunk_psi

    numThreads = getNumberOfThreads() if isFreeThreadingSupported() else 1

    # Target ~1000 to 5000 elements per chunk depending on mesh size
    chunk_size = max(1, len(elements) // (numThreads * 4)) if numThreads > 1 else min(len(elements), 4000)
    plan = _chunkedGatherPlan(elements, Un1.entitiesInDofVector, chunk_size)

    if numThreads == 1:
        # avoid ThreadPoolExecutor/task dispatch overhead when there is nothing to parallelize
        psi_total = sum(compute_chunk(plannedChunk) for plannedChunk in plan)
    else:
        executor = getThreadPool(numThreads)
        # map returns the chunk_psi from each worker
        psi_total = sum(executor.map(compute_chunk, plan))

    scatter_P.assembleInto(P)

    return P, psi_total
