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

import concurrent.futures
import itertools

from edelweissfe.elements.base.baseelement import BaseElement
from edelweissfe.numerics.dofmanager import DofVector, VIJSystemMatrix
from edelweissfe.numerics.parallelizationutilities import (
    getNumberOfThreads,
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

    def computeElementsWorker(element: BaseElement):
        Pe = scatter_P[element]
        Ue = Un1[element]
        dUe = dU[element]
        Ke = K[element]
        element.computeKernels(Ke, Pe, Ue, dUe, time, dT)

    numThreads = getNumberOfThreads() if isFreeThreadingSupported() else 1

    with concurrent.futures.ThreadPoolExecutor(max_workers=numThreads) as executor:
        list(executor.map(computeElementsWorker, elements.values()))

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


def computeElementsInParallelForExplicit(
    elements: dict, Un1: DofVector, dU: DofVector, P: DofVector, timeStep: TimeStep
) -> tuple[DofVector, float]:

    scatter_P = P.createScatterVector()
    time = timeStep.totalTime
    dT = timeStep.timeIncrement

    # Define the worker to process a CHUNK of elements, not just one.
    def compute_chunk(element_chunk) -> float:
        chunk_psi = 0.0
        for element in element_chunk:
            Pe = scatter_P[element]
            Ue = Un1[element]
            dUe = dU[element]

            element.computeKernelsExplicit(Pe, Ue, dUe, time, dT)
            chunk_psi += element.computeInternalEnergy()

        return chunk_psi

    numThreads = getNumberOfThreads() if isFreeThreadingSupported() else 1

    # Target ~1000 to 5000 elements per chunk depending on mesh size
    chunk_size = max(1, len(elements) // (numThreads * 4))
    chunks = chunked_iterable(elements.values(), chunk_size)

    psi_total = 0.0
    with concurrent.futures.ThreadPoolExecutor(max_workers=numThreads) as executor:
        # map returns the chunk_psi from each worker
        results = executor.map(compute_chunk, chunks)
        psi_total = sum(results)

    scatter_P.assembleInto(P)

    return P, psi_total
