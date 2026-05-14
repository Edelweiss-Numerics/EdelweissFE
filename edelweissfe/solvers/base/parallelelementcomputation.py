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

import numpy as np

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

    time = np.array([timeStep.stepTime, timeStep.totalTime])
    dT = timeStep.timeIncrement

    def computeElementsWorker(element: BaseElement):
        Pe = scatter_P[element]
        Ue = Un1[element]
        dUe = dU[element]
        Ke = K[element]
        element.computeYourself(Ke, Pe, Ue, dUe, time, dT)

    numThreads = getNumberOfThreads() if isFreeThreadingSupported() else 1

    with concurrent.futures.ThreadPoolExecutor(max_workers=numThreads) as executor:
        list(executor.map(computeElementsWorker, elements.values()))

    scatter_P.assembleInto(P)
    scatter_P.assembleInto(F, absolute=True)

    return P, K, F


def computeElementsInParallelForExplicit(
    elements: dict, Un1: DofVector, dU: DofVector, P: DofVector, timeStep: TimeStep
) -> DofVector:
    """
    Compute the elements in parallel for explicit analysis.

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
    timeStep : TimeStep
        The time step.

    Returns
    -------
    P : DofVector
        The internal force vector.
    """

    scatter_P = (
        P.createScatterVector()
    )  # make a scatter vector; which gives 1) contiguous memory access and 2) thread safety

    time = np.array([timeStep.stepTime, timeStep.totalTime])
    dT = timeStep.timeIncrement

    def computeElementsWorker(element: BaseElement):
        Pe = scatter_P[element]
        Ue = Un1[element]
        dUe = dU[element]
        element.computeYourselfExplicit(Pe, Ue, dUe, time, dT)

    numThreads = getNumberOfThreads() if isFreeThreadingSupported() else 1

    with concurrent.futures.ThreadPoolExecutor(max_workers=numThreads) as executor:
        list(executor.map(computeElementsWorker, elements.values()))

    scatter_P.assembleInto(P)

    return P
