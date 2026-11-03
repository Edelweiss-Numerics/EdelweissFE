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
"""``FEModel.advanceToTime`` lets every element accept its computed state, once.

Explicit dynamics does that on every increment, so a model with fewer elements than threads does not pay for handing
its few elements to the thread pool: the calling thread accepts them. A model with many more elements than threads
hands them to the pool in chunks (:func:`~edelweissfe.numerics.parallelizationutilities.mapChunksInParallel`).
"""

import threading

import pytest

import edelweissfe.models.femodel as femodel


class _StubElement:
    """What ``FEModel.advanceToTime`` asks of an element: to accept its last state. This one notes down who accepted."""

    def __init__(self, number: int, acceptances: list):
        self.number = number
        self.acceptances = acceptances

    def acceptLastState(self):
        self.acceptances.append((self.number, threading.get_ident()))


def _modelOfStubElements(numElements: int, acceptances: list) -> femodel.FEModel:
    model = femodel.FEModel(2)
    for number in range(numElements):
        model.elements[number] = _StubElement(number, acceptances)
    return model


@pytest.fixture
def fourThreads(workerThreads, monkeypatch):
    """Four worker threads for the pool helper, and for the model's own comparison with its number of elements."""
    workerThreads(4)
    # The model imported getNumberOfWorkerThreads by name: its copy is not the one workerThreads replaces.
    monkeypatch.setattr(femodel, "getNumberOfWorkerThreads", lambda: 4)


@pytest.mark.usefixtures("fourThreads")
def test_a_model_with_fewer_elements_than_threads_accepts_every_state_once_in_the_calling_thread():
    acceptances = []

    _modelOfStubElements(3, acceptances).advanceToTime(1.0)

    assert [number for number, _ in acceptances] == [0, 1, 2]
    assert {thread for _, thread in acceptances} == {threading.get_ident()}


@pytest.mark.usefixtures("fourThreads")
def test_a_model_with_many_more_elements_than_threads_accepts_every_state_once_on_the_pool():
    acceptances = []

    _modelOfStubElements(500, acceptances).advanceToTime(1.0)

    assert sorted(number for number, _ in acceptances) == list(range(500))
    assert threading.get_ident() not in {thread for _, thread in acceptances}, "the calling thread did the work"
