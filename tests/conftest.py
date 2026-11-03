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
"""Fixtures shared by the tests of this directory."""

import pytest

from edelweissfe.numerics import parallelizationutilities


@pytest.fixture
def workerThreads(monkeypatch):
    """Choose the number of worker threads of the pool helpers, whatever the GIL of this process is.

    Without free threading :func:`~edelweissfe.numerics.parallelizationutilities.getNumberOfWorkerThreads` is 1 however
    many threads ``OMP_NUM_THREADS`` asks for, and the helpers then run serially. A test of the chunking, or of the
    order of the results, needs the pool and not a speed-up, so it calls ``workerThreads(numThreads)``: 1 is the serial
    arm of a comparison, more is the threaded arm.

    The helpers read ``getNumberOfWorkerThreads`` as a name of their own module, which is what is replaced here. A
    module that imported the name has its own copy, which stays as it is: on a free-threaded interpreter it reads the
    environment variable, which is set as well.
    """

    def choose(numThreads: int):
        monkeypatch.setenv("OMP_NUM_THREADS", str(numThreads))
        monkeypatch.setattr(parallelizationutilities, "getNumberOfWorkerThreads", lambda: numThreads)

    return choose
