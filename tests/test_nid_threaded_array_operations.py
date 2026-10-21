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
#  ---------------------------------------------------------------------
"""The array operations the implicit solvers run on the thread pool, slice by slice
(:func:`~edelweissfe.numerics.parallelizationutilities.forEachSliceInParallel`): zeroing the tangent,
adding the scaled mass to it, and NID's products with the mass matrix, in row blocks of it.

Every slice computes its entries exactly as the operation on the whole array does, so the result must
be bitwise the serial one -- not merely close. The test models are far smaller than one slice or one
mass row block, so the solvers run them serially; here both are shrunk until the same live-refining
deck takes the threaded path. SuperLU solves the linear systems: it is serial and deterministic, so
the two runs can only differ through the operations under test.
"""

import numpy as np
from scipy.sparse import csr_matrix
from test_nid_amr import (
    _CANTILEVER_STEP,
    _REFINE_THE_FIXED_END_TWICE,
    _deck,
    _finalState,
    _hangingSlaveNodes,
    _run,
)

import edelweissfe.solvers.nonlinearimplicitdynamic as nonlinearimplicitdynamic
import edelweissfe.solvers.nonlinearimplicitstatic as nonlinearimplicitstatic
from edelweissfe.numerics.parallelizationutilities import forEachSliceInParallel
from edelweissfe.solvers.nonlinearimplicitdynamic import NonlinearImplicitDynamic


def test_the_slices_cover_every_entry_once():
    visits = np.zeros(1000, dtype=int)

    def visit(s: slice):
        visits[s] += 1

    forEachSliceInParallel(visit, len(visits), sliceLength=64)

    assert np.all(visits == 1)


def test_the_mass_row_blocks_multiply_bitwise_like_the_whole_matrix():
    rng = np.random.default_rng(0)
    dense = rng.standard_normal((300, 300))
    dense[rng.random((300, 300)) > 0.05] = 0.0
    M = csr_matrix(dense)
    x = rng.standard_normal(300)

    blocks = NonlinearImplicitDynamic._rowBlocks(M, 64)

    assert all(np.shares_memory(block.data, M.data) for block in blocks), "a block copied the entries"
    assert np.concatenate([block @ x for block in blocks]).tobytes() == (M @ x).tobytes()


def test_the_threaded_path_reproduces_the_serial_one_bitwise(tmp_path, monkeypatch):
    deck = _deck(modifier=_REFINE_THE_FIXED_END_TWICE, nX=2, lX=4.0, steps=_CANTILEVER_STEP).replace(
        "*solver, solver=NID, name=theSolver", "*solver, solver=NID, name=theSolver\nlinsolver=superlu"
    )
    monkeypatch.setenv("OMP_NUM_THREADS", "4")
    serialModel, _ = _run(tmp_path, "serial", deck)

    # Slices of 8 entries, and mass row blocks of 8 rows to match them.
    slicedLengths = []

    def inSlicesOfEight(work, length, sliceLength=None):
        slicedLengths.append(length)
        forEachSliceInParallel(work, length, 8)

    for module in (nonlinearimplicitstatic, nonlinearimplicitdynamic):
        monkeypatch.setattr(module, "forEachSliceInParallel", inSlicesOfEight)
    monkeypatch.setattr(nonlinearimplicitdynamic, "_MASS_ROWS_PER_BLOCK", 8)
    threadedModel, _ = _run(tmp_path, "threaded", deck)

    assert _hangingSlaveNodes(threadedModel) > 0, "no hanging node; the MPC path was not exercised"
    assert slicedLengths and min(slicedLengths) > 8, "an operation was not split into several slices"
    assert threadedModel.time == serialModel.time
    for serial, threaded, name in zip(_finalState(serialModel), _finalState(threadedModel), "UVA"):
        assert threaded.tobytes() == serial.tobytes(), name
