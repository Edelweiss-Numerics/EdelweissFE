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
adding the scaled mass to it, NID's products with the mass matrix, in row blocks of it, and the passes over the
entries of the mass; and the loop over the elements that assembles the mass and the damping, in chunks of elements
(:func:`~edelweissfe.numerics.parallelizationutilities.mapChunksInParallel`).

Every slice computes its entries exactly as the operation on the whole array does, and the chunks of elements come
back in the order of the elements, so the result must be bitwise the serial one -- not merely close. The test models are far smaller than one slice or one
mass row block, so the solvers run them serially; here both are shrunk until the same live-refining
deck takes the threaded path. SuperLU solves the linear systems: it is serial and deterministic, so
the two runs can only differ through the operations under test.

The tests choose their number of threads with the ``workerThreads`` fixture (``conftest.py``) and not with the
environment alone: what they check is the chunking and the order of the results, which has to hold with or without
free threading.
"""

import threading
import time

import numpy as np
import pytest
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
from edelweissfe.numerics.parallelizationutilities import (
    forEachSliceInParallel,
    mapChunksInParallel,
)
from edelweissfe.solvers.nonlinearimplicitdynamic import NonlinearImplicitDynamic


def test_the_slices_cover_every_entry_once():
    visits = np.zeros(1000, dtype=int)

    def visit(s: slice):
        visits[s] += 1

    forEachSliceInParallel(visit, len(visits), sliceLength=64)

    assert np.all(visits == 1)


def test_the_slices_return_what_the_work_returns_in_the_order_of_the_slices(workerThreads):
    workerThreads(4)

    starts = forEachSliceInParallel(lambda s: s.start, 1000, sliceLength=64)

    assert starts == list(range(0, 1000, 64))


def test_the_chunks_cover_every_item_once_and_come_back_in_order(workerThreads):
    items = list(range(1000))
    workerThreads(4)

    chunks = mapChunksInParallel(items, lambda chunk: list(chunk))

    assert len(chunks) > 4, "the items were not split into several chunks per thread"
    assert [item for chunk in chunks for item in chunk] == items


def test_the_chunks_are_one_serial_chunk_for_one_thread(workerThreads):
    items = list(range(1000))
    workerThreads(1)

    assert mapChunksInParallel(items, lambda chunk: list(chunk)) == [items]


def test_the_first_exception_of_the_chunks_is_the_one_of_the_serial_loop(workerThreads):
    workerThreads(4)

    def work(chunk):
        for item in chunk:
            if item % 100 == 37:
                raise ValueError(item)

    with pytest.raises(ValueError) as error:
        mapChunksInParallel(list(range(1000)), work)

    assert error.value.args == (37,)


def _twoHundredChunks(work):
    """``work(item)`` for the first item of each of 200 chunks of 4 items (800 items, 4 threads, 50 chunks per thread),
    through ``mapChunksInParallel``: the first call gets 0, the next ones 4, 8, ..."""
    return mapChunksInParallel(list(range(800)), lambda chunk: work(chunk[0]), chunksPerThread=50)


def _twoHundredSlices(work):
    """``work(entry)`` for each of 200 entries, in a slice of its own, through ``forEachSliceInParallel``: the first
    call gets 0, the next ones 1, 2, ..."""
    return forEachSliceInParallel(lambda s: work(s.start), 200, sliceLength=1)


@pytest.mark.parametrize("runTwoHundredCalls", [_twoHundredChunks, _twoHundredSlices])
def test_a_raising_call_stops_the_unstarted_calls_and_waits_for_the_running_ones(runTwoHundredCalls, workerThreads):
    """Of 200 calls on four threads, call 0 raises while three other calls are running. Afterwards no call is still
    running (``list(pool.map(...))`` returned the exception while they were), and the calls that had not started were
    not made (waiting for every call, without cancelling the rest, would run all 200: about a second)."""
    workerThreads(4)
    started, finished = [], []
    othersStarted = threading.Event()

    def work(call):
        started.append(call)
        if call == 0:
            # Raise only once three other calls are running: they cannot be stopped, they have to be waited for.
            assert othersStarted.wait(timeout=10), "the pool did not start three other calls beside call 0"
            raise ValueError(call)
        if sum(1 for other in started if other != 0) >= 3:  # call 0 is waiting for this, and is not counted
            othersStarted.set()
        time.sleep(0.02)  # still busy when call 0 raises
        finished.append(call)

    with pytest.raises(ValueError):
        runTwoHundredCalls(work)

    assert (
        set(finished) == set(started) - {0} and len(finished) == len(started) - 1
    ), "a call was still running when the exception was raised"
    assert len(started) < 100, "the calls that had not started were made"


def test_the_mass_row_blocks_multiply_bitwise_like_the_whole_matrix():
    rng = np.random.default_rng(0)
    dense = rng.standard_normal((300, 300))
    dense[rng.random((300, 300)) > 0.05] = 0.0
    M = csr_matrix(dense)
    x = rng.standard_normal(300)

    blocks = NonlinearImplicitDynamic._rowBlocks(M, 64)

    assert all(np.shares_memory(block.data, M.data) for block in blocks), "a block copied the entries"
    assert np.concatenate([block @ x for block in blocks]).tobytes() == (M @ x).tobytes()


def test_the_threaded_path_reproduces_the_serial_one_bitwise(tmp_path, monkeypatch, workerThreads):
    deck = _deck(modifier=_REFINE_THE_FIXED_END_TWICE, nX=2, lX=4.0, steps=_CANTILEVER_STEP).replace(
        "*solver, solver=NID, name=theSolver", "*solver, solver=NID, name=theSolver\nlinsolver=superlu"
    )
    workerThreads(4)
    serialModel, _ = _run(tmp_path, "serial", deck)

    # Slices of 8 entries, and mass row blocks of 8 rows to match them.
    slicedLengths = []

    def inSlicesOfEight(work, length, sliceLength=None):
        slicedLengths.append(length)
        return forEachSliceInParallel(work, length, 8)

    for module in (nonlinearimplicitstatic, nonlinearimplicitdynamic):
        monkeypatch.setattr(module, "forEachSliceInParallel", inSlicesOfEight)
    monkeypatch.setattr(nonlinearimplicitdynamic, "_MASS_ROWS_PER_BLOCK", 8)
    threadedModel, _ = _run(tmp_path, "threaded", deck)

    assert _hangingSlaveNodes(threadedModel) > 0, "no hanging node; the MPC path was not exercised"
    assert slicedLengths and min(slicedLengths) > 8, "an operation was not split into several slices"
    assert threadedModel.time == serialModel.time
    for serial, threaded, name in zip(_finalState(serialModel), _finalState(threadedModel), "UVA"):
        assert threaded.tobytes() == serial.tobytes(), name


def test_the_mass_and_the_damping_assembled_on_the_thread_pool_are_the_serial_ones_bitwise(
    tmp_path, monkeypatch, workerThreads
):
    """Every assembly of the mass and the damping of the live-refining deck, once with one thread, and once with four
    threads, the elements in chunks and the passes over the entries of the mass in slices of 8."""
    deck = _deck(modifier=_REFINE_THE_FIXED_END_TWICE, nX=2, lX=4.0, steps=_CANTILEVER_STEP).replace(
        "*solver, solver=NID, name=theSolver", "*solver, solver=NID, name=theSolver\nlinsolver=superlu"
    )
    assembleMassAndDamping = NonlinearImplicitDynamic._assembleMassAndDamping

    def recordingAssemblies(assemblies):
        def assemble(self, model, dynamicDofs, dynamicFields):
            result = assembleMassAndDamping(self, model, dynamicDofs, dynamicFields)
            Mvij, damping, M, C, massByField = result
            assemblies.append((np.asarray(Mvij).tobytes(), damping, M.data.tobytes(), C.data.tobytes()))
            return result

        return assemble

    serialAssemblies, threadedAssemblies = [], []
    workerThreads(1)
    monkeypatch.setattr(NonlinearImplicitDynamic, "_assembleMassAndDamping", recordingAssemblies(serialAssemblies))
    _run(tmp_path, "serial", deck)

    chunkCounts = []

    def inChunks(items, work, chunksPerThread=4):
        chunks = mapChunksInParallel(items, work, chunksPerThread)
        chunkCounts.append(len(chunks))
        return chunks

    def inSlicesOfEight(work, length, sliceLength=None):
        return forEachSliceInParallel(work, length, 8)

    workerThreads(4)
    monkeypatch.setattr(nonlinearimplicitdynamic, "mapChunksInParallel", inChunks)
    monkeypatch.setattr(nonlinearimplicitdynamic, "forEachSliceInParallel", inSlicesOfEight)
    monkeypatch.setattr(nonlinearimplicitdynamic, "_MASS_ROWS_PER_BLOCK", 8)  # the slices of 8 take blocks of 8 rows
    monkeypatch.setattr(NonlinearImplicitDynamic, "_assembleMassAndDamping", recordingAssemblies(threadedAssemblies))
    _run(tmp_path, "threaded", deck)

    assert len(serialAssemblies) > 1, "the mass was assembled once; the refinement did not change it"
    assert chunkCounts and min(chunkCounts) > 1, "the elements were not split into several chunks"
    assert len(threadedAssemblies) == len(serialAssemblies)
    for serial, threaded in zip(serialAssemblies, threadedAssemblies):
        assert threaded[0] == serial[0], "the mass in the VIJ layout"
        assert threaded[2] == serial[2], "the mass as CSR"
        assert threaded[3] == serial[3], "the damping as CSR"
