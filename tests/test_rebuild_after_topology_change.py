#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""What a topology change makes the implicit solvers rebuild, and what they now build in parallel:

* the VIJ pattern of the DofManager is filled by the entities in chunks on the thread pool, and comes out bitwise the
  serial one;
* the element result collectors of the field outputs gather the result arrays of the elements in chunks on the thread
  pool, in the order of the elements.

The tests choose their number of threads with the ``workerThreads`` fixture (``conftest.py``) and not with the
environment alone: what they check is the chunking and the order of the results, which has to hold with or without
free threading.
"""

import numpy as np
from test_dofmanager_refresh import _buildTwoElementModel, _dofManager, _FakeConstraint

import edelweissfe.numerics.dofmanager as dofmanager
from edelweissfe.numerics.parallelizationutilities import mapChunksInParallel
from edelweissfe.utils.elementresultcollector import ElementResultCollector


def test_the_vij_pattern_filled_on_the_thread_pool_is_the_serial_one_bitwise(monkeypatch, workerThreads):
    model, n = _buildTwoElementModel()
    constraint = _FakeConstraint([n[0], n[5]], ["displacement"])

    workerThreads(1)
    serial = _dofManager(model, [constraint])

    chunkCounts = []

    def inChunks(items, work, chunksPerThread=4):
        chunks = mapChunksInParallel(items, work, chunksPerThread)
        chunkCounts.append(len(chunks))
        return chunks

    workerThreads(4)
    monkeypatch.setattr(dofmanager, "mapChunksInParallel", inChunks)
    threaded = _dofManager(model, [constraint])

    assert chunkCounts and max(chunkCounts) > 1, "the entities were not split into several chunks"
    assert threaded.I.tobytes() == serial.I.tobytes()
    assert threaded.J.tobytes() == serial.J.tobytes()
    assert list(threaded.idcsOfHigherOrderEntitiesInVIJ.values()) == list(
        serial.idcsOfHigherOrderEntitiesInVIJ.values()
    )


class _ElementWithResults:
    """The face of an element the result collector reads: persistent arrays of its results."""

    def __init__(self, number: int):
        self._results = {("sigma", qp): np.array([number, qp, number * 10.0 + qp]) for qp in range(3)}

    def getResultArray(self, result, quadraturePoint, getPersistentView=True):
        return self._results[(result, quadraturePoint)]


def test_the_result_collector_gathers_the_elements_in_order_and_sees_their_arrays_change(workerThreads):
    elements = [_ElementWithResults(i) for i in range(500)]

    workerThreads(1)
    serial = ElementResultCollector(elements, range(3), "sigma").getCurrentResults().copy()

    workerThreads(4)
    collector = ElementResultCollector(elements, range(3), "sigma")
    threaded = collector.getCurrentResults().copy()

    assert threaded.tobytes() == serial.tobytes()
    assert threaded.shape == (500, 3, 3)
    assert np.array_equal(threaded[:, 0, 0], np.arange(500))

    # a persistent view: the collector reads what the element writes into its array afterwards
    elements[250]._results[("sigma", 1)][2] = -1.0
    assert collector.getCurrentResults()[250, 1, 2] == -1.0
