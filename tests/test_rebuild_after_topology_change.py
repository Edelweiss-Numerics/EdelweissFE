#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""What a topology change makes the implicit solvers rebuild, and what they now build in parallel:

* the VIJ pattern of the DofManager is filled by the entities in chunks on the thread pool, and comes out bitwise the
  serial one;
* the DofManager locates its entities in the DofVector in chunks on the thread pool, in the order of the entities and as
  a single thread does; an entity that couples a field its nodes do not have is left out, and an entity that is faulty
  in any other way raises, for few entities as for many;
* the element result collectors of the field outputs gather the result arrays of the elements in chunks on the thread
  pool, in the order of the elements.

The tests choose their number of threads with the ``workerThreads`` fixture (``conftest.py``) and not with the
environment alone: what they check is the chunking and the order of the results, which has to hold with or without
free threading.
"""

import numpy as np
import pytest
from test_dofmanager_refresh import _buildTwoElementModel, _dofManager, _FakeConstraint

import edelweissfe.numerics.dofmanager as dofmanager
from edelweissfe.numerics.parallelizationutilities import mapChunksInParallel
from edelweissfe.utils.elementresultcollector import ElementResultCollector


def _countChunks(monkeypatch) -> list:
    """Make the DofManager note down into how many chunks each of its calls of ``mapChunksInParallel`` splits its
    entities: the list returned gets one number per call."""
    chunkCounts = []

    def inChunks(items, work, chunksPerThread=4):
        chunks = mapChunksInParallel(items, work, chunksPerThread)
        chunkCounts.append(len(chunks))
        return chunks

    monkeypatch.setattr(dofmanager, "mapChunksInParallel", inChunks)
    return chunkCounts


def test_the_vij_pattern_filled_on_the_thread_pool_is_the_serial_one_bitwise(monkeypatch, workerThreads):
    model, n = _buildTwoElementModel()
    constraint = _FakeConstraint([n[0], n[5]], ["displacement"])

    workerThreads(1)
    serial = _dofManager(model, [constraint])

    workerThreads(4)
    chunkCounts = _countChunks(monkeypatch)
    threaded = _dofManager(model, [constraint])

    assert chunkCounts and max(chunkCounts) > 1, "the entities were not split into several chunks"
    assert threaded.I.tobytes() == serial.I.tobytes()
    assert threaded.J.tobytes() == serial.J.tobytes()
    assert list(threaded.idcsOfHigherOrderEntitiesInVIJ.values()) == list(
        serial.idcsOfHigherOrderEntitiesInVIJ.values()
    )


class _StubEntity:
    """The face of an entity the DofManager locates: its nodes, the fields it couples on each of them, and the
    permutation of its dofs."""

    def __init__(self, nodes, fields, dofIndicesPermutation=None):
        self.nodes = nodes
        self.fields = fields
        self.dofIndicesPermutation = dofIndicesPermutation


def _dofManagerAndEntities(numEntities):
    """A DofManager of the two-element model, and entities that all couple the displacement on the four nodes of its
    first element."""
    model, n = _buildTwoElementModel()
    nodes = [n[0], n[1], n[4], n[5]]
    entities = [_StubEntity(nodes, [["displacement"] for _ in nodes]) for _ in range(numEntities)]
    return _dofManager(model, []), entities


def test_the_entities_located_on_the_thread_pool_are_the_serial_ones_in_order(monkeypatch, workerThreads):
    dofManager, entities = _dofManagerAndEntities(600)  # not one of the few that are located in the calling thread

    # The indices of the displacement of the entities' nodes, which is where an entity without a permutation is
    # located, and the orders the entities take their dofs in: as they are, reversed and rotated.
    unpermuted = np.concatenate(
        [dofManager.idcsOfFieldVariablesInDofVector[node.fields["displacement"]] for node in entities[0].nodes]
    )
    inOrder = np.arange(len(unpermuted))
    permutations = [None, inOrder[::-1], np.roll(inOrder, 3)]
    for i, entity in enumerate(entities):
        entity.dofIndicesPermutation = permutations[i % len(permutations)]

    workerThreads(1)
    serial = dofManager._locateNodeCouplingEntitiesInDofVector(entities)

    workerThreads(4)
    chunkCounts = _countChunks(monkeypatch)
    pooled = dofManager._locateNodeCouplingEntitiesInDofVector(entities)

    assert chunkCounts and max(chunkCounts) > 1, "the entities were not split into several chunks"
    assert list(serial) == entities and list(pooled) == entities, "not the entities, in the order they were given"
    for entity in entities:
        permutation = entity.dofIndicesPermutation
        assert np.array_equal(serial[entity], unpermuted if permutation is None else unpermuted[permutation])
        assert np.array_equal(pooled[entity], serial[entity])


@pytest.mark.parametrize("numThreads", [1, 4])
@pytest.mark.parametrize("numEntities", [3, 600])
def test_an_entity_coupling_a_field_its_nodes_do_not_have_is_left_out(numEntities, numThreads, workerThreads):
    dofManager, entities = _dofManagerAndEntities(numEntities)
    withoutLocation = entities[numEntities // 2]
    withoutLocation.fields = [["noSuchField"] for _ in withoutLocation.nodes]

    workerThreads(numThreads)
    located = dofManager._locateNodeCouplingEntitiesInDofVector(entities)

    assert list(located) == [entity for entity in entities if entity is not withoutLocation]


@pytest.mark.parametrize("numEntities", [3, 600])
def test_an_entity_that_lists_no_fields_for_its_nodes_raises_instead_of_being_left_out(numEntities, workerThreads):
    dofManager, entities = _dofManagerAndEntities(numEntities)
    entities[numEntities // 2].fields = []

    workerThreads(4)
    with pytest.raises(IndexError):
        dofManager._locateNodeCouplingEntitiesInDofVector(entities)


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
