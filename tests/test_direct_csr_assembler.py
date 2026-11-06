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
"""The direct CSR assembly (:class:`~edelweissfe.numerics.csrgeneratorv2.DirectCSRAssembler`) against the VIJ path's
:class:`~edelweissfe.numerics.csrgeneratorv2.CSRGenerator`, built independently from the same entities: the pattern
the direct path derives row by row equals the one sort-and-sum derives from the VIJ pairs, and the blocks it scatters
sum to the matrix the generator gathers -- dense blocks, listed entries (including a list with exactly nDof**2
entries), several threads, entries added to the pattern. The integer trick (counts, then sums of entry indices) checks
the threaded, atomic scatter exactly.
"""

import numpy as np
import pytest
from scipy.sparse import csr_matrix

from edelweissfe.numerics.csrgeneratorv2 import (
    CSRGenerator,
    DirectCSRAssembler,
    csrPatternOfEntities,
)
from edelweissfe.numerics.systemmatrixcontributor import (
    SystemMatrixContributorBase,
    SystemMatrixEntities,
    countEntriesOfBlock,
)

# --- the pattern the direct path derives row by row, against the one the VIJ path derives by sort-and-sum


class _DenseEntity(SystemMatrixContributorBase):
    def __init__(self, nDof):
        super().__init__(nDof)


class _ListedEntity(SystemMatrixContributorBase):
    """An entity that lists its entries: random local (row, column) pairs, duplicates included, or the given ones.
    Overriding :meth:`localEntriesOfBlock` is all it does to declare its layout."""

    def __init__(self, nDof, localRows, localColumns):
        super().__init__(nDof)
        self._entries = (np.asarray(localRows, dtype=np.intc), np.asarray(localColumns, dtype=np.intc))

    def localEntriesOfBlock(self):
        return self._entries


def test_the_layout_of_a_block_follows_from_the_override_of_its_entries():
    assert _DenseEntity(3).couplesAllItsDofs
    assert not _ListedEntity(3, [0], [1]).couplesAllItsDofs


def test_a_second_declaration_of_the_layout_is_refused():
    """The direct path reads the flag, the VIJ path the list: an entity that declared both could make them disagree."""
    with pytest.raises(TypeError, match="couplesAllItsDofs"):

        class _DeclaresItsLayout(_ListedEntity):
            couplesAllItsDofs = True

    with pytest.raises(TypeError, match="couplesAllItsDofs"):

        class _DeclaresItsLayoutAsAProperty(SystemMatrixContributorBase):
            @property
            def couplesAllItsDofs(self):
                return True


def _vijPairs(dofsOfEntity):
    rows, columns = [np.zeros(0, dtype=np.intc)], [np.zeros(0, dtype=np.intc)]
    for entity, dofs in dofsOfEntity.items():
        localRows, localColumns = entity.localEntriesOfBlock()
        rows.append(dofs[localRows])
        columns.append(dofs[localColumns])
    return np.concatenate(rows), np.concatenate(columns)


def _randomEntities(nDof, seed):
    """Dense entities, entities with random listed entries, and one listing exactly nDof**2 entries in an order that
    is not column-major (a one-slave rigid body contact's count); DOF nDof - 1 belongs to no entity."""
    rng = np.random.default_rng(seed)
    dofsOfEntity = {}
    for _ in range(30):
        dofsOfEntity[_DenseEntity(4)] = rng.choice(nDof - 1, size=4, replace=False).astype(np.intc)
    for _ in range(10):
        n = int(rng.integers(2, 7))
        nEntries = int(rng.integers(1, 2 * n))
        entity = _ListedEntity(n, rng.integers(0, n, nEntries), rng.integers(0, n, nEntries))
        dofsOfEntity[entity] = rng.choice(nDof - 1, size=n, replace=False).astype(np.intc)
    order = rng.permutation(9)
    squareCount = _ListedEntity(3, np.repeat(np.arange(3), 3)[order], np.tile(np.arange(3), 3)[order])
    dofsOfEntity[squareCount] = rng.choice(nDof - 1, size=3, replace=False).astype(np.intc)
    assert countEntriesOfBlock(squareCount) == squareCount.nDof**2
    return dofsOfEntity


@pytest.mark.parametrize("numThreads", [1, 4])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_the_row_wise_pattern_is_the_sort_and_sum_pattern(numThreads, seed):
    nDof = 25
    dofsOfEntity = _randomEntities(nDof, seed)
    I, J = _vijPairs(dofsOfEntity)  # noqa: E741
    generator = CSRGenerator(I, J, nDof)

    indptr, indices = csrPatternOfEntities(SystemMatrixEntities.ofEntities(dofsOfEntity), nDof, numThreads)

    np.testing.assert_array_equal(indptr, generator.csrMatrix.indptr)
    np.testing.assert_array_equal(indices, generator.csrMatrix.indices)
    assert indptr[-1] == indptr[-2], "the DOF of no entity has an empty row"


def test_the_pattern_of_no_entities_is_empty():
    indptr, indices = csrPatternOfEntities(SystemMatrixEntities.ofEntities({}), 3, 2)
    np.testing.assert_array_equal(indptr, [0, 0, 0, 0])
    assert indices.shape == (0,)


def test_a_listed_entry_outside_the_entity_is_refused():
    entity = _ListedEntity(2, [0, 2], [1, 0])
    with pytest.raises(IndexError):
        csrPatternOfEntities(SystemMatrixEntities.ofEntities({entity: np.array([0, 1], dtype=np.intc)}), 2)


def test_entries_are_added_to_the_sort_and_sum_pattern_by_the_same_merge():
    """The fill-in of the multi-point-constraint condensation enters the pattern through one merge
    (``CSRPattern::addEntries``), shared by both derivations; on the VIJ path the new entries gather nothing."""
    from scipy.sparse import csr_matrix

    nDof = 25
    dofsOfEntity = _randomEntities(nDof, 7)
    I, J = _vijPairs(dofsOfEntity)  # noqa: E741
    generator = CSRGenerator(I, J, nDof)
    pattern = generator.csrMatrix.copy()
    pattern.data[:] = 1.0
    fillIn = csr_matrix(np.ones((nDof, nDof))) - pattern
    fillIn.eliminate_zeros()
    fillIn.sort_indices()

    generator.addToPattern(fillIn)
    V = np.arange(len(I), dtype=float)
    assembled = generator.updateInPlace(V).toarray()

    assert generator.csrMatrix.nnz == nDof * nDof
    reference = np.zeros((nDof, nDof))
    np.add.at(reference, (I, J), V)
    np.testing.assert_array_equal(assembled, reference)


# --- the scatter, against the gather of the same entities


def _scatterAll(direct, dofsOfEntity, valuesOf, numThreads):
    """Scatter the block of every entity, the entities dealt out to ``numThreads`` threads in turn."""
    from concurrent.futures import ThreadPoolExecutor

    entities = list(dofsOfEntity)
    direct.beginAssembly()

    def work(tid):
        for e in range(tid, len(entities), numThreads):
            direct.scatterBlock(tid, e, valuesOf(entities[e]))

    with ThreadPoolExecutor(numThreads) as pool:
        list(pool.map(work, range(numThreads)))
    return direct.csrMatrix


def _vijValues(dofsOfEntity, valuesOf):
    return np.concatenate([np.zeros(0)] + [valuesOf(entity) for entity in dofsOfEntity])


@pytest.mark.parametrize("numThreads", [1, 4])
@pytest.mark.parametrize("seed", [0, 1])
def test_the_scatter_sums_the_blocks_as_the_gather_does(numThreads, seed):
    """The integer trick: every value 1 makes each CSR entry the number of its contributions, every value its own
    index in the VIJ layout the sum of these indices. Both are integers, exact in any summation order, so the threaded
    atomic scatter matches the gather bit for bit."""
    nDof = 25
    dofsOfEntity = _randomEntities(nDof, seed)
    I, J = _vijPairs(dofsOfEntity)  # noqa: E741
    generator = CSRGenerator(I, J, nDof)
    direct = DirectCSRAssembler(SystemMatrixEntities.ofEntities(dofsOfEntity), nDof, numThreads)
    starts = dict(zip(dofsOfEntity, direct.entryStart))

    np.testing.assert_array_equal(direct.entryStart[-1], len(I))
    for valuesOf in (
        lambda entity: np.ones(countEntriesOfBlock(entity)),
        lambda entity: np.arange(countEntriesOfBlock(entity), dtype=float) + starts[entity],
    ):
        scattered = _scatterAll(direct, dofsOfEntity, valuesOf, numThreads)
        gathered = generator.updateInPlace(_vijValues(dofsOfEntity, valuesOf))
        np.testing.assert_array_equal(scattered.indptr, gathered.indptr)
        np.testing.assert_array_equal(scattered.indices, gathered.indices)
        np.testing.assert_array_equal(scattered.data, gathered.data)


def test_a_second_assembly_starts_from_zero():
    nDof = 25
    dofsOfEntity = _randomEntities(nDof, 3)
    direct = DirectCSRAssembler(SystemMatrixEntities.ofEntities(dofsOfEntity), nDof, 2)

    first = _scatterAll(direct, dofsOfEntity, lambda entity: np.ones(countEntriesOfBlock(entity)), 2).data.copy()
    second = _scatterAll(direct, dofsOfEntity, lambda entity: np.ones(countEntriesOfBlock(entity)), 2).data
    np.testing.assert_array_equal(second, first)


def test_entries_added_to_the_pattern_are_stored_zeros_on_both_paths():
    """``patternFillIn`` adds the same entries to the direct pattern as ``addToPattern`` to the generator's; they are
    assembled as zeros, and the scatter places every other value as before."""
    nDof = 25
    dofsOfEntity = _randomEntities(nDof, 5)
    I, J = _vijPairs(dofsOfEntity)  # noqa: E741

    def fillIn(pattern):
        full = csr_matrix(np.ones(pattern.shape))
        existing = pattern.copy()
        existing.data[:] = 1.0
        entries = full - existing
        entries.eliminate_zeros()
        entries.sort_indices()
        return entries

    generator = CSRGenerator(I, J, nDof)
    generator.addToPattern(fillIn(generator.csrMatrix))
    direct = DirectCSRAssembler(SystemMatrixEntities.ofEntities(dofsOfEntity), nDof, 2, fillIn)

    def valuesOf(entity):
        return np.full(countEntriesOfBlock(entity), 0.5)

    scattered = _scatterAll(direct, dofsOfEntity, valuesOf, 2)
    gathered = generator.updateInPlace(_vijValues(dofsOfEntity, valuesOf))
    assert scattered.nnz == nDof * nDof
    np.testing.assert_array_equal(scattered.indices, gathered.indices)
    np.testing.assert_array_equal(scattered.data, gathered.data)


def test_a_listed_block_with_nDof_squared_entries_is_scattered_entry_by_entry():
    """A list of exactly nDof**2 entries in an order that is not column-major: scattered as a dense block, its values
    would land on the wrong rows and columns. The entity declares nothing but the list, as an entity migrated from
    ``initializeVIJContribution`` by renaming it does."""
    order = np.array([4, 0, 8, 2, 6, 1, 3, 7, 5])
    entity = _ListedEntity(3, np.repeat(np.arange(3), 3)[order], np.tile(np.arange(3), 3)[order])
    dofs = np.array([2, 0, 1], dtype=np.intc)
    direct = DirectCSRAssembler(SystemMatrixEntities.ofEntities({entity: dofs}), 3)

    values = np.arange(9, dtype=float)
    direct.beginAssembly()
    direct.scatterBlock(0, 0, values)

    reference = np.zeros((3, 3))
    localRows, localColumns = entity.localEntriesOfBlock()
    np.add.at(reference, (dofs[localRows], dofs[localColumns]), values)
    np.testing.assert_array_equal(direct.csrMatrix.toarray(), reference)


def test_an_entity_without_entries_scatters_nothing():
    dense = _DenseEntity(2)
    empty = _ListedEntity(2, [], [])
    direct = DirectCSRAssembler(
        SystemMatrixEntities.ofEntities({dense: np.array([0, 1], dtype=np.intc), empty: np.array([1, 2])}), 3
    )
    direct.beginAssembly()
    direct.scatterBlock(0, 0, np.ones(4))
    direct.scatterBlock(0, 1, np.zeros(0))
    np.testing.assert_array_equal(direct.csrMatrix.toarray(), [[1, 1, 0], [1, 1, 0], [0, 0, 0]])
