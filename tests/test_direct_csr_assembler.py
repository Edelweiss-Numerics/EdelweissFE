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
entries), several threads, entries added to the pattern, rows longer than a 16-bit offset within the row reaches. The
integer trick (counts, then sums of entry indices) checks the threaded, atomic scatter exactly.
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


def test_a_matrix_that_outlives_its_assembly_keeps_only_the_pattern_alive():
    """A linear solver may keep the pattern arrays of the last matrix after the equation system was rebuilt (PARDISO
    does, to compare the next pattern with). They keep the pattern alive, not the map of the assembly that is gone:
    the offsets and positions of the direct path, the gather map of the VIJ path."""
    nDof = 25
    dofsOfEntity = _randomEntities(nDof, 4)
    rows, columns = _vijPairs(dofsOfEntity)

    for makeAssembly in (
        lambda: DirectCSRAssembler(SystemMatrixEntities.ofEntities(dofsOfEntity), nDof, 2),
        lambda: CSRGenerator(rows, columns, nDof),
    ):
        assembly = makeAssembly()
        indptr, indices = assembly.csrMatrix.indptr, assembly.csrMatrix.indices
        pattern = indptr.base
        patternBytes = indptr.nbytes + indices.nbytes
        assert pattern.memoryBytesKeptAlive > patternBytes

        del assembly
        assert pattern.memoryBytesKeptAlive == patternBytes
        np.testing.assert_array_equal(indptr[-1], len(indices))


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


# --- rows longer than a 16-bit offset within the row reaches


def _hubTable(nSprings):
    """``nSprings`` dense two-DOF entities that couple a hub DOF 0 with a DOF of their own (springs to a reference
    point), a listed entity that couples the hub with every spring DOF and each of them with itself (a rigid body's
    reference point), and dense two-DOF entities between neighbouring spring DOFs, away from the hub. The row of the
    hub has ``nSprings + 1`` entries. Built as the arrays of the table directly, to keep the entities cheap."""
    springDofs = np.arange(1, nSprings + 1, dtype=np.intc)
    neighbours = np.arange(1, nSprings, dtype=np.intc)

    dofsOfSprings = np.stack([np.zeros(nSprings, dtype=np.intc), springDofs], axis=1).ravel()
    dofsOfNeighbours = np.stack([neighbours, neighbours + 1], axis=1).ravel()
    dofsOfHub = np.concatenate([[0], springDofs]).astype(np.intc)
    # the listed entity: its local DOF 0 is the hub; entries (0, b) and (b, b) for every spring b
    localSprings = np.arange(1, nSprings + 1, dtype=np.intc)
    localRows = np.concatenate([np.zeros(nSprings, dtype=np.intc), localSprings])
    localColumns = np.concatenate([localSprings, localSprings])

    nDense = nSprings + len(neighbours)
    dofStart = np.zeros(nDense + 2, dtype=np.int64)
    dofStart[1 : nDense + 1] = 2 * np.arange(1, nDense + 1)
    dofStart[-1] = dofStart[-2] + len(dofsOfHub)
    localEntryStart = np.zeros(nDense + 2, dtype=np.int64)
    localEntryStart[-1] = len(localRows)
    table = SystemMatrixEntities(
        dofStart=dofStart,
        dofs=np.concatenate([dofsOfSprings, dofsOfNeighbours, dofsOfHub]).astype(np.intc),
        couplesAllItsDofs=np.concatenate([np.ones(nDense, dtype=np.uint8), [0]]).astype(np.uint8),
        localEntryStart=localEntryStart,
        localRows=localRows,
        localColumns=localColumns,
    )

    # the VIJ pairs of the same entities, in the same order of the values
    dense = np.concatenate([dofsOfSprings, dofsOfNeighbours]).reshape(-1, 2)
    I = np.concatenate([dense[:, [0, 1, 0, 1]].ravel(), dofsOfHub[localRows]])  # noqa: E741
    J = np.concatenate([dense[:, [0, 0, 1, 1]].ravel(), dofsOfHub[localColumns]])
    return table, nSprings + 1, I.astype(np.intc), J.astype(np.intc)


def _scatterTable(direct, table, values, numThreads):
    """Scatter the values of every block of a table, the entities dealt out to ``numThreads`` threads in turn."""
    from concurrent.futures import ThreadPoolExecutor

    nEntities = len(table.dofStart) - 1
    starts = direct.entryStart
    direct.beginAssembly()

    def work(tid):
        for e in range(tid, nEntities, numThreads):
            direct.scatterBlock(tid, e, values[starts[e] : starts[e + 1]])

    with ThreadPoolExecutor(numThreads) as pool:
        list(pool.map(work, range(numThreads)))
    return direct.csrMatrix


@pytest.mark.parametrize("numThreads", [1, 4])
def test_a_row_longer_than_16_bit_offsets_reach_is_assembled(numThreads):
    """The row of a hub -- a reference point coupled with every one of its slave nodes -- grows with the model, not
    with the stencil: here it has 70,001 entries, past what a 16-bit offset within the row reaches. The blocks with a
    DOF in it, dense or listed, are scattered value by value to their positions; the others row by row. The integer
    trick checks the result exactly against the gather of the same entities."""
    table, nDof, I, J = _hubTable(70_000)  # noqa: E741
    generator = CSRGenerator(I, J, nDof)
    direct = DirectCSRAssembler(table, nDof, numThreads)
    assert generator.csrMatrix.indptr[1] - generator.csrMatrix.indptr[0] == 70_001

    np.testing.assert_array_equal(direct.entryStart[-1], len(I))
    for values in (np.ones(len(I)), np.arange(len(I), dtype=float)):
        scattered = _scatterTable(direct, table, values, numThreads)
        gathered = generator.updateInPlace(values)
        np.testing.assert_array_equal(scattered.indptr, gathered.indptr)
        np.testing.assert_array_equal(scattered.indices, gathered.indices)
        np.testing.assert_array_equal(scattered.data, gathered.data)


def test_a_row_made_long_by_the_entries_added_to_the_pattern_is_assembled():
    """The fill-in of the multi-point-constraint condensation can make a row long, too: a dense block whose value
    lies past the 16-bit reach of a row only once the entries are added is scattered to its position."""
    nDof = 70_002
    entity = _DenseEntity(2)
    dofsOfEntity = {entity: np.array([0, nDof - 1], dtype=np.intc)}

    def fillIn(pattern):
        # the hub row 0 gets every column between the two DOFs of the entity
        columns = np.arange(1, nDof - 1, dtype=np.intc)
        indptr = np.zeros(nDof + 1, dtype=np.intc)
        indptr[1:] = len(columns)
        return csr_matrix((np.zeros(len(columns)), columns, indptr), shape=pattern.shape)

    I, J = _vijPairs(dofsOfEntity)  # noqa: E741
    generator = CSRGenerator(I, J, nDof)
    generator.addToPattern(fillIn(generator.csrMatrix))
    direct = DirectCSRAssembler(SystemMatrixEntities.ofEntities(dofsOfEntity), nDof, 2, fillIn)

    values = np.arange(4, dtype=float) + 1.0
    direct.beginAssembly()
    direct.scatterBlock(0, 0, values)
    gathered = generator.updateInPlace(values)
    np.testing.assert_array_equal(direct.csrMatrix.indices, gathered.indices)
    np.testing.assert_array_equal(direct.csrMatrix.data, gathered.data)
    assert direct.csrMatrix[0, nDof - 1] == 3.0


# --- the input of the public entry points, which the compiled code does not check


def test_the_generator_refuses_pairs_that_do_not_fit_the_matrix():
    with pytest.raises(ValueError, match="differ in length"):
        CSRGenerator(np.array([0, 1]), np.array([0]), 2)
    with pytest.raises(IndexError, match="row"):
        CSRGenerator(np.array([0, 2]), np.array([0, 1]), 2)
    with pytest.raises(IndexError, match="column"):
        CSRGenerator(np.array([0, 1]), np.array([-1, 1]), 2)
    with pytest.raises(ValueError, match="negative"):
        CSRGenerator(np.array([0]), np.array([0]), -1)

    generator = CSRGenerator(np.array([0, 1]), np.array([0, 1]), 2)
    with pytest.raises(ValueError, match="pairs"):
        generator.updateInPlace(np.ones(3))


def test_the_generator_refuses_strided_values():
    """The C++ core reads the values one after the other: a strided view, every second value of a longer array, would
    be read as the first values of that array, silently."""
    generator = CSRGenerator(np.array([0, 1]), np.array([0, 1]), 2)
    values = np.array([1.0, -1.0, 2.0, -2.0])
    with pytest.raises(ValueError, match="contiguous"):
        generator.updateInPlace(values[::2])
    with pytest.raises(ValueError, match="contiguous"):
        generator.updateCSR(values[::2])
    np.testing.assert_array_equal(generator.updateCSR(values[::2].copy()).toarray(), [[1.0, 0.0], [0.0, 2.0]])


@pytest.mark.parametrize(
    "dofStart, couplesAllItsDofs, localEntryStart, message",
    [
        pytest.param([0, 2, 1, 3], [1, 1, 1], [0, 0, 0, 0], "entity 1 end before they begin", id="dofStart-decreases"),
        pytest.param([1, 2, 3], [1, 1], [0, 0, 0], "begin at 0", id="dofStart-begins-at-1"),
        pytest.param([0, 1, 2, 3], [0, 0, 0], [0, 2, 1, 3], "entity 1 end before they begin", id="entries-decrease"),
        pytest.param([0, 1, 2, 3], [0, 0, 0], [1, 1, 2, 3], "begin at 0", id="entries-begin-at-1"),
    ],
)
def test_a_table_whose_lists_do_not_lie_one_after_the_other_is_refused(
    dofStart, couplesAllItsDofs, localEntryStart, message
):
    """The DOF lists of the entities, and their listed entries, lie one after the other in the table: the starts begin
    at 0 and never decrease. The C++ passes size their arrays by the last start and fill them entity by entity; a
    start that decreases gives an entity a negative length, and with it they can write past the end of an array (these
    tables only give a wrong pattern, so that the test fails cleanly without the check)."""
    nListed = localEntryStart[-1]
    table = SystemMatrixEntities(
        dofStart=np.array(dofStart, dtype=np.int64),
        dofs=np.arange(dofStart[-1], dtype=np.intc),
        couplesAllItsDofs=np.array(couplesAllItsDofs, dtype=np.uint8),
        localEntryStart=np.array(localEntryStart, dtype=np.int64),
        localRows=np.zeros(nListed, dtype=np.intc),
        localColumns=np.zeros(nListed, dtype=np.intc),
    )
    with pytest.raises(ValueError, match=message):
        csrPatternOfEntities(table, 3)
    with pytest.raises(ValueError, match=message):
        DirectCSRAssembler(table, 3)


def test_entries_to_add_are_sorted_csr_rows():
    generator = CSRGenerator(np.array([0, 1]), np.array([0, 1]), 3)
    unsorted = csr_matrix((np.zeros(2), np.array([2, 1]), np.array([0, 2, 2, 2])), shape=(3, 3))
    with pytest.raises(ValueError, match="sorted"):
        generator.addToPattern(unsorted)
    with pytest.raises(ValueError, match="CSR"):
        generator.addToPattern(unsorted.tocoo())


@pytest.mark.parametrize("column", [3, -1])
def test_entries_to_add_lie_inside_the_matrix(column):
    """scipy builds a CSR matrix with a column outside its shape without complaint, and reports its rows as sorted.
    Merged into the pattern, such a column would be read out of bounds by every later use of the matrix (a product,
    the linear solver). Both paths refuse it, and keep their pattern."""
    outside = csr_matrix((np.zeros(1), np.array([column]), np.array([0, 0, 1, 1])), shape=(3, 3))

    generator = CSRGenerator(np.array([0, 1]), np.array([0, 1]), 3)
    matrix = generator.csrMatrix
    with pytest.raises(IndexError, match="outside"):
        generator.addToPattern(outside)
    assert generator.csrMatrix is matrix
    np.testing.assert_array_equal(matrix.indices, [0, 1])

    dense = _DenseEntity(2)
    entities = SystemMatrixEntities.ofEntities({dense: np.array([0, 1], dtype=np.intc)})
    with pytest.raises(IndexError, match="outside"):
        DirectCSRAssembler(entities, 3, 1, lambda pattern: outside)


def test_the_scatter_refuses_a_block_that_is_not_the_entitys():
    dense = _DenseEntity(2)
    direct = DirectCSRAssembler(SystemMatrixEntities.ofEntities({dense: np.array([0, 1], dtype=np.intc)}), 2, 2)
    direct.beginAssembly()
    with pytest.raises(IndexError, match="thread"):
        direct.scatterBlock(2, 0, np.ones(4))
    with pytest.raises(IndexError, match="entity"):
        direct.scatterBlock(0, 1, np.ones(4))
    with pytest.raises(ValueError, match="values given"):
        direct.scatterBlock(0, 0, np.ones(3))
    np.testing.assert_array_equal(direct.csrMatrix.data, 0.0)
    with pytest.raises(ValueError, match="negative"):
        DirectCSRAssembler(SystemMatrixEntities.ofEntities({}), -1)


def test_the_matrix_of_an_assembly_keeps_the_arrays_the_assembly_writes_into():
    """The direct assembler scatters into the data array of its matrix through a raw pointer, which it holds on to,
    and both assemblies write into their matrix in place: neither the matrix of an assembly nor one of its arrays can
    be replaced, while the values can be changed in place."""
    dense = _DenseEntity(2)
    dofsOfEntity = {dense: np.array([0, 1], dtype=np.intc)}
    rows, columns = _vijPairs(dofsOfEntity)

    for assembly in (
        DirectCSRAssembler(SystemMatrixEntities.ofEntities(dofsOfEntity), 2),
        CSRGenerator(rows, columns, 2),
    ):
        with pytest.raises(AttributeError):
            assembly.csrMatrix = None

        matrix = assembly.csrMatrix
        with pytest.raises(RuntimeError, match="reassign 'data'"):
            matrix.data = np.zeros_like(matrix.data)
        with pytest.raises(RuntimeError, match="reassign 'indices'"):
            matrix.indices = matrix.indices.copy()
        with pytest.raises(RuntimeError, match="reassign 'indptr'"):
            matrix.indptr = matrix.indptr.copy()

        data = matrix.data
        matrix.data += 1.0
        assert matrix.data is data
        np.testing.assert_array_equal(assembly.csrMatrix.data, 1.0)
