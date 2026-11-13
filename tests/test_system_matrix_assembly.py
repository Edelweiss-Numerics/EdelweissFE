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
"""The two assemblies of the system matrix (:mod:`edelweissfe.numerics.systemmatrixassembly`) assemble the same
matrix: from dense blocks and from blocks with entries of their own, from several worker threads, with blocks scaled
by a factor, with whole matrices added during the assembly, and after an abandoned assembly; a matrix copied onto
their sparsity pattern holds its values there; the constraints with entries of their own list them in their local
numbering; and the worker loop deals out its chunks as documented."""

from types import SimpleNamespace

import numpy as np
import pytest
from scipy.sparse import csr_matrix

from edelweissfe.constraints.base.rigidbodycontactstiffness import (
    rigidBodyContactLocalEntries,
)
from edelweissfe.constraints.nodetodeformablesurfacepenalty import (
    Constraint as NodeToDeformableSurfacePenalty,
)
from edelweissfe.constraints.nodetodiscreterigidbodypenalty import (
    Constraint as NodeToDiscreteRigidBodyPenalty,
)
from edelweissfe.constraints.rigidbody import Constraint as RigidBody
from edelweissfe.constraints.surfacetodeformablesurfacepenalty import (
    Constraint as SurfaceToDeformableSurfacePenalty,
)
from edelweissfe.constraints.surfacetodiscreterigidbodypenalty import (
    Constraint as SurfaceToDiscreteRigidBodyPenalty,
)
from edelweissfe.numerics.parallelizationutilities import mapChunksOnWorkers
from edelweissfe.numerics.systemmatrixassembly import (
    DirectCSRSystemMatrixAssembly,
    VIJSystemMatrixAssembly,
)
from edelweissfe.numerics.systemmatrixcontributor import (
    SystemMatrixContributorBase,
    countEntriesOfBlock,
)
from edelweissfe.numerics.vijsystemmatrix import VIJSystemMatrix


class _DenseEntity(SystemMatrixContributorBase):
    """An element-like entity: a dense block on its DOFs."""

    def __init__(self, dofs):
        super().__init__(len(dofs))
        self.dofs = np.asarray(dofs, dtype=np.intc)


class _SparseEntity(SystemMatrixContributorBase):
    """A constraint-like entity on three DOFs that couples only the first and the last, and the middle one with
    itself."""

    def __init__(self, dofs):
        super().__init__(len(dofs))
        self.dofs = np.asarray(dofs, dtype=np.intc)

    def localEntriesOfBlock(self):
        return np.array([2, 0, 1], dtype=np.intc), np.array([0, 2, 1], dtype=np.intc)


class _OneSlaveRigidBodyContact(SystemMatrixContributorBase):
    """The block of a contact between one node and a rigid body in 3D, as
    :class:`~edelweissfe.constraints.nodetodiscreterigidbodypenalty.Constraint` lays it out: the six degrees of
    freedom of the reference point (first) coupled with each other, the three of the node (last) with each other, and
    the two groups both ways. That is 36 + 9 + 18 + 18 = 81 entries on nDof = 9 degrees of freedom: exactly as many as
    a dense 9 x 9 block, but ordered block by block, not column by column."""

    rpDofCount = 6

    def __init__(self, dofs):
        super().__init__(len(dofs))
        self.dofs = np.asarray(dofs, dtype=np.intc)

    def localEntriesOfBlock(self):
        return rigidBodyContactLocalEntries(np.arange(self.rpDofCount), [np.arange(self.rpDofCount, self.nDof)])


class _DofManager:
    """The part of a DofManager the assemblies use: the number of DOFs and the global DOF list of every entity."""

    def __init__(self, entities, nDof):
        self.nDof = nDof
        self.idcsOfHigherOrderEntitiesInDofVector = {entity: entity.dofs for entity in entities}


def _model():
    rng = np.random.default_rng(42)
    entities = [_DenseEntity(rng.choice(12, size=4, replace=False)) for _ in range(40)]
    entities += [_SparseEntity([0, 5, 11]), _SparseEntity([3, 7, 2])]
    blocks = {entity: rng.standard_normal(countEntriesOfBlock(entity)) for entity in entities}
    return entities, _DofManager(entities, 12), blocks


def _fillInOfTheWholeMatrix(pattern):
    """Every entry that is not yet in the pattern: room for a condensation that couples everything."""
    existing = pattern.copy()
    existing.data[:] = 1.0
    entries = csr_matrix(np.ones(pattern.shape)) - existing
    entries.eliminate_zeros()
    entries.sort_indices()
    return entries


def _writeBlocks(K, entities, blocks, numberOfWorkers):
    """Every entity adds its block, the entities in chunks on the workers, as the element loop does."""

    def work(worker, chunk):
        writer = K.blocksOfWorker(worker)
        for entity in chunk:
            block = writer[entity]
            block += blocks[entity].reshape(block.shape, order="F")
        writer.flush()

    mapChunksOnWorkers(entities, work, numberOfWorkers)


def _assemble(assemblyType, numberOfWorkers, factor=None, model=_model, patternFillIn=None):
    entities, dofManager, blocks = model()
    K = assemblyType(dofManager, numberOfWorkers, patternFillIn)

    # an abandoned assembly: a block handed out, never committed
    K.beginAssembly()
    K[entities[0]][...] = 1e10

    K.beginAssembly()
    _writeBlocks(K, entities, blocks, numberOfWorkers)

    if factor is not None:
        scaled = K.scaledBy(factor)
        for entity in entities[:5]:
            block = scaled[entity]
            block += blocks[entity].reshape(block.shape, order="F")
        scaled.flush()

    assembled = K.assembledMatrix()
    assert assembled is K.csrMatrix
    return assembled.toarray()


def _reference(factor=None, model=_model):
    entities, dofManager, blocks = model()
    reference = np.zeros((dofManager.nDof, dofManager.nDof))
    for entity in entities:
        localRows, localColumns = entity.localEntriesOfBlock()
        np.add.at(reference, (entity.dofs[localRows], entity.dofs[localColumns]), blocks[entity])
    if factor is not None:
        for entity in entities[:5]:
            localRows, localColumns = entity.localEntriesOfBlock()
            np.add.at(reference, (entity.dofs[localRows], entity.dofs[localColumns]), factor * blocks[entity])
    return reference


@pytest.mark.parametrize("patternFillIn", [None, _fillInOfTheWholeMatrix])
@pytest.mark.parametrize("assemblyType", [VIJSystemMatrixAssembly, DirectCSRSystemMatrixAssembly])
@pytest.mark.parametrize("numberOfWorkers", [1, 3])
def test_both_assemblies_assemble_the_same_matrix(assemblyType, numberOfWorkers, patternFillIn):
    np.testing.assert_allclose(
        _assemble(assemblyType, numberOfWorkers, patternFillIn=patternFillIn), _reference(), rtol=1e-14, atol=1e-14
    )


@pytest.mark.parametrize("patternFillIn", [None, _fillInOfTheWholeMatrix])
def test_both_assemblies_derive_the_same_pattern(patternFillIn):
    """The VIJ assembly sorts and sums the pairs, the direct assembly unions the entities' columns row by row: two
    independent derivations of one pattern, with the same entries added."""
    entities, dofManager, _ = _model()
    vij = VIJSystemMatrixAssembly(dofManager, 1, patternFillIn).csrMatrix
    direct = DirectCSRSystemMatrixAssembly(dofManager, 3, patternFillIn).csrMatrix
    np.testing.assert_array_equal(direct.indptr, vij.indptr)
    np.testing.assert_array_equal(direct.indices, vij.indices)


@pytest.mark.parametrize("assemblyType", [VIJSystemMatrixAssembly, DirectCSRSystemMatrixAssembly])
def test_scaled_blocks_enter_scaled(assemblyType):
    np.testing.assert_allclose(_assemble(assemblyType, 2, factor=0.3), _reference(factor=0.3), rtol=1e-14, atol=1e-14)


def _modelWithOneSlaveRigidBodyContact():
    """Elements on twelve degrees of freedom, and one rigid body contact on nine of them whose block has exactly
    nDof**2 entries."""
    rng = np.random.default_rng(3)
    entities = [_DenseEntity(rng.choice(12, size=4, replace=False)) for _ in range(10)]
    entities += [_OneSlaveRigidBodyContact([11, 0, 4, 9, 2, 7, 5, 10, 1])]
    assert countEntriesOfBlock(entities[-1]) == entities[-1].nDof ** 2
    blocks = {entity: rng.standard_normal(countEntriesOfBlock(entity)) for entity in entities}
    return entities, _DofManager(entities, 12), blocks


@pytest.mark.parametrize("assemblyType", [VIJSystemMatrixAssembly, DirectCSRSystemMatrixAssembly])
@pytest.mark.parametrize("numberOfWorkers", [1, 3])
def test_sparse_entity_whose_size_is_a_square_is_scattered_entry_by_entry(assemblyType, numberOfWorkers):
    """How many entries a block has does not say how they are laid out. A rigid body contact with one slave node has
    exactly as many as a dense block on its nine degrees of freedom; scattered as if it were one, its values would
    land on the wrong rows and columns. It lists its entries (``localEntriesOfBlock``), so both assemblies place every
    value at its own entry."""
    model = _modelWithOneSlaveRigidBodyContact
    np.testing.assert_allclose(
        _assemble(assemblyType, numberOfWorkers, model=model), _reference(model=model), rtol=1e-14, atol=1e-14
    )


@pytest.mark.parametrize(
    "constraintType",
    [
        RigidBody,
        NodeToDiscreteRigidBodyPenalty,
        SurfaceToDiscreteRigidBodyPenalty,
        NodeToDeformableSurfacePenalty,
        SurfaceToDeformableSurfacePenalty,
    ],
)
def test_constraints_with_entries_of_their_own_say_so(constraintType):
    """Every constraint that lists the entries of its block itself overrides ``localEntriesOfBlock``, and that alone
    makes its block a list for both assemblies, so that no assembly mistakes it for a dense one."""
    assert constraintType.localEntriesOfBlock is not SystemMatrixContributorBase.localEntriesOfBlock
    # the layout is a property of the class: an instance without its state tells it
    assert object.__new__(constraintType).couplesAllItsDofs is False


def test_the_entries_of_a_one_slave_rigid_body_contact():
    """2D: the three degrees of freedom of the reference point (local 0, 1, 2) and the two of the slave node (3, 4).
    Each sub-block row by row: the reference point with itself, the node with itself, node-to-RP, RP-to-node."""
    localRows, localColumns = rigidBodyContactLocalEntries(np.arange(3), [np.arange(3, 5)])
    rpRp = [(a, b) for a in range(3) for b in range(3)]
    nodeNode = [(3, 3), (3, 4), (4, 3), (4, 4)]
    nodeRp = [(a, b) for a in (3, 4) for b in range(3)]
    rpNode = [(a, b) for a in range(3) for b in (3, 4)]
    expected = np.array(rpRp + nodeNode + nodeRp + rpNode)
    np.testing.assert_array_equal(localRows, expected[:, 0])
    np.testing.assert_array_equal(localColumns, expected[:, 1])
    assert localRows.dtype == np.intc and localColumns.dtype == np.intc


def test_the_entries_of_a_rigid_body():
    """2D, one slave node: its displacement (local 0, 1), the reference point's displacement (2, 3) and rotation (4),
    and the two Lagrange multipliers of the slave (5, 6). The rotation with itself, then the five coupled degrees of
    freedom with the multipliers and back, then the diagonal of all seven."""
    rigidBody = SimpleNamespace(
        nDim=2,
        nRot=1,
        nDofsOnNodes=5,
        nDof=7,
        slaveNodes=[None],
        indicesOfSlaveNodesInP=[[0, 1]],
        indicesOfRPUinP=[2, 3],
        indicesOfRPPhiInP=[4],
    )
    localRows, localColumns = RigidBody.localEntriesOfBlock(rigidBody)
    coupledToMultipliers = [(a, b) for a in range(5) for b in (5, 6)]
    multipliersToCoupled = [(a, b) for a in (5, 6) for b in range(5)]
    diagonal = [(a, a) for a in range(7)]
    expected = np.array([(4, 4)] + coupledToMultipliers + multipliersToCoupled + diagonal)
    np.testing.assert_array_equal(localRows, expected[:, 0])
    np.testing.assert_array_equal(localColumns, expected[:, 1])


def test_the_entries_of_a_surface_to_deformable_surface_contact():
    """Two active contact points, on two and on three degrees of freedom: one dense block each, row by row."""
    contact = SimpleNamespace(_activeBlockSizes=lambda: [2, 3])
    localRows, localColumns = SurfaceToDeformableSurfacePenalty.localEntriesOfBlock(contact)
    expected = np.array([(a, b) for a in (0, 1) for b in (0, 1)] + [(a, b) for a in (2, 3, 4) for b in (2, 3, 4)])
    np.testing.assert_array_equal(localRows, expected[:, 0])
    np.testing.assert_array_equal(localColumns, expected[:, 1])


def test_a_contact_without_active_points_has_no_entries():
    contact = SimpleNamespace(_activeBlockSizes=lambda: [])
    localRows, localColumns = SurfaceToDeformableSurfacePenalty.localEntriesOfBlock(contact)
    assert localRows.shape == (0,) and localColumns.shape == (0,)


def test_direct_assembly_with_workers_agrees_with_itself_to_round_off():
    """The workers add to the one CSR matrix atomically, in an order set by their timing: repeated, the matrix agrees
    to round-off, not necessarily bit for bit."""
    first = _assemble(DirectCSRSystemMatrixAssembly, 3)
    for _ in range(5):
        np.testing.assert_allclose(_assemble(DirectCSRSystemMatrixAssembly, 3), first, rtol=1e-14, atol=1e-14)


_ASSEMBLIES = [VIJSystemMatrixAssembly, DirectCSRSystemMatrixAssembly]


def _onThePattern(K, values):
    """A matrix of its own (not sharing an array with the system matrix) on the sparsity pattern of ``K``."""
    pattern = K.csrMatrix
    return csr_matrix((values, pattern.indices.copy(), pattern.indptr.copy()), shape=pattern.shape)


@pytest.mark.parametrize("assemblyType", _ASSEMBLIES)
@pytest.mark.parametrize("numberOfWorkers", [1, 3])
def test_a_matrix_and_a_lumped_matrix_enter_during_the_assembly(assemblyType, numberOfWorkers):
    """K = sum of the blocks + 0.7 M - 0.2 C, as a dynamic solver forms its effective tangent: the same however often
    the assembled matrix is asked for, and gone in the next assembly."""
    entities, dofManager, blocks = _model()
    K = assemblyType(dofManager, numberOfWorkers)
    rng = np.random.default_rng(7)

    source = _onThePattern(K, rng.standard_normal(K.csrMatrix.nnz))
    M = K.copyOntoSparsityPattern(source)
    lumpedMatrix = rng.standard_normal(dofManager.nDof)
    lumpedMatrix[[1, 4]] = 0.0
    coupled = np.unique(np.concatenate([entity.dofs for entity in entities if isinstance(entity, _DenseEntity)]))
    lumpedMatrix[np.setdiff1d(np.arange(dofManager.nDof), coupled)] = 0.0

    K.beginAssembly()
    _writeBlocks(K, entities, blocks, numberOfWorkers)
    K.addMatrix(M, 0.7)
    K.addLumpedMatrix(lumpedMatrix, -0.2)

    expected = _reference() + 0.7 * source.toarray() - 0.2 * np.diag(lumpedMatrix)
    np.testing.assert_allclose(K.assembledMatrix().toarray(), expected, rtol=1e-14, atol=1e-14)
    np.testing.assert_allclose(K.assembledMatrix().toarray(), expected, rtol=1e-14, atol=1e-14)

    K.beginAssembly()
    _writeBlocks(K, entities, blocks, numberOfWorkers)
    np.testing.assert_allclose(K.assembledMatrix().toarray(), _reference(), rtol=1e-14, atol=1e-14)


def _sparseModel():
    """Two dense blocks and one block with entries of its own on eight degrees of freedom: the pattern is known, with
    entries outside it, and dofs 4 and 6 without a diagonal entry (dof 7 is coupled to nothing)."""
    entities = [_DenseEntity([0, 1]), _DenseEntity([2, 3]), _SparseEntity([4, 5, 6])]
    blocks = {entity: np.arange(1.0, countEntriesOfBlock(entity) + 1.0) for entity in entities}
    return entities, _DofManager(entities, 8), blocks


@pytest.mark.parametrize("assemblyType", _ASSEMBLIES)
@pytest.mark.parametrize("numberOfWorkers", [1, 3])
def test_a_lumped_matrix_off_the_diagonal_of_the_pattern_is_refused(assemblyType, numberOfWorkers):
    _, dofManager, _ = _sparseModel()
    K = assemblyType(dofManager, numberOfWorkers)
    K.beginAssembly()

    onTheDiagonal = np.zeros(dofManager.nDof)
    onTheDiagonal[[0, 3, 5]] = 1.0
    K.addLumpedMatrix(onTheDiagonal, 2.0)
    np.testing.assert_array_equal(K.assembledMatrix().diagonal(), 2.0 * onTheDiagonal)

    offTheDiagonal = np.zeros(dofManager.nDof)
    offTheDiagonal[4] = 1.0
    with pytest.raises(ValueError, match="no diagonal entry"):
        K.addLumpedMatrix(offTheDiagonal, 1.0)

    with pytest.raises(ValueError, match="one value per degree of freedom"):
        K.addLumpedMatrix(np.ones(dofManager.nDof + 1), 1.0)


@pytest.mark.parametrize("assemblyType", _ASSEMBLIES)
def test_a_lumped_matrix_finds_no_diagonal_in_an_empty_pattern(assemblyType):
    """A model whose degrees of freedom are coupled to nothing has a sparsity pattern without entries: a lumped matrix
    with a nonzero value is refused, a zero one adds nothing."""
    K = assemblyType(_DofManager([], 3), 1)
    K.beginAssembly()

    K.addLumpedMatrix(np.zeros(3), 1.0)
    assert K.assembledMatrix().nnz == 0

    K.beginAssembly()
    with pytest.raises(ValueError, match="no diagonal entry"):
        K.addLumpedMatrix(np.array([0.0, 1.0, 0.0]), 1.0)


@pytest.mark.parametrize("assemblyType", _ASSEMBLIES)
@pytest.mark.parametrize("numberOfWorkers", [1, 3])
def test_a_copy_on_the_same_pattern_has_values_of_its_own(assemblyType, numberOfWorkers):
    entities, dofManager, blocks = _sparseModel()
    K = assemblyType(dofManager, numberOfWorkers)
    K.beginAssembly()
    _writeBlocks(K, entities, blocks, numberOfWorkers)
    assembled = K.assembledMatrix()
    expected = assembled.toarray()

    copy = K.copyOntoSparsityPattern(assembled)

    assert copy.indptr is K.csrMatrix.indptr and copy.indices is K.csrMatrix.indices
    assert not np.shares_memory(copy.data, K.csrMatrix.data)
    K.beginAssembly()
    K.assembledMatrix()
    np.testing.assert_array_equal(copy.toarray(), expected)

    # the pattern is the system matrix's: a structural change is refused, a change of the values is not
    with pytest.raises(RuntimeError):
        copy.eliminate_zeros()
    copy.data *= 2.0
    np.testing.assert_array_equal(copy.toarray(), 2.0 * expected)

    with pytest.raises(ValueError, match="not on the sparsity pattern"):
        K.addMatrix(assembled.copy(), 1.0)


@pytest.mark.parametrize("assemblyType", _ASSEMBLIES)
@pytest.mark.parametrize("numberOfWorkers", [1, 3])
def test_a_copy_from_a_part_of_the_pattern_is_zero_elsewhere(assemblyType, numberOfWorkers):
    """A matrix on a part of the pattern (as the mass of the discrete system before a contact search widened the
    pattern), with a stored zero outside the pattern, and a duplicate entry, which is summed."""
    _, dofManager, _ = _sparseModel()
    K = assemblyType(dofManager, numberOfWorkers)

    # row 0: (0, 1) and a stored zero at (0, 3), outside; row 1: (1, 1); row 3: (3, 2) twice; row 5: (5, 5); row 6: (6, 4)
    values = np.array([1.0, 0.0, 2.0, 3.0, 4.0, 6.0, 5.0])
    columns = np.array([1, 3, 1, 2, 2, 5, 4])
    rowStarts = np.array([0, 2, 3, 3, 5, 5, 6, 7, 7])
    part = csr_matrix((values, columns, rowStarts), shape=(dofManager.nDof, dofManager.nDof))

    copy = K.copyOntoSparsityPattern(part)

    assert copy.indices is K.csrMatrix.indices
    np.testing.assert_array_equal(copy.toarray(), part.toarray())
    assert copy.nnz == K.csrMatrix.nnz


@pytest.mark.parametrize("assemblyType", _ASSEMBLIES)
@pytest.mark.parametrize("numberOfWorkers", [1, 3])
def test_a_nonzero_outside_the_pattern_is_not_copied_onto_it(assemblyType, numberOfWorkers):
    _, dofManager, _ = _sparseModel()
    K = assemblyType(dofManager, numberOfWorkers)

    outside = csr_matrix(([1.0, 2.0], ([0, 0], [1, 3])), shape=(dofManager.nDof, dofManager.nDof))
    with pytest.raises(ValueError, match=r"nonzero entry \(0, 3\) outside the sparsity pattern"):
        K.copyOntoSparsityPattern(outside)

    with pytest.raises(ValueError, match="shape"):
        K.copyOntoSparsityPattern(csr_matrix((dofManager.nDof + 1, dofManager.nDof + 1)))


def test_workers_get_their_chunks_in_turn():
    items = list(range(100))
    seen = mapChunksOnWorkers(items, lambda worker, chunk: (worker, chunk), numberOfWorkers=4, chunksPerWorker=2)
    assert [item for _, chunk in seen for item in chunk] == items
    assert [worker for worker, _ in seen] == [chunkNumber % 4 for chunkNumber in range(len(seen))]


def test_the_first_failing_chunk_raises():
    """As in a serial loop: the exception of the first failing chunk in the order of the chunks."""

    def work(worker, chunk):
        if chunk[0] >= 30:
            raise ValueError(chunk[0])

    with pytest.raises(ValueError) as error:
        mapChunksOnWorkers(list(range(100)), work, numberOfWorkers=4, chunksPerWorker=2)
    assert error.value.args[0] == 36


def test_an_error_in_the_view_of_an_entity_is_not_taken_for_an_array_index():
    """The VIJ value array hands an entity its block through the entity's own view of its values. An error in that view
    is the entity's, and comes out as it is, not as an index of the array that numpy does not understand."""

    class _BrokenView(_DenseEntity):
        def shapeBlock(self, values):
            raise AttributeError("the view of this entity is broken")

    entity = _BrokenView([0, 1])
    V = VIJSystemMatrix(4, {entity: slice(0, 4)})
    with pytest.raises(AttributeError, match="the view of this entity is broken"):
        V[entity]


def test_the_value_array_keeps_its_layout_in_a_copy_and_answers_array_indices():
    entity = _DenseEntity([0, 1])
    V = VIJSystemMatrix(4, {entity: slice(0, 4)})
    V[entity] = [1.0, 2.0, 3.0, 4.0]
    np.testing.assert_array_equal(V.copy()[entity], [[1.0, 3.0], [2.0, 4.0]])
    np.testing.assert_array_equal(V[1:3], [2.0, 3.0])
    assert V[np.int64(3)] == 4.0
    np.testing.assert_array_equal(V[np.array([0, 3])], [1.0, 4.0])
