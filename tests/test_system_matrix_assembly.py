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
matrix: from dense blocks and from a sparse custom VIJ pattern, from several worker threads, with blocks scaled by a
factor, and after an abandoned assembly; and the worker loop deals out its chunks as documented."""

import numpy as np
import pytest

from edelweissfe.constraints.base.rigidbodycontactstiffness import (
    fillRigidBodyContactIndices,
    rigidBodyContactContributionSize,
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
from edelweissfe.numerics.csrgeneratorv2 import CSRGenerator
from edelweissfe.numerics.parallelizationutilities import mapChunksOnWorkers
from edelweissfe.numerics.systemmatrixassembly import (
    DirectCSRSystemMatrixAssembly,
    VIJSystemMatrixAssembly,
)
from edelweissfe.numerics.vijentitybase import VIJEntityBase
from edelweissfe.numerics.vijsystemmatrix import VIJSystemMatrix


class _DenseEntity(VIJEntityBase):
    """An element-like entity: a dense block on its DOFs."""

    def __init__(self, dofs):
        super().__init__(len(dofs))
        self.dofs = np.asarray(dofs, dtype=np.intc)


class _SparseEntity(VIJEntityBase):
    """A constraint-like entity on three DOFs that couples only the first and the last, and the middle one with
    itself."""

    def __init__(self, dofs):
        super().__init__(len(dofs))
        self.dofs = np.asarray(dofs, dtype=np.intc)

    @property
    def couplesAllItsDofs(self):
        return False

    def getVIJContributionSize(self):
        return 3

    def initializeVIJContribution(self, idcs, I_, J_, offset):
        I_[offset : offset + 3] = [idcs[2], idcs[0], idcs[1]]
        J_[offset : offset + 3] = [idcs[0], idcs[2], idcs[1]]


class _OneSlaveRigidBodyContact(VIJEntityBase):
    """The pattern of a contact between one node and a rigid body in 3D, as
    :class:`~edelweissfe.constraints.nodetodiscreterigidbodypenalty.Constraint` lays it out: the six degrees of
    freedom of the reference point (first) coupled with each other, the three of the node (last) with each other, and
    the two groups both ways. That is 36 + 9 + 18 + 18 = 81 entries on nDof = 9 degrees of freedom: exactly as many as
    a dense 9 x 9 block, but ordered block by block, not column by column."""

    rpDofCount = 6

    def __init__(self, dofs):
        super().__init__(len(dofs))
        self.dofs = np.asarray(dofs, dtype=np.intc)

    @property
    def couplesAllItsDofs(self):
        return False

    def getVIJContributionSize(self):
        return rigidBodyContactContributionSize(self.rpDofCount, [self.nDof - self.rpDofCount])

    def initializeVIJContribution(self, idcs, I_, J_, offset):
        fillRigidBodyContactIndices(idcs[: self.rpDofCount], [idcs[self.rpDofCount :]], I_, J_, offset)


class _DofManager:
    """The part of a DofManager the assemblies use: the VIJ pattern of the entities."""

    def __init__(self, entities, nDof):
        self.nDof = nDof
        self.idcsOfHigherOrderEntitiesInVIJ = {}
        size = sum(entity.getVIJContributionSize() for entity in entities)
        self.I = np.zeros(size, dtype=np.intc)  # noqa: E741
        self.J = np.zeros(size, dtype=np.intc)
        offset = 0
        for entity in entities:
            self.idcsOfHigherOrderEntitiesInVIJ[entity] = offset
            entity.initializeVIJContribution(entity.dofs, self.I, self.J, offset)
            offset += entity.getVIJContributionSize()

    def constructVIJSystemMatrix(self):
        return VIJSystemMatrix(self.nDof, self.I, self.J, self.idcsOfHigherOrderEntitiesInVIJ)


def _model():
    rng = np.random.default_rng(42)
    entities = [_DenseEntity(rng.choice(12, size=4, replace=False)) for _ in range(40)]
    entities += [_SparseEntity([0, 5, 11]), _SparseEntity([3, 7, 2])]
    blocks = {entity: rng.standard_normal(entity.getVIJContributionSize()) for entity in entities}
    return entities, _DofManager(entities, 12), blocks


def _assemble(assemblyType, numberOfWorkers, factor=None, model=_model):
    entities, dofManager, blocks = model()
    generator = CSRGenerator(dofManager, patternOnly=not assemblyType.needsGatherMap)
    K = assemblyType(generator, dofManager, numberOfWorkers)

    # an abandoned assembly: a block handed out, never committed
    K.beginAssembly()
    K[entities[0]][...] = 1e10

    K.beginAssembly()

    def work(worker, chunk):
        writer = K.blocksOfWorker(worker)
        for entity in chunk:
            block = writer[entity]
            block += blocks[entity].reshape(block.shape, order="F")
        writer.flush()

    mapChunksOnWorkers(entities, work, numberOfWorkers)

    if factor is not None:
        scaled = K.scaledBy(factor)
        for entity in entities[:5]:
            block = scaled[entity]
            block += blocks[entity].reshape(block.shape, order="F")
        scaled.flush()

    return K.assembledMatrix().toarray()


def _reference(factor=None, model=_model):
    entities, dofManager, blocks = model()
    reference = np.zeros((dofManager.nDof, dofManager.nDof))
    for entity in entities:
        start = dofManager.idcsOfHigherOrderEntitiesInVIJ[entity]
        size = entity.getVIJContributionSize()
        for k in range(size):
            reference[dofManager.I[start + k], dofManager.J[start + k]] += blocks[entity][k]
    if factor is not None:
        for entity in entities[:5]:
            start = dofManager.idcsOfHigherOrderEntitiesInVIJ[entity]
            for k in range(entity.getVIJContributionSize()):
                reference[dofManager.I[start + k], dofManager.J[start + k]] += factor * blocks[entity][k]
    return reference


@pytest.mark.parametrize("assemblyType", [VIJSystemMatrixAssembly, DirectCSRSystemMatrixAssembly])
@pytest.mark.parametrize("numberOfWorkers", [1, 3])
def test_both_assemblies_assemble_the_same_matrix(assemblyType, numberOfWorkers):
    np.testing.assert_allclose(_assemble(assemblyType, numberOfWorkers), _reference(), rtol=1e-14, atol=1e-14)


@pytest.mark.parametrize("assemblyType", [VIJSystemMatrixAssembly, DirectCSRSystemMatrixAssembly])
def test_scaled_blocks_enter_scaled(assemblyType):
    np.testing.assert_allclose(_assemble(assemblyType, 2, factor=0.3), _reference(factor=0.3), rtol=1e-14, atol=1e-14)


def _modelWithOneSlaveRigidBodyContact():
    """Elements on twelve degrees of freedom, and one rigid body contact on nine of them whose pattern has exactly
    nDof**2 entries."""
    rng = np.random.default_rng(3)
    entities = [_DenseEntity(rng.choice(12, size=4, replace=False)) for _ in range(10)]
    entities += [_OneSlaveRigidBodyContact([11, 0, 4, 9, 2, 7, 5, 10, 1])]
    assert entities[-1].getVIJContributionSize() == entities[-1].nDof ** 2
    blocks = {entity: rng.standard_normal(entity.getVIJContributionSize()) for entity in entities}
    return entities, _DofManager(entities, 12), blocks


@pytest.mark.parametrize("assemblyType", [VIJSystemMatrixAssembly, DirectCSRSystemMatrixAssembly])
@pytest.mark.parametrize("numberOfWorkers", [1, 3])
def test_sparse_entity_whose_size_is_a_square_is_scattered_entry_by_entry(assemblyType, numberOfWorkers):
    """How many entries a block has does not say how they are laid out. A rigid body contact with one slave node has
    exactly as many as a dense block on its nine degrees of freedom; scattered as if it were one, its values would
    land on the wrong rows and columns. It says it is sparse (``couplesAllItsDofs``), so both assemblies place every
    value at its own (I, J) pair."""
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
def test_constraints_with_a_pattern_of_their_own_say_so(constraintType):
    """Every constraint that lays out a sparse pattern of its own declares it, so that no assembly mistakes its block
    for a dense one."""
    assert constraintType.initializeVIJContribution is not VIJEntityBase.initializeVIJContribution
    assert constraintType.couplesAllItsDofs is not VIJEntityBase.couplesAllItsDofs
    assert constraintType.couplesAllItsDofs.fget(None) is False


def test_direct_assembly_with_workers_agrees_with_itself_to_round_off():
    """The workers add to the one CSR matrix atomically, in an order set by their timing: repeated, the matrix agrees
    to round-off, not necessarily bit for bit."""
    first = _assemble(DirectCSRSystemMatrixAssembly, 3)
    for _ in range(5):
        np.testing.assert_allclose(_assemble(DirectCSRSystemMatrixAssembly, 3), first, rtol=1e-14, atol=1e-14)


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
