#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unit tests for edelweissfe.numerics (DofVector, ScatterDofVector, VIJSystemMatrix, VIJEntityBase, DofManager).
"""

import unittest

import numpy as np

from edelweissfe.numerics.dofvector import DofVector
from edelweissfe.numerics.parallelizationutilities import (
    getNumberOfThreads,
    isFreeThreadingSupported,
)
from edelweissfe.numerics.scatterdofvector import ScatterDofVector, ScatterDofVectorTemplate
from edelweissfe.numerics.vijentitybase import VIJEntityBase
from edelweissfe.numerics.vijsystemmatrix import VIJSystemMatrix


class DummyEntity(VIJEntityBase):
    """Mock entity for testing."""

    def __init__(self, nDof: int, entity_id: int):
        super().__init__(nDof)
        self.entity_id = entity_id

    def __hash__(self):
        return hash(self.entity_id)

    def __eq__(self, other):
        return isinstance(other, DummyEntity) and self.entity_id == other.entity_id


class TestNumerics(unittest.TestCase):
    def setUp(self):
        self.ent1 = DummyEntity(nDof=2, entity_id=1)
        self.ent2 = DummyEntity(nDof=3, entity_id=2)
        self.entity_map = {
            self.ent1: np.array([0, 1], dtype=int),
            self.ent2: np.array([1, 2, 3], dtype=int),
        }

    def test_dof_vector_creation_and_indexing(self):
        dof_vec = DofVector(nDof=4, entitiesInDofVector=self.entity_map)
        self.assertEqual(dof_vec.size, 4)
        self.assertTrue(np.allclose(dof_vec, 0.0))

        # Standard indexing
        dof_vec[0] = 10.0
        self.assertEqual(dof_vec[0], 10.0)

        # Slice indexing
        dof_vec[1:3] = [20.0, 30.0]
        self.assertTrue(np.allclose(dof_vec[1:3], [20.0, 30.0]))

        # Tuple indexing
        self.assertEqual(dof_vec[(0,)], 10.0)

        # Entity-based indexing
        dof_vec[self.ent1] = [1.0, 2.0]
        self.assertTrue(np.allclose(dof_vec[self.ent1], [1.0, 2.0]))
        self.assertEqual(dof_vec[0], 1.0)
        self.assertEqual(dof_vec[1], 2.0)

        # Copying
        dof_copy = dof_vec.copy()
        self.assertIsInstance(dof_copy, DofVector)
        self.assertTrue(np.allclose(dof_copy, dof_vec))
        self.assertIn(self.ent1, dof_copy.entitiesInDofVector)

        # Create Scatter Vector
        scatter_vec = dof_vec.createScatterVector()
        self.assertIsInstance(scatter_vec, ScatterDofVector)
        self.assertEqual(scatter_vec.size, 5)  # 2 + 3 entries

    def test_scatter_dof_vector(self):
        template = ScatterDofVectorTemplate(self.entity_map, nDof=4)
        scatter_vec = ScatterDofVector(template)
        self.assertEqual(scatter_vec.size, 5)

        # Entity-based setitem and getitem
        scatter_vec[self.ent1] = [5.0, 6.0]
        scatter_vec[self.ent2] = [7.0, 8.0, 9.0]
        self.assertTrue(np.allclose(scatter_vec[self.ent1], [5.0, 6.0]))
        self.assertTrue(np.allclose(scatter_vec[self.ent2], [7.0, 8.0, 9.0]))

        # Assemble into global DofVector
        target = DofVector(4, self.entity_map)
        scatter_vec.assembleInto(target)
        # index 0: 5.0 (from ent1)
        # index 1: 6.0 (from ent1) + 7.0 (from ent2) = 13.0
        # index 2: 8.0 (from ent2)
        # index 3: 9.0 (from ent2)
        self.assertTrue(np.allclose(target, [5.0, 13.0, 8.0, 9.0]))

        # toDofVector
        as_dof = scatter_vec.toDofVector()
        self.assertTrue(np.allclose(as_dof, [5.0, 13.0, 8.0, 9.0]))

        # Absolute assembly
        scatter_vec[self.ent1] = [-5.0, 6.0]
        as_abs_dof = scatter_vec.toDofVector(absolute=True)
        self.assertTrue(np.allclose(as_abs_dof, [5.0, 13.0, 8.0, 9.0]))

        # Copy
        scatter_copy = scatter_vec.copy()
        self.assertIsInstance(scatter_copy, ScatterDofVector)
        self.assertTrue(np.allclose(scatter_copy, scatter_vec))

    def test_vij_system_matrix(self):
        # 2 elements: ent1 (2x2=4 entries), ent2 (3x3=9 entries) -> size = 13
        I_arr = np.zeros(13, dtype=np.intc)
        J_arr = np.zeros(13, dtype=np.intc)
        entities_in_vij = {self.ent1: 0, self.ent2: 4}

        self.ent1.initializeVIJContribution(self.entity_map[self.ent1], I_arr, J_arr, 0)
        self.ent2.initializeVIJContribution(self.entity_map[self.ent2], I_arr, J_arr, 4)

        vij_mat = VIJSystemMatrix(nDof=4, I=I_arr, J=J_arr, entitiesInVIJ=entities_in_vij)
        self.assertEqual(vij_mat.size, 13)

        # Entity read/write
        K1 = np.array([[1.0, 2.0], [3.0, 4.0]])
        vij_mat[self.ent1] = K1.flatten("F")
        self.assertTrue(np.allclose(vij_mat[self.ent1], K1))

        # Copy
        vij_copy = vij_mat.copy()
        self.assertIsInstance(vij_copy, VIJSystemMatrix)
        self.assertTrue(np.allclose(vij_copy, vij_mat))

    def test_parallelization_utilities(self):
        self.assertIsInstance(isFreeThreadingSupported(), bool)
        self.assertIsInstance(getNumberOfThreads(), int)
        self.assertGreaterEqual(getNumberOfThreads(), 1)


if __name__ == "__main__":
    unittest.main()
