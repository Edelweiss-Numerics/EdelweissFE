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
"""Unit tests for the geometry and pose helpers of :class:`~edelweissfe.rigidbodies.discreterigidbody.DiscreteRigidBody`."""

import os
import tempfile
import unittest

import numpy as np
import pyvista as pv

from edelweissfe.generators.discreterigidbodygenerator import (
    generateDiscreteRigidBodyFromMeshFile,
)
from edelweissfe.journal.journal import Journal
from edelweissfe.models.femodel import FEModel
from edelweissfe.utils.facetcontactgeometry import tria3ClosestPoint
from edelweissfe.utils.rotations import rightJacobianSO3, rotationMatrixFromPseudoVector


class TestDiscreteRigidBody(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.cubeCenter = np.array([1.0, 2.0, 3.0])

    def tearDown(self):
        self.directory.cleanup()

    def _rigidBodyFromSurface(self, surface: pv.PolyData, name: str = "cube"):
        filename = os.path.join(self.directory.name, f"{name}.stl")
        surface.save(filename)
        model = FEModel(3)
        with model.topologyChanges():
            return generateDiscreteRigidBodyFromMeshFile(model, Journal(), name, filename)

    def _unitCube(self) -> pv.PolyData:
        return pv.Cube(center=self.cubeCenter, x_length=1.0, y_length=1.0, z_length=1.0).triangulate()

    def test_reference_triangles_have_outward_normals(self):
        triangles, normals = self._rigidBodyFromSurface(self._unitCube()).referenceTriangles()

        self.assertEqual(triangles.shape, (12, 3, 3))
        np.testing.assert_allclose(np.linalg.norm(normals, axis=1), 1.0, rtol=0, atol=1e-12)
        outwardness = np.einsum("ti,ti->t", triangles.mean(axis=1) - self.cubeCenter, normals)
        self.assertTrue(np.all(outwardness > 0.0))

    def test_reference_triangles_agree_with_surface_query(self):
        rigidBody = self._rigidBodyFromSurface(self._unitCube())
        triangles, normals = rigidBody.referenceTriangles()
        point = self.cubeCenter + np.array([0.8, 0.1, -0.2])

        distances = [tria3ClosestPoint(point, *triangle)[1] for triangle in triangles]
        closest = int(np.argmin(distances))

        u = np.zeros(3)
        queryDistances, queryNormals = rigidBody.querySurface(
            point[None, :], kinematics=(u, np.eye(3), rigidBody.rpNode.coordinates)
        )
        self.assertAlmostEqual(distances[closest], queryDistances[0], places=6)
        np.testing.assert_allclose(normals[closest], queryNormals[0], rtol=0, atol=1e-12)

    def test_open_surface_is_rejected(self):
        openCube = self._unitCube().extract_cells(range(10)).extract_surface(algorithm="dataset_surface")
        rigidBody = self._rigidBodyFromSurface(openCube, "openCube")
        with self.assertRaises(ValueError) as context:
            rigidBody.referenceTriangles()
        self.assertIn("not closed", str(context.exception))

    def test_pose_from_dofs(self):
        rigidBody = self._rigidBodyFromSurface(self._unitCube())
        rpDisplacement = np.array([0.1, -0.2, 0.3])
        rpRotation = np.array([0.2, 0.4, -0.1])

        currentRpPosition, rotationMatrix, dPhysicalSpin_dTheta = rigidBody.poseFromDofs(rpDisplacement, rpRotation)

        np.testing.assert_array_equal(currentRpPosition, rigidBody.rpNode.coordinates + rpDisplacement)
        np.testing.assert_array_equal(rotationMatrix, rotationMatrixFromPseudoVector(rpRotation))
        np.testing.assert_array_equal(dPhysicalSpin_dTheta, rotationMatrix @ rightJacobianSO3(rpRotation))


if __name__ == "__main__":
    unittest.main()
