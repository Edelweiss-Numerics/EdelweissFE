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
"""The contract of the contact points, for both kinds of point of a finite element surface.

The slave side is the top face of one hexa20 cube, tiled by the surface element generator. A contact point moves with
its interpolation vector :math:`c` and its force is distributed with the same vector; the tests check the pieces of
:class:`~edelweissfe.constraints.base.contactpoints.ContactPoints` against each other and against what an affine
displacement must give, since both kinds of point reproduce an affine field exactly.
"""

import unittest

import numpy as np
from _hexa20cube import (  # noqa: E402  (tests/ is on sys.path)
    _SIDE,
    _YMAX,
    _hexa20Coordinates,
)

import edelweissfe.utils.inputfileparser  # noqa: F401 bootstrap input language
from edelweissfe.constraints.base.contactpointsatslavenodes import (
    ContactPointsAtSlaveNodes,
)
from edelweissfe.constraints.base.contactpointsonslavesurface import (
    ContactPointsOnSlaveSurface,
)
from edelweissfe.elements.displacementelement.element import DisplacementElement
from edelweissfe.generators.surfaceelementgenerator import buildContactFacets
from edelweissfe.journal.journal import Journal
from edelweissfe.models.femodel import FEModel
from edelweissfe.points.node import Node
from edelweissfe.sets.elementset import ElementSet


def _slaveSurfaceOfOneCube() -> tuple:
    """One hexa20 cube; its top face (y = side) tiled by contact facets.

    Returns
    -------
    tuple
        ``(model, slaveSurface)``.
    """

    model = FEModel(3)
    with model.topology.changes():
        nodes = []
        for label, x in enumerate(_hexa20Coordinates(0.0), start=1):
            nodes.append(Node(label, x))
            model.nodes[label] = nodes[-1]
        (elNumber,) = model.topology.reserveElementNumbers(1)
        element = DisplacementElement("C3D20", elNumber)
        element.setNodes(nodes)
        model.createElement(element)
        model.surfaces["slaveFace"] = {_YMAX: ElementSet("s", [element])}
        slaveSetName, _ = buildContactFacets(model, "slaveFace", "slv", "midside", "facetConsistent", Journal())
    return model, model.elementSets[slaveSetName]


class TestContactPoints(unittest.TestCase):
    def _kindsOfPoints(self) -> list:
        """Both kinds of contact point of the same slave surface, with the model."""

        model, slaveSurface = _slaveSurfaceOfOneCube()
        return [
            (model, ContactPointsOnSlaveSurface(slaveSurface, 3, "theContact")),
            (model, ContactPointsAtSlaveNodes(slaveSurface, "theContact")),
        ]

    def test_areas_add_up_to_the_area_of_the_face(self):
        for _, points in self._kindsOfPoints():
            self.assertAlmostEqual(points.areas.sum(), _SIDE**2, places=12, msg=points.describe())
            self.assertEqual(len(points.areas), points.nPoints, points.describe())

    def test_located_points_are_where_the_search_sees_them(self):
        """Located in any order, a point's origin plus its displacement is its position; with no displacement field,
        the current coordinates of the search are the origins."""

        for model, points in self._kindsOfPoints():
            located = np.arange(points.nPoints)[::2]
            nodes = points.locatePoints(located)
            origins = points.originsOfLocatedPoints()
            np.testing.assert_allclose(origins, points.currentPointCoordinates(model)[located], atol=1e-14)

            zero = np.zeros(3 * len(nodes))
            np.testing.assert_array_equal(points.coordinatesOfLocatedPoints(zero, zero), origins)

    def test_an_affine_displacement_moves_every_point_affinely(self):
        """The shape functions of a face and :math:`c = [1]` reproduce :math:`u = A X + b` at the points."""

        rng = np.random.default_rng(0)
        A, b = 1e-2 * rng.standard_normal((3, 3)), 1e-2 * rng.standard_normal(3)
        for _, points in self._kindsOfPoints():
            nodes = points.locatePoints(np.arange(points.nPoints))
            U = np.concatenate([A @ node.coordinates + b for node in nodes])
            displacements = points.displacementsOfLocatedPoints(U, np.zeros_like(U))
            np.testing.assert_allclose(
                displacements, points.originsOfLocatedPoints() @ A.T + b, atol=1e-14, err_msg=points.describe()
            )

    def test_the_interpolation_of_one_point_is_its_row_of_the_matrix(self):
        """:math:`C` holds the interpolation vectors over the nodes of the block, and :math:`C X` are the origins."""

        for _, points in self._kindsOfPoints():
            nodes = points.locatePoints(np.arange(points.nPoints))
            C = points.interpolationMatrixOfLocatedPoints().toarray()
            self.assertEqual(C.shape, (points.nPoints, len(nodes)), points.describe())
            np.testing.assert_allclose(
                C @ np.array([node.coordinates for node in nodes]), points.originsOfLocatedPoints()
            )

            for k in range(points.nPoints):
                dofs, c = points.interpolationOfLocatedPoint(k)
                row = np.zeros(len(nodes))
                row[dofs[::3] // 3] = c
                np.testing.assert_array_equal(C[k], row, err_msg=f"{points.describe()}, point {k}")
                np.testing.assert_array_equal(dofs, (dofs[::3, None] + np.arange(3)).ravel())

    def test_the_force_is_distributed_with_the_transpose_of_the_motion(self):
        """The residual contribution of forces :math:`f` along directions :math:`n` at some points is
        :math:`C^T (f n)`: so the work of the nodal forces on any motion of the nodes is the work of the point forces
        on the motion of the points."""

        rng = np.random.default_rng(1)
        for _, points in self._kindsOfPoints():
            nodes = points.locatePoints(np.arange(points.nPoints))
            C = points.interpolationMatrixOfLocatedPoints().toarray()
            carrying = np.arange(points.nPoints)[1::3]
            forces = rng.standard_normal(len(carrying))
            directions = rng.standard_normal((len(carrying), 3))
            directions /= np.linalg.norm(directions, axis=1)[:, None]

            P = np.zeros(3 * len(nodes))
            points.addForceResidual(P, carrying, forces, directions)
            np.testing.assert_allclose(
                P.reshape((-1, 3)),
                C[carrying].T @ (forces[:, None] * directions),
                atol=1e-14,
                err_msg=points.describe(),
            )

    def test_nodal_normal_forces_keep_the_resultant(self):
        """Distributed to the surface nodes, the normal forces of the points keep their sum, with the sign turned: a
        point in compression (negative force) repels its nodes."""

        rng = np.random.default_rng(2)
        for _, points in self._kindsOfPoints():
            normalForces = -rng.random(points.nPoints)
            nodalForces = points.nodalNormalForces(normalForces)
            self.assertEqual(len(nodalForces), len(points.surfaceNodes))
            self.assertAlmostEqual(nodalForces.sum(), -normalForces.sum(), places=12, msg=points.describe())
            np.testing.assert_allclose(points.normalPressures(normalForces), -normalForces / points.areas)

    def test_the_finite_element_nodes_are_the_nodes_of_the_face(self):
        for model, points in self._kindsOfPoints():
            faceNodes = {node for node in model.nodes.values() if node.coordinates[1] == _SIDE}
            self.assertEqual(len(faceNodes), 8)
            self.assertEqual(points.finiteElementNodes(), faceNodes, points.describe())


if __name__ == "__main__":
    unittest.main()
