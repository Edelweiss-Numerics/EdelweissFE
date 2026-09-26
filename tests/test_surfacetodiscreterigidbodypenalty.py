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
"""Unit tests for :mod:`edelweissfe.constraints.surfacetodiscreterigidbodypenalty`."""

import os
import tempfile
import unittest

import numpy as np
import pyvista as pv
from _hexa20cube import (  # noqa: E402  (tests/ is on sys.path)
    _SIDE,
    _YMIN,
    _hexa20Coordinates,
)

import edelweissfe.utils.inputfileparser  # noqa: F401 bootstrap input language
from edelweissfe.constraints.surfacetodiscreterigidbodypenalty import (
    Constraint as RigidSurfaceContact,
)
from edelweissfe.constraints.surfacetodiscreterigidbodypenalty import (
    SurfaceToDiscreteRigidBodyPenaltySchema,
)
from edelweissfe.elements.displacementelement.element import DisplacementElement
from edelweissfe.fields.nodefield import NodeField
from edelweissfe.generators.discreterigidbodygenerator import (
    generateDiscreteRigidBodyFromMeshFile,
)
from edelweissfe.generators.surfaceelementgenerator import buildContactFacets
from edelweissfe.journal.journal import Journal
from edelweissfe.models.femodel import FEModel
from edelweissfe.models.modelchange import ModelChange
from edelweissfe.models.modelchangeobserver import ModelChangeType
from edelweissfe.points.node import Node
from edelweissfe.sets.elementset import ElementSet
from edelweissfe.sets.nodeset import NodeSet


class TestSurfaceToDiscreteRigidBodyContact(unittest.TestCase):
    def setUp(self):
        self.journal = Journal()
        self.directory = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.directory.cleanup()

    def _blockOnRigidSupport(
        self,
        penetration: float,
        openSurface: bool = False,
        mixedElements: bool = False,
        supportXMax: float = _SIDE + 1.0,
        reverseTriangleOrder: bool = False,
    ) -> tuple:
        """A hexa20 cube on y in [0, 2], and a rigid box below it whose top face at y = ``penetration``
        overlaps the cube's Ymin face (the slave surface). The rigid body's reference point sits at
        the center of the contact face. A ``supportXMax`` below ``_SIDE`` lets the cube overhang the
        support's edge, and ``reverseTriangleOrder`` stores the support's triangles in reverse order.

        Returns
        -------
        tuple
            ``(model, slaveSurface, rigidBody)``.
        """

        stlFile = os.path.join(self.directory.name, "support.stl")
        box = pv.Box(bounds=(-1.0, supportXMax, penetration - 1.0, penetration, -1.0, _SIDE + 1.0)).triangulate()
        if openSurface:
            box = box.extract_cells(range(10)).extract_surface(algorithm="dataset_surface")
        if reverseTriangleOrder:
            box = pv.PolyData(box.points, faces=box.faces.reshape(-1, 4)[::-1].ravel())
        box.save(stlFile)

        model = FEModel(3)
        with model.topologyChanges():
            nodes = []
            coordinates = _hexa20Coordinates(0.0)
            for label, x in zip(model.reserveNodeNumbers(len(coordinates)), coordinates):
                node = Node(label, x)
                model.nodes[label] = node
                nodes.append(node)
            (elNumber,) = model.reserveElementNumbers(1)
            element = DisplacementElement("C3D20", elNumber)
            element.setNodes(nodes)
            model.createElement(element)

            slaveElements = [element]
            if mixedElements:
                # A hexa8 cube next to the hexa20 one, for a slave surface of two element types.
                hexa8Nodes = []
                for label, x in zip(model.reserveNodeNumbers(8), coordinates[:8]):
                    node = Node(label, x + np.array([_SIDE, 0.0, 0.0]))
                    model.nodes[label] = node
                    hexa8Nodes.append(node)
                (elNumber,) = model.reserveElementNumbers(1)
                hexa8 = DisplacementElement("C3D8", elNumber)
                hexa8.setNodes(hexa8Nodes)
                model.createElement(hexa8)
                slaveElements.append(hexa8)
                nodes += hexa8Nodes

            model.surfaces["slaveFace"] = {_YMIN: ElementSet("s", slaveElements)}
            slaveSetName, _ = buildContactFacets(model, "slaveFace", "slv", "midside", "facetConsistent", self.journal)

            rigidBody = generateDiscreteRigidBodyFromMeshFile(
                model, self.journal, "support", stlFile, rpCoordinate=np.array([0.5 * _SIDE, 0.0, 0.5 * _SIDE])
            )

        # Zeroed displacement and rotation fields, read by the contact search.
        for node in nodes + [rigidBody.rpNode]:
            node.fields["displacement"] = 3
        rigidBody.rpNode.fields["rotation"] = 3
        for fieldName in ("displacement", "rotation"):
            field = NodeField(fieldName, 3, NodeSet("all", nodes + [rigidBody.rpNode]))
            field.createFieldValueEntry("U")
            model.nodeFields[fieldName] = field

        return model, model.elementSets[slaveSetName], rigidBody

    def _constraint(self, model, slaveSurface, rigidBody, **options) -> RigidSurfaceContact:
        configuration = SurfaceToDiscreteRigidBodyPenaltySchema(
            penalty=options.pop("penalty", 1.0e4), searchDistance=options.pop("searchDistance", 1.0), **options
        )
        constraint = RigidSurfaceContact(
            "theContact", model, slaveSurface, rigidBody, self.journal, configuration=configuration
        )
        constraint.updateConnectivity(model)
        return constraint

    def _forces(self, constraint, U) -> np.ndarray:
        PExt = np.zeros(constraint.nDof)
        constraint.applyConstraint(U, np.zeros_like(U), PExt, None, None)
        return PExt

    def _assembledTangent(self, constraint, U) -> np.ndarray:
        """The tangent as one dense ``nDof x nDof`` matrix, scattered from the shared-RP layout."""

        flat = np.zeros(constraint.getVIJContributionSize())
        view = constraint.shapeVIJContribution(flat)
        constraint.applyConstraint(U, np.zeros_like(U), np.zeros(constraint.nDof), view, None)

        K = np.zeros((constraint.nDof, constraint.nDof))
        rp = np.arange(constraint.nSlaveDof, constraint.nDof)
        K[np.ix_(rp, rp)] += view.K_rprp
        for f, (first, m) in enumerate(constraint._slaveBlockOffsets()):
            slave = np.arange(first, first + m)
            K[np.ix_(slave, slave)] += view.K_ss[f]
            K[np.ix_(slave, rp)] += view.K_srp[f]
            K[np.ix_(rp, slave)] += view.K_rps[f]
        return K

    def test_tangent_is_the_negative_force_jacobian(self):
        """``K == -dPExt/dU`` by central differences, for a moved and rotated rigid body and both
        penalty laws -- except for the documented, omitted rotation-rotation block of the RP."""

        for contactType in ("linear", "quadratic"):
            model, slaveSurface, rigidBody = self._blockOnRigidSupport(penetration=0.05)
            constraint = self._constraint(model, slaveSurface, rigidBody, contactType=contactType)

            rng = np.random.default_rng(0)
            U = 1e-3 * rng.standard_normal(constraint.nDof)
            U[-3:] = [0.01, -0.005, 0.008]

            K = self._assembledTangent(constraint, U)

            h = 1e-7
            KNumeric = np.zeros_like(K)
            for i in range(constraint.nDof):
                UPlus, UMinus = U.copy(), U.copy()
                UPlus[i] += h
                UMinus[i] -= h
                KNumeric[:, i] = -(self._forces(constraint, UPlus) - self._forces(constraint, UMinus)) / (2.0 * h)

            rotation = slice(constraint.nDof - 3, constraint.nDof)
            K[rotation, rotation] = 0.0
            KNumeric[rotation, rotation] = 0.0

            self.assertTrue(np.all(constraint.getGaps() < 0.0), "every contact point must stay closed")
            scale = np.abs(K).max()
            np.testing.assert_allclose(K, KNumeric, rtol=0, atol=1e-6 * scale, err_msg=f"type={contactType}")

    def test_forces_balance_and_corners_are_pulled(self):
        """The contact forces on the block and on the rigid body are in equilibrium, and the corner
        nodes of the serendipity face carry tensile (negative) forces."""

        model, slaveSurface, rigidBody = self._blockOnRigidSupport(penetration=0.01)
        constraint = self._constraint(model, slaveSurface, rigidBody)
        PExt = self._forces(constraint, np.zeros(constraint.nDof))

        slaveForce = PExt[: constraint.nSlaveDof].reshape((-1, 3)).sum(axis=0)
        rigidBodyForce = PExt[constraint.nSlaveDof : constraint.nSlaveDof + 3]
        np.testing.assert_allclose(slaveForce + rigidBodyForce, 0.0, rtol=0, atol=1e-10)

        # A uniform penetration of 0.01 on a face of area 4: total force penalty * 0.01 * 4. STL files
        # store single-precision coordinates, so the rigid face sits at 0.01 only to ~1e-9.
        expectedForce = 1.0e4 * 0.01 * _SIDE**2
        self.assertAlmostEqual(slaveForce[1] / expectedForce, 1.0, places=6)

        nodalForces = constraint.getSlaveNodalNormalForces()
        self.assertEqual(np.sum(nodalForces < 0.0), 4, "the 4 corner nodes must be pulled")
        self.assertAlmostEqual(nodalForces.sum(), slaveForce[1], places=10)

    def test_open_rigid_surface_is_rejected(self):
        model, slaveSurface, rigidBody = self._blockOnRigidSupport(penetration=0.01, openSurface=True)
        with self.assertRaises(ValueError):
            self._constraint(model, slaveSurface, rigidBody)

    def test_invalid_options_are_rejected(self):
        model, slaveSurface, rigidBody = self._blockOnRigidSupport(penetration=0.01)
        with self.assertRaises(ValueError) as context:
            self._constraint(model, slaveSurface, rigidBody, penalty=0.0)
        self.assertIn("penalty must be positive", str(context.exception))

        with self.assertRaises(ValueError) as context:
            self._constraint(model, slaveSurface, rigidBody, sliding="finite")
        self.assertIn("only 'small' is implemented", str(context.exception))

        with self.assertRaises(ValueError) as context:
            RigidSurfaceContact("theContact", FEModel(2), slaveSurface, rigidBody, self.journal)
        self.assertIn("only implemented for 3D", str(context.exception))

    def test_mixed_element_types_are_rejected(self):
        model, slaveSurface, rigidBody = self._blockOnRigidSupport(penetration=0.01, mixedElements=True)
        with self.assertRaises(ValueError) as context:
            self._constraint(model, slaveSurface, rigidBody)
        self.assertIn("a single element type is required", str(context.exception))

    def test_search_distance_limits_the_assignment(self):
        """A point farther than searchDistance from the rigid surface is not assigned a triangle, and
        a point is never assigned a triangle whose plane it lies behind by more than searchDistance."""

        for gap in (5.0, 1.2):
            model, slaveSurface, rigidBody = self._blockOnRigidSupport(penetration=-gap)
            constraint = self._constraint(model, slaveSurface, rigidBody, searchDistance=1.0)
            np.testing.assert_array_equal(constraint._frozenBodyNormals, 0.0)
            self._forces(constraint, np.zeros(constraint.nDof))
            np.testing.assert_array_equal(constraint.getGaps(), 0.0)

        model, slaveSurface, rigidBody = self._blockOnRigidSupport(penetration=1.5)
        constraint = self._constraint(model, slaveSurface, rigidBody, searchDistance=1.0)
        self.assertTrue(np.all(constraint._frozenBodyNormals[:, 1] < 0.5), "the top face lies 1.5 above the points")

    def test_without_search_distance_every_point_is_assigned(self):
        model, slaveSurface, rigidBody = self._blockOnRigidSupport(penetration=-5.0)
        constraint = self._constraint(model, slaveSurface, rigidBody, searchDistance=None)
        np.testing.assert_array_equal(constraint._frozenBodyNormals, [[0.0, 1.0, 0.0]] * constraint.nPoints)
        self._forces(constraint, np.zeros(constraint.nDof))
        np.testing.assert_allclose(constraint.getGaps(), 5.0, rtol=1e-6)

    def test_overhanging_points_are_not_supported_by_an_extended_plane(self):
        """A cube overhanging the support's edge at x = 1, slightly above it: beyond the edge, the top
        and the side triangles share the closest point. The side triangle must win whatever the
        triangle order, so pushing the cube down loads only the points above the support."""

        for reverseTriangleOrder in (False, True):
            model, slaveSurface, rigidBody = self._blockOnRigidSupport(
                penetration=-0.01, supportXMax=1.0, reverseTriangleOrder=reverseTriangleOrder
            )
            constraint = self._constraint(model, slaveSurface, rigidBody)
            pointX = constraint.slave.currentPointCoordinates(model)[:, 0]
            overhanging = pointX > 1.0 + 1e-6
            self.assertTrue(overhanging.any() and (~overhanging).any())
            np.testing.assert_allclose(constraint._frozenBodyNormals[overhanging, 0], 1.0, atol=1e-6)

            U = np.zeros(constraint.nDof)
            U[1 : constraint.nSlaveDof : 3] = -0.03
            self._forces(constraint, U)
            gaps = constraint.getGaps()
            self.assertTrue(np.all(gaps[overhanging] > 0.0), f"reverseTriangleOrder={reverseTriangleOrder}")
            self.assertTrue(np.all(gaps[~overhanging] < 0.0), f"reverseTriangleOrder={reverseTriangleOrder}")

    def test_refresh_ignores_changes_that_do_not_touch_the_slave_surface(self):
        model, slaveSurface, rigidBody = self._blockOnRigidSupport(penetration=0.01)
        constraint = self._constraint(model, slaveSurface, rigidBody)
        self.assertFalse(constraint.refresh(model, ModelChange(kind=ModelChangeType.REFINEMENT)))

    def test_empty_slave_surface_carries_no_force(self):
        model, _, rigidBody = self._blockOnRigidSupport(penetration=0.01)
        constraint = self._constraint(model, ElementSet("empty", []), rigidBody)
        self.assertEqual(constraint.nDof, 6)
        np.testing.assert_array_equal(self._forces(constraint, np.zeros(constraint.nDof)), 0.0)

    def test_slave_surface_nodes_are_the_facet_nodes(self):
        model, slaveSurface, rigidBody = self._blockOnRigidSupport(penetration=0.01)
        constraint = self._constraint(model, slaveSurface, rigidBody)
        self.assertEqual(len(constraint.slaveSurfaceNodes), 8)
        self.assertEqual(len(constraint.getSlaveNodalNormalForces()), 8)


if __name__ == "__main__":
    unittest.main()
