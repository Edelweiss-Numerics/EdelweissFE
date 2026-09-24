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

import edelweissfe.utils.inputfileparser  # noqa: F401 bootstrap input language
from edelweissfe.constraints.surfacetodiscreterigidbodypenalty import (
    Constraint as RigidSurfaceContact,
)
from edelweissfe.constraints.surfacetodiscreterigidbodypenalty import (
    SurfaceToDiscreteRigidBodyPenaltySchema,
)
from edelweissfe.constraints.test_surfacetodeformablesurfacepenalty import (
    _SIDE,
    _YMIN,
    _hexa20Coordinates,
)
from edelweissfe.elements.displacementelement.element import DisplacementElement
from edelweissfe.fields.nodefield import NodeField
from edelweissfe.generators.discreterigidbodygenerator import (
    generateDiscreteRigidBodyFromMeshFile,
)
from edelweissfe.generators.surfaceelementgenerator import buildContactFacets
from edelweissfe.journal.journal import Journal
from edelweissfe.models.femodel import FEModel
from edelweissfe.points.node import Node
from edelweissfe.sets.elementset import ElementSet
from edelweissfe.sets.nodeset import NodeSet


class TestSurfaceToDiscreteRigidBodyContact(unittest.TestCase):
    def setUp(self):
        self.journal = Journal()
        self.directory = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.directory.cleanup()

    def _blockOnRigidSupport(self, penetration: float, openSurface: bool = False) -> tuple:
        """A hexa20 cube on y in [0, 2], and a rigid box below it whose top face at y = ``penetration``
        overlaps the cube's Ymin face (the slave surface). The rigid body's reference point sits at
        the center of the contact face.

        Returns
        -------
        tuple
            ``(model, slaveSurface, rigidBody)``.
        """

        stlFile = os.path.join(self.directory.name, "support.stl")
        box = pv.Box(bounds=(-1.0, _SIDE + 1.0, -1.0, penetration, -1.0, _SIDE + 1.0)).triangulate()
        if openSurface:
            box = box.extract_cells(range(10)).extract_surface(algorithm="dataset_surface")
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

            model.surfaces["slaveFace"] = {_YMIN: ElementSet("s", [element])}
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


if __name__ == "__main__":
    unittest.main()
