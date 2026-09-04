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
"""Unit tests for the contact facet generator's per-node weighting options.

The analytic reference values are those of a *flat, straight-edged* quadratic face, for which the
midside triangulation's four corner triangles each have a quarter of the face area and the central
midside quad has half of it.
"""

import unittest

import numpy as np

import edelweissfe.utils.inputfileparser  # noqa: F401 bootstrap input language
from edelweissfe.constraints.nodetodeformablesurfacepenalty import (
    Constraint as ContactConstraint,
)
from edelweissfe.constraints.nodetodeformablesurfacepenalty import (
    NodeToDeformableSurfacePenaltySchema,
)
from edelweissfe.elements.contactsurfaceelement import Tria3ContactFacet
from edelweissfe.elements.displacementelement.element import DisplacementElement
from edelweissfe.generators.surfaceelementgenerator import buildContactFacets
from edelweissfe.journal.journal import Journal
from edelweissfe.models.femodel import FEModel
from edelweissfe.points.node import Node
from edelweissfe.sets.elementset import ElementSet

#: Edge-to-midside-node map of the hexa20 node ordering, local indices 8..19 in order.
_HEXA20_EDGES = (
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 0),
    (4, 5),
    (5, 6),
    (6, 7),
    (7, 4),
    (0, 4),
    (1, 5),
    (2, 6),
    (3, 7),
)

#: Side length of the test cube; a face therefore has area 4, matching the reference values below.
_SIDE = 2.0

#: The Ymin face (face number 1) of the hexa20 node ordering: corner then midside local indices.
_YMIN_CORNERS = (0, 1, 2, 3)
_YMIN_MIDSIDES = (8, 9, 10, 11)


class TestContactFacetNodalWeights(unittest.TestCase):
    def setUp(self):
        self.journal = Journal()

    def _modelWithOneHexa20(self) -> FEModel:
        """A single straight-edged hexa20 cube of side ``_SIDE``, with its Ymin face registered as
        the surface ``theSurface``."""

        corners = [
            np.array([0.0, 0.0, 0.0]),
            np.array([_SIDE, 0.0, 0.0]),
            np.array([_SIDE, 0.0, _SIDE]),
            np.array([0.0, 0.0, _SIDE]),
            np.array([0.0, _SIDE, 0.0]),
            np.array([_SIDE, _SIDE, 0.0]),
            np.array([_SIDE, _SIDE, _SIDE]),
            np.array([0.0, _SIDE, _SIDE]),
        ]
        coordinates = corners + [0.5 * (corners[a] + corners[b]) for a, b in _HEXA20_EDGES]

        model = FEModel(3)
        nodes = [Node(i + 1, x) for i, x in enumerate(coordinates)]
        for node in nodes:
            model.nodes[node.label] = node

        with model.topologyChanges():
            (elNumber,) = model.reserveElementNumbers(1)
            element = DisplacementElement("C3D20", elNumber)
            element.setNodes(nodes)
            model.createElement(element)
        model.surfaces["theSurface"] = {1: ElementSet("theFace", [element])}

        return model

    def _sharesPerNode(self, model: FEModel, facetsSetName: str) -> dict:
        """The tributary area of every facet node, assembled exactly as the contact constraint
        does."""

        shares = {}
        for facet in model.elementSets[facetsSetName]:
            for node, share in zip(facet.nodes, facet.nodalAreaShares):
                shares[node] = shares.get(node, 0.0) + share
        return shares

    def _facetSharesByRole(self, model: FEModel, nodalWeights: str) -> tuple:
        """Generate the Ymin face's facets under ``nodalWeights`` and return its (corner shares,
        midside shares) as arrays, plus the facet element set name."""

        with model.topologyChanges():
            facetsSetName, _ = buildContactFacets(model, "theSurface", "pfx", "midside", nodalWeights, self.journal)
        shares = self._sharesPerNode(model, facetsSetName)
        (sourceElement,) = model.surfaces["theSurface"][1]
        sourceNodes = sourceElement.nodes
        cornerShares = np.array([shares[sourceNodes[i]] for i in _YMIN_CORNERS])
        midsideShares = np.array([shares[sourceNodes[i]] for i in _YMIN_MIDSIDES])
        return cornerShares, midsideShares, facetsSetName

    def test_facetConsistent_shares_are_the_facet_tiling_lumping(self):
        """The default weighting: A/24 per corner, 5A/24 per midside node, for A = 4."""

        model = self._modelWithOneHexa20()
        cornerShares, midsideShares, _ = self._facetSharesByRole(model, "facetConsistent")

        np.testing.assert_allclose(cornerShares, 4.0 / 24.0)
        np.testing.assert_allclose(midsideShares, 5.0 * 4.0 / 24.0)
        self.assertAlmostEqual(cornerShares.sum() + midsideShares.sum(), 4.0)

    def test_serendipityOptimal_zeroes_the_corners_and_stays_resultant_exact(self):
        """The modified weighting: zero per corner, A/4 per midside node, same total area."""

        model = self._modelWithOneHexa20()
        cornerShares, midsideShares, facetsSetName = self._facetSharesByRole(model, "serendipityOptimal")

        np.testing.assert_allclose(cornerShares, 0.0)
        np.testing.assert_allclose(midsideShares, 4.0 / 4.0)
        self.assertAlmostEqual(cornerShares.sum() + midsideShares.sum(), 4.0)

        for facet in model.elementSets[facetsSetName]:
            self.assertTrue(np.all(facet.nodalAreaShares >= 0.0))

    def test_serendipityOptimal_halves_the_mismatch_against_the_consistent_loads(self):
        """The mismatch against the quad8 face's own consistent nodal loads (-A/12 per corner,
        +A/3 per midside) drops from A/8 to A/12 per node -- by a third, and that is the optimum
        over all resultant-exact, non-negative weightings."""

        area = 4.0
        consistentCorner, consistentMidside = -area / 12.0, area / 3.0

        default = self._facetSharesByRole(self._modelWithOneHexa20(), "facetConsistent")
        modified = self._facetSharesByRole(self._modelWithOneHexa20(), "serendipityOptimal")

        for cornerShares, midsideShares, expectedMismatch in (
            (*default[:2], area / 8.0),
            (*modified[:2], area / 12.0),
        ):
            np.testing.assert_allclose(cornerShares - consistentCorner, expectedMismatch)
            np.testing.assert_allclose(midsideShares - consistentMidside, -expectedMismatch)

    def test_serendipityOptimal_installs_a_force_preserving_transform_on_corner_triangles(self):
        """Only the four corner triangles carry a transform, and it is column-stochastic (so the
        transferred force is redistributed, never scaled)."""

        model = self._modelWithOneHexa20()
        with model.topologyChanges():
            facetsSetName, _ = buildContactFacets(
                model, "theSurface", "pfx", "midside", "serendipityOptimal", self.journal
            )
        facets = list(model.elementSets[facetsSetName])
        self.assertEqual(len(facets), 6)

        transformed = [facet for facet in facets if facet.weightTransform is not None]
        self.assertEqual(len(transformed), 4)
        for facet in transformed:
            np.testing.assert_allclose(facet.weightTransform.sum(axis=0), 1.0)
            # the corner (local index 1 of a (m_prev, c, m_next) triangle) receives nothing
            np.testing.assert_allclose(facet.weightTransform[1, :], 0.0)

    def test_facetConsistent_installs_no_transform(self):
        model = self._modelWithOneHexa20()
        with model.topologyChanges():
            facetsSetName, _ = buildContactFacets(
                model, "theSurface", "pfx", "midside", "facetConsistent", self.journal
            )
        for facet in model.elementSets[facetsSetName]:
            self.assertIsNone(facet.weightTransform)

    def test_serendipityOptimal_rejects_the_corner_triangulation(self):
        """With the corner reduction there are no midside nodes to reassign the corner shares to,
        which would silently zero every weight of a quadratic face."""

        model = self._modelWithOneHexa20()
        with self.assertRaises(ValueError) as ctx, model.topologyChanges():
            buildContactFacets(model, "theSurface", "pfx", "corner", "serendipityOptimal", self.journal)
        self.assertIn("requires triangulation='midside'", str(ctx.exception))

    def test_unknown_nodalWeights_is_rejected(self):
        model = self._modelWithOneHexa20()
        with self.assertRaises(ValueError) as ctx, model.topologyChanges():
            buildContactFacets(model, "theSurface", "pfx", "midside", "nonsense", self.journal)
        self.assertIn("nodalWeights 'nonsense' is not supported", str(ctx.exception))

    def test_transform_reassigns_the_corner_weight_of_the_contact_point(self):
        """End-to-end on the constraint: a slave whose closest point falls inside a corner triangle
        gets the corner's interpolation weight split equally onto the two adjacent midside nodes,
        so no force reaches the corner node."""

        model = self._modelWithOneHexa20()
        with model.topologyChanges():
            facetsSetName, _ = buildContactFacets(
                model, "theSurface", "pfx", "midside", "serendipityOptimal", self.journal
            )

        # The Ymin corner triangle (9, 1, 8) of the side-2 cube spans (2, 0, 1), (2, 0, 0) and
        # (1, 0, 0); its centroid is (5/3, 0, 1/3). Three slave nodes just outside the face (the
        # outward normal is -y), the first exactly above that centroid.
        slaveCoordinates = (
            np.array([5.0 / 3.0, -0.01, 1.0 / 3.0]),
            np.array([1.70, -0.01, 0.35]),
            np.array([1.75, -0.01, 0.30]),
        )
        slaveNodes = [Node(1000 + i, x) for i, x in enumerate(slaveCoordinates)]
        slaveFacet = Tria3ContactFacet("Tria3ContactFacet", 1000)
        slaveFacet.setNodes(slaveNodes)
        slaveFacet.initializeElement()

        constraint = ContactConstraint(
            "theContact",
            model,
            ElementSet("slave_facets", [slaveFacet]),
            model.elementSets[facetsSetName],
            self.journal,
            configuration=NodeToDeformableSurfacePenaltySchema(penalty=1.0, sliding="small", searchDistance=1.0),
        )
        constraint.updateConnectivity(model)

        for s, weights in enumerate(constraint._frozenWeights):
            self.assertIsNotNone(weights, f"slave {s} was not assigned a facet")
            self.assertAlmostEqual(weights.sum(), 1.0, msg="the transform must preserve the force")
            self.assertAlmostEqual(weights[1], 0.0, msg="the corner node must receive no weight")

        # the centroid slave: barycentric (1/3, 1/3, 1/3) -> (1/2, 0, 1/2)
        np.testing.assert_allclose(constraint._frozenWeights[0], [0.5, 0.0, 0.5], atol=1e-12)

    def test_transformed_master_facets_require_small_sliding(self):
        """A weight transform is variationally admissible only in the frozen-projection
        formulation; finite sliding must refuse it rather than half-support it."""

        model = self._modelWithOneHexa20()
        with model.topologyChanges():
            facetsSetName, _ = buildContactFacets(
                model, "theSurface", "pfx", "midside", "serendipityOptimal", self.journal
            )
        masterSurface = model.elementSets[facetsSetName]
        slaveSurface = ElementSet("slave_facets", [])

        for sliding, expectation in (("small", None), ("finite", "requires sliding=small")):
            configuration = NodeToDeformableSurfacePenaltySchema(penalty=1.0, sliding=sliding)
            if expectation is None:
                ContactConstraint(
                    "theContact", model, slaveSurface, masterSurface, self.journal, configuration=configuration
                )
            else:
                with self.assertRaises(ValueError) as ctx:
                    ContactConstraint(
                        "theContact", model, slaveSurface, masterSurface, self.journal, configuration=configuration
                    )
                self.assertIn(expectation, str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
