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
"""Unit tests for the integrated (Gauss-point-to-segment) surface contact constraint.

The consistency of the tangent is checked here by finite differences rather than left to the
regression tests: the hexa20 contact patch test is very nearly a linear problem, so it converges in
two iterations even with a wrong tangent and would not expose one.
"""

import unittest

import numpy as np

import edelweissfe.utils.inputfileparser  # noqa: F401 bootstrap input language
from edelweissfe.constraints.surfacetodeformablesurfacepenalty import (
    Constraint as IntegratedContact,
)
from edelweissfe.constraints.surfacetodeformablesurfacepenalty import (
    SurfaceToDeformableSurfacePenaltySchema,
)
from edelweissfe.elements.contactsurfaceelement import Tria3ContactFacet
from edelweissfe.elements.displacementelement.element import DisplacementElement
from edelweissfe.generators.surfaceelementgenerator import buildContactFacets
from edelweissfe.journal.journal import Journal
from edelweissfe.models.femodel import FEModel
from edelweissfe.points.node import Node
from edelweissfe.sets.elementset import ElementSet

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

_SIDE = 2.0

#: Face numbers of the hexa20 node ordering, per the generator's face tables.
_YMIN, _YMAX = 1, 2


def _hexa20Coordinates(yOffset: float) -> list:
    """The 20 node coordinates of a side-``_SIDE`` cube whose y span starts at ``yOffset``.

    The corner ring order is boxGen's own -- (0,0), (0,S), (S,S), (S,0) in the (x, z) plane -- which
    is what the generator's face tables were verified against. Transposing two corners still yields
    a geometrically valid cube whose face areas and shape-function integrals are unchanged, but it
    reverses the face normals, so a contact fixture built that way reports penetration where there
    is separation. Hence the orientation assertion in the tests below.
    """

    corners = [
        np.array([0.0, yOffset, 0.0]),
        np.array([0.0, yOffset, _SIDE]),
        np.array([_SIDE, yOffset, _SIDE]),
        np.array([_SIDE, yOffset, 0.0]),
        np.array([0.0, yOffset + _SIDE, 0.0]),
        np.array([0.0, yOffset + _SIDE, _SIDE]),
        np.array([_SIDE, yOffset + _SIDE, _SIDE]),
        np.array([_SIDE, yOffset + _SIDE, 0.0]),
    ]
    return corners + [0.5 * (corners[a] + corners[b]) for a, b in _HEXA20_EDGES]


class TestIntegratedSurfaceContact(unittest.TestCase):
    def setUp(self):
        self.journal = Journal()

    def _twoBlockModel(self, penetration: float) -> tuple:
        """Two hexa20 cubes meeting at y = 0, the lower one overlapping the upper by
        ``penetration``, with the lower block's Ymax face as slave and the upper block's Ymin face
        as master.

        Returns
        -------
        tuple
            ``(model, slaveSurface, masterSurface)``.
        """

        model = FEModel(3)
        label = 1
        elements = {}
        with model.topologyChanges():
            for key, yOffset in (("upper", 0.0), ("lower", -_SIDE + penetration)):
                nodes = []
                for x in _hexa20Coordinates(yOffset):
                    node = Node(label, x)
                    model.nodes[label] = node
                    nodes.append(node)
                    label += 1
                (elNumber,) = model.reserveElementNumbers(1)
                element = DisplacementElement("C3D20", elNumber)
                element.setNodes(nodes)
                model.createElement(element)
                elements[key] = element

            model.surfaces["masterFace"] = {_YMIN: ElementSet("m", [elements["upper"]])}
            model.surfaces["slaveFace"] = {_YMAX: ElementSet("s", [elements["lower"]])}

            masterSetName, _ = buildContactFacets(
                model, "masterFace", "mst", "midside", "facetConsistent", self.journal
            )
            slaveSetName, _ = buildContactFacets(model, "slaveFace", "slv", "midside", "facetConsistent", self.journal)

        return model, model.elementSets[slaveSetName], model.elementSets[masterSetName]

    def _constraint(self, model, slaveSurface, masterSurface, **options) -> IntegratedContact:
        configuration = SurfaceToDeformableSurfacePenaltySchema(
            penalty=options.pop("penalty", 1.0e4),
            searchDistance=options.pop("searchDistance", 1.0),
            **options,
        )
        constraint = IntegratedContact(
            "theContact", model, slaveSurface, masterSurface, self.journal, configuration=configuration
        )
        constraint.updateConnectivity(model)
        return constraint

    def _forces(self, constraint, U) -> np.ndarray:
        PExt = np.zeros(constraint.nDof)
        constraint.applyConstraint(U, np.zeros_like(U), PExt, None, None)
        return PExt

    def _assembledTangent(self, constraint, U) -> np.ndarray:
        """The tangent as one dense ``nDof x nDof`` matrix, scattered from the per-point blocks."""

        flat = np.zeros(constraint.getVIJContributionSize())
        view = constraint.shapeVIJContribution(flat)
        PExt = np.zeros(constraint.nDof)
        constraint.applyConstraint(U, np.zeros_like(U), PExt, view, None)

        K = np.zeros((constraint.nDof, constraint.nDof))
        localOffset = 0
        for block in view.blocks:
            m = block.shape[0]
            idcs = np.arange(localOffset, localOffset + m)
            K[np.ix_(idcs, idcs)] += block
            localOffset += m
        return K

    def test_tangent_is_the_negative_force_jacobian(self):
        """``K == -dPExt/dU``, by central differences, for both penalty laws.

        Every contact point stays well inside contact over the perturbation, so the gap-activation
        switch never fires and the comparison is against a smooth branch.
        """

        for contactType in ("linear", "quadratic"):
            model, slaveSurface, masterSurface = self._twoBlockModel(penetration=0.05)
            constraint = self._constraint(model, slaveSurface, masterSurface, contactType=contactType)

            rng = np.random.default_rng(0)
            U = 1e-3 * rng.standard_normal(constraint.nDof)

            K = self._assembledTangent(constraint, U)

            h = 1e-8
            KNumeric = np.zeros_like(K)
            for i in range(constraint.nDof):
                UPlus, UMinus = U.copy(), U.copy()
                UPlus[i] += h
                UMinus[i] -= h
                KNumeric[:, i] = -(self._forces(constraint, UPlus) - self._forces(constraint, UMinus)) / (2.0 * h)

            scale = max(np.abs(K).max(), 1.0)
            np.testing.assert_allclose(K, KNumeric, atol=1e-5 * scale, err_msg=f"type={contactType}")

    def test_batched_and_per_point_paths_agree(self):
        """The batched evaluation must reproduce the per-point loop, which is the reference
        implementation. Without this the loop is unreachable in tests -- every ordinary model takes
        the batched path -- and the two could drift apart unnoticed.

        Checked for both penalty laws, on forces and on the assembled tangent, at a displacement
        state where some points are closed and some are open.
        """

        for contactType in ("linear", "quadratic"):
            model, slaveSurface, masterSurface = self._twoBlockModel(penetration=0.02)
            constraint = self._constraint(model, slaveSurface, masterSurface, contactType=contactType)

            rng = np.random.default_rng(7)
            U = 2e-2 * rng.standard_normal(constraint.nDof)

            self.assertIsNotNone(constraint._batched, "this model should take the batched path")
            gaps = None

            results = {}
            for label in ("batched", "loop"):
                if label == "loop":
                    constraint._batched = None
                P = self._forces(constraint, U)
                K = self._assembledTangent(constraint, U)
                results[label] = (P.copy(), K.copy(), constraint.getGaps().copy(), constraint.totalNormalForce)
                if gaps is None:
                    gaps = constraint.getGaps().copy()

            # a mixed active set, or the comparison proves little
            self.assertTrue((gaps < 0.0).any() and (gaps >= 0.0).any(), "expected a mixed active set")

            pB, kB, gB, fB = results["batched"]
            pL, kL, gL, fL = results["loop"]
            np.testing.assert_allclose(gB, gL, rtol=1e-13, atol=1e-15, err_msg=f"gaps, {contactType}")
            np.testing.assert_allclose(pB, pL, rtol=1e-11, atol=1e-13, err_msg=f"forces, {contactType}")
            np.testing.assert_allclose(kB, kL, rtol=1e-11, atol=1e-13, err_msg=f"tangent, {contactType}")
            self.assertAlmostEqual(fB, fL, delta=1e-10 * max(abs(fL), 1.0))

    def test_tangent_is_symmetric(self):
        """The frozen-projection formulation must produce a symmetric operator; a weight
        substitution that broke the gradient/distribution transpose relation would show up here."""

        model, slaveSurface, masterSurface = self._twoBlockModel(penetration=0.05)
        constraint = self._constraint(model, slaveSurface, masterSurface)
        K = self._assembledTangent(constraint, np.zeros(constraint.nDof))
        np.testing.assert_allclose(K, K.T, atol=1e-12 * max(np.abs(K).max(), 1.0))

    def test_explicit_path_matches_the_implicit_forces(self):
        """``applyConstraintExplicit`` must deliver exactly the forces the implicit path does."""

        model, slaveSurface, masterSurface = self._twoBlockModel(penetration=0.05)
        constraint = self._constraint(model, slaveSurface, masterSurface)
        U = np.zeros(constraint.nDof)

        PImplicit = np.zeros(constraint.nDof)
        flat = np.zeros(constraint.getVIJContributionSize())
        constraint.applyConstraint(U, np.zeros_like(U), PImplicit, constraint.shapeVIJContribution(flat), None)

        PExplicit = np.zeros(constraint.nDof)
        constraint.applyConstraintExplicit(U, np.zeros_like(U), PExplicit, None)

        np.testing.assert_array_equal(PImplicit, PExplicit)

    def test_corner_nodes_receive_tensile_force(self):
        """The whole point of the formulation: with the two faces pressed flat together, the corner
        nodes of the slave face carry a NEGATIVE (tensile) normal force, of exactly the consistent
        magnitude -p*A/12, while the midside nodes carry +p*A/3."""

        model, slaveSurface, masterSurface = self._twoBlockModel(penetration=0.05)
        constraint = self._constraint(model, slaveSurface, masterSurface, penalty=1.0e4)
        self._forces(constraint, np.zeros(constraint.nDof))

        pressures = constraint.getNormalPressures()
        np.testing.assert_allclose(pressures, pressures[0], rtol=1e-12)
        p = pressures[0]
        area = _SIDE**2

        forces = constraint.getSlaveNodalNormalForces()
        nodalForce = dict(zip(constraint.slaveSurfaceNodes, forces))
        sourceNodes = model.surfaces["slaveFace"][_YMAX][0].nodes
        # Ymax face of the hexa20 ordering: corners 4..7, midsides 12..15
        corners = np.array([nodalForce[sourceNodes[i]] for i in (4, 5, 6, 7)])
        midsides = np.array([nodalForce[sourceNodes[i]] for i in (12, 13, 14, 15)])

        np.testing.assert_allclose(corners, -p * area / 12.0, rtol=1e-10)
        np.testing.assert_allclose(midsides, p * area / 3.0, rtol=1e-10)
        self.assertTrue(np.all(corners < 0.0), "the corner nodal forces must be tensile")
        self.assertAlmostEqual(corners.sum() + midsides.sum(), p * area, delta=1e-8 * p * area)

    def test_open_gap_carries_no_force(self):
        """Separated surfaces produce no contact force at all."""

        model, slaveSurface, masterSurface = self._twoBlockModel(penetration=-0.01)
        constraint = self._constraint(model, slaveSurface, masterSurface)
        PExt = self._forces(constraint, np.zeros(constraint.nDof))
        np.testing.assert_array_equal(PExt, 0.0)
        self.assertEqual(constraint.totalNormalForce, 0.0)

    def test_quadrature_point_count_is_honoured(self):
        for nQuadraturePoints in (1, 3, 6):
            model, slaveSurface, masterSurface = self._twoBlockModel(penetration=0.05)
            constraint = self._constraint(model, slaveSurface, masterSurface, nQuadraturePoints=nQuadraturePoints)
            self.assertEqual(constraint.nPoints, len(slaveSurface) * nQuadraturePoints)
            self.assertEqual(len(constraint.getNormalPressures()), constraint.nPoints)

    def test_resultant_is_independent_of_the_quadrature_rule(self):
        """A uniform pressure is integrated exactly by every rule, so the transmitted resultant must
        not depend on the point count -- whereas the nodal *distribution* needs at least the 3-point
        rule to be exact."""

        resultants = []
        for nQuadraturePoints in (1, 3, 6):
            model, slaveSurface, masterSurface = self._twoBlockModel(penetration=0.05)
            constraint = self._constraint(model, slaveSurface, masterSurface, nQuadraturePoints=nQuadraturePoints)
            self._forces(constraint, np.zeros(constraint.nDof))
            resultants.append(constraint.totalNormalForce)
        np.testing.assert_allclose(resultants, resultants[0], rtol=1e-12)

    def test_finite_sliding_is_rejected(self):
        model, slaveSurface, masterSurface = self._twoBlockModel(penetration=0.05)
        with self.assertRaises(ValueError) as ctx:
            self._constraint(model, slaveSurface, masterSurface, sliding="finite")
        self.assertIn("only 'small' is implemented", str(ctx.exception))

    def test_unstamped_facets_are_rejected(self):
        """A facet built by hand, without a parent face, cannot be integrated over."""

        model, _, masterSurface = self._twoBlockModel(penetration=0.05)
        handMade = Tria3ContactFacet("Tria3ContactFacet", 9999)
        handMade.setNodes([Node(9000 + i, x) for i, x in enumerate(np.eye(3))])
        handMade.initializeElement()

        with self.assertRaises(ValueError) as ctx:
            self._constraint(model, ElementSet("hand_made", [handMade]), masterSurface)
        self.assertIn("carries no parent face", str(ctx.exception))

    def test_self_contact_is_rejected(self):
        model, slaveSurface, _ = self._twoBlockModel(penetration=0.05)
        with self.assertRaises(ValueError) as ctx:
            self._constraint(model, slaveSurface, slaveSurface)
        self.assertIn("self-contact is not supported", str(ctx.exception))

    def test_invalid_penalty_and_type_are_rejected(self):
        model, slaveSurface, masterSurface = self._twoBlockModel(penetration=0.05)
        with self.assertRaises(ValueError) as ctx:
            self._constraint(model, slaveSurface, masterSurface, penalty=0.0)
        self.assertIn("penalty must be positive", str(ctx.exception))

        with self.assertRaises(ValueError) as ctx:
            self._constraint(model, slaveSurface, masterSurface, contactType="cubic")
        self.assertIn("is not supported", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
