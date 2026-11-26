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
from unittest import mock

import numpy as np
from _hexa20cube import (  # noqa: E402  (tests/ is on sys.path)
    _SIDE,
    _YMAX,
    _YMIN,
    _hexa20Coordinates,
    twoBlockModel,
)

import edelweissfe.utils.inputfileparser  # noqa: F401 bootstrap input language
from edelweissfe.constraints.base.constraintbase import ConstraintBase
from edelweissfe.constraints.base.contactpointsatslavenodes import (
    ContactPointsAtSlaveNodes,
)
from edelweissfe.constraints.base.contactpointsonslavesurface import (
    ContactPointsOnSlaveSurface,
)
from edelweissfe.constraints.base.penaltylaw import PenaltyLaw, penaltyLaws
from edelweissfe.constraints.base.penaltysprings import PenaltySprings
from edelweissfe.constraints.surfacetodeformablesurfacepenalty import (
    Constraint as IntegratedContact,
)
from edelweissfe.constraints.surfacetodeformablesurfacepenalty import (
    ContactState,
    SurfaceToDeformableSurfacePenaltySchema,
)
from edelweissfe.elements.contactsurfaceelement import Tria3ContactFacet
from edelweissfe.elements.displacementelement.element import DisplacementElement
from edelweissfe.fields.nodefield import NodeField
from edelweissfe.generators.surfaceelementgenerator import buildContactFacets
from edelweissfe.journal.journal import Journal
from edelweissfe.models.femodel import FEModel
from edelweissfe.models.modelchange import ModelChange
from edelweissfe.models.modelchangeobserver import ModelChangeType
from edelweissfe.numerics.systemmatrixcontributor import countEntriesOfBlock
from edelweissfe.points.node import Node
from edelweissfe.sets.elementset import ElementSet
from edelweissfe.sets.nodeset import NodeSet
from edelweissfe.solvers.base.criticaltimestep import NextIncrement
from edelweissfe.timesteppers.timestep import TimeStep
from edelweissfe.utils.meshtools import currentNodeCoordinates
from edelweissfe.utils.residualtangentcheck import (
    finiteDifferenceTangent,
    residualContributionAndTangent,
)


def assertTheBoundIsTheOneOfTheTangent(constraint, U: np.ndarray, timeStep: TimeStep | None = None):
    """The bound of the stiffness the critical time step of ``NED`` takes from the constraint,
    :meth:`~edelweissfe.constraints.surfacetodeformablesurfacepenalty.Constraint.addBoundOfAbsoluteStiffnessTimes`
    (from the springs of the law), equals the row sums of the absolute value of its tangent (the default of
    :class:`~edelweissfe.constraints.base.constraintbase.ConstraintBase`), weighted by random columns: one spring per
    point, or springs along different axes, which act on different components, so the bound is the row sum itself.
    A bound that took other points as closed, another gap or another direction than the force would differ."""

    weights = np.random.default_rng(3).uniform(0.5, 2.0, constraint.nDof)
    bound, boundOfTheTangent = np.zeros(constraint.nDof), np.zeros(constraint.nDof)
    constraint.addBoundOfAbsoluteStiffnessTimes(U, np.zeros_like(U), weights, bound, timeStep)
    ConstraintBase.addBoundOfAbsoluteStiffnessTimes(
        constraint, U, np.zeros_like(U), weights, boundOfTheTangent, timeStep
    )
    assert boundOfTheTangent.max() > 0.0
    np.testing.assert_allclose(bound, boundOfTheTangent, rtol=1e-12, atol=1e-12 * boundOfTheTangent.max())


class TestIntegratedSurfaceContact(unittest.TestCase):
    def setUp(self):
        self.journal = Journal()

    def _constraint(self, model, slaveSurface, masterSurface, **options) -> IntegratedContact:
        configuration = SurfaceToDeformableSurfacePenaltySchema(
            penalty=options.pop("penalty", 1.0e4),
            searchDistance=options.pop("searchDistance", 1.0),
            **options,
        )
        if configuration.contactPoints == "nodes":
            contactPoints = ContactPointsAtSlaveNodes(slaveSurface, "theContact")
        else:
            contactPoints = ContactPointsOnSlaveSurface(slaveSurface, configuration.nQuadraturePoints, "theContact")
        constraint = IntegratedContact(
            "theContact", model, contactPoints, masterSurface, self.journal, configuration=configuration
        )
        constraint.updateConnectivity(model)
        return constraint

    def _forces(self, constraint, U) -> np.ndarray:
        """The contact forces on the nodes, the negative of the constraint's residual contribution."""

        P = np.zeros(constraint.nDof)
        constraint.computeKernels(U, np.zeros_like(U), P, None, None)
        return -P

    def _assembledTangent(self, constraint, U) -> np.ndarray:
        """The tangent as one dense ``nDof x nDof`` matrix, scattered with the constraint's VIJ pattern."""

        return residualContributionAndTangent(constraint, U, np.zeros_like(U))[1]

    def test_tangent_is_the_derivative_of_the_residual_contribution(self):
        """``K == dP/dU``, by central differences, for both penalty laws.

        Every contact point stays well inside contact over the perturbation, so the gap-activation
        switch never fires and the comparison is against a smooth branch.
        """

        for contactType, contactPoints in (
            ("linear", "quadraturePoints"),
            ("quadratic", "quadraturePoints"),
            ("linear", "nodes"),
            ("quadratic", "nodes"),
        ):
            model, slaveSurface, masterSurface = twoBlockModel(self.journal, penetration=0.05)
            constraint = self._constraint(
                model, slaveSurface, masterSurface, contactType=contactType, contactPoints=contactPoints
            )

            rng = np.random.default_rng(0)
            U = 1e-3 * rng.standard_normal(constraint.nDof)

            K = self._assembledTangent(constraint, U)
            dP_dU = finiteDifferenceTangent(constraint, U, np.zeros_like(U), h=1e-8)

            scale = max(np.abs(K).max(), 1.0)
            np.testing.assert_allclose(K, dP_dU, atol=1e-5 * scale, err_msg=f"type={contactType}, {contactPoints}")

            # The check above passes a P and a K negated together, so the direction is checked as well: the contact
            # pushes the faces apart, the slave face (the lower block's, at y = 0.05) down and the master face up.
            forces = self._forces(constraint, U).reshape((-1, 3))
            onTheSlaveFace = np.array([node.coordinates[1] > 0.025 for node in constraint.nodes])
            self.assertLess(forces[onTheSlaveFace, 1].sum(), 0.0, f"type={contactType}")
            self.assertGreater(forces[~onTheSlaveFace, 1].sum(), 0.0, f"type={contactType}")

    def test_tangent_is_symmetric(self):
        """The frozen-projection formulation must produce a symmetric operator; a weight
        substitution that broke the gradient/distribution transpose relation would show up here."""

        model, slaveSurface, masterSurface = twoBlockModel(self.journal, penetration=0.05)
        constraint = self._constraint(model, slaveSurface, masterSurface)
        K = self._assembledTangent(constraint, np.zeros(constraint.nDof))
        np.testing.assert_allclose(K, K.T, atol=1e-12 * max(np.abs(K).max(), 1.0))

    def test_explicit_path_matches_the_implicit_forces(self):
        """``computeKernelsExplicit`` must deliver exactly the forces the implicit path does."""

        model, slaveSurface, masterSurface = twoBlockModel(self.journal, penetration=0.05)
        constraint = self._constraint(model, slaveSurface, masterSurface)
        U = np.zeros(constraint.nDof)

        PImplicit = np.zeros(constraint.nDof)
        flat = np.zeros(countEntriesOfBlock(constraint))
        constraint.computeKernels(U, np.zeros_like(U), PImplicit, constraint.shapeBlock(flat), None)

        PExplicit = np.zeros(constraint.nDof)
        constraint.computeKernelsExplicit(U, np.zeros_like(U), PExplicit, None)

        np.testing.assert_array_equal(PImplicit, PExplicit)

    def test_corner_nodes_receive_tensile_force(self):
        """The whole point of the formulation: with the two faces pressed flat together, the corner
        nodes of the slave face carry a NEGATIVE (tensile) normal force, of exactly the consistent
        magnitude -p*A/12, while the midside nodes carry +p*A/3."""

        model, slaveSurface, masterSurface = twoBlockModel(self.journal, penetration=0.05)
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

    def test_unassigned_points_report_no_stale_gap(self):
        """A point that loses its master facet must report gap 0, not the value it had while still
        assigned.

        ``_gapCurrent`` backs the public :meth:`getGaps`, whose callers cannot mask unassigned
        entries because they do not know the assignment, and the connectivity diagnostic counts
        negative entries as 'closed'. Without the reset, a point that penetrated and then dropped
        out of the search radius would keep being reported as closed forever.
        """

        model, slaveSurface, masterSurface = twoBlockModel(self.journal, penetration=0.05)
        field = self._addDisplacementField(model)
        constraint = self._constraint(model, slaveSurface, masterSurface, searchDistance=1.0)
        self._forces(constraint, np.zeros(constraint.nDof))

        self.assertTrue((constraint.getGaps() < 0.0).any(), "expected penetrating points to start")

        # Pull the slave block away, beyond the search distance: a point that has left the master body there is free
        # in the update that sees it leave.
        for node in slaveSurface.extractNodeSet():
            field["U"][field._indicesOfNodesInArray[node]] = [0.0, -2.0, 0.0]
        constraint.updateConnectivity(model)
        self.assertTrue((constraint.stateOfPoints == ContactState.FREE).all())

        self.assertTrue((constraint._assignedFacetIdx < 0).all(), "expected no assignments")
        self.assertFalse(constraint.withinReach)
        np.testing.assert_array_equal(
            constraint.getGaps(), 0.0, err_msg="unassigned points must not retain a stale gap"
        )

    def _twoElementMasterModel(self, penetration: float) -> tuple:
        """Two hexa20 cubes side by side in x forming the master surface, and one hexa20 cube
        straddling their seam as the slave, overlapping by ``penetration``.

        The point of the second master element is that its Ymin face is a *different* parent face:
        a contact point sliding in +x crosses from one parent face to the other, which is the
        reassignment that :meth:`test_reprojection_across_parent_faces_leaves_the_gap_unchanged`
        pins. The two cubes do not share nodes -- their coincident faces are topologically
        separate -- which is exactly the situation the closest-point search sees on a real mesh
        once each element has contributed its own facets.

        Returns
        -------
        tuple
            ``(model, slaveSurface, masterSurface)``.
        """

        model = FEModel(3)
        label = 1
        elements = {}
        with model.topology.changes():
            for key, shift in (
                ("masterLeft", np.array([0.0, 0.0, 0.0])),
                ("masterRight", np.array([_SIDE, 0.0, 0.0])),
                ("slave", np.array([0.5 * _SIDE, -_SIDE + penetration, 0.0])),
            ):
                nodes = []
                for x in _hexa20Coordinates(0.0):
                    node = Node(label, x + shift)
                    model.nodes[label] = node
                    nodes.append(node)
                    label += 1
                (elNumber,) = model.topology.reserveElementNumbers(1)
                element = DisplacementElement("C3D20", elNumber)
                element.setNodes(nodes)
                model.createElement(element)
                elements[key] = element

            model.surfaces["masterFace"] = {_YMIN: ElementSet("m", [elements["masterLeft"], elements["masterRight"]])}
            model.surfaces["slaveFace"] = {_YMAX: ElementSet("s", [elements["slave"]])}

            masterSetName, _ = buildContactFacets(
                model, "masterFace", "mst", "midside", "facetConsistent", self.journal
            )
            slaveSetName, _ = buildContactFacets(model, "slaveFace", "slv", "midside", "facetConsistent", self.journal)

        return model, model.elementSets[slaveSetName], model.elementSets[masterSetName]

    def _addDisplacementField(self, model) -> NodeField:
        """Give ``model`` a zeroed displacement field, so the constraint's connectivity search can
        be re-run at a configuration other than the reference one."""

        for node in model.nodes.values():
            node.fields["displacement"] = model.domainSize
        field = NodeField("displacement", model.domainSize, NodeSet("all", list(model.nodes.values())))
        model.nodeFields["displacement"] = field
        return field

    def _gapsFromFrozenProjection(self, constraint, model) -> np.ndarray:
        """The gap of every assigned point at the model's *current* configuration, evaluated from
        whatever projection is frozen right now.

        Deliberately independent of :meth:`computeKernels`: it takes the frozen data directly, so
        calling it either side of an ``updateConnectivity`` isolates the effect of the re-projection
        from everything else.
        """

        slavePoints = constraint.slave.currentPointCoordinates(model)
        gaps = np.full(constraint.nPoints, np.nan)
        for p in range(constraint.nPoints):
            facetIdx = constraint._assignedFacetIdx[p]
            if facetIdx < 0:
                continue
            masterCoords = currentNodeCoordinates(
                constraint.facetElements[facetIdx].parentFaceNodes,
                model,
                constraint._masterParentRefCoords[facetIdx],
            )
            masterPoint = constraint._frozenMasterShapeFunctions[p] @ masterCoords
            gaps[p] = constraint._frozenNormals[p] @ (slavePoints[p] - masterPoint)
        return gaps

    def _slideAndReproject(self, constraint, model, field, slaveNodes, offset: np.ndarray) -> tuple:
        """Translate ``slaveNodes`` tangentially by ``offset``, re-run the connectivity search, and
        return ``(assignmentBefore, assignmentAfter, gapsBefore, gapsAfter)``, both gap fields taken
        at the *same* (slid) configuration.

        The master surface here is flat and undeformed and the slide is in its plane, so the normal
        gap is a geometric invariant of the slide: ``n`` is constant and ``n . x`` is unchanged for
        a tangential motion of either surface. Whatever the search then picks -- a different facet,
        a different parent face, a different parametric location within one -- must reproduce the
        same gap exactly. This is the sharp form of the statement that re-projection is a change of
        *representation*, not of physics.
        """

        for node in slaveNodes:
            field["U"][field._indicesOfNodesInArray[node]] = offset

        assignmentBefore = list(constraint._assignedFacetIdx)
        gapsBefore = self._gapsFromFrozenProjection(constraint, model)
        constraint.updateConnectivity(model)
        return (
            assignmentBefore,
            list(constraint._assignedFacetIdx),
            gapsBefore,
            self._gapsFromFrozenProjection(constraint, model),
        )

    def test_reprojection_within_one_parent_face_leaves_the_gap_unchanged(self):
        """Sliding the slave tangentially reassigns points among the facets tiling one parent face,
        and the gap field is unchanged to the last bit.

        This is the benign half of the reassignment behaviour, and it is worth pinning because the
        mapping it exercises -- clamped facet barycentrics through ``vertexParametricCoords`` into
        the parent face's own parametric space -- is the one place a facet swap could silently move
        the evaluation point. A single hexa20 face is tiled by six Tria3 facets, so this happens
        without any second element being involved.
        """

        model, slaveSurface, masterSurface = twoBlockModel(self.journal, penetration=0.01)
        field = self._addDisplacementField(model)
        constraint = self._constraint(model, slaveSurface, masterSurface, searchDistance=1.0)
        slaveNodes = [node for node in slaveSurface.extractNodeSet()]

        before, after, gapsBefore, gapsAfter = self._slideAndReproject(
            constraint, model, field, slaveNodes, np.array([0.31, 0.0, 0.17])
        )

        reassigned = [p for p in range(constraint.nPoints) if before[p] != after[p]]
        self.assertTrue(reassigned, "the slide must actually reassign facets or the test is vacuous")
        for p in reassigned:
            self.assertEqual(
                constraint.facetElements[before[p]].parentFaceNodes,
                constraint.facetElements[after[p]].parentFaceNodes,
                "a single-element master offers only one parent face to reassign within",
            )

        np.testing.assert_array_equal(
            gapsAfter, gapsBefore, err_msg="re-projecting within one parent face must not move the gap"
        )

    def test_reprojection_across_parent_faces_leaves_the_gap_unchanged(self):
        """A point sliding from one master element's parent face onto the next one's reproduces the
        same gap exactly.

        This is the reassignment that actually occurs in service -- 166 times over the
        NEDSurfaceContact run -- and the one that makes an explicit trajectory a step function of
        rounding, because *which* side of the seam a borderline point lands on is decided at the
        1e-16 level. That sensitivity is legitimate only if the two branches agree where they meet,
        which is what this asserts. On a flat master they agree exactly; what remains in a deformed
        model is the genuine kink between two non-coplanar element faces, i.e. the frozen facet
        normal changing -- a property of facet-based contact, not of this mapping. See the header of
        testfiles/edelweiss-only/NEDSurfaceContact/test.inp.
        """

        model, slaveSurface, masterSurface = self._twoElementMasterModel(penetration=0.01)
        field = self._addDisplacementField(model)
        constraint = self._constraint(model, slaveSurface, masterSurface, searchDistance=1.0)
        slaveNodes = [node for node in slaveSurface.extractNodeSet()]

        before, after, gapsBefore, gapsAfter = self._slideAndReproject(
            constraint, model, field, slaveNodes, np.array([0.37, 0.0, 0.0])
        )

        crossings = [
            p
            for p in range(constraint.nPoints)
            if before[p] != after[p]
            and constraint.facetElements[before[p]].parentFaceNodes
            != constraint.facetElements[after[p]].parentFaceNodes
        ]
        self.assertTrue(crossings, "the slide must carry points across the seam or the test is vacuous")

        np.testing.assert_array_equal(
            gapsAfter, gapsBefore, err_msg="crossing a parent-face boundary must not move the gap"
        )

    def test_getGaps_returns_a_copy(self):
        """Mutating the returned array must not corrupt the state a later Uzawa update would read."""

        model, slaveSurface, masterSurface = twoBlockModel(self.journal, penetration=0.05)
        constraint = self._constraint(model, slaveSurface, masterSurface)
        self._forces(constraint, np.zeros(constraint.nDof))

        gaps = constraint.getGaps()
        before = gaps.copy()
        gaps[:] = 1234.0
        np.testing.assert_array_equal(constraint.getGaps(), before)

    def test_open_gap_carries_no_force(self):
        """Separated surfaces produce no contact force at all."""

        model, slaveSurface, masterSurface = twoBlockModel(self.journal, penetration=-0.01)
        constraint = self._constraint(model, slaveSurface, masterSurface)
        forces = self._forces(constraint, np.zeros(constraint.nDof))
        np.testing.assert_array_equal(forces, 0.0)
        self.assertEqual(constraint.totalNormalForce, 0.0)

    def test_quadrature_point_count_is_honoured(self):
        for nQuadraturePoints in (1, 3, 6):
            model, slaveSurface, masterSurface = twoBlockModel(self.journal, penetration=0.05)
            constraint = self._constraint(model, slaveSurface, masterSurface, nQuadraturePoints=nQuadraturePoints)
            self.assertEqual(constraint.nPoints, len(slaveSurface) * nQuadraturePoints)
            self.assertEqual(len(constraint.getNormalPressures()), constraint.nPoints)

    def test_resultant_is_independent_of_the_quadrature_rule(self):
        """A uniform pressure is integrated exactly by every rule, so the transmitted resultant must
        not depend on the point count -- whereas the nodal *distribution* needs at least the 3-point
        rule to be exact."""

        resultants = []
        for nQuadraturePoints in (1, 3, 6):
            model, slaveSurface, masterSurface = twoBlockModel(self.journal, penetration=0.05)
            constraint = self._constraint(model, slaveSurface, masterSurface, nQuadraturePoints=nQuadraturePoints)
            self._forces(constraint, np.zeros(constraint.nDof))
            resultants.append(constraint.totalNormalForce)
        np.testing.assert_allclose(resultants, resultants[0], rtol=1e-12)

    def test_finite_sliding_is_rejected(self):
        model, slaveSurface, masterSurface = twoBlockModel(self.journal, penetration=0.05)
        with self.assertRaises(ValueError) as ctx:
            self._constraint(model, slaveSurface, masterSurface, sliding="finite")
        self.assertIn("only 'small' is implemented", str(ctx.exception))

    def test_unstamped_facets_are_rejected(self):
        """A facet built by hand, without a parent face, cannot be integrated over."""

        model, _, masterSurface = twoBlockModel(self.journal, penetration=0.05)
        handMade = Tria3ContactFacet("Tria3ContactFacet", 9999)
        handMade.setNodes([Node(9000 + i, x) for i, x in enumerate(np.eye(3))])
        handMade.initializeElement()

        with self.assertRaises(ValueError) as ctx:
            self._constraint(model, ElementSet("hand_made", [handMade]), masterSurface)
        self.assertIn("carries no parent face", str(ctx.exception))

    def test_self_contact_is_rejected(self):
        model, slaveSurface, _ = twoBlockModel(self.journal, penetration=0.05)
        with self.assertRaises(ValueError) as ctx:
            self._constraint(model, slaveSurface, slaveSurface)
        self.assertIn("self-contact is not supported", str(ctx.exception))

    def test_serendipity_weighted_surfaces_are_rejected(self):
        """A surface generated with nodalWeights='serendipityOptimal' must be refused, not silently
        ignored.

        There is no code path here that could honour a per-node weighting: the pressure is
        distributed with the parent face's shape functions at the quadrature points. Accepting the
        surface would give a different answer from the node-based constraint on identical input,
        with nothing in the log to say why -- and the corner mismatch the weighting exists to
        minimise is one this formulation removes outright.
        """

        model, slaveSurface, masterSurface = twoBlockModel(
            self.journal, penetration=0.05, nodalWeights="serendipityOptimal"
        )
        with self.assertRaises(ValueError) as ctx:
            self._constraint(model, slaveSurface, masterSurface)
        self.assertIn("cannot honour", str(ctx.exception))
        self.assertIn("nodalWeights='facetConsistent'", str(ctx.exception))

    def test_invalid_penalty_and_type_are_rejected(self):
        model, slaveSurface, masterSurface = twoBlockModel(self.journal, penetration=0.05)
        with self.assertRaises(ValueError) as ctx:
            self._constraint(model, slaveSurface, masterSurface, penalty=0.0)
        self.assertIn("penalty must be positive", str(ctx.exception))

        with self.assertRaises(ValueError) as ctx:
            self._constraint(model, slaveSurface, masterSurface, contactType="cubic")
        self.assertIn("is not supported", str(ctx.exception))

    def test_nodes_and_quadrature_points_transmit_the_same_resultant(self):
        """A uniform overlap of two flat faces: the slave nodes, each standing for its tributary area, and the
        quadrature points transmit the same resultant, the penalty times the area times the overlap."""

        resultants = {}
        for contactPoints in ("nodes", "quadraturePoints"):
            model, slaveSurface, masterSurface = twoBlockModel(self.journal, penetration=0.05)
            constraint = self._constraint(model, slaveSurface, masterSurface, contactPoints=contactPoints)
            self._forces(constraint, np.zeros(constraint.nDof))
            resultants[contactPoints] = constraint.totalNormalForce
            np.testing.assert_allclose(
                constraint.contactForceOnSlave + constraint.contactForceOnMaster, 0.0, atol=1e-10
            )
        self.assertAlmostEqual(resultants["nodes"], -1.0e4 * _SIDE**2 * 0.05, delta=1e-8)
        self.assertAlmostEqual(resultants["quadraturePoints"], resultants["nodes"], delta=1e-8)

    def test_a_point_that_tunnels_through_the_search_distance_gets_no_force(self):
        """A point out of reach at one update and inside the master body at the next has not been seen
        approaching: it is found inside, gets no force, and is reported -- the closest facet of a point inside a
        body is no indication of the way out."""

        model, slaveSurface, masterSurface = twoBlockModel(self.journal, penetration=-0.5)
        field = self._addDisplacementField(model)
        constraint = self._constraint(model, slaveSurface, masterSurface, searchDistance=0.1)
        self.assertTrue((constraint.stateOfPoints == ContactState.FREE).all())

        for node in slaveSurface.extractNodeSet():
            field["U"][field._indicesOfNodesInArray[node]] = [0.0, 0.52, 0.0]
        constraint.updateConnectivity(model)
        self.assertEqual(constraint.numberOfPointsFoundInside, constraint.nPoints)
        self.assertFalse(constraint.withinReach)
        self.assertEqual(constraint.nDof, 0)

    def test_an_overlap_deeper_than_the_search_distance_at_the_first_update_is_found_inside(self):
        """At the first update, the bodies are as the model gives them: every point is searched at any depth, and an
        overlap within the search distance is contact. A point that overlaps deeper has not been seen outside: it is
        found inside, gets no force, and is reported -- whether its overlap would be pushed apart or adjusted."""

        for initialOverlap, forceWithin in (("pushApart", -1.0e4 * _SIDE**2 * 0.05), ("adjust", 0.0)):
            model, slaveSurface, masterSurface = twoBlockModel(self.journal, penetration=0.05)
            with mock.patch.object(self.journal, "message") as message:
                deep = self._constraint(
                    model, slaveSurface, masterSurface, searchDistance=0.01, initialOverlap=initialOverlap
                )
            self.assertTrue((deep.stateOfPoints == ContactState.FOUND_INSIDE).all(), initialOverlap)
            self.assertFalse(deep.withinReach)
            np.testing.assert_array_equal(deep._initialOverlap, 0.0)
            self.assertTrue(
                any(
                    "deeper than the search distance at the first contact search" in c.args[0]
                    for c in message.mock_calls
                )
            )

            within = self._constraint(
                model, slaveSurface, masterSurface, searchDistance=0.1, initialOverlap=initialOverlap
            )
            self.assertTrue((within.stateOfPoints == ContactState.IN_CONTACT).all(), initialOverlap)
            self._forces(within, np.zeros(within.nDof))
            self.assertAlmostEqual(within.totalNormalForce, forceWithin, delta=1e-9, msg=initialOverlap)

    def test_a_point_found_inside_counts_until_it_is_seen_outside_also_deeper_than_the_search_distance(self):
        """A point found inside that goes on, deeper than the search distance, is out of sight of every facet. It has
        not left the master body: it still counts as found inside, so that a count of zero after a run means that no
        point is inside unseen. Seen outside again within the search distance, it approaches."""

        model, slaveSurface, masterSurface = twoBlockModel(self.journal, penetration=-0.5)
        field = self._addDisplacementField(model)
        constraint = self._constraint(model, slaveSurface, masterSurface, searchDistance=0.1)

        def moveTheSlaveFaceUpBy(dy: float):
            for node in slaveSurface.extractNodeSet():
                field["U"][field._indicesOfNodesInArray[node]] = [0.0, dy, 0.0]
            constraint.updateConnectivity(model)

        # tunnelled 0.02 into the master body, then on to 0.25 (beyond the search distance), and back to 0.02
        for dy in (0.52, 0.75, 0.52):
            moveTheSlaveFaceUpBy(dy)
            self.assertTrue((constraint.stateOfPoints == ContactState.FOUND_INSIDE).all(), f"dy={dy}")
            self.assertEqual(constraint.numberOfPointsFoundInside, constraint.nPoints)
            self.assertFalse(constraint.withinReach)

        moveTheSlaveFaceUpBy(0.45)
        self.assertTrue((constraint.stateOfPoints == ContactState.APPROACHING).all())
        self.assertEqual(constraint.numberOfPointsFoundInside, 0)

    def test_a_point_that_leaves_its_contact_beyond_the_search_distance_is_free_at_once(self):
        """A point in contact is not lost to the search distance; one that has left the master body is. Beyond the
        search distance it is free, and the constraint out of reach, in the update that sees it leave."""

        model, slaveSurface, masterSurface = twoBlockModel(self.journal, penetration=0.05)
        field = self._addDisplacementField(model)
        constraint = self._constraint(model, slaveSurface, masterSurface, searchDistance=0.1)
        self.assertTrue((constraint.stateOfPoints == ContactState.IN_CONTACT).all())

        for node in slaveSurface.extractNodeSet():
            field["U"][field._indicesOfNodesInArray[node]] = [0.0, -0.3, 0.0]
        constraint.updateConnectivity(model)
        self.assertTrue((constraint.stateOfPoints == ContactState.FREE).all())
        self.assertFalse(constraint.withinReach)
        self.assertEqual(constraint.nDof, 0)

    def test_restart_data_continues_the_pairing(self):
        """The frozen projection and the pairing states written to a checkpoint are adopted by a constraint on the
        same model."""

        model, slaveSurface, masterSurface = twoBlockModel(self.journal, penetration=0.05)
        constraint = self._constraint(model, slaveSurface, masterSurface)
        U = 1e-3 * np.random.default_rng(3).standard_normal(constraint.nDof)
        forces = self._forces(constraint, U)

        restored = self._constraint(model, slaveSurface, masterSurface, searchDistance=1e-12)
        restored.setRestartData(constraint.getRestartData())
        np.testing.assert_array_equal(restored.stateOfPoints, constraint.stateOfPoints)
        np.testing.assert_array_equal(self._forces(restored, U), forces)

    def test_an_initial_overlap_is_ramped_out_over_the_ramp_time(self):
        """With ``initialOverlapRampTime``, the overlap at the first search is not counted at first and counted fully
        after the ramp time, starting and ending at rest: no force at the start, half of it half way, all of it at
        the end. The offset is a constant of the gap, so the tangent stays the derivative of the residual
        contribution; and a checkpoint carries it."""

        def at(time: float) -> TimeStep:
            return TimeStep(1, 0.0, 0.0, 0.0, time, time)

        model, slaveSurface, masterSurface = twoBlockModel(self.journal, penetration=0.05)
        constraint = self._constraint(model, slaveSurface, masterSurface, initialOverlapRampTime=2.0)
        np.testing.assert_allclose(constraint._initialOverlap, -0.05, rtol=1e-12)
        U = np.zeros(constraint.nDof)

        fullForce = -1.0e4 * _SIDE**2 * 0.05
        for time, fraction in ((0.0, 0.0), (0.5, 0.25 - 1.0 / (2.0 * np.pi)), (1.0, 0.5), (2.0, 1.0), (3.0, 1.0)):
            P = np.zeros(constraint.nDof)
            constraint.computeKernels(U, U, P, None, at(time))
            self.assertAlmostEqual(constraint.totalNormalForce, fraction * fullForce, delta=1e-9, msg=f"t={time}")

        # Overlapping further than at the start, the point is closed before the end of the ramp.
        U = 1e-3 * np.random.default_rng(1).standard_normal(constraint.nDof)
        K = residualContributionAndTangent(constraint, U, np.zeros_like(U), at(1.0))[1]
        dP_dU = finiteDifferenceTangent(constraint, U, np.zeros_like(U), h=1e-8, timeStep=at(1.0))
        np.testing.assert_allclose(K, dP_dU, atol=1e-5 * max(np.abs(K).max(), 1.0))

        # Only the first search records the overlap.
        constraint.updateConnectivity(model)
        np.testing.assert_allclose(constraint._initialOverlap, -0.05, rtol=1e-12)

        restored = self._constraint(model, slaveSurface, masterSurface, initialOverlapRampTime=2.0)
        restored.setRestartData(constraint.getRestartData())
        for c in (constraint, restored):
            c.computeKernels(U, np.zeros_like(U), np.zeros(c.nDof), None, at(1.0))
        self.assertEqual(restored.totalNormalForce, constraint.totalNormalForce)

    def test_over_the_next_increment_a_point_that_could_close_has_its_spring(self):
        """The slave face lies 0.05 below the master face: open now, its bound is zero. Over a next increment in which
        the slave moves up by 0.1, from its velocity or from its acceleration, every point closes within it, and the
        bound over it is the one at its end, with the gradients of now (the same here); over one in which it moves up by
        0.02, or down, no point closes (``NED``'s ``follow-critical-time-step``,
        :func:`~edelweissfe.constraints.base.penaltysprings.smallestGapsWithin`)."""

        model, slaveSurface, masterSurface = twoBlockModel(self.journal, -0.05)
        constraint = self._constraint(model, slaveSurface, masterSurface)
        nSlaveDof = constraint._nSlaveDof
        zero = np.zeros(constraint.nDof)
        weights = np.random.default_rng(3).uniform(0.5, 2.0, constraint.nDof)
        timeStep = TimeStep(1, 0.1, 0.1, 0.1, 0.1, 0.1)

        def slaveMovedUpBy(motion: float) -> np.ndarray:
            dU = np.zeros(constraint.nDof)
            dU[:nSlaveDof] = np.tile((0.0, motion, 0.0), nSlaveDof // 3)
            return dU

        def boundOver(nextIncrement: NextIncrement | None, U: np.ndarray = zero) -> np.ndarray:
            bound = np.zeros(constraint.nDof)
            constraint.addBoundOfAbsoluteStiffnessTimes(U, zero, weights, bound, timeStep, nextIncrement)
            return bound

        self.assertFalse(boundOver(None).any())
        closing = boundOver(NextIncrement(slaveMovedUpBy(0.1), zero, timeStep))
        atTheEnd = boundOver(None, slaveMovedUpBy(0.1))
        self.assertGreater(atTheEnd.max(), 0.0)
        np.testing.assert_allclose(closing, atTheEnd, rtol=1e-12)
        accelerating = boundOver(NextIncrement(zero, slaveMovedUpBy(0.1), timeStep))
        np.testing.assert_allclose(accelerating, atTheEnd, rtol=1e-12)
        self.assertFalse(boundOver(NextIncrement(slaveMovedUpBy(0.02), zero, timeStep)).any())
        self.assertFalse(boundOver(NextIncrement(slaveMovedUpBy(-0.1), zero, timeStep)).any())

    def test_the_bound_of_the_stiffness_is_the_one_of_the_tangent_also_during_the_ramp(self):
        """For both kinds of contact points: at the start of the ramp of an initial overlap only the points that
        overlap further than at the first search are closed, and the bound of the stiffness counts just them; later,
        with more of the overlap counted, all."""

        def at(time: float) -> TimeStep:
            return TimeStep(1, 0.0, 0.0, 0.0, time, time)

        for contactPoints in ("quadraturePoints", "nodes"):
            model, slaveSurface, masterSurface = twoBlockModel(self.journal, penetration=0.05)
            constraint = self._constraint(
                model, slaveSurface, masterSurface, initialOverlapRampTime=2.0, contactPoints=contactPoints
            )
            U = 1e-3 * np.random.default_rng(2).standard_normal(constraint.nDof)
            for time in (0.0, 1.0):
                constraint.computeKernels(U, np.zeros_like(U), np.zeros(constraint.nDof), None, at(time))
                closed = constraint.numberOfPointsWithASpring
                self.assertEqual(closed < constraint.nPoints, time == 0.0, f"{contactPoints}, t={time}: {closed}")
                assertTheBoundIsTheOneOfTheTangent(constraint, U, at(time))

    def _slaveDisplacedBy(self, constraint, displacement: np.ndarray) -> np.ndarray:
        """The block's solution with the contact points' nodes displaced uniformly, the master at rest."""

        U = np.zeros(constraint.nDof)
        U[: constraint._nSlaveDof] = np.tile(displacement, constraint._nSlaveDof // 3)
        return U

    def test_an_adjusted_initial_overlap_is_the_state_at_rest(self):
        """With ``initialOverlap=adjust``, points that overlap the master at the first search rest there without a
        force, for good, and their gap is measured from there: pressed further in by ``d``, they carry exactly the
        force of points that just touched the master at the start, with the same tangent (the derivative of the
        residual contribution) and the same bound of the stiffness. For both kinds of contact points."""

        approach = np.array([0.0, 0.01, 0.0])
        for contactPoints in ("quadraturePoints", "nodes"):
            model, slaveSurface, masterSurface = twoBlockModel(self.journal, penetration=0.05)
            adjusted = self._constraint(
                model, slaveSurface, masterSurface, initialOverlap="adjust", contactPoints=contactPoints
            )
            self.assertTrue((adjusted.stateOfPoints == ContactState.IN_CONTACT).all())
            np.testing.assert_allclose(adjusted._initialOverlap, -0.05, rtol=1e-12)
            # at rest, also when both sides move together: touching, without a force (to round-off)
            for U in (np.zeros(adjusted.nDof), np.tile([0.3, -0.2, 0.1], adjusted.nDof // 3)):
                np.testing.assert_allclose(self._forces(adjusted, U), 0.0, atol=1e-10, err_msg=contactPoints)
                np.testing.assert_allclose(adjusted.getGaps(), 0.0, atol=1e-14)
                self.assertLess(adjusted.penaltyEnergy, 1e-20)

            model, slaveSurface, masterSurface = twoBlockModel(self.journal, penetration=0.0)
            touching = self._constraint(model, slaveSurface, masterSurface, contactPoints=contactPoints)
            self.assertEqual(touching.nDof, adjusted.nDof)

            U = self._slaveDisplacedBy(adjusted, approach)
            forces, K = residualContributionAndTangent(adjusted, U, np.zeros_like(U))
            forcesTouching, KTouching = residualContributionAndTangent(touching, U, np.zeros_like(U))
            self.assertAlmostEqual(adjusted.totalNormalForce, -1.0e4 * _SIDE**2 * 0.01, delta=1e-9)
            np.testing.assert_allclose(forces, forcesTouching, rtol=1e-10, atol=1e-10, err_msg=contactPoints)
            np.testing.assert_allclose(K, KTouching, rtol=1e-12, atol=1e-12 * np.abs(K).max())
            self.assertAlmostEqual(adjusted.deepestPenetration, 0.01, delta=1e-12)
            self.assertAlmostEqual(adjusted.penaltyEnergy, touching.penaltyEnergy, delta=1e-12)

            V = U + 1e-4 * np.random.default_rng(4).standard_normal(adjusted.nDof)
            K = residualContributionAndTangent(adjusted, V, np.zeros_like(V))[1]
            dP_dU = finiteDifferenceTangent(adjusted, V, np.zeros_like(V), h=1e-8)
            np.testing.assert_allclose(K, dP_dU, atol=1e-5 * np.abs(K).max())
            assertTheBoundIsTheOneOfTheTangent(adjusted, V)

    def test_a_contact_far_from_the_origin_follows_a_penetration_below_the_round_off_of_its_coordinates(self):
        """The gap is measured from the origins of the points, not as a difference of their coordinates (see the class
        documentation of the constraint, *Measured from the origins*). A contact a thousand lengths from the origin of
        the coordinates, whose coordinates are rounded to about 1e-13, with an initial overlap far above that
        round-off adjusted, rests exactly where it was found, and its force follows a penetration of a fraction of the
        round-off linearly: it is a smooth function of the displacements, not a staircase of the bits of the
        coordinates, on which a Newton iteration would stall."""

        origin = np.array([1e3, -1e3, 1e3])
        roundOffOfTheCoordinates = np.spacing(np.abs(origin).max())
        penalty, area = 1.0e4, _SIDE**2
        for contactPoints in ("quadraturePoints", "nodes"):
            model, slaveSurface, masterSurface = twoBlockModel(self.journal, 1e-9, origin=origin)
            contact = self._constraint(
                model, slaveSurface, masterSurface, initialOverlap="adjust", contactPoints=contactPoints
            )
            self.assertTrue((contact.stateOfPoints == ContactState.IN_CONTACT).all())
            np.testing.assert_array_equal(self._forces(contact, np.zeros(contact.nDof)), 0.0, err_msg=contactPoints)

            for fraction in (1e-3, 1e-2, 1e-1):
                penetration = fraction * roundOffOfTheCoordinates
                # the slave, below, moves up into the master by the penetration, and along the faces
                self._forces(contact, self._slaveDisplacedBy(contact, penetration * np.array([1.0, 1.0, -0.5])))
                force = penalty * area * penetration
                np.testing.assert_allclose(
                    contact.contactForceOnSlave, [0.0, -force, 0.0], rtol=1e-6, atol=1e-9 * force, err_msg=contactPoints
                )

    def test_a_point_that_separates_comes_back_to_the_adjusted_surface(self):
        """A point that leaves the master after an adjusted initial overlap and comes back meets the master where it
        touched it at the start: no force until it is back there, the force of its further approach beyond."""

        model, slaveSurface, masterSurface = twoBlockModel(self.journal, penetration=0.05)
        field = self._addDisplacementField(model)
        constraint = self._constraint(model, slaveSurface, masterSurface, initialOverlap="adjust")

        def movedTo(y: float) -> float:
            for node in slaveSurface.extractNodeSet():
                field["U"][field._indicesOfNodesInArray[node]] = [0.0, y, 0.0]
            constraint.updateConnectivity(model)
            U = np.concatenate([field["U"][field.indexOfNode(node)] for node in constraint.nodes])
            self._forces(constraint, U)
            return constraint.totalNormalForce

        self.assertEqual(movedTo(-0.08), 0.0)  # out of the master, within the search distance
        self.assertTrue((constraint.stateOfPoints == ContactState.APPROACHING).all())
        self.assertEqual(movedTo(-0.01), 0.0)  # in the master again, short of where it touched
        self.assertTrue((constraint.stateOfPoints == ContactState.IN_CONTACT).all())
        self.assertAlmostEqual(movedTo(0.02), -1.0e4 * _SIDE**2 * 0.02, delta=1e-9)

    def test_a_checkpoint_and_a_new_master_keep_the_adjusted_overlap(self):
        """The offsets of an adjusted initial overlap are written to a checkpoint and restored, and they belong to the
        contact points: a new mesh of the master keeps them, regenerated contact points lose them (reported)."""

        model, slaveSurface, masterSurface = twoBlockModel(self.journal, penetration=0.05)
        constraint = self._constraint(model, slaveSurface, masterSurface, initialOverlap="adjust")
        U = self._slaveDisplacedBy(constraint, np.array([0.0, 0.01, 0.0]))
        forces = self._forces(constraint, U)
        self.assertLess(constraint.totalNormalForce, 0.0)

        restored = self._constraint(model, slaveSurface, masterSurface, initialOverlap="adjust", searchDistance=1e-12)
        restored.setRestartData(constraint.getRestartData())
        np.testing.assert_array_equal(restored._initialOverlap, constraint._initialOverlap)
        np.testing.assert_array_equal(self._forces(restored, U), forces)
        restored.updateConnectivity(model)  # not the first search: nothing is recorded anew
        np.testing.assert_array_equal(restored._initialOverlap, constraint._initialOverlap)

        for touched in ("master", "slave"):
            model, slaveSurface, masterSurface = twoBlockModel(self.journal, penetration=0.05)
            constraint = self._constraint(model, slaveSurface, masterSurface, initialOverlap="adjust")
            facets = constraint._masterSurfaceSetName if touched == "master" else constraint.slave.setName
            surface = model.contactFacetRecipes[facets][0]
            change = ModelChange(kind=ModelChangeType.REFINEMENT, changedSurfaces={surface})
            self.assertTrue(constraint.refresh(model, change))
            constraint.updateConnectivity(model)
            self._forces(constraint, np.zeros(constraint.nDof))
            if touched == "master":
                np.testing.assert_allclose(constraint._initialOverlap, -0.05, rtol=1e-12)
                self.assertAlmostEqual(constraint.totalNormalForce, 0.0, delta=1e-10)
            else:
                np.testing.assert_array_equal(constraint._initialOverlap, 0.0)
                self.assertAlmostEqual(constraint.totalNormalForce, -1.0e4 * _SIDE**2 * 0.05, delta=1e-9)

    def test_a_new_master_keeps_the_states_of_the_points(self):
        """A rebuild of the master mid-contact (the retiling itself: the crest test) keeps what the pairing has learnt
        about the points (the class documentation, *A new mesh*): a point in contact deeper than the search distance
        is still searched at any depth and carries the same force, and a point found inside stays found inside,
        without a force."""

        def refineTheMaster(constraint, model):
            surface = model.contactFacetRecipes[constraint._masterSurfaceSetName][0]
            self.assertTrue(
                constraint.refresh(model, ModelChange(kind=ModelChangeType.REFINEMENT, changedSurfaces={surface}))
            )
            constraint.updateConnectivity(model)

        def moveTheSlaveFaceUpBy(dy: float):
            for node in slaveSurface.extractNodeSet():
                field["U"][field._indicesOfNodesInArray[node]] = [0.0, dy, 0.0]
            constraint.updateConnectivity(model)

        def forces() -> np.ndarray:
            U = np.concatenate([field["U"][field.indexOfNode(node)] for node in constraint.nodes])
            return self._forces(constraint, U)

        # in contact at the first update, pressed on to 0.05, five times the search distance
        model, slaveSurface, masterSurface = twoBlockModel(self.journal, penetration=0.005)
        field = self._addDisplacementField(model)
        constraint = self._constraint(model, slaveSurface, masterSurface, searchDistance=0.01)
        moveTheSlaveFaceUpBy(0.045)
        inContact = forces()
        self.assertAlmostEqual(constraint.totalNormalForce, -1.0e4 * _SIDE**2 * 0.05, delta=1e-9)
        refineTheMaster(constraint, model)
        self.assertTrue((constraint.stateOfPoints == ContactState.IN_CONTACT).all())
        np.testing.assert_allclose(forces(), inContact, rtol=1e-12, atol=1e-12 * np.abs(inContact).max())

        # tunnelled 0.02 into the master body
        model, slaveSurface, masterSurface = twoBlockModel(self.journal, penetration=-0.5)
        field = self._addDisplacementField(model)
        constraint = self._constraint(model, slaveSurface, masterSurface, searchDistance=0.1)
        moveTheSlaveFaceUpBy(0.52)
        refineTheMaster(constraint, model)
        self.assertEqual(constraint.numberOfPointsFoundInside, constraint.nPoints)
        self.assertFalse(constraint.withinReach)

    def test_an_initial_overlap_is_pushed_apart_or_adjusted_and_nothing_else(self):
        model, slaveSurface, masterSurface = twoBlockModel(self.journal, 0.05)
        for options in (
            {"initialOverlap": "close"},
            {"initialOverlap": "adjust", "initialOverlapRampTime": 1.0},
        ):
            with self.assertRaises(ValueError, msg=str(options)):
                self._constraint(model, slaveSurface, masterSurface, **options)
        pushedApart = self._constraint(model, slaveSurface, masterSurface, initialOverlap="PUSHAPART")
        self.assertFalse(pushedApart.adjustsTheInitialOverlap)
        np.testing.assert_array_equal(pushedApart._initialOverlap, 0.0)  # at once: nothing to record

    def test_refresh_ignores_changes_that_do_not_touch_the_surfaces(self):
        model, slaveSurface, masterSurface = twoBlockModel(self.journal, penetration=0.05)
        constraint = self._constraint(model, slaveSurface, masterSurface)
        self.assertFalse(constraint.refresh(model, ModelChange(kind=ModelChangeType.REFINEMENT)))


class TestContactAtCornersOfTheMaster(unittest.TestCase):
    """A contact point sliding over a plane strain master with corners, the slave a small square whose bottom left
    node is moved along a path with a prescribed overlap: around the re-entrant root of a tooth, and off the convex
    crest of a block. The contact points are the slave's nodes (``contactPoints=nodes``), so that the point followed
    is a node of the model."""

    #: The overlap of the moved node, a hundredth of the element size.
    overlap = 0.01

    def setUp(self):
        self.journal = Journal()

    def _model(self, masterBlocks: dict, masterFaces: dict) -> tuple:
        """Plane strain CPE4 squares (lower left and upper right corner each); the master surface is the given faces
        (face 1 bottom, 2 right, 3 top, 4 left). See :meth:`_modelOfQuadrilaterals`."""

        return self._modelOfQuadrilaterals(
            {name: ((x0, y0), (x1, y0), (x1, y1), (x0, y1)) for name, ((x0, y0), (x1, y1)) in masterBlocks.items()},
            masterFaces,
        )

    def _modelOfQuadrilaterals(self, masterQuadrilaterals: dict, masterFaces: dict) -> tuple:
        """Plane strain CPE4 elements (their four corners, counterclockwise), nodes shared where they coincide; the
        master surface is the given faces (face 1 from the first corner to the second, and so on), the slave the
        bottom of a small square far away, which a displacement field moves into place.

        Returns
        -------
        tuple
            ``(model, slaveSurface, masterSurface, field, slaveNodes)``.
        """

        model = FEModel(2)
        nodeAt = {}
        elements = {}
        slave = ((5.0, 5.0), (5.01, 5.0), (5.01, 5.01), (5.0, 5.01))
        with model.topology.changes():
            for name, corners in (masterQuadrilaterals | {"slave": slave}).items():
                nodes = []
                for x in corners:
                    if x not in nodeAt:
                        label = len(nodeAt) + 1
                        nodeAt[x] = model.nodes[label] = Node(label, np.array(x))
                    nodes.append(nodeAt[x])
                (elNumber,) = model.topology.reserveElementNumbers(1)
                element = DisplacementElement("CPE4", elNumber)
                element.setNodes(nodes)
                model.createElement(element)
                elements[name] = element

            model.surfaces["masterFaces"] = {
                face: ElementSet(f"m{face}", [elements[name] for name in names]) for face, names in masterFaces.items()
            }
            model.surfaces["slaveFace"] = {1: ElementSet("s", [elements["slave"]])}
            masterSetName, _ = buildContactFacets(
                model, "masterFaces", "mst", "corner", "facetConsistent", self.journal
            )
            slaveSetName, _ = buildContactFacets(model, "slaveFace", "slv", "corner", "facetConsistent", self.journal)

        for node in model.nodes.values():
            node.fields["displacement"] = 2
        field = NodeField("displacement", 2, NodeSet("all", list(model.nodes.values())))
        model.nodeFields["displacement"] = field
        return (
            model,
            model.elementSets[slaveSetName],
            model.elementSets[masterSetName],
            field,
            elements["slave"].nodes,
        )

    def _slide(self, model, slaveSurface, masterSurface, field, slaveNodes, path: np.ndarray) -> tuple:
        """Move the slave square so that its bottom left node runs along ``path``, searching and evaluating at every
        position.

        Returns
        -------
        tuple
            Per position the moved node's state, gap, normal force and the direction of its frozen gap; and the
            constraint and its displacements at the end of the path.
        """

        def moveTo(point):
            for node in slaveNodes:
                field["U"][field._indicesOfNodesInArray[node]] = point - np.array([5.0, 5.0])

        moveTo(path[0])
        configuration = SurfaceToDeformableSurfacePenaltySchema(penalty=1.0e4, contactPoints="nodes")
        constraint = IntegratedContact(
            "theContact",
            model,
            ContactPointsAtSlaveNodes(slaveSurface, "theContact"),
            masterSurface,
            self.journal,
            configuration=configuration,
        )
        followed = constraint.slave.surfaceNodes.index(slaveNodes[0])

        states, gaps, forces, directions = [], [], [], []
        for point in path:
            moveTo(point)
            constraint.updateConnectivity(model)
            U = np.array([field["U"][field._indicesOfNodesInArray[node]] for node in constraint.nodes]).ravel()
            constraint.computeKernels(U, np.zeros_like(U), np.zeros(constraint.nDof), None, None)
            states.append(constraint.stateOfPoints[followed])
            gaps.append(constraint.getGaps()[followed])
            forces.append(constraint._normalForcesOfPoints()[followed])
            directions.append(constraint._frozenNormals[followed].copy())
        return np.array(states), np.array(gaps), np.array(forces), np.array(directions), constraint, U

    def test_a_point_slides_around_a_reentrant_root_without_sticking_or_jumping(self):
        """The root of a tooth: the flank x = 1 (the tooth [0,1] x [0,2] on a core [0,3] x [-1,0]) meets the top of
        the core y = 0 at a re-entrant corner. A point inside the master by ``overlap`` slides down the flank and on
        along the core: it stays in contact, its gap is the signed distance to the master throughout -- the overlap
        along the faces, the distance to the corner in between -- and it is pushed out along the flank's normal, then
        diagonally, then along the core's normal."""

        d = self.overlap
        model, slaveSurface, masterSurface, field, slaveNodes = self._model(
            {
                "core1": ((0.0, -1.0), (1.0, 0.0)),
                "core2": ((1.0, -1.0), (2.0, 0.0)),
                "core3": ((2.0, -1.0), (3.0, 0.0)),
                "tooth1": ((0.0, 0.0), (1.0, 1.0)),
                "tooth2": ((0.0, 1.0), (1.0, 2.0)),
            },
            {2: ("tooth1", "tooth2"), 3: ("core2", "core3")},
        )
        down = np.column_stack((np.full(70, 1.0 - d), np.linspace(0.6, -d, 70)))
        along = np.column_stack((np.linspace(1.0 - d, 1.6, 70), np.full(70, -d)))[1:]
        path = np.vstack((down, along))

        states, gaps, forces, directions, _, _ = self._slide(
            model, slaveSurface, masterSurface, field, slaveNodes, path
        )

        self.assertTrue((states == ContactState.IN_CONTACT).all(), "the point must stay in contact")
        expected = -np.where((path[:, 0] < 1.0) & (path[:, 1] < 0.0), np.hypot(1.0 - path[:, 0], path[:, 1]), d)
        np.testing.assert_allclose(gaps, expected, rtol=1e-12)
        self.assertTrue((forces < 0.0).all(), "the point must carry a force throughout")
        np.testing.assert_allclose(directions[0], [1.0, 0.0], atol=1e-15)
        np.testing.assert_allclose(directions[69], [np.sqrt(0.5), np.sqrt(0.5)], rtol=1e-12)
        np.testing.assert_allclose(directions[-1], [0.0, 1.0], atol=1e-15)

        # At the corner the gap is frozen along the diagonal; the tangent is still the derivative of the residual, and
        # the bound of the stiffness in the critical time step takes the spring along the diagonal too.
        *_, constraint, U = self._slide(model, slaveSurface, masterSurface, field, slaveNodes, down)
        K = residualContributionAndTangent(constraint, U, np.zeros_like(U))[1]
        dP_dU = finiteDifferenceTangent(constraint, U, np.zeros_like(U), h=1e-8)
        np.testing.assert_allclose(K, dP_dU, atol=1e-5 * max(np.abs(K).max(), 1.0))
        assertTheBoundIsTheOneOfTheTangent(constraint, U)

    def test_a_point_that_slides_off_a_convex_crest_is_released(self):
        """The crest of a block [0,1] x [0,1], its top and its side the master: a point inside the block by
        ``overlap`` below the top slides in +x. Near the edge, inside, it is still pushed up -- not out of the side,
        which is closer, but across a convex crease from the top it entered through. Beyond the edge it is beside
        the block, outside of it: it is released, with the distance to the side as its gap, where measuring along
        the top's normal would have kept pushing it down."""

        d = self.overlap
        model, slaveSurface, masterSurface, field, slaveNodes = self._model(
            {"block": ((0.0, 0.0), (1.0, 1.0))}, {2: ("block",), 3: ("block",)}
        )
        path = np.column_stack((np.linspace(0.7, 1.3, 61), np.full(61, 1.0 - d)))

        states, gaps, forces, directions, _, _ = self._slide(
            model, slaveSurface, masterSurface, field, slaveNodes, path
        )

        inside = path[:, 0] < 1.0 - 1e-12
        self.assertTrue((states[inside] == ContactState.IN_CONTACT).all())
        np.testing.assert_allclose(gaps[inside], -d, rtol=1e-12)
        np.testing.assert_allclose(directions[inside], np.tile([0.0, 1.0], (inside.sum(), 1)), atol=1e-15)

        beyond = path[:, 0] > 1.0 + 1e-12
        self.assertTrue((states[beyond] == ContactState.APPROACHING).all())
        np.testing.assert_allclose(gaps[beyond], path[beyond, 0] - 1.0, rtol=1e-12)
        np.testing.assert_array_equal(forces[beyond], 0.0)

    def test_a_point_beside_a_crest_keeps_its_side_across_a_new_mesh_of_the_master(self):
        """The crest of :meth:`test_a_point_that_slides_off_a_convex_crest_is_released`: a point in contact slides to
        half its overlap from the side, which is now closer than the top it entered through. The block is refined
        into four, and the master retiled: the point's facet is carried over to the new top facet where the old one
        lay, and the point walks on from there -- it is still pushed up, not out of the side, as a search from scratch
        would push it."""

        d = self.overlap
        blocks = {"block": ((0.0, 0.0), (1.0, 1.0))} | {
            f"fine{i}{j}": ((0.5 * i, 0.5 * j), (0.5 * (i + 1), 0.5 * (j + 1))) for i in range(2) for j in range(2)
        }
        model, slaveSurface, masterSurface, field, slaveNodes = self._model(blocks, {2: ("block",), 3: ("block",)})
        path = np.array([[0.7, 1.0 - d], [1.0 - d / 2, 1.0 - d]])
        _, _, _, directions, constraint, _ = self._slide(model, slaveSurface, masterSurface, field, slaveNodes, path)
        np.testing.assert_allclose(directions[-1], [0.0, 1.0], atol=1e-15)

        def elementOf(corners: tuple):
            (lowerLeft, upperRight) = corners
            return next(
                element
                for element in model.elements.values()
                if np.allclose(element.nodes[0].coordinates, lowerLeft)
                and np.allclose(element.nodes[2].coordinates, upperRight)
            )

        with model.topology.changes():
            model.surfaces["masterFaces"] = {
                2: ElementSet("m2", [elementOf(blocks["fine10"]), elementOf(blocks["fine11"])]),
                3: ElementSet("m3", [elementOf(blocks["fine01"]), elementOf(blocks["fine11"])]),
            }
            buildContactFacets(model, "masterFaces", "mst", "corner", "facetConsistent", self.journal)
        self.assertTrue(
            constraint.refresh(model, ModelChange(kind=ModelChangeType.REFINEMENT, changedSurfaces={"masterFaces"}))
        )
        self.assertEqual(len(constraint.facetElements), 4)
        constraint.updateConnectivity(model)

        followed = constraint.slave.surfaceNodes.index(slaveNodes[0])
        self.assertEqual(constraint.stateOfPoints[followed], ContactState.IN_CONTACT)
        np.testing.assert_allclose(constraint._frozenNormals[followed], [0.0, 1.0], atol=1e-15)
        U = np.array([field["U"][field._indicesOfNodesInArray[node]] for node in constraint.nodes]).ravel()
        constraint.computeKernels(U, np.zeros_like(U), np.zeros(constraint.nDof), None, None)
        self.assertAlmostEqual(constraint.getGaps()[followed], -d, delta=1e-12)

        *_, fromScratch, _ = self._slide(model, slaveSurface, masterSurface, field, slaveNodes, path[1:])
        np.testing.assert_allclose(fromScratch._frozenNormals[followed], [1.0, 0.0], atol=1e-15)

    def _pointsAroundACurvedMaster(self, path: np.ndarray) -> tuple:
        """A half ring (radii 0.5 and 1, its outer face the master) tiled by nine elements of 20 degrees: a curved
        master whose facets turn by less than the crease angle from one to the next, but by more than it over two. The
        bottom left node of the slave square is moved along ``path``, given as angles in degrees, by ``overlap`` inside
        the facet it is in front of.

        Returns
        -------
        tuple
            As :meth:`_slide`.
        """

        def onCircle(radius: float, degrees: float) -> tuple:
            return (radius * np.cos(np.deg2rad(degrees)), radius * np.sin(np.deg2rad(degrees)))

        segments = {
            f"ring{k}": (
                onCircle(1.0, 20 * k),
                onCircle(1.0, 20 * (k + 1)),
                onCircle(0.5, 20 * (k + 1)),
                onCircle(0.5, 20 * k),
            )
            for k in range(9)
        }
        model, slaveSurface, masterSurface, field, slaveNodes = self._modelOfQuadrilaterals(
            segments, {1: tuple(segments)}
        )
        apothem = np.cos(np.deg2rad(10.0))
        points = np.array([onCircle(apothem - self.overlap, degrees) for degrees in path])
        return model, slaveSurface, masterSurface, field, slaveNodes, points

    @staticmethod
    def _normalAt(degrees: float) -> np.ndarray:
        return np.array([np.cos(np.deg2rad(degrees)), np.sin(np.deg2rad(degrees))])

    def test_a_point_that_slid_around_a_curved_master_in_one_update_pairs_as_a_search_from_scratch(self):
        """A point in contact at the top of the half ring (90 degrees) slides to 10 degrees within one update: past
        four facets, its closest facet across a convex crease from its facet. Walking over the master to a fixed point,
        it pairs with the facet a search from scratch finds, the one it is in front of, with the overlap as its gap. A
        single step would stop at 70 degrees, and move on by a facet in every repeated update."""

        model, slaveSurface, masterSurface, field, slaveNodes, path = self._pointsAroundACurvedMaster([90.0, 10.0])
        states, gaps, _, directions, constraint, _ = self._slide(
            model, slaveSurface, masterSurface, field, slaveNodes, path
        )
        followed = constraint.slave.surfaceNodes.index(slaveNodes[0])

        self.assertTrue((states == ContactState.IN_CONTACT).all())
        np.testing.assert_allclose(directions[0], self._normalAt(90.0), atol=1e-15)
        np.testing.assert_allclose(directions[1], self._normalAt(10.0), atol=1e-14)
        np.testing.assert_allclose(gaps[1], -self.overlap, rtol=1e-10)

        *_, fromScratch, _ = self._slide(model, slaveSurface, masterSurface, field, slaveNodes, path[1:])
        self.assertEqual(constraint._assignedFacetIdx[followed], fromScratch._assignedFacetIdx[followed])

    def test_an_update_repeated_after_a_checkpoint_pairs_the_same(self):
        """A point in contact on the facet at 70 degrees slides to 37 degrees within one update, onto the facet at 30
        degrees, across a convex crease from its facet; the facet at 50 degrees, between them, is not. One step goes
        to 50 degrees, and an update repeated there to 30: the walk goes on to 30 at once. An update repeated at the
        same configuration -- by the constraint itself, or by one restored from its checkpoint, as a resumed step
        repeats it -- then changes neither the pairing nor the forces."""

        model, slaveSurface, masterSurface, field, slaveNodes, path = self._pointsAroundACurvedMaster([70.0, 37.0])
        _, _, _, directions, constraint, U = self._slide(model, slaveSurface, masterSurface, field, slaveNodes, path)
        np.testing.assert_allclose(directions[1], self._normalAt(30.0), atol=1e-14)
        assignment, forces = constraint._assignedFacetIdx.copy(), constraint._normalForcesOfPoints()

        restored = IntegratedContact(
            "theContact",
            model,
            ContactPointsAtSlaveNodes(slaveSurface, "theContact"),
            masterSurface,
            self.journal,
            configuration=SurfaceToDeformableSurfacePenaltySchema(penalty=1.0e4, contactPoints="nodes"),
        )
        restored.setRestartData(constraint.getRestartData())
        for repeated in (constraint, restored):
            self.assertFalse(repeated.updateConnectivity(model), "the nodes of the constraint must not change")
            np.testing.assert_array_equal(repeated._assignedFacetIdx, assignment)
            repeated.computeKernels(U, np.zeros_like(U), np.zeros(repeated.nDof), None, None)
            np.testing.assert_array_equal(repeated._normalForcesOfPoints(), forces)

    def test_a_walk_beyond_its_bound_is_reported(self):
        """With a bound of two steps, the point stops two facets on, at 50 degrees, and the update is reported."""

        model, slaveSurface, masterSurface, field, slaveNodes, path = self._pointsAroundACurvedMaster([90.0, 10.0])
        with (
            mock.patch("edelweissfe.constraints.surfacetodeformablesurfacepenalty.maximumStepsOfThePairingWalk", 2),
            mock.patch.object(self.journal, "message") as message,
        ):
            _, _, _, directions, _, _ = self._slide(model, slaveSurface, masterSurface, field, slaveNodes, path)

        np.testing.assert_allclose(directions[1], self._normalAt(50.0), atol=1e-14)
        self.assertTrue(any("after 2 steps of the pairing" in call.args[0] for call in message.call_args_list))


class BondedNormalLaw(PenaltyLaw):
    """A toy law, defined here and nowhere in the library: every paired point is bonded to the master in the direction
    of the normal only, free to slide along the surface -- one linear spring along the frozen normal at every paired
    point, of the stretch :math:`g`, in tension and in compression."""

    name = "bonded"

    def springs(self, penaltyTimesArea, gaps, normals):
        return [
            PenaltySprings(
                np.arange(len(gaps)),
                normals,
                penaltyTimesArea * gaps,
                penaltyTimesArea,
                0.5 * penaltyTimesArea * gaps**2,
            )
        ]


class TestALawDefinedOutsideTheLibrary(unittest.TestCase):
    """A law is a class with its springs (:class:`BondedNormalLaw`): it plugs into the constraint through the registry
    of laws, and the constraint pairs, assembles, bounds and reports it without a change."""

    _constraint = TestIntegratedSurfaceContact._constraint
    _forces = TestIntegratedSurfaceContact._forces
    _slaveDisplacedBy = TestIntegratedSurfaceContact._slaveDisplacedBy

    #: The penalty, per unit area.
    PENALTY = 1.0e4

    def setUp(self):
        self.journal = Journal()
        registered = mock.patch.dict(penaltyLaws, {BondedNormalLaw.name: BondedNormalLaw})
        registered.start()
        self.addCleanup(registered.stop)

    def _bonded(self, **options):
        model, slaveSurface, masterSurface = twoBlockModel(self.journal, 0.0)
        options = {"contactType": BondedNormalLaw.name, "penalty": self.PENALTY} | options
        return model, self._constraint(model, slaveSurface, masterSurface, **options)

    def test_it_holds_the_paired_points_along_the_normal_only(self):
        """The two faces touching: every point is paired, and pulled back along the normal in tension as in
        compression, whatever it slides along the faces."""

        model, bonded = self._bonded()
        self.assertIsInstance(bonded.law, BondedNormalLaw)
        self.assertTrue((bonded.stateOfPoints == ContactState.APPROACHING).all())

        area = _SIDE**2
        normal = np.array([0.0, -1e-3, 0.0])
        self._forces(bonded, self._slaveDisplacedBy(bonded, normal))
        np.testing.assert_allclose(
            bonded.getTractions(), np.tile(-self.PENALTY * normal, (bonded.nPoints, 1)), atol=1e-9
        )
        np.testing.assert_allclose(bonded.contactForceOnSlave, -self.PENALTY * area * normal, rtol=1e-9, atol=1e-9)
        np.testing.assert_allclose(bonded.contactForceOnMaster, self.PENALTY * area * normal, rtol=1e-9, atol=1e-9)
        self.assertAlmostEqual(bonded.penaltyEnergy, 0.5 * self.PENALTY * area * 1e-6, delta=1e-12)
        self.assertEqual(bonded.numberOfPointsWithASpring, bonded.nPoints)

        sliding = np.array([1e-3, 0.0, -2e-3])
        np.testing.assert_allclose(self._forces(bonded, self._slaveDisplacedBy(bonded, sliding)), 0.0, atol=1e-9)
        np.testing.assert_allclose(bonded.getTractions(), 0.0, atol=1e-9)

    def test_its_tangent_and_bound_are_those_of_its_springs(self):
        for contactPoints in ("quadraturePoints", "nodes"):
            model, bonded = self._bonded(contactPoints=contactPoints)
            U = 1e-3 * np.random.default_rng(11).standard_normal(bonded.nDof)
            K, P = residualContributionAndTangent(bonded, U, np.zeros_like(U))[1], self._forces(bonded, U)
            dP_dU = finiteDifferenceTangent(bonded, U, np.zeros_like(U), h=1e-8)
            scale = np.abs(K).max()
            np.testing.assert_allclose(K, dP_dU, atol=1e-6 * scale, err_msg=contactPoints)
            np.testing.assert_allclose(K, K.T, atol=1e-12 * scale)
            assertTheBoundIsTheOneOfTheTangent(bonded, U)

            PExplicit = np.zeros(bonded.nDof)
            bonded.computeKernelsExplicit(U, np.zeros_like(U), PExplicit, None)
            np.testing.assert_array_equal(-PExplicit, P)


if __name__ == "__main__":
    unittest.main()
