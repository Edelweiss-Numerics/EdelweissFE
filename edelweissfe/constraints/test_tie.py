#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Unit tests for the tie constraint and multi-point constraint infrastructure."""

import unittest

import numpy as np

import edelweissfe.utils.inputfileparser  # noqa: F401 bootstrap input language
from edelweissfe.constraints.base.multipointconstraintbase import (
    MultiPointConstraintBase,
)
from edelweissfe.constraints.tie import Constraint as TieConstraint
from edelweissfe.elements.contactsurfaceelement import Line2ContactFacet
from edelweissfe.models.femodel import FEModel
from edelweissfe.numerics.dofmanager import DofManager
from edelweissfe.points.node import Node
from edelweissfe.sets.elementset import ElementSet
from edelweissfe.sets.nodeset import NodeSet
from edelweissfe.solvers.nonlinearexplicitstatic import NEST
from edelweissfe.solvers.nonlinearimplicitstatic import NIST


class TestTieConstraint(unittest.TestCase):
    def setUp(self):
        self.model = FEModel(2)

    def test_empty_surface_validation(self):
        # Empty master surface
        self.model.elementSets["slave_facets"] = ElementSet("slave_facets", [])
        self.model.elementSets["master_facets"] = ElementSet("master_facets", [])

        with self.assertRaises(ValueError) as ctx:
            TieConstraint("my_tie", self.model, slaveSurface="slave_facets", masterSurface="master_facets")
        self.assertIn("contains no facet elements", str(ctx.exception))

    def test_missing_displacement_field_error(self):
        # Create 2D line2 facets
        n1 = Node(1, np.array([0.0, 0.0]))
        n2 = Node(2, np.array([1.0, 0.0]))
        n3 = Node(3, np.array([0.0, 0.0]))
        n4 = Node(4, np.array([1.0, 0.0]))

        f1 = Line2ContactFacet("Line2", 100)
        f1.setNodes([n1, n2])
        f2 = Line2ContactFacet("Line2", 101)
        f2.setNodes([n3, n4])

        self.model.elementSets["slave_facets"] = ElementSet("slave_facets", [f1])
        self.model.elementSets["master_facets"] = ElementSet("master_facets", [f2])

        tie = TieConstraint("my_tie", self.model, slaveSurface="slave_facets", masterSurface="master_facets")

        dofManager = DofManager([], [], [], [], [])
        with self.assertRaises(KeyError) as ctx:
            tie.getMultiPointConstraints(dofManager)
        self.assertIn("has no 'displacement' field defined", str(ctx.exception))

    def test_existing_nodeset_collision_raises(self):
        n1 = Node(1, np.array([0.0, 0.0]))
        n2 = Node(2, np.array([1.0, 0.0]))
        n3 = Node(3, np.array([0.0, 0.0]))
        n4 = Node(4, np.array([1.0, 0.0]))

        f1 = Line2ContactFacet("Line2", 100)
        f1.setNodes([n1, n2])
        f2 = Line2ContactFacet("Line2", 101)
        f2.setNodes([n3, n4])

        self.model.elementSets["slave_facets"] = ElementSet("slave_facets", [f1])
        self.model.elementSets["master_facets"] = ElementSet("master_facets", [f2])

        # Pre-populate nodeSet
        pre_existing_set = NodeSet("my_tie_tied", [n1])
        self.model.nodeSets["my_tie_tied"] = pre_existing_set

        with self.assertRaises(ValueError) as ctx:
            TieConstraint("my_tie", self.model, slaveSurface="slave_facets", masterSurface="master_facets")
        self.assertIn("already exists in the model", str(ctx.exception))

    def test_mpc_lifecycle_advance_to_time(self):
        class StatefulMPC(MultiPointConstraintBase):
            def __init__(self):
                self.accepted = False

            def getMultiPointConstraints(self, dofManager):
                return []

            def acceptLastState(self):
                self.accepted = True

        mpc = StatefulMPC()
        self.model.multiPointConstraints["custom_mpc"] = mpc
        self.model.advanceToTime(1.0)
        self.assertTrue(mpc.accepted)

    def test_solver_mpc_capability_validation(self):
        self.model.multiPointConstraints["dummy"] = None
        nest = NEST({}, None)
        with self.assertRaises(NotImplementedError) as ctx:
            nest.validateModelCapabilities(self.model)
        self.assertIn("not supported", str(ctx.exception))

        jobInfo = {
            "fieldCorrectionTolerance": {},
            "fluxResidualTolerance": {},
            "fluxResidualToleranceAlternative": {},
        }
        nest = NEST(jobInfo, None)
        with self.assertRaises(NotImplementedError) as ctx:
            nest.validateModelCapabilities(self.model)
        self.assertIn("not supported", str(ctx.exception))

        nist = NIST(jobInfo, None)
        # Should not raise
        nist.validateModelCapabilities(self.model)


if __name__ == "__main__":
    unittest.main()
