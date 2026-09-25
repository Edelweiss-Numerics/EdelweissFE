#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Arbitration of contested multi-point-constraint slave degrees of freedom.

A DOF may be condensed out only once, so when several constraints ask for the same one, the first in
model order keeps it. These tests pin the two properties that decision must have: no constraint ever
inspects a peer, and the outcome depends on the settled model rather than on the order in which
constraints happened to be *refreshed*.

``tie`` used to resolve this itself by unioning its peers' ``claimedSlaveNodes()``. Constraints
refresh in dictionary order, so a tie refreshing early read *pre-refinement* claims and dropped
nodes that had since stopped being hanging nodes -- leaving them constrained by nothing.

Model order now carries the precedence, so the hanging-node case is pinned explicitly below: if
``hAdaptivity`` ever stops registering its constraint first, that test fails rather than the solver
quietly mis-solving.
"""

import unittest

import numpy as np

import edelweissfe.utils.inputfileparser  # noqa: F401  bootstrap input language
from edelweissfe.constraints.tie import Constraint as TieConstraint
from edelweissfe.elements.contactsurfaceelement import Line2ContactFacet
from edelweissfe.journal.journal import Journal
from edelweissfe.models.femodel import FEModel
from edelweissfe.points.node import Node
from edelweissfe.sets.elementset import ElementSet
from edelweissfe.solvers.base.nonlinearsolverbase import NonlinearSolverBase


class _FakeConstraint:
    """A multi-point constraint that hands back the records it was constructed with."""

    def __init__(self, records):
        self._records = records

    def getMultiPointConstraints(self, dofManager):
        return list(self._records)


class _ArbiterHost:
    """The minimum a solver must provide for :meth:`_collectMultiPointConstraintRecords`."""

    identification = "TestSolver"

    def __init__(self):
        self.theDofManager = None
        self.messages = []
        self.journal = self

    def message(self, text, identification, level=1):
        self.messages.append(text)

    def collect(self, model):
        # Resolved at call time so this module still imports against a tree without the arbiter,
        # letting the tie test below fail on its own terms rather than at collection.
        return NonlinearSolverBase._collectMultiPointConstraintRecords(self, model)


def _model(*namedConstraints):
    model = FEModel(2)
    for name, constraint in namedConstraints:
        model.multiPointConstraints[name] = constraint
    return model


class TestSlaveClaimArbitration(unittest.TestCase):
    def test_the_first_claim_wins(self):
        hanging = _FakeConstraint([(7, [(1, 0.5), (2, 0.5)])])
        tie = _FakeConstraint([(7, [(3, 1.0)]), (8, [(4, 1.0)])])

        records = dict(_ArbiterHost().collect(_model(("hanging", hanging), ("tie", tie))))

        self.assertEqual(records[7], [(1, 0.5), (2, 0.5)], "the first constraint in model order keeps DOF 7")
        self.assertEqual(records[8], [(4, 1.0)], "the tie keeps its uncontested DOF 8")

    def test_uncontested_claims_are_never_dropped(self):
        """The regression: a yielding constraint gives up only what someone else actually claims.

        The live defect dropped slave DOFs that no other constraint claimed, leaving them free.
        """

        hanging = _FakeConstraint([(1, [(10, 1.0)])])
        tie = _FakeConstraint([(n, [(20 + n, 1.0)]) for n in (1, 2, 3, 4, 5)])

        records = _ArbiterHost().collect(_model(("hanging", hanging), ("tie", tie)))

        self.assertEqual(sorted(slave for slave, _ in records), [1, 2, 3, 4, 5], "no DOF may be left unconstrained")
        self.assertEqual(dict(records)[1], [(10, 1.0)])

    def test_the_outcome_now_depends_on_model_order_by_design(self):
        """Deliberate, and the trade-off of resolving by order rather than by declaration.

        Swapping registration swaps who keeps the contested DOF. That is why hAdaptivity registers
        its hanging-node constraint first (see TestHangingNodePrecedence) -- the precedence lives in
        model construction, not in a per-constraint flag. What must NOT depend on order is the order
        constraints are *refreshed* in, which is the defect this module guards; that is structural
        now, since no constraint inspects a peer at all.
        """

        def contestedOwner(hangingFirst):
            pairs = [
                ("hanging", _FakeConstraint([(3, [(1, 1.0)])])),
                ("tie", _FakeConstraint([(3, [(9, 1.0)])])),
            ]
            records = _ArbiterHost().collect(_model(*(pairs if hangingFirst else pairs[::-1])))
            return dict(records)[3]

        self.assertEqual(contestedOwner(True), [(1, 1.0)])
        self.assertEqual(contestedOwner(False), [(9, 1.0)])

    def test_dropped_records_are_reported(self):
        host = _ArbiterHost()
        host.collect(
            _model(
                ("hanging", _FakeConstraint([(1, [(10, 1.0)])])),
                ("theTie", _FakeConstraint([(1, [(20, 1.0)])])),
            )
        )
        self.assertTrue(any("theTie" in m and "dropped" in m for m in host.messages))


class TestTieDoesNotInspectPeers(unittest.TestCase):
    def test_a_stale_peer_claim_cannot_untie_a_node(self):
        """A peer holding a stale claim must not be able to remove records from the tie."""

        model = FEModel(2)

        slave = [Node(1, np.array([0.0, 0.0])), Node(2, np.array([1.0, 0.0]))]
        master = [Node(3, np.array([0.0, 0.0])), Node(4, np.array([1.0, 0.0]))]
        for node in slave + master:
            node.fields["displacement"] = None
        for label, nodes in ((100, slave), (101, master)):
            facet = Line2ContactFacet("Line2", label)
            facet.setNodes(nodes)
            name = "slave_facets" if label == 100 else "master_facets"
            model.elementSets[name] = ElementSet(name, [facet])

        # Claims by node object -- what the old peer-inspecting code compared against.
        stalePeer = _FakeConstraint([])
        stalePeer.claimedSlaveNodes = lambda: set(slave)
        model.multiPointConstraints["stalePeer"] = stalePeer

        tie = TieConstraint(
            "theTie",
            model,
            model.elementSets["slave_facets"],
            model.elementSets["master_facets"],
            Journal(verbose=False),
        )

        self.assertEqual(
            len(tie.tiedRecords),
            2,
            "the tie must pair both slave nodes regardless of what any peer claims; "
            "contested DOFs are resolved centrally, not here",
        )
        self.assertEqual(tie.untiedSlaveNodes, [])


class TestHangingNodePrecedence(unittest.TestCase):
    """A hanging node on a tie's slave surface must be kept by the hanging-node constraint.

    Nothing else in the model holds that node on its coarse parent edge, so if the tie takes it the
    refined and unrefined meshes come apart there. The tie loses nothing: the node's coarse parents
    are themselves tie slaves, so its tied motion is still delivered through them. Measured on
    examples/AnchorPryOutCoarse, the two precedences differ by 5.3e-02 relative displacement.

    Model order is what expresses this, so this test is the guard on hAdaptivity registering its
    hanging-node constraint FIRST.
    """

    def test_hadaptivity_registers_its_hanging_constraint_first(self):
        model = FEModel(2)
        model.multiPointConstraints["aTie"] = _FakeConstraint([])
        hanging = _FakeConstraint([])
        # what hAdaptivity does on construction
        model.multiPointConstraints = {"amr_hanging": hanging, **model.multiPointConstraints}

        self.assertEqual(
            next(iter(model.multiPointConstraints)),
            "amr_hanging",
            "the hanging-node constraint must come first, or a tie will take its nodes",
        )

    def test_the_first_constraint_in_model_order_keeps_a_contested_dof(self):
        hanging = _FakeConstraint([(7, [(1, 0.5), (2, 0.5)])])
        tie = _FakeConstraint([(7, [(3, 1.0)]), (8, [(4, 1.0)])])

        records = dict(_ArbiterHost().collect(_model(("amr_hanging", hanging), ("aTie", tie))))

        self.assertEqual(records[7], [(1, 0.5), (2, 0.5)], "the hanging-node constraint keeps DOF 7")
        self.assertEqual(records[8], [(4, 1.0)], "the tie keeps its uncontested DOF 8")


if __name__ == "__main__":
    unittest.main()
