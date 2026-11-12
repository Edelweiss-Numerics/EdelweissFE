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
"""The model classifies what it changed before an increment (``FEModel.updateConnectivity`` returns a ``Change``), and the
topology update hands every solver the same answer. A constraint reports only that its nodes changed, and sets
``withinReach``; the model compares which constraints take part and which scalar variables they bring."""

import pytest

from edelweissfe.config import registry
from edelweissfe.config.configurator import loadConfiguration
from edelweissfe.constraints.amrtransparencyprobe import (
    Constraint as AmrTransparencyProbe,
)
from edelweissfe.constraints.base.constraintbase import ConstraintBase
from edelweissfe.constraints.nodetodeformablesurfacepenalty import (
    Constraint as NodeToDeformableSurfacePenalty,
)
from edelweissfe.constraints.nodetodiscreterigidbodypenalty import (
    Constraint as NodeToDiscreteRigidBodyPenalty,
)
from edelweissfe.constraints.nodetorigidsurfacepenalty import (
    Constraint as NodeToRigidSurfacePenalty,
)
from edelweissfe.constraints.surfacetodeformablesurfacepenalty import (
    Constraint as SurfaceToDeformableSurfacePenalty,
)
from edelweissfe.constraints.surfacetodiscreterigidbodypenalty import (
    Constraint as SurfaceToDiscreteRigidBodyPenalty,
)
from edelweissfe.journal.journal import Journal
from edelweissfe.models.change import Change
from edelweissfe.models.femodel import FEModel
from edelweissfe.solvers.nonlinearimplicitstatic import NIST
from edelweissfe.variables.scalarvariable import ScalarVariable


class _Constraint(ConstraintBase):
    """A constraint that finds itself within reach or not (``inReach``), reports that its nodes changed or not
    (``nodesChange``), and brings scalar variables if it is given any."""

    def __init__(self, name, nScalarVariables=0):
        super().__init__(name, None)
        self.name = name
        self.inReach = True
        self.nodesChange = False
        self.scalarVariables = [ScalarVariable(f"{name}_{i}") for i in range(nScalarVariables)]

    @property
    def nodes(self):
        return []

    @property
    def fieldsOnNodes(self):
        return []

    @property
    def nDof(self):
        return 0

    def updateConnectivity(self, model):
        self.withinReach = self.inReach
        return self.nodesChange

    def computeKernels(self, U_np, dU, P, K, timeStep):
        pass


def _model(*constraints) -> FEModel:
    model = FEModel(2)
    for constraint in constraints:
        model.constraints[constraint.name] = constraint
    return model


def test_the_changes_are_ordered_by_how_much_of_the_discrete_system_they_invalidate():
    assert Change.NONE < Change.LOCATIONS < Change.NUMBERING < Change.MESH


def test_nothing_changed_is_none():
    model = _model(_Constraint("contact"), _Constraint("tie", nScalarVariables=1))

    assert model.updateConnectivity() == Change.NONE
    assert model.updateConnectivity() == Change.NONE


def test_a_constraint_coupling_other_nodes_is_a_change_of_the_locations():
    contact = _Constraint("contact")
    model = _model(contact, _Constraint("tie", nScalarVariables=1))

    contact.nodesChange = True

    assert model.updateConnectivity() == Change.LOCATIONS


def test_a_constraint_without_scalar_variables_leaving_and_entering_reach_is_a_change_of_the_locations():
    contact = _Constraint("contact")
    model = _model(contact)

    contact.inReach = False
    assert model.updateConnectivity() == Change.LOCATIONS
    assert model.updateConnectivity() == Change.NONE, "out of reach and staying there"

    contact.inReach = True
    assert model.updateConnectivity() == Change.LOCATIONS


def test_a_constraint_with_scalar_variables_leaving_and_entering_reach_is_a_change_of_the_numbering():
    tie = _Constraint("tie", nScalarVariables=2)
    model = _model(_Constraint("contact"), tie)

    tie.inReach = False
    assert model.updateConnectivity() == Change.NUMBERING
    assert model.activeScalarVariables() == []

    tie.inReach = True
    assert model.updateConnectivity() == Change.NUMBERING
    assert model.activeScalarVariables() == tie.scalarVariables


def test_a_change_of_the_numbering_outweighs_a_change_of_the_locations():
    contact, tie = _Constraint("contact"), _Constraint("tie", nScalarVariables=1)
    model = _model(contact, tie)

    contact.nodesChange = True
    tie.inReach = False

    assert model.updateConnectivity() == Change.NUMBERING


def test_every_constraint_that_is_switched_on_is_updated_even_after_the_first_change():
    first, second = _Constraint("first"), _Constraint("second")
    model = _model(first, second)
    first.nodesChange = True
    second.inReach = False

    model.updateConnectivity()

    assert not second.withinReach, "the second constraint found itself out of reach, though the first one changed"


def test_a_constraint_switched_off_from_outside_is_not_updated_so_its_nodes_are_not_reported():
    """The model compares which constraints take part from the start of its own call: a switch from outside is for
    the time between steps, when the discrete system is built anew."""

    contact = _Constraint("contact")
    model = _model(contact)

    contact.active = False
    contact.nodesChange = True

    assert model.updateConnectivity() == Change.NONE


def _solverAndModelWithTopologyUpdate(monkeypatch, meshChanged: bool, meshDependentsRefreshed: bool, *constraints):
    model = _model(*constraints)
    monkeypatch.setattr(model.topology, "update", lambda step=None: meshChanged)
    monkeypatch.setattr(model.topology, "refreshMeshDependents", lambda: meshDependentsRefreshed)
    return NIST(loadConfiguration({}), Journal(verbose=False)), model


@pytest.mark.parametrize("meshDependentsRefreshed", [False, True])
def test_a_change_of_the_mesh_is_the_largest_change_the_topology_update_reports(monkeypatch, meshDependentsRefreshed):
    tie = _Constraint("tie", nScalarVariables=1)
    solver, model = _solverAndModelWithTopologyUpdate(monkeypatch, True, meshDependentsRefreshed, tie)
    tie.inReach = False

    assert solver.updateTopologyAndConnectivity(model, None) == Change.MESH
    assert not tie.withinReach, "the connectivity is brought up to date on the changed mesh, too"


def test_the_topology_update_without_any_change_is_none(monkeypatch):
    solver, model = _solverAndModelWithTopologyUpdate(monkeypatch, False, False, _Constraint("contact"))

    assert solver.updateTopologyAndConnectivity(model, None) == Change.NONE


def test_a_mesh_dependent_catching_up_is_a_change_of_the_mesh(monkeypatch):
    """A mesh-dependent consumer refreshes only when the topology version advanced, i.e. when nodes or elements did
    change, so its catching up is a change of the mesh whoever made it: the same answer for every solver."""

    solver, model = _solverAndModelWithTopologyUpdate(monkeypatch, False, True, _Constraint("contact"))
    assert solver.updateTopologyAndConnectivity(model, None) == Change.MESH

    tie = _Constraint("tie", nScalarVariables=1)
    solver, model = _solverAndModelWithTopologyUpdate(monkeypatch, False, True, tie)
    tie.inReach = False
    assert solver.updateTopologyAndConnectivity(model, None) == Change.MESH


def test_the_model_modifiers_are_not_offered_a_retry(monkeypatch):
    solver, model = _solverAndModelWithTopologyUpdate(monkeypatch, True, False)

    assert solver.updateTopologyAndConnectivity(model, None, offerModelModifiers=False) == Change.NONE


def test_a_constraint_out_of_reach_whose_nodes_change_is_still_a_change_of_the_locations():
    """Every constraint that is switched on is updated, those out of reach included, and a change of its nodes is
    reported whether or not it takes part."""

    contact = _Constraint("contact")
    model = _model(contact)
    contact.inReach = False
    assert model.updateConnectivity() == Change.LOCATIONS

    contact.nodesChange = True
    assert model.updateConnectivity() == Change.LOCATIONS
    assert not contact.active


def test_only_a_constraint_that_searches_as_the_bodies_move_says_so():
    assert not ConstraintBase.connectivityChangesWithMotion
    for contact in (
        NodeToDeformableSurfacePenalty,
        SurfaceToDeformableSurfacePenalty,
        SurfaceToDiscreteRigidBodyPenalty,
    ):
        assert contact.connectivityChangesWithMotion
    # fixed nodes, a stateless gap: only refreshed after a change of the mesh
    for fixedNodes in (NodeToRigidSurfacePenalty, NodeToDiscreteRigidBodyPenalty):
        assert not fixedNodes.connectivityChangesWithMotion


def _registeredConstraintClasses() -> list[type]:
    """The constraints of EdelweissFE that derive from ``ConstraintBase``, which is where ``updateConnectivity`` is
    declared: the linear multi-point constraints do not have it. The probe of the transparency of the adaptive refinement
    is the one constraint that has work in ``updateConnectivity`` and none as the bodies move: it checks its references
    once the topology version advanced, which only a change of the mesh does."""

    classes = [registry.lookup("constraint", name)[0] for name in registry.availableNames("constraint")]
    return [
        constraintClass
        for constraintClass in classes
        if constraintClass.__module__.startswith("edelweissfe.")
        and issubclass(constraintClass, ConstraintBase)
        and constraintClass is not AmrTransparencyProbe
    ]


@pytest.mark.parametrize("constraintClass", _registeredConstraintClasses(), ids=lambda cls: cls.__module__)
def test_every_constraint_that_searches_in_update_connectivity_says_so(constraintClass):
    """NED and NEST rely on ``connectivityChangesWithMotion`` to know that ``updateConnectivity`` has work to do as the
    bodies move, whether or not the nodes of the constraint change (the rigid-body surface contact finds its closest
    triangles anew). A constraint that only follows a change of the mesh implements ``MeshDependent.refresh`` instead.
    """

    if constraintClass.updateConnectivity is not ConstraintBase.updateConnectivity:
        assert constraintClass.connectivityChangesWithMotion
