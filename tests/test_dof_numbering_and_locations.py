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

"""The numbering and the locations of a small model, pinned: the global index of every degree of freedom, the location
of every element and constraint, the counts of the fluxes of the convergence tests, and the digest of the numbering.

``DofLocations.withNewLocationsOf()`` must reproduce, for a changed constraint connectivity, exactly the
locations built from scratch would hold -- and must hand out a NEW mapping of the entities, because the locations are
immutable: whoever still holds the old locations holds a mapping that stays valid.

The explicit solver asks for it (``DiscreteSystem.withNewLocations()``) instead of building the whole discrete system
when a contact search changed which nodes a constraint couples. A numbering that is built that often must not pay for
what only an implicit solver reads: the map from index to node is built on first use.
"""

import re

import numpy as np
import pytest

from edelweissfe.config.elementlibrary import getElementClass
from edelweissfe.config.materiallibrary import getMaterialClass
from edelweissfe.journal.journal import Journal
from edelweissfe.models.femodel import FEModel
from edelweissfe.numerics.doflocations import DofLocations
from edelweissfe.numerics.dofnumbering import DofNumbering
from edelweissfe.numerics.systemmatrixcontributor import denseBlockLocalEntries
from edelweissfe.points.node import Node
from edelweissfe.sections.plane import PlaneSectionSchema, Section
from edelweissfe.sets.elementset import ElementSet
from edelweissfe.sets.nodeset import NodeSet
from edelweissfe.variables.scalarvariable import ScalarVariable


class _FakeConstraint:
    """The face of a constraint the locations read: its nodes, the fields it couples per node,
    its scalar variables and its DOF count; and, for a system matrix assembly, the entries of its block."""

    #: a dense block, like an element's
    couplesAllItsDofs = True

    name = "fakeConstraint"

    def __init__(self, nodes, fieldsPerNode):
        self.nodes = nodes
        self.fieldsOnNodes = [fieldsPerNode for _ in nodes]
        self.scalarVariables = []
        # a 2D displacement field: two degrees of freedom per node and field
        self.nDof = 2 * len(nodes) * len(fieldsPerNode)

    def localEntriesOfBlock(self):
        return denseBlockLocalEntries(self.nDof)


class _FakeEntity:
    """The face of an element the locations read: its nodes, the fields it couples per node (``fieldsPerNode``, or per
    node ``fieldsOfSecondNode`` for the second), no permutation of its degrees of freedom, and its DOF count."""

    dofIndicesPermutation = None

    def __init__(self, nodes, fieldsPerNode, fieldsOfSecondNode=None):
        self.nodes = nodes
        self.fields = [fieldsPerNode for _ in nodes]
        if fieldsOfSecondNode is not None:
            self.fields[1] = fieldsOfSecondNode
        self.nDof = 2 * len(nodes) * len(fieldsPerNode)


def _buildTwoElementModel():
    n = [Node(i + 1, np.array(xy)) for i, xy in enumerate([(0, 0), (1, 0), (2, 0), (2, 1), (1, 1), (0, 1)])]
    ElementClass = getElementClass("CPE4", "edelweiss")
    e1 = ElementClass("CPE4", 1)
    e1.setNodes([n[0], n[1], n[4], n[5]])
    e2 = ElementClass("CPE4", 2)
    e2.setNodes([n[1], n[2], n[3], n[4]])
    material = getMaterialClass("linearelastic", "edelweiss")(np.array([1000.0, 0.3]))

    model = FEModel(2)
    for node in n:
        model.nodes[node.label] = node
    for element in (e1, e2):
        model.elements[element.elNumber] = element
    model.elementSets["all"] = ElementSet("all", [e1, e2])
    model.nodeSets["all"] = NodeSet("all", n)
    model.materials["linearelastic"] = material
    section = Section(
        "section1", model, material, [model.elementSets["all"]], configuration=PlaneSectionSchema(thickness=1.0)
    )
    model.sections["section1"] = section
    for element in (e1, e2):
        section.assignSectionPropertiesToElement(element)
    model.prepareYourself(Journal(verbose=False))
    model._linkFieldVariableObjects(model.nodeSets["all"])
    return model, n


def _numbering(model):
    return DofNumbering(model.nodeFields.values(), model.scalarVariables.values(), model.nodeSets.values())


def _locations(model, constraints):
    return DofLocations(_numbering(model), model.elements.values(), constraints)


def _assertSameLocations(relocated, fresh):
    assert relocated.numbering.nDof == fresh.numbering.nDof
    assert relocated.nAccumulatedNodalFluxesFieldwise == fresh.nAccumulatedNodalFluxesFieldwise

    assert relocated.elements == fresh.elements
    assert relocated.constraints == fresh.constraints
    assert relocated.ofEntity.keys() == fresh.ofEntity.keys()
    for entity in fresh.ofEntity:
        np.testing.assert_array_equal(relocated[entity], fresh[entity])


def test_new_locations_match_fresh_locations():
    model, n = _buildTwoElementModel()
    before = _FakeConstraint([n[0], n[5]], ["displacement"])
    after = _FakeConstraint([n[2], n[3], n[4]], ["displacement"])

    stale = _locations(model, [before])
    staleMapping = stale.ofEntity
    staleElementIndices = {e: stale[e].copy() for e in stale.elements}
    # asked on the old locations first: the new ones must not take over a count made for the old constraint
    staleCounts = stale.nAccumulatedNodalFluxesFieldwise

    relocated = stale.withNewLocationsOf([after])

    fresh = DofLocations(stale.numbering, model.elements.values(), [after])
    _assertSameLocations(relocated, fresh)

    # the same numbering, a NEW mapping object, the elements untouched, and the old locations unchanged
    assert relocated.numbering is stale.numbering
    assert relocated.ofEntity is not staleMapping
    assert stale.ofEntity is staleMapping and before in stale
    assert before not in relocated
    assert relocated.constraints == [after] and stale.constraints == [before]
    assert stale.nAccumulatedNodalFluxesFieldwise == staleCounts != relocated.nAccumulatedNodalFluxesFieldwise
    for element, indices in staleElementIndices.items():
        np.testing.assert_array_equal(relocated[element], indices)

    # and a DofVector constructed afterwards knows the new constraint and not the old one
    vector = relocated.constructDofVector()
    assert after in vector.locations and before not in vector.locations


def test_new_locations_with_no_constraints_at_all():
    """A search may leave a constraint with nothing to couple; the locations must simply hold none."""
    model, n = _buildTwoElementModel()
    before = _FakeConstraint([n[0]], ["displacement"])

    relocated = _locations(model, [before]).withNewLocationsOf([])
    fresh = _locations(model, [])

    _assertSameLocations(relocated, fresh)
    assert relocated.constraints == []


def test_moved_further_entities_are_located_anew_and_the_others_keep_their_locations():
    model, n = _buildTwoElementModel()
    moving = _FakeEntity([n[0], n[1]], ["displacement"])
    resting = _FakeEntity([n[4], n[5]], ["displacement"])
    locations = DofLocations(_numbering(model), [], [], [moving, resting])
    indicesOfResting = locations[resting]

    moving.nodes = [n[2], n[3]]
    relocated = locations.withNewLocationsOf([], [moving])

    np.testing.assert_array_equal(relocated[moving], [4, 5, 6, 7])
    assert relocated[resting] is indicesOfResting
    np.testing.assert_array_equal(locations[moving], [0, 1, 2, 3])
    assert relocated.elements == [moving, resting]


def test_the_discrete_system_of_the_model_leaves_out_an_element_without_kernels_and_the_nodes_only_it_touches(
    monkeypatch,
):
    """A contact facet carries geometry only: it has no rows, so it has no location, and no solver evaluates it. A node
    that only such an entity touches would be numbered, but have no row in the system matrix: the model does not build
    a discrete system with it, and names the node."""
    model, n = _buildTwoElementModel()
    e2 = model.elements[2]
    monkeypatch.setattr(type(e2), "hasKernels", property(lambda self: self is not e2))

    with pytest.raises(ValueError, match=r"field 'displacement' of node 3 is numbered.*no row in the system matrix"):
        model.buildDiscreteSystem()


def test_every_numbered_degree_of_freedom_must_be_located():
    model, n = _buildTwoElementModel()
    numbering = _numbering(model)
    e1, e2 = model.elements[1], model.elements[2]

    DofLocations(numbering, [e1, e2]).checkEveryDofIsLocated()

    with pytest.raises(ValueError, match=r"field 'displacement' of node 3 is numbered"):
        DofLocations(numbering, [e1]).checkEveryDofIsLocated()


def test_a_degree_of_freedom_that_only_a_multi_point_constraint_relates_is_located():
    """The reference point of a rigid body, say, is in no entity: its rows come through the condensation."""
    model, n = _buildTwoElementModel()
    locations = DofLocations(_numbering(model), [model.elements[1]])

    # the displacement of the nodes 3 and 4, which only the second element couples, as slave and as masters
    locations.checkEveryDofIsLocated(np.array([4, 5, 6, 7]))

    with pytest.raises(ValueError, match=r"field 'displacement' of node 4 is numbered"):
        locations.checkEveryDofIsLocated(np.array([4, 5]))


def test_the_accumulated_flux_count_is_that_of_the_elements_with_kernels(monkeypatch):
    """The relative residual test of the implicit solvers divides by the number of accumulated nodal fluxes
    (``nAccumulatedNodalFluxesFieldwise``): a facet adds none, as it has no location, so the count is that of the
    elements that carry kernels, here one CPE4 with four nodes of two displacement degrees of freedom each."""
    model, n = _buildTwoElementModel()
    e1, e2 = model.elements[1], model.elements[2]
    monkeypatch.setattr(type(e2), "hasKernels", property(lambda self: self is not e2))

    locations = DofLocations(_numbering(model), model.elementsWithKernels())

    assert locations.nAccumulatedNodalFluxesFieldwise == {"displacement": 4 * 2}
    assert (
        locations.nAccumulatedNodalFluxesFieldwise
        == DofLocations(_numbering(model), [e1]).nAccumulatedNodalFluxesFieldwise
    )


def test_the_flux_counts_of_the_elements_are_counted_once_for_all_locations_made_from_them(monkeypatch):
    model, n = _buildTwoElementModel()
    locations = _locations(model, [_FakeConstraint([n[0], n[5]], ["displacement"])])
    entitiesCounted = []
    count = DofLocations._accumulateNodalFluxesFieldwise
    monkeypatch.setattr(
        DofLocations,
        "_accumulateNodalFluxesFieldwise",
        lambda self, entities: entitiesCounted.append(len(entities)) or count(self, entities),
    )

    locations.nAccumulatedNodalFluxesFieldwise
    assert entitiesCounted == [2, 0, 1], "the elements, the further entities, the constraints"
    entitiesCounted.clear()

    relocated = locations.withNewLocationsOf([_FakeConstraint([n[2], n[3], n[4]], ["displacement"])])
    relocated.nAccumulatedNodalFluxesFieldwise
    assert entitiesCounted == [1], "only the constraints are counted again"


def test_a_moved_entity_must_be_located_already():
    model, n = _buildTwoElementModel()
    resting = _FakeEntity([n[4], n[5]], ["displacement"])
    locations = DofLocations(_numbering(model), [], [], [resting])

    with pytest.raises(ValueError, match="not located"):
        locations.withNewLocationsOf([], [_FakeEntity([n[0], n[1]], ["displacement"])])


def test_an_entity_that_couples_a_field_its_node_does_not_have_is_an_error():
    """Neither an element nor a constraint is dropped silently: the model is wrong, and the error says where."""
    model, n = _buildTwoElementModel()
    numbering = _numbering(model)

    entityClass = rf"{re.escape(_FakeEntity.__module__)}\._FakeEntity"
    with pytest.raises(ValueError, match=entityClass + r" coupling the nodes \[1, 2\].*node 2 has no field 'pressure'"):
        DofLocations(numbering, [_FakeEntity([n[0], n[1]], ["displacement"], ["displacement", "pressure"])])

    constraintClass = rf"{re.escape(_FakeConstraint.__module__)}\._FakeConstraint 'fakeConstraint'"
    with pytest.raises(
        ValueError, match=constraintClass + r" coupling the nodes \[1\].*node 1 has no field 'pressure'"
    ):
        DofLocations(numbering, [], [_FakeConstraint([n[0]], ["pressure"])])

    with pytest.raises(ValueError, match=r"coupling the nodes \[1, 1, 1, 1, 1, 1\] \.\.\. \(8 nodes\)"):
        DofLocations(numbering, [], [_FakeConstraint([n[0]] * 8, ["pressure"])])

    unnumbered = ScalarVariable("lambda_7")
    constraint = _FakeConstraint([n[0]], ["displacement"])
    constraint.scalarVariables = [unnumbered]
    with pytest.raises(ValueError, match="scalar variable 'lambda_7' is not numbered"):
        DofLocations(numbering, [], [constraint])


def test_the_node_of_an_index_is_mapped_on_first_use_only(monkeypatch):
    """Building the map from index to node costs a pass over every degree of freedom, and only the residual report of
    an implicit solver reads it: a numbering builds it when it is first asked, once."""
    builds = []
    determine = DofNumbering._determineIndexToNodeMap
    monkeypatch.setattr(DofNumbering, "_determineIndexToNodeMap", lambda self: builds.append(1) or determine(self))

    model, n = _buildTwoElementModel()
    numbering = _numbering(model)
    assert builds == []

    index = numbering.idcsOfFieldVariablesInDofVector[n[2].fields["displacement"]][1]
    assert numbering.getNodeForIndexInDofVector(index) is n[2]
    assert numbering.getNodeForIndexInDofVector(index) is n[2]
    assert builds == [1]


# The numbering and the locations of the two-element model, written down by hand: the nodes 0 .. 5 carry two
# displacement components each, in the order of the nodes.


def test_the_numbering_of_the_two_element_model():
    model, n = _buildTwoElementModel()
    scalarVariable = ScalarVariable("lambda_0")
    numbering = DofNumbering(model.nodeFields.values(), [scalarVariable], model.nodeSets.values())

    assert numbering.nDof == 13
    assert numbering.fields == ["displacement"]
    assert numbering.idcsOfFieldsInDofVector == {"displacement": slice(0, 12)}
    for i, node in enumerate(n):
        np.testing.assert_array_equal(
            numbering.idcsOfFieldVariablesInDofVector[node.fields["displacement"]], [2 * i, 2 * i + 1]
        )
    assert numbering.idcsOfScalarVariablesInDofVector == {scalarVariable: 12}

    indices, nodePositions = numbering.locateFieldOnNodeSet("displacement", NodeSet("some", [n[4], n[1]]))
    np.testing.assert_array_equal(indices, [8, 9, 2, 3])
    np.testing.assert_array_equal(nodePositions, [0, 1])
    assert numbering.getNodeForIndexInDofVector(9) is n[4]


def test_the_locations_of_the_two_element_model():
    model, n = _buildTwoElementModel()
    constraint = _FakeConstraint([n[0], n[5]], ["displacement"])
    locations = _locations(model, [constraint])
    e1, e2 = model.elements[1], model.elements[2]

    np.testing.assert_array_equal(locations[e1], [0, 1, 2, 3, 8, 9, 10, 11])
    np.testing.assert_array_equal(locations[e2], [2, 3, 4, 5, 6, 7, 8, 9])
    np.testing.assert_array_equal(locations[constraint], [0, 1, 10, 11])
    assert list(locations.ofEntity) == [e1, e2, constraint]
    assert locations.elements == [e1, e2] and locations.constraints == [constraint]

    # the counts the convergence tests use: Σ_entities Σ_nodes nDof(field)
    assert locations.nAccumulatedNodalFluxesFieldwise == {"displacement": 2 * 8 + 2 * 2}

    vector = locations.constructDofVector()
    assert vector.shape == (locations.numbering.nDof,)
    vector[e2] = 1.0
    np.testing.assert_array_equal(np.flatnonzero(vector), [2, 3, 4, 5, 6, 7, 8, 9])


def test_the_digest_of_the_numbering_is_pinned():
    """The digest is written to restart checkpoints, so its values for a model must not change."""
    model, _ = _buildTwoElementModel()

    assert _numbering(model).digest == "c825c3f1f666320e0355b0bd01fc6cdb"
    assert (
        DofNumbering(model.nodeFields.values(), [ScalarVariable("lambda_0")]).digest
        == "2d4979ecb7ee8b340ea4c100ebbb050a"
    )
