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

"""``DofManager.refreshConstraintIndices()`` must reproduce, for a changed constraint connectivity,
exactly the constraint-derived bookkeeping a freshly constructed DofManager would hold -- and must
hand out a NEW merged entity mapping, because consumers cache plans keyed on that mapping's identity.

The explicit solver calls it instead of rebuilding the whole DofManager when a contact search
changed which nodes a constraint couples.
"""

import numpy as np

from edelweissfe.config.elementlibrary import getElementClass
from edelweissfe.config.materiallibrary import getMaterialClass
from edelweissfe.journal.journal import Journal
from edelweissfe.models.femodel import FEModel
from edelweissfe.numerics.dofmanager import DofManager
from edelweissfe.points.node import Node
from edelweissfe.sections.plane import PlaneSectionSchema, Section
from edelweissfe.sets.elementset import ElementSet
from edelweissfe.sets.nodeset import NodeSet


class _FakeConstraint:
    """The face of a constraint the DofManager reads: its nodes, the fields it couples per node,
    its scalar variables, its DOF count and its VIJ footprint."""

    def __init__(self, nodes, fieldsPerNode):
        self.nodes = nodes
        self.fieldsOnNodes = [fieldsPerNode for _ in nodes]
        self.scalarVariables = []
        # a 2D displacement field: two degrees of freedom per node and field
        self.nDof = 2 * len(nodes) * len(fieldsPerNode)

    def getVIJContributionSize(self):
        return self.nDof**2

    def initializeVIJContribution(self, indices, I, J, idxInVIJ):  # noqa: E741
        # a dense block, like an element's
        n = len(indices)
        I[idxInVIJ : idxInVIJ + n * n] = np.repeat(indices, n)
        J[idxInVIJ : idxInVIJ + n * n] = np.tile(indices, n)


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
    for nodeField in model.nodeFields.values():
        nodeField.createFieldValueEntry("U")
        nodeField.createFieldValueEntry("P")
    model._linkFieldVariableObjects(model.nodeSets["all"])
    return model, n


def _dofManager(model, constraints, **kwargs):
    return DofManager(
        model.nodeFields.values(),
        model.scalarVariables.values(),
        model.elements.values(),
        constraints,
        model.nodeSets.values(),
        **kwargs,
    )


def _assertSameConstraintBookkeeping(refreshed, fresh, withVIJ):
    assert refreshed.nDof == fresh.nDof
    assert refreshed.accumulatedConstraintNDof == fresh.accumulatedConstraintNDof
    assert refreshed.largestNumberOfConstraintNDof == fresh.largestNumberOfConstraintNDof
    assert refreshed.nAccumulatedNodalFluxesFieldwise == fresh.nAccumulatedNodalFluxesFieldwise

    assert refreshed.idcsOfConstraintsInDofVector.keys() == fresh.idcsOfConstraintsInDofVector.keys()
    for constraint, indices in fresh.idcsOfConstraintsInDofVector.items():
        np.testing.assert_array_equal(refreshed.idcsOfConstraintsInDofVector[constraint], indices)

    assert refreshed.idcsOfHigherOrderEntitiesInDofVector.keys() == fresh.idcsOfHigherOrderEntitiesInDofVector.keys()
    for entity, indices in fresh.idcsOfHigherOrderEntitiesInDofVector.items():
        np.testing.assert_array_equal(refreshed.idcsOfHigherOrderEntitiesInDofVector[entity], indices)

    if withVIJ:
        assert refreshed._sizeVIJ == fresh._sizeVIJ
        np.testing.assert_array_equal(refreshed.I, fresh.I)
        np.testing.assert_array_equal(refreshed.J, fresh.J)
        assert refreshed.idcsOfHigherOrderEntitiesInVIJ.keys() == fresh.idcsOfHigherOrderEntitiesInVIJ.keys()
        for entity, indices in fresh.idcsOfHigherOrderEntitiesInVIJ.items():
            np.testing.assert_array_equal(refreshed.idcsOfHigherOrderEntitiesInVIJ[entity], indices)


def test_refresh_matches_a_fresh_dofmanager_without_the_vij_pattern():
    """The explicit solver's configuration: no sparse-matrix pattern, no index-to-node map."""
    model, n = _buildTwoElementModel()
    before = _FakeConstraint([n[0], n[5]], ["displacement"])
    after = _FakeConstraint([n[2], n[3], n[4]], ["displacement"])

    refreshed = _dofManager(model, [before], initializeVIJPattern=False, determiningIndexToHostObjectMapping=False)
    staleMapping = refreshed.idcsOfHigherOrderEntitiesInDofVector
    staleElementIndices = {e: i.copy() for e, i in refreshed.idcsOfElementsInDofVector.items()}

    refreshed.refreshConstraintIndices([after])

    fresh = _dofManager(model, [after], initializeVIJPattern=False, determiningIndexToHostObjectMapping=False)
    _assertSameConstraintBookkeeping(refreshed, fresh, withVIJ=False)

    # a NEW mapping object, and the elements untouched
    assert refreshed.idcsOfHigherOrderEntitiesInDofVector is not staleMapping
    assert before not in refreshed.idcsOfHigherOrderEntitiesInDofVector
    for element, indices in staleElementIndices.items():
        np.testing.assert_array_equal(refreshed.idcsOfElementsInDofVector[element], indices)

    # and a DofVector constructed afterwards knows the new constraint and not the old one
    vector = refreshed.constructDofVector()
    assert after in vector.entitiesInDofVector and before not in vector.entitiesInDofVector


def test_refresh_keeps_the_vij_pattern_consistent_when_it_was_initialized():
    """With a VIJ pattern the constraint's block of I/J moves too; it must match a fresh build."""
    model, n = _buildTwoElementModel()
    before = _FakeConstraint([n[0], n[1]], ["displacement"])
    after = _FakeConstraint([n[3]], ["displacement"])

    refreshed = _dofManager(model, [before])
    refreshed.refreshConstraintIndices([after])
    fresh = _dofManager(model, [after])

    _assertSameConstraintBookkeeping(refreshed, fresh, withVIJ=True)


def test_refresh_to_no_constraints_at_all():
    """A search may leave a constraint with nothing to couple; the bookkeeping must simply be
    empty."""
    model, n = _buildTwoElementModel()
    before = _FakeConstraint([n[0]], ["displacement"])

    refreshed = _dofManager(model, [before], initializeVIJPattern=False)
    refreshed.refreshConstraintIndices([])
    fresh = _dofManager(model, [], initializeVIJPattern=False)

    _assertSameConstraintBookkeeping(refreshed, fresh, withVIJ=False)
    assert refreshed.accumulatedConstraintNDof == 0
