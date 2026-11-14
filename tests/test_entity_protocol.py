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

"""One protocol of the location for every entity of a discrete system.

Elements, constraints and, in EdelweissMeshfree, cells, cell elements and particles are all derived from
:class:`~edelweissfe.nodecouplingentity.base.nodecouplingentity.BaseNodeCouplingEntity`: the locations read the same
attributes from each of them. What an entity computes is the business of its kind; which entities a solver asks for
inertia and damping is a fact of the discrete system.
"""

import numpy as np
from test_dof_numbering_and_locations import _buildTwoElementModel, _numbering

from edelweissfe.constraints.base.constraintbase import ConstraintBase
from edelweissfe.models.discretesystem import DiscreteSystem
from edelweissfe.nodecouplingentity.base.nodecouplingentity import (
    BaseNodeCouplingEntity,
)
from edelweissfe.numerics.doflocations import DofLocations
from edelweissfe.numerics.systemmatrixcontributor import SystemMatrixContributorBase


class _Entity(BaseNodeCouplingEntity, SystemMatrixContributorBase):
    """An entity that only couples the displacement of its nodes, as a cell or a particle does: it declares nothing
    else."""

    def __init__(self, nodes):
        self._nodes = nodes

    @property
    def nNodes(self):
        return len(self._nodes)

    @property
    def nodes(self):
        return self._nodes

    @property
    def nDof(self):
        return 2 * len(self._nodes)

    @property
    def fields(self):
        return [["displacement"] for _ in self._nodes]

    @property
    def dofIndicesPermutation(self):
        return None


class _Constraint(ConstraintBase):
    """A constraint that declares only what the base requires."""

    def __init__(self, nodes):
        super().__init__("constraint", None)
        self._nodes = nodes

    @property
    def nodes(self):
        return self._nodes

    @property
    def fields(self):
        return [["displacement"] for _ in self._nodes]

    @property
    def nDof(self):
        return 2 * len(self._nodes)

    def computeKernels(self, U_np, dU, P, K, timeStep):
        pass


def test_an_entity_with_the_protocol_of_the_location_only_is_located():
    model, n = _buildTwoElementModel()
    entity = _Entity([n[0], n[1]])

    locations = DofLocations(_numbering(model), [entity])

    np.testing.assert_array_equal(locations[entity], [0, 1, 2, 3])
    assert entity.connectivityChangesWithMotion is False
    assert list(entity.scalarVariables) == []


def test_a_constraint_answers_the_protocol_of_the_location_of_every_entity():
    model, n = _buildTwoElementModel()
    constraint = _Constraint([n[0], n[1], n[2]])

    assert constraint.nNodes == 3
    assert constraint.fields == [["displacement"]] * 3
    assert constraint.dofIndicesPermutation is None
    assert list(constraint.scalarVariables) == []


def test_the_elements_whose_inertia_a_solver_assembles_at_the_nodes_are_those_of_a_finite_element_model():
    model, n = _buildTwoElementModel()
    elements = list(model.elements.values())

    system = DiscreteSystem(model, DofLocations(_numbering(model), elements))

    assert system.elementsWithNodalInertia == system.elements == elements
