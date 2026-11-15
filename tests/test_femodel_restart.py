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
"""A round-trip test for FEModel.writeRestart/readRestart, covering the
three kinds of state a checkpoint carries -- node fields, element quadrature-point history, and
scalar variables -- plus a stateful constraint's own internal history (see
ConstraintBase.getRestartData/setRestartData). Model topology (nodes/elements/sections/materials)
is *not* round-tripped; per the plan, restart reconstructs the model from the ``.inp`` file and
only overwrites converged state, so this test builds the model once and never rebuilds it."""

import h5py
import numpy as np
import pytest

from edelweissfe.config.elementlibrary import getElementClass
from edelweissfe.config.materiallibrary import getMaterialClass
from edelweissfe.journal.journal import Journal
from edelweissfe.models.femodel import FEModel
from edelweissfe.points.node import Node
from edelweissfe.sections.plane import PlaneSectionSchema, Section
from edelweissfe.sets.elementset import ElementSet
from edelweissfe.sets.nodeset import NodeSet
from edelweissfe.utils.exceptions import RestartError
from edelweissfe.variables.scalarvariable import ScalarVariable


class _FakeRestartableConstraint:
    """A minimal stand-in that only exercises FEModel's constraint-restart wiring, without the
    scaffolding a real ConstraintBase implementation (e.g. NodeToDeformableSurfacePenaltyConstraint)
    would need. FEModel calls getRestartData/setRestartData duck-typed, so this is sufficient."""

    def __init__(self):
        self.history = np.zeros(2)

    def getRestartData(self):
        return {"history": self.history}

    def setRestartData(self, data):
        self.history = np.asarray(data["history"])


class _FakeElement:
    """A stand-in with only what FEModel's checkpoint asks of an element: its state-variable buffer, of a given size
    (zero for an element without state)."""

    def __init__(self, elNumber: int, nStateVars: int):
        self.elNumber = elNumber
        self._stateVars = np.zeros(nStateVars)

    def getStateVars(self):
        return self._stateVars

    def setStateVars(self, values):
        self._stateVars[:] = values


def _buildSingleElementModel():
    E, nu, thickness = 1000.0, 0.3, 1.0

    n1 = Node(1, np.array([0.0, 0.0]))
    n2 = Node(2, np.array([1.0, 0.0]))
    n3 = Node(3, np.array([1.0, 1.0]))
    n4 = Node(4, np.array([0.0, 1.0]))

    ElementClass = getElementClass("CPE4", "edelweiss")
    element = ElementClass("CPE4", 1)
    element.setNodes([n1, n2, n3, n4])

    # von Mises (not linear-elastic): carries nontrivial per-quadrature-point history, so the
    # round-trip actually exercises getStateVars/setStateVars instead of comparing zeros to zeros.
    material = getMaterialClass("vonmises", "edelweiss")(np.array([E, nu, 10.0, 1.0, 5.0, 1.0]))

    model = FEModel(2)
    for n in (n1, n2, n3, n4):
        model.nodes[n.label] = n
    model.elements[element.elNumber] = element
    model.elementSets["all"] = ElementSet("all", [element])
    model.nodeSets["all"] = NodeSet("all", [n1, n2, n3, n4])
    model.materials["vonmises"] = material

    section = Section(
        "section1",
        model,
        material,
        [model.elementSets["all"]],
        configuration=PlaneSectionSchema(thickness=thickness),
    )
    model.sections["section1"] = section
    section.assignSectionPropertiesToElement(element)

    journal = Journal(verbose=False)
    model.prepareYourself(journal)

    model._linkFieldVariableObjects(model.nodeSets["all"])

    return model, element


def test_femodel_restart_roundtrip(tmp_path):
    model, element = _buildSingleElementModel()

    model.scalarVariables["dummy"] = ScalarVariable("dummy")
    model.scalarVariables["dummy"].value = 1.23

    fakeConstraint = _FakeRestartableConstraint()
    fakeConstraint.history[:] = [4.0, 5.0]
    model.constraints["fake"] = fakeConstraint

    U = model.nodeFields["displacement"]["U"]
    expectedU = np.arange(U.size, dtype=float).reshape(U.shape)
    U[:] = expectedU

    expectedStateVars = np.arange(element.getStateVars().size, dtype=float)
    element.setStateVars(expectedStateVars)

    model.time = 3.5

    checkpointPath = tmp_path / "restart.h5"
    with h5py.File(checkpointPath, "w") as f:
        model.writeRestart(f)

    # mutate everything the checkpoint should restore, so the read-back proves it actually did
    model.time = 0.0
    U[:] = 0.0
    element.setStateVars(np.zeros_like(expectedStateVars))
    model.scalarVariables["dummy"].value = 0.0
    fakeConstraint.history[:] = 0.0

    with h5py.File(checkpointPath, "r") as f:
        model.readRestart(f)

    assert model.time == pytest.approx(3.5)
    np.testing.assert_allclose(model.nodeFields["displacement"]["U"], expectedU)
    np.testing.assert_allclose(element.getStateVars(), expectedStateVars)
    assert model.scalarVariables["dummy"].value == pytest.approx(1.23)
    np.testing.assert_allclose(fakeConstraint.history, [4.0, 5.0])


def test_the_state_of_elements_of_different_sizes_round_trips(tmp_path):
    """The checkpoint holds the state of all elements in one array: each element gets back exactly its own part,
    whatever the sizes of the others, an element without state included."""
    model, _ = _buildSingleElementModel()
    for elNumber, nStateVars in ((7, 3), (8, 0), (9, 5), (10, 0)):
        model.elements[elNumber] = _FakeElement(elNumber, nStateVars)
    expected = {
        elNumber: 10.0 * elNumber + np.arange(element.getStateVars().size, dtype=float)
        for elNumber, element in model.elements.items()
    }
    for elNumber, element in model.elements.items():
        element.setStateVars(expected[elNumber])

    checkpointPath = tmp_path / "restart.h5"
    with h5py.File(checkpointPath, "w") as f:
        model.writeRestart(f)
    for element in model.elements.values():
        element.setStateVars(np.zeros(element.getStateVars().size))
    with h5py.File(checkpointPath, "r") as f:
        model.readRestart(f)

    for elNumber, element in model.elements.items():
        assert np.array_equal(element.getStateVars(), expected[elNumber]), elNumber


def test_state_for_an_element_the_model_lacks_is_refused(tmp_path):
    model, _ = _buildSingleElementModel()
    model.elements[7] = _FakeElement(7, 2)
    checkpointPath = tmp_path / "restart.h5"
    with h5py.File(checkpointPath, "w") as f:
        model.writeRestart(f)
    del model.elements[7]

    with h5py.File(checkpointPath, "r") as f, pytest.raises(RestartError, match="element 7"):
        model.readRestart(f)


def test_a_constraint_declares_the_state_it_carries():
    """A constraint carrying nothing between increments declares an empty mapping; one that has not
    declared anything cannot be checkpointed -- it is refused, not silently treated as stateless."""

    from edelweissfe.constraints.base.constraintbase import ConstraintBase
    from edelweissfe.utils.exceptions import RestartError

    class _Constraint(ConstraintBase):
        def __init__(self, name, model):
            super().__init__(name, model)

        @property
        def nodes(self):
            return []

        @property
        def fields(self):
            return []

        @property
        def nDof(self):
            return 0

        def computeKernels(self, U_np, dU, P, K, timeStep):
            pass

    class _StatelessConstraint(_Constraint):
        checkpointedState = {}

    assert _StatelessConstraint("stateless", None).getRestartData() == {}
    with pytest.raises(RestartError):
        _Constraint("undeclared", None).getRestartData()
