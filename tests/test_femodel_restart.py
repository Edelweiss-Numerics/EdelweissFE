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

    for nodeField in model.nodeFields.values():
        nodeField.createFieldValueEntry("U")
        nodeField.createFieldValueEntry("P")
    model._linkFieldVariableObjects(model.nodeSets["all"])

    return model, element


def test_femodel_restart_roundtrip(tmp_path):
    model, element = _buildSingleElementModel()

    model.scalarVariables["dummy"] = ScalarVariable()
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


def test_constraint_base_default_restart_data_is_none():
    """A stateless constraint (the vast majority) must not force FEModel to serialize anything for
    it -- getRestartData's default of None is the opt-out."""

    from edelweissfe.constraints.base.constraintbase import ConstraintBase

    class _StatelessConstraint(ConstraintBase):
        def __init__(self, name, model):
            super().__init__(name, model)

        @property
        def nodes(self):
            return []

        @property
        def fieldsOnNodes(self):
            return []

        @property
        def nDof(self):
            return 0

        def applyConstraint(self, U_np, dU, PExt, V, timeStep):
            pass

    constraint = _StatelessConstraint("stateless", None)
    assert constraint.getRestartData() is None
