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
"""What the implicit solvers ask the model (see the documentation section ``model-equation-system``): whether the
coupling changed (``FEModel.updateConnectivity``), what the equation system is made of (``FEModel.buildEquationSystem``
and its ``EquationSystemLayout``), and what to update before the elements are computed
(``EquationSystemLayout.updateMaterialPoints``).

The end-to-end test solves a strip of four elements whose model answers that only its first two elements form the
equation system, as the active domain of a meshfree model does: its node sets reach beyond the equation system, its
nodes outside of it carry no solution, and it reports a change of the coupling in every increment. It must give, bit for
bit, the solution of a model that consists of these two elements only -- with ``NIST``, and with the threaded element
loop of ``NISTParallel``.
"""

import numpy as np
import pytest

from edelweissfe.config.configurator import loadConfiguration
from edelweissfe.config.elementlibrary import getElementClass
from edelweissfe.config.materiallibrary import getMaterialClass
from edelweissfe.fields.nodefield import NodeField
from edelweissfe.journal.journal import Journal
from edelweissfe.models.equationsystemlayout import EquationSystemLayout
from edelweissfe.models.femodel import FEModel
from edelweissfe.numerics.dofmanager import DofManager
from edelweissfe.points.node import Node
from edelweissfe.sections.plane import PlaneSectionSchema, Section
from edelweissfe.sets.elementset import ElementSet
from edelweissfe.sets.nodeset import NodeSet
from edelweissfe.solvers.nonlinearimplicitstatic import NIST
from edelweissfe.solvers.nonlinearimplicitstaticparallel import NISTParallel
from edelweissfe.stepactions.base.dirichletbase import DirichletBase
from edelweissfe.stepactions.dirichlet import StepAction as DirichletStepAction
from edelweissfe.stepactions.nodeforces import StepAction as NodeForcesStepAction
from edelweissfe.steps.adaptivestep import AdaptiveStep
from edelweissfe.steps.stepmanager import StepActionCollection
from edelweissfe.timesteppers.timestep import TimeStep
from edelweissfe.utils.fieldoutput import FieldOutputController
from edelweissfe.variables.fieldvariable import FieldVariable

_FIELD = "displacement"


def _nodesWithDisplacement(labels):
    nodes = [Node(label, np.array([float(label), 0.0])) for label in labels]
    for node in nodes:
        node.fields[_FIELD] = FieldVariable(node, _FIELD)
    return nodes


def test_a_node_set_is_located_on_its_nodes_in_the_equation_system():
    """Nodes 1..5 carry the field, the equation system numbers 1, 3 and 5: a node set of all five, in a mixed order,
    is located on these three, in the order of the set, with their positions in it."""
    nodes = _nodesWithDisplacement(range(1, 6))
    byLabel = {node.label: node for node in nodes}
    numbered = NodeField(_FIELD, 2, NodeSet("numbered", [byLabel[1], byLabel[3], byLabel[5]]))
    dofManager = DofManager([numbered])

    nodeSet = NodeSet("mixed", [byLabel[label] for label in (5, 2, 1, 4, 3)])
    dofIndices, nodePositions = dofManager.locateFieldOnNodeSet(_FIELD, nodeSet)

    # node 1 is numbered first (dofs 0, 1), node 3 second (2, 3), node 5 third (4, 5)
    np.testing.assert_array_equal(dofIndices, [4, 5, 0, 1, 2, 3])
    np.testing.assert_array_equal(nodePositions, [0, 2, 4])

    # a set of nodes outside the equation system only
    dofIndices, nodePositions = dofManager.locateFieldOnNodeSet(_FIELD, NodeSet("outside", [byLabel[2]]))
    assert dofIndices.size == 0 and nodePositions.size == 0


def test_a_dirichlet_boundary_condition_prescribes_its_nodes_in_the_equation_system():
    """The prescribed increment of a boundary condition is given per node of its set; the solver takes the rows of the
    nodes in the equation system, in the order of the constrained degrees of freedom."""

    nodes = _nodesWithDisplacement(range(1, 5))
    numbered = NodeField(_FIELD, 2, NodeSet("numbered", [nodes[1], nodes[3]]))

    class _DistinctPerNode(DirichletBase):
        field = _FIELD
        fieldSize = 2
        components = [1]
        nSet = NodeSet("all", nodes)

        def __init__(self):
            pass

        def getPrescribedIncrement(self, timeStep):
            # the label of each node of the set, so that a wrong row shows
            return np.array([[float(node.label)] for node in self.nSet])

    solver = NIST(loadConfiguration({}), Journal(verbose=False))
    solver.theDofManager = DofManager([numbered])
    dirichlet = _DistinctPerNode()
    solver.locateConstrainedDofs([dirichlet])

    # the second component of nodes 2 and 4, numbered first and second
    np.testing.assert_array_equal(dirichlet.constrainedDofIndices, [1, 3])
    np.testing.assert_array_equal(dirichlet.getPrescribedIncrementOfConstrainedDofs(None), [2.0, 4.0])

    R = solver.applyDirichletToResidual(None, np.zeros(4), [dirichlet])
    np.testing.assert_array_equal(R, [0.0, 2.0, 0.0, 4.0])


def test_a_node_force_loads_its_nodes_in_the_equation_system():
    nodes = _nodesWithDisplacement(range(1, 5))
    model = FEModel(2)
    numbered = NodeField(_FIELD, 2, NodeSet("numbered", [nodes[0], nodes[2]]))
    dofManager = DofManager([numbered])

    nodeForce = NodeForcesStepAction("pull", NodeSet("all", nodes), _FIELD, np.array([0.0, 1.5]), model, None)
    P = dofManager.constructDofVector()
    nodeForce.assembleLoad(P, None, None, None, TimeStep(1, 1.0, 1.0, 1.0, 1.0, 1.0), dofManager)

    # a load enters the residual with the negative sign, at the nodes 1 and 3 only
    np.testing.assert_array_equal(P, [0.0, -1.5, 0.0, -1.5])


# --- a model whose equation system covers a part of it -------------------------------------------------------------


class _FirstElementsLayout(EquationSystemLayout):
    """The layout of the equation system of the first elements of the strip: only their nodes are numbered, carry a
    solution from increment to increment, and receive results; the other nodes of the model are left at zero."""

    def __init__(self, model, dofManager, elements, nodesInSystem, nodeFieldsInSystem):
        super().__init__(model, dofManager, elements)
        self.nodesInSystem = nodesInSystem
        self.nodeFieldsInSystem = nodeFieldsInSystem

    def writeSolutionToDofVector(self, U):
        for nodeField in self.model.nodeFields.values():
            U = self.dofManager.writeNodeFieldToDofVector(U, nodeField, "U", self.nodesInSystem)
        return U

    def writeResultsToModel(self, U, dU, P):
        for nodeField in self.nodeFieldsInSystem.values():
            for vector, entry in ((U, "U"), (dU, "dU"), (P, "P")):
                self.dofManager.writeDofVectorToNodeField(vector, nodeField, entry)
            self.model.nodeFields[nodeField.name].copyEntriesFromOther(nodeField)

    def updateMaterialPoints(self, dU, timeStep, forEachChunk):
        self.model.materialPointUpdates += 1


class _StripWithEquationSystemOnFirstElements(FEModel):
    """A model whose equation system holds its first elements only, and which reports a change of the coupling in every
    increment: the shape of the active domain of a meshfree model."""

    def __init__(self, dimension, nElementsInSystem):
        super().__init__(dimension)
        self.nElementsInSystem = nElementsInSystem
        self.connectivityUpdates = 0
        self.equationSystemBuilds = 0
        self.materialPointUpdates = 0

    def updateConnectivity(self):
        self.connectivityUpdates += 1
        super().updateConnectivity()
        return True

    def buildEquationSystem(self, withSystemMatrix=True):
        self.equationSystemBuilds += 1
        elements = list(self.elements.values())[: self.nElementsInSystem]
        nodesInSystem = NodeSet("in system", sorted({n for el in elements for n in el.nodes}, key=lambda n: n.label))
        nodeFieldsInSystem = {
            name: NodeField(name, nodeField.dimension, nodesInSystem) for name, nodeField in self.nodeFields.items()
        }
        dofManager = DofManager(
            nodeFieldsInSystem.values(),
            elements=elements,
            nodeSets=self.nodeSets.values(),
            initializeVIJPattern=withSystemMatrix,
            determiningIndexToHostObjectMapping=withSystemMatrix,
        )
        return _FirstElementsLayout(self, dofManager, elements, nodesInSystem, nodeFieldsInSystem)


def _strip(model: FEModel, nElements: int) -> FEModel:
    """A strip of plane strain quadrilaterals along x, one element high: nodes 1..nElements+1 at the bottom, the ones
    above them numbered on from there, clamped vertically at the bottom and horizontally at the left end, and pressed
    down at the top by node forces that reach beyond the strip's first elements."""

    nNodesPerRow = nElements + 1
    nodes = {}
    for row in range(2):
        for i in range(nNodesPerRow):
            label = 1 + i + row * nNodesPerRow
            nodes[label] = Node(label, np.array([float(i), float(row)]))

    ElementClass = getElementClass("CPE4", "edelweiss")
    elements = {}
    for i in range(nElements):
        element = ElementClass("CPE4", i + 1)
        bottom, top = 1 + i, 1 + i + nNodesPerRow
        element.setNodes([nodes[bottom], nodes[bottom + 1], nodes[top + 1], nodes[top]])
        elements[i + 1] = element

    model.nodes.update(nodes)
    model.elements.update(elements)
    model.elementSets["all"] = ElementSet("all", list(elements.values()))
    model.nodeSets["all"] = NodeSet("all", list(nodes.values()))
    model.nodeSets["bottom"] = NodeSet("bottom", [nodes[1 + i] for i in range(nNodesPerRow)])
    model.nodeSets["top"] = NodeSet("top", [nodes[1 + i + nNodesPerRow] for i in range(nNodesPerRow)])
    model.nodeSets["left"] = NodeSet("left", [nodes[1], nodes[1 + nNodesPerRow]])

    material = getMaterialClass("linearelastic", "edelweiss")(np.array([1000.0, 0.3]))
    model.materials["linearelastic"] = material
    section = Section(
        "section", model, material, [model.elementSets["all"]], configuration=PlaneSectionSchema(thickness=1.0)
    )
    model.sections["section"] = section
    for element in elements.values():
        section.assignSectionPropertiesToElement(element)

    return model


def _solve(model: FEModel, solverClass, journal: Journal) -> FEModel:
    model.prepareYourself(journal)
    model.advanceToTime(0.0)
    jobInfo = loadConfiguration(dict())
    model._linkFieldVariableObjects(model.nodeSets["all"])

    fieldOutputController = FieldOutputController(model, journal)
    model.fieldOutputController = fieldOutputController
    fieldOutputController.initializeJob()

    stepActions = StepActionCollection()
    stepActions["dirichlet"]["bottom"] = DirichletStepAction(
        "bottom", model.nodeSets["bottom"], _FIELD, {1: 0.0}, model, journal
    )
    stepActions["dirichlet"]["left"] = DirichletStepAction(
        "left", model.nodeSets["left"], _FIELD, {0: 0.0}, model, journal
    )
    stepActions["nodeforces"]["press"] = NodeForcesStepAction(
        "press", model.nodeSets["top"], _FIELD, np.array([0.0, -5.0]), model, journal
    )

    # the rebuilt systems number the degrees of freedom as before: the prediction survives them
    solver = solverClass(jobInfo, journal, extrapolateAcrossConnectivityChange="True")
    step = AdaptiveStep(
        0, model, fieldOutputController, journal, jobInfo, solver, [], stepActions, startInc=0.25, maxInc=0.25
    )
    step.solve()
    fieldOutputController.finalizeJob()
    return model


@pytest.mark.parametrize("solverClass", [NIST, NISTParallel])
@pytest.mark.parametrize("numThreads", [1, 2])
def test_an_equation_system_on_a_part_of_the_model(solverClass, numThreads, workerThreads):
    workerThreads(numThreads)
    journal = Journal(verbose=False)

    reference = _solve(_strip(FEModel(2), 2), solverClass, journal)
    partial = _solve(_strip(_StripWithEquationSystemOnFirstElements(2, nElementsInSystem=2), 4), solverClass, journal)

    # the nodes of the first two elements: the bottom nodes 1-3 and the top nodes 6-8 of the four-element strip
    inSystem = [1, 2, 3, 6, 7, 8]
    referenceField = reference.nodeFields[_FIELD]
    partialField = partial.nodeFields[_FIELD]
    rows = [partialField.indexOfNode(partial.nodes[label]) for label in inSystem]
    outside = [partialField.indexOfNode(partial.nodes[label]) for label in (4, 5, 9, 10)]

    for entry in ("U", "P"):
        np.testing.assert_array_equal(partialField[entry][rows], referenceField[entry])
        np.testing.assert_array_equal(partialField[entry][outside], 0.0)
    assert np.abs(referenceField["U"]).max() > 0.0

    # asked at every attempt of an increment, and rebuilt every time it reported a change
    assert partial.connectivityUpdates == 5 and partial.equationSystemBuilds == 5
    assert partial.materialPointUpdates > partial.connectivityUpdates
