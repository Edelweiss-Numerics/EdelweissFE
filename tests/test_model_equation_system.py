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


def _twoFieldModel() -> tuple[FEModel, DofManager]:
    """A model of two nodes with two fields, displacement and rotation, and the numbering of both."""
    nodes = _nodesWithDisplacement((1, 2))
    for node in nodes:
        node.fields["rotation"] = FieldVariable(node, "rotation")
    model = FEModel(2)
    model.nodeFields[_FIELD] = NodeField(_FIELD, 2, NodeSet("all", nodes))
    model.nodeFields["rotation"] = NodeField("rotation", 1, NodeSet("all", nodes))
    return model, DofManager(model.nodeFields.values())


def test_the_layout_of_a_finite_element_model_reads_the_state_of_every_node():
    model, dofManager = _twoFieldModel()
    for value, entry in enumerate(NodeField.ENTRIES, start=1):
        for nodeField in model.nodeFields.values():
            nodeField[entry][:] = value
    layout = EquationSystemLayout(model, dofManager, [])

    U, V, A = (dofManager.constructDofVector() for _ in range(3))
    layout.readStateFromModel(U, V, A, dynamicFields=[_FIELD])

    displacement, rotation = (dofManager.idcsOfFieldsInDofVector[name] for name in (_FIELD, "rotation"))
    np.testing.assert_array_equal(U, 1.0)
    # the velocity and the acceleration of the fields given only; the results dU and P are not read
    np.testing.assert_array_equal(V[displacement], 4.0)
    np.testing.assert_array_equal(A[displacement], 5.0)
    np.testing.assert_array_equal(V[rotation], 0.0)
    np.testing.assert_array_equal(A[rotation], 0.0)


def test_the_layout_of_a_finite_element_model_writes_the_results_to_every_node():
    model, dofManager = _twoFieldModel()
    for nodeField in model.nodeFields.values():
        nodeField["V"][:] = nodeField["A"][:] = -1.0
    layout = EquationSystemLayout(model, dofManager, [])

    U = dofManager.constructDofVector()
    U[:] = np.arange(dofManager.nDof)
    # every node carries its total solution: nothing is incremental
    layout.zeroIncrementalCoefficients(U)
    np.testing.assert_array_equal(U, np.arange(dofManager.nDof))

    layout.writeResultsToModel(U, 2 * U, 3 * U)
    for nodeField in model.nodeFields.values():
        rows = np.asarray(U)[dofManager.idcsOfFieldsInDofVector[nodeField.name]].reshape(-1, nodeField.dimension)
        for entry, factor in (("U", 1), ("dU", 2), ("P", 3)):
            np.testing.assert_array_equal(nodeField[entry], factor * rows)
        # passed no velocity and no acceleration, the layout leaves them as they are
        np.testing.assert_array_equal(nodeField["V"], -1.0)
        np.testing.assert_array_equal(nodeField["A"], -1.0)

    # the velocity and the acceleration, the state of a solver that integrates in time, go their own way, to every field
    layout.writeVelocityAndAccelerationToModel(4 * U, 5 * U)
    for nodeField in model.nodeFields.values():
        rows = np.asarray(U)[dofManager.idcsOfFieldsInDofVector[nodeField.name]].reshape(-1, nodeField.dimension)
        np.testing.assert_array_equal(nodeField["V"], 4 * rows)
        np.testing.assert_array_equal(nodeField["A"], 5 * rows)
    layout.writeVelocityAndAccelerationToModel(None, 6 * U)
    for nodeField in model.nodeFields.values():
        rows = np.asarray(U)[dofManager.idcsOfFieldsInDofVector[nodeField.name]].reshape(-1, nodeField.dimension)
        np.testing.assert_array_equal(nodeField["V"], 4 * rows)
        np.testing.assert_array_equal(nodeField["A"], 6 * rows)


def test_the_layout_writes_the_state_of_a_quasi_static_solver_at_rest():
    model, dofManager = _twoFieldModel()
    for nodeField in model.nodeFields.values():
        for entry in NodeField.ENTRIES:
            nodeField[entry][:] = -1.0
    layout = EquationSystemLayout(model, dofManager, [])

    layout.writeZeroVelocityAndAccelerationToModel()
    for nodeField in model.nodeFields.values():
        np.testing.assert_array_equal(nodeField["V"], 0.0)
        np.testing.assert_array_equal(nodeField["A"], 0.0)
        # the state U and the results are not touched
        for entry in ("U", "dU", "P"):
            np.testing.assert_array_equal(nodeField[entry], -1.0)


def test_the_layout_of_a_finite_element_model_has_the_inertia_at_the_nodes():
    model, dofManager = _twoFieldModel()
    layout = EquationSystemLayout(model, dofManager, [])

    # every node carries the mass the elements assemble
    displacement = dofManager.idcsOfFieldsInDofVector[_FIELD]
    np.testing.assert_array_equal(
        layout.dofsWithNodalInertia([_FIELD]), np.arange(displacement.start, displacement.stop)
    )
    np.testing.assert_array_equal(layout.dofsWithNodalInertia([_FIELD, "rotation"]), np.arange(dofManager.nDof))
    # no particles, no lumped mass of particles, and no material points
    assert layout.assembleLumpedMassOfParticles(lambda entities, work: work(entities)) is None
    assert layout.materialPoints == []


def test_the_layout_writes_the_results_in_two_parts():
    model, dofManager = _twoFieldModel()
    layout = EquationSystemLayout(model, dofManager, [])
    U = dofManager.constructDofVector()
    U[:] = np.arange(dofManager.nDof)

    # the increment on the system it was taken on, then the force on the system it is evaluated on
    layout.writeResultsToModel(U, 2 * U, None)
    layout.writeResultsToModel(U, None, 3 * U)
    for nodeField in model.nodeFields.values():
        rows = np.asarray(U)[dofManager.idcsOfFieldsInDofVector[nodeField.name]].reshape(-1, nodeField.dimension)
        np.testing.assert_array_equal(nodeField["dU"], 2 * rows)
        np.testing.assert_array_equal(nodeField["P"], 3 * rows)


# --- a model whose equation system covers a part of it -------------------------------------------------------------


class _FirstElementsLayout(EquationSystemLayout):
    """The layout of the equation system of the first elements of the strip: only their nodes are numbered, carry a
    solution from increment to increment, and receive results; the other nodes of the model are left as they are, at
    zero."""

    def __init__(self, model, dofManager, elements, nodesInSystem, nodeFieldsInSystem):
        super().__init__(model, dofManager, elements)
        self.nodesInSystem = nodesInSystem
        self.nodeFieldsInSystem = nodeFieldsInSystem

    def readEntryOfNodeField(self, dofVector, nodeField, entry):
        # straight from the model's node fields, at the nodes in the system
        self.dofManager.writeNodeFieldToDofVector(dofVector, nodeField, entry, self.nodesInSystem)

    def writeEntryOfNodeField(self, dofVector, nodeField, entry):
        # straight into the model's node fields, at the nodes in the system: the other nodes keep their values
        self.dofManager.writeDofVectorToNodeField(dofVector, nodeField, entry, self.nodesInSystem)

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
