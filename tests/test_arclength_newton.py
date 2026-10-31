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
"""Arc length control (``NISTPArcLength``): the controller is the step action the solver option
``arcLengthController`` names, from an input file or from Python alone.

Most decks are the cantilever of ``testfiles/edelweiss-only/ArcLengthPenaltyConstraint``, bent down onto a rigid floor
(a quadratic penalty, so the path is nonlinear) by a tip load whose load parameter follows the tip displacement. Its end
state does not depend on the path, so every option that changes the iterations must lead to the same end state.
"""

from pathlib import Path

import numpy as np
import pytest

from edelweissfe.config.configurator import loadConfiguration
from edelweissfe.config.elementlibrary import getElementClass
from edelweissfe.config.materiallibrary import getMaterialClass
from edelweissfe.drivers.inputfiledrivensimulation import finiteElementSimulation
from edelweissfe.journal.journal import Journal
from edelweissfe.models.femodel import FEModel
from edelweissfe.points.node import Node
from edelweissfe.sections.plane import PlaneSectionSchema, Section
from edelweissfe.sets.elementset import ElementSet
from edelweissfe.sets.nodeset import NodeSet
from edelweissfe.solvers.nonlinearimplicitstatic import NIST
from edelweissfe.solvers.nonlinearimplicitstaticparallelarclength import NISTPArcLength
from edelweissfe.stepactions.dirichlet import StepAction as DirichletStepAction
from edelweissfe.stepactions.indirectcontrol import StepAction as IndirectControl
from edelweissfe.stepactions.nodeforces import StepAction as NodeForcesStepAction
from edelweissfe.steps.adaptivestep import AdaptiveStep
from edelweissfe.steps.stepmanager import StepActionCollection
from edelweissfe.utils.exceptions import ReachedMaxIterations
from edelweissfe.utils.fieldoutput import FieldOutputController
from edelweissfe.utils.inputfileparser import parseInputFile

_MODEL = (
    Path(__file__).resolve().parent.parent / "testfiles" / "edelweiss-only" / "ArcLengthPenaltyConstraint" / "model.inp"
)

INCREMENT = 0.1

#: The step of ``ArcLengthPenaltyConstraint/test.inp``: the tip is controlled down by 0.25.
_STEP = """
*step, solver={solver}
startInc={increment}, maxInc={increment}, minInc=1e-6, maxNumInc=100, maxIter=25, stepLength=1{stepOptions}
>>options, name={solver}, arcLengthController={controller}{solverOptions}
>>indirectcontrol, name=tipDisplacement, dof1='model.nodeSets["gen_rightTop"][0].fields["displacement"]', cVector1='0, 1', dof2='model.nodeSets["gen_leftBottom"][0].fields["displacement"]', cVector2='0, 0', L=-0.25
>>dirichlet, name=clamp, nSet=gen_left, field=displacement, 1=0.0, 2=0.0
>>nodeforces, name=tipLoad, nSet=gen_rightTop, field=displacement, 2=-1.0
>>bodyforce, name=selfWeight, elSet=gen_all, forceVector='0.0, -0.005', f(t)='1'
"""


def deck(solverOptions: str = "", stepOptions: str = "", controller: str = "tipDisplacement") -> str:
    return "*include, input={:}\n*solver, solver=NISTPArcLength, name=theSolver\n".format(_MODEL) + _STEP.format(
        solver="theSolver",
        increment=INCREMENT,
        stepOptions=stepOptions,
        controller=controller,
        solverOptions=solverOptions,
    )


def run(tmp_path, text: str):
    path = tmp_path / "test.inp"
    path.write_text(text)
    model, _ = finiteElementSimulation(parseInputFile(str(path)), verbose=False, suppressPlots=True)
    return model


def displacements(model) -> np.ndarray:
    return model.nodeFields["displacement"]["U"].copy()


def tipDisplacement(model) -> float:
    field = model.nodeFields["displacement"]
    return field["U"][field.indexOfNode(model.nodeSets["gen_rightTop"][0]), 1]


def assertSameEndState(model, reference):
    """The same end state, within what the convergence criterion leaves open."""
    U, UReference = displacements(model), displacements(reference)
    np.testing.assert_allclose(U, UReference, rtol=0.0, atol=1e-6 * np.max(np.abs(UReference)))
    np.testing.assert_allclose(
        model.solvers["theSolver"].Lambda, reference.solvers["theSolver"].Lambda, rtol=1e-6, atol=0.0
    )


@pytest.fixture
def attempts(monkeypatch):
    """Records every attempt of an arc length increment -- (number, size, extrapolated) -- and the system matrices of
    its linear solves; ``attempts.fail`` makes chosen attempts fail as if Newton's method had not converged."""

    class Attempts:
        made = []
        matrices = []
        fail = set()

    originalSolveIncrement = NISTPArcLength.solveIncrement
    originalLinearSolve = NIST.linearSolve

    def solveIncrement(self, U_n, dU, P, K, stepActions, model, timeStep, prevTimeStep, extrapolation, *args):
        attempt = (timeStep.number, timeStep.stepProgressIncrement, self.extrapolates(extrapolation, prevTimeStep))
        Attempts.made.append(attempt)
        Attempts.matrices.append([])
        if attempt in Attempts.fail:
            Attempts.fail.discard(attempt)
            raise ReachedMaxIterations("failure forced by the test")
        return originalSolveIncrement(
            self, U_n, dU, P, K, stepActions, model, timeStep, prevTimeStep, extrapolation, *args
        )

    def linearSolve(self, A, b):
        if Attempts.matrices:
            Attempts.matrices[-1].append(A.toarray())
        return originalLinearSolve(self, A, b)

    monkeypatch.setattr(NISTPArcLength, "solveIncrement", solveIncrement)
    monkeypatch.setattr(NIST, "linearSolve", linearSolve)
    return Attempts


# --- the controller ---------------------------------------------------------------------------------------------------


def test_the_controller_is_the_step_action_the_option_names(tmp_path):
    model = run(tmp_path, deck())

    solver = model.solvers["theSolver"]
    assert solver.arcLengthController.name == "tipDisplacement"
    np.testing.assert_allclose(tipDisplacement(model), -0.25, rtol=1e-12)
    assert solver.Lambda > 0.0


def test_a_controller_the_step_does_not_have_is_refused(tmp_path):
    with pytest.raises(ValueError, match="arcLengthController=crackOpening: the step has 0 arc length controller"):
        run(tmp_path, deck(controller="crackOpening"))


def test_without_a_controller_the_loads_follow_the_time(tmp_path):
    model = run(tmp_path, deck(controller="off"))

    solver = model.solvers["theSolver"]
    assert solver.arcLengthController is None
    assert solver.Lambda == 0.0
    assert "arc length parameter" not in model.additionalParameters


def _cantilever(journal: Journal) -> FEModel:
    """A cantilever of four CPE4 elements, built from Python objects: nodes 1..5 at the bottom, 6..10 at the top."""

    nodes = {label: Node(label, np.array([float((label - 1) % 5), float((label - 1) // 5)])) for label in range(1, 11)}
    ElementClass = getElementClass("CPE4", "edelweiss")
    elements = {}
    for i in range(4):
        element = ElementClass("CPE4", i + 1)
        element.setNodes([nodes[i + 1], nodes[i + 2], nodes[i + 7], nodes[i + 6]])
        elements[i + 1] = element

    model = FEModel(2)
    model.nodes.update(nodes)
    model.elements.update(elements)
    model.elementSets["all"] = ElementSet("all", list(elements.values()))
    model.nodeSets["all"] = NodeSet("all", list(nodes.values()))
    model.nodeSets["left"] = NodeSet("left", [nodes[1], nodes[6]])
    model.nodeSets["tip"] = NodeSet("tip", [nodes[10]])

    material = getMaterialClass("linearelastic", "edelweiss")(np.array([1000.0, 0.3]))
    model.materials["linearelastic"] = material
    section = Section(
        "section", model, material, [model.elementSets["all"]], configuration=PlaneSectionSchema(thickness=1.0)
    )
    model.sections["section"] = section
    for element in elements.values():
        section.assignSectionPropertiesToElement(element)

    model.prepareYourself(journal)
    model.advanceToTime(0.0)
    for nodeField in model.nodeFields.values():
        nodeField.createFieldValueEntry("U")
        nodeField.createFieldValueEntry("P")
    model._linkFieldVariableObjects(model.nodeSets["all"])
    return model


def test_an_arc_length_step_built_from_python_objects_alone():
    """No input file: the solver gets its controller by the option, as a constructor argument, and the controller is a
    step action of the step, built through its typed constructor."""

    journal = Journal(verbose=False)
    model = _cantilever(journal)
    fieldOutputController = FieldOutputController(model, journal)
    model.fieldOutputController = fieldOutputController
    fieldOutputController.initializeJob()

    tip = model.nodes[10].fields["displacement"]
    stepActions = StepActionCollection()
    stepActions["dirichlet"]["clamp"] = DirichletStepAction(
        "clamp", model.nodeSets["left"], "displacement", {0: 0.0, 1: 0.0}, model, journal
    )
    stepActions["nodeforces"]["tipLoad"] = NodeForcesStepAction(
        "tipLoad", model.nodeSets["tip"], "displacement", np.array([0.0, -1.0]), model, journal
    )
    # the tip moves down by 0.01; the second variable does not take part
    stepActions["indirectcontrol"]["tipDisplacement"] = IndirectControl(
        "tipDisplacement", tip, model.nodes[1].fields["displacement"], [0.0, 1.0], [0.0, 0.0], -0.01, model, journal
    )

    jobInfo = loadConfiguration({})
    solver = NISTPArcLength(jobInfo, journal, arcLengthController="tipDisplacement")
    step = AdaptiveStep(
        0,
        model,
        fieldOutputController,
        journal,
        jobInfo,
        solver,
        [],
        stepActions,
        stepLength=1.0,
        startInc=0.25,
        maxInc=0.25,
    )
    step.solve()
    fieldOutputController.finalizeJob()

    field = model.nodeFields["displacement"]
    np.testing.assert_allclose(field["U"][field.indexOfNode(model.nodes[10]), 1], -0.01, rtol=1e-10)
    # the reactions at the clamp balance the tip load scaled by the load parameter, -1 * lambda: the elements'
    # internal forces sum to zero
    reaction = sum(field["P"][field.indexOfNode(node), 1] for node in model.nodeSets["left"])
    np.testing.assert_allclose(reaction, solver.Lambda * 1.0, rtol=1e-10)
    assert solver.Lambda > 0.0


def test_a_redeclared_controller_counts_its_absolute_target_from_the_quantity_reached():
    """indirectcontrol, absolute formulation: re-declared under its name in a later step, the controller aims at the
    absolute value of the controlled quantity, so the target of the step counts from the value the steps before
    reached."""

    class FieldVariableValues:
        def __init__(self):
            self.values = np.zeros(2)

    dof1, dof2 = FieldVariableValues(), FieldVariableValues()
    opening = (np.array([0.0, -1.0]), np.array([0.0, 1.0]))
    controller = IndirectControl("opening", dof1, dof2, *opening, 0.1, None, Journal(verbose=False))
    assert controller.L == 0.1

    # the first step ended at an opening of 0.08, e.g. by a stop condition
    dof1.values[:], dof2.values[:] = [0.0, -0.04], [0.0, 0.04]
    controller.applyAtStepEnd(None)
    controller.updateStepAction(dof1, dof2, *opening, 0.3)
    assert controller.L == pytest.approx(0.22)

    dof1.values[:], dof2.values[:] = [0.0, -0.15], [0.0, 0.15]
    controller.applyAtStepEnd(None)
    controller.updateStepAction(dof1, dof2, *opening, 0.5)
    assert controller.L == pytest.approx(0.2)
