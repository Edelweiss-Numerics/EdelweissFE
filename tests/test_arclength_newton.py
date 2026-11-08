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
"""Newton's method under arc length control (``NISTPArcLength``).

- The controller is the step action the solver option ``arcLengthController`` names, from an input file or from
  Python alone.
- The line search and the quasi-Newton fallback act on the pair of the solution increment and the increment of the
  load parameter: the line search scales the correction the out-of-balance force causes, and the controller completes
  every trial state, so that none leaves its equation unsatisfied.
- The prediction extrapolates both with the same weights, quadratically, too, and a failed extrapolated increment is
  retried without it.

What the load parameter scales, and how it carries over to the next step, is ``tests/test_arclength_load_parameter.py``.

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
from edelweissfe.solvers.nonlinearimplicitstaticparallelarclength import (
    ArcLengthIncrement,
    NISTPArcLength,
)
from edelweissfe.stepactions.dirichlet import StepAction as DirichletStepAction
from edelweissfe.stepactions.indirectcontractioncontrol import (
    StepAction as IndirectContractionControl,
)
from edelweissfe.stepactions.indirectcontrol import StepAction as IndirectControl
from edelweissfe.stepactions.nodeforces import StepAction as NodeForcesStepAction
from edelweissfe.steps.adaptivestep import AdaptiveStep
from edelweissfe.steps.stepmanager import StepActionCollection
from edelweissfe.timesteppers.timestep import TimeStep
from edelweissfe.utils.exceptions import ReachedMaxIterations
from edelweissfe.utils.fieldoutput import FieldOutputController
from edelweissfe.utils.inputfileparser import parseInputFile
from edelweissfe.variables.fieldvariable import FieldVariable

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


#: A second arc length step: it names the controller of the first step again, without a stop condition, with the
#: absolute target of the tip, and adds the change of the tip load of the first step on top of the load reached.
_SECOND_STEP = """
*step, solver=theSolver
startInc={increment}, maxInc={increment}, minInc=1e-6, maxNumInc=100, maxIter=25, stepLength=1
>>options, name=theSolver, arcLengthController=tipDisplacement
>>indirectcontrol, name=tipDisplacement, dof1='model.nodeSets["gen_rightTop"][0].fields["displacement"]', cVector1='0, 1', dof2='model.nodeSets["gen_leftBottom"][0].fields["displacement"]', cVector2='0, 0', L=-0.25
>>nodeforces, name=tipLoad, 2=-1.0
""".format(
    increment=INCREMENT
)


def test_a_stop_condition_belongs_to_the_controller_it_is_declared_with(tmp_path, monkeypatch):
    """A stop condition ends the arc length step it is declared for, and no later step that names its controller anew:
    the first step stops before its second increment, by a condition that holds from then on; the second step names the
    controller again, without a stop condition, and runs to its end, where the tip reaches the target. (A stop condition
    carried over would end the second step at its first check, too.)"""

    tipAtStepEnds = []
    originalStepEnd = NISTPArcLength.applyStepActionsAtStepEnd

    def applyStepActionsAtStepEnd(self, model, stepActions):
        tipAtStepEnds.append(tipDisplacement(model))
        return originalStepEnd(self, model, stepActions)

    monkeypatch.setattr(NISTPArcLength, "applyStepActionsAtStepEnd", applyStepActionsAtStepEnd)
    model = run(tmp_path, deck(solverOptions=", stopCondition='True'") + _SECOND_STEP)

    stopped, completed = tipAtStepEnds
    # the first step's one increment: a tenth of the way to the target
    np.testing.assert_allclose(stopped, INCREMENT * -0.25, rtol=1e-10)
    np.testing.assert_allclose(completed, -0.25, rtol=1e-10)
    assert model.solvers["theSolver"].options["stopCondition"] is None


def test_a_stop_condition_of_the_solver_definition_holds_for_every_step_that_names_a_controller(tmp_path):
    """A stop condition in the ``*solver`` datalines belongs to the solver: a step whose ``>>options`` name the
    controller without a stop condition of their own keeps it (here, one that holds from the start ends the step after
    its first increment), and the solver returns to it after a step that declared its own."""

    definition = "*include, input={:}\n*solver, solver=NISTPArcLength, name=theSolver\nstopCondition=True\n".format(
        _MODEL
    )
    model = run(
        tmp_path,
        definition
        + _STEP.format(
            solver="theSolver", increment=INCREMENT, stepOptions="", controller="tipDisplacement", solverOptions=""
        ),
    )
    np.testing.assert_allclose(tipDisplacement(model), INCREMENT * -0.25, rtol=1e-10)

    solver = NISTPArcLength(loadConfiguration({}), Journal(verbose=False), stopCondition="True")
    solver.applyOptionsOverride({"arcLengthController": "opening", "stopCondition": "False"})
    assert solver.options["stopCondition"] == "False"
    solver.applyOptionsOverride({"arcLengthController": "reopening"})
    assert solver.options["stopCondition"] == "True"


def test_a_stop_condition_without_arc_length_control_is_refused(tmp_path):
    """A stop condition is checked under arc length control only: an ``>>options`` block that sets one for a step
    without a controller is refused, and changes nothing."""

    with pytest.raises(ValueError, match="arcLengthController=off"):
        run(tmp_path, deck(controller="off", solverOptions=", stopCondition='True'"))

    solver = NISTPArcLength(loadConfiguration({}), Journal(verbose=False))
    with pytest.raises(ValueError, match="arcLengthController=off"):
        solver.applyOptionsOverride({"stopCondition": "True"})
    assert solver.options["stopCondition"] is None
    solver.applyOptionsOverride({"arcLengthController": "opening", "stopCondition": "True"})
    with pytest.raises(ValueError, match="arcLengthController=off"):
        solver.applyOptionsOverride({"arcLengthController": "off", "stopCondition": "True"})
    assert solver.options["arcLengthController"] == "opening"


def test_the_stop_condition_is_checked_after_every_accepted_increment_of_nonzero_size(tmp_path):
    """A stop condition that holds from the start ends the step after its first increment of nonzero size: not after the
    zero increment, which leaves the step where it started, and, in a step without it, right after its first
    increment."""

    for stepOptions in ("", ", makeZeroIncrementFirst=False"):
        model = run(tmp_path, deck(solverOptions=", stopCondition='True'", stepOptions=stepOptions))
        np.testing.assert_allclose(tipDisplacement(model), INCREMENT * -0.25, rtol=1e-10)


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
    assert model.additionalParameters["arc length parameter"] == solver.Lambda


def test_a_redeclared_controller_counts_its_absolute_target_from_the_quantity_reached():
    """indirectcontrol, absolute formulation: re-declared under its name in a later step, the controller aims at the
    absolute value of the controlled quantity, so the change of the step counts from the value the steps before
    reached -- which a restart checkpoint therefore holds."""

    class FieldVariableValues:
        def __init__(self):
            self.values = np.zeros(2)

    dof1, dof2 = FieldVariableValues(), FieldVariableValues()
    opening = (np.array([0.0, -1.0]), np.array([0.0, 1.0]))
    controller = IndirectControl("opening", dof1, dof2, *opening, 0.1, None, Journal(verbose=False))
    assert controller.changeOverStep == 0.1

    # the first step ended at an opening of 0.08, e.g. by a stop condition
    dof1.values[:], dof2.values[:] = [0.0, -0.04], [0.0, 0.04]
    controller.applyAtStepEnd(None)
    controller.updateStepAction(dof1, dof2, *opening, 0.3)
    assert controller.changeOverStep == pytest.approx(0.22)

    dof1.values[:], dof2.values[:] = [0.0, -0.15], [0.0, 0.15]
    controller.applyAtStepEnd(None)
    controller.updateStepAction(dof1, dof2, *opening, 0.5)
    assert controller.changeOverStep == pytest.approx(0.2)

    assert set(controller.getRestartData()) == {"changeOverStep"}


#: A step without arc length control that bends the cantilever down by a fraction of the target, and an arc length step
#: in which the controller is declared for the first time, with the absolute target of the tip.
_DEFORM_THEN_CONTROL = """
*step, solver=theSolver
startInc=0.5, maxInc=0.5, minInc=1e-6, maxNumInc=100, maxIter=25, stepLength=1
>>options, name=theSolver, arcLengthController=off
>>dirichlet, name=clamp, nSet=gen_left, field=displacement, 1=0.0, 2=0.0
>>nodeforces, name=tipLoad, nSet=gen_rightTop, field=displacement, 2=-0.05
>>bodyforce, name=selfWeight, elSet=gen_all, forceVector='0.0, -0.005', f(t)='1'

*step, solver=theSolver
startInc={increment}, maxInc={increment}, minInc=1e-6, maxNumInc=100, maxIter=25, stepLength=1
>>options, name=theSolver, arcLengthController=tipDisplacement
>>indirectcontrol, name=tipDisplacement, dof1='model.nodeSets["gen_rightTop"][0].fields["displacement"]', cVector1='0, 1', dof2='model.nodeSets["gen_leftBottom"][0].fields["displacement"]', cVector2='0, 0', L=-0.25
>>nodeforces, name=tipLoad, 2=-1.0
""".format(
    increment=INCREMENT
)


def test_a_controller_first_declared_on_a_deformed_model_reaches_its_absolute_target(tmp_path, monkeypatch):
    """The absolute target of a controller first declared in a later step is the value of the controlled quantity at the
    end of that step, as for a controller re-declared: the change of the step counts from the quantity of the deformed
    model the step starts from, not from zero."""

    tipAtStepEnds = []
    originalStepEnd = NISTPArcLength.applyStepActionsAtStepEnd

    def applyStepActionsAtStepEnd(self, model, stepActions):
        tipAtStepEnds.append(tipDisplacement(model))
        return originalStepEnd(self, model, stepActions)

    monkeypatch.setattr(NISTPArcLength, "applyStepActionsAtStepEnd", applyStepActionsAtStepEnd)
    run(
        tmp_path,
        "*include, input={:}\n*solver, solver=NISTPArcLength, name=theSolver\n".format(_MODEL) + _DEFORM_THEN_CONTROL,
    )

    deformed, controlled = tipAtStepEnds
    assert deformed < -0.01
    np.testing.assert_allclose(controlled, -0.25, rtol=1e-10)


def test_a_contraction_control_redeclared_after_a_step_without_arc_length_counts_from_the_model():
    """indirectcontractioncontrol, absolute formulation: the change of a step counts from the contraction of the model
    when the controller is re-declared, also after a step that it did not control, in which the ring contracted."""

    ring = NodeSet(
        "ring", [Node(label, np.array(xy)) for label, xy in enumerate([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])]
    )
    for node in ring:
        node.fields["displacement"] = FieldVariable(node, "displacement")
        node.fields["displacement"].values = np.zeros(2)
    controller = IndirectContractionControl("contraction", ring, 0.1, None, Journal(verbose=False))
    assert controller.changeOverStep == 0.1

    # a step without arc length control contracts the ring by 0.04: every node moves towards the center (0, 0.5)
    for node in ring:
        towardsCenter = np.array([0.0, 0.5]) - node.coordinates
        node.fields["displacement"].values[:] = 0.04 * towardsCenter / np.linalg.norm(towardsCenter)
    controller.applyAtStepEnd(None)
    controller.updateStepAction(ring, 0.1)
    assert controller.changeOverStep == pytest.approx(0.06)

    assert set(controller.getRestartData()) == {"changeOverStep"}


# --- line search and quasi-Newton fallback -----------------------------------------------------------------------------


def test_the_line_search_scales_the_load_parameter_with_the_solution(tmp_path, monkeypatch):
    """The trial states of the line search move the load parameter with the solution, and every one of them satisfies
    the controller's equation: the tip moves down by the target of the increment. So does the iterate the line search
    chooses, which the next iteration evaluates."""

    reference = run(tmp_path, deck())

    #: per evaluation of the model: the load parameter, and the residual of the controller's equation
    loadParameters = []
    controllerResiduals = []
    lineSearches = []
    originalAssembleLoads = NISTPArcLength.assembleLoadsAtIncrementEnd
    originalLineSearch = NIST.quadraticLineSearch

    def assembleLoadsAtIncrementEnd(self, stepActions, U_np, dU, P, K, timeStep):
        controller = self.arcLengthController
        controlled = controller.controlledDofIndices(self.theDofManager)
        loadParameters.append(self.arcLengthIncrement.loadParameter)
        controllerResiduals.append(
            controller.cVector.dot(dU[controlled]) - timeStep.stepProgressIncrement * controller.changeOverStep
        )
        return originalAssembleLoads(self, stepActions, U_np, dU, P, K, timeStep)

    def quadraticLineSearch(self, largestResidualAt):
        before = len(loadParameters)
        stepLength = originalLineSearch(self, largestResidualAt)
        lineSearches.append((before, len(loadParameters), stepLength))
        return stepLength

    monkeypatch.setattr(NISTPArcLength, "assembleLoadsAtIncrementEnd", assembleLoadsAtIncrementEnd)
    monkeypatch.setattr(NIST, "quadraticLineSearch", quadraticLineSearch)
    searched = run(
        tmp_path, deck(solverOptions=", lineSearch=True, lineSearchAfterIterations=0, lineSearchEveryIterations=1")
    )

    assertSameEndState(searched, reference)
    assert lineSearches
    trialStepLengths = searched.solvers["theSolver"].lineSearchStepLengths()
    target = searched.solvers["theSolver"].arcLengthController.changeOverStep
    for first, last, stepLength in lineSearches:
        # the trials move the load parameter along the correction of the pair: an affine function of the step length
        trials = loadParameters[first:last]
        assert len(trials) == len(trialStepLengths)
        slope = (trials[-1] - trials[0]) / (trialStepLengths[-1] - trialStepLengths[0])
        np.testing.assert_allclose(
            trials, trials[0] + slope * (np.array(trialStepLengths) - trialStepLengths[0]), rtol=1e-12, atol=1e-15
        )
        # every trial state satisfies the controller's equation
        np.testing.assert_allclose(controllerResiduals[first:last], 0.0, rtol=0.0, atol=1e-12 * abs(target))
        # and the next iteration takes the load parameter at the chosen step length, which satisfies it too
        if last < len(loadParameters):
            np.testing.assert_allclose(
                loadParameters[last], trials[0] + slope * (stepLength - trialStepLengths[0]), rtol=1e-12, atol=1e-15
            )
            np.testing.assert_allclose(controllerResiduals[last], 0.0, rtol=0.0, atol=1e-12 * abs(target))


def test_every_trial_state_of_the_line_search_satisfies_the_controllers_equation():
    """The line search scales the correction the out-of-balance force causes, and the controller completes every trial
    state with the correction of the load parameter its equation asks for, as it completes the full correction. So every
    trial state satisfies the controller's equation, also from an iterate that does not satisfy it yet: here the start
    of an increment, from which the controlled opening still has to grow by the target of the increment. (Scaling the
    full correction of the pair instead would miss the target by the part of the step not taken.)"""

    class FieldVariableAtRest:
        """A stand-in for the two controlled field variables, at rest."""

        def __init__(self):
            self.values = np.zeros(2)

    dof1, dof2 = FieldVariableAtRest(), FieldVariableAtRest()
    opening = IndirectControl("opening", dof1, dof2, [0.0, -1.0], [0.0, 1.0], 0.5, None, Journal(verbose=False))

    class DofManager:
        """The degrees of freedom of the two field variables."""

        idcsOfFieldVariablesInDofVector = {dof1: np.array([0, 1]), dof2: np.array([2, 3])}

    def openingOf(dU):
        return opening.cVector.dot(dU)

    timeStep = TimeStep(3, 0.2, 0.6, 0.2, 0.6, 0.6)
    target = timeStep.stepProgressIncrement * opening.changeOverStep
    solver = NISTPArcLength(loadConfiguration({}), Journal(verbose=False), arcLengthController="opening")
    solver.arcLengthController = opening
    solver.theDofManager = DofManager()
    solver.arcLengthIncrement = ArcLengthIncrement(
        loadParameterAtStart=1.5,
        timeStep=timeStep,
        referenceTimeStep=TimeStep(3, 1.0, 1.0, 0.0, 0.0, 0.0),
        deadLoad=None,
        referenceLoad=None,
        rightHandSides=None,
        referenceCorrection=np.array([0.0, -0.1, 0.05, 0.3]),
    )
    # the corrections the out-of-balance force and the reference load cause, from the start of the increment
    ddU_0 = np.array([0.5, 0.2, -0.1, 0.05])
    dU = np.zeros(4)
    assert openingOf(dU) != target

    for stepLength in (0.3, 0.8, 1.0, 1.2):
        trial = solver.setTrialState(dU, ddU_0, stepLength)
        np.testing.assert_allclose(openingOf(trial), target, rtol=1e-14)
        # the trial state's load parameter: the one of the iterate, moved by the correction the controller asks for
        ddLambda = (target - stepLength * openingOf(ddU_0)) / openingOf(solver.arcLengthIncrement.referenceCorrection)
        np.testing.assert_allclose(solver.arcLengthIncrement.loadParameter, 1.5 + ddLambda, rtol=1e-14)
        np.testing.assert_array_equal(dU, 0.0)

    # the iterate the line search chooses satisfies it as well, with the load parameter that goes with it
    ddU = solver.applyCorrection(dU, ddU_0, 0.8)
    np.testing.assert_array_equal(dU, ddU)
    np.testing.assert_allclose(openingOf(dU), target, rtol=1e-14)
    ddLambda = (target - 0.8 * openingOf(ddU_0)) / openingOf(solver.arcLengthIncrement.referenceCorrection)
    np.testing.assert_allclose(solver.arcLengthIncrement.loadParameterIncrement, ddLambda, rtol=1e-14)
    np.testing.assert_allclose(solver.arcLengthIncrement.loadParameter, 1.5 + ddLambda, rtol=1e-14)


def test_the_fallback_continues_with_the_tangent_of_the_first_iteration(tmp_path, attempts):
    """With maxGrowIter=0, an increment counts as diverging at its first check that does not converge: once the
    cantilever touches the floor, the increments are cut back until the step fails -- unless they fall back to the
    tangent of their first iteration, the same matrix in every linear solve of an attempt, for both right-hand
    sides."""

    reference = run(tmp_path, deck())

    assert run(tmp_path, deck(stepOptions=", maxGrowIter=0")).time < 1.0

    attempts.matrices.clear()
    fallBack = run(tmp_path, deck(stepOptions=", maxGrowIter=0", solverOptions=", fallBackToQuasiNewton=True"))

    assert fallBack.time == pytest.approx(1.0)
    assert any(len(matrices) > 1 for matrices in attempts.matrices)
    for matrices in attempts.matrices:
        for matrix in matrices[1:]:
            np.testing.assert_array_equal(matrix, matrices[0])
    assertSameEndState(fallBack, reference)


# --- prediction --------------------------------------------------------------------------------------------------------


def solverWithHistory(extrapolation: str, sizes: list, dLambdas: list) -> NISTPArcLength:
    """An arc length solver whose accepted history are increments of the given sizes and load parameter increments."""
    solver = NISTPArcLength(loadConfiguration({}), Journal(verbose=False), extrapolation=extrapolation)
    timeSteps = [TimeStep(i + 1, size, 0.0, size, 0.0, 0.0) for i, size in enumerate(sizes)]
    solver.prevTimeStep, solver.dLambda = timeSteps[-1], dLambdas[-1]
    if len(sizes) > 1:
        solver.timeStepBeforePrev, solver.dLambdaBeforePrev = timeSteps[-2], dLambdas[-2]
    return solver


def test_quadratic_prediction_continues_a_parabolic_load_parameter_exactly():
    def loadParameter(s):
        return 3.0 * s**2 - s + 1.0

    # nonuniform sizes: 0.1 to 0.3, 0.3 to 0.35, then 0.35 to 0.6
    solver = solverWithHistory(
        "quadratic", [0.2, 0.05], [loadParameter(0.3) - loadParameter(0.1), loadParameter(0.35) - loadParameter(0.3)]
    )
    timeStep = TimeStep(9, 0.25, 0.0, 0.25, 0.0, 0.0)

    predicted = solver.extrapolateLoadParameterIncrement("quadratic", timeStep, solver.prevTimeStep)
    np.testing.assert_allclose(predicted, loadParameter(0.6) - loadParameter(0.35), rtol=1e-13)

    # bit for bit the original linear prediction: the last increment times the ratio of the sizes
    linear = solverWithHistory("linear", [0.3], [0.7])
    assert linear.extrapolateLoadParameterIncrement("linear", timeStep, linear.prevTimeStep) == 0.7 * (0.25 / 0.3)
    # no prediction without a last increment, or when off
    assert linear.extrapolateLoadParameterIncrement("linear", timeStep, None) == 0.0
    assert linear.extrapolateLoadParameterIncrement("off", timeStep, linear.prevTimeStep) == 0.0


def test_the_load_parameter_is_predicted_with_the_weights_of_the_solution():
    """Solution and load parameter lie on one predicted path: the same weights for both."""
    dUs = [np.array([1.0, -2.0]), np.array([1.5, -2.5])]
    dLambdas = [0.4, 0.55]
    solver = solverWithHistory("quadratic", [0.2, 0.1], dLambdas)
    solver.dUBeforePrev, solver.dU = dUs
    timeStep = TimeStep(9, 0.15, 0.0, 0.15, 0.0, 0.0)

    weightOfLast, weightOfBeforeLast = solver.extrapolationWeights("quadratic", timeStep, solver.prevTimeStep)
    dU, extrapolated = solver.extrapolateLastIncrement(
        "quadratic", timeStep, np.zeros(2), (), solver.prevTimeStep, None
    )
    dLambda = solver.extrapolateLoadParameterIncrement("quadratic", timeStep, solver.prevTimeStep)

    assert extrapolated
    np.testing.assert_allclose(dU, weightOfLast * dUs[1] + weightOfBeforeLast * dUs[0], rtol=1e-15)
    np.testing.assert_allclose(dLambda, weightOfLast * dLambdas[1] + weightOfBeforeLast * dLambdas[0], rtol=1e-15)


def test_quadratic_prediction_under_arc_length_control(tmp_path, monkeypatch):
    reference = run(tmp_path, deck())

    quadraticPredictions = []
    originalPrediction = NISTPArcLength.extrapolateLoadParameterIncrement

    def extrapolateLoadParameterIncrement(self, extrapolation, timeStep, prevTimeStep):
        if self.extrapolates(extrapolation, prevTimeStep):
            quadraticPredictions.append(self.extrapolationWeights(extrapolation, timeStep, prevTimeStep)[1] is not None)
        return originalPrediction(self, extrapolation, timeStep, prevTimeStep)

    monkeypatch.setattr(NISTPArcLength, "extrapolateLoadParameterIncrement", extrapolateLoadParameterIncrement)
    quadratic = run(tmp_path, deck(solverOptions=", extrapolation=quadratic"))

    assertSameEndState(quadratic, reference)
    # the first prediction is linear, the later ones quadratic
    assert quadraticPredictions[0] is False and quadraticPredictions.count(True) >= 3


@pytest.mark.parametrize("retry", [False, True])
def test_a_failed_extrapolated_increment_is_retried_without_extrapolation(tmp_path, attempts, retry):
    reference = run(tmp_path, deck())
    referenceAttempts = list(attempts.made)
    attempts.made.clear()

    attempts.fail = {(3, INCREMENT, True)}
    result = run(tmp_path, deck(solverOptions=", retryWithoutExtrapolation={:}".format(retry)))
    attemptsOfThird = [attempt for attempt in attempts.made if attempt[0] == 3]

    if retry:
        # the same increment, from a zero prediction of the solution and of the load parameter: no cutback
        assert attemptsOfThird == [(3, INCREMENT, True), (3, INCREMENT, False)]
        assert [attempt for attempt in attempts.made if attempt != (3, INCREMENT, False)] == referenceAttempts
    else:
        # cut back, and the prediction is dropped
        assert attemptsOfThird[:2] == [(3, INCREMENT, True), (3, 0.25 * INCREMENT, False)]
    assertSameEndState(result, reference)


# --- re-equilibration after a model change -----------------------------------------------------------------------------


def test_the_re_equilibration_after_a_model_change_holds_the_controlled_quantity(tmp_path, monkeypatch):
    """``equilibrateAfterModelChange`` under arc length control: the re-equilibration of a changed model is an arc
    length increment of size zero. Here the floor stiffens while the cantilever presses on it, a change of the model:
    the re-equilibration holds the controlled tip displacement where the last increment left it, and solves for the load
    parameter, which rises -- the stiffer floor takes more load to hold the tip down -- until the changed model is in
    equilibrium. The increments that follow start from there, and the step reaches its target."""

    changeBeforeIncrement = 8
    reEquilibrations = []
    acceptedLoadParameters = {}
    originalPrepareIncrement = NISTPArcLength.prepareIncrement
    originalSolveIncrement = NISTPArcLength.solveIncrement
    originalAcceptIncrement = NISTPArcLength.acceptIncrement

    def prepareIncrement(self, step, model, isRetry):
        originalPrepareIncrement(self, step, model, isRetry)
        if step.timeStepper.numberOfIncrementsDone() == changeBeforeIncrement and not isRetry:
            # the cantilever presses on the floor at y = -0.1
            assert tipDisplacement(model) < -0.1
            model.constraints["floor"].penalty *= 4.0
            self._modelHasChanged = True

    def solveIncrement(self, U_n, dU, P, K, stepActions, model, timeStep, *args):
        controller = self.arcLengthController
        controlled = controller.controlledDofIndices(self.theDofManager)
        controlledQuantityAtStart = controller.cVector.dot(U_n[controlled])
        result = originalSolveIncrement(self, U_n, dU, P, K, stepActions, model, timeStep, *args)
        if timeStep.number == changeBeforeIncrement and timeStep.stepProgressIncrement == 0.0:
            U_np, _, _, _, residualHistory = result
            controlledQuantityAtEnd = controller.cVector.dot(U_np[controlled])
            reEquilibrations.append((controlledQuantityAtStart, controlledQuantityAtEnd, self.Lambda, residualHistory))
        return result

    def acceptIncrement(self, step, model, timeStep):
        originalAcceptIncrement(self, step, model, timeStep)
        acceptedLoadParameters[timeStep.number] = self.Lambda

    monkeypatch.setattr(NISTPArcLength, "prepareIncrement", prepareIncrement)
    monkeypatch.setattr(NISTPArcLength, "solveIncrement", solveIncrement)
    monkeypatch.setattr(NISTPArcLength, "acceptIncrement", acceptIncrement)
    model = run(tmp_path, deck(solverOptions=", equilibrateAfterModelChange=True"))

    ((controlledQuantityAtStart, controlledQuantityAtEnd, loadParameter, residualHistory),) = reEquilibrations
    # the controlled quantity is held, the load parameter rises
    np.testing.assert_allclose(controlledQuantityAtEnd, controlledQuantityAtStart, rtol=0.0, atol=1e-12)
    assert loadParameter > 1.001 * acceptedLoadParameters[changeBeforeIncrement - 1]
    # the changed model is in equilibrium: the last iteration of the re-equilibration converged in every field
    assert all(
        history[-1].fluxResidualConverged and history[-1].correctionConverged for history in residualHistory.values()
    )
    np.testing.assert_allclose(tipDisplacement(model), -0.25, rtol=1e-10)
