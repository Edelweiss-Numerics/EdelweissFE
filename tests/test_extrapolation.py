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
"""The prediction an increment of the implicit solvers starts from (the solver option ``extrapolation``), the retry of a
failed extrapolated increment (``retryWithoutExtrapolation``), and the prediction across a rebuild of the discrete
system after a change of the connectivity (``extrapolateAcrossConnectivityChange``).

The decks are a linear elastic plate whose edge displacement follows a prescribed path in the step progress. For a
linear problem, the solution follows that path: an extrapolation that reproduces the path exactly lets the increment
converge without a single correction, which is what the tests count.
"""

from types import SimpleNamespace

import numpy as np
import pytest

from edelweissfe.config.configurator import loadConfiguration
from edelweissfe.drivers.inputfiledrivensimulation import finiteElementSimulation
from edelweissfe.fields.nodefield import NodeField
from edelweissfe.journal.journal import Journal
from edelweissfe.models.change import Change
from edelweissfe.numerics.dofnumbering import DofNumbering
from edelweissfe.solvers.nonlinearimplicitstatic import NIST
from edelweissfe.stepactions.base.dirichletbase import DirichletBase
from edelweissfe.timesteppers.timestep import TimeStep
from edelweissfe.utils.exceptions import ReachedMaxIterations
from edelweissfe.utils.inputfileparser import parseInputFile
from edelweissfe.variables.scalarvariable import ScalarVariable

INCREMENT = 0.125


def deck(path: str = "t**2", options: str = "extrapolation=quadratic") -> str:
    return f"""
*material, name=linearelastic, id=linearelastic, provider=edelweiss
210000.0, 0.3

*section, name=section1, thickness=1.0, material=linearelastic, type=plane
all

*job, name=job, domain=2d
*solver, solver=NIST, name=theSolver

*modelGenerator, generator=planeRectQuad, name=gen
x0=0, l=10
y0=0, h=10
elType=CPE4
elProvider=edelweiss
nX=2
nY=2

*step, solver=theSolver
stepLength=1, startInc={INCREMENT}, maxInc={INCREMENT}, minInc=1e-3, maxNumInc=100, maxIter=10
>>options, name=theSolver, {options}
>>dirichlet, name=left, nSet=gen_left, field=displacement, 1=0, 2=0
>>dirichlet, name=right, nSet=gen_right, field=displacement, 1=0.1, f(t)='{path}'
"""


def run(tmp_path, text: str):
    path = tmp_path / "test.inp"
    path.write_text(text)
    model, _ = finiteElementSimulation(parseInputFile(str(path)), verbose=False, suppressPlots=True)
    return model


@pytest.fixture
def increments(monkeypatch):
    """Records every attempt of an increment -- (number, size, extrapolated) -- and every accepted increment -- (size,
    iterations); ``increments.failAttempts`` makes chosen attempts fail as if Newton's method had not converged."""

    class Increments:
        attempts = []
        accepted = []
        failAttempts = set()

    originalSolveIncrement = NIST.solveIncrement
    originalAcceptIncrement = NIST.acceptIncrement

    def solveIncrement(self, U_n, dU, P, K, stepActions, model, timeStep, prevTimeStep, extrapolation, *args):
        attempt = (timeStep.number, timeStep.stepProgressIncrement, self.extrapolates(extrapolation, prevTimeStep))
        Increments.attempts.append(attempt)
        if attempt in Increments.failAttempts:
            Increments.failAttempts.discard(attempt)
            raise ReachedMaxIterations("failure forced by the test")
        return originalSolveIncrement(
            self, U_n, dU, P, K, stepActions, model, timeStep, prevTimeStep, extrapolation, *args
        )

    def acceptIncrement(self, step, model, timeStep):
        Increments.accepted.append((timeStep.stepProgressIncrement, self._iterationCounter))
        return originalAcceptIncrement(self, step, model, timeStep)

    monkeypatch.setattr(NIST, "solveIncrement", solveIncrement)
    monkeypatch.setattr(NIST, "acceptIncrement", acceptIncrement)
    return Increments


def iterationsAfterTwoNonzeroIncrements(accepted) -> list:
    """The iterations of every accepted increment preceded by two accepted increments of nonzero size."""
    return [
        iterations
        for i, (size, iterations) in enumerate(accepted)
        if i >= 2 and accepted[i - 1][0] > 0.0 and accepted[i - 2][0] > 0.0
    ]


def displacements(model) -> np.ndarray:
    return model.nodeFields["displacement"]["U"].copy()


# --- the extrapolation formulas ---------------------------------------------------------------------------------------


def solverWithHistory(extrapolation, sizes, dUs, dofNumbering="the numbering", **options) -> NIST:
    """A NIST solver whose accepted history is the given increments (oldest first), sizes in step progress, accepted in
    a discrete system with the given digest of its DOF numbering."""
    solver = NIST(loadConfiguration({}), Journal(verbose=False), extrapolation=extrapolation, **options)
    timeSteps = [TimeStep(i + 1, size, 0.0, size, 0.0, 0.0) for i, size in enumerate(sizes)]
    solver.prevTimeStep, solver.dUOfLastIncrement = timeSteps[-1], np.array(dUs[-1], dtype=float)
    if len(sizes) > 1:
        solver.timeStepOfIncrementBeforeLast, solver.dUOfIncrementBeforeLast = timeSteps[-2], np.array(
            dUs[-2], dtype=float
        )
    solver.dofNumberingOfLastIncrement = dofNumbering
    return solver


def predict(solver, size, dirichlets=()) -> tuple[np.ndarray, bool]:
    dU = np.full(len(solver.dUOfLastIncrement), np.nan)
    timeStep = TimeStep(9, size, 0.0, size, 0.0, 0.0)
    return solver.extrapolateLastIncrement(
        solver.options["extrapolation"], timeStep, dU, list(dirichlets), solver.prevTimeStep, None
    )


def test_quadratic_extrapolation_continues_a_quadratic_path_exactly():
    def u(s):
        return np.array([3.0 * s**2 - s + 1.0, -2.0 * s**2])

    # nonuniform sizes: 0.1 to 0.3, 0.3 to 0.35, then 0.35 to 0.6
    solver = solverWithHistory("quadratic", [0.2, 0.05], [u(0.3) - u(0.1), u(0.35) - u(0.3)])
    dU, extrapolated = predict(solver, 0.25)

    assert extrapolated
    np.testing.assert_allclose(dU, u(0.6) - u(0.35), rtol=1e-13)


def test_quadratic_extrapolation_is_linear_while_one_increment_is_known():
    solver = solverWithHistory("quadratic", [0.2], [[1.0, -2.0]])
    np.testing.assert_array_equal(predict(solver, 0.1)[0], [0.5, -1.0])

    # a zero increment before the last one is no history either
    solver = solverWithHistory("quadratic", [0.0, 0.2], [[0.0, 0.0], [1.0, -2.0]])
    np.testing.assert_array_equal(predict(solver, 0.1)[0], [0.5, -1.0])


def test_linear_extrapolation_scales_the_last_increment():
    lastIncrement = np.array([0.3, -1.7, 2.9])
    solver = solverWithHistory("linear", [0.3], [lastIncrement])
    dU, extrapolated = predict(solver, 0.1)

    assert extrapolated
    # bit for bit the original formula: the last increment times the ratio of the sizes
    np.testing.assert_array_equal(dU, lastIncrement * (0.1 / 0.3))


def test_no_extrapolation_without_a_nonzero_last_increment_or_when_off():
    for solver in (
        solverWithHistory("off", [0.3], [[1.0, 2.0]]),
        solverWithHistory("quadratic", [0.0], [[1.0, 2.0]]),
    ):
        dU, extrapolated = predict(solver, 0.1)
        assert not extrapolated
        np.testing.assert_array_equal(dU, 0.0)


def test_the_prescribed_degrees_of_freedom_take_their_prescribed_increment():
    class Dirichlet(DirichletBase):
        components = [0]
        constrainedDofIndices = np.array([1])
        constrainedNodePositions = np.array([0])

        def __init__(self):
            pass

        def getPrescribedIncrement(self, timeStep):
            return np.array([[0.75]])

    solver = solverWithHistory("quadratic", [0.2, 0.2], [[1.0, 1.0], [2.0, 2.0]])
    dU, _ = predict(solver, 0.2, [Dirichlet()])
    assert dU[1] == 0.75
    assert dU[0] == pytest.approx(3.0)


# --- the decks --------------------------------------------------------------------------------------------------------


def test_a_quadratic_path_converges_without_corrections(tmp_path, increments):
    quadratic = displacements(run(tmp_path, deck(options="extrapolation=quadratic")))
    iterationsQuadratic = iterationsAfterTwoNonzeroIncrements(increments.accepted)
    increments.accepted.clear()

    linear = displacements(run(tmp_path, deck(options="extrapolation=linear")))
    iterationsLinear = iterationsAfterTwoNonzeroIncrements(increments.accepted)

    assert len(iterationsQuadratic) >= 4
    assert all(iterations == 0 for iterations in iterationsQuadratic)
    assert all(iterations >= 1 for iterations in iterationsLinear)
    np.testing.assert_allclose(quadratic, linear, rtol=0.0, atol=1e-12)


def test_a_linear_path_converges_without_corrections_with_linear_extrapolation(tmp_path, increments):
    run(tmp_path, deck(path="t", options="extrapolation=linear"))
    assert all(iterations == 0 for iterations in iterationsAfterTwoNonzeroIncrements(increments.accepted))


def test_an_unknown_extrapolation_is_refused(tmp_path):
    with pytest.raises(ValueError, match="Unknown extrapolation"):
        run(tmp_path, deck(options="extrapolation=quadratik"))


@pytest.mark.parametrize("option", ["retryWithoutExtrapolation", "extrapolateAcrossConnectivityChange"])
def test_an_option_of_the_prediction_is_refused_without_one(tmp_path, option):
    """Both options act on a prediction: a declaration that switches one on with ``extrapolation=off`` is refused, the
    definition of the solver as a step's ``>>options`` block, which then changes nothing. Switched on for earlier
    steps, the option rests in a step that switches the prediction off."""

    with pytest.raises(ValueError, match=option):
        NIST(loadConfiguration({}), Journal(verbose=False), extrapolation="off", **{option: "True"})
    with pytest.raises(ValueError, match=option):
        run(tmp_path, deck(options=f"extrapolation=off, {option}=True"))

    solver = NIST(loadConfiguration({}), Journal(verbose=False), extrapolation="off")
    with pytest.raises(ValueError, match=option):
        solver.applyOptionsOverride({option: True})
    assert not solver.options[option]

    solver.applyOptionsOverride({"extrapolation": "linear", option: True})
    solver.applyOptionsOverride({"extrapolation": "off"})
    assert solver.options[option]


@pytest.mark.parametrize("retry", [False, True])
def test_a_failed_extrapolated_increment_is_retried_without_extrapolation(tmp_path, increments, retry):
    reference = displacements(run(tmp_path, deck(options="extrapolation=quadratic")))
    referenceSizes = [size for size, _ in increments.accepted]
    increments.attempts.clear()
    increments.accepted.clear()

    increments.failAttempts = {(3, INCREMENT, True)}
    result = displacements(run(tmp_path, deck(options=f"extrapolation=quadratic, retryWithoutExtrapolation={retry}")))
    attemptsOfThird = [attempt for attempt in increments.attempts if attempt[0] == 3]

    if retry:
        # the same increment, from a zero prediction: no cutback
        assert attemptsOfThird == [(3, INCREMENT, True), (3, INCREMENT, False)]
        assert [size for size, _ in increments.accepted] == referenceSizes
    else:
        # cut back, and the prediction is dropped
        assert attemptsOfThird[:2] == [(3, INCREMENT, True), (3, 0.25 * INCREMENT, False)]
    np.testing.assert_allclose(result, reference, rtol=0.0, atol=1e-12)


@pytest.mark.parametrize("retry", [False, True])
def test_the_accepted_history_survives_a_cutback_with_the_retry(tmp_path, increments, retry):
    increments.failAttempts = {(3, INCREMENT, True), (3, INCREMENT, False)}
    run(tmp_path, deck(options=f"extrapolation=quadratic, retryWithoutExtrapolation={retry}"))

    attemptAfterCutback = [attempt for attempt in increments.attempts if attempt[:2] == (3, 0.25 * INCREMENT)][0]
    assert attemptAfterCutback[2] == retry


@pytest.mark.parametrize("acrossConnectivityChange", [False, True])
def test_prediction_across_a_change_of_the_connectivity(tmp_path, increments, monkeypatch, acrossConnectivityChange):
    reference = displacements(run(tmp_path, deck(options="extrapolation=quadratic")))
    referenceAttempts = list(increments.attempts)
    increments.attempts.clear()

    # every increment reports a change of the connectivity, e.g., of a contact search: the system is rebuilt each time
    monkeypatch.setattr(
        NIST,
        "updateTopologyAndConnectivity",
        lambda self, model, step, offerModelModifiers=True: Change.LOCATIONS,
    )
    result = displacements(
        run(
            tmp_path,
            deck(options=f"extrapolation=quadratic, extrapolateAcrossConnectivityChange={acrossConnectivityChange}"),
        )
    )

    if acrossConnectivityChange:
        # the numbering of the degrees of freedom is unchanged: the same predictions as without the rebuilds
        assert increments.attempts == referenceAttempts
    else:
        assert not any(extrapolated for _, _, extrapolated in increments.attempts)
    np.testing.assert_allclose(result, reference, rtol=0.0, atol=1e-12)


def test_a_rebuild_that_renumbers_the_degrees_of_freedom_drops_the_prediction(tmp_path):
    model = run(tmp_path, deck())
    field = model.nodeFields["displacement"]
    digest = DofNumbering([field]).digest

    assert DofNumbering([field]).digest == digest
    renumbered = NodeField("displacement", field.dimension, list(reversed(field.nodes)))
    assert DofNumbering([renumbered]).digest != digest

    lastIncrement = np.zeros(field.dimension * len(field.nodes))
    solver = solverWithHistory("linear", [0.1], [lastIncrement], digest, extrapolateAcrossConnectivityChange=True)
    # a rebuild after a change of the connectivity alone (LOCATIONS), or at the first increment of a step that continues
    # from a checkpoint (NONE): the prediction survives only if the rebuilt system numbers the DOFs alike
    for change in (Change.LOCATIONS, Change.NONE):
        solver.system = SimpleNamespace(numbering=DofNumbering([renumbered]))
        assert not solver.predictionSurvivesRebuild(change)
        solver.system = SimpleNamespace(numbering=DofNumbering([field]))
        assert solver.predictionSurvivesRebuild(change)
    # never after a change of the mesh
    assert not solver.predictionSurvivesRebuild(Change.MESH)
    # a prediction without the digest of its numbering cannot be compared, and does not survive
    solver.dofNumberingOfLastIncrement = None
    assert not solver.predictionSurvivesRebuild(Change.NONE)
    solver.dofNumberingOfLastIncrement = digest
    # across a change of the connectivity only with the option
    solver.options["extrapolateAcrossConnectivityChange"] = False
    assert not solver.predictionSurvivesRebuild(Change.LOCATIONS)
    assert solver.predictionSurvivesRebuild(Change.NONE)


def test_a_rebuild_that_numbers_other_scalar_variables_drops_the_prediction():
    """Two constraints with one Lagrange multiplier each, one switched on and the other off, then the other way round
    (a constraint may switch within a step): the two discrete systems have as many degrees of freedom, but the last one
    numbers another multiplier. The digest names the scalar variables, by the names the model holds them under, so the
    prediction does not survive the switch."""

    digest = DofNumbering([], [ScalarVariable("tieA_0")]).digest

    # the same names in another run, with other objects: the same digest
    assert DofNumbering([], [ScalarVariable("tieA_0")]).digest == digest
    assert DofNumbering([], [ScalarVariable("tieB_0")]).digest != digest
    assert (
        DofNumbering([], [ScalarVariable("tieA_0"), ScalarVariable("tieB_0")]).digest
        != DofNumbering([], [ScalarVariable("tieB_0"), ScalarVariable("tieA_0")]).digest
    )

    solver = solverWithHistory("linear", [0.1], [[0.5]], digest, extrapolateAcrossConnectivityChange=True)
    solver.system = SimpleNamespace(numbering=DofNumbering([], [ScalarVariable("tieB_0")]))
    assert not solver.predictionSurvivesRebuild(Change.LOCATIONS)
    solver.system = SimpleNamespace(numbering=DofNumbering([], [ScalarVariable("tieA_0")]))
    assert solver.predictionSurvivesRebuild(Change.LOCATIONS)


def test_the_history_of_a_quadratic_extrapolation_is_checkpointed():
    solver = solverWithHistory("quadratic", [0.2, 0.1], [[1.0, 2.0], [3.0, 4.0]])
    restored = NIST(loadConfiguration({}), Journal(verbose=False), extrapolation="quadratic")
    restored.setRestartData(solver.getRestartData())

    np.testing.assert_array_equal(restored.dUOfIncrementBeforeLast, [1.0, 2.0])
    assert restored.timeStepOfIncrementBeforeLast.stepProgressIncrement == 0.2
    np.testing.assert_array_equal(predict(restored, 0.1)[0], predict(solver, 0.1)[0])
    # with the digest of the numbering the history is a vector of
    assert restored.dofNumberingOfLastIncrement == "the numbering"

    # a linear extrapolation keeps no older history
    linear = solverWithHistory("linear", [0.1], [[3.0, 4.0]])
    assert set(linear.getRestartData()) == {"prevTimeStep", "dUOfLastIncrement", "dofNumberingOfLastIncrement"}
