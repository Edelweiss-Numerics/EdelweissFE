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
"""What the load parameter of the arc length method scales (``NISTPArcLength``; see the solver documentation, "Arc
length control").

- The prescribed displacements of a step follow the load parameter, like its loads: a prescribed degree of freedom
  moves with the load parameter, and only with it.
- At the end of an arc length step, every load keeps the value the step applied, f(0) + lambda (f(1) - f(0)) with its
  own amplitude, and the next step starts from it.
- Every arc length step scales its own reference load -- the change of the loads over the step -- from the load
  parameter zero.

The model is the cantilever of ``testfiles/edelweiss-only/ArcLengthPenaltyConstraint``: linear elastic, bent down onto
a rigid floor (an elastic penalty), by loads that do not follow the deformation. Its state depends only on the
prescribed displacements and the loads it ends with, not on the path that led there, so NIST, with the same values
prescribed, is an independent reference for every end state.
"""

from pathlib import Path

import numpy as np
import pytest

from edelweissfe.drivers.inputfiledrivensimulation import finiteElementSimulation
from edelweissfe.solvers.nonlinearimplicitstaticparallelarclength import NISTPArcLength
from edelweissfe.timesteppers.timestep import TimeStep
from edelweissfe.utils.exceptions import ReachedMaxIterations
from edelweissfe.utils.inputfileparser import parseInputFile

_DECKS = Path(__file__).resolve().parent.parent / "testfiles" / "edelweiss-only"
_MODEL = _DECKS / "ArcLengthPenaltyConstraint" / "model.inp"

#: The vertical displacement of the tip at the end of the decks' arc length steps.
TIP_DISPLACEMENT = -0.25

#: The change of the clamp's vertical displacement that ``ArcLengthPrescribedDisplacement`` prescribes for its step.
SETTLEMENT = -0.2

#: The weight of a quarter of a square element of edge 0.5 under the body force (0, -0.005): what the self-weight puts on
#: the tip, a corner node of the mesh.
SELF_WEIGHT_ON_THE_TIP = -0.005 * 0.5 * 0.5 / 4.0

#: The model solved by NIST with the end state prescribed: the clamp's settlement, and the tip displacement.
_NIST_REFERENCE = """
*include, input={model}

*solver, solver=NIST, name=theSolver

*step, solver=theSolver
startInc=0.1, maxInc=0.1, minInc=1e-6, maxNumInc=100, maxIter=25, stepLength=1
>>dirichlet, name=clamp, nSet=gen_left, field=displacement, 1=0.0, 2={settlement}
>>dirichlet, name=tipDisplacement, nSet=gen_rightTop, field=displacement, 2={tip}
>>bodyforce, name=selfWeight, elSet=gen_all, forceVector='0.0, -0.005', f(t)='1'
"""


#: A step without arc length control that declares no change: it holds every load.
_HOLDING_STEP = """
*step, solver=theSolver
startInc=0.5, maxInc=0.5, minInc=1e-6, maxNumInc=100, maxIter=25, stepLength=1
>>options, name=theSolver, arcLengthController=off
"""


def solve(inputFile: Path):
    model, _ = finiteElementSimulation(parseInputFile(str(inputFile)), verbose=False, suppressPlots=True)
    return model


def solveWithNIST(tmp_path, settlement: float, tip: float):
    """The reference: NIST with the clamp's settlement and the tip displacement prescribed."""
    path = tmp_path / "reference.inp"
    # repr of a float: the shortest text that reads back as the same number
    path.write_text(_NIST_REFERENCE.format(model=_MODEL, settlement=repr(float(settlement)), tip=repr(float(tip))))
    return solve(path)


def nodeRows(model, nodeSetName: str) -> list:
    field = model.nodeFields["displacement"]
    return [field.indexOfNode(node) for node in model.nodeSets[nodeSetName]]


def arcLengthPenaltyConstraintDeck() -> str:
    """The deck ``ArcLengthPenaltyConstraint``, to be written elsewhere: its arc length step bends the cantilever down
    by the tip load, ramped linearly, under the self-weight, which the step holds at a constant amplitude."""
    deck = (_DECKS / "ArcLengthPenaltyConstraint" / "test.inp").read_text()
    return deck.replace("*include, input=model.inp", "*include, input={:}".format(_MODEL))


def assertSameExternalLoad(load, reference):
    """The same external load, up to the order in which its parts were summed."""
    np.testing.assert_allclose(load, reference, rtol=0.0, atol=1e-12 * np.max(np.abs(reference)))


def assertReachesTheReferenceState(model, reference, tipLoadFactor: float):
    """The same displacements and internal forces as the reference, and a tip load that the reference's internal force
    at the tip balances: at the tip, f_int = -1 * tipLoadFactor + the self-weight on the tip."""

    U, P = model.nodeFields["displacement"]["U"], model.nodeFields["displacement"]["P"]
    UReference, PReference = reference.nodeFields["displacement"]["U"], reference.nodeFields["displacement"]["P"]
    np.testing.assert_allclose(U, UReference, rtol=0.0, atol=1e-8)
    np.testing.assert_allclose(P, PReference, rtol=0.0, atol=1e-8)

    (tip,) = nodeRows(reference, "gen_rightTop")
    np.testing.assert_allclose(tipLoadFactor, SELF_WEIGHT_ON_THE_TIP - PReference[tip, 1], rtol=1e-7)


@pytest.fixture
def clampHistory(monkeypatch):
    """What the clamp of ``ArcLengthPrescribedDisplacement`` sees in the iterations of every increment: per evaluation of
    the model, the increment of the load parameter of the trial state, the clamp's increment, and its prescribed change
    over the step; per convergence check, the clamp's rows of the out-of-balance force."""

    class ClampHistory:
        evaluations = []
        checkedResiduals = []
        constrainedDofIndices = None

    originalAssembleLoads = NISTPArcLength.assembleLoadsAtIncrementEnd
    originalCheckConvergence = NISTPArcLength.checkConvergence

    def assembleLoadsAtIncrementEnd(self, stepActions, U_np, dU, P, K, timeStep):
        clamp = stepActions["dirichlet"]["clamp"]
        ClampHistory.constrainedDofIndices = clamp.constrainedDofIndices
        changeOverTheStep = clamp.getPrescribedIncrement(TimeStep(timeStep.number, 1.0, 1.0, 0.0, 0.0, 0.0)).flatten()
        result = originalAssembleLoads(self, stepActions, U_np, dU, P, K, timeStep)
        loadParameterIncrement = self.arcLengthIncrement.loadParameter - self.Lambda
        ClampHistory.evaluations.append(
            (loadParameterIncrement, np.array(dU[clamp.constrainedDofIndices]), changeOverTheStep)
        )
        return result

    def checkConvergence(self, R, *args):
        ClampHistory.checkedResiduals.append(np.array(R[ClampHistory.constrainedDofIndices]))
        return originalCheckConvergence(self, R, *args)

    monkeypatch.setattr(NISTPArcLength, "assembleLoadsAtIncrementEnd", assembleLoadsAtIncrementEnd)
    monkeypatch.setattr(NISTPArcLength, "checkConvergence", checkConvergence)
    return ClampHistory


def test_the_prescribed_degrees_of_freedom_are_no_unknowns_of_the_iterations(tmp_path, monkeypatch, clampHistory):
    """In every iteration, the clamp of ``ArcLengthPrescribedDisplacement`` stands where the load parameter of the
    iterate puts it: its increment is the increment of the load parameter times its prescribed change over the step.
    Its rows of the out-of-balance force hold no equation of the problem, and are zero where the convergence is
    checked."""

    monkeypatch.chdir(tmp_path)
    solve(_DECKS / "ArcLengthPrescribedDisplacement" / "test.inp")

    assert clampHistory.evaluations and clampHistory.checkedResiduals
    for loadParameterIncrement, clampIncrement, changeOverTheStep in clampHistory.evaluations:
        assert np.any(changeOverTheStep != 0.0)
        np.testing.assert_allclose(clampIncrement, loadParameterIncrement * changeOverTheStep, rtol=1e-12, atol=1e-15)
    for clampResidual in clampHistory.checkedResiduals:
        np.testing.assert_array_equal(clampResidual, 0.0)


def test_a_prescribed_displacement_follows_the_load_parameter(tmp_path, monkeypatch):
    """``ArcLengthPrescribedDisplacement``: the clamp settles by its prescribed change scaled by the load parameter, while
    the tip load grows with it. Where the step ends, the settlement is the load parameter times the prescribed change --
    whichever path the iterations took -- and the state is the one NIST reaches with that settlement prescribed."""

    monkeypatch.chdir(tmp_path)
    model = solve(_DECKS / "ArcLengthPrescribedDisplacement" / "test.inp")
    loadParameter = model.solvers["theSolver"].Lambda
    U = model.nodeFields["displacement"]["U"]

    (tip,) = nodeRows(model, "gen_rightTop")
    np.testing.assert_allclose(U[tip, 1], TIP_DISPLACEMENT, rtol=1e-12, err_msg="the end of the step")
    assert loadParameter > 0.0
    clamp = nodeRows(model, "gen_left")
    np.testing.assert_allclose(U[clamp, 1], loadParameter * SETTLEMENT, rtol=1e-12)
    np.testing.assert_array_equal(U[clamp, 0], 0.0)

    reference = solveWithNIST(tmp_path, settlement=loadParameter * SETTLEMENT, tip=TIP_DISPLACEMENT)
    assertReachesTheReferenceState(model, reference, tipLoadFactor=loadParameter)


@pytest.fixture
def steps(monkeypatch):
    """Per step of a run: the external load at every evaluation of the model (the negative of the residual
    contribution of the loads), the load parameter at the first evaluation of an arc length step, and the displacements
    and the load parameter at the end of the step."""

    class Steps:
        externalLoads = []
        startLoadParameters = []
        endDisplacements = []
        endLoadParameters = []

    originalBeginStep = NISTPArcLength.beginStep
    originalAssembleLoads = NISTPArcLength.assembleLoadsAtIncrementEnd
    originalStepEnd = NISTPArcLength.applyStepActionsAtStepEnd

    def beginStep(self, *args):
        Steps.externalLoads.append([])
        return originalBeginStep(self, *args)

    def assembleLoadsAtIncrementEnd(self, stepActions, U_np, dU, P, K, timeStep):
        before = np.array(P)
        result = originalAssembleLoads(self, stepActions, U_np, dU, P, K, timeStep)
        if not Steps.externalLoads[-1] and self.arcLengthIncrement is not None:
            Steps.startLoadParameters.append(self.arcLengthIncrement.loadParameter)
        Steps.externalLoads[-1].append(before - np.array(P))
        return result

    def applyStepActionsAtStepEnd(self, model, stepActions):
        Steps.endDisplacements.append(model.nodeFields["displacement"]["U"].copy())
        Steps.endLoadParameters.append(self.Lambda)
        return originalStepEnd(self, model, stepActions)

    monkeypatch.setattr(NISTPArcLength, "beginStep", beginStep)
    monkeypatch.setattr(NISTPArcLength, "assembleLoadsAtIncrementEnd", assembleLoadsAtIncrementEnd)
    monkeypatch.setattr(NISTPArcLength, "applyStepActionsAtStepEnd", applyStepActionsAtStepEnd)
    return Steps


def test_a_step_after_an_arc_length_step_holds_the_loads_it_ended_with(tmp_path, monkeypatch, steps):
    """At the end of an arc length step, every load keeps the value the step applied, f(0) + lambda (f(1) - f(0)) with
    its own amplitude: in ``ArcLengthPenaltyConstraint``, the tip load, declared in the step and ramped linearly,
    keeps lambda times its change, and the self-weight, which the step holds at a constant amplitude (a dead load),
    keeps its full value, whatever the load parameter. A step that holds every load then starts from the external load
    the arc length step ended with, and leaves the model where it is."""

    monkeypatch.chdir(tmp_path)
    path = tmp_path / "test.inp"
    path.write_text(arcLengthPenaltyConstraintDeck() + _HOLDING_STEP)
    solve(path)

    arcLengthStep, holdingStep = steps.externalLoads
    assert 0.0 < steps.endLoadParameters[0] < 1.0
    assertSameExternalLoad(holdingStep[0], arcLengthStep[-1])
    np.testing.assert_allclose(steps.endDisplacements[1], steps.endDisplacements[0], rtol=0.0, atol=1e-8)


def test_every_arc_length_step_scales_its_own_reference_load_from_zero(tmp_path, monkeypatch, steps):
    """``ArcLengthTwoSteps``: the second arc length step scales its own change of the tip load from the load parameter
    zero, so it starts from the external load the first step ended with -- as the step after it, which holds every
    load, starts from the external load the second ended with."""

    monkeypatch.chdir(tmp_path)
    solve(_DECKS / "ArcLengthTwoSteps" / "test.inp")

    first, second, holding = steps.externalLoads
    assertSameExternalLoad(second[0], first[-1])
    assertSameExternalLoad(holding[0], second[-1])
    assert steps.startLoadParameters == [0.0, 0.0]


def test_two_arc_length_steps_reach_the_state_of_one(tmp_path, monkeypatch, steps):
    """``ArcLengthTwoSteps`` ends where NIST takes the model with the tip displacement prescribed. Its tip load is the
    change the two steps declared, each scaled by the load parameter its step ended at, and the model publishes the
    load parameter of the second step; the step that holds every load at the end keeps the state."""

    monkeypatch.chdir(tmp_path)
    model = solve(_DECKS / "ArcLengthTwoSteps" / "test.inp")
    first, second, holding = steps.endLoadParameters
    assert first > 0.0 and second > 0.0 and holding == 0.0

    reference = solveWithNIST(tmp_path, settlement=0.0, tip=TIP_DISPLACEMENT)
    assertReachesTheReferenceState(model, reference, tipLoadFactor=first + second)
    assert model.additionalParameters["arc length parameter"] == second


def test_a_cut_back_increment_retries_from_the_accepted_load_parameter(tmp_path, monkeypatch):
    """The load parameter advances with the solution, only when an increment has converged: an attempt that fails after
    its iterations have moved the load parameter leaves it where the last accepted increment left it, the cut back
    increment starts from there, and the step ends in the state an uninterrupted run reaches.

    ``ArcLengthPenaltyConstraint``, with its third increment failing once, at its first convergence check: by
    then the prediction (or the first correction) has moved the load parameter."""

    monkeypatch.chdir(tmp_path)
    deck = tmp_path / "test.inp"
    deck.write_text(arcLengthPenaltyConstraintDeck())
    uninterrupted = solve(deck)

    #: the load parameter at the start of the step and after every accepted increment
    acceptedLoadParameters = []
    #: per attempt of an increment: the solver's load parameter at its start, and the last accepted one
    loadParametersAtTheStartOfAttempts = []
    failedIterate = {}

    originalSolveIncrement = NISTPArcLength.solveIncrement
    originalCheckConvergence = NISTPArcLength.checkConvergence
    originalAcceptIncrement = NISTPArcLength.acceptIncrement

    def solveIncrement(self, *args):
        loadParametersAtTheStartOfAttempts.append((self.Lambda, acceptedLoadParameters[-1]))
        return originalSolveIncrement(self, *args)

    def checkConvergence(self, R, ddU, dU, F, FConstraints, iterationCounter, incrementResidualHistory):
        if len(acceptedLoadParameters) == 3 and not failedIterate:
            failedIterate["loadParameter"] = self.arcLengthIncrement.loadParameter
            raise ReachedMaxIterations("failing once, for the test")
        return originalCheckConvergence(self, R, ddU, dU, F, FConstraints, iterationCounter, incrementResidualHistory)

    def acceptIncrement(self, step, model, timeStep):
        originalAcceptIncrement(self, step, model, timeStep)
        acceptedLoadParameters.append(self.Lambda)

    originalBeginStep = NISTPArcLength.beginStep

    def beginStep(self, *args):
        result = originalBeginStep(self, *args)
        acceptedLoadParameters.append(self.Lambda)
        return result

    monkeypatch.setattr(NISTPArcLength, "beginStep", beginStep)
    monkeypatch.setattr(NISTPArcLength, "solveIncrement", solveIncrement)
    monkeypatch.setattr(NISTPArcLength, "checkConvergence", checkConvergence)
    monkeypatch.setattr(NISTPArcLength, "acceptIncrement", acceptIncrement)

    model = solve(deck)

    assert failedIterate, "no attempt failed, so this proves nothing"
    assert failedIterate["loadParameter"] != acceptedLoadParameters[2], "the failed attempt never moved the parameter"
    # one attempt per accepted increment, and the failed one; the step start counts as accepted
    assert len(loadParametersAtTheStartOfAttempts) == len(acceptedLoadParameters)
    for atTheStart, lastAccepted in loadParametersAtTheStartOfAttempts:
        assert atTheStart == lastAccepted

    loadParameter = model.solvers["theSolver"].Lambda
    assert model.additionalParameters["arc length parameter"] == loadParameter
    np.testing.assert_allclose(loadParameter, uninterrupted.solvers["theSolver"].Lambda, rtol=1e-8)
    np.testing.assert_allclose(
        model.nodeFields["displacement"]["U"], uninterrupted.nodeFields["displacement"]["U"], rtol=0.0, atol=1e-8
    )
