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
"""A step of the explicit solver (``NED``) starts from the velocity and the acceleration the model holds, and leaves
the velocity at its end time (see the module documentation of
:mod:`~edelweissfe.solvers.nonlinearexplicitdynamic`).

* Two consecutive explicit steps are one: a column released under its own weight, solved by one step, and by two
  steps of half its length -- with one solver and with two -- ends in the same state. Each increment of motion adds
  the impulse of the net force of the increment before; across the step boundary that impulse is the one of the model's
  state, and the half increment the velocity lags behind at the end of the first step is completed by the model.
* The acceleration a step starts from is the model's, whoever wrote it: a body given a velocity and an acceleration,
  and no force, takes the half increment of the leapfrog's start with that acceleration and then moves uniformly.

The time increment is fixed (the critical one is replaced), so that the steps divide it exactly.
"""

from pathlib import Path

import numpy as np
import pytest

from edelweissfe.drivers.inputfiledrivensimulation import finiteElementSimulation
from edelweissfe.solvers.nonlinearexplicitdynamic import NED
from edelweissfe.utils.inputfileparser import parseInputFile

#: The time increment of every increment of motion, well below the stable one of the column.
_TIME_INCREMENT = 2.0e-3

#: The length of the explicit time, over which the column swings through about a period.
_DURATION = 0.2

_MODEL = """
*material, name=LinearElastic, id=linearelastic, provider=edelweiss
**E       nu     density
1.0e4,    0.0,   1.0

*section, name=section1, material=linearelastic, type=plane, thickness=1
all

*job, name=nedstepstartjob, domain=2d

*modelGenerator, generator=planeRectQuad, name=gen
l=1
h=4
nX=1
nY=4
elType=CPE4
elProvider=edelweiss
"""

_SOLVER = """
*solver, solver=NED, name={name}
courant-number=1.0
output-frequency=1
"""

_STEP = """
*step, type=adaptiveForExplicitSimulations, solver={solver}
maxInc=1, minInc=1e-12, maxNumInc=1000, maxIter=25, stepLength={length}
"""

_RELEASED_UNDER_ITS_WEIGHT = """
>>dirichlet, name=base, nSet=gen_bottom, field=displacement, 1=0.0, 2=0.0
>>bodyforce, name=weight, elSet=all, forceVector='0.0, -1.0', f(t)='1'
"""


@pytest.fixture
def fixedTimeIncrement(monkeypatch):
    """Every equation system's stable time increment is replaced by the fixed one (with a courant number of 1)."""

    monkeypatch.setattr(NED, "getCriticalTimeStepForExplicitDynamics", lambda self, model, U: _TIME_INCREMENT)


def _run(deck: str, directory: Path):
    """The model at the end of the deck's last step."""

    directory.mkdir(parents=True, exist_ok=True)
    inputFile = directory / "test.inp"
    inputFile.write_text(deck)
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.chdir(directory)
    try:
        model, _ = finiteElementSimulation(parseInputFile(str(inputFile)), verbose=False, suppressPlots=True)
    finally:
        monkeypatch.undo()
    return model


def _state(model) -> dict:
    """The entries of the state of the displacement field, by name."""

    field = model.nodeFields["displacement"]
    return {entry: np.array(field[entry]) for entry in ("U", "V", "A")}


@pytest.mark.parametrize("solversOfTheSteps", [("theSolver", "theSolver"), ("first", "second")])
def test_two_explicit_steps_are_one(tmp_path, fixedTimeIncrement, solversOfTheSteps):
    oneStep = _run(
        _MODEL
        + _SOLVER.format(name="theSolver")
        + _STEP.format(solver="theSolver", length=_DURATION)
        + _RELEASED_UNDER_ITS_WEIGHT,
        tmp_path / "one",
    )

    first, second = solversOfTheSteps
    solvers = "".join(_SOLVER.format(name=name) for name in sorted(set(solversOfTheSteps)))
    twoSteps = _run(
        _MODEL
        + solvers
        + _STEP.format(solver=first, length=0.5 * _DURATION)
        + _RELEASED_UNDER_ITS_WEIGHT
        + _STEP.format(solver=second, length=0.5 * _DURATION),
        tmp_path / "two",
    )

    expected, actual = _state(oneStep), _state(twoSteps)
    # the column swings: at the step boundary, about half a period after its release, it moves and accelerates
    assert np.max(np.abs(expected["V"])) > 0.1 * np.max(np.abs(expected["U"])) / _DURATION
    for entry in ("U", "V", "A"):
        scale = np.max(np.abs(expected[entry]))
        np.testing.assert_allclose(actual[entry], expected[entry], rtol=0.0, atol=1e-9 * scale, err_msg=entry)


def test_a_step_starts_from_the_acceleration_the_model_holds(tmp_path, fixedTimeIncrement, monkeypatch):
    velocity, acceleration = np.array([0.3, -0.2]), np.array([5.0, 10.0])
    beginStep = NED.beginStep

    def beginStepInAMovingState(self, step, model, fieldOutputController, outputmanagers):
        # as an implicit dynamic step would leave it: a velocity and an acceleration at the start time
        field = model.nodeFields["displacement"]
        field["V"][:] = velocity
        field["A"][:] = acceleration
        beginStep(self, step, model, fieldOutputController, outputmanagers)

    monkeypatch.setattr(NED, "beginStep", beginStepInAMovingState)

    # no support and no load: the body has no force, so after the first half increment it moves uniformly
    nIncrements = 25
    model = _run(
        _MODEL
        + _SOLVER.format(name="theSolver")
        + _STEP.format(solver="theSolver", length=nIncrements * _TIME_INCREMENT),
        tmp_path,
    )

    state = _state(model)
    velocityAfterTheHalfStep = velocity + 0.5 * _TIME_INCREMENT * acceleration
    np.testing.assert_allclose(state["V"], np.broadcast_to(velocityAfterTheHalfStep, state["V"].shape), rtol=1e-12)
    np.testing.assert_allclose(
        state["U"],
        np.broadcast_to(nIncrements * _TIME_INCREMENT * velocityAfterTheHalfStep, state["U"].shape),
        rtol=1e-10,
    )
    np.testing.assert_allclose(state["A"], 0.0, atol=1e-9 * np.max(np.abs(acceleration)))
