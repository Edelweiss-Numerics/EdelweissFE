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
"""The backtracking safeguard of the Newton corrections of NIST (solver option ``backtracking``).

The backtracking is checked on residuals of known shape along the correction, where the step length it has to choose is
known (both merit functions, trial states that cannot be evaluated), and on a plastic plate, where it must not change
the converged solution.
"""

import numpy as np
import pytest

from edelweissfe.config.configurator import loadConfiguration
from edelweissfe.drivers.inputfiledrivensimulation import finiteElementSimulation
from edelweissfe.journal.journal import Journal
from edelweissfe.solvers.nonlinearimplicitstatic import NIST
from edelweissfe.utils.exceptions import CutbackRequest
from edelweissfe.utils.inputfileparser import parseInputFile

# --- on residuals of known shape ----------------------------------------------------------------------------------------

R0 = np.array([3.0, -4.0])  # the residual of the iterate
DDU = np.array([0.3, -0.4])  # the correction, a descent direction: DDU . R0 > 0


def backtrack(residualAt, merit="residual", maxCuts=5, stepLength=1.0):
    """The step length the backtracking chooses if the residual at step length a is residualAt(a), and the step
    lengths it tried."""
    solver = NIST(
        loadConfiguration({}),
        Journal(verbose=False),
        backtracking=True,
        backtrackingMerit=merit,
        backtrackingMaxCuts=maxCuts,
    )
    tried = []

    def evaluateAt(a):
        tried.append(a)
        return residualAt(a)

    chosen, evaluatedLast = solver.backtrack(evaluateAt, R0, DDU, stepLength)
    assert evaluatedLast == (chosen == tried[-1] and residualAt(chosen) is not None)
    return chosen, tried


def test_an_exact_newton_correction_is_taken_in_full():
    for merit in ("residual", "energy"):
        assert backtrack(lambda a: (1.0 - a) * R0, merit) == (1.0, [1.0])


def test_the_residual_merit_halves_until_the_residual_decreases():
    # the residual grows beyond a = 0.3 along the correction
    chosen, tried = backtrack(lambda a: (1.0 - a) * R0 + 10.0 * max(a - 0.3, 0.0) * R0)
    assert tried == [1.0, 0.5, 0.25] and chosen == 0.25


def test_the_energy_merit_cuts_to_the_zero_of_the_secant():
    # the work of the residual along the correction, g(a) = DDU . R(a), is linear with its zero at a = 0.5: the full step
    # overshoots by g(1) = -g(0), the secant hits the zero in one cut
    chosen, tried = backtrack(lambda a: (1.0 - 2.0 * a) * R0, "energy")
    assert tried == [1.0, 0.5] and chosen == pytest.approx(0.5)
    # an overshoot by less than 0.8 g(0) is accepted
    assert backtrack(lambda a: (1.0 - 1.7 * a) * R0, "energy") == (1.0, [1.0])


def test_the_energy_merit_of_no_descent_direction_falls_back_to_the_residual():
    solver = NIST(loadConfiguration({}), Journal(verbose=False), backtracking=True, backtrackingMerit="energy")
    tried = []

    def evaluateAt(a):
        tried.append(a)
        return (1.0 - a) * R0 + 10.0 * max(a - 0.3, 0.0) * R0

    assert solver.backtrack(evaluateAt, R0, -DDU)[0] == 0.25
    assert tried == [1.0, 0.5, 0.25]


def test_a_trial_state_that_cannot_be_evaluated_is_too_long_a_step():
    # e.g. a material that cannot integrate the increment: R is None beyond a = 0.6
    for merit in ("residual", "energy"):
        chosen, tried = backtrack(lambda a: None if a > 0.6 else (1.0 - a) * R0, merit)
        assert tried == [1.0, 0.5] and chosen == 0.5


def test_without_an_accepted_trial_the_best_one_is_taken():
    # the residual grows at every step length; the shortest trial grows least
    chosen, tried = backtrack(lambda a: (1.0 + a) * R0, maxCuts=2)
    assert tried == [1.0, 0.5, 0.25] and chosen == 0.25
    # of the trials that could be evaluated
    chosen, tried = backtrack(lambda a: None if a < 0.4 else (1.0 + a) * R0, maxCuts=2)
    assert tried == [1.0, 0.5, 0.25] and chosen == 0.5


def test_if_no_trial_state_can_be_evaluated_the_increment_is_cut_back():
    with pytest.raises(CutbackRequest, match="no step length of the backtracking"):
        backtrack(lambda a: None, maxCuts=3)


def test_the_backtracking_schedule():
    class Dirichlet:
        constrainedDofIndices = np.array([0])

    solver = NIST(loadConfiguration({}), Journal(verbose=False), backtracking=True)
    # the first correction of an increment only if it prescribes no Dirichlet increment
    assert not solver.backtrackingIsDue(0, np.array([0.1, 1.0]), [Dirichlet()])
    assert solver.backtrackingIsDue(0, np.array([0.0, 1.0]), [Dirichlet()])
    assert solver.backtrackingIsDue(1, np.array([0.1, 1.0]), [Dirichlet()])
    assert not NIST(loadConfiguration({}), Journal(verbose=False)).backtrackingIsDue(3, DDU, [])  # off by default


# --- on a plastic plate -------------------------------------------------------------------------------------------------


def deck(solverOptions: str = "extrapolation=off") -> str:
    """A plate in plane strain of von Mises plasticity with hardening, stretched beyond its yield strain."""
    return f"""
*material, name=VonMises, id=plasticity, provider=edelweiss
210000, 0.3, 550, 1000, 200, 1400

*section, name=section1, material=plasticity, type=plane, thickness=1
all

*job, name=job, domain=2d
*solver, solver=NIST, name=theSolver

*modelGenerator, generator=planeRectQuad, name=gen
l=10
h=10
nX=2
nY=2
elType=CPE4
elProvider=edelweiss

*step, solver=theSolver
stepLength=1, startInc=0.25, maxInc=0.25, minInc=1e-2, maxNumInc=100, maxIter=50
>>options, name=theSolver, {solverOptions}
>>dirichlet, name=left, nSet=gen_left, field=displacement, 1=0
>>dirichlet, name=bottom, nSet=gen_leftBottom, field=displacement, 2=0
>>dirichlet, name=right, nSet=gen_right, field=displacement, 1=0.08
"""


def run(tmp_path, text: str):
    path = tmp_path / "test.inp"
    path.write_text(text)
    model, _ = finiteElementSimulation(parseInputFile(str(path)), verbose=False, suppressPlots=True)
    return model


@pytest.mark.parametrize("merit", ["residual", "energy"])
def test_the_backtracking_leaves_the_converged_solution(tmp_path, monkeypatch, merit):
    reference = run(tmp_path, deck())

    stepLengths = []
    originalBacktrack = NIST.backtrack

    def backtrack(self, *args):
        stepLength, evaluated = originalBacktrack(self, *args)
        stepLengths.append(stepLength)
        return stepLength, evaluated

    monkeypatch.setattr(NIST, "backtrack", backtrack)
    safeguarded = run(tmp_path, deck(f"extrapolation=off, backtracking=True, backtrackingMerit={merit}"))

    assert safeguarded.time == pytest.approx(1.0)
    assert len(stepLengths) > 0
    U = safeguarded.nodeFields["displacement"]["U"]
    UReference = reference.nodeFields["displacement"]["U"]
    np.testing.assert_allclose(U, UReference, rtol=0.0, atol=1e-6 * np.max(np.abs(UReference)))


def test_an_unknown_merit_is_refused(tmp_path):
    with pytest.raises(ValueError, match="Unknown backtrackingMerit"):
        run(tmp_path, deck("backtracking=True, backtrackingMerit=energie"))
