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
"""The quadratic line search (solver option ``lineSearch``) and the quasi-Newton fallback (``fallBackToQuasiNewton``)
of the implicit solvers.

The line search is checked on residuals of known shape, where the step length it has to choose is known, and on a
plastic plate, where it must not change the converged solution. The fallback is checked on the same plate with
``maxGrowIter=0``, which declares an increment diverging at its first check that does not converge: without the
fallback, the step fails; with it, every increment converges with the tangent of its first iteration -- the same matrix
in every linear solve of the increment.
"""

import numpy as np
import pytest

from edelweissfe.config.configurator import loadConfiguration
from edelweissfe.drivers.inputfiledrivensimulation import finiteElementSimulation
from edelweissfe.journal.journal import Journal
from edelweissfe.solvers.nonlinearimplicitstatic import NIST
from edelweissfe.solvers.nonlinearimplicitstaticparallelarclength import NISTPArcLength
from edelweissfe.utils.inputfileparser import parseInputFile


def deck(stepOptions: str = "", solverOptions: str = "extrapolation=off") -> str:
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
stepLength=1, startInc=0.25, maxInc=0.25, minInc=1e-2, maxNumInc=100, maxIter=50{stepOptions}
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


@pytest.fixture
def solves(monkeypatch):
    """The system matrices of the linear solves, a list per attempt of an increment, and the step lengths the line
    searches chose."""

    class Solves:
        matricesPerAttempt = []
        stepLengths = []

    originalSolveIncrement = NIST.solveIncrement
    originalLinearSolve = NIST.linearSolve
    originalLineSearch = NIST.quadraticLineSearch

    def solveIncrement(self, *args):
        Solves.matricesPerAttempt.append([])
        return originalSolveIncrement(self, *args)

    def linearSolve(self, A, b):
        Solves.matricesPerAttempt[-1].append(A.toarray())
        return originalLinearSolve(self, A, b)

    def quadraticLineSearch(self, ddU, *args):
        scaled = originalLineSearch(self, ddU, *args)
        Solves.stepLengths.append(float(np.max(np.abs(scaled))) / float(np.max(np.abs(ddU))))
        return scaled

    monkeypatch.setattr(NIST, "solveIncrement", solveIncrement)
    monkeypatch.setattr(NIST, "linearSolve", linearSolve)
    monkeypatch.setattr(NIST, "quadraticLineSearch", quadraticLineSearch)
    return Solves


# --- the line search on residuals of known shape ------------------------------------------------------------------------


def lineSearchStepLength(largestResidualOf, stepLengths: str = "0.8, 1.0, 1.2") -> float:
    """The step length the line search chooses if the largest residual at step length a is largestResidualOf(a)."""
    solver = NIST(loadConfiguration({}), Journal(verbose=False), lineSearchStepLengths=stepLengths)

    # one degree of freedom, with zero increment so far and the correction 1: the trial increment is the step length
    def assembleResidualAndTangent(U_n, dU, U_np, R, *args):
        R[:] = largestResidualOf(dU[0])
        return None, None, None

    solver.assembleResidualAndTangent = assembleResidualAndTangent
    vector = np.zeros(1)
    scaled = solver.quadraticLineSearch(
        np.ones(1),
        vector,
        vector,
        vector.copy(),
        np.zeros(1),
        None,
        None,
        None,
        None,
        None,
        {"dirichlet": {}},
        None,
        None,
    )
    return float(scaled[0])


def test_the_line_search_takes_the_minimum_of_the_parabola():
    # a Newton correction that is exact: the residual |1 - a| r_0 is smallest at the full step
    assert lineSearchStepLength(lambda a: abs(1.0 - a) * 7.0) == pytest.approx(1.0)
    # a residual that is itself a parabola: its minimum
    assert lineSearchStepLength(lambda a: (a - 0.9) ** 2 + 0.1) == pytest.approx(0.9)
    # more trial step lengths than three: a least squares fit, here exact
    assert lineSearchStepLength(lambda a: (a - 0.3) ** 2 + 0.1, "0.15, 0.4, 0.7, 1.0") == pytest.approx(0.3)


def test_the_line_search_step_length_is_limited():
    assert lineSearchStepLength(lambda a: (a - 3.0) ** 2) == pytest.approx(1.5)
    assert lineSearchStepLength(lambda a: (a + 1.0) ** 2) == pytest.approx(0.1)


def test_a_parabola_without_minimum_gives_the_end_of_the_range_where_it_is_smaller():
    assert lineSearchStepLength(lambda a: 2.0 - (a - 1.1) ** 2) == pytest.approx(0.1)
    assert lineSearchStepLength(lambda a: 2.0 - (a - 0.7) ** 2) == pytest.approx(1.5)
    # a straight line, whose fitted curvature is round-off of either sign
    assert lineSearchStepLength(lambda a: 1.0 + a) == pytest.approx(0.1)
    assert lineSearchStepLength(lambda a: 3.0 - a) == pytest.approx(1.5)


def test_the_line_search_schedule():
    solver = NIST(loadConfiguration({}), Journal(verbose=False), lineSearch=True)
    assert [iteration for iteration in range(12) if solver.lineSearchIsDue(iteration)] == [6, 8, 10]

    solver = NIST(
        loadConfiguration({}), Journal(verbose=False), lineSearchAfterIterations=0, lineSearchEveryIterations=1
    )
    assert not any(solver.lineSearchIsDue(iteration) for iteration in range(12))  # off by default


# --- on a plastic plate -------------------------------------------------------------------------------------------------


def test_the_line_search_leaves_the_converged_solution(tmp_path, solves):
    reference = run(tmp_path, deck())
    assert not solves.stepLengths

    searched = run(
        tmp_path,
        deck(
            solverOptions="extrapolation=off, lineSearch=True, lineSearchAfterIterations=0, lineSearchEveryIterations=1"
        ),
    )

    assert searched.time == pytest.approx(1.0)
    assert len(solves.stepLengths) > 0
    assert all(0.1 <= stepLength <= 1.5 for stepLength in solves.stepLengths)
    U = searched.nodeFields["displacement"]["U"]
    UReference = reference.nodeFields["displacement"]["U"]
    np.testing.assert_allclose(U, UReference, rtol=0.0, atol=1e-6 * np.max(np.abs(UReference)))


def test_too_few_line_search_step_lengths_are_refused(tmp_path):
    with pytest.raises(ValueError, match="three different positive step lengths"):
        run(tmp_path, deck(solverOptions="lineSearch=True, lineSearchStepLengths='1.0, 1.0, 0.5'"))


def test_without_the_fallback_a_diverging_increment_is_cut_back_until_the_step_fails(tmp_path):
    model = run(tmp_path, deck(stepOptions=", maxGrowIter=0"))
    assert model.time < 1.0


def test_the_fallback_continues_with_the_tangent_of_the_first_iteration(tmp_path, solves):
    # The first loaded increment stays elastic; the later ones are extrapolated into the plastic range, so that the
    # tangent of their first iteration, at the prediction, is a plastic one. (From the converged state, the fallback
    # would iterate with the elastic tangent through a plastic increment, which converges far too slowly.)
    reference = run(tmp_path, deck(solverOptions="extrapolation=linear"))
    newtonMatrices = [matrices for matrices in solves.matricesPerAttempt if len(matrices) > 1]
    solves.matricesPerAttempt.clear()

    fallBack = run(
        tmp_path, deck(stepOptions=", maxGrowIter=0", solverOptions="extrapolation=linear, fallBackToQuasiNewton=True")
    )

    # Newton's method updates the tangent of the plastic plate, the fallback does not
    assert newtonMatrices and any(not np.array_equal(matrices[0], matrices[-1]) for matrices in newtonMatrices)
    assert fallBack.time == pytest.approx(1.0)
    assert any(len(matrices) > 1 for matrices in solves.matricesPerAttempt)
    for matrices in solves.matricesPerAttempt:
        for matrix in matrices[1:]:
            np.testing.assert_array_equal(matrix, matrices[0])
    U = fallBack.nodeFields["displacement"]["U"]
    UReference = reference.nodeFields["displacement"]["U"]
    np.testing.assert_allclose(U, UReference, rtol=0.0, atol=1e-6 * np.max(np.abs(UReference)))


def test_neither_is_available_under_an_arc_length_controller():
    for option in ("lineSearch", "fallBackToQuasiNewton"):
        solver = NISTPArcLength(loadConfiguration({}), Journal(verbose=False), **{option: "True"})
        solver._arcLengthOptions = {"arcLengthController": "theController"}

        class Model:
            multiPointConstraints = {}
            modelModifiers = {}
            additionalParameters = {}

        class Step:
            actions = {"indirectcontrol": {"theController": object()}}

        with pytest.raises(NotImplementedError, match="arc length controller"):
            solver.beginStep(Step(), Model(), None, {})
