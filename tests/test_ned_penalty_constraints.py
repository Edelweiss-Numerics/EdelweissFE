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
"""``NED`` and the stiffness of penalty constraints in the critical time step, on the smallest model that has them:
point masses held and joined by penalty springs.

A penalty constraint owns no degrees of freedom; it adds stiffness to the nodes it couples, and with it raises their
frequencies (:class:`~edelweissfe.solvers.base.criticaltimestep.CriticalTimeStepWithThePenaltyConstraints`, theory page
"The critical time step of an explicit analysis"). Here the masses are point masses, whose elements have no stiffness, so
the springs are all there is, and the eigenvalues of the system are known exactly:

* a mass held by a penalty spring has the critical time step :math:`2\\sqrt{m/k}`, and so does the estimate;
* two equal masses joined by a penalty spring likewise; for unequal masses, the estimate is a lower bound;
* a step that asks for a larger increment than the springs allow is unstable without the estimate -- the oscillation
  grows without bound, energy appears from nowhere -- and stable with it.
"""

import numpy as np
import pytest

from edelweissfe.config.configurator import loadConfiguration
from edelweissfe.constraints.directionalspringpenalty import (
    Constraint as DirectionalSpringPenalty,
)
from edelweissfe.constraints.directionalspringpenalty import (
    DirectionalSpringPenaltySchema,
)
from edelweissfe.constraints.equalvaluepenalty import Constraint as EqualValuePenalty
from edelweissfe.constraints.equalvaluepenalty import EqualValuePenaltySchema
from edelweissfe.elements.pointmass import PointMass
from edelweissfe.journal.journal import Journal
from edelweissfe.models.femodel import FEModel
from edelweissfe.points.node import Node
from edelweissfe.sets.elementset import ElementSet
from edelweissfe.sets.nodeset import NodeSet
from edelweissfe.solvers.base.criticaltimestep import (
    CriticalTimeStepEstimate,
    CriticalTimeStepWithThePenaltyConstraints,
)
from edelweissfe.solvers.nonlinearexplicitdynamic import NED
from edelweissfe.steps.adaptivestepforexplicitsimulations import (
    AdaptiveStepForExplicitSimulations,
)
from edelweissfe.steps.stepmanager import StepActionCollection
from edelweissfe.utils.fieldoutput import FieldOutputController

#: The velocity along x the first mass starts with.
VELOCITY = 0.3


def massesOnSprings(masses: tuple, ground: float, tie: float) -> FEModel:
    """Point masses on a line, the first held to the ground by a spring of the stiffness ``ground`` along x
    (``directionalSpringPenalty``), all joined along x by an ``equalValuePenalty`` of the penalty ``tie``, which for two
    masses is a spring of :math:`k/2` between them. A spring of zero stiffness is left out."""

    model = FEModel(3)
    for number, mass in enumerate(masses, start=1):
        node = Node(number, np.array([float(number), 0.0, 0.0]))
        model.nodes[number] = node
        model.elements[number] = PointMass(number, [node], model, mass)
    model.elementSets["all"] = ElementSet("all", list(model.elements.values()))
    model.nodeSets["all"] = NodeSet("all", list(model.nodes.values()))
    model.nodeSets["first"] = NodeSet("first", [model.nodes[1]])
    if ground:
        model.constraints["ground"] = DirectionalSpringPenalty(
            "ground",
            model,
            model.nodeSets["first"],
            configuration=DirectionalSpringPenaltySchema(
                field="displacement", component=0, penalty=ground, nSet="first"
            ),
        )
    if tie:
        model.constraints["tie"] = EqualValuePenalty(
            "tie",
            model,
            model.nodeSets["all"],
            configuration=EqualValuePenaltySchema(field="displacement", component=0, penalty=tie, nSet="all"),
        )
    return model


def exactCriticalTimeStep(masses: tuple, ground: float, tie: float) -> float:
    """:math:`2/\\omega_\\mathrm{max}` of the masses on their springs along x, from the eigenvalues."""

    n = len(masses)
    K = tie * (np.eye(n) - np.full((n, n), 1.0 / n))
    K[0, 0] += ground
    s = 1.0 / np.sqrt(np.array(masses))
    return 2.0 / np.sqrt(np.max(np.linalg.eigvalsh(s[:, None] * K * s[None, :])))


class _RecordingNED(NED):
    """``NED``, recording the increment and the displacement along x of the first mass after each increment."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.history = []

    def acceptIncrement(self, step, model, timeStep):
        super().acceptIncrement(step, model, timeStep)
        displacement = model.nodeFields["displacement"]
        self.history.append((timeStep.timeIncrement, displacement["U"][displacement.indexOfNode(model.nodes[1]), 0]))


def run(model: FEModel, stepLength: float, timeIncrement: float, **options) -> _RecordingNED:
    """One explicit step of the given length, which asks for the given time increment; the first mass starts with
    :data:`VELOCITY` along x."""

    journal = Journal(verbose=False)
    model.prepareYourself(journal)
    displacement = model.nodeFields["displacement"]
    displacement["V"][displacement.indexOfNode(model.nodes[1]), 0] = VELOCITY

    fieldOutputController = FieldOutputController(model, journal)
    fieldOutputController.initializeJob()
    jobInfo = loadConfiguration(dict())
    solver = _RecordingNED(jobInfo, journal, **options)
    AdaptiveStepForExplicitSimulations(
        0,
        model,
        fieldOutputController,
        journal,
        jobInfo,
        solver,
        [],
        StepActionCollection(),
        stepLength=stepLength,
        startInc=timeIncrement / stepLength,
        maxInc=timeIncrement / stepLength,
        maxNumInc=100000,
    ).solve()
    fieldOutputController.finalizeJob()
    return solver


@pytest.mark.parametrize(
    "masses, ground, tie",
    [((2.0,), 50.0, 0.0), ((2.0, 2.0), 0.0, 80.0)],
    ids=["a mass held by a spring", "two equal masses joined by a spring"],
)
def test_the_estimate_is_exact_for_a_mass_held_by_a_spring_and_for_two_equal_masses_joined_by_one(masses, ground, tie):
    exact = exactCriticalTimeStep(masses, ground, tie)
    solver = run(massesOnSprings(masses, ground, tie), 5 * exact, exact)

    assert solver._criticalTimeStepWithThePenaltyConstraints.value == pytest.approx(exact, rel=1e-14)
    timeIncrements = np.array([timeIncrement for timeIncrement, _ in solver.history])
    np.testing.assert_allclose(timeIncrements[1:-1], NED.SolverSpecificOptions["courant-number"] * exact, rtol=1e-14)


@pytest.mark.parametrize("masses", [(1.0, 3.0), (3.0, 1.0), (1.0, 2.0, 5.0)])
def test_for_unequal_masses_the_estimate_is_a_lower_bound_within_the_bound_of_the_row_sums(masses):
    """The row sums bound the largest eigenvalue from above (Gershgorin): the estimate is below the exact critical time
    step, and not far below for springs of comparable frequency."""

    ground, tie = 50.0, 80.0
    exact = exactCriticalTimeStep(masses, ground, tie)
    solver = run(massesOnSprings(masses, ground, tie), 5 * exact, exact)

    estimate = solver._criticalTimeStepWithThePenaltyConstraints.value
    assert 0.6 * exact < estimate <= exact


def test_a_step_that_asks_for_more_than_the_springs_allow_is_unstable_without_the_estimate_and_stable_with_it(
    monkeypatch,
):
    """The step asks for twice the critical time step of the springs. The finite elements (point masses) have no stiffness,
    so without the estimate of the penalty constraints -- as before it existed -- the step takes that increment, and
    the oscillation grows by orders of magnitude within a few periods. With it, the increment is the courant number
    times the critical time step, and the first mass stays within the amplitude its energy allows against its spring
    to the ground, :math:`v_0 \\sqrt{m/k}`, up to the factor by which the central difference stretches the amplitude
    of the highest mode near the critical time step."""

    masses, ground, tie = (2.0, 2.0), 50.0, 80.0
    exact = exactCriticalTimeStep(masses, ground, tie)
    amplitude = VELOCITY * np.sqrt(masses[0] / ground)

    withTheEstimate = run(massesOnSprings(masses, ground, tie), 40 * exact, 2 * exact)
    assert max(abs(x) for _, x in withTheEstimate.history) < 2 * amplitude

    monkeypatch.setattr(
        CriticalTimeStepWithThePenaltyConstraints,
        "estimate",
        lambda self, *args: self._take(CriticalTimeStepEstimate(np.inf)),
    )
    withoutTheEstimate = run(massesOnSprings(masses, ground, tie), 40 * exact, 2 * exact)
    assert all(timeIncrement in (0.0, 2 * exact) for timeIncrement, _ in withoutTheEstimate.history[:-1])
    assert max(abs(x) for _, x in withoutTheEstimate.history) > 1e3 * amplitude
