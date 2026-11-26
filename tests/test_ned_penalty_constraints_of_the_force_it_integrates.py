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
"""``NED`` bounds the penalty constraints with the stiffness of the net force the next increment integrates: right after
every evaluation of the net force, at the state it was evaluated at (theory page "The critical time step of an explicit
analysis", "At which state").

The central difference method integrates in each increment the net force evaluated at the end of the increment before,
over the average of its own time increment and the one before. The tests take, after every increment, the estimate of
the constraints as they acted in its net force -- on its connectivity and its mesh, at its state -- and pin that ``NED``
holds that estimate when the next increment starts, and that the next increment is within it:

* a point paired anew by the periodic contact search, already in the master, is in the net force of the increment the
  search runs in, and so in the critical time step of the next increment; the first increment that integrates its force
  does so over the average of its own time increment and the one taken without it;
* after a refinement in the middle of a step, the constraints are bounded at once at the state on the refined mesh,
  with its smaller masses: the first increment on it, which integrates no force, and the one after it, which integrates
  the first force on it, are both within the critical time step of the springs on the new masses.

With ``follow-critical-time-step``, the constraints are bounded over the states the next increment can reach as well,
and once more after a contact search, on the new pairing (theory page, "Over the next increment"): a point paired anew
by the search, already in the master, is in the time increment of the first increment that evaluates its force, not
half an increment late; every increment is still within the critical time step of the force it integrates.
"""

from typing import NamedTuple

import numpy as np
import pytest

from edelweissfe.drivers.inputfiledrivensimulation import finiteElementSimulation
from edelweissfe.numerics.doflocations import DofLocations
from edelweissfe.solvers.base.criticaltimestep import (
    CriticalTimeStepWithThePenaltyConstraints,
)
from edelweissfe.solvers.nonlinearexplicitdynamic import NED
from edelweissfe.utils.inputfileparser import parseInputFile

#: A block that starts 0.035 deep in a block held at its bottom, beyond the search distance of their penalty contact,
#: and is pulled out of it: a search, every 20th increment, pairs its nodes while they are still in the master.
_SEARCH_DECK = """
*material, name=LinearElastic, id=linearelastic, provider=edelweiss
1.8e4, 0.3, 1.0

*job, name=nedPenaltyAfterTheSearch, domain=3d

*solver, solver=NED, name=theSolver
output-frequency=100000
contact-update-frequency=20

*modelGenerator, generator=boxGen, name=lower
nX=2
nY=2
nZ=1
lX=2
lY=2
lZ=1
elProvider=edelweiss
elType=C3D8

*modelGenerator, generator=boxGen, name=upper
nX=2
nY=2
nZ=2
x0=0.3
y0=0.3
z0=0.965
lX=0.6
lY=0.6
lZ=1
elProvider=edelweiss
elType=C3D8

*modelGenerator, generator=surfaceElementGenerator, name=slaveFacets
surface=upper_back
name=slaveSurf

*modelGenerator, generator=surfaceElementGenerator, name=masterFacets
surface=lower_front
name=masterSurf

*section, name=section1, material=linearelastic, type=solid
lower_all
upper_all

*constraint, name=contact, type=nodeToDeformableSurfacePenalty
slaveSurface=slaveSurf_facets, masterSurface=masterSurf_facets
penalty=5e5, type=linear, searchDistance=0.02, sliding=small

*step, type=adaptiveForExplicitSimulations, solver=theSolver
maxInc=1, minInc=1e-12, maxNumInc=100000, maxIter=25, stepLength=0.15
>>dirichlet, name=fixLower, nSet=lower_back, field=displacement, 1=0.0, 2=0.0, 3=0.0
>>dirichlet, name=pullUp, nSet=upper_front, field=displacement, 1=0.0, 2=0.0, 3=0.045
"""

#: A bar held at its left end and by stiff springs along x at its right end, refined once, live: the springs set the
#: critical time step, and the refinement lowers the masses they act on.
_REFINEMENT_DECK = """
*material, name=LinearElastic, id=linearelastic
2.0e4, 0.2, 1.0

*section, name=section1, material=linearelastic, type=solid
gen_all

*job, name=nedPenaltyAfterTheRefinement, domain=3d

*solver, solver=NED, name=theSolver
output-frequency=20
topology-check-frequency=20

*modelGenerator, generator=boxGen, name=gen
nX=2
nY=1
nZ=1
lX=20
lY=10
lZ=10
elType=C3D20R

*modelModifier, type=hAdaptivity, name=amrLive
>>marker, type=elementSet, elSet=gen_all, initialOnly=False
refineElSet=gen_all
maxLevel=1

*constraint, name=springs, type=directionalSpringPenalty
field=displacement, component=0, penalty=1e7, nSet=gen_right

*step, type=adaptiveForExplicitSimulations, solver=theSolver
maxInc=1, minInc=1e-14, maxNumInc=100000, maxIter=25, stepLength=0.1
>>dirichlet, name=holdLeft, nSet=gen_left, field=displacement, 1=0.0, 2=0.0, 3=0.0
"""


class Increment(NamedTuple):
    """An increment that moves, and the net force it integrates."""

    #: Its time increment.
    timeIncrement: float
    #: The critical time step of the parts with the penalty constraints, as ``NED`` holds it when the increment starts.
    held: float
    #: The same, estimated anew from the constraints as they acted in the net force the increment integrates: at the
    #: state and on the connectivity that force was evaluated at, the end of the increment before (the start of the
    #: step for the first).
    ofTheForceItIntegrates: float
    #: The locations of the discrete system the increment evaluates its net force on, new after a search at its start.
    locations: DofLocations
    #: The number of degrees of freedom, larger after a refinement.
    nDof: int
    #: The courant number of the solver.
    courantNumber: float


def _run(tmp_path, monkeypatch, deck: str) -> list[Increment]:
    """Run a deck and take every increment that moves (:class:`Increment`)."""

    monkeypatch.setenv("OMP_NUM_THREADS", "1")
    increments = []
    ofTheLastForce = []
    solveIncrement = NED.solveIncrement

    def estimateAt(solver, U, dU, timeStep) -> float:
        estimate = CriticalTimeStepWithThePenaltyConstraints()
        estimate.estimate(
            solver.system,
            solver._rowSumsOfThePenaltyConstraints(U, dU, timeStep),
            solver._criticalTimeStepOfTheElements,
            solver._criticalTimeStepOfTheEntities,
            solver.mpcTransformation,
        )
        return estimate.value

    def recordingSolveIncrement(self, U_n, dU, V, netForce, internalForce, Minv, stepActions, model, timeStep, *args):
        if not ofTheLastForce:
            # the net force the step starts from is the one of the model's state, with no increment
            ofTheLastForce.append(
                estimateAt(
                    self,
                    U_n,
                    self.system.locations.constructDofVector(),
                    self._zeroLengthTimeStepAtTheModelsState(model),
                )
            )
        if timeStep.timeIncrement > 0.0:
            increments.append(
                Increment(
                    timeStep.timeIncrement,
                    self._criticalTimeStepWithThePenaltyConstraints.value,
                    ofTheLastForce[-1],
                    self.system.locations,
                    self.system.numbering.nDof,
                    self.options["courant-number"],
                )
            )
        U, V, netForce = solveIncrement(
            self, U_n, dU, V, netForce, internalForce, Minv, stepActions, model, timeStep, *args
        )
        if timeStep.timeIncrement > 0.0:
            ofTheLastForce.append(estimateAt(self, U, dU, timeStep))
        return U, V, netForce

    monkeypatch.setattr(NED, "solveIncrement", recordingSolveIncrement)
    path = tmp_path / "deck.inp"
    path.write_text(deck)
    finiteElementSimulation(parseInputFile(str(path)), verbose=False, suppressPlots=True)
    return increments


def _assertEveryIncrementIsWithinTheConstraintsOfTheForceItIntegrates(increments: list[Increment]):
    for number, increment in enumerate(increments):
        message = "increment {:}".format(number)
        largestTimeIncrement = increment.courantNumber * increment.ofTheForceItIntegrates
        assert increment.timeIncrement <= largestTimeIncrement * (1.0 + 1e-14), message


def test_a_point_paired_anew_by_the_search_is_in_the_critical_time_step_of_the_first_increment_that_integrates_its_force(
    tmp_path, monkeypatch
):
    increments = _run(tmp_path, monkeypatch, _SEARCH_DECK)

    _assertEveryIncrementIsWithinTheConstraintsOfTheForceItIntegrates(increments)
    for number, increment in enumerate(increments):
        assert increment.held == increment.ofTheForceItIntegrates, "increment {:}: not estimated at its force".format(
            number
        )
    # The test proves something only if a search paired points that were already in the master: the critical time step
    # of the constraints drops with the force of the increment the search ran in, and the next increment, the first
    # that integrates that force, is the shorter one.
    ofTheForce = np.array([increment.ofTheForceItIntegrates for increment in increments])
    drops = np.flatnonzero(ofTheForce[1:] < 0.9 * ofTheForce[:-1]) + 1
    assert drops.size, "no search paired a point that was already in the master"
    searched, first = increments[drops[0] - 1], increments[drops[0]]
    assert searched.locations is not increments[drops[0] - 2].locations, "the first drop is not after a search"
    assert first.timeIncrement == pytest.approx(first.courantNumber * first.ofTheForceItIntegrates, rel=1e-14)
    assert first.timeIncrement < 0.9 * searched.timeIncrement


#: The search deck, with the time increment following the critical time step.
_FOLLOWED_SEARCH_DECK = _SEARCH_DECK.replace(
    "contact-update-frequency=20", "contact-update-frequency=20\nfollow-critical-time-step=True"
)


def _firstIncrementThatEvaluatesTheForceOfANewPairing(increments: list[Increment]) -> tuple[int, np.ndarray]:
    """The first increment whose net force, evaluated at its end, is far stiffer than the one before: the first on the
    connectivity of a search that paired points already in the master. Returns its index, and the critical time step of
    the force each increment evaluates (the one the next increment integrates; the last increment has none)."""

    ofTheForceItEvaluates = np.array([increment.ofTheForceItIntegrates for increment in increments[1:]])
    drops = np.flatnonzero(ofTheForceItEvaluates[1:] < 0.9 * ofTheForceItEvaluates[:-1]) + 1
    assert drops.size, "no search paired a point that was already in the master"
    first = drops[0]
    assert increments[first].locations is not increments[first - 1].locations, "the first drop is not after a search"
    return first, ofTheForceItEvaluates


def test_followed_a_point_paired_anew_by_the_search_is_in_the_time_increment_of_the_first_increment_on_it(
    tmp_path, monkeypatch
):
    increments = _run(tmp_path, monkeypatch, _FOLLOWED_SEARCH_DECK)

    _assertEveryIncrementIsWithinTheConstraintsOfTheForceItIntegrates(increments)
    first, ofTheForceItEvaluates = _firstIncrementThatEvaluatesTheForceOfANewPairing(increments)
    increment = increments[first]
    assert increment.timeIncrement <= increment.courantNumber * ofTheForceItEvaluates[first] * (1.0 + 1e-14)
    assert increment.timeIncrement < 0.9 * increments[first - 1].timeIncrement


def test_followed_without_the_bound_after_the_search_the_first_increment_on_the_new_pairing_is_the_long_one(
    tmp_path, monkeypatch
):
    """The control: the search before the increment as without ``follow-critical-time-step``, with no bound over the
    next increment on the new pairing; the bound over the next increment taken with the force before it has the old
    pairing."""

    def searchAlone(self, step, model):
        self.updateDiscreteSystem(model, model.updateConnectivity())

    monkeypatch.setattr(NED, "_searchTheContactsAnew", searchAlone)
    increments = _run(tmp_path, monkeypatch, _FOLLOWED_SEARCH_DECK)

    first, ofTheForceItEvaluates = _firstIncrementThatEvaluatesTheForceOfANewPairing(increments)
    increment = increments[first]
    assert increment.timeIncrement > 1.1 * increment.courantNumber * ofTheForceItEvaluates[first]


def test_after_a_refinement_the_constraints_are_bounded_with_the_masses_of_the_refined_mesh_at_once(
    tmp_path, monkeypatch
):
    increments = _run(tmp_path, monkeypatch, _REFINEMENT_DECK)

    _assertEveryIncrementIsWithinTheConstraintsOfTheForceItIntegrates(increments)
    nDof = np.array([increment.nDof for increment in increments])
    (refined,) = np.flatnonzero(nDof[1:] > nDof[:-1]) + 1
    # The first increment on the refined mesh integrates no force (the net force of a new numbering starts at zero);
    # the next one integrates the first force on the refined mesh, over the average of the two increments. The springs
    # set the critical time step on both meshes, and the refined one, with its smaller masses, a shorter one.
    before, first, second = increments[refined - 1], increments[refined], increments[refined + 1]
    assert second.held == second.ofTheForceItIntegrates
    assert second.ofTheForceItIntegrates < 0.9 * before.ofTheForceItIntegrates
    assert second.timeIncrement == pytest.approx(second.courantNumber * second.ofTheForceItIntegrates, rel=1e-14)
    # The springs are linear, so their critical time step on the refined mesh does not depend on the state: the first
    # increment on it already holds the one of the force the second integrates, and is taken with it.
    assert first.held == pytest.approx(second.ofTheForceItIntegrates, rel=1e-14)
    assert first.timeIncrement == pytest.approx(first.courantNumber * first.held, rel=1e-14)
