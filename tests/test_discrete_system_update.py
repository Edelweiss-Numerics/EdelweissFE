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
"""One build path for every solver that derives from ``NIST``: what the solver does for each
:class:`~edelweissfe.models.change.Change` (``NonlinearSolverBase.updateDiscreteSystem``), what a vector keeps when its
locations change (``DofVector.onLocations``), what new locations refuse (a change of the scalar variables), and that a
contact deck gives the same results through new locations as through a discrete system built anew every time."""

from types import SimpleNamespace

import numpy as np
import pytest
from test_arclength_load_parameter import arcLengthPenaltyConstraintDeck
from test_dof_numbering_and_locations import (
    _buildTwoElementModel,
    _FakeConstraint,
    _locations,
    _numbering,
)
from test_nid_contact import _DECK as _NID_CONTACT_DECK

from edelweissfe.drivers.inputfiledrivensimulation import finiteElementSimulation
from edelweissfe.linsolve.composition.solver import ComposedLinearSolver
from edelweissfe.models.change import Change
from edelweissfe.numerics.doflocations import DofLocations
from edelweissfe.numerics.dofnumbering import DofNumbering
from edelweissfe.numerics.dofvector import DofVector
from edelweissfe.solvers.base.nonlinearsolverbase import NonlinearSolverBase
from edelweissfe.solvers.nonlinearimplicitstatic import NIST
from edelweissfe.utils.inputfileparser import parseInputFile
from edelweissfe.variables.scalarvariable import ScalarVariable


class _System:
    """A discrete system that says how it came about."""

    def __init__(self, record: list, origin: str):
        self.record = record
        self.origin = origin

    def withNewLocations(self):
        return _System(self.record, "new locations of " + self.origin)


class _Solver:
    """The part of a solver the update of the discrete system touches: it records what is called, in order."""

    def __init__(self, system=None):
        self.record = []
        self.system = system

    def forgetDiscreteSystem(self):
        self.record.append("forget")
        self.system = None

    def onNewNumbering(self):
        self.record.append("onNewNumbering " + self.system.origin)

    def onNewLocations(self):
        self.record.append("onNewLocations " + self.system.origin)


def _update(solver, change):
    model = SimpleNamespace(buildDiscreteSystem=lambda: _System(solver.record, "built"))
    NonlinearSolverBase.updateDiscreteSystem(solver, model, change)

    return solver.record


@pytest.mark.parametrize("change", list(Change))
def test_the_first_discrete_system_is_built_whatever_the_change(change):
    assert _update(_Solver(), change) == ["forget", "onNewNumbering built", "onNewLocations built"]


@pytest.mark.parametrize("change", [Change.NUMBERING, Change.MESH])
def test_a_change_of_the_numbering_or_the_mesh_builds_a_new_discrete_system(change):
    solver = _Solver(_System([], "old"))

    assert _update(solver, change) == ["forget", "onNewNumbering built", "onNewLocations built"]


def test_a_change_of_the_locations_makes_new_locations_on_the_same_numbering():
    solver = _Solver(_System([], "old"))
    solver.system.record = solver.record

    assert _update(solver, Change.LOCATIONS) == ["onNewLocations new locations of old"]


def test_no_change_leaves_the_discrete_system_as_it_is():
    system = _System([], "old")
    solver = _Solver(system)

    assert _update(solver, Change.NONE) == []
    assert solver.system is system


def test_a_vector_wrapped_on_new_locations_keeps_its_values_and_sees_the_new_locations():
    model, n = _buildTwoElementModel()
    before = _FakeConstraint([n[0], n[5]], ["displacement"])
    after = _FakeConstraint([n[2], n[3]], ["displacement"])
    locations = _locations(model, [before])
    newLocations = locations.withNewLocationsOf([after])

    vector = locations.constructDofVector()
    vector[:] = np.arange(vector.size)
    rewrapped = vector.onLocations(newLocations)

    assert isinstance(rewrapped, DofVector) and rewrapped.locations is newLocations and vector.locations is locations
    np.testing.assert_array_equal(rewrapped, np.arange(vector.size))
    assert np.shares_memory(rewrapped, vector), "a view, not a copy"
    np.testing.assert_array_equal(rewrapped[after], newLocations[after] * 1.0)
    np.testing.assert_array_equal(vector[before], locations[before] * 1.0)
    # the old locations do not know the new constraint, and the new ones do not know the old
    assert after not in vector.locations and before not in rewrapped.locations
    assert rewrapped[model.elements[1]].tolist() == vector[model.elements[1]].tolist()
    # the scatter layout follows the locations the vector is wrapped with
    assert rewrapped.createScatterVector().offsetMap.keys() == newLocations.ofEntity.keys()
    rewrapped[0] = -1.0
    assert vector[0] == -1.0, "the same buffer"


def test_new_locations_refuse_a_change_of_the_scalar_variables():
    model, n = _buildTwoElementModel()
    numbered = ScalarVariable("lambda_0")
    other = ScalarVariable("lambda_1")
    numbering = DofNumbering(model.nodeFields.values(), [numbered], model.nodeSets.values())
    constraint = _FakeConstraint([n[0], n[5]], ["displacement"])
    constraint.scalarVariables = [numbered]
    locations = DofLocations(numbering, model.elements.values(), [constraint])

    assert locations.withNewLocationsOf([constraint]).constraints == [constraint]

    gained = _FakeConstraint([n[1]], ["displacement"])
    gained.scalarVariables = [other]
    with pytest.raises(ValueError, match="lambda_1"):
        locations.withNewLocationsOf([constraint, gained])
    with pytest.raises(ValueError, match="needs a new numbering"):
        locations.withNewLocationsOf([])
    assert _numbering(model).nDof == numbering.nDof - 1


_CONTACT_DECK = """
*material, name=linearelastic, id=mat
18000, 0.0, 1.0e-3

*section, name=sec, material=mat, type=solid
lower_all
upper_all

*modelGenerator, generator=boxGen, name=lower
nX=2
nY=2
nZ=1
lX=1
lY=1
lZ=1
elType=C3D8

*modelGenerator, generator=boxGen, name=upper
nX=1
nY=1
nZ=1
x0=0.2
y0=0.2
z0=1.02
lX=0.6
lY=0.6
lZ=0.5
elType=C3D8

*modelGenerator, generator=surfaceElementGenerator, name=gen1
surface=upper_back
name=slaveSurf

*modelGenerator, generator=surfaceElementGenerator, name=gen2
surface=lower_front
name=masterSurf

*constraint, name=contact, type=nodeToDeformableSurfacePenalty
slaveSurface=slaveSurf_facets, masterSurface=masterSurf_facets, penalty=1e6, type=linear,
searchDistance=2.0, sliding=finite

*job, name=nistcontact, domain=3d
*solver, solver=NIST, name=theSolver

*step, solver=theSolver
maxInc=0.05, startInc=0.05, minInc=1e-6, maxNumInc=100, maxIter=25, stepLength=0.1
>>options, name=theSolver, extrapolation=off
>>dirichlet, name=fixLower, nSet=lower_back, field=displacement, 1=0.0, 2=0.0, 3=0.0
>>dirichlet, name=pushUpper, nSet=upper_front, field=displacement, 1=0.4, 2=0.0, 3=-0.05, f(t)='t*t*(3-2*t)'
"""


def _run(tmp_path, name: str, deck: str, monkeypatch, buildAnewEveryTime: bool):
    """Run a deck whose solver derives from NIST; every update of the discrete system asked for is recorded, and
    answered as asked, or (``buildAnewEveryTime``) with a new discrete system whatever the change."""
    changes = []
    updateDiscreteSystem = NonlinearSolverBase.updateDiscreteSystem

    def update(self, model, change):
        changes.append(change)
        updateDiscreteSystem(self, model, max(change, Change.NUMBERING) if buildAnewEveryTime else change)

    monkeypatch.setattr(NIST, "updateDiscreteSystem", update)

    path = tmp_path / f"{name}.inp"
    path.write_text(deck)
    model, _ = finiteElementSimulation(parseInputFile(str(path)), verbose=False, suppressPlots=True)
    field = model.nodeFields["displacement"]
    return changes, [np.array(field[entry]) for entry in ("U", "dU", "V", "A", "P")]


def test_a_contact_deck_gives_the_same_results_through_new_locations_as_through_a_new_discrete_system(
    tmp_path, monkeypatch
):
    changes, results = _run(tmp_path, "relocated", _CONTACT_DECK, monkeypatch, buildAnewEveryTime=False)
    _, builtAnew = _run(tmp_path, "builtAnew", _CONTACT_DECK, monkeypatch, buildAnewEveryTime=True)

    assert Change.LOCATIONS in changes, "the contact never changed its connectivity; the deck exercises nothing"
    for relocated, fresh in zip(results, builtAnew):
        np.testing.assert_array_equal(relocated, fresh)


def test_a_dynamic_contact_deck_keeps_its_velocity_and_acceleration_through_new_locations(tmp_path, monkeypatch):
    """NID keeps its velocity and acceleration, and its mass moved onto the new pattern, instead of reading them from
    the model and assembling the mass anew."""
    changes, results = _run(tmp_path, "relocated", _NID_CONTACT_DECK, monkeypatch, buildAnewEveryTime=False)
    _, builtAnew = _run(tmp_path, "builtAnew", _NID_CONTACT_DECK, monkeypatch, buildAnewEveryTime=True)

    assert Change.LOCATIONS in changes, "the contact never changed its connectivity; the deck exercises nothing"
    assert np.any(results[2]) and np.any(results[3]), "nothing moves in the deck"
    for relocated, fresh in zip(results, builtAnew):
        np.testing.assert_array_equal(relocated, fresh)


@pytest.mark.parametrize("change", [Change.LOCATIONS, Change.NUMBERING])
def test_an_arc_length_run_does_not_depend_on_how_the_discrete_system_follows_a_change(tmp_path, monkeypatch, change):
    """Every increment reports a change of the locations (or of the numbering): the arc length solver reaches the state
    of the run without any change."""
    # the prediction survives the change, so that the runs take the same steps
    deck = arcLengthPenaltyConstraintDeck().replace(
        ">>options, name=theSolver, ", ">>options, name=theSolver, extrapolateAcrossConnectivityChange=True, "
    )
    assert "extrapolateAcrossConnectivityChange" in deck
    _, reference = _run(tmp_path, "unchanged", deck, monkeypatch, buildAnewEveryTime=False)

    monkeypatch.setattr(
        NIST, "updateTopologyAndConnectivity", lambda self, model, step, offerModelModifiers=True: change
    )
    changes, results = _run(tmp_path, "changing", deck, monkeypatch, buildAnewEveryTime=False)

    assert set(changes) >= {change}
    for result, expected in zip(results, reference):
        np.testing.assert_allclose(result, expected, rtol=0.0, atol=1e-10 * max(1.0, np.max(np.abs(expected))))


def test_new_locations_tell_the_linear_solver_of_the_new_pattern_before_it_is_built_and_leave_its_model_alone(
    tmp_path, monkeypatch
):
    events = []
    setModel = ComposedLinearSolver.setModel
    onNewMatrixPattern = ComposedLinearSolver.onNewMatrixPattern
    createSystemMatrixAssembly = NIST.createSystemMatrixAssembly
    updateDiscreteSystem = NonlinearSolverBase.updateDiscreteSystem
    eventsPerChange = []

    def recordedSetModel(self, *arguments):
        events.append("setModel")
        return setModel(self, *arguments)

    def recordedOnNewMatrixPattern(self):
        events.append("onNewMatrixPattern")
        return onNewMatrixPattern(self)

    def recordedCreateSystemMatrixAssembly(self, *arguments):
        events.append("createSystemMatrixAssembly")
        return createSystemMatrixAssembly(self, *arguments)

    def update(self, model, change):
        events.clear()
        hadSystem = self.system is not None
        updateDiscreteSystem(self, model, change)
        eventsPerChange.append((change if hadSystem else None, list(events)))

    monkeypatch.setattr(ComposedLinearSolver, "setModel", recordedSetModel)
    monkeypatch.setattr(ComposedLinearSolver, "onNewMatrixPattern", recordedOnNewMatrixPattern)
    monkeypatch.setattr(NIST, "createSystemMatrixAssembly", recordedCreateSystemMatrixAssembly)
    monkeypatch.setattr(NIST, "updateDiscreteSystem", update)

    path = tmp_path / "events.inp"
    path.write_text(_CONTACT_DECK)
    finiteElementSimulation(parseInputFile(str(path)), verbose=False, suppressPlots=True)

    afterLocations = [recorded for change, recorded in eventsPerChange if change == Change.LOCATIONS]
    assert afterLocations, "the contact never changed its connectivity; the deck exercises nothing"
    for recorded in afterLocations:
        assert recorded == ["onNewMatrixPattern", "createSystemMatrixAssembly"]

    # the first discrete system of a step (no system yet, whatever the change then): a new numbering
    afterNumbering = [recorded for change, recorded in eventsPerChange if recorded[:1] == ["setModel"]]
    assert afterNumbering
    for recorded in afterNumbering:
        assert recorded[-2:] == ["onNewMatrixPattern", "createSystemMatrixAssembly"]
