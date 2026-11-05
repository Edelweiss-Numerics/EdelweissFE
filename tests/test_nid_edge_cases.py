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
"""Edge cases of the Newmark-beta implicit dynamic solver (``NID``) not covered by
``test_nid_newmark.py``'s closed-form checks: non-default Newmark parameters and the
guard against a dynamic degree of freedom that received no mass.

Reuses the SDOF deck from ``test_nid_newmark`` (one Marmot ``C3D8``/``LinearElastic`` bar).
"""

import functools
import re
from pathlib import Path

import numpy as np
import pytest

# tests/ carries no __init__.py, so pytest's default (prepend) import mode puts this directory on
# sys.path and sibling test modules import by their bare module name -- same convention as
# test_nid_restart.py.
from test_model_equation_system import _solve, _strip
from test_nid_newmark import (
    BETA,
    FORCE,
    GAMMA,
    MASS,
    RHO,
    STIFFNESS,
    U_STATIC,
    E,
    _deck,
    _run,
    _tipHistories,
)

from edelweissfe.config.configurator import loadConfiguration
from edelweissfe.drivers.inputfiledrivensimulation import finiteElementSimulation
from edelweissfe.journal.journal import Journal
from edelweissfe.models.equationsystemlayout import EquationSystemLayout
from edelweissfe.models.femodel import FEModel
from edelweissfe.solvers.nonlinearimplicitdynamic import NonlinearImplicitDynamic
from edelweissfe.solvers.nonlinearimplicitstatic import NIST
from edelweissfe.utils.inputfileparser import parseInputFile

#: A deck of the explicit static solver NEST.
_NEST_DECK = Path(__file__).resolve().parent.parent / "testfiles" / "marmot" / "NEST" / "test.inp"


def test_backward_euler_damps_energy_monotonically(tmp_path):
    """newmarkGamma=1.0 is the pairing this solver's own production use (the hef70 pry-out run on
    rabbit) picks for maximum numerical damping -- the Newmark family's backward-Euler limit. It
    must actually dissipate energy, not merely fail to conserve it by accident."""

    dt = 0.02
    deck = (
        _deck(dt, maxNumInc=30)
        .replace(f"newmarkGamma={GAMMA}", "newmarkGamma=1.0")
        .replace(f"newmarkBeta={BETA}", "newmarkBeta=1.0")
    )
    _, fieldOutputController = _run(tmp_path, "nid_backward_euler", deck)
    _, u, v, _ = _tipHistories(fieldOutputController)

    energy = 0.5 * MASS * v**2 + 0.5 * STIFFNESS * u**2 - FORCE * u
    energyScale = FORCE * U_STATIC

    # zero at rest at t=0 (same reference as the undamped, exactly-conserved case), and never
    # increasing by more than a tiny fraction of one increment's own dissipation -- backward Euler
    # is globally dissipative, not pointwise monotonic in this exact energy functional, so a little
    # slack (four orders below the ~3.5e-4 * energyScale per-increment decrement measured here) is
    # the honest bound, not machine round-off.
    assert np.max(np.diff(energy)) < 1e-4 * energyScale
    # and it must actually damp something, not merely stay flat at zero
    assert energy[-1] < -1e-2 * energyScale, "backward Euler did not visibly dissipate energy"


def test_warns_but_runs_outside_the_stable_region(tmp_path):
    """beta < (gamma + 1/2)^2 / 4 is outside the unconditionally stable region Newmark's average-
    acceleration rule normally guarantees. The solver must warn, not refuse -- the time increment
    is then the user's own responsibility, same as an explicit solver's CFL limit."""

    dt = 0.02
    deck = _deck(dt, maxNumInc=5).replace(f"newmarkBeta={BETA}", "newmarkBeta=0.1")
    # beta=0.1 < 0.25*(0.5+0.5)^2 = 0.25 -- outside the stable region at gamma=1/2 unchanged.
    _, fieldOutputController = _run(tmp_path, "nid_unstable_region", deck)
    t, u, _, _ = _tipHistories(fieldOutputController)
    assert len(t) == 5
    assert np.all(np.isfinite(u))


def test_refuses_a_dynamic_dof_with_no_mass(tmp_path):
    """A zero density is a valid, finite material property -- unlike a MISSING one, which Marmot's
    own material raises on -- so this solver carries its own guard: a mass-carrying field left
    with no mass anywhere would otherwise integrate as though it had inertia, silently."""

    deck = _deck(0.02, maxNumInc=1).replace(f"{E!r}, 0.0, {RHO!r}", f"{E!r}, 0.0, 0.0")
    with pytest.raises(ValueError, match="received no mass"):
        _run(tmp_path, "nid_massless", deck)


def test_refuses_an_element_without_a_consistent_mass_with_a_clear_message(tmp_path):
    """The pure-Python element library implements only the lumped mass the explicit solver needs.
    Refused at the first increment, naming the element and what to use instead -- not with the bare
    ``NotImplementedError`` of the element base class from deep inside the assembly."""

    deck = (
        _deck(0.02, maxNumInc=1)
        .replace("*material, name=LinearElastic, id=bar", "*material, name=linearelastic, id=bar, provider=edelweiss")
        .replace("elType=C3D8", "elType=C3D8\nelProvider=edelweiss")
    )
    with pytest.raises(NotImplementedError, match="needs a consistent mass matrix.*NED"):
        _run(tmp_path, "nid_python_element", deck)


def test_a_reduced_integration_element_starts_under_load(tmp_path):
    """The initial acceleration solves the consistent mass on its own, M a0 = R. Integrated with a
    reduced element's own rule, that mass is singular (a C3D20R's, from 8 points, has rank 24 of
    60), and a load present at the step's start then produced an arbitrary a0 of order 1e14 and a
    failed increment. Marmot integrates the consistent mass with the full rule of the shape since;
    the reduced element must now behave like the fully integrated one of the same mesh, whose
    stiffness it does not share but whose mass it does."""

    def maxima(elType):
        deck = _deck(0.02, maxNumInc=30).replace("elType=C3D8", f"elType={elType}").replace("nX=1", "nX=3")
        model, _ = _run(tmp_path, f"nid_loaded_start_{elType}", deck)
        field = model.nodeFields["displacement"]
        return model.time, [float(np.max(np.abs(field[entry]))) for entry in "UVA"]

    timeReduced, (uReduced, vReduced, aReduced) = maxima("C3D20R")
    timeFull, (uFull, _, aFull) = maxima("C3D20")

    assert timeReduced == timeFull > 0.0, "the reduced-integration run did not reach the end"
    assert np.isfinite([uReduced, vReduced, aReduced]).all()
    # same mass, softer stiffness: comparable response, nothing of the 1e14 of a singular a0
    assert aReduced < 10.0 * aFull, (aReduced, aFull)
    assert uReduced < 10.0 * uFull, (uReduced, uFull)


def _corruptTheAssembledMass(monkeypatch, value: float):
    """Replace the first nonzero entry of the assembled mass matrix (still in VIJ form) by
    ``value`` -- the elements' own output is out of reach of a test, the Marmot elements being
    compiled. The discarded-inertia warning is the hook: it is handed the mass right before the
    CSR conversion and the checks."""

    from edelweissfe.solvers.nonlinearimplicitdynamic import NonlinearImplicitDynamic

    trueWarn = NonlinearImplicitDynamic._warnAboutDiscardedInertia

    def corruptingWarn(self, Mvij, dampingVIJIndices, couplesDynamicOnly):
        trueWarn(self, Mvij, dampingVIJIndices, couplesDynamicOnly)
        Mvij[np.flatnonzero(np.asarray(Mvij))[0]] = value

    monkeypatch.setattr(NonlinearImplicitDynamic, "_warnAboutDiscardedInertia", corruptingWarn)


def _reportADamping(monkeypatch, value: float):
    """Let every element report ``value`` as the damping of its first (dynamic) dof -- the
    elements' own output is out of reach of a test, the Marmot elements being compiled."""

    from edelweissfe.solvers.nonlinearimplicitdynamic import NonlinearImplicitDynamic

    def lumpedDampingOfElement(self, el):
        start = self._K.vijSliceOfEntity[el].start
        return np.array([start], dtype=np.int64), np.array([value])

    monkeypatch.setattr(NonlinearImplicitDynamic, "_lumpedDampingOfElement", lumpedDampingOfElement)


def test_refuses_a_non_finite_mass(tmp_path, monkeypatch):
    _corruptTheAssembledMass(monkeypatch, np.nan)
    with pytest.raises(ValueError, match="non-finite entries"):
        _run(tmp_path, "nid_non_finite_mass", _deck(0.02, maxNumInc=1))


def test_refuses_a_non_finite_damping(tmp_path, monkeypatch):
    _reportADamping(monkeypatch, np.inf)
    with pytest.raises(ValueError, match="non-finite entries"):
        _run(tmp_path, "nid_non_finite_damping", _deck(0.02, maxNumInc=1))


def test_refuses_a_negative_damping(tmp_path, monkeypatch):
    _reportADamping(monkeypatch, -1.0)
    with pytest.raises(ValueError, match="negative entries"):
        _run(tmp_path, "nid_negative_damping", _deck(0.02, maxNumInc=1))


def test_reports_the_assembled_total_mass(tmp_path, monkeypatch):
    from edelweissfe.solvers.nonlinearimplicitdynamic import NonlinearImplicitDynamic

    reported = []
    trueTotalMassByField = NonlinearImplicitDynamic._totalMassByField

    def recordingTotalMassByField(*args):
        totals = trueTotalMassByField(*args)
        reported.append(totals)
        return totals

    monkeypatch.setattr(NonlinearImplicitDynamic, "_totalMassByField", staticmethod(recordingTotalMassByField))
    _run(tmp_path, "nid_mass_report", _deck(0.02, maxNumInc=1))

    assert reported and set(reported[0]) == {"displacement"}
    # the whole unit bar: rho * volume
    np.testing.assert_allclose(reported[0]["displacement"], RHO, rtol=1e-12)


def test_the_quasi_static_and_the_dynamic_solver_give_the_entities_their_newmark_parameters():
    """The solver is the one source of the Newmark parameters of the entities that integrate their own inertia (the
    particles of a meshfree model): NIST passes zero, NID its own with meshfreeInertia=particles and zero with
    meshfreeInertia=nodes, where it integrates the particles' inertia at the nodes instead."""

    journal = Journal(verbose=False)
    jobInfo = loadConfiguration(dict())
    assert NIST(jobInfo, journal).newmarkParametersOfEntities() == (0.0, 0.0)

    inParticles = NonlinearImplicitDynamic(jobInfo, journal, newmarkBeta=0.3, newmarkGamma=0.6)
    assert inParticles.newmarkParametersOfEntities() == (0.3, 0.6)

    atNodes = NonlinearImplicitDynamic(jobInfo, journal, newmarkBeta=0.3, newmarkGamma=0.6, meshfreeInertia="nodes")
    assert atNodes.newmarkParametersOfEntities() == (0.0, 0.0)

    with pytest.raises(ValueError, match=re.escape("meshfreeInertia=particles|nodes")):
        NonlinearImplicitDynamic(jobInfo, journal, meshfreeInertia="grid")


class _NewmarkParametersGiven(Exception):
    """Stops the run once the solver has given the entities their Newmark parameters."""


def test_the_explicit_static_solver_gives_the_entities_no_inertia(tmp_path, monkeypatch):
    """NEST is quasi-static: at the start of its step, it gives the entities that integrate their own inertia zero
    Newmark parameters, as NIST does, so that particles that a dynamic step before gave its parameters carry no inertia
    under it."""

    given = []

    def recordAndStop(self, beta, gamma):
        given.append((beta, gamma))
        raise _NewmarkParametersGiven()

    monkeypatch.setattr(FEModel, "setNewmarkParametersOfEntities", recordAndStop)
    monkeypatch.chdir(tmp_path)
    with pytest.raises(_NewmarkParametersGiven):
        finiteElementSimulation(parseInputFile(str(_NEST_DECK)), verbose=False, suppressPlots=True)

    assert given == [(0.0, 0.0)]


class _LayoutWithoutNodalInertia(EquationSystemLayout):
    """The layout of a model none of whose nodes carries a mass the solver integrates: as in a meshfree model whose
    particles carry the inertia and integrate it themselves (``meshfreeInertia=particles``). It reports the motion of
    the particles where the model says."""

    def dofsWithNodalInertia(self, fieldNames):
        return np.zeros(0, dtype=int)

    def carriersOfTheMotionOfParticles(self):
        return set(self.model.motionOfParticlesHeldIn)


class _ModelWithoutNodalInertia(FEModel):
    """A finite element model that answers with :class:`_LayoutWithoutNodalInertia`, and holds the motion of
    particles (which it does not model) in the given places."""

    def __init__(self, dimension, motionOfParticlesHeldIn=()):
        super().__init__(dimension)
        self.motionOfParticlesHeldIn = motionOfParticlesHeldIn

    def buildEquationSystem(self):
        layout = super().buildEquationSystem()
        return _LayoutWithoutNodalInertia(self, layout.dofManager, layout.elements)


def test_a_model_without_nodal_inertia_gets_no_mass_assembled():
    """With no degree of freedom to integrate in time, NID assembles no mass and adds no Newmark terms: the elements of
    the strip, from the pure-Python library, have no consistent mass, which an assembly would ask them for. What is
    left is the quasi-static answer."""

    journal = Journal(verbose=False)
    quasiStatic = _solve(_strip(FEModel(2), 2), NIST, journal)
    withoutNodalInertia = _solve(_strip(_ModelWithoutNodalInertia(2), 2), NonlinearImplicitDynamic, journal)

    expected, actual = quasiStatic.nodeFields["displacement"], withoutNodalInertia.nodeFields["displacement"]
    assert np.max(np.abs(expected["U"])) > 0.0
    for entry in ("U", "P"):
        np.testing.assert_allclose(
            actual[entry], expected[entry], rtol=1e-10, atol=1e-14 * np.max(np.abs(expected[entry]))
        )
    for entry in ("V", "A"):
        np.testing.assert_array_equal(actual[entry], 0.0)


@pytest.mark.parametrize(
    "meshfreeInertia, motionOfParticlesHeldIn, refused",
    [
        ("nodes", ("particles",), True),
        ("particles", ("nodes",), True),
        ("particles", ("particles",), False),
        ("nodes", ("nodes",), False),
        ("particles", ("particles", "nodes"), False),
        ("nodes", (), False),
    ],
)
def test_a_step_refuses_to_start_where_the_motion_of_the_particles_is_not_taken_from(
    meshfreeInertia, motionOfParticlesHeldIn, refused
):
    """NID takes the motion of meshfree particles from their state (``particles``) or from the nodes of their kernel
    functions (``nodes``), and nothing transfers it between the two: a step that would take it from where the model
    does not hold it refuses to start, rather than starting the particles from rest. Held in both places (after the
    explicit meshfree solver), or nowhere (at rest), the step starts."""

    journal = Journal(verbose=False)
    model = _strip(_ModelWithoutNodalInertia(2, motionOfParticlesHeldIn), 2)
    solverClass = functools.partial(NonlinearImplicitDynamic, meshfreeInertia=meshfreeInertia)

    if refused:
        with pytest.raises(ValueError, match="meshfreeInertia={:} takes it".format(meshfreeInertia)):
            _solve(model, solverClass, journal)
    else:
        _solve(model, solverClass, journal)
        assert np.max(np.abs(model.nodeFields["displacement"]["U"])) > 0.0
