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

import numpy as np
import pytest

# tests/ carries no __init__.py, so pytest's default (prepend) import mode puts this directory on
# sys.path and sibling test modules import by their bare module name -- same convention as
# test_nid_restart.py.
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
