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
"""The explicit solver (``NED``) after a quasi-static step: ``testfiles/edelweiss-only/NEDAfterNISTGravity``, a column
loaded by its own weight with ``NIST``, then continued by ``NED`` under the same weight and support.

* The explicit step starts at rest in equilibrium and stays there. Its net force is solver state, zero at the start of
  a step, which is the net force of a state in equilibrium. Read back from the model's ``P``, as it once was, it would
  be NIST's internal force, and the column would swing about its static deflection by about a tenth of it.
* The published force ``P`` means the same in both steps (see the documentation page *Node field results*): the
  internal force at the base, which holds the column, and the internal plus the inertial force elsewhere -- in
  equilibrium, the weight there. The net force has the opposite sign at the base, and is zero elsewhere.
"""

from pathlib import Path

import numpy as np
import pytest

from edelweissfe.drivers.inputfiledrivensimulation import finiteElementSimulation
from edelweissfe.utils.inputfileparser import parseInputFile

_DECK = Path(__file__).resolve().parent.parent / "testfiles" / "edelweiss-only" / "NEDAfterNISTGravity" / "test.inp"

#: the end of the quasi-static step
_END_OF_PRELOAD = 1.0


@pytest.fixture(scope="module")
def histories(tmp_path_factory):
    """Per output: the times, and the results at the end of the quasi-static step and at every later output."""

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.chdir(tmp_path_factory.mktemp("nedafternist"))
    try:
        _, fieldOutputController = finiteElementSimulation(
            parseInputFile(str(_DECK)), verbose=False, suppressPlots=True
        )
    finally:
        monkeypatch.undo()

    result = {}
    for name in ("U", "P"):
        output = fieldOutputController.fieldOutputs[name]
        t = output.getTimeHistory()
        values = output.getResultHistory()
        endOfPreload = np.flatnonzero(np.isclose(t, _END_OF_PRELOAD))[0]
        result[name] = values[endOfPreload], values[endOfPreload + 1 :], t[endOfPreload + 1 :]
    return result


def test_the_explicit_step_starts_at_rest_in_equilibrium(histories):
    static, explicit, t = histories["U"]

    assert t[-1] > _END_OF_PRELOAD + 0.15, "the explicit step must cover about a period of the column"
    deflection = np.max(np.abs(static))
    assert deflection > 0.0
    for U in explicit:
        np.testing.assert_allclose(U, static, rtol=0.0, atol=1e-8 * deflection)


def test_the_published_force_means_the_same_after_the_switch(histories):
    quasiStatic, explicit, _ = histories["P"]

    # the weight loads the column, so the force is not zero
    assert np.max(np.abs(quasiStatic)) > 0.0
    for P in explicit:
        np.testing.assert_allclose(P, quasiStatic, rtol=0.0, atol=1e-8 * np.max(np.abs(quasiStatic)))
