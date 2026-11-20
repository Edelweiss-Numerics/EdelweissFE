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
"""A failed simulation is a failure for everything that runs it: the ``edelweissfe`` command exits non-zero and writes
no ``--output``, and ``run_tests_edelweissfe`` counts the deck as FAILED and, with ``--create``, writes no reference.

The deck fails deterministically: with ``maxIter=1``, the load increment, which Newton's method solves in its first
iteration but can show to be converged only in a second one, fails, and it cannot be cut back (``minInc=maxInc``).
"""

import numpy as np
import pytest

from edelweissfe._cli import _edelweissfe, _run_tests_edelweissfe

FAILING_DECK = """
*material, name=linearelastic, id=le, provider=edelweiss
1000.0, 0.3
*section, name=section1, thickness=1.0, material=le, type=plane
all
*job, name=job, domain=2d
*solver, solver=NIST, name=theSolver
*fieldOutput
>>perNode, elSet=all, field=displacement, result=U, name=displacement
*modelGenerator, generator=planeRectQuad, name=gen
x0=0, l=1
y0=0, h=1
elType=CPE4
elProvider=edelweiss
nX=1
nY=1
*step, solver=theSolver
startInc=1, maxInc=1, minInc=1, maxNumInc=10, maxIter=1, stepLength=1
>>dirichlet, name=bottom, nSet=gen_bottom, field=displacement, 1=0, 2=0
>>nodeforces, name=top, nSet=gen_top, field=displacement, 2=10
"""


@pytest.fixture
def failingDeckDirectory(tmp_path, monkeypatch):
    """A test directory holding the failing deck as ``FailingDeck/test.inp``; the scripts change into it."""
    monkeypatch.chdir(tmp_path)
    deckDirectory = tmp_path / "decks" / "FailingDeck"
    deckDirectory.mkdir(parents=True)
    (deckDirectory / "test.inp").write_text(FAILING_DECK)
    return deckDirectory


def test_the_command_exits_non_zero_and_writes_no_output(failingDeckDirectory, runScript):
    output = failingDeckDirectory / "U.out"

    exitCode = runScript(
        _edelweissfe.main,
        str(failingDeckDirectory / "test.inp"),
        "--quiet",
        "--noplot",
        "--output",
        str(output),
    )

    assert exitCode != 0
    assert not output.exists()


def test_create_writes_no_reference(failingDeckDirectory, runScript):
    exitCode = runScript(_run_tests_edelweissfe.main, str(failingDeckDirectory.parent), "--create")

    assert exitCode != 0
    assert not (failingDeckDirectory / "U.ref").exists()


def test_a_failed_run_does_not_pass_against_the_state_it_failed_in(failingDeckDirectory, runScript):
    """The state of the failed run, the initial one, as the reference: still FAILED."""
    np.savetxt(failingDeckDirectory / "U.ref", np.zeros(8))

    exitCode = runScript(_run_tests_edelweissfe.main, str(failingDeckDirectory.parent))

    assert exitCode != 0
