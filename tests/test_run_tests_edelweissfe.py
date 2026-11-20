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
"""The deck runner ``run_tests_edelweissfe``: a deck with a missing requirement it declares is skipped, any other
exception fails it, its solution is compared relative to the magnitude of the reference, ``--create`` refuses a
reference that is zero, and ``--expected-skips`` fails the run on an unexpected skip and on a stale list.
"""

import numpy as np
import pytest

from edelweissfe._cli import _run_tests_edelweissfe

#: One linear elastic element pulled at its top by a node force.
_DECK = """{requirements}
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
startInc=1, maxInc=1, minInc=1, maxNumInc=10, maxIter=25, stepLength=1
>>dirichlet, name=bottom, nSet=gen_bottom, field=displacement, 1=0, 2=0
>>nodeforces, name=top, nSet=gen_top, field=displacement, 2={force}
"""


@pytest.fixture
def decks(tmp_path, monkeypatch):
    """A test directory ``decks``; the runner changes into the directories of its decks."""
    monkeypatch.chdir(tmp_path)
    directory = tmp_path / "decks"
    directory.mkdir()
    return directory


def writeDeck(decks, name: str, force: float = 10.0, requirements: str = ""):
    (decks / name).mkdir()
    (decks / name / "test.inp").write_text(_DECK.format(requirements=requirements, force=force))
    return decks / name


def test_a_deck_with_a_missing_requirement_is_skipped_with_the_reason(decks, runScript, capsys):
    deck = writeDeck(decks, "NeedsAModule", requirements="** requires: python-module edelweissfe_no_such_module")

    assert runScript(_run_tests_edelweissfe.main, str(decks), "--create") == 0
    assert not (deck / "U.ref").exists()
    # rich wraps long lines
    assert "Python module edelweissfe_no_such_module is not installed" in " ".join(capsys.readouterr().out.split())


def test_an_unknown_kind_of_requirement_fails_the_deck(decks, runScript):
    writeDeck(decks, "UnknownRequirement", requirements="** requires: something-else name")

    assert runScript(_run_tests_edelweissfe.main, str(decks), "--create") == 1


@pytest.mark.parametrize("exception", [NotImplementedError, ModuleNotFoundError])
def test_any_exception_of_the_simulation_fails_the_deck(decks, runScript, monkeypatch, exception):
    """E.g. a solver that lost a capability, or a core extension that failed to build."""
    deck = writeDeck(decks, "Raises")
    np.savetxt(deck / "U.ref", np.ones(8))

    def raising(*args, **kwargs):
        raise exception("raised by the simulation")

    monkeypatch.setattr(_run_tests_edelweissfe, "finiteElementSimulation", raising)

    assert runScript(_run_tests_edelweissfe.main, str(decks)) == 1


@pytest.mark.parametrize(
    "force, relativeDeviation, exitCode",
    [(0.01, 1e-5, 1), (1e6, 1e-7, 0)],
    ids=["small solution, deviating too much", "large solution, deviating within the tolerance"],
)
def test_the_solution_is_compared_relative_to_the_magnitude_of_the_reference(
    decks, runScript, force, relativeDeviation, exitCode
):
    deck = writeDeck(decks, "Pulled", force=force)
    assert runScript(_run_tests_edelweissfe.main, str(decks), "--create") == 0
    reference = np.loadtxt(deck / "U.ref")
    np.savetxt(deck / "U.ref", reference * (1.0 + relativeDeviation))

    assert runScript(_run_tests_edelweissfe.main, str(decks)) == exitCode


def test_create_refuses_a_reference_that_is_zero(decks, runScript):
    deck = writeDeck(decks, "Unloaded", force=0.0)

    assert runScript(_run_tests_edelweissfe.main, str(decks), "--create") == 1
    assert not (deck / "U.ref").exists()


@pytest.fixture
def skippedAndRunDecks(decks, runScript):
    """A deck that is skipped, and one that runs, with its reference."""
    writeDeck(decks, "Skipped", requirements="** requires: python-module edelweissfe_no_such_module")
    writeDeck(decks, "Runs")
    assert runScript(_run_tests_edelweissfe.main, str(decks), "--create", "--tests", "Runs") == 0
    return decks


@pytest.mark.parametrize(
    "listed, exitCode",
    [
        (["decks/Skipped"], 0),
        ([], 1),
        (["decks/Skipped", "decks/Runs"], 1),
        (["decks/Skipped", "decks/NoSuchDeck"], 1),
        (["decks/Skipped", "otherDecks/Runs"], 0),
    ],
    ids=["as listed", "unlisted skip", "listed deck runs", "listed deck missing", "list of another directory"],
)
def test_expected_skips(skippedAndRunDecks, runScript, tmp_path, listed, exitCode):
    expectedSkips = tmp_path / "expected-skips.txt"
    expectedSkips.write_text("# decks that may be skipped\n" + "".join(entry + "  # a reason\n" for entry in listed))

    assert (
        runScript(_run_tests_edelweissfe.main, str(skippedAndRunDecks), "--expected-skips", str(expectedSkips))
        == exitCode
    )
