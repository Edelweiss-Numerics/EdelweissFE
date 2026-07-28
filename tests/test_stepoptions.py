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
"""
P3 (see PLAN_INPUT_SYSTEM.md): guards for the *single* remaining step-options mechanism.

``getOptionsOfCategory`` recovers "what did the user actually write" by stripping ``None``-valued
entries, which is only correct while **every** option on the shared ``options`` step keyword carries
a runtime default of ``None``. That used to be a convention, backed up by a second, redundant
mechanism (``StepAction.explicitlySetOptions``) that was written on every update and read by nothing.
With the redundant mechanism deleted, the invariant is load-bearing on its own, so it is asserted
here rather than left to reviewers: one test pins the ``None`` defaults, one pins that
``registerOptionsArg`` is the only registrar (a future call site adding an arg directly would bypass
the ``None``), and one pins that the documentation-only default does not leak into the runtime one.

P3(c) moved the strip itself from ``getOptionsOfCategory`` into the ``options`` step action's L4
seam, so that the container stores what a consumer receives rather than a parser dict to be
re-stripped on every query. Two further tests cover that seam: one that it keeps only what the user
set, and one that a later step's block *replaces* its category's options instead of merging into
them -- the one behaviour that stopped being implied by the mechanism and had to become explicit.
"""

import subprocess
import sys
from pathlib import Path

import pytest

from edelweissfe.stepactions.options import StepAction as OptionsStepAction
from edelweissfe.stepactions.options import getOptionsOfCategory

_REPO_ROOT = Path(__file__).resolve().parents[1]


def _optionsKeywordArgs() -> list:
    """Every optional arg registered on the shared ``options`` keyword, across all step types.

    Rendered in a subprocess for the same reason as the golden grammar surface: the full input
    language is only populated once the parser has imported every module, and doing that in-process
    would depend on what other test modules have already imported.
    """
    script = (
        "from edelweissfe.utils.inputlanguage import InputLanguage\n"
        "language = InputLanguage()\n"
        "language.ensureParserLoaded()\n"
        "for module in language['step'].modules:\n"
        "    for arg in module.getKeyword('options').optionalArgs:\n"
        "        print(f'{module.name}|{arg.name}|{arg.default!r}|{arg.documentedDefault!r}')\n"
    )
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, cwd=_REPO_ROOT, check=False)
    if result.returncode != 0:
        pytest.fail(f"Could not render the 'options' keyword:\n{result.stderr}")
    return [line.split("|", 3) for line in result.stdout.splitlines() if line]


def test_every_option_on_the_shared_keyword_defaults_to_none():
    """The invariant getOptionsOfCategory's strip-``None``s rests on."""
    args = _optionsKeywordArgs()
    assert args, "no options were registered at all -- the render must be broken"

    offenders = [(module, name, default) for module, name, default, _ in args if default != "None"]
    assert not offenders, (
        "options on the shared 'options' step keyword must default to None, otherwise "
        "getOptionsOfCategory cannot tell a user's entry from a foreign module's default and would "
        "silently leak these into every category: " + repr(offenders)
    )


def test_registerOptionsArg_is_the_only_registrar_on_the_shared_keyword():
    """What makes the ``None`` default above structural rather than a convention.

    A source-level check, deliberately: the failure mode is a *new* call site added years from now,
    which no amount of exercising the current call sites would catch.
    """
    hits = subprocess.run(
        ["grep", "-rn", "--include=*.py", 'getKeyword("options")', "edelweissfe"],
        capture_output=True,
        text=True,
        cwd=_REPO_ROOT,
        check=False,
    ).stdout.splitlines()

    assert len(hits) == 1 and hits[0].startswith("edelweissfe/stepactions/options.py:"), (
        "the shared 'options' keyword must only ever be extended through registerOptionsArg, which "
        "forces the runtime default to None; found other call sites: " + repr(hits)
    )


def test_documented_default_is_rendered_but_does_not_change_the_runtime_default():
    """The P3 docs-regression fix: docs show the real default, the parser still sees ``None``."""
    byName = {name: (default, documented) for _, name, default, documented in _optionsKeywordArgs()}

    # a solver option whose effective default lives in NIST.SolverSpecificOptions
    assert byName["defaultMaxIter"] == ("None", "10")
    # ... and one whose value is a string, to catch a repr/format slip
    assert byName["linsolver"] == ("None", "'pardiso'")


#: A parsed ``>>options, category=NISTSolver, defaultMaxIter=25`` block, as the parser hands it over:
#: every arg any module registered on the shared keyword is present, ``None`` where unset, plus the
#: parser's own bookkeeping keys.
_PARSED_OPTIONS_BLOCK = {
    "category": "NISTSolver",
    "defaultMaxIter": "25",
    "linsolver": None,
    "inputFile": "/some/path/test.inp",
    "datalines": [],
    "explicitlySetArgs": {"defaultmaxiter"},
}


def _optionsAction(name, definition):
    """Build a real ``options`` step action from a parsed block, through the L4 seam."""

    return OptionsStepAction.fromStepActionDefinition(name, definition, None, None, None, None)


def test_parsed_options_block_keeps_only_what_the_user_set():
    """Consumers get the user's entries only -- no ``None``s, no parser internals, no category tag.

    The stripping is asserted on the L4 seam rather than on ``getOptionsOfCategory``, because that is
    where it moved in P3(c): the container now stores what a consumer receives. Note that these
    actions are built with no ``jobInfo``/``model``/``fieldOutputController``/``journal`` at all,
    which is the property the port bought -- the stand-in class this test used to need existed only
    because the real one could not be constructed without a parser-shaped dict.
    """

    options = getOptionsOfCategory(
        {"options": {"NISTSolver": _optionsAction("NISTSolver", _PARSED_OPTIONS_BLOCK)}}, "nistsolver"
    )

    # a CaseInsensitiveDict stores its keys casefolded, hence the lookup rather than a dict compare
    assert len(options) == 1
    assert options["defaultMaxIter"] == "25"


def test_a_later_step_replaces_the_options_of_its_category_rather_than_merging():
    """An option set in an earlier step and omitted in a later one must not leak into it.

    Pre-port this fell out of the mechanism: ``self.options.update(parsedBlock)`` was handed a
    mapping carrying *every* declared arg, so an omitted option arrived as ``None`` and was dropped
    by the read-time strip. Once the strip moves to the seam the mapping is sparse, and a merge would
    silently carry the earlier value over -- which no simulation test would catch, because the only
    input that re-declares one category with differing option sets
    (``testfiles/marmot/IndirectDisplacementControl2``) skips for a missing Marmot material.
    """

    action = _optionsAction("NISTSolver", _PARSED_OPTIONS_BLOCK)
    assert getOptionsOfCategory({"options": {"NISTSolver": action}}, "NISTSolver")["defaultMaxIter"] == "25"

    laterStep = dict(_PARSED_OPTIONS_BLOCK, defaultMaxIter=None, linsolver="klu")
    action.updateStepActionFromDefinition(laterStep, None, None, None, None)

    options = getOptionsOfCategory({"options": {"NISTSolver": action}}, "NISTSolver")
    assert options["linsolver"] == "klu"
    assert "defaultMaxIter" not in options


def test_getOptionsOfCategory_rejects_two_blocks_for_one_category():
    actions = {
        "options": {
            "a": _optionsAction("a", {"category": "NISTSolver"}),
            "b": _optionsAction("b", {"category": "nistsolver"}),
        }
    }

    with pytest.raises(ValueError, match="Multiple 'options' step action definitions"):
        getOptionsOfCategory(actions, "NISTSolver")
