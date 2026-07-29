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
#  ---------------------------------------------------------------------
"""Guards for the name-based ``>>options`` override mechanism (see
``edelweissfe/stepactions/options.py``'s module docstring for the design).

Three concerns, mirrored by three groups of tests below:

- **Resolution.** ``name`` must resolve to exactly one of ``model.solvers``/``model.outputManagers``,
  with a loud error for "neither" and for "both" (an instance name reused across the two, which would
  otherwise silently pick one).
- **The shared keyword's grammar.** Every solver's and output manager's option names must be
  pre-declared on ``>>options`` (:func:`~edelweissfe.stepactions.options.registerSchemaOptions`), with
  a runtime default of ``None`` -- the invariant :func:`~edelweissfe.stepactions.options._writtenOptions`
  rests on to tell "the user wrote this" apart from "some other module's option sharing this keyword".
- **The override itself.** Only what the user actually wrote is validated against the resolved
  target's own schema and applied, and an override sticks until changed again -- confirmed empirically
  (this module used to claim otherwise) against the pre-existing mechanism this one replaces: a step
  that does not re-declare ``>>options`` for a given target leaves that target's last-set options in
  effect, it does not revert them.
"""

import subprocess
import sys
from pathlib import Path

import pytest

from edelweissfe.journal.journal import Journal
from edelweissfe.solvers.nonlinearimplicitstatic import NIST
from edelweissfe.stepactions.options import StepAction, _resolveTarget, _writtenOptions

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


def test_name_is_the_only_required_arg_on_the_shared_keyword():
    """Every other option is target-specific and therefore optional at the keyword level; only
    ``name`` -- which every ``>>options`` block needs regardless of what it resolves to -- is not."""

    script = (
        "from edelweissfe.utils.inputlanguage import InputLanguage\n"
        "language = InputLanguage()\n"
        "language.ensureParserLoaded()\n"
        "for module in language['step'].modules:\n"
        "    kw = module.getKeyword('options')\n"
        "    print(','.join(arg.name for arg in kw.requiredArgs))\n"
    )
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, cwd=_REPO_ROOT, check=False)
    if result.returncode != 0:
        pytest.fail(f"Could not render the 'options' keyword:\n{result.stderr}")

    lines = [line for line in result.stdout.splitlines() if line]
    assert lines, "no step type registers the 'options' keyword at all -- the render must be broken"
    assert all(
        line == "name" for line in lines
    ), "the 'options' keyword must declare exactly one required arg, 'name' -- found: " + repr(lines)


def test_every_option_on_the_shared_keyword_defaults_to_none():
    """The invariant :func:`~edelweissfe.stepactions.options._writtenOptions`'s strip-``None``s
    rests on."""
    args = _optionsKeywordArgs()
    assert args, "no options were registered at all -- the render must be broken"

    offenders = [(module, name, default) for module, name, default, _ in args if default != "None"]
    assert not offenders, (
        "options on the shared 'options' step keyword must default to None, otherwise "
        "_writtenOptions cannot tell a user's entry from a foreign module's default and would "
        "silently leak these into every resolved target: " + repr(offenders)
    )


def test_registerSchemaOptions_is_the_only_registrar_on_the_shared_keyword():
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
        "the shared 'options' keyword must only ever be extended through registerSchemaOptions, "
        "which forces the runtime default to None; found other call sites: " + repr(hits)
    )


def test_documented_default_is_rendered_but_does_not_change_the_runtime_default():
    """The docs show the real (schema) default, the parser still sees ``None``."""
    byName = {name: (default, documented) for _, name, default, documented in _optionsKeywordArgs()}

    # a solver option whose effective default lives in NISTSchema
    assert byName["defaultMaxIter"] == ("None", "10")
    # ... and one whose value is a string, to catch a repr/format slip
    assert byName["linsolver"] == ("None", "'pardiso'")


def _journal() -> Journal:
    return Journal()


def _jobInfo() -> dict:
    return {"fieldCorrectionTolerance": {}, "fluxResidualTolerance": {}, "fluxResidualToleranceAlternative": {}}


class _FakeModel:
    """A minimal stand-in for FEModel carrying only what _resolveTarget consults, exercising the
    same "no InputLanguage/parser involvement needed" property every other ported step action's
    tests rely on."""

    def __init__(self, solvers=None, outputManagers=None):
        self.solvers = solvers or {}
        self.outputManagers = outputManagers or {}


#: A parsed ``>>options, name=theSolver, extrapolation=off`` block, as the parser hands it over:
#: every option any module registered on the shared keyword is present, ``None`` where unset, plus
#: the parser's own bookkeeping keys.
def _parsedBlock(name: str, **written) -> dict:
    block = {
        "name": name,
        "defaultMaxIter": None,
        "defaultCriticalIter": None,
        "defaultMaxGrowingIter": None,
        "extrapolation": None,
        "extrapolateAfterModelChange": None,
        "equilibrateAfterModelChange": None,
        "linsolver": None,
        "linsolverConfigFile": None,
        "inputFile": "/some/path/test.inp",
        "datalines": [],
    }
    block.update(written)
    return block


def test_writtenOptions_keeps_only_what_the_user_set():
    """No ``None``s, no parser internals, no ``name`` -- what's left is exactly what the user wrote."""

    written = _writtenOptions(_parsedBlock("theSolver", extrapolation="off"))
    assert written == {"extrapolation": "off"}


def test_resolveTarget_finds_a_solver_by_name():
    solver = NIST(_jobInfo(), _journal())
    model = _FakeModel(solvers={"theSolver": solver})
    assert _resolveTarget("theSolver", model) is solver


def test_resolveTarget_finds_an_output_manager_by_name():
    class FakeOutputManager:
        schema = object()

    manager = FakeOutputManager()
    model = _FakeModel(outputManagers={"myExport": manager})
    assert _resolveTarget("myExport", model) is manager


def test_resolveTarget_rejects_an_unknown_name():
    model = _FakeModel()
    with pytest.raises(ValueError, match="not the name of any declared"):
        _resolveTarget("nonexistent", model)


def test_resolveTarget_rejects_a_name_shared_by_a_solver_and_an_output_manager():
    """Searched deliberately, rather than returning whichever is found first: silently picking one
    would apply an override to the wrong object with no diagnostic at all."""

    class FakeOutputManager:
        schema = object()

    solver = NIST(_jobInfo(), _journal())
    model = _FakeModel(solvers={"shared": solver}, outputManagers={"shared": FakeOutputManager()})
    with pytest.raises(ValueError, match="names both a solver and an output manager"):
        _resolveTarget("shared", model)


def test_resolveTarget_rejects_a_target_with_no_schema():
    class SchemalessTarget:
        schema = None

    model = _FakeModel(solvers={"bare": SchemalessTarget()})
    with pytest.raises(ValueError, match="declares no option schema"):
        _resolveTarget("bare", model)


def test_creation_applies_the_written_options_immediately():
    solver = NIST(_jobInfo(), _journal())
    model = _FakeModel(solvers={"theSolver": solver})

    StepAction.fromStepActionDefinition(
        "theSolver", _parsedBlock("theSolver", extrapolation="off"), None, model, None, _journal()
    )

    assert solver.options["extrapolation"] == "off"


def test_an_override_sticks_across_a_step_that_does_not_repeat_it():
    """The behavior this whole mechanism replaces was, despite a comment claiming otherwise,
    empirically sticky: an option set once stays in effect until a later declaration changes it, not
    just for the step that declared it. This pins that -- now genuinely intentional -- behavior."""

    solver = NIST(_jobInfo(), _journal())
    model = _FakeModel(solvers={"theSolver": solver})

    action = StepAction.fromStepActionDefinition(
        "theSolver", _parsedBlock("theSolver", extrapolation="off"), None, model, None, _journal()
    )

    # A later step's block omits 'extrapolation' entirely and sets an unrelated option only.
    action.updateStepActionFromDefinition(_parsedBlock("theSolver", linsolver="klu"), None, model, None, _journal())

    assert solver.options["extrapolation"] == "off"
    assert solver.options["linsolver"] == "klu"


def test_writing_an_option_the_target_does_not_declare_raises():
    """A typo, or an option meant for a different target, is a loud error -- not a silent no-op."""

    solver = NIST(_jobInfo(), _journal())
    model = _FakeModel(solvers={"theSolver": solver})

    with pytest.raises(ValueError, match="not a valid option"):
        StepAction.fromStepActionDefinition(
            "theSolver",
            _parsedBlock("theSolver", **{"runge-kutta-stages": "3"}),
            None,
            model,
            None,
            _journal(),
        )
