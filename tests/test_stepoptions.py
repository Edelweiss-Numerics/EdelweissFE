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

Four concerns, mirrored by four groups of tests below:

- **Resolution.** ``name`` must resolve to exactly one of ``model.solvers``/``model.outputManagers``,
  with a loud error for "neither" and for "both" (an instance name reused across the two, which would
  otherwise silently pick one).
- **The shared keyword's grammar is minimal and dynamic.** ``>>options`` declares only ``name``
  statically; no solver's or output manager's option names are pre-declared on it at all (unlike the
  mechanism this one replaces, which required a static, hand-synchronized aggregate -- see U3c of
  ``PLAN_INPUT_SYSTEM_UNIFICATION.md``). Any other ``key=value`` pair is accepted unvalidated by the
  parser (:meth:`~edelweissfe.utils.inputlanguage.InputFileKeyword.allowArbitraryOptionalArgs`) and
  handed on raw for :func:`~edelweissfe.stepactions.options._writtenOptions` to recover.
- **The keyword is declared in exactly one place.** A source-level guard against a future call site
  reintroducing a static, pre-declared aggregate.
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


def _optionsKeywordShape() -> list:
    """The required/optional arg names, and the ``acceptsArbitraryArgs`` flag, of the shared
    ``options`` keyword, across all step types.

    Rendered in a subprocess for the same reason as the golden grammar surface: the full input
    language is only populated once the parser has imported every module, and doing that in-process
    would depend on what other test modules have already imported.
    """
    script = (
        "from edelweissfe.utils.inputlanguage import InputLanguage\n"
        "language = InputLanguage()\n"
        "language.ensureParserLoaded()\n"
        "for module in language['step'].modules:\n"
        "    kw = module.getKeyword('options')\n"
        "    required = ','.join(arg.name for arg in kw.requiredArgs)\n"
        "    optional = ','.join(arg.name for arg in kw.optionalArgs)\n"
        "    print(f'{module.name}|{required}|{optional}|{kw.acceptsArbitraryArgs}')\n"
    )
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, cwd=_REPO_ROOT, check=False)
    if result.returncode != 0:
        pytest.fail(f"Could not render the 'options' keyword:\n{result.stderr}")
    return [line.split("|", 3) for line in result.stdout.splitlines() if line]


def test_name_is_the_only_required_arg_on_the_shared_keyword():
    """Every other option is target-specific and therefore not declared at the keyword level at
    all; only ``name`` -- which every ``>>options`` block needs regardless of what it resolves to --
    is."""

    shape = _optionsKeywordShape()
    assert shape, "no step type registers the 'options' keyword at all -- the render must be broken"
    assert all(
        required == "name" for _, required, _, _ in shape
    ), "the 'options' keyword must declare exactly one required arg, 'name' -- found: " + repr(shape)


def test_the_shared_keyword_declares_no_optional_args_and_accepts_arbitrary_ones():
    """The shared keyword pre-declares nothing beyond ``name``: every solver's/output manager's
    option is validated dynamically, downstream, once ``name`` has resolved -- never against a
    static list here. ``acceptsArbitraryArgs`` is what lets the parser's static grammar check accept
    an option it does not itself know the name of (see
    :meth:`~edelweissfe.utils.inputlanguage.InputFileKeyword.allowArbitraryOptionalArgs`)."""

    shape = _optionsKeywordShape()
    assert shape, "no step type registers the 'options' keyword at all -- the render must be broken"

    offenders = [(module, optional) for module, _, optional, _ in shape if optional]
    assert not offenders, (
        "the shared 'options' keyword must declare no optional args at all -- found some "
        "pre-declared, i.e. a static aggregate crept back in: " + repr(offenders)
    )

    notArbitrary = [module for module, _, _, arbitrary in shape if arbitrary != "True"]
    assert not notArbitrary, (
        "every step type's 'options' keyword must accept arbitrary optional args (the dynamic "
        "validation this mechanism relies on) -- these do not: " + repr(notArbitrary)
    )


def test_the_options_keyword_is_declared_in_exactly_one_place():
    """A source-level guard, deliberately: the failure mode is a *new* call site added years from
    now that pre-declares options on this keyword again -- statically -- which no amount of
    exercising today's inputs would catch, since a static declaration is a strict superset of what
    dynamic validation already accepts.
    """
    declarationHits = subprocess.run(
        ["grep", "-rn", "--include=*.py", 'addOptionalKeyword("options"', "edelweissfe"],
        capture_output=True,
        text=True,
        cwd=_REPO_ROOT,
        check=False,
    ).stdout.splitlines()
    assert len(declarationHits) == 1 and declarationHits[0].startswith(
        "edelweissfe/stepactions/options.py:"
    ), "the shared 'options' keyword must be declared in exactly one place: " + repr(declarationHits)

    extensionHits = subprocess.run(
        ["grep", "-rn", "--include=*.py", 'getKeyword("options")', "edelweissfe"],
        capture_output=True,
        text=True,
        cwd=_REPO_ROOT,
        check=False,
    ).stdout.splitlines()
    assert extensionHits == [], (
        "no module may reach back into the shared 'options' keyword to add options onto it "
        "statically -- its grammar is deliberately minimal and its options are resolved "
        "dynamically at runtime instead: found " + repr(extensionHits)
    )


#: Both remaining tests drive the real parser entry point end-to-end, which requires the full
#: 'step' grammar (every step type + step action module) to already be loaded via
#: 'edelweissfe.utils.inputfileparser' -- exactly the import-order-sensitive state every other
#: grammar-surface test in this suite renders in a **fresh subprocess** rather than in-process (see
#: ``tests/test_inputlanguage_golden.py``'s module docstring): another test module importing e.g.
#: ``edelweissfe.steps.adaptivestep`` standalone earlier in the same shared pytest session would
#: otherwise leave that step type's ``Module`` built but never attached to the ``InputLanguage``
#: singleton's ``'step'`` keyword (the very "if keyword in inputLanguage: silently no-op" hazard
#: this whole redesign exists to remove), making these two tests' outcome depend on test order.
_PARSE_OPTIONS_SNIPPET = (
    "import sys\n"
    "from edelweissfe.utils.inputfileparser import parseModuleKeywordLine\n"
    "line = sys.argv[1]\n"
    "try:\n"
    "    keyword, options = parseModuleKeywordLine(line, 'test.inp', 'step', {'type': 'adaptive'}, {'step': []})\n"
    "except ValueError as e:\n"
    "    print(f'ERROR|{e}')\n"
    "else:\n"
    '    print(f\'OK|{keyword}|{options["name"]}|{options["extrapolation"]}|{options["acompletelymadeupoption"]}\')\n'
)


def _parseOptionsLineInFreshInterpreter(line: str) -> str:
    result = subprocess.run(
        [sys.executable, "-c", _PARSE_OPTIONS_SNIPPET, line],
        capture_output=True,
        text=True,
        cwd=_REPO_ROOT,
        check=False,
    )
    if result.returncode != 0:
        pytest.fail(f"Could not parse {line!r}:\n{result.stderr}")
    return result.stdout.strip()


def test_an_option_belonging_to_no_pre_declared_list_still_parses_statically():
    """The parser accepts *any* ``key=value`` pair here, precisely because none is pre-declared --
    proving the claim end-to-end through the real parser entry point, not just against the
    lower-level pieces the other tests in this file exercise directly."""

    output = _parseOptionsLineInFreshInterpreter(
        ">>options, name=mySolver, extrapolation=off, aCompletelyMadeUpOption=42"
    )
    assert output == "OK|options|mySolver|off|42", output


def test_missing_name_still_raises_at_parse_time():
    """``name`` is the one thing every ``>>options`` block needs regardless of what it resolves to,
    so it stays a statically-enforced required arg even though everything else is dynamic."""

    output = _parseOptionsLineInFreshInterpreter(">>options, extrapolation=off")
    assert output.startswith("ERROR|"), output
    assert "required keyword argument" in output


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
