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

"""Audits every step action's option handling against its declared grammar.

Two failure modes are pinned here, both of which this project has been bitten by repeatedly and
neither of which any simulation test reliably catches.

**A read of an undeclared option is a latent ``KeyError``.** Which options a definition dict carries
is not fixed: the parser validates a *full* declaration against the module's own keyword, but a
*partial* re-declaration of an already-defined step action in a later step against the module's
``update<keyword>`` companion (``utils/inputfileparser.py``'s ``parseModuleKeywordLine``), whose arg
list is deliberately smaller -- it omits exactly those args that cannot change, such as a load's
node set. So ``updateStepActionFromDefinition`` may only read what the *update* keyword declares.
``testfiles/marmot/NodeForces`` and ``testfiles/marmot/GeoStatic`` are the two inputs that exercise
this path; a module without such coverage would fail only in a user's multi-step input.

**A declared option that nothing reads is silently ignored.** The user writes it, the parser accepts
it, and it has no effect -- the failure mode behind the ``legend``/``axpSec`` family in ``meshplot``
and the ``c``/``ls`` family in ``Plotter.plotXYData``. The current set of such options is pinned in
:data:`KNOWN_UNREAD_OPTIONS` so a *new* one cannot be introduced unnoticed.

The audit is **source-level**, deliberately, for the same reason ``tests/test_stepoptions.py``'s
registrar test is: no amount of exercising today's inputs can catch an option read that a future
edit adds, and 41 of the ``testfiles/marmot`` cases skip on a Marmot build without their material,
so simulation coverage of these modules is far thinner than the file count suggests.
"""

import ast
import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).parent.parent
STEP_ACTION_DIR = REPO_ROOT / "edelweissfe" / "stepactions"
GRAMMAR_SCRIPT = Path(__file__).parent / "_stepaction_grammar.py"

#: Parameter names that hold a parsed option mapping. ``definition`` is the name the ported modules
#: use; the other two are the legacy spellings, kept so that an unported module is still audited.
DEFINITION_PARAMETER_NAMES = frozenset({"definition", "action", "options"})

#: Keys that are never declared as args but legitimately appear in a definition mapping: the parser
#: injects them, or a helper pops them before the step action sees the dict.
NON_OPTION_KEYS = frozenset({"name", "inputfile", "inputFile", "datalines", "explicitlySetArgs", "moduleOptions"})

#: Options that are declared on a keyword but read by nothing -- i.e. silently ignored today. Every
#: entry is a latent bug, recorded rather than fixed because fixing means deciding what the option
#: should *mean*, which is a product decision and not part of a behaviour-neutral port.
#:
#: Empty now that all twelve step actions use schema-based option handling: a schema-based module's
#: genuinely-unread fields -- e.g. `distributedload`'s `field` (never consumed: `DistributedLoadBase`
#: has no notion of a field), `bodyforce`'s `delta` (declared for an incremental update, but
#: unreachable since `bodyforce` offers no `updatebodyforce` keyword, so a re-declaration is always
#: full and its required `forceVector` always wins), or `indirectcontrol`'s `exportCVector` (declared
#: but never implemented -- only its sibling `indirectcontractioncontrol` does) -- do not need an
#: entry here: `buildSchemaFromOptions`/`coercePresentOptions` still type-validate them even though
#: their coerced value is then discarded, which is a milder failure mode than this dict exists to
#: flag -- a value with *no* validation and *no* effect, e.g. `meshplot`'s `legend`/`axpSec` or
#: `Plotter.plotXYData`'s `c`/`ls`. Kept as a dict (not deleted) since those two other, unrelated
#: modules may yet need it.
KNOWN_UNREAD_OPTIONS = {}

#: Modules exempt from the "every declared option is read" check, with the reason.
UNREAD_CHECK_EXEMPT_MODULES = {
    # `options` is a *shared container*, not a consumer: every solver and output manager registers
    # its own args on this one keyword via `registerOptionsArg` and reads them back through
    # `getOptionsOfCategory`. The container reading none of them is the design, not a bug.
    "options",
}

#: The two validation entry points a schema-based module hands its `definition`/`self.schema` to (see
#: ``utils/schema.py``). Neither can ``KeyError`` on a key its schema declares but the caller's
#: dict happens to lack: ``buildSchemaFromOptions`` lets the dataclass default fill it in (its only
#: failure mode for an absent key is "missing *required*", a distinct, already-enforced check), and
#: ``coercePresentOptions`` is explicitly absence-tolerant for *any* field, required or not. So a
#: call to either is a delegate this file's AST walk cannot follow (the option names it "reads" live
#: in the schema dataclass, in *this* file, not in a locally-defined function) -- resolved by name
#: instead -- but its reads count only toward "read *anywhere* in the module", never toward a
#: per-method "does this risk a KeyError for an option the validated keyword doesn't declare" check,
#: since that risk cannot arise through either function. The real risk schema-based modules carry is
#: different -- calling the *wrong one* of the two in an update path -- and is pinned separately by
#: :func:`test_update_path_does_not_use_build_schema_from_options_when_a_partial_update_is_possible`.
SCHEMA_VALIDATION_FUNCTIONS = frozenset({"buildSchemaFromOptions", "coercePresentOptions"})

#: Synthetic method-name key under which module-wide, KeyError-safe reads are recorded (2-arg
#: `.pop(key, default)` calls, and whatever a :data:`SCHEMA_VALIDATION_FUNCTIONS` call validates) --
#: picked up by the aggregate "not silently unread" check but invisible to the per-method subset
#: checks, which only ever look up real method names.
SCHEMA_LENIENT_READS_KEY = "__lenient_reads__"


def _stepActionGrammar() -> dict:
    """Read the declared step-action grammar from a fresh subprocess.

    Returns
    -------
    dict
        Maps module name -> keyword name -> list of declared arg names.
    """

    result = subprocess.run(
        [sys.executable, str(GRAMMAR_SCRIPT)],
        capture_output=True,
        text=True,
        check=True,
        cwd=str(REPO_ROOT),
    )
    return json.loads(result.stdout)


def _definitionParameters(function: ast.FunctionDef) -> set[str]:
    """The parameters of ``function`` that hold a parsed option mapping.

    Parameters
    ----------
    function
        The function definition to inspect.

    Returns
    -------
    set[str]
        The names of those parameters.
    """

    arguments = function.args
    allArgs = arguments.posonlyargs + arguments.args + arguments.kwonlyargs
    return {arg.arg for arg in allArgs} & DEFINITION_PARAMETER_NAMES


def _readsAndDelegates(function: ast.FunctionDef) -> tuple[set[str], set[str], set[str]]:
    """Collect the option keys ``function`` reads, and the helpers it hands the mapping on to.

    Parameters
    ----------
    function
        The function definition to inspect.

    Returns
    -------
    tuple[set[str], set[str], set[str]]
        The string-literal option keys subscripted (or ``.pop(key)``-ed, with no default) out of a
        definition mapping; the keys read in a way that cannot ``KeyError`` (``.pop(key, default)``,
        two-argument form); and the names of the methods called with the mapping as an argument (so
        their reads count as this function's).
    """

    definitionNames = _definitionParameters(function)
    reads: set[str] = set()
    lenientReads: set[str] = set()
    delegates: set[str] = set()

    if not definitionNames:
        return reads, lenientReads, delegates

    for node in ast.walk(function):
        if isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name) and node.value.id in definitionNames:
            if isinstance(node.slice, ast.Constant) and isinstance(node.slice.value, str):
                reads.add(node.slice.value)
            elif isinstance(node.slice, ast.Call) and isinstance(node.slice.func, ast.Name):
                # `definition[str(index + 1)]`: the numbered component options, iterated over the
                # field size. The set is finite and declared as `1`..`6`, so credit all of them --
                # that is what this idiom reads for a field of maximal size.
                if node.slice.func.id == "str":
                    reads.update(str(i) for i in range(1, 7))
            # A bare `definition[key]` where `key` is a parameter is not guessed at here; the option
            # name is supplied by the caller and is picked up as a literal below.

        if isinstance(node, ast.Call):
            # `definition.pop("key")` / `definition.pop("key", default)`: a structural option (e.g.
            # a node/element set name) popped before the rest is validated against a schema. The
            # one-argument form KeyErrors if `key` is absent -- exactly the risk this audit exists
            # for -- so it counts as a strict read; the two-argument form cannot, so it is only
            # ever a lenient one (see `SCHEMA_LENIENT_READS_KEY`).
            if (
                isinstance(node.func, ast.Attribute)
                and node.func.attr == "pop"
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id in definitionNames
                and node.args
                and isinstance(node.args[0], ast.Constant)
                and isinstance(node.args[0].value, str)
            ):
                if len(node.args) == 1:
                    reads.add(node.args[0].value)
                else:
                    lenientReads.add(node.args[0].value)

            if any(isinstance(arg, ast.Name) and arg.id in definitionNames for arg in node.args):
                if isinstance(node.func, ast.Attribute):
                    delegates.add(node.func.attr)
                elif isinstance(node.func, ast.Name):
                    delegates.add(node.func.id)

                # A call that hands on the mapping together with a string literal is naming the
                # option for the callee to read, e.g. `cls._dofFromDefinition(definition, "dof1",
                # model)`.
                for arg in list(node.args) + [keyword.value for keyword in node.keywords]:
                    if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                        reads.add(arg.value)

    return reads, lenientReads, delegates


def _classAttributeSchemaName(tree: ast.Module) -> str | None:
    """The dataclass name assigned to a class-level ``schema = ...`` attribute in this module.

    Every ported step action declares exactly one such attribute (``schema = XSchema``, per
    ``OptionSchemaProvider``); this recovers ``"XSchema"`` so its declared option names can be
    looked up in the same file.

    Parameters
    ----------
    tree
        The parsed module.

    Returns
    -------
    str | None
        The schema class's name, or ``None`` if the module declares no such attribute.
    """
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            for stmt in node.body:
                if (
                    isinstance(stmt, ast.Assign)
                    and len(stmt.targets) == 1
                    and isinstance(stmt.targets[0], ast.Name)
                    and stmt.targets[0].id == "schema"
                    and isinstance(stmt.value, ast.Name)
                ):
                    return stmt.value.id
    return None


def _schemaFieldOptionNames(tree: ast.Module, schemaClassName: str) -> set[str]:
    """The ``.inp``-facing option names declared by a schema dataclass's ``schemaField(...)``
    class attributes: the ``optionName=`` override if given, else the field's own name.

    Parameters
    ----------
    tree
        The parsed module.
    schemaClassName
        The name of the schema dataclass to inspect.

    Returns
    -------
    set[str]
        The option names the schema accepts.
    """
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == schemaClassName:
            for stmt in node.body:
                if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
                    optionName = stmt.target.id
                    if isinstance(stmt.value, ast.Call):
                        for keyword in stmt.value.keywords:
                            if keyword.arg == "optionName" and isinstance(keyword.value, ast.Constant):
                                optionName = keyword.value.value
                    names.add(optionName)
    return names


def _optionReadsByMethod(modulePath: Path) -> dict[str, set[str]]:
    """Which option keys each method of a step action module reads, transitively.

    A method that hands the definition mapping to a helper is credited with the helper's reads, so
    that a module which factors its translation into ``_xFromDefinition`` staticmethods -- as the
    ported ones do -- is audited as a whole. A method that hands the mapping to either of
    :data:`SCHEMA_VALIDATION_FUNCTIONS` is credited with every option its own ``schema = XSchema``
    class attribute declares, but only under :data:`SCHEMA_LENIENT_READS_KEY` -- see there for why
    neither function's reads belong in a per-method subset check.

    Parameters
    ----------
    modulePath
        The path of the step action module.

    Returns
    -------
    dict[str, set[str]]
        Maps method name to the option keys reachable from it, plus (if applicable)
        :data:`SCHEMA_LENIENT_READS_KEY` mapping to the KeyError-safe reads of the whole module.
    """

    tree = ast.parse(modulePath.read_text())

    directReads: dict[str, set[str]] = {}
    lenientDirectReads: dict[str, set[str]] = {}
    delegatesOf: dict[str, set[str]] = {}
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            reads, lenientReads, delegates = _readsAndDelegates(node)
            directReads[node.name] = directReads.get(node.name, set()) | reads
            lenientDirectReads[node.name] = lenientDirectReads.get(node.name, set()) | lenientReads
            delegatesOf[node.name] = delegatesOf.get(node.name, set()) | delegates

    schemaClassName = _classAttributeSchemaName(tree)
    schemaOptionNames = _schemaFieldOptionNames(tree, schemaClassName) if schemaClassName else set()

    def resolve(name: str, seen: frozenset[str]) -> set[str]:
        if name in seen or name not in directReads:
            return set()
        reads = set(directReads[name])
        for delegate in delegatesOf.get(name, ()):
            reads |= resolve(delegate, seen | {name})
        return reads

    result = {name: resolve(name, frozenset()) for name in directReads}

    lenientAggregate = {key for reads in lenientDirectReads.values() for key in reads}
    if any(delegate in SCHEMA_VALIDATION_FUNCTIONS for delegates in delegatesOf.values() for delegate in delegates):
        lenientAggregate |= schemaOptionNames
    if lenientAggregate:
        result[SCHEMA_LENIENT_READS_KEY] = lenientAggregate

    return result


def _schemaValidationCallsByMethod(modulePath: Path) -> dict[str, set[str]]:
    """Which of :data:`SCHEMA_VALIDATION_FUNCTIONS` each method of a step action module
    transitively calls.

    Parameters
    ----------
    modulePath
        The path of the step action module.

    Returns
    -------
    dict[str, set[str]]
        Maps method name to the subset of :data:`SCHEMA_VALIDATION_FUNCTIONS` reachable from it.
    """

    tree = ast.parse(modulePath.read_text())

    delegatesOf: dict[str, set[str]] = {}
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            _, _, delegates = _readsAndDelegates(node)
            delegatesOf[node.name] = delegatesOf.get(node.name, set()) | delegates

    def resolve(name: str, seen: frozenset[str]) -> set[str]:
        if name in seen or name not in delegatesOf:
            return set()
        used = {delegate for delegate in delegatesOf[name] if delegate in SCHEMA_VALIDATION_FUNCTIONS}
        for delegate in delegatesOf[name]:
            used |= resolve(delegate, seen | {name})
        return used

    return {name: resolve(name, frozenset()) for name in delegatesOf}


def _moduleNames() -> list[str]:
    return sorted(_stepActionGrammar())


@pytest.fixture(scope="module")
def grammar() -> dict:
    return _stepActionGrammar()


@pytest.fixture(scope="module")
def optionReads() -> dict:
    return {name: _optionReadsByMethod(STEP_ACTION_DIR / f"{name}.py") for name in _moduleNames()}


@pytest.fixture(scope="module")
def schemaValidationCalls() -> dict:
    return {name: _schemaValidationCallsByMethod(STEP_ACTION_DIR / f"{name}.py") for name in _moduleNames()}


def test_every_builtin_step_action_declares_its_own_keyword(grammar):
    """Each registered step action module declares a keyword named after itself.

    Without this, a module could be reachable by name from the registry while being unusable from an
    input file -- the shape of the ``timemonitor`` bug (a module registered under a name nothing
    could reach), which is what taught this branch to check reachability explicitly.
    """

    missing = [name for name, keywords in grammar.items() if name not in {kw.casefold() for kw in keywords}]
    assert not missing, f"step action modules declaring no keyword of their own name: {missing}"


def test_every_step_action_is_ported_to_the_typed_constructor_seam(optionReads):
    """Every step action overrides ``fromStepActionDefinition``.

    ``StepActionBase`` deliberately provides a legacy default for this hook so that the port could
    proceed one module at a time (see its docstring). This asserts the port is finished, so the
    default can only ever be reached by a *new* module -- and makes it visible if one is added that
    reverts to handing a parser-shaped dict to ``__init__``.
    """

    unported = sorted(name for name, methods in optionReads.items() if "fromStepActionDefinition" not in methods)
    assert not unported, f"step actions still consuming a parsed definition dict in __init__: {unported}"


@pytest.mark.parametrize("moduleName", _moduleNames())
def test_creation_path_reads_only_options_its_keyword_declares(moduleName, grammar, optionReads):
    """``fromStepActionDefinition`` may only read args declared on the module's own keyword."""

    declared = {arg.casefold() for arg in grammar[moduleName].get(moduleName, [])}
    reads = {key.casefold() for key in optionReads[moduleName].get("fromStepActionDefinition", set())}

    undeclared = reads - declared - {key.casefold() for key in NON_OPTION_KEYS}
    assert not undeclared, (
        f"{moduleName}.fromStepActionDefinition reads option(s) {sorted(undeclared)}, which the "
        f"'{moduleName}' keyword does not declare -- a KeyError for every input using it"
    )


@pytest.mark.parametrize("moduleName", _moduleNames())
def test_update_path_reads_only_options_the_update_keyword_declares(moduleName, grammar, optionReads):
    """``updateStepActionFromDefinition`` may only read args declared on the keyword the parser
    validates a re-declaration against.

    That is the ``update<keyword>`` companion when the module declares one -- and its arg list is
    *smaller* than the main keyword's, which is precisely the trap: reading ``nSet`` here works for a
    full re-declaration and raises for the partial one in ``testfiles/marmot/NodeForces``.
    """

    keywords = grammar[moduleName]
    updateKeyword = next((kw for kw in keywords if kw.casefold() == "update" + moduleName), None)
    validatedAgainst = updateKeyword if updateKeyword is not None else moduleName

    declared = {arg.casefold() for arg in keywords.get(validatedAgainst, [])}
    reads = {key.casefold() for key in optionReads[moduleName].get("updateStepActionFromDefinition", set())}

    undeclared = reads - declared - {key.casefold() for key in NON_OPTION_KEYS}
    assert not undeclared, (
        f"{moduleName}.updateStepActionFromDefinition reads option(s) {sorted(undeclared)}, which the "
        f"'{validatedAgainst}' keyword -- the one a re-declaration is validated against -- does not "
        f"declare"
    )


@pytest.mark.parametrize("moduleName", _moduleNames())
def test_update_path_does_not_use_build_schema_from_options_when_a_partial_update_is_possible(
    moduleName, grammar, schemaValidationCalls
):
    """A module offering a dedicated ``update<keyword>`` grammar accepts a *partial*
    re-declaration -- exactly the shape :func:`~edelweissfe.utils.schema.buildSchemaFromOptions`
    cannot handle, since it enforces every field the schema calls required.

    This is not hypothetical: ``dirichlet``'s update path was first written this way and broke on
    the very re-declaration ``testfiles/marmot/GosfordSandstone`` exercises (``>>dirichlet,
    name=top, 2=-1``, omitting the required ``field``), raising ``ValueError: Missing required
    option(s)`` instead of applying the partial update.
    :func:`~edelweissfe.utils.schema.coercePresentOptions` is the function built for this path -- it
    validates only whatever keys are actually present, with no missing-required check -- so a module
    that declares an update keyword must reach it, not ``buildSchemaFromOptions``, from
    ``updateStepActionFromDefinition``.
    """

    keywords = grammar[moduleName]
    hasUpdateKeyword = any(kw.casefold() == "update" + moduleName for kw in keywords)
    if not hasUpdateKeyword:
        pytest.skip(f"'{moduleName}' declares no update{moduleName} keyword; a re-declaration is always full")

    used = schemaValidationCalls[moduleName].get("updateStepActionFromDefinition", set())
    assert "buildSchemaFromOptions" not in used, (
        f"{moduleName}.updateStepActionFromDefinition calls buildSchemaFromOptions, but '{moduleName}' "
        f"declares an update{moduleName} keyword for partial re-declarations -- a re-declaration omitting "
        f"a required field would raise 'Missing required option(s)' instead of applying the partial "
        f"update. Use coercePresentOptions instead."
    )


@pytest.mark.parametrize("moduleName", _moduleNames())
def test_no_new_silently_ignored_options(moduleName, grammar, optionReads):
    """Every declared option is read somewhere, apart from the recorded exceptions."""

    if moduleName in UNREAD_CHECK_EXEMPT_MODULES:
        pytest.skip(f"'{moduleName}' is exempt: see UNREAD_CHECK_EXEMPT_MODULES")

    everythingRead = {key.casefold() for reads in optionReads[moduleName].values() for key in reads}

    for keywordName, declaredArgs in grammar[moduleName].items():
        allowed = {arg.casefold() for arg in KNOWN_UNREAD_OPTIONS.get((moduleName, keywordName.casefold()), set())}
        unread = {
            arg
            for arg in declaredArgs
            if arg.casefold() not in everythingRead
            and arg.casefold() not in {key.casefold() for key in NON_OPTION_KEYS}
            and arg.casefold() not in allowed
        }
        assert not unread, (
            f"'{keywordName}' declares option(s) {sorted(unread)} that {moduleName}.py never reads -- "
            f"a user writing them gets no effect and no warning. Either consume them, or record them "
            f"in KNOWN_UNREAD_OPTIONS with the reason."
        )
