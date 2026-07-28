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
"""P1 tests (see PLAN_INPUT_SYSTEM.md) for ``edelweissfe/config/registry.py``, the L3 lazy
registry.

Covers: zero-eager-import (subprocess, since ``sys.modules`` pollution from other test modules
would otherwise make an in-process check meaningless -- the same reasoning
``tests/test_inputlanguage_golden.py`` already documents), built-in resolution without any
entry-point metadata, synthetic entry-point discovery via a directly-constructed
``importlib.metadata.EntryPoint`` (a documented, public seam -- no on-disk package install
involved), helpful lookup-failure messages, and thread-safety of the memoized lookup under
concurrent access.
"""

import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from importlib.metadata import EntryPoint
from unittest.mock import patch

import pytest

from edelweissfe.config import registry
from edelweissfe.outputmanagers.base.outputmanagerbase import OutputManagerBase
from edelweissfe.utils.schema import schemaOf


class _PluginSchema:
    """Stand-in for a third-party package's L2 option schema."""


class _PluginOutputManagerWithoutSchema(OutputManagerBase):
    """A plugin that has not declared an L2 schema, inheriting OptionSchemaProvider's None."""

    def __init__(self, name, definitionLines, model, fieldOutputController, journal, plotter):
        pass

    def initializeJob(self):
        pass

    def initializeStep(self, step):
        pass

    def finalizeIncrement(self, timeStep, **kwargs):
        pass

    def finalizeFailedIncrement(self, **kwargs):
        pass

    def finalizeStep(self):
        pass

    def finalizeJob(self):
        pass


class _PluginOutputManager(OutputManagerBase):
    """Stand-in for an output manager contributed by an external package via an entry point.

    Defined at module level (not inside the test) because the entry point must resolve it by
    dotted string, exactly as a real installed plugin would. The abstract methods are stubbed
    only so the class is concrete; none of them is called -- the test asserts on resolution.
    """

    schema = _PluginSchema

    def __init__(self, name, definitionLines, model, fieldOutputController, journal, plotter):
        pass

    def initializeJob(self):
        pass

    def initializeStep(self, step):
        pass

    def finalizeIncrement(self, timeStep, **kwargs):
        pass

    def finalizeFailedIncrement(self, **kwargs):
        pass

    def finalizeStep(self):
        pass

    def finalizeJob(self):
        pass


def test_importing_registry_does_not_import_any_builtin_category_module():
    """Importing ``edelweissfe.config.registry`` must not import any element, material, solver,
    or output-manager module -- only resolving a specific name (via :func:`lookup`) may.

    Run in a fresh subprocess: importing anything from ``edelweissfe`` in-process here would
    already have pulled in an unpredictable subset of these modules via whichever other test
    module ran first in the shared pytest session (exactly the import-order fragility this whole
    redesign exists to remove -- see ``tests/test_inputlanguage_golden.py``'s docstring for the
    same argument applied to ``InputLanguage``).
    """
    probeCategoryPrefixes = (
        "edelweissfe.outputmanagers.",
        "edelweissfe.materials.",
        "edelweissfe.elements.",
        "edelweissfe.solvers.",
        "edelweissfe.stepactions.",
        "edelweissfe.generators.",
        "edelweissfe.sections.",
        "edelweissfe.constraints.",
        "edelweissfe.analyticalfields.",
        "edelweissfe.modelmodifiers.",
        "edelweissfe.adaptivity.",
    )
    code = (
        "import sys\n"
        "before = set(sys.modules)\n"
        "import edelweissfe.config.registry\n"
        "after = set(sys.modules)\n"
        "newlyImported = after - before\n"
        "for m in sorted(newlyImported):\n"
        "    print(m)\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=True,
    )
    newlyImported = [line for line in result.stdout.splitlines() if line]
    offending = [m for m in newlyImported if m.startswith(probeCategoryPrefixes)]
    assert offending == [], f"Importing registry.py eagerly imported: {offending}"


def test_builtin_lookup_resolves_without_any_entry_point_metadata():
    """The built-in table must work standalone: no ``pip install -e .`` / metadata regeneration
    is required to resolve EdelweissFE's own modules.

    Verified empirically here by forcing ``entry_points`` to return nothing (as it would for a
    package whose editable-install metadata has gone stale) and confirming the lookup still
    succeeds purely off the static ``_BUILTINS`` table.
    """
    with patch.object(registry, "entry_points", return_value=[]):
        target, schema = registry.lookup("outputmanager", "ensight")

    from edelweissfe.outputmanagers.ensight import OutputManager

    assert target is OutputManager
    # What this test is about is the *target* resolving off the static table. The schema is asserted
    # against what the class itself declares rather than against a literal, so that porting a module
    # to L1/L2 (which gives it a non-None schema) cannot fail this test as a side effect -- the
    # contract "the registry hands out exactly the schema the class declares" holds either way.
    assert schema is schemaOf(OutputManager)


@pytest.mark.parametrize(
    "category,name,moduleName,attrName",
    [
        ("outputmanager", "ensight", "edelweissfe.outputmanagers.ensight", "OutputManager"),
        ("stepaction", "dirichlet", "edelweissfe.stepactions.dirichlet", "StepAction"),
        ("solver", "NIST", "edelweissfe.solvers.nonlinearimplicitstatic", "NIST"),
        ("section", "plane", "edelweissfe.sections.plane", "Section"),
    ],
)
def test_builtin_lookup_matches_direct_import(category, name, moduleName, attrName):
    """A representative sample of built-in categories resolves to exactly the class a direct
    ``import`` would give, proving the registry is not inventing a second, divergent object."""
    import importlib

    expected = getattr(importlib.import_module(moduleName), attrName)
    target, schema = registry.lookup(category, name)
    assert target is expected
    # As above: pinned to the class's own declaration, not to a literal None, so this stays true as
    # categories are ported. `schema is None` is still what it asserts for the unported ones.
    assert schema is schemaOf(expected)


def test_case_insensitive_lookup():
    target_lower, _ = registry.lookup("outputmanager", "ensight")
    target_mixed, _ = registry.lookup("OutputManager", "EnSight")
    assert target_lower is target_mixed


def test_synthetic_entry_point_is_discoverable():
    """An external package registers an implementation via the
    ``edelweissfe.plugins`` entry-point group in its own ``pyproject.toml``; simulate that
    without touching the installed environment by constructing a real
    ``importlib.metadata.EntryPoint`` in-process (this is a documented, public constructor -- not
    a private/undocumented seam) and patching ``registry.entry_points`` to return it.

    The "plugin" points at a real, already-importable attribute
    (``edelweissfe.utils.misc:asBool``) purely as a stand-in target -- the point being tested is
    discovery and resolution of the entry point itself, not any particular category's semantics.
    """
    fakeEntryPoint = EntryPoint(
        name="synthetictestcategory.syntheticname",
        value="edelweissfe.utils.misc:asBool",
        group=registry.ENTRY_POINT_GROUP,
    )

    with patch.object(registry, "entry_points", return_value=[fakeEntryPoint]):
        target, schema = registry.lookup("synthetictestcategory", "syntheticname")

    from edelweissfe.utils.misc import asBool

    assert target is asBool
    assert schema is None


def test_synthetic_entry_point_is_not_found_once_patch_is_removed():
    """Sanity check for the previous test: without the patched entry point, the same category is
    genuinely unknown (proving the previous test exercised entry-point discovery, not some
    unrelated fallback)."""
    with pytest.raises(registry.RegistryLookupError):
        registry.lookup("anothersynthetictestcategory", "syntheticname")


def test_lookup_failure_lists_available_names_for_the_category():
    with pytest.raises(registry.RegistryLookupError) as excinfo:
        registry.lookup("outputmanager", "totallyBogusName")
    message = str(excinfo.value)
    assert "outputmanager" in message
    assert "ensight" in message  # a real, available name should be listed


def test_lookup_failure_for_unknown_category_does_not_crash():
    """A category with zero registered names must still produce a clean error, not an internal
    exception from an empty-list edge case (the failure mode this replaces:
    ``inputlanguage.py``'s ``findSimilarString`` raises a bare, unhelpful ``Exception`` when its
    candidate list is empty)."""
    with pytest.raises(registry.RegistryLookupError) as excinfo:
        registry.lookup("thisCategoryDoesNotExistAtAll", "whatever")
    assert "thisCategoryDoesNotExistAtAll" in str(excinfo.value)


def test_memoized_lookup_is_thread_safe_under_concurrent_access():
    """Hammer ``lookup()`` for the same, not-yet-resolved key from many threads at once and
    assert every thread observes the identical resolved object -- the property the ``_lock``
    double-checked-locking in :func:`registry.lookup` is meant to guarantee under
    ``PYTHON_GIL=0``.
    """
    category, name = "outputmanager", "conditionalstop"
    key = (category.casefold(), name.casefold())

    # Force a genuine cache miss for every thread's first attempt, regardless of what earlier
    # tests in this process may have already resolved.
    registry._resolved.pop(key, None)

    barrier = threading.Barrier(16)

    def resolve():
        barrier.wait()
        return registry.lookup(category, name)

    with ThreadPoolExecutor(max_workers=16) as pool:
        results = list(pool.map(lambda _: resolve(), range(16)))

    targets = [target for target, _ in results]
    assert all(t is targets[0] for t in targets), "Concurrent lookups returned inconsistent objects"

    from edelweissfe.outputmanagers.conditionalstop import OutputManager

    assert targets[0] is OutputManager


def test_register_allows_manual_registration_bypassing_builtins_and_entrypoints():
    class _FakeImplementation:
        pass

    class _FakeSchema:
        pass

    registry.register("registrytestcategory", "manualentry", _FakeImplementation, schema=_FakeSchema)
    target, schema = registry.lookup("registrytestcategory", "manualentry")
    assert target is _FakeImplementation
    assert schema is _FakeSchema


def test_schema_reaches_the_caller_through_the_builtin_dotted_string_path():
    """A schema declared as a class attribute on a *built-in* must come back from ``lookup``.

    This is the P2 blocker's resolution: ``register(..., schema=...)`` writes straight into the
    memo cache and was previously the *only* way a non-``None`` schema could ever be returned, so
    the built-in table and the entry-point paths were structurally schema-blind. ``lookup`` now
    obtains the schema from the resolved target itself via ``schemaOf``.

    Asserted against a real built-in (not a synthetic stand-in) so that a regression in either the
    registry or the output manager's own declaration fails this test.
    """
    target, schema = registry.lookup("outputmanager", "computetimemonitor")

    from edelweissfe.outputmanagers.computetimemonitor import (
        ComputeTimeMonitorSchema,
        OutputManager,
    )

    assert target is OutputManager
    assert schema is ComputeTimeMonitorSchema


def test_schema_reaches_the_caller_through_a_third_party_entry_point():
    """The case that actually motivates the whole convention (PLAN_INPUT_SYSTEM.md §4).

    An external package -- EdelweissMeshfree, or any plugin -- registers a class via an entry
    point in its own ``pyproject.toml``. There is no way to pass a ``schema=`` argument alongside a
    dotted string there, so if the schema did not travel *on the class*, every third-party module
    would silently lose its schema while EdelweissFE's own modules kept theirs. The plugin here
    derives from ``OutputManagerBase`` exactly as a real one would.
    """
    fakeEntryPoint = EntryPoint(
        name="outputmanager.syntheticpluginwithschema",
        # Derived from __name__ rather than hardcoded so the test does not depend on pytest's
        # import mode (prepend imports this file as "test_registry", importlib as "tests.test_registry").
        value=f"{__name__}:_PluginOutputManager",
        group=registry.ENTRY_POINT_GROUP,
    )

    with patch.object(registry, "entry_points", return_value=[fakeEntryPoint]):
        target, schema = registry.lookup("outputmanager", "syntheticpluginwithschema")

    assert target is _PluginOutputManager
    assert schema is _PluginSchema


def test_schema_is_none_for_a_class_that_declares_none():
    """A class that has not been given an L2 schema inherits ``OptionSchemaProvider``'s ``None``
    default rather than failing lookup -- which is what lets a base class adopt the mixin before its
    subclasses are ported.

    Deliberately asserted against a synthetic subclass rather than a real not-yet-ported module:
    the set of unported modules shrinks with every port, so naming one here would make this test
    fail as a side effect of unrelated progress (it did exactly that when statusfile was ported).
    The contract, unlike the module list, is stable.
    Resolved through an entry point rather than ``register()`` on purpose: ``register()`` writes
    ``(target, schema)`` straight into the memo cache, so a lookup that hits it never calls
    ``schemaOf`` at all and the test would prove nothing.
    """
    fakeEntryPoint = EntryPoint(
        name="outputmanager.syntheticpluginwithoutschema",
        value=f"{__name__}:_PluginOutputManagerWithoutSchema",
        group=registry.ENTRY_POINT_GROUP,
    )

    with patch.object(registry, "entry_points", return_value=[fakeEntryPoint]):
        target, schema = registry.lookup("outputmanager", "syntheticpluginwithoutschema")

    assert target is _PluginOutputManagerWithoutSchema
    assert schema is None
    assert _PluginOutputManagerWithoutSchema.schema is None, "the inherited default itself"


def test_schema_is_none_for_a_function_target_rather_than_raising():
    """Every ``generator`` entry resolves to the module-level function ``generateModelData``, which
    cannot inherit a class attribute. ``schemaOf`` must report ``None`` for it, not raise
    ``AttributeError`` -- this is the case that rules out a bare ``target.schema`` in ``lookup``."""
    target, schema = registry.lookup("generator", "boxgen")

    assert callable(target) and not isinstance(target, type)
    assert schema is None


def test_registering_the_identical_target_twice_is_idempotent():
    """Tests and repeated imports rely on this, so identity is compared before raising."""

    class _Implementation:
        pass

    registry.register("registryconflictcategory", "idempotent", _Implementation)
    registry.register("registryconflictcategory", "idempotent", _Implementation)

    assert registry.lookup("registryconflictcategory", "idempotent")[0] is _Implementation


def test_registering_a_different_target_under_a_taken_name_raises_naming_both():
    class _Incumbent:
        pass

    class _Newcomer:
        pass

    registry.register("registryconflictcategory", "contested", _Incumbent)

    with pytest.raises(registry.RegistryConflictError) as excInfo:
        registry.register("registryconflictcategory", "contested", _Newcomer)

    assert "_Incumbent" in str(excInfo.value) and "_Newcomer" in str(excInfo.value)
    assert registry.lookup("registryconflictcategory", "contested")[0] is _Incumbent, "unchanged"

    registry.register("registryconflictcategory", "contested", _Newcomer, override=True)
    assert registry.lookup("registryconflictcategory", "contested")[0] is _Newcomer


def test_registering_over_a_builtin_name_raises_without_importing_it():
    """The core hazard: names are casefolded, so `register("outputmanager", "Ensight", ...)` used to
    make the built-in `ensight` simply stop existing, with no diagnostic anywhere. Checked against
    the static table of strings, so it neither imports the built-in nor depends on whether anything
    resolved it earlier -- collision detection must not itself be import-order dependent."""

    class _Impostor:
        pass

    with pytest.raises(registry.RegistryConflictError, match="built-in"):
        registry.register("outputmanager", "Ensight", _Impostor)

    from edelweissfe.outputmanagers.ensight import OutputManager

    assert registry.lookup("outputmanager", "ensight")[0] is OutputManager, "built-in still reachable"


def test_two_entry_points_claiming_one_name_raise_rather_than_resolving_arbitrarily():
    """`entry_points()` has no documented ordering, so returning the first match made the winner
    depend on installation order and differ between identical environments."""
    duplicates = [
        EntryPoint(
            name="outputmanager.contestedplugin",
            value=f"{__name__}:_PluginOutputManager",
            group=registry.ENTRY_POINT_GROUP,
        ),
        EntryPoint(
            name="outputmanager.contestedplugin",
            value=f"{__name__}:_PluginOutputManagerWithoutSchema",
            group=registry.ENTRY_POINT_GROUP,
        ),
    ]

    with patch.object(registry, "entry_points", return_value=duplicates):
        with pytest.raises(registry.RegistryConflictError, match="claim the 'outputmanager' name"):
            registry.lookup("outputmanager", "contestedplugin")


def test_an_entry_point_shadowing_a_builtin_name_is_reported_not_silently_ignored():
    """The mirror image of the `register` hazard: built-ins resolve first, so a plugin claiming a
    built-in name would never load and its author would get no diagnostic at all."""
    shadowing = EntryPoint(
        name="outputmanager.ensight",
        value=f"{__name__}:_PluginOutputManager",
        group=registry.ENTRY_POINT_GROUP,
    )

    with patch.object(registry, "entry_points", return_value=[shadowing]):
        assert registry._entryPointsShadowingBuiltins() == {
            "outputmanager.ensight": (
                "edelweissfe.outputmanagers.ensight:OutputManager",
                f"{__name__}:_PluginOutputManager",
            )
        }

        # The audit runs once per process and may already have run; force it for this assertion.
        registry._entryPointsAudited = False
        try:
            with pytest.raises(registry.RegistryConflictError, match="claim built-in names"):
                registry._auditEntryPoints()
        finally:
            registry._entryPointsAudited = True


def test_isRegistered_answers_without_importing_the_implementation():
    """The property that lets ``isRegistered`` replace ``importlib.util.find_spec`` in
    :class:`~edelweissfe.steps.stepmanager.StepActionCollection`.

    That collection is asked about categories a consumer defines *no* actions in (a solver checking
    for ``indirectcontrol``, say), so an existence check that imported the implementation would pull
    in modules the simulation never uses -- and would do it for a name the user never wrote.

    A fresh subprocess again, because by the time this runs in a shared pytest session some other
    test has certainly imported half of ``edelweissfe.stepactions`` already.
    """
    code = (
        "import sys\n"
        "from edelweissfe.config import registry\n"
        "assert registry.isRegistered('stepaction', 'Dirichlet')\n"
        "assert not registry.isRegistered('stepaction', 'nosuchstepaction')\n"
        "for m in sorted(m for m in sys.modules if m.startswith('edelweissfe.stepactions.')):\n"
        "    print(m)\n"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    imported = [line for line in result.stdout.splitlines() if line]
    assert imported == [], f"isRegistered imported step action modules: {imported}"


def test_isRegistered_sees_all_three_registration_mechanisms():
    """Built-in table, entry point and in-process ``register`` must all be visible to the predicate,
    otherwise it would disagree with :func:`lookup` about what exists."""
    assert registry.isRegistered("stepaction", "dirichlet")  # built-in table

    plugin = EntryPoint(
        name="stepaction.pluginaction",
        value=f"{__name__}:_PluginOutputManager",
        group=registry.ENTRY_POINT_GROUP,
    )
    with patch.object(registry, "entry_points", return_value=[plugin]):
        assert registry.isRegistered("stepaction", "PluginAction")

    registry.register("stepaction", "inprocessaction", _PluginOutputManager)
    assert registry.isRegistered("stepaction", "InProcessAction")
    assert "inprocessaction" in registry.availableNames("stepaction")
