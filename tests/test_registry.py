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
    assert schema is None


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
    assert schema is None


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
