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

"""No documented module may have an import-time side effect that depends on, or damages, the
import order:

1. **Every documented module imports standalone**, in an interpreter where nothing else from
   ``edelweissfe`` has been imported -- what makes a module reusable by an external caller
   (EdelweissMeshfree, a script) rather than only by the parser.
2. **No documented module poisons the parser by being imported first.** A module that declares
   its grammar via an import-time side effect (e.g. a shared keyword like ``options``, registered
   lazily and idempotently by ``_ensureOptionsKeyword``) must not fail silently if imported before
   the step types it depends on exist.

Each check runs in a **fresh subprocess** per module, which is the only way to test an import-order
property, and the reason this file is slower than the rest of the suite. The subprocesses run
concurrently, and a failure reports *every* offending module rather than only the first, because
these bugs come in families with one shared cause -- seeing all of them together is what points at
the cause instead of at a symptom.
"""

import concurrent.futures
import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).parent.parent
SNAPSHOT_SCRIPT = Path(__file__).parent / "_inputlanguage_snapshot.py"

#: Cap on concurrent interpreters. Each imports numpy and friends, so this is memory-bound rather
#: than CPU-bound; 8 keeps the wall time near a few seconds without thrashing.
_MAX_WORKERS = 8


def _documentedModuleNames() -> list[str]:
    """The dotted names of every ``edelweissfe`` module exposing a ``documentation`` list.

    Discovered by ``tests/_inputlanguage_snapshot.py --list-modules``, i.e. by exactly the same walk
    the golden grammar-surface test uses, so the two cannot disagree about the module set.

    Returns
    -------
    list[str]
        The module names, sorted.
    """

    result = subprocess.run(
        [sys.executable, str(SNAPSHOT_SCRIPT), "--list-modules"],
        capture_output=True,
        text=True,
        check=True,
        cwd=str(REPO_ROOT),
    )
    return json.loads(result.stdout)


def _runInFreshInterpreter(code: str) -> tuple[int, str]:
    """Run ``code`` in a fresh interpreter and return its exit status and last stderr line.

    Parameters
    ----------
    code
        The Python source to execute.

    Returns
    -------
    tuple[int, str]
        The return code, and the last line of stderr (the exception message, for a traceback).
    """

    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
    )
    lastLine = (result.stderr.strip().splitlines() or [""])[-1]
    return result.returncode, lastLine


def _failuresAcrossModules(codeForModule) -> list[tuple[str, str]]:
    """Run one fresh interpreter per documented module and collect every failure.

    Parameters
    ----------
    codeForModule
        Callable mapping a module name to the source to execute for it.

    Returns
    -------
    list[tuple[str, str]]
        ``(moduleName, errorMessage)`` for each module whose interpreter exited non-zero, sorted.
    """

    names = _documentedModuleNames()
    assert names, "no documented modules were discovered at all -- the discovery must be broken"

    def probe(name):
        returnCode, message = _runInFreshInterpreter(codeForModule(name))
        return name, returnCode, message

    with concurrent.futures.ThreadPoolExecutor(max_workers=_MAX_WORKERS) as executor:
        results = list(executor.map(probe, names))

    return sorted((name, message) for name, returnCode, message in results if returnCode != 0)


def _formatFailures(failures: list[tuple[str, str]]) -> str:
    return "\n".join(f"  {name}\n      {message}" for name, message in failures)


@pytest.mark.slow
def test_every_documented_module_imports_standalone():
    """Each documented module must import in an otherwise-untouched interpreter."""

    failures = _failuresAcrossModules(lambda name: f"import {name}\n")

    assert not failures, (
        "these documented modules cannot be imported on their own, so nothing but the input file "
        "parser can reach them:\n" + _formatFailures(failures)
    )


@pytest.mark.slow
def test_no_documented_module_poisons_the_parser_when_imported_first():
    """Importing a documented module before the parser must not break the parser.

    A module that registers grammar at import time can run *before* the keyword it registers on
    exists. If it then caches that failure -- by being in ``sys.modules`` when the parser gets round
    to importing it properly -- the parser breaks for every input, and only in that import order.
    That is not hypothetical: it is what ``ensight`` and ``options`` did until the lazy declaration
    of the shared ``options`` keyword landed.
    """

    failures = _failuresAcrossModules(lambda name: f"import {name}\nimport edelweissfe.utils.inputfileparser\n")

    assert (
        not failures
    ), "importing these documented modules before the parser leaves the input language broken:\n" + _formatFailures(
        failures
    )
