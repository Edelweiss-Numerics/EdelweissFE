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

Both checks run in one **fresh subprocess** per module, which is the only way to test an import-order
property, and the reason this file is slower than the rest of the suite: it imports the module, then
the parser, and reports when the module has imported, so that the two failures are told apart (a crash
included). The subprocesses run concurrently, and a failure reports *every* offending module rather
than only the first, because these bugs come in families with one shared cause -- seeing all of them
together is what points at the cause instead of at a symptom.
"""

import concurrent.futures
import functools
import json
import subprocess
import sys
import tempfile
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


def _runInFreshInterpreter(code: str) -> tuple[int, str, str]:
    """Run ``code`` in a fresh interpreter and return its exit status, its stdout and its last stderr line.

    Parameters
    ----------
    code
        The Python source to execute.

    Returns
    -------
    tuple[int, str, str]
        The return code, the standard output, and the last line of stderr (the exception message, for a
        traceback).
    """

    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        # NOT cwd=REPO_ROOT: `-c` sets sys.path[0] to the (resolved) cwd, and the repo root
        # contains an uncompiled edelweissfe/ source tree (no built Cython extensions -- those
        # only exist in the real, installed site-packages copy under a non-editable `pip install
        # .`). That local package would shadow the installed one and break on the very import
        # this test is trying to exercise. A neutral cwd keeps the import resolving to whatever is
        # actually on sys.path via the installed package, which is the property under test.
        cwd=tempfile.gettempdir(),
    )
    lastLine = (result.stderr.strip().splitlines() or [""])[-1]
    return result.returncode, result.stdout, lastLine


#: What a probe prints once its module has imported: a probe that does not print it failed (or crashed) importing the
#: module on its own.
_IMPORTED = "probe: the module imported on its own"

#: The probe of a module: import it alone -- the first statement of the interpreter --, then the parser.
_PROBE = """
import {module}
print("{imported}", flush=True)
import edelweissfe.utils.inputfileparser
"""


@functools.cache
def _probeEveryModule() -> dict[str, tuple[int, str, str]]:
    """Probe every documented module in a fresh interpreter of its own (see :data:`_PROBE`).

    Returns
    -------
    dict[str, tuple[int, str, str]]
        By module name, what :func:`_runInFreshInterpreter` returned for its probe.
    """

    names = _documentedModuleNames()
    assert names, "no documented modules were discovered at all -- the discovery must be broken"

    def probe(name):
        return _runInFreshInterpreter(_PROBE.format(module=name, imported=_IMPORTED))

    with concurrent.futures.ThreadPoolExecutor(max_workers=_MAX_WORKERS) as executor:
        return dict(zip(names, executor.map(probe, names)))


def _failures(failed) -> list[tuple[str, str]]:
    """``(moduleName, errorMessage)`` for each module whose probe ``failed(returnCode, imported)`` says failed,
    sorted; ``imported`` tells whether the module imported on its own."""

    return sorted(
        (name, message)
        for name, (returnCode, stdout, message) in _probeEveryModule().items()
        if failed(returnCode, _IMPORTED in stdout)
    )


def _formatFailures(failures: list[tuple[str, str]]) -> str:
    return "\n".join(f"  {name}\n      {message}" for name, message in failures)


@pytest.mark.slow
def test_every_documented_module_imports_standalone():
    """Each documented module must import in an otherwise-untouched interpreter."""

    failures = _failures(lambda returnCode, imported: not imported)

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

    failures = _failures(lambda returnCode, imported: returnCode != 0)

    assert (
        not failures
    ), "importing these documented modules before the parser leaves the input language broken:\n" + _formatFailures(
        failures
    )
