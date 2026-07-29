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
Renders EdelweissFE's current input-language surface to stdout.

Deliberately NOT named ``test_*.py`` so pytest does not collect it as a test module: it must run
as a **fresh interpreter process** (see ``test_inputlanguage_golden.py``), because the whole
point it demonstrates -- that ``InputLanguage`` registration is import-order dependent
(``if keyword in inputLanguage:`` silently no-ops if a module is imported before its parent
keyword exists, see PLAN_INPUT_SYSTEM.md) -- means the rendered surface is only deterministic
for a clean import, i.e. exactly what a real simulation run or the Sphinx doc build does. Running
it in-process inside the shared pytest session would make its result depend on which other test
modules happened to import which ``edelweissfe`` submodules first -- a form of test pollution
that would make the golden comparison flaky rather than a real regression signal.
"""

import contextlib
import importlib
import io
import pkgutil

import edelweissfe
from edelweissfe.utils.inputlanguage import InputLanguage


def _discoverModulesWithDocumentation() -> list[tuple[str, object]]:
    """Import every submodule of ``edelweissfe`` and collect those exposing a module-level
    ``documentation`` attribute (list-of-keywords style, or the legacy dict style used by
    ``outputmanagers.meshplot`` and ``analyticalfields.mapped``).

    Modules that fail to import (e.g. an optional Marmot-backed Cython extension that was not
    built) are silently skipped, mirroring how ``printDocumentation`` already tolerates a
    ``ModuleNotFoundError``.
    """
    # Force the full grammar to be built, exactly as the Sphinx build does
    # (doc/source/conf.py:36-38), so `if keyword in inputLanguage:` registration guards elsewhere
    # succeed instead of silently no-opping.
    InputLanguage().ensureParserLoaded()

    # microstructuregenerator is documented but -- per the analysis in PLAN_INPUT_SYSTEM.md -- is
    # never imported by the normal parser import order, so ensureParserLoaded() alone will not
    # reach it. Import it explicitly so this snapshot still captures its grammar surface.
    importlib.import_module("edelweissfe.generators.microstructuregenerator")

    discovered = []
    for moduleInfo in pkgutil.walk_packages(edelweissfe.__path__, prefix="edelweissfe."):
        try:
            mod = importlib.import_module(moduleInfo.name)
        except Exception:
            continue
        documentation = mod.__dict__.get("documentation")
        if documentation is not None:
            discovered.append((moduleInfo.name, mod))

    return sorted(discovered, key=lambda pair: pair[0])


def _renderDocumentation(documentation) -> str:
    if isinstance(documentation, dict):
        # legacy dict-style documentation: {optionName: description}
        return "\n".join(f"  {key}: {documentation[key]}" for key in sorted(documentation))
    return "\n".join(str(item.__doc__()) for item in documentation)


def renderCurrentInputLanguageSurface() -> str:
    from edelweissfe.utils.inputfileparser import printKeywords

    parts = []

    keywordsOutput = io.StringIO()
    with contextlib.redirect_stdout(keywordsOutput):
        printKeywords()
    parts.append("===== printKeywords() =====")
    parts.append(keywordsOutput.getvalue().rstrip("\n"))

    for name, mod in _discoverModulesWithDocumentation():
        parts.append(f"===== module documentation: {name} =====")
        if mod.__doc__:
            parts.append(mod.__doc__.strip())
        parts.append(_renderDocumentation(mod.documentation))

    # Strip per-line trailing whitespace. printKeywords() pads its columns via textwrap, so
    # description-less arguments render with trailing spaces. Without this normalisation the
    # `trailing-whitespace` pre-commit hook rewrites the committed golden file on every commit,
    # which would make this comparison fail permanently.
    rendered = "\n".join(parts)

    return "\n".join(line.rstrip() for line in rendered.split("\n")) + "\n"


if __name__ == "__main__":
    import sys

    if "--list-modules" in sys.argv:
        # The names only, as JSON, for tests/test_module_import_independence.py. It shares this
        # discovery rather than repeating it, so the two cannot disagree about what "a documented
        # module" is -- if the golden surface covers a module, the import gate must cover it too.
        import json

        print(json.dumps([name for name, _ in _discoverModulesWithDocumentation()]))
    else:
        print(renderCurrentInputLanguageSurface(), end="")
