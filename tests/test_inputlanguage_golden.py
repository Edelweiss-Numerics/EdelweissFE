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
A golden/snapshot capture of EdelweissFE's entire input-language surface -- the
``InputLanguage`` singleton's ``printKeywords()`` dump, plus the rendered documentation of every
module that exposes a module-level ``documentation`` list (the same mechanism
``edelweissfe/utils/printdocumentation.py`` and the Sphinx ``pprint`` directive use) -- that pins
the current surface against drift.

The point of this test is not the specific content of the golden file -- it is that any future
change to the input-language grammar (a renamed keyword, a changed default, a module that stops
registering itself) shows up as a reviewable text diff instead of silently vanishing, which is
exactly the failure mode ``inputlanguage.py``'s silent-no-op registration guard
(``if keyword in inputLanguage:``) currently allows.

The render itself runs in a **fresh subprocess** (``tests/_inputlanguage_snapshot.py``), not
in-process: ``InputLanguage`` registration is import-order dependent (see above), so capturing it
in the shared pytest process would make the result depend on which other test modules happened to
import which ``edelweissfe`` submodules first, standalone, before this test runs -- turning a form
of the very bug being tracked into flaky test pollution instead of a real regression signal. A
fresh interpreter always reaches the grammar the same way a real simulation run or the Sphinx doc
build does.

To regenerate the golden file after an intentional grammar change::

    EDELWEISS_UPDATE_GOLDEN=1 python -m pytest tests/test_inputlanguage_golden.py
"""

import os
import subprocess
import sys
from pathlib import Path

import pytest

GOLDEN_PATH = Path(__file__).parent / "golden" / "inputlanguage_surface.txt"
SNAPSHOT_SCRIPT = Path(__file__).parent / "_inputlanguage_snapshot.py"


def _renderCurrentInputLanguageSurface() -> str:
    result = subprocess.run(
        [sys.executable, str(SNAPSHOT_SCRIPT)],
        capture_output=True,
        text=True,
        check=True,
        cwd=str(Path(__file__).parent.parent),
    )
    return result.stdout


def test_inputlanguage_surface_matches_golden():
    current = _renderCurrentInputLanguageSurface()

    if os.environ.get("EDELWEISS_UPDATE_GOLDEN"):
        GOLDEN_PATH.parent.mkdir(parents=True, exist_ok=True)
        GOLDEN_PATH.write_text(current)
        pytest.skip("Regenerated the golden file because EDELWEISS_UPDATE_GOLDEN was set.")

    if not GOLDEN_PATH.exists():
        pytest.fail(
            f"Golden file missing at {GOLDEN_PATH}. Regenerate with: "
            f"EDELWEISS_UPDATE_GOLDEN=1 python -m pytest {Path(__file__).name}"
        )

    expected = GOLDEN_PATH.read_text()
    assert current == expected, (
        "The input-language surface (printKeywords() + module documentation) changed. "
        "If this is an intentional grammar change, review the diff carefully and then "
        "regenerate with: EDELWEISS_UPDATE_GOLDEN=1 python -m pytest " + Path(__file__).name
    )
