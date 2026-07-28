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

"""Dump the declared step-action grammar as JSON, for ``tests/test_stepaction_option_coverage.py``.

Runs as a **fresh subprocess** for the same reason ``tests/_inputlanguage_snapshot.py`` does:
``InputLanguage`` registration is import-order dependent, so reading the grammar inside the shared
pytest process would make the result depend on which other test module imported what first.
"""

import json

from edelweissfe.config.registry import _BUILTINS
from edelweissfe.utils.inputlanguage import InputLanguage


def dumpStepActionGrammar() -> dict:
    """Collect every declared keyword of every built-in step action module.

    Returns
    -------
    dict
        Maps the step action module name to a mapping of declared keyword name -> list of declared
        argument names. A module declares its main keyword (named after the module) and possibly an
        ``update<keyword>`` companion, which the parser uses to validate a *partial* re-declaration
        of an already-defined step action in a later step.
    """

    inputLanguage = InputLanguage()
    inputLanguage.ensureParserLoaded()

    moduleNames = sorted(name for category, name in _BUILTINS if category == "stepaction")

    grammar = {}
    for moduleName in moduleNames:
        keywords = {}
        # every step type registers the same keywords, so the first match per name is representative
        for stepModule in inputLanguage["step"].modules:
            for keyword in stepModule.keywords:
                if keyword.name.casefold() in (moduleName, "update" + moduleName):
                    keywords.setdefault(keyword.name, sorted({arg.name for arg in keyword.args}))
        grammar[moduleName] = keywords

    return grammar


if __name__ == "__main__":
    print(json.dumps(dumpStepActionGrammar(), indent=2, sort_keys=True))
