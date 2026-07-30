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

Sourced entirely from the registry's ``"stepaction"`` category and each module's own L2 schema
(``PLAN_INPUT_SYSTEM_UNIFICATION.md``, U4) -- there is no other grammar representation left to read.
Still runs as a **fresh subprocess**, matching ``tests/_inputlanguage_snapshot.py``: this keeps the
grammar dump isolated from whatever else the shared pytest session happens to have imported, even
though nothing about the registry is import-order dependent any more.
"""

import dataclasses
import json

from edelweissfe.config.registry import _BUILTINS, lookup
from edelweissfe.utils.inputfileparser import _updateStepActionSchema
from edelweissfe.utils.schema import fieldSchemaMeta


def _declaredArgNames(schemaCls: type | None) -> list[str]:
    """Every option name a schema declares on its own line/dataline-less grammar -- scalar fields
    only (no dataline payload, no ``>>`` sub-keyword), including ``structuralOnly``/
    ``optionsOverrideOnly``/``updateOnly`` ones: all of these are names a step action's own code may
    legitimately read from a parsed definition dict, exactly like every other declared option.
    """
    if schemaCls is None:
        return []
    names = []
    for field in dataclasses.fields(schemaCls):
        meta = fieldSchemaMeta(field)
        if meta.isDataline or meta.subSchema is not None:
            continue
        names.append(meta.optionName or field.name)
    return sorted(names)


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

    moduleNames = sorted(name for category, name in _BUILTINS if category == "stepaction")

    grammar = {}
    for moduleName in moduleNames:
        _target, schema = lookup("stepaction", moduleName)
        keywords = {moduleName: _declaredArgNames(schema)}

        updateSchema = _updateStepActionSchema(moduleName)
        if updateSchema is not None:
            updateName = "update" + moduleName
            keywords[updateName] = _declaredArgNames(updateSchema)

        grammar[moduleName] = keywords

    return grammar


if __name__ == "__main__":
    print(json.dumps(dumpStepActionGrammar(), indent=2, sort_keys=True))
