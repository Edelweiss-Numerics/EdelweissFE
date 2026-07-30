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

"""``*output``: the name-dispatched keyword defining an output module (see
``PLAN_INPUT_SYSTEM_UNIFICATION.md``, U2b).

Verbatim transcription of ``inputLanguage.addKeyword("output", ...)`` in
``edelweissfe/utils/inputfileparser.py:355-357`` -- its own line args only. The resolved
``type=``-dispatched output-manager class (``ensight``, ``monitor``, ...) supplies its own schema
for the rest of validation, out of scope for U2b (see ``edelweissfe.keywords.element`` for the
general note on this phase's scope).
"""

from __future__ import annotations

from dataclasses import dataclass

from edelweissfe.keywords.base.keywordbase import KeywordBase
from edelweissfe.utils.schema import schemaField


@dataclass(frozen=True)
class OutputSchema:
    """L2: the options of the ``*output`` keyword. No dataline payload.

    ``outputType`` answers to the input-file option ``type``; a dataclass field literally called
    ``type`` would shadow the builtin, which this project's conventions avoid (see
    ``edelweissfe.keywords.element.ElementSchema.elementType`` for the identical precedent).
    """

    outputType: str | None = schemaField(
        description="output module", dtype=str, default=None, required=True, optionName="type"
    )
    name: str | None = schemaField(description="name of output manager", dtype=str, default=None)


class OutputKeyword(KeywordBase):
    """``*output``: define an output module."""

    #: L2 schema declared for the L3 registry, per OptionSchemaProvider.
    schema = OutputSchema

    keywordName = "output"
    keywordDescription = "define an output module"
