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

"""``*modelModifier``: the name-dispatched keyword defining a model modifier (see
``PLAN_INPUT_SYSTEM_UNIFICATION.md``, U2b).

Verbatim transcription of ``inputLanguage.addKeyword("modelModifier", ...)`` in
``edelweissfe/utils/inputfileparser.py:436-439`` -- its own line args and dataline payload only.
The resolved ``type=``-dispatched model-modifier class supplies its own schema for any further
options, out of scope for U2b (see ``edelweissfe.keywords.element`` for the general note on this
phase's scope).
"""

from __future__ import annotations

from dataclasses import dataclass

from edelweissfe.keywords.base.keywordbase import KeywordBase
from edelweissfe.utils.schema import datalineField, schemaField


@dataclass(frozen=True)
class ModelModifierSchema:
    """L2: the options and dataline payload of the ``*modelModifier`` keyword.

    ``modelModifierType`` answers to the input-file option ``type``; a dataclass field literally
    called ``type`` would shadow the builtin, which this project's conventions avoid (see
    ``edelweissfe.keywords.element.ElementSchema.elementType`` for the identical precedent).
    """

    modelModifierType: str | None = schemaField(
        description="model modifier type", dtype=str, default=None, required=True, optionName="type"
    )
    datalines: list | None = datalineField(description="definition of the model modifier", required=True)
    name: str | None = schemaField(description="name of the model modifier", dtype=str, default=None)


class ModelModifierKeyword(KeywordBase):
    """``*modelModifier``: define a model modifier."""

    #: L2 schema declared for the L3 registry, per OptionSchemaProvider.
    schema = ModelModifierSchema

    keywordName = "modelModifier"
    keywordDescription = "define a model modifier"
