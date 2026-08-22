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

"""``*modelModifier``: defines a model modifier, dispatched by ``type`` to a model-modifier class.

The resolved model-modifier class supplies its own schema for any further options beyond this
keyword's own line args and dataline payload.
"""

from __future__ import annotations

from dataclasses import dataclass

from edelweissfe.keywords.base.keywordbase import KeywordBase
from edelweissfe.utils.schema import datalineField, schemaField


@dataclass(frozen=True)
class ModelModifierSchema:
    """Options and dataline payload of the ``*modelModifier`` keyword.

    ``modelModifierType`` corresponds to the input-file option ``type``; the field is not named
    ``type`` to avoid shadowing the Python builtin.
    """

    modelModifierType: str | None = schemaField(
        description="model modifier type", dtype=str, default=None, required=True, optionName="type"
    )
    datalines: list | None = datalineField(description="definition of the model modifier", required=True)
    name: str | None = schemaField(description="name of the model modifier", dtype=str, default=None)


class ModelModifierKeyword(KeywordBase):
    """``*modelModifier``: define a model modifier."""

    #: Schema class describing this keyword's options and dataline payload.
    schema = ModelModifierSchema

    keywordName = "modelModifier"
    keywordDescription = "define a model modifier"
