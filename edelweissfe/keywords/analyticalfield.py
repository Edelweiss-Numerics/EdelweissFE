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

"""``*analyticalField``: defines an analytical field, dispatched by its ``type`` option.

Declares no dataline payload of its own.
"""

from __future__ import annotations

from dataclasses import dataclass

from edelweissfe.keywords.base.keywordbase import KeywordBase
from edelweissfe.utils.schema import schemaField


@dataclass(frozen=True)
class AnalyticalFieldSchema:
    """The options of the ``*analyticalField`` keyword. No dataline payload -- see the module
    docstring.

    ``analyticalFieldType`` answers to the input-file option ``type``; a dataclass field literally
    called ``type`` would shadow the builtin, so it is renamed (see also
    ``edelweissfe.keywords.element.ElementSchema.elementType`` for the same convention).
    """

    name: str | None = schemaField(description="name of analytical field", dtype=str, default=None, required=True)
    analyticalFieldType: str | None = schemaField(
        description="type of analytical field", dtype=str, default=None, required=True, optionName="type"
    )


class AnalyticalFieldKeyword(KeywordBase):
    """``*analyticalField``: define an analytical field."""

    #: Schema declared for the registry, per ``OptionSchemaProvider``.
    schema = AnalyticalFieldSchema

    keywordName = "analyticalField"
    keywordDescription = "define an analytical field"
