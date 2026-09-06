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

"""``*element``: defines element(s), assigning an element type and (optionally) an element set."""

from __future__ import annotations

from dataclasses import dataclass

from edelweissfe.keywords.base.keywordbase import KeywordBase
from edelweissfe.utils.schema import datalineField, schemaField


@dataclass(frozen=True)
class ElementSchema:
    """The options and dataline payload of the ``*element`` keyword.

    ``elementType`` answers to the input-file option ``type``; a dataclass field literally called
    ``type`` would shadow the builtin, so it is renamed (see also
    ``edelweissfe.sections.base.sectionbase.MaterialParameterFromFieldSchema`` for the same
    convention). It is declared ``required=True`` explicitly, but still given a ``default=None``
    so the schema remains constructible with no arguments.
    """

    elementType: str | None = schemaField(
        description="assign one of the types defined in the elementlibrary",
        dtype=str,
        default=None,
        required=True,
        optionName="type",
    )
    elSet: str | None = schemaField(description="name of elSet to be created", dtype=str, default=None)
    provider: str = schemaField(
        description="provider (library) for the element type. Default: Marmot", dtype=str, default="Marmot"
    )
    datalines: list | None = datalineField(description="Abaqus like element definition lines", required=True)


class ElementKeyword(KeywordBase):
    """``*element``: definition of element(s)."""

    #: Schema declared for the registry, per ``OptionSchemaProvider``.
    schema = ElementSchema

    keywordName = "element"
    keywordDescription = "definition of element(s)"
