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

"""``*elSet``: defines an element set from explicit element numbers or a generated range."""

from __future__ import annotations

from dataclasses import dataclass

from edelweissfe.keywords.base.keywordbase import KeywordBase
from edelweissfe.utils.schema import datalineField, schemaField


@dataclass(frozen=True)
class ElSetSchema:
    """The options and dataline payload of the ``*elSet`` keyword."""

    elSet: str | None = schemaField(description="name", dtype=str, default=None, required=True)
    generate: bool = schemaField(
        description="set True to generate from data line 1: start-element, end-element, step",
        dtype=bool,
        default=False,
    )
    datalines: list | None = datalineField(description="Abaqus like element set definition lines", required=True)


class ElSetKeyword(KeywordBase):
    """``*elSet``: definition of an element set."""

    #: Schema declared for the registry, per ``OptionSchemaProvider``.
    schema = ElSetSchema

    keywordName = "elSet"
    keywordDescription = "definition of an element set"
