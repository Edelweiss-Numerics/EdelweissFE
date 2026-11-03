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

"""``*section``: defines a section, dispatched by ``type`` (e.g. ``plane``, ``solid``).

The hosted section module's own extra options (e.g. ``plane``'s ``thickness``) are declared on
that module's own schema, not here.
"""

from __future__ import annotations

from dataclasses import dataclass

from edelweissfe.keywords.base.keywordbase import KeywordBase
from edelweissfe.utils.schema import datalineField, schemaField


@dataclass(frozen=True)
class SectionSchema:
    """Options and dataline payload of the ``*section`` keyword.

    ``sectionType`` corresponds to the input-file option ``type``; the field is not named ``type``
    to avoid shadowing the Python builtin.
    """

    name: str | None = schemaField(description="name", dtype=str, default=None, required=True)
    material: str | None = schemaField(
        description="associated id of defined material", dtype=str, default=None, required=True
    )
    sectionType: str | None = schemaField(
        description="type of the section", dtype=str, default=None, required=True, optionName="type"
    )
    datalines: list | None = datalineField(description="list of associated element sets", required=True)


class SectionKeyword(KeywordBase):
    """``*section``: definition of a section."""

    #: Schema class describing this keyword's options and dataline payload.
    schema = SectionSchema

    keywordName = "section"
    keywordDescription = "definition of a section"
