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

"""``*section``: the pluggable-module keyword defining a section (see
``PLAN_INPUT_SYSTEM_UNIFICATION.md``, U2b).

Verbatim transcription of ``inputLanguage.addKeyword("section", ...)`` in
``edelweissfe/utils/inputfileparser.py:243-247`` -- its own line args and dataline payload only.
``*section`` is a name-dispatched keyword (``type=plane``/``solid``/...): the hosted module's own
extra args (e.g. ``plane``'s ``thickness``) are declared on that module's own schema, not here --
U2b mirrors only the keyword's own grammar, not the dispatch target's (see
``edelweissfe.keywords.element`` for the general note on this phase's scope).
"""

from __future__ import annotations

from dataclasses import dataclass

from edelweissfe.keywords.base.keywordbase import KeywordBase
from edelweissfe.utils.schema import datalineField, schemaField


@dataclass(frozen=True)
class SectionSchema:
    """L2: the options and dataline payload of the ``*section`` keyword.

    ``sectionType`` answers to the input-file option ``type``; a dataclass field literally called
    ``type`` would shadow the builtin, which this project's conventions avoid (see
    ``edelweissfe.keywords.element.ElementSchema.elementType`` for the identical precedent).
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

    #: L2 schema declared for the L3 registry, per OptionSchemaProvider.
    schema = SectionSchema

    keywordName = "section"
    keywordDescription = "definition of a section"
