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

"""``*advancedmaterial``: defines an advanced material via a provider-specific option set.

Uses the same option grammar as ``*material`` (see ``edelweissfe.keywords.material``).
"""

from __future__ import annotations

from dataclasses import dataclass

from edelweissfe.keywords.base.keywordbase import KeywordBase
from edelweissfe.utils.schema import datalineField, schemaField


@dataclass(frozen=True)
class AdvancedMaterialSchema:
    """The options and dataline payload of the ``*advancedmaterial`` keyword.

    ``materialId`` answers to the input-file option ``id``; a dataclass field literally called
    ``id`` would shadow the builtin, so it is renamed (see also
    ``edelweissfe.keywords.element.ElementSchema.elementType`` for the same convention).
    """

    name: str | None = schemaField(description="name of material", dtype=str, default=None, required=True)
    materialId: str | None = schemaField(
        description="id of material", dtype=str, default=None, required=True, optionName="id"
    )
    provider: str = schemaField(description="material provider", dtype=str, default="marmotmaterial")
    datalines: list | None = datalineField(description="material properties", required=True)


class AdvancedMaterialKeyword(KeywordBase):
    """``*advancedmaterial``: definition of an advanced material."""

    #: Schema declared for the registry, per ``OptionSchemaProvider``.
    schema = AdvancedMaterialSchema

    keywordName = "advancedmaterial"
    keywordDescription = "definition of an advanced material"
