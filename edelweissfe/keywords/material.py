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

"""``*material``: defines a material.

``provider`` selects the resolver strategy used to construct the material; any provider-specific
options are validated by that provider, not by this schema.
"""

from __future__ import annotations

from dataclasses import dataclass

from edelweissfe.keywords.base.keywordbase import KeywordBase
from edelweissfe.utils.schema import datalineField, schemaField


@dataclass(frozen=True)
class MaterialSchema:
    """Options and dataline payload of the ``*material`` keyword.

    ``materialId`` corresponds to the input-file option ``id``; the field is not named ``id`` to
    avoid shadowing the Python builtin.
    """

    name: str | None = schemaField(description="name of material", dtype=str, default=None, required=True)
    materialId: str | None = schemaField(
        description="id of material", dtype=str, default=None, required=True, optionName="id"
    )
    provider: str = schemaField(description="material provider", dtype=str, default="marmotmaterial")
    datalines: list | None = datalineField(description="material properties", required=True)


class MaterialKeyword(KeywordBase):
    """``*material``: definition of a material."""

    #: Schema class describing this keyword's options and dataline payload.
    schema = MaterialSchema

    keywordName = "material"
    keywordDescription = "definition of a material"
