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

"""``*surface``: defines a surface set."""

from __future__ import annotations

from dataclasses import dataclass

from edelweissfe.keywords.base.keywordbase import KeywordBase
from edelweissfe.utils.schema import datalineField, schemaField


@dataclass(frozen=True)
class SurfaceSchema:
    """Options and dataline payload of the ``*surface`` keyword.

    ``surfaceType`` corresponds to the input-file option ``type``; the field is not named ``type``
    to avoid shadowing the Python builtin.
    """

    name: str | None = schemaField(description="name", dtype=str, default=None, required=True)
    surfaceType: str | None = schemaField(
        description="type of surface (currently 'element' only)",
        dtype=str,
        default=None,
        required=True,
        optionName="type",
    )
    datalines: list | None = datalineField(
        description="Abaqus like definition. Type 'element': elSet, faceID", required=True
    )


class SurfaceKeyword(KeywordBase):
    """``*surface``: definition of surface set."""

    #: Schema class describing this keyword's options and dataline payload.
    schema = SurfaceSchema

    keywordName = "surface"
    keywordDescription = "definition of surface set"
