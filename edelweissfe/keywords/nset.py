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

"""``*nSet``: defines a node set.

Note: :attr:`NSetKeyword.keywordDescription` reads "definition of an element set" rather than
"node set" -- a long-standing copy-paste artifact from ``*elSet`` that is kept as-is for backward
compatibility, not a typo introduced here.
"""

from __future__ import annotations

from dataclasses import dataclass

from edelweissfe.keywords.base.keywordbase import KeywordBase
from edelweissfe.utils.schema import datalineField, schemaField


@dataclass(frozen=True)
class NSetSchema:
    """Options and dataline payload of the ``*nSet`` keyword."""

    nSet: str | None = schemaField(description="name", dtype=str, default=None, required=True)
    generate: bool = schemaField(
        description="set True to generate from data line 1: start-node, end-node, step",
        dtype=bool,
        default=False,
    )
    datalines: list | None = datalineField(description="Abaqus like node set definition lines", required=True)


class NSetKeyword(KeywordBase):
    """``*nSet``: definition of an element set.

    See the module docstring: this description is kept as-is from a long-standing copy-paste
    artifact, not a typo introduced here.
    """

    #: Schema class describing this keyword's options and dataline payload.
    schema = NSetSchema

    keywordName = "nSet"
    # Kept as-is (copy-paste artifact from *elSet); see the module docstring.
    keywordDescription = "definition of an element set"
