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

"""``*step``: defines an analysis step, dispatched by ``type`` (e.g. ``adaptive``,
``adaptiveForExplicitSimulations``).

The dispatched step class's own options, and the large family of ``>>``-declared step actions a
step can host, are not part of this keyword's own schema.
"""

from __future__ import annotations

from dataclasses import dataclass

from edelweissfe.keywords.base.keywordbase import KeywordBase
from edelweissfe.utils.schema import schemaField


@dataclass(frozen=True)
class StepSchema:
    """Options of the ``*step`` keyword. No dataline payload.

    ``stepType`` corresponds to the input-file option ``type``; the field is not named ``type`` to
    avoid shadowing the Python builtin.
    """

    solver: str | None = schemaField(description="solver to be used", dtype=str, default=None, required=True)
    stepType: str = schemaField(description="step type", dtype=str, default="adaptive", optionName="type")


class StepKeyword(KeywordBase):
    """``*step``: define steps."""

    #: Schema class describing this keyword's options.
    schema = StepSchema

    keywordName = "step"
    keywordDescription = "define steps"
