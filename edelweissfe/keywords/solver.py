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

"""``*solver``: defines a solver.

Declares two required options: ``name`` (the solver instance's own name) and ``solver`` (the
solver type, e.g. ``"NIST"``). Its dataline payload is optional, unlike most other keywords.
"""

from __future__ import annotations

from dataclasses import dataclass

from edelweissfe.keywords.base.keywordbase import KeywordBase
from edelweissfe.utils.schema import datalineField, schemaField


@dataclass(frozen=True)
class SolverSchema:
    """Options and dataline payload of the ``*solver`` keyword."""

    name: str | None = schemaField(description="solver name", dtype=str, default=None, required=True)
    solver: str | None = schemaField(description="solver type", dtype=str, default=None, required=True)
    datalines: list | None = datalineField(
        description="define options which are passed to the respective solver instance."
    )


class SolverKeyword(KeywordBase):
    """``*solver``: define a solver."""

    #: Schema class describing this keyword's options and dataline payload.
    schema = SolverSchema

    keywordName = "solver"
    keywordDescription = "define a solver"
