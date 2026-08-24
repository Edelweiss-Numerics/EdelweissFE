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

"""``*job``: defines an analysis job (domain, start time, name, solver).

Unlike other structural keywords, ``*job`` has no dataline payload.
"""

from __future__ import annotations

from dataclasses import dataclass

from edelweissfe.keywords.base.keywordbase import KeywordBase
from edelweissfe.utils.schema import schemaField


@dataclass(frozen=True)
class JobSchema:
    """Options of the ``*job`` keyword. No dataline payload."""

    domain: str | None = schemaField(
        description="define spatial domain: 1d, 2d, 3d", dtype=str, default=None, required=True
    )
    startTime: float = schemaField(description="(optional) start time of job", dtype=float, default=0.0)
    name: str = schemaField(description="Name of job.", dtype=str, default="defaultJob")
    solver: str = schemaField(description="(deprecated) define the solver to be used", dtype=str, default="NIST")


class JobKeyword(KeywordBase):
    """``*job``: definition of an analysis job."""

    #: Schema class describing this keyword's options.
    schema = JobSchema

    keywordName = "job"
    keywordDescription = "definition of an analysis job"
