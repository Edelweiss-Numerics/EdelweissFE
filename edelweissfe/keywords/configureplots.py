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

"""``*configurePlots``: the keyword customizing figures and axes (see
``PLAN_INPUT_SYSTEM_UNIFICATION.md``, U2b).

Verbatim transcription of ``inputLanguage.addKeyword("configurePlots", ...)`` in
``edelweissfe/utils/inputfileparser.py:449-450`` -- a required dataline payload only, no line args
at all (like ``*exportPlots``, see ``edelweissfe.keywords.exportplots``). See
``edelweissfe.keywords.element`` for the general note on this phase's scope.
"""

from __future__ import annotations

from dataclasses import dataclass

from edelweissfe.keywords.base.keywordbase import KeywordBase
from edelweissfe.utils.schema import datalineField


@dataclass(frozen=True)
class ConfigurePlotsSchema:
    """L2: the dataline payload of the ``*configurePlots`` keyword. No line args at all."""

    datalines: list | None = datalineField(
        description="key=value pairs for configuration of figures and axes", required=True
    )


class ConfigurePlotsKeyword(KeywordBase):
    """``*configurePlots``: customize the figures and axes."""

    #: L2 schema declared for the L3 registry, per OptionSchemaProvider.
    schema = ConfigurePlotsSchema

    keywordName = "configurePlots"
    keywordDescription = "customize the figures and axes"
