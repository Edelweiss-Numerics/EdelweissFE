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

"""``*updateConfiguration``: the keyword updating a previously defined configuration (see
``PLAN_INPUT_SYSTEM_UNIFICATION.md``, U2b).

Verbatim transcription of ``inputLanguage.addKeyword("updateConfiguration", ...)`` in
``edelweissfe/utils/inputfileparser.py:377-379``. See ``edelweissfe.keywords.element`` for the
general note on this phase's scope (schema only, no runtime wiring).
"""

from __future__ import annotations

from dataclasses import dataclass

from edelweissfe.keywords.base.keywordbase import KeywordBase
from edelweissfe.utils.schema import datalineField, schemaField


@dataclass(frozen=True)
class UpdateConfigurationSchema:
    """L2: the options and dataline payload of the ``*updateConfiguration`` keyword."""

    configuration: str | None = schemaField(
        description="name of configuration to be changed", dtype=str, default=None, required=True
    )
    datalines: list | None = datalineField(description="keyword arguments", required=True)


class UpdateConfigurationKeyword(KeywordBase):
    """``*updateConfiguration``: update a configuration."""

    #: L2 schema declared for the L3 registry, per OptionSchemaProvider.
    schema = UpdateConfigurationSchema

    keywordName = "updateConfiguration"
    keywordDescription = "update a configuration"
