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

"""``*modelGenerator``: defines a model generator, dispatched by name to a generator module.

``*modelGenerator`` declares no dataline payload of its own; the resolved generator class supplies
its own schema for any further options.
"""

from __future__ import annotations

from dataclasses import dataclass

from edelweissfe.keywords.base.keywordbase import KeywordBase
from edelweissfe.utils.schema import schemaField


@dataclass(frozen=True)
class ModelGeneratorSchema:
    """Options of the ``*modelGenerator`` keyword. No dataline payload."""

    name: str | None = schemaField(description="name of the generator", dtype=str, default=None, required=True)
    generator: str | None = schemaField(description="name of generator module", dtype=str, default=None, required=True)
    executeAfterManualGeneration: bool = schemaField(
        description="Delay the execution of the generator after model generation", dtype=bool, default=False
    )


class ModelGeneratorKeyword(KeywordBase):
    """``*modelGenerator``: define a model generator, loaded from a module."""

    #: Schema class describing this keyword's options.
    schema = ModelGeneratorSchema

    keywordName = "modelGenerator"
    keywordDescription = "define a model generator, loaded from a module"
