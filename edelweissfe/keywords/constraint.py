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

"""``*constraint``: defines a constraint, dispatched by its ``type`` option.

Declares only its own line options and dataline payload; the resolved ``type=`` constraint class
supplies its own schema for any further options.
"""

from __future__ import annotations

from dataclasses import dataclass

from edelweissfe.keywords.base.keywordbase import KeywordBase
from edelweissfe.utils.schema import datalineField, schemaField


@dataclass(frozen=True)
class ConstraintSchema:
    """The options and dataline payload of the ``*constraint`` keyword.

    ``constraintType`` answers to the input-file option ``type``; a dataclass field literally
    called ``type`` would shadow the builtin, so it is renamed (see also
    ``edelweissfe.keywords.element.ElementSchema.elementType`` for the same convention).
    """

    constraintType: str | None = schemaField(
        description="constraint type", dtype=str, default=None, required=True, optionName="type"
    )
    datalines: list | None = datalineField(description="definition of the constraint", required=True)
    name: str | None = schemaField(description="name of the constraint", dtype=str, default=None)


class ConstraintKeyword(KeywordBase):
    """``*constraint``: define a constraint."""

    #: Schema declared for the registry, per ``OptionSchemaProvider``.
    schema = ConstraintSchema

    keywordName = "constraint"
    keywordDescription = "define a constraint"
