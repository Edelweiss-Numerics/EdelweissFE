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

"""``*node``: the structural keyword defining nodes (see ``PLAN_INPUT_SYSTEM_UNIFICATION.md``,
U2a).

Verbatim transcription of ``inputLanguage.addKeyword("node", ...)`` in
``edelweissfe/utils/inputfileparser.py:221-223``. See ``edelweissfe.keywords.element`` for the
general note on U2a's scope (schema only, no runtime wiring).
"""

from __future__ import annotations

from dataclasses import dataclass

from edelweissfe.keywords.base.keywordbase import KeywordBase
from edelweissfe.utils.inputcontext import InputContext
from edelweissfe.utils.schema import datalineField, schemaField


@dataclass(frozen=True)
class NodeSchema:
    """L2: the options and dataline payload of the ``*node`` keyword."""

    nSet: str | None = schemaField(description="name of nSet to be created", dtype=str, default=None)
    datalines: list | None = datalineField(
        description="Abaqus like node definition lines: label, x, [y], [z]", required=True
    )


class NodeKeyword(KeywordBase):
    """``*node``: definition of nodes."""

    #: L2 schema declared for the L3 registry, per OptionSchemaProvider.
    schema = NodeSchema

    @classmethod
    def fromKeywordDefinition(cls, name: str, definition: dict, context: InputContext) -> "KeywordBase | None":
        """Not yet implemented -- U2a only mirrors the grammar as a schema.

        Raises
        ------
        NotImplementedError
            Always. Construction from a parsed ``*node`` definition is wired in U3, once the
            runtime parser is swapped over (see ``PLAN_INPUT_SYSTEM_UNIFICATION.md``, U3).
        """
        raise NotImplementedError("NodeKeyword.fromKeywordDefinition is wired in U3, not U2a.")
