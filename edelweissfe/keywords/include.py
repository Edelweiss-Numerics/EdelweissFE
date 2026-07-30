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

"""``*include``: the keyword loading the contents of an extra file (see
``PLAN_INPUT_SYSTEM_UNIFICATION.md``, U2b).

Verbatim transcription of ``inputLanguage.addKeyword("include", ...)`` in
``edelweissfe/utils/inputfileparser.py:461-462``. No dataline payload -- the referenced file's
contents are spliced into the parse, not carried as datalines of this keyword itself. See
``edelweissfe.keywords.element`` for the general note on this phase's scope.
"""

from __future__ import annotations

from dataclasses import dataclass

from edelweissfe.keywords.base.keywordbase import KeywordBase
from edelweissfe.utils.inputcontext import InputContext
from edelweissfe.utils.schema import schemaField


@dataclass(frozen=True)
class IncludeSchema:
    """L2: the options of the ``*include`` keyword. No dataline payload -- see the module
    docstring."""

    input: str | None = schemaField(
        description="path to file (use relative path to current .inp)", dtype=str, default=None, required=True
    )


class IncludeKeyword(KeywordBase):
    """``*include``: load contents of extra file."""

    #: L2 schema declared for the L3 registry, per OptionSchemaProvider.
    schema = IncludeSchema

    keywordName = "include"
    keywordDescription = "load contents of extra file"

    @classmethod
    def fromKeywordDefinition(cls, name: str, definition: dict, context: InputContext) -> "KeywordBase | None":
        """Not yet implemented -- U2b only mirrors the grammar as a schema.

        Raises
        ------
        NotImplementedError
            Always. Construction from a parsed ``*include`` definition is wired in U3, once the
            runtime parser is swapped over (see ``PLAN_INPUT_SYSTEM_UNIFICATION.md``, U3).
        """
        raise NotImplementedError("IncludeKeyword.fromKeywordDefinition is wired in U3, not U2b.")
