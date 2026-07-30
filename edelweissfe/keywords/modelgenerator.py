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

"""``*modelGenerator``: the name-dispatched keyword defining a model generator (see
``PLAN_INPUT_SYSTEM_UNIFICATION.md``, U2b).

Verbatim transcription of ``inputLanguage.addKeyword("modelGenerator", ...)`` in
``edelweissfe/utils/inputfileparser.py:384-392`` -- its own line args only. The legacy
declaration's ``# kw.addRequiredDatalines("keyword arguments", "")`` is commented out, i.e. dead --
this keyword declares no dataline payload of its own. The resolved ``generator=``-dispatched
generator class supplies its own schema for any further options, out of scope for U2b (see
``edelweissfe.keywords.element`` for the general note on this phase's scope).
"""

from __future__ import annotations

from dataclasses import dataclass

from edelweissfe.keywords.base.keywordbase import KeywordBase
from edelweissfe.utils.inputcontext import InputContext
from edelweissfe.utils.schema import schemaField


@dataclass(frozen=True)
class ModelGeneratorSchema:
    """L2: the options of the ``*modelGenerator`` keyword. No dataline payload -- see the module
    docstring."""

    name: str | None = schemaField(description="name of the generator", dtype=str, default=None, required=True)
    generator: str | None = schemaField(description="name of generator module", dtype=str, default=None, required=True)
    executeAfterManualGeneration: bool = schemaField(
        description="Delay the execution of the generator after model generation", dtype=bool, default=False
    )


class ModelGeneratorKeyword(KeywordBase):
    """``*modelGenerator``: define a model generator, loaded from a module."""

    #: L2 schema declared for the L3 registry, per OptionSchemaProvider.
    schema = ModelGeneratorSchema

    keywordName = "modelGenerator"
    keywordDescription = "define a model generator, loaded from a module"

    @classmethod
    def fromKeywordDefinition(cls, name: str, definition: dict, context: InputContext) -> "KeywordBase | None":
        """Not yet implemented -- U2b only mirrors the grammar as a schema.

        Raises
        ------
        NotImplementedError
            Always. Construction from a parsed ``*modelGenerator`` definition is wired in U3, once
            the runtime parser is swapped over (see ``PLAN_INPUT_SYSTEM_UNIFICATION.md``, U3).
        """
        raise NotImplementedError("ModelGeneratorKeyword.fromKeywordDefinition is wired in U3, not U2b.")
