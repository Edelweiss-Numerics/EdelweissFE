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

"""``*job``: the structural keyword defining an analysis job (see
``PLAN_INPUT_SYSTEM_UNIFICATION.md``, U2a).

Verbatim transcription of ``inputLanguage.addKeyword("job", ...)`` in
``edelweissfe/utils/inputfileparser.py:305-309``. Unlike the other five structural keywords,
``*job`` declares no ``addRequiredDatalines``/``addOptionalDatalines`` call in the legacy grammar,
so this schema has no :func:`~edelweissfe.utils.schema.datalineField`. See
``edelweissfe.keywords.element`` for the general note on U2a's scope (schema only, no runtime
wiring).
"""

from __future__ import annotations

from dataclasses import dataclass

from edelweissfe.keywords.base.keywordbase import KeywordBase
from edelweissfe.utils.inputcontext import InputContext
from edelweissfe.utils.schema import schemaField


@dataclass(frozen=True)
class JobSchema:
    """L2: the options of the ``*job`` keyword. No dataline payload -- see the module docstring."""

    domain: str | None = schemaField(
        description="define spatial domain: 1d, 2d, 3d", dtype=str, default=None, required=True
    )
    startTime: float = schemaField(description="(optional) start time of job", dtype=float, default=0.0)
    name: str = schemaField(description="Name of job.", dtype=str, default="defaultJob")
    solver: str = schemaField(description="(deprecated) define the solver to be used", dtype=str, default="NIST")


class JobKeyword(KeywordBase):
    """``*job``: definition of an analysis job."""

    #: L2 schema declared for the L3 registry, per OptionSchemaProvider.
    schema = JobSchema

    @classmethod
    def fromKeywordDefinition(cls, name: str, definition: dict, context: InputContext) -> "KeywordBase | None":
        """Not yet implemented -- U2a only mirrors the grammar as a schema.

        Raises
        ------
        NotImplementedError
            Always. Construction from a parsed ``*job`` definition is wired in U3, once the
            runtime parser is swapped over (see ``PLAN_INPUT_SYSTEM_UNIFICATION.md``, U3).
        """
        raise NotImplementedError("JobKeyword.fromKeywordDefinition is wired in U3, not U2a.")
