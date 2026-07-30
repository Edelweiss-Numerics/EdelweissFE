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

"""``*fieldOutput``: the pluggable-module keyword defining a field output (see
``PLAN_INPUT_SYSTEM_UNIFICATION.md``, U2b).

Verbatim transcription of ``inputLanguage.addKeyword("fieldOutput", ...)`` in
``edelweissfe/utils/inputfileparser.py:280`` -- unlike every other keyword ported so far,
``*fieldOutput`` declares **no** ``addRequiredArg``/``addOptionalArg``/``addRequiredDatalines`` call
of its own at all; its entire grammar lives in the hosted ``edelweissfe.utils.fieldoutput`` module
(reached via ``inputLanguage["fieldOutput"].addModule(...)``), which is out of scope for U2b (see
``edelweissfe.keywords.element`` for the general note on this phase's scope: only a keyword's own
line args are mirrored here, never a hosted module's). So this schema is ``None``.
"""

from __future__ import annotations

from edelweissfe.keywords.base.keywordbase import KeywordBase
from edelweissfe.utils.inputcontext import InputContext


class FieldOutputKeyword(KeywordBase):
    """``*fieldOutput``: define fieldoutput, which is used by outputmanagers."""

    #: ``*fieldOutput`` declares no line args of its own -- see the module docstring.
    schema = None

    keywordName = "fieldOutput"
    keywordDescription = "define fieldoutput, which is used by outputmanagers"

    @classmethod
    def fromKeywordDefinition(cls, name: str, definition: dict, context: InputContext) -> "KeywordBase | None":
        """Not yet implemented -- U2b only mirrors the grammar as a schema.

        Raises
        ------
        NotImplementedError
            Always. Construction from a parsed ``*fieldOutput`` definition is wired in U3, once the
            runtime parser is swapped over (see ``PLAN_INPUT_SYSTEM_UNIFICATION.md``, U3).
        """
        raise NotImplementedError("FieldOutputKeyword.fromKeywordDefinition is wired in U3, not U2b.")
