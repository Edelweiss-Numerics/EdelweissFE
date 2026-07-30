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

"""``*fieldOutput``: the pluggable-module keyword defining a field output.

Unlike every other top-level keyword, ``*fieldOutput`` declares no line options of its own at all;
its entire grammar is the ``>>perNode``/``>>perElement``/``>>fromExpression`` repeatable sub-keyword
blocks, described by ``edelweissfe.utils.fieldoutput.FieldOutputSchema`` -- the same
repeatable-``>>``-blocks shape as e.g. ``edelweissfe.outputmanagers.ensight.EnsightSchema``.
Construction is unrelated to this schema: ``abqmodelconstructor``/``inputfilehelpers`` still build
``_FieldOutputBase`` instances from the raw parsed dict.
"""

from __future__ import annotations

from edelweissfe.keywords.base.keywordbase import KeywordBase
from edelweissfe.utils.fieldoutput import FieldOutputSchema


class FieldOutputKeyword(KeywordBase):
    """``*fieldOutput``: define fieldoutput, which is used by outputmanagers."""

    #: ``*fieldOutput`` declares no line options of its own; its grammar is entirely the three
    #: repeatable ``>>`` blocks declared on this schema -- see the module docstring.
    schema = FieldOutputSchema

    keywordName = "fieldOutput"
    keywordDescription = "define fieldoutput, which is used by outputmanagers"
