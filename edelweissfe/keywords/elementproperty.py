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

"""``*elementProperty``: the structural keyword overriding a named property on all elements of an
element set (see ``edelweissfe.elements.elementproperty.ElementProperty``, consumed by
``generators.abqmodelconstructor.AbqModelConstructor``).

Verbatim transcription of the legacy ``inputLanguage.addKeyword("elementproperty", ...)`` grammar
-- missed by U2a/U2b's otherwise-exhaustive enumeration of top-level keywords.
"""

from __future__ import annotations

from dataclasses import dataclass

from edelweissfe.keywords.base.keywordbase import KeywordBase
from edelweissfe.utils.schema import datalineField, schemaField


@dataclass(frozen=True)
class ElementPropertySchema:
    """L2: the options and dataline payload of the ``*elementProperty`` keyword."""

    elSet: str | None = schemaField(
        description="element set the property is assigned to", dtype=str, default=None, required=True
    )
    propertyName: str | None = schemaField(
        description="name of the property, as understood by the target elements' assignProperty method",
        dtype=str,
        default=None,
        required=True,
    )
    datalines: list | None = datalineField(description="the property's values", required=True)


class ElementPropertyKeyword(KeywordBase):
    """``*elementProperty``: override a named property on all elements of an element set."""

    #: L2 schema declared for the L3 registry, per OptionSchemaProvider.
    schema = ElementPropertySchema

    keywordName = "elementProperty"
    keywordDescription = "override a named property on all elements of an element set"
