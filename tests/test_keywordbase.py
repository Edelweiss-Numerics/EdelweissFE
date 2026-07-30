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
"""Tests for ``edelweissfe/keywords/base/keywordbase.py`` (see ``PLAN_INPUT_SYSTEM_UNIFICATION.md``,
U4).

``KeywordBase`` is not an ABC: construction from a parsed ``.inp`` definition happens in the
existing per-category L4 adapters (``abqmodelconstructor``/``inputfilehelpers``/``StepManager``),
not on this class (the ``fromKeywordDefinition`` seam U1-U3 explored was never wired to anything and
was deleted in U4 -- see the plan's "U3d-2 -- SKIPPED" note). ``KeywordBase`` is therefore just an
``OptionSchemaProvider`` that additionally owns a keyword's own spelling/description, mirroring
``tests/test_registry.py``'s and ``test_schema.py``'s coverage of the equivalent property for
``GeneratorBase``/``StepActionBase``.
"""

from dataclasses import dataclass

from edelweissfe.keywords.base.keywordbase import KeywordBase
from edelweissfe.utils.schema import OptionSchemaProvider, schemaField, schemaOf


def test_keywordbase_is_an_option_schema_provider_with_no_schema_by_default():
    """The declared default of ``None`` is what lets :func:`schemaOf` avoid ``hasattr`` probing --
    see ``OptionSchemaProvider``'s own docstring."""
    assert issubclass(KeywordBase, OptionSchemaProvider)
    assert KeywordBase.schema is None


def test_keywordbase_is_directly_instantiable():
    """No abstract seam is left on the base class -- a bare ``KeywordBase()`` is legal, unlike
    ``GeneratorBase``/``StepActionBase``, which still declare a real abstract construction seam."""
    keyword = KeywordBase()
    assert isinstance(keyword, KeywordBase)


@dataclass(frozen=True)
class _FindClosestNodeLikeSchema:
    location: str = schemaField(description="Query point.", dtype=str, required=True, default="unset")


class _FindClosestNodeLikeKeyword(KeywordBase):
    """A minimal concrete subclass: only ``schema``/``keywordName``/``keywordDescription``, exactly
    what every real ``keywords/*.py`` module declares."""

    schema = _FindClosestNodeLikeSchema
    keywordName = "findclosestnode"
    keywordDescription = "Find the node closest to a given spatial position."


def test_a_concrete_subclass_is_instantiable_and_exposes_its_schema_via_schemaOf():
    keyword = _FindClosestNodeLikeKeyword()
    assert isinstance(keyword, KeywordBase)
    assert schemaOf(_FindClosestNodeLikeKeyword) is _FindClosestNodeLikeSchema


def test_a_concrete_subclass_exposes_its_name_and_description():
    assert _FindClosestNodeLikeKeyword.keywordName == "findclosestnode"
    assert _FindClosestNodeLikeKeyword.keywordDescription == "Find the node closest to a given spatial position."
