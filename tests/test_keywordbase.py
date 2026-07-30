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
"""U1 tests (see ``PLAN_INPUT_SYSTEM_UNIFICATION.md``) for
``edelweissfe/keywords/base/keywordbase.py``.

Nothing here is wired into the running parser or the registry's ``"keyword"`` category -- U1 is
purely additive. These tests only pin the shape :class:`KeywordBase` itself must offer to a future
concrete subclass (U2): the abstract ``fromKeywordDefinition`` seam, and that it is discoverable as
an :class:`~edelweissfe.utils.schema.OptionSchemaProvider`, mirroring
``tests/test_registry.py``'s and ``test_schema.py``'s coverage of the equivalent property for
``GeneratorBase``/``StepActionBase``.
"""

from dataclasses import dataclass

import pytest

from edelweissfe.keywords.base.keywordbase import KeywordBase
from edelweissfe.utils.inputcontext import InputContext
from edelweissfe.utils.schema import OptionSchemaProvider, schemaField, schemaOf


def test_keywordbase_cannot_be_instantiated_directly():
    """``fromKeywordDefinition`` is declared abstract -- a bare ``KeywordBase()`` (or a subclass
    that forgets to override it) must be rejected by the ABC machinery, not silently constructible
    with a no-op seam."""
    with pytest.raises(TypeError, match="abstract"):
        KeywordBase()


def test_a_subclass_that_does_not_override_fromKeywordDefinition_stays_abstract():
    class _IncompleteKeyword(KeywordBase):
        pass

    with pytest.raises(TypeError, match="abstract"):
        _IncompleteKeyword()


def test_keywordbase_is_an_option_schema_provider_with_no_schema_by_default():
    """The declared default of ``None`` is what lets :func:`schemaOf` avoid ``hasattr`` probing --
    see ``OptionSchemaProvider``'s own docstring."""
    assert issubclass(KeywordBase, OptionSchemaProvider)
    assert KeywordBase.schema is None


@dataclass(frozen=True)
class _FindClosestNodeLikeSchema:
    location: str = schemaField(description="Query point.", dtype=str, required=True, default="unset")


class _FindClosestNodeLikeKeyword(KeywordBase):
    """A minimal structural-keyword-shaped subclass: mutates nothing here (no model is threaded
    through this fixture), just proves the seam is callable and returns whatever the override
    decides to -- ``None`` for a structural keyword, per the base class's own contract."""

    schema = _FindClosestNodeLikeSchema

    @classmethod
    def fromKeywordDefinition(cls, name, definition, context):
        return None


def test_a_concrete_subclass_is_instantiable_and_exposes_its_schema_via_schemaOf():
    keyword = _FindClosestNodeLikeKeyword()
    assert isinstance(keyword, KeywordBase)
    assert schemaOf(_FindClosestNodeLikeKeyword) is _FindClosestNodeLikeSchema


def test_fromKeywordDefinition_is_the_declared_L4_seam_and_may_return_None():
    """Structural keywords (``*element``, ``*node``, ...) mutate ``context.model`` directly and
    return ``None`` -- there is nothing further to hand back to a caller by name (see the base
    class's docstring). This fixture does not touch ``context.model`` (no model is constructed
    here), it only pins that returning ``None`` is a legitimate, type-annotated outcome."""
    context = InputContext(model=None, journal=None)
    result = _FindClosestNodeLikeKeyword.fromKeywordDefinition("myKeyword", {"location": "0,0,0"}, context)
    assert result is None
