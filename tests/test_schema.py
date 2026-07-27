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
"""P1 tests (see PLAN_INPUT_SYSTEM.md) for ``edelweissfe/utils/schema.py``, the L2 primitives.

Covers frozen-ness, field-metadata round-trip, and coercion parity against
``edelweissfe.utils.inputlanguage``'s existing (pre-P1, still-in-production) casting logic for a
representative set of types (``int``, ``float``, ``str``, ``bool``), including the
already-correct-type passthrough case that the legacy code only got right for ``bool`` (via the
P0 ``asBool`` fix) and relied on constructor idempotence for everywhere else.
"""

import dataclasses

import pytest

from edelweissfe.utils.inputlanguage import KeywordArg
from edelweissfe.utils.schema import (
    SchemaFieldMeta,
    buildSchemaFromOptions,
    coerceValue,
    fieldSchemaMeta,
    resolveCaseInsensitiveOptions,
    schemaField,
    schemaFields,
)


@dataclasses.dataclass(frozen=True)
class _SampleSchema:
    """A representative L2 schema mixing a required field and defaulted fields of every
    primitive type used across EdelweissFE's ``.inp`` grammar."""

    name: str = schemaField(description="A required name.", dtype=str, required=True, default="unnamed")
    overwrite: bool = schemaField(description="Overwrite existing output.", dtype=bool, default=False)
    intermediateSaveInterval: int = schemaField(
        description="Save an intermediate output every N increments.", dtype=int, default=10
    )
    scaleFactor: float = schemaField(description="A scale factor.", dtype=float, default=1.0)


# --- frozen-ness -------------------------------------------------------------------------------


def test_schema_instance_is_frozen():
    instance = _SampleSchema(name="foo")
    with pytest.raises(dataclasses.FrozenInstanceError):
        instance.name = "bar"


# --- field metadata round-trip ------------------------------------------------------------------


def test_schema_field_metadata_round_trip():
    meta = schemaFields(_SampleSchema)

    assert meta["name"] == SchemaFieldMeta(description="A required name.", dtype=str, required=True)
    assert meta["overwrite"] == SchemaFieldMeta(description="Overwrite existing output.", dtype=bool, required=False)
    assert meta["intermediateSaveInterval"].dtype is int
    assert meta["intermediateSaveInterval"].required is False
    assert meta["scaleFactor"].dtype is float


def test_required_is_inferred_from_absence_of_a_default_when_not_given_explicitly():
    # A dataclass cannot place a field without a default after one that has a default, so the
    # "genuinely required, no default at all" field must come first:
    @dataclasses.dataclass(frozen=True)
    class _RequiredFirst:
        mandatory: str = schemaField(description="m", dtype=str)
        optional: int = schemaField(description="o", dtype=int, default=1)

    meta = schemaFields(_RequiredFirst)
    assert meta["mandatory"].required is True
    assert meta["optional"].required is False


def test_fieldSchemaMeta_raises_for_a_plain_dataclasses_field_without_schema_metadata():
    @dataclasses.dataclass(frozen=True)
    class _Plain:
        x: int = 0

    (plainField,) = dataclasses.fields(_Plain)
    with pytest.raises(KeyError):
        fieldSchemaMeta(plainField)


# --- coercion: already-correct-type passthrough --------------------------------------------------


@pytest.mark.parametrize(
    "value,dtype,expected",
    [
        (5, int, 5),
        ("5", int, 5),
        (5.5, float, 5.5),
        ("5.5", float, 5.5),
        ("hello", str, "hello"),
        (True, bool, True),
        (False, bool, False),
        ("yes", bool, True),
        ("no", bool, False),
        ("True", bool, True),
        ("False", bool, False),
    ],
)
def test_coerceValue(value, dtype, expected):
    result = coerceValue(value, dtype)
    assert result == expected
    assert isinstance(result, dtype)


def test_coerceValue_raises_ValueError_on_bad_conversion():
    with pytest.raises(ValueError):
        coerceValue("not-a-number", int)


# --- coercion parity with inputlanguage.py's existing casting ------------------------------------


@pytest.mark.parametrize(
    "dtype,rawValue",
    [
        (int, "42"),
        (float, "3.14"),
        (str, "hello"),
        (bool, "yes"),
        (bool, "no"),
        (bool, "true"),
        (bool, "false"),
    ],
)
def test_coerceValue_matches_legacy_KeywordArg_casting_for_string_input(dtype, rawValue):
    """For genuinely string-valued input (the only case the legacy code ever had to handle), the
    new coercion must agree exactly with ``KeywordArg.getValueFromKwargs`` -- P1 factors the
    logic out, it does not change what a given ``.inp`` file produces.
    """
    legacyArg = KeywordArg("value", "a test arg", dtype)
    legacyResult = legacyArg.getValueFromKwargs({"value": rawValue})

    assert coerceValue(rawValue, dtype) == legacyResult


def test_coerceValue_deliberately_diverges_from_legacy_for_an_already_bool_value():
    """The one deliberate divergence: the legacy ``KeywordArg.getValueFromKwargs`` path calls
    ``strtobool()`` unconditionally, which crashes with ``AttributeError`` on a real ``bool``
    (this was P0's bugfix, see ``tests/test_ensight_bugfixes.py``). ``coerceValue`` must not
    reproduce that crash -- it is a bug, not a "surprise" worth preserving for parity.
    """
    legacyArg = KeywordArg("value", "a test arg", bool)
    with pytest.raises(AttributeError):
        legacyArg.getValueFromKwargs({"value": True})

    assert coerceValue(True, bool) is True


# --- case-insensitive key resolution (an L4-facing utility, not part of the schema itself) -------


def test_resolveCaseInsensitiveOptions_matches_regardless_of_case():
    resolved = resolveCaseInsensitiveOptions(_SampleSchema, {"NAME": "foo", "OverWrite": "true"})
    assert resolved == {"name": "foo", "overwrite": "true"}


def test_resolveCaseInsensitiveOptions_raises_with_hint_for_an_unknown_key():
    with pytest.raises(ValueError, match="overwrite"):
        resolveCaseInsensitiveOptions(_SampleSchema, {"overwrit": "true"})


# --- end-to-end schema construction from raw (string, mis-cased) options ------------------------


def test_buildSchemaFromOptions_end_to_end():
    instance = buildSchemaFromOptions(
        _SampleSchema,
        {"NAME": "myOutput", "OVERWRITE": "true", "intermediatesaveinterval": "5", "scalefactor": "2.5"},
    )
    assert instance == _SampleSchema(name="myOutput", overwrite=True, intermediateSaveInterval=5, scaleFactor=2.5)


def test_buildSchemaFromOptions_fills_defaults_for_omitted_optional_fields():
    instance = buildSchemaFromOptions(_SampleSchema, {"name": "onlyName"})
    assert instance.overwrite is False
    assert instance.intermediateSaveInterval == 10
    assert instance.scaleFactor == 1.0


def test_buildSchemaFromOptions_raises_for_missing_required_field():
    @dataclasses.dataclass(frozen=True)
    class _RequiresName:
        name: str = schemaField(description="m", dtype=str, required=True, default="placeholder")

    with pytest.raises(ValueError, match="name"):
        buildSchemaFromOptions(_RequiresName, {})


def test_buildSchemaFromOptions_passes_already_correct_types_through_unchanged():
    """A programmatic caller (e.g. EdelweissMeshfree) may already hold correctly-typed Python
    values rather than strings -- these must not be mangled."""
    instance = buildSchemaFromOptions(
        _SampleSchema,
        {"name": "myOutput", "overwrite": True, "intermediateSaveInterval": 7, "scaleFactor": 3.0},
    )
    assert instance == _SampleSchema(name="myOutput", overwrite=True, intermediateSaveInterval=7, scaleFactor=3.0)
