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
    optionNames,
    resolveCaseInsensitiveOptions,
    scalarOptionNames,
    schemaField,
    schemaFields,
    subKeywordField,
    subKeywordFieldNames,
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


def test_option_name_may_differ_from_the_field_name():
    """An input-file option that is not a valid Python identifier -- ``f(x)`` is the real case --
    must still be reachable, without renaming the user-facing option."""

    @dataclasses.dataclass(frozen=True)
    class _Schema:
        f_x: str = schemaField(
            description="A result-transforming expression.", dtype=str, default="x", optionName="f(x)"
        )

    built = buildSchemaFromOptions(_Schema, {"f(x)": "x**2"})
    assert built.f_x == "x**2"

    assert optionNames(_Schema) == {"f(x)": dataclasses.fields(_Schema)[0]}

    # Case-insensitively too, as for any other option, and the field name itself is *not* accepted
    # as an alias -- the schema answers to the declared option name only.
    assert buildSchemaFromOptions(_Schema, {"F(X)": "2*x"}).f_x == "2*x"
    with pytest.raises(ValueError, match="not a valid option"):
        buildSchemaFromOptions(_Schema, {"f_x": "2*x"})


def test_missing_required_option_is_reported_by_its_input_file_name():
    """The diagnostic must name the spelling the user has to type, not the internal field name."""

    @dataclasses.dataclass(frozen=True)
    class _Schema:
        f_x: str = schemaField(description="Required expression.", dtype=str, optionName="f(x)")

    with pytest.raises(ValueError, match=r"f\(x\)"):
        buildSchemaFromOptions(_Schema, {})


def test_an_explicit_none_falls_back_to_the_field_default():
    """The parser spells "user did not specify this optional argument" as a present key with value
    None. Coercing that would turn it into the *string* "None" for a str-typed option -- truthy,
    and silently wrong. See buildSchemaFromOptions' docstring."""

    @dataclasses.dataclass(frozen=True)
    class _Schema:
        export: str | None = schemaField(description="Export filename.", dtype=str, default=None)
        count: int = schemaField(description="A count.", dtype=int, default=7)

    built = buildSchemaFromOptions(_Schema, {"export": None, "count": None})

    assert built.export is None, "an explicit None must not become the string 'None'"
    assert built.count == 7

    # A misspelled option still raises even when its value is None, so the None-tolerance cannot
    # mask a typo.
    with pytest.raises(ValueError, match="not a valid option"):
        buildSchemaFromOptions(_Schema, {"exprot": None})


# --- nested >> sub-keyword blocks ----------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class _BlockSchema:
    """One repeatable sub-keyword block."""

    fieldOutput: str | None = schemaField(description="A field output.", dtype=str, default=None, required=True)


@dataclasses.dataclass(frozen=True)
class _NestedSchema:
    """A schema mixing a scalar option with repeatable and renamed sub-keyword blocks, mirroring
    ensight's ``>>perNode`` / ``>>configuration`` shape."""

    label: str = schemaField(description="A label.", dtype=str, default="unnamed")
    perNode: tuple[_BlockSchema, ...] = subKeywordField(description="Per-node blocks.", schema=_BlockSchema)
    configurations: tuple[_BlockSchema, ...] = subKeywordField(
        description="Configuration blocks.", schema=_BlockSchema, optionName="configuration"
    )


def test_sub_keyword_blocks_are_built_in_file_order():
    built = buildSchemaFromOptions(
        _NestedSchema,
        {"label": "myExport"},
        {"perNode": [{"fieldOutput": "Displacement"}, {"fieldOutput": "Temperature"}]},
    )

    assert built.label == "myExport"
    assert [block.fieldOutput for block in built.perNode] == ["Displacement", "Temperature"]
    assert isinstance(built.perNode, tuple), "must be immutable, so the enclosing frozen schema is"
    assert built.configurations == (), "a block kind that was not supplied defaults to empty"


def test_sub_keyword_blocks_are_matched_case_insensitively_and_by_option_name():
    """The parser stores the sub-keyword name exactly as the user wrote it, so the fold happens
    here. `configurations` answers to the declared option name `configuration`, not its field name.
    """
    built = buildSchemaFromOptions(_NestedSchema, {}, {"PERNODE": [{"fieldOutput": "U"}]})
    assert [block.fieldOutput for block in built.perNode] == ["U"]

    built = buildSchemaFromOptions(_NestedSchema, {}, {"configuration": [{"fieldOutput": "U"}]})
    assert [block.fieldOutput for block in built.configurations] == ["U"]

    with pytest.raises(ValueError, match="not a valid sub-keyword"):
        buildSchemaFromOptions(_NestedSchema, {}, {"configurations": [{"fieldOutput": "U"}]})


def test_a_sub_keyword_is_not_reachable_as_a_scalar_option():
    """Writing `perNode=1` as a plain option must be rejected -- with an error that explains the
    remedy rather than offering `perNode` as a spelling suggestion."""
    with pytest.raises(ValueError, match=r"is a sub-keyword .*>>perNode"):
        buildSchemaFromOptions(_NestedSchema, {"perNode": "1"})


def test_an_unknown_sub_keyword_is_rejected_rather_than_silently_dropped():
    with pytest.raises(ValueError, match="not a valid sub-keyword"):
        buildSchemaFromOptions(_NestedSchema, {}, {"perFace": [{"fieldOutput": "U"}]})


def test_a_validation_error_inside_a_block_names_the_sub_keyword():
    with pytest.raises(ValueError, match=r"In sub-keyword 'perNode'.*fieldOutput"):
        buildSchemaFromOptions(_NestedSchema, {}, {"perNode": [{}]})


def test_a_required_sub_keyword_must_appear_at_least_once():
    @dataclasses.dataclass(frozen=True)
    class _RequiresBlock:
        marker: tuple[_BlockSchema, ...] = subKeywordField(
            description="At least one is required.", schema=_BlockSchema, required=True
        )

    with pytest.raises(ValueError, match="marker"):
        buildSchemaFromOptions(_RequiresBlock, {}, {})

    # An empty list is treated as "not supplied", not as "supplied with zero blocks".
    with pytest.raises(ValueError, match="marker"):
        buildSchemaFromOptions(_RequiresBlock, {}, {"marker": []})

    built = buildSchemaFromOptions(_RequiresBlock, {}, {"marker": [{"fieldOutput": "U"}]})
    assert len(built.marker) == 1


def test_scalar_and_sub_keyword_names_are_partitioned():
    assert list(scalarOptionNames(_NestedSchema)) == ["label"]
    assert list(subKeywordFieldNames(_NestedSchema)) == ["perNode", "configuration"]


# --- ensight: the first real user of subKeywordField ---------------------------------------------


def test_ensight_schema_is_buildable_from_the_parser_shaped_module_options():
    """End-to-end over the real ensight schema, from the shape the ``.inp`` parser produces for::

    *output, type=ensight, name=myExport
    >>perNode,    fieldOutput=Displacement
    >>perElement, fieldOutput=Stress
    >>perElement, fieldOutput=Strain
    >>configuration, overwrite=yes
    """
    from edelweissfe.outputmanagers.ensight import EnsightSchema

    built = buildSchemaFromOptions(
        EnsightSchema,
        {},
        {
            "perNode": [{"fieldOutput": "Displacement"}],
            "perElement": [{"fieldOutput": "Stress"}, {"fieldOutput": "Strain"}],
            "configuration": [
                {"overwrite": "yes", "intermediateSaveInterval": 10, "elSet": None, "nSet": None, "transient": True}
            ],
        },
    )

    assert [b.fieldOutput for b in built.perNode] == ["Displacement"]
    assert [b.fieldOutput for b in built.perElement] == ["Stress", "Strain"]
    assert len(built.configurations) == 1
    assert built.configurations[0].overwrite is True
    # elSet/nSet arrive as explicit Nones from the parser's default-filling and must stay None
    # rather than becoming the string "None".
    assert built.configurations[0].elSet is None
    assert built.configurations[0].nSet is None


def test_ensight_configuration_defaults_are_the_declared_ones():
    """Pins the defaults that apply when no ``>>configuration`` block is given at all -- the branch
    that used to be read out of the ``Module`` declaration, and which no simulation test covers
    (the one input without ``overwrite=`` is not named ``test.inp``, so it is never run)."""
    from edelweissfe.outputmanagers.ensight import EnsightConfigurationSchema

    defaults = EnsightConfigurationSchema()
    assert defaults.overwrite is False, "False means the export directory gets a timestamp suffix"
    assert defaults.intermediateSaveInterval == 10
    assert defaults.transient is True
    assert defaults.elSet is None and defaults.nSet is None


def test_ensight_schema_is_constructible_with_no_arguments_and_no_input_language():
    """The L1 constructor default `configuration=EnsightSchema()` must not require the parser."""
    from edelweissfe.outputmanagers.ensight import EnsightSchema, OutputManager

    assert EnsightSchema() == EnsightSchema(perNode=(), perElement=(), configurations=())
    assert OutputManager.schema is EnsightSchema


def test_parser_bookkeeping_keys_are_stripped_from_sub_keyword_blocks():
    """The parser injects `inputFile` into every `>>` block (and `datalines`/`explicitlySetArgs`
    into some), which are not user-facing options. Forgetting to strip them makes every ensight
    input fail validation, so this is pinned directly."""
    from edelweissfe.utils.misc import withoutParserBookkeeping

    stripped = withoutParserBookkeeping(
        [{"fieldOutput": "U", "inputFile": "/some/path/test.inp", "datalines": [], "explicitlySetArgs": {"x"}}]
    )
    assert stripped == [{"fieldOutput": "U"}]


def test_meshplot_datalines_are_dispatched_to_the_arm_selected_by_create():
    """meshplot aggregates a heterogeneous job list, one or more per dataline, with the arm chosen
    by the `create=` tag -- the `DatalineAggregatingSchema` pattern."""
    from edelweissfe.outputmanagers.meshplot import (
        MeshPlotMeshOnlyJob,
        MeshPlotPerElementJob,
        MeshPlotPerNodeJob,
        MeshPlotSchema,
        MeshPlotXYDataJob,
    )

    configuration = MeshPlotSchema.fromDatalines(
        [
            {"create": "xyData", "x": "time", "y": "RF", "integral": "True"},
            {"create": "perNode", "fieldOutput": "S"},
            {"create": "perElement", "fieldOutput": "alphaP"},
            {"create": "meshOnly", "configuration": "deformed", "warpBy": "U", "scaleFactor": "5"},
        ]
    )

    assert [type(job) for job in configuration.jobs] == [
        MeshPlotXYDataJob,
        MeshPlotPerNodeJob,
        MeshPlotPerElementJob,
        MeshPlotMeshOnlyJob,
    ]
    assert configuration.jobs[0].integral is True
    assert configuration.jobs[3].scaleFactor == 5.0


def test_meshplot_rejects_a_dataline_selecting_two_jobs_at_once():
    """Legacy `updateDefinition` tested for `saveFigure` and `create` independently, so one line
    could emit two jobs -- but only because each arm tolerated the other's options, which is the
    silent-typo tolerance the closed schema removes. No input file does this."""
    from edelweissfe.outputmanagers.meshplot import MeshPlotSchema

    with pytest.raises(ValueError, match="must select a single job"):
        MeshPlotSchema.fromDatalines([{"create": "xyData", "x": "time", "y": "RF", "saveFigure": None}])


def test_meshplot_rejects_an_unknown_create_value_and_a_dataline_selecting_no_job():
    from edelweissfe.outputmanagers.meshplot import MeshPlotSchema

    with pytest.raises(ValueError, match="not a valid 'create' value"):
        MeshPlotSchema.fromDatalines([{"create": "perGaussPoint", "fieldOutput": "S"}])

    with pytest.raises(ValueError, match="must select a job"):
        MeshPlotSchema.fromDatalines([{"figure": "1", "label": "orphan"}])


def test_meshplot_rejects_a_misspelled_option_within_an_arm():
    """The whole point of the closed schema: options the plotter never reads (`legend` for `label`,
    `axpSec` for `axSpec`) used to be silently ignored."""
    from edelweissfe.outputmanagers.meshplot import MeshPlotSchema

    with pytest.raises(ValueError, match="legend"):
        MeshPlotSchema.fromDatalines([{"create": "xyData", "x": "time", "y": "RF", "legend": "ignored"}])


def test_meshplot_schema_is_constructible_programmatically_without_the_parser():
    """The hard requirement: an external caller builds jobs directly, no `.inp` file involved."""
    from edelweissfe.outputmanagers.meshplot import (
        MeshPlotSchema,
        MeshPlotXYDataJob,
        OutputManager,
    )

    configuration = MeshPlotSchema(jobs=(MeshPlotXYDataJob(x="time", y="RF", label="reaction force"),))

    assert MeshPlotSchema() == MeshPlotSchema(jobs=())
    assert OutputManager.schema is MeshPlotSchema
    assert configuration.jobs[0].label == "reaction force"
