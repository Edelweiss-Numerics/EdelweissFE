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
"""U1 tests (see ``PLAN_INPUT_SYSTEM_UNIFICATION.md``) for ``edelweissfe/utils/schemasurface.py``.

Every expected string here is transcribed verbatim from ``tests/golden/inputlanguage_surface.txt``
(the frozen output of the *legacy* ``Module.__doc__``/``InputFileKeyword.__doc__``/
``OptionalKeywordArg.__doc__`` renderer), for the corresponding real module -- proving
``renderSchemaSurface`` reproduces that exact textual format from a schema alone, with no
dependency on ``edelweissfe.utils.inputlanguage``. U1 does not wire this renderer into anything
running; U2 drives it over the whole grammar and asserts byte-identical output against that golden
file.
"""

from dataclasses import dataclass

from edelweissfe.utils.schema import datalineField, schemaField, subKeywordField
from edelweissfe.utils.schemasurface import KeywordSurfaceSpec, renderSchemaSurface

# --- scalar-only shape (mirrors edelweissfe.generators.findclosestnode) -------------------------


@dataclass(frozen=True)
class _FindClosestNodeSchema:
    location: str = schemaField(description="Query point.", dtype=str, required=True, default="unset")
    storeIn: str = schemaField(
        description="Node set to store closest node in.", dtype=str, required=True, default="unset"
    )


def test_renders_a_flat_required_scalar_schema():
    """Transcribed from the golden ``edelweissfe.generators.findclosestnode`` entry."""
    rendered = renderSchemaSurface(
        [
            KeywordSurfaceSpec(
                name="findclosestnode",
                description="Find the node closest to a given spatial position, and store it in an existing or "
                "new node set.",
                schema=_FindClosestNodeSchema,
            )
        ]
    )
    expected = (
        "[findclosestnode] Find the node closest to a given spatial position, and store it in an existing or "
        "new node set.\n"
        "  required arguments\n"
        "    [location] Query point. (<class 'str'>)\n"
        "    [storeIn] Node set to store closest node in. (<class 'str'>)"
    )
    assert rendered == expected


# --- sub-keyword shape (mirrors edelweissfe.outputmanagers.ensight) ------------------------------


@dataclass(frozen=True)
class _PerNodeSchema:
    fieldOutput: str | None = schemaField(
        description="Name of the result, defined on an elSet (also for perNode results!)",
        dtype=str,
        default=None,
        required=True,
    )


@dataclass(frozen=True)
class _PerElementSchema:
    fieldOutput: str | None = schemaField(
        description="Name of the result, defined on an elSet (also for perNode results!)",
        dtype=str,
        default=None,
        required=True,
    )


@dataclass(frozen=True)
class _ConfigurationSchema:
    overwrite: bool = schemaField(description="Overwrite results.", dtype=bool, default=False)
    intermediateSaveInterval: int = schemaField(description="Set intermediate save interval.", dtype=int, default=10)
    elSet: str | None = schemaField(description="Element set.", dtype=str, default=None)
    nSet: str | None = schemaField(description="Node set.", dtype=str, default=None)
    transient: bool = schemaField(description="Set transient ensight output.", dtype=bool, default=True)


@dataclass(frozen=True)
class _EnsightLikeSchema:
    """Mirrors ``edelweissfe.outputmanagers.ensight.EnsightSchema`` exactly for its three
    sub-keyword blocks (that module's own two extra top-level scalar fields exist only for
    ``>>options`` overrides and are not part of its *keyword-line* grammar -- see its docstring --
    so they are deliberately not reproduced here)."""

    perNode: tuple = subKeywordField(description="Node-based Ensight export.", schema=_PerNodeSchema)
    perElement: tuple = subKeywordField(description="Element-based Ensight export.", schema=_PerElementSchema)
    configurations: tuple = subKeywordField(description="", schema=_ConfigurationSchema, optionName="configuration")


def test_renders_repeatable_sub_keyword_blocks_with_their_own_required_and_optional_arguments():
    """Transcribed verbatim from the golden ``edelweissfe.outputmanagers.ensight`` entry (the
    ``[ensight] ...`` line through its last ``>>configuration`` option) -- proving the nested
    indentation (``indent2`` prepended to *every* line of a sub-block's own rendering, not only
    prefixed once) matches ``Module.__doc__``'s ``optional keywords`` branch exactly.
    """
    rendered = renderSchemaSurface(
        [KeywordSurfaceSpec(name="ensight", description="Ensight export.", schema=_EnsightLikeSchema)]
    )
    expected = (
        "[ensight] Ensight export.\n"
        "  optional keywords\n"
        "    < perNode > Node-based Ensight export.\n"
        "      required arguments\n"
        "        [fieldOutput] Name of the result, defined on an elSet (also for perNode results!) "
        "(<class 'str'>)\n"
        "    < perElement > Element-based Ensight export.\n"
        "      required arguments\n"
        "        [fieldOutput] Name of the result, defined on an elSet (also for perNode results!) "
        "(<class 'str'>)\n"
        "    < configuration >\n"
        "      optional arguments\n"
        "        [overwrite] Overwrite results. (<class 'bool'>, default = False)\n"
        "        [intermediateSaveInterval] Set intermediate save interval. (<class 'int'>, default = 10)\n"
        "        [elSet] Element set. (<class 'str'>, default = None)\n"
        "        [nSet] Node set. (<class 'str'>, default = None)\n"
        "        [transient] Set transient ensight output. (<class 'bool'>, default = True)"
    )
    assert rendered == expected


# --- dataline-only shape (mirrors edelweissfe.generators.executepythoncode) ----------------------


@dataclass(frozen=True)
class _ExecutePythonCodeLikeSchema:
    code: str = datalineField(description="Python code to run", required=True)


def test_renders_a_required_dataline_only_schema():
    """Transcribed from the golden ``edelweissfe.generators.executepythoncode`` entry."""
    rendered = renderSchemaSurface(
        [
            KeywordSurfaceSpec(
                name="executePythoncode",
                description="Directly execute Python code to create the model tree.",
                schema=_ExecutePythonCodeLikeSchema,
            )
        ]
    )
    expected = (
        "[executePythoncode] Directly execute Python code to create the model tree.\n"
        "  required datalines\n"
        "    Python code to run"
    )
    assert rendered == expected


def test_a_dataline_field_is_never_rendered_as_a_scalar_option():
    """A datalineField must not leak into the required/optional *arguments* sections -- it has no
    ``key=value`` spelling on the keyword line at all."""
    rendered = renderSchemaSurface(
        [KeywordSurfaceSpec(name="executePythoncode", description="", schema=_ExecutePythonCodeLikeSchema)]
    )
    assert "arguments" not in rendered
    assert "required datalines" in rendered


# --- combined shape: sub-keywords + required datalines, no top-level scalars ---------------------
# (mirrors edelweissfe.sections.solid, reusing its *real*, already-ported sub-keyword schemas)


def test_renders_sub_keywords_together_with_a_top_level_required_dataline():
    """Transcribed verbatim from the golden ``edelweissfe.sections.solid`` entry. Reuses the real
    ``MaterialParameterFromFieldSchema``/``WriteMaterialPropertiesToFileSchema`` from
    ``edelweissfe.sections.base.sectionbase`` (already-ported L2 schemas, not a synthetic stand-in)
    so this test also pins that ``renderSchemaSurface`` renders an ``optionName`` containing
    parentheses (``f(p,f)``) correctly.
    """
    from edelweissfe.sections.base.sectionbase import (
        MaterialParameterFromFieldSchema,
        WriteMaterialPropertiesToFileSchema,
    )

    @dataclass(frozen=True)
    class _SolidLikeSchema:
        materialParameterFromField: tuple = subKeywordField(
            description="use material properties given by an analytical field",
            schema=MaterialParameterFromFieldSchema,
        )
        writeMaterialPropertiesToFile: tuple = subKeywordField(
            description="export material properties to file",
            schema=WriteMaterialPropertiesToFileSchema,
        )
        elementSets: str = datalineField(
            description="elementSets as comma separated list of element sets for this section", required=True
        )

    rendered = renderSchemaSurface(
        [
            KeywordSurfaceSpec(
                name="solid",
                description="This section represents a classical solid materal section.",
                schema=_SolidLikeSchema,
            )
        ]
    )
    expected = (
        "[solid] This section represents a classical solid materal section.\n"
        "  optional keywords\n"
        "    < materialParameterFromField > use material properties given by an analytical field\n"
        "      required arguments\n"
        "        [index] index of material parameter (<class 'int'>)\n"
        "        [field] name of analytical field (<class 'str'>)\n"
        "        [type] either 'setToValue' or 'scale' (<class 'str'>)\n"
        "      optional arguments\n"
        "        [f(p,f)] p...value of parameter from material definition; f...value of analytical field "
        "(<class 'str'>, default = f)\n"
        "    < writeMaterialPropertiesToFile > export material properties to file\n"
        "      required arguments\n"
        "        [filename] file name for material property export (<class 'str'>)\n"
        "  required datalines\n"
        "    elementSets as comma separated list of element sets for this section"
    )
    assert rendered == expected


# --- multiple top-level keywords are joined with a blank line ------------------------------------


def test_multiple_keyword_specs_are_joined_by_a_blank_line():
    rendered = renderSchemaSurface(
        [
            KeywordSurfaceSpec(name="a", description="First.", schema=None),
            KeywordSurfaceSpec(name="b", description="Second.", schema=None),
        ]
    )
    assert rendered == "[a] First.\n\n[b] Second."


def test_a_schemaless_keyword_renders_only_its_header_line():
    rendered = renderSchemaSurface([KeywordSurfaceSpec(name="fieldOutput", description="", schema=None)])
    assert rendered == "[fieldOutput]"
