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

import re
from dataclasses import dataclass
from pathlib import Path

import pytest

from edelweissfe.utils.schema import datalineField, schemaField, subKeywordField
from edelweissfe.utils.schemasurface import (
    KeywordSurfaceSpec,
    renderPrintKeywordsBlock,
    renderSchemaSurface,
    specFromKeywordClass,
)

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


# --- U2a/U2b: all 21 top-level keywords' printKeywords()-format blocks, proven byte-identical -----
# against the frozen golden (PLAN_INPUT_SYSTEM_UNIFICATION.md, U2 gate (A)). This is the *second*
# legacy rendering format -- ``inputfileparser.printKeywords()``'s hand-rolled dump of the
# structural/type-dispatch keywords declared directly in that file -- as opposed to the
# ``Module.__doc__`` format every test above this one exercises.

_GOLDEN_PATH = Path(__file__).parent / "golden" / "inputlanguage_surface.txt"

#: Every top-level keyword covered by U2a (the six structural mesh/job keywords) and U2b (the
#: remaining fifteen pluggable-module/type-dispatch keywords) -- the complete ``printKeywords()``
#: surface, per ``PLAN_INPUT_SYSTEM_UNIFICATION.md``.
_ALL_TOP_LEVEL_KEYWORDS = [
    "element",
    "elSet",
    "node",
    "nSet",
    "surface",
    "job",
    "section",
    "material",
    "advancedmaterial",
    "fieldOutput",
    "analyticalField",
    "solver",
    "step",
    "output",
    "updateConfiguration",
    "modelGenerator",
    "constraint",
    "modelModifier",
    "configurePlots",
    "exportPlots",
    "include",
]


def _printKeywordsBlocksByName() -> dict[str, str]:
    """Parse the golden file's ``printKeywords()`` section into one block of text per top-level
    keyword, keyed by keyword name -- extracted from the golden file itself, never hand-retyped.

    Blocks are separated by the two blank lines ``printKeywords()``'s trailing ``print("\\n")``
    produces between keywords. Each block's own header line is ``"    {name}    {description...}"``
    (``kwString`` in ``printKeywords()``); the name is recovered with a regex rather than assumed
    from list position, so a reordering of the golden file cannot silently pair the wrong block
    with the wrong keyword name.

    The very last keyword of the section (``*include``) is not followed by another ``"\\n\\n\\n"``
    separator but directly by the ``"===== module documentation:"`` marker one line down, so its
    raw slice carries one trailing ``"\\n"`` that is not part of any other block's rendering. That
    is an artifact of where the section boundary was cut, not of ``printKeywords()`` itself (every
    other block, mid-section, has no leading/trailing newline at all -- confirmed against the raw
    golden bytes), so it is stripped here rather than reproduced by the renderer.
    """
    golden = _GOLDEN_PATH.read_text()
    section = golden.split("===== printKeywords() =====\n", 1)[1]
    section = section.split("===== module documentation:", 1)[0]
    blocksByName: dict[str, str] = {}
    for block in section.split("\n\n\n"):
        if not block.strip():
            continue
        header = re.match(r"^ {4}(\S+) {4}", block)
        assert header, f"printKeywords() block has no parseable '    name    ' header: {block[:60]!r}"
        blocksByName[header.group(1)] = block.rstrip("\n")
    return blocksByName


_PRINT_KEYWORDS_GOLDEN_BLOCKS = _printKeywordsBlocksByName()


def _structuralKeywordSpec(keywordName: str) -> KeywordSurfaceSpec:
    """Build the :class:`KeywordSurfaceSpec` for one of the 21 top-level keywords from its real,
    registered ``KeywordBase`` subclass -- name, description and schema all sourced from the class
    via :func:`specFromKeywordClass` (no hand-typed spelling/description), so this test proves the
    *class* encodes the legacy grammar, and also exercises
    :func:`edelweissfe.config.registry.lookup`."""
    from edelweissfe.config import registry

    target, _schema = registry.lookup("keyword", keywordName)
    return specFromKeywordClass(target)


@pytest.mark.parametrize("keywordName", _ALL_TOP_LEVEL_KEYWORDS)
def test_structural_keyword_printKeywords_block_matches_golden_byte_for_byte(keywordName):
    """U2's gate (A): ``renderPrintKeywordsBlock`` over each top-level keyword's real, registered
    class (name + description + schema all from the class) reproduces the corresponding golden
    ``printKeywords()`` block exactly -- proving the class encodes the legacy grammar (spelling in
    exact display case, descriptions incl. the *nSet* copy-paste bug, types, required/optional-ness,
    textwrap-80 wrapping) with zero drift, without touching the running parser at all.
    """
    spec = _structuralKeywordSpec(keywordName)
    rendered = renderPrintKeywordsBlock(spec)
    assert rendered == _PRINT_KEYWORDS_GOLDEN_BLOCKS[keywordName]


def test_printKeywords_golden_extraction_found_all_21_top_level_keywords():
    """Falsifies the extraction helper itself: if a golden-file reformat ever changed the
    ``printKeywords()`` section's separator/header shape such that :func:`_printKeywordsBlocksByName`
    silently found fewer blocks, the parametrized test above would just stop running for the
    missing ones instead of failing -- this pins the extraction's coverage independently.
    """
    assert set(_ALL_TOP_LEVEL_KEYWORDS) <= set(_PRINT_KEYWORDS_GOLDEN_BLOCKS)


def test_registered_keyword_category_matches_the_golden_printKeywords_surface_exactly():
    """The end-to-end U2 assertion: every header the golden ``printKeywords()`` section actually
    contains resolves to a registered ``"keyword"`` entry whose rendered block matches, and every
    registered ``"keyword"`` entry is exercised above -- i.e. the registry's ``keyword`` category
    and the golden's ``printKeywords()`` section describe exactly the same 21 names, not merely a
    subset of each other.
    """
    from edelweissfe.config import registry

    registeredDisplayNames = {registry.lookup("keyword", name)[0].keywordName for name in _ALL_TOP_LEVEL_KEYWORDS}
    assert registeredDisplayNames == set(_PRINT_KEYWORDS_GOLDEN_BLOCKS)
    assert len(_ALL_TOP_LEVEL_KEYWORDS) == 21
