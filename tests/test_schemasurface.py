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
"""Tests for ``edelweissfe/utils/schemasurface.py``.

Every expected string here is transcribed verbatim from ``tests/golden/inputlanguage_surface.txt``
(the frozen output of the *legacy* ``Module.__doc__``/``InputFileKeyword.__doc__``/
``OptionalKeywordArg.__doc__`` renderer), for the corresponding real module -- proving
``renderSchemaSurface`` reproduces that exact textual format from a schema alone, with no
dependency on ``edelweissfe.utils.inputlanguage``. This renderer is not yet wired into anything
running; a broader battery of tests drives it over the whole grammar and asserts byte-identical
output against that golden file.
"""

import importlib
import re
from dataclasses import dataclass
from pathlib import Path

import pytest

from edelweissfe.utils.schema import datalineField, schemaField, subKeywordField
from edelweissfe.utils.schemasurface import (
    KeywordSurfaceSpec,
    renderDictDocumentation,
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
    ``edelweissfe.sections.base.sectionbase`` (already-ported schemas, not a synthetic stand-in)
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


# --- all 21 top-level keywords' printKeywords()-format blocks, proven byte-identical against -------
# the frozen golden file. This is the *second* legacy rendering format -- ``inputfileparser.printKeywords()``'s
# hand-rolled dump of the structural/type-dispatch keywords declared directly in that file -- as opposed to the
# ``Module.__doc__`` format every test above this one exercises.

_GOLDEN_PATH = Path(__file__).parent / "golden" / "inputlanguage_surface.txt"

#: Every top-level keyword covered by U2a (the six structural mesh/job keywords) and U2b (the
#: remaining fifteen pluggable-module/type-dispatch keywords) -- the complete ``printKeywords()``
#: surface -- plus ``restart``, the first top-level keyword added after that gate.
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
    "restart",
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
    """Build the :class:`KeywordSurfaceSpec` for one of the 22 top-level keywords from its real,
    registered ``KeywordBase`` subclass -- name, description and schema all sourced from the class
    via :func:`specFromKeywordClass` (no hand-typed spelling/description), so this test proves the
    *class* encodes the legacy grammar, and also exercises
    :func:`edelweissfe.config.registry.lookup`."""
    from edelweissfe.config import registry

    target, _schema = registry.lookup("keyword", keywordName)
    return specFromKeywordClass(target)


@pytest.mark.parametrize("keywordName", _ALL_TOP_LEVEL_KEYWORDS)
def test_structural_keyword_printKeywords_block_matches_golden_byte_for_byte(keywordName):
    """``renderPrintKeywordsBlock`` over each top-level keyword's real, registered
    class (name + description + schema all from the class) reproduces the corresponding golden
    ``printKeywords()`` block exactly -- proving the class encodes the legacy grammar (spelling in
    exact display case, descriptions incl. the *nSet* copy-paste bug, types, required/optional-ness,
    textwrap-80 wrapping) with zero drift, without touching the running parser at all.
    """
    spec = _structuralKeywordSpec(keywordName)
    rendered = renderPrintKeywordsBlock(spec)
    assert rendered == _PRINT_KEYWORDS_GOLDEN_BLOCKS[keywordName]


def test_printKeywords_golden_extraction_found_all_22_top_level_keywords():
    """Falsifies the extraction helper itself: if a golden-file reformat ever changed the
    ``printKeywords()`` section's separator/header shape such that :func:`_printKeywordsBlocksByName`
    silently found fewer blocks, the parametrized test above would just stop running for the
    missing ones instead of failing -- this pins the extraction's coverage independently.
    """
    assert set(_ALL_TOP_LEVEL_KEYWORDS) <= set(_PRINT_KEYWORDS_GOLDEN_BLOCKS)


def test_registered_keyword_category_matches_the_golden_printKeywords_surface_exactly():
    """The end-to-end assertion: every header the golden ``printKeywords()`` section actually
    contains resolves to a registered ``"keyword"`` entry whose rendered block matches, and every
    registered ``"keyword"`` entry is exercised above -- i.e. the registry's ``keyword`` category
    and the golden's ``printKeywords()`` section describe exactly the same 22 names, not merely a
    subset of each other.
    """
    from edelweissfe.config import registry

    registeredDisplayNames = {registry.lookup("keyword", name)[0].keywordName for name in _ALL_TOP_LEVEL_KEYWORDS}
    assert registeredDisplayNames == set(_PRINT_KEYWORDS_GOLDEN_BLOCKS)
    assert len(_ALL_TOP_LEVEL_KEYWORDS) == 22


# --- the ``Module.__doc__`` "module documentation" sections, over every registry entry that --------
# declares a real (non-``None``) schema -- as opposed to the ``printKeywords()`` surface above, which
# only ever covered the 21 top-level keywords. This is the *other* legacy rendering format
# (``renderSchemaSurface``, not ``renderPrintKeywordsBlock``), and it is checked across every
# category discovered to carry both a schema and a golden "module documentation:" section --
# ``outputmanager``, ``section``, ``analyticalfield``, ``generator``, ``modelmodifier``,
# ``statetransferstrategy``, plus ``constraint`` and ``stepaction`` (found to qualify too; see
# ``_DEFERRED_TO_U3``/``_NON_BRACKET_FORMAT_WITH_GOLDEN_SECTION`` below for why most of the latter
# two still differ).

_MODULE_SECTION_CATEGORIES = [
    "outputmanager",
    "section",
    "analyticalfield",
    "generator",
    "modelmodifier",
    "statetransferstrategy",
    "constraint",
    "stepaction",
]


def _rawModuleDocGoldenSections() -> dict[str, str]:
    """Parse the golden file's "module documentation" sections into one *raw* body of text per
    module, keyed by the module's dotted import path -- extracted from the golden file itself,
    never hand-retyped, mirroring :func:`_printKeywordsBlocksByName`.

    "Raw" means no assumption about header shape at all: the full text between a module's own
    ``===== module documentation: X =====`` marker and the next one (or end of file), rstripped of
    trailing newlines. :func:`_moduleDocGoldenBodies` narrows this down further, for the
    ``[name] ...``-headed subset only; the step-action (``< name >``-headed, possibly repeated) and
    meshplot (legacy dict-style) sections use this raw form directly instead, since they have no
    ``[name] ...`` line to anchor a header-stripping extraction on.
    """
    golden = _GOLDEN_PATH.read_text()
    marker = re.compile(r"===== module documentation: (\S+) =====\n")
    boundaries = list(marker.finditer(golden))
    sections: dict[str, str] = {}
    for i, m in enumerate(boundaries):
        start = m.end()
        end = boundaries[i + 1].start() if i + 1 < len(boundaries) else len(golden)
        sections[m.group(1)] = golden[start:end].rstrip("\n")
    return sections


_RAW_MODULE_DOC_GOLDEN_SECTIONS = _rawModuleDocGoldenSections()


def _moduleDocGoldenBodies() -> dict[str, str]:
    """Narrow :data:`_RAW_MODULE_DOC_GOLDEN_SECTIONS` to one body of text per ``[name] ...``-headed
    module.

    A section's raw content sometimes carries a leading line or two that is **not** part of
    ``Module.__doc__``'s own rendering at all: ``tests/_inputlanguage_snapshot.py`` additionally
    prints the *Python* module's own docstring (``mod.__doc__.strip()``), when non-empty, directly
    above it -- e.g. ``edelweissfe.generators.findclosestnode``'s golden section starts with its
    ``.py`` file's docstring line before the ``[findclosestnode] ...`` grammar line.
    :func:`renderSchemaSurface` has no access to (and does not reproduce) that Python-level
    docstring -- it only renders from the schema -- so the "body" extracted here starts at the
    first line matching ``^\\[`` (the ``[name] ...`` header ``KeywordSurfaceSpec`` produces),
    discarding everything before it -- i.e. only the lines AFTER the ``[name]``
    header line. A module documented instead with a ``< name >`` header (the
    step actions) or the legacy dict style (``meshplot``) has no such line, so it is not usable as
    a golden body here at all -- see :func:`_stepActionExpectedSection`/
    :func:`_meshplotExpectedSection` below, which compare against
    :data:`_RAW_MODULE_DOC_GOLDEN_SECTIONS` directly instead.
    """
    bodies: dict[str, str] = {}
    for modpath, section in _RAW_MODULE_DOC_GOLDEN_SECTIONS.items():
        lines = section.split("\n")
        headerIndex = next((idx for idx, line in enumerate(lines) if re.match(r"^\[\S", line)), None)
        if headerIndex is not None:
            bodies[modpath] = "\n".join(lines[headerIndex + 1 :])
    return bodies


_MODULE_DOC_GOLDEN_BODIES = _moduleDocGoldenBodies()


def _registrySchemaEntries() -> dict[str, type]:
    """Every dotted module path, across :data:`_MODULE_SECTION_CATEGORIES`, that resolves to a
    registered class declaring a real schema *and* has a ``[name] ...``-headed golden "module
    documentation" section -- the population :func:`renderSchemaSurface` can meaningfully be
    checked against here (a ``schema=None`` entry cannot be rendered at all; a ``< name >``-headed
    entry, e.g. every step action, was never extracted into :data:`_MODULE_DOC_GOLDEN_BODIES`).
    """
    from edelweissfe.config import registry

    entries: dict[str, type] = {}
    for category in _MODULE_SECTION_CATEGORIES:
        for name in registry.availableNames(category):
            target, schema = registry.lookup(category, name)
            if schema is None:
                continue
            modpath = target.__module__
            if modpath in _MODULE_DOC_GOLDEN_BODIES:
                entries[modpath] = schema
    return entries


_REGISTRY_SCHEMA_ENTRIES = _registrySchemaEntries()

#: The 21 module-documentation sections that were already byte-identical (measured directly
#: against the golden file, not transcribed from any planning notes).
_PREVIOUSLY_BYTE_IDENTICAL_MODULES = frozenset(
    {
        "edelweissfe.analyticalfields.fromvtk",
        "edelweissfe.analyticalfields.randomscalar",
        "edelweissfe.analyticalfields.scalarexpression",
        "edelweissfe.constraints.hangingnode",
        "edelweissfe.generators.boxgen",
        "edelweissfe.generators.cubit",
        "edelweissfe.generators.cuboidlatticegenerator",
        "edelweissfe.generators.discreterigidbodygenerator",
        "edelweissfe.generators.findclosestnode",
        "edelweissfe.generators.microstructuregenerator",
        "edelweissfe.generators.pipegen",
        "edelweissfe.generators.planerectquad",
        "edelweissfe.generators.surfaceelementgenerator",
        "edelweissfe.outputmanagers.computetimemonitor",
        "edelweissfe.outputmanagers.conditionalstop",
        "edelweissfe.outputmanagers.fractureenergyintegrator",
        "edelweissfe.outputmanagers.meshdatatofile",
        "edelweissfe.outputmanagers.monitor",
        "edelweissfe.outputmanagers.plotalongpath",
        "edelweissfe.outputmanagers.statusfile",
        "edelweissfe.outputmanagers.timemonitor",
    }
)
assert len(_PREVIOUSLY_BYTE_IDENTICAL_MODULES) == 21

#: Three additional closures: the renderer now reproduces ``datalineField`` (closing ``section/plane``
#: and ``section/solid``) and the ``optionsOverrideOnly`` marker excludes ``ensight``'s two
#: ``>>options``-only fields from its module section.
_NEWLY_BYTE_IDENTICAL_MODULES = frozenset(
    {
        "edelweissfe.outputmanagers.ensight",
        "edelweissfe.sections.plane",
        "edelweissfe.sections.solid",
    }
)

#: The 11 constraints whose structural args (`slaveSurface`/`masterSurface`/`nSet`/
#: `referencePoint`/`constrainedNSet`/`loadNSet`/`rigidBody`) were resolved in
#: `fromConstraintDefinition` (popped from the raw definition) rather than declared on the schema,
#: leaving the schema under-describing the grammar. These constraints now declare them as required
#: schema fields (dtype/order matching the golden) and `fromConstraintDefinition` reads them off the
#: built schema instance instead. 2 of these 11 (`nodetodeformablesurfacepenalty.augmentedLagrange`,
#: `tie.adjust`) also carry a `str`->`bool` improvement, whose golden lines were updated alongside.
#: `constraints/hangingnode` is the twelfth registered constraint but is NOT here -- it
#: was already byte-identical (see `_PREVIOUSLY_BYTE_IDENTICAL_MODULES`).
_U3A_CONSTRAINT_MODULES = frozenset(
    {
        "edelweissfe.constraints.amrtransparencyprobe",
        "edelweissfe.constraints.directionalspringpenalty",
        "edelweissfe.constraints.equalvaluelagrangian",
        "edelweissfe.constraints.equalvaluepenalty",
        "edelweissfe.constraints.linearizedrigidbody",
        "edelweissfe.constraints.nodetodeformablesurfacepenalty",
        "edelweissfe.constraints.nodetodiscreterigidbodypenalty",
        "edelweissfe.constraints.nodetorigidsurfacepenalty",
        "edelweissfe.constraints.penaltyindirectcontrol",
        "edelweissfe.constraints.rigidbody",
        "edelweissfe.constraints.tie",
    }
)
assert len(_U3A_CONSTRAINT_MODULES) == 11

#: hadaptivity gained a real (documentation-only -- construction is untouched, see
#: HAdaptivitySchema's own docstring) schema, reproducing its golden module section byte-for-byte.
_U3C_MODULES = frozenset({"edelweissfe.modelmodifiers.adaptivity.hadaptivity"})

#: The first new schema-bearing module after the U2/U3 gate: the ``*output, type=restart`` output
#: manager writing restart checkpoints.
_RESTART_MODULES = frozenset({"edelweissfe.outputmanagers.restart"})

_EXPECTED_BYTE_IDENTICAL_MODULES = (
    _PREVIOUSLY_BYTE_IDENTICAL_MODULES
    | _NEWLY_BYTE_IDENTICAL_MODULES
    | _U3A_CONSTRAINT_MODULES
    | _U3C_MODULES
    | _RESTART_MODULES
)

#: Every module documentation section that HAS a ``[name] ...``-headed golden body (i.e. is a member
#: of :data:`_REGISTRY_SCHEMA_ENTRIES`) but is *not* byte-identical, deliberately deferred -- the
#: "documented, deliberately-excluded list" so coverage can only grow, never
#: shrink silently. All 11 constraints are now closed (see :data:`_U3A_CONSTRAINT_MODULES`), so this set
#: is now empty; the remaining not-yet-schema-described modules are the ``schema=None`` ones tracked
#: in :data:`_SCHEMA_NONE_WITH_GOLDEN_SECTION` and the ``< name >``-headed ones in
#: :data:`_NON_BRACKET_FORMAT_WITH_GOLDEN_SECTION`, neither of which is a member of
#: :data:`_REGISTRY_SCHEMA_ENTRIES` in the first place.
_DEFERRED_TO_U3 = frozenset()

#: The schema=None modules that additionally have a
#: golden "module documentation" section -- they cannot be rendered at all today, let alone compared,
#: so they are tracked separately from `_DEFERRED_TO_U3` (which is exclusively "has a schema, differs
#: from golden"). Real schemas may be added later. (`sections/planerandomthickness` and the three
#: `statetransferstrategy` entries are also `schema=None` but have NO golden section at all --
#: they were never `Module`-documented in the legacy grammar -- so they are outside this tracking
#: entirely, not merely deferred.)
#:
#: `hadaptivity` is already closed (see :data:`_U3C_MODULES`). `stepactions/options` deliberately stays
#: `schema=None` permanently, not just for now: it is a dispatcher onto another object's schema, not
#: a leaf option consumer (see that module's own docstring) -- there is no schema of its own to add.
_SCHEMA_NONE_WITH_GOLDEN_SECTION = frozenset(
    {
        "edelweissfe.generators.executepythoncode",
        "edelweissfe.stepactions.options",
    }
)

#: Modules that DO have a real schema and a golden "module documentation:" marker, but whose golden
#: content is not headed by a ``[name] ...`` line at all -- a fundamentally different rendering
#: *shape* than :func:`renderSchemaSurface` produces as a single top-level spec, discovered while
#: extending this test beyond the categories originally measured (``stepaction``,
#: ``outputmanager``). Because :func:`_moduleDocGoldenBodies` only extracts a body for a
#: ``[name] ...``-headed section, none of these ever enter :data:`_REGISTRY_SCHEMA_ENTRIES` at all.
#:
#: Both shapes that used to populate this set are now closed: the 12 step actions (`stepactions/options`,
#: the thirteenth registered one, stays
#: `schema=None` -- entangled with the ``>>options`` rework, see
#: :data:`_SCHEMA_NONE_WITH_GOLDEN_SECTION`) render their ``< name >``/``< updateName >`` pair,
#: repeated once per registered step type, via :func:`_stepActionExpectedSection` below; and
#: ``outputmanagers/meshplot``'s legacy dict-style ``documentation = {...}`` renders via
#: :func:`~edelweissfe.utils.schemasurface.renderDictDocumentation`
#: (:func:`_meshplotExpectedSection`). Neither shape fits the single-``[name]``-spec machinery
#: above, so both are verified by their own dedicated tests rather than folded into
#: :data:`_EXPECTED_BYTE_IDENTICAL_MODULES`; this set is kept (empty) to record that no
#: non-bracket-format module remains unaccounted for, following the same "keep it, note why it's
#: empty" convention as :data:`_DEFERRED_TO_U3`.
_NON_BRACKET_FORMAT_WITH_GOLDEN_SECTION = frozenset()


def _moduleSectionBody(schema: type) -> str:
    """The lines of :func:`renderSchemaSurface` after its own ``[name] ...`` header line, for a
    single schema -- the "grammar body" this test compares against the golden body
    extracted by :func:`_moduleDocGoldenBodies`. ``name``/``description`` are placeholders: the
    header line itself is never compared (only the golden extraction's own header line is
    discarded), so what is written here is immaterial.
    """
    rendered = renderSchemaSurface([KeywordSurfaceSpec(name="_", description="_", schema=schema)])
    _, _, body = rendered.partition("\n")
    return body


@pytest.mark.parametrize("modpath", sorted(_EXPECTED_BYTE_IDENTICAL_MODULES))
def test_module_section_matches_golden_byte_for_byte(modpath):
    """Every module documentation section not deferred -- the
    21 already-identical modules, the ``ensight``/``section.plane``/``section.solid`` trio closed
    by the renderer's ``datalineField`` support and ``optionsOverrideOnly`` marker, and the 11 constraints closed by
    adding their structural args to their schemas -- renders byte-identical to its golden grammar body.
    """
    schema = _REGISTRY_SCHEMA_ENTRIES[modpath]
    assert _moduleSectionBody(schema) == _MODULE_DOC_GOLDEN_BODIES[modpath]


def test_module_section_byte_identical_set_is_exactly_the_expected_closed_set():
    """The end-to-end assertion: computing byte-identity
    fresh for every qualifying registry entry (not trusting the parametrized list above, which could
    in principle omit an entry) yields exactly ``_EXPECTED_BYTE_IDENTICAL_MODULES`` -- no regression
    among the previous 21, the three additional closures, and the 11 constraints, and no more.
    """
    matching = {
        modpath
        for modpath, schema in _REGISTRY_SCHEMA_ENTRIES.items()
        if _moduleSectionBody(schema) == _MODULE_DOC_GOLDEN_BODIES[modpath]
    }
    assert matching == _EXPECTED_BYTE_IDENTICAL_MODULES


def test_deferred_and_matching_module_sections_partition_every_schema_bearing_entry():
    """Falsifies both the matching set and ``_DEFERRED_TO_U3`` against drift: every registry entry
    across :data:`_MODULE_SECTION_CATEGORIES` that declares a real schema and has a golden section is
    either byte-identical or explicitly deferred -- never silently unaccounted for. A future module
    gaining a schema (or a golden-format change) that is neither ported to match nor added to
    ``_DEFERRED_TO_U3`` fails here first, before it could fail silently by omission.
    """
    assert set(_REGISTRY_SCHEMA_ENTRIES) == _EXPECTED_BYTE_IDENTICAL_MODULES | _DEFERRED_TO_U3


def test_schema_none_modules_with_a_golden_section_are_tracked_and_unrenderable():
    """Falsifies :data:`_SCHEMA_NONE_WITH_GOLDEN_SECTION`: every entry in it really is registered
    with ``schema=None`` today (so a schema would need to be added for it to gain one), and really does have a golden
    "module documentation" section (otherwise it would belong outside this tracking entirely, like
    ``sections/planerandomthickness``).
    """
    from edelweissfe.config import registry

    golden = _GOLDEN_PATH.read_text()
    for modpath in _SCHEMA_NONE_WITH_GOLDEN_SECTION:
        assert f"===== module documentation: {modpath} =====" in golden
        found = False
        for category in _MODULE_SECTION_CATEGORIES:
            for name in registry.availableNames(category):
                target, schema = registry.lookup(category, name)
                if target.__module__ == modpath:
                    assert schema is None, f"{modpath} now has a schema -- move it out of this set."
                    found = True
        assert found, f"{modpath} is not registered under any of {_MODULE_SECTION_CATEGORIES}."


def test_non_bracket_format_modules_have_a_schema_but_no_extractable_golden_body():
    """Falsifies :data:`_NON_BRACKET_FORMAT_WITH_GOLDEN_SECTION`: every entry in it really is
    registered with a real (non-``None``) schema today -- so it is not miscategorized as
    ``_SCHEMA_NONE_WITH_GOLDEN_SECTION`` -- really does have a golden "module documentation" marker,
    and really is excluded from :func:`_moduleDocGoldenBodies`'s extraction (confirming the "no
    ``[name] ...`` header line" premise that keeps it out of :data:`_REGISTRY_SCHEMA_ENTRIES`
    entirely, rather than showing up there as a spurious "differs" entry).
    """
    from edelweissfe.config import registry

    golden = _GOLDEN_PATH.read_text()
    for modpath in _NON_BRACKET_FORMAT_WITH_GOLDEN_SECTION:
        assert f"===== module documentation: {modpath} =====" in golden
        assert modpath not in _MODULE_DOC_GOLDEN_BODIES
        assert modpath not in _REGISTRY_SCHEMA_ENTRIES
        found = False
        for category in _MODULE_SECTION_CATEGORIES:
            for name in registry.availableNames(category):
                target, schema = registry.lookup(category, name)
                if target.__module__ == modpath:
                    assert schema is not None, f"{modpath} has schema=None -- move it to that set instead."
                    found = True
        assert found, f"{modpath} is not registered under any of {_MODULE_SECTION_CATEGORIES}."


def test_module_doc_golden_extraction_is_not_vacuous():
    """Falsifies :func:`_moduleDocGoldenBodies` itself: if the golden file's header-line format ever
    changed such that the ``^\\[`` regex silently stopped matching, every body would disappear and
    every test above would vacuously pass on empty dicts/sets instead of failing. Pin a lower bound
    (21 previously-identical + at least one differing/deferred entry) so that cannot happen quietly.
    """
    assert len(_MODULE_DOC_GOLDEN_BODIES) >= len(_PREVIOUSLY_BYTE_IDENTICAL_MODULES) + 1
    assert _PREVIOUSLY_BYTE_IDENTICAL_MODULES <= set(_MODULE_DOC_GOLDEN_BODIES)


# --- the 12 step actions' `< name >`/`< updateName >` sections, and meshplot's dict-style section --
# the two non-bracket-format shapes `_NON_BRACKET_FORMAT_WITH_GOLDEN_SECTION` used to
# track. Neither fits `_moduleSectionBody`'s "one `[name]`
# spec, header line discarded" comparison above, so each gets its own reconstruction mirroring
# `tests/_inputlanguage_snapshot.py::renderCurrentInputLanguageSurface`'s own per-module assembly
# (the module's real `__doc__`, if any, then the rendered grammar, joined by one newline) and is
# compared against the *raw*, unstripped golden section from `_RAW_MODULE_DOC_GOLDEN_SECTIONS`.


def _expectedModuleSection(modpath: str, renderedGrammar: str) -> str:
    """Reconstruct the full expected "module documentation" section text for `modpath`.

    Mirrors ``tests/_inputlanguage_snapshot.py::renderCurrentInputLanguageSurface``'s per-module
    assembly exactly: ``mod.__doc__.strip()`` (only if the module actually has one -- see that
    function's own ``if mod.__doc__:`` guard) followed by the rendered grammar, joined by a single
    newline. Reconstructing the *whole* section this way -- rather than guessing how many leading
    lines a docstring spans, the way :func:`_moduleDocGoldenBodies` strips up to a ``[name] ...``
    anchor -- means this helper needs no bracket-shaped header to anchor on, which is exactly why
    it is used for the ``< name >``-headed step actions and the header-less meshplot dict section.

    Parameters
    ----------
    modpath
        The dotted module path, imported to read its real ``__doc__``.
    renderedGrammar
        The grammar rendering to append (from :func:`renderSchemaSurface`/
        :func:`~edelweissfe.utils.schemasurface.renderDictDocumentation`).

    Returns
    -------
    str
        The reconstructed section text, comparable byte-for-byte against
        ``_RAW_MODULE_DOC_GOLDEN_SECTIONS[modpath]``.
    """
    mod = importlib.import_module(modpath)
    parts = []
    if mod.__doc__:
        parts.append(mod.__doc__.strip())
    parts.append(renderedGrammar)
    return "\n".join(parts)


@dataclass(frozen=True)
class _StepActionRenderSpec:
    """Everything needed to reconstruct one step action's expected golden section: the keyword's
    own name/description (transcribed verbatim from the golden ``< name > description`` header --
    step actions carry no ``keywordName``/``keywordDescription`` class attributes the way the
    top-level ``KeywordBase`` subclasses do, so unlike :func:`_structuralKeywordSpec` these
    cannot be read off the class), and -- only for the 3 modules with an ``update<keyword>``
    partial-redeclaration pair -- the same for the update keyword and its documentation-only schema
    attribute name.
    """

    modpath: str
    name: str
    description: str
    schemaAttr: str
    updateName: str | None = None
    updateDescription: str | None = None
    updateSchemaAttr: str | None = None


#: One entry per closed step action (`stepactions/options`, the 13th registered one, is not
#: here -- see :data:`_SCHEMA_NONE_WITH_GOLDEN_SECTION`). Every schema is real code, imported by
#: attribute name from the actual module below, not re-declared here.
_U3B_STEP_ACTION_SPECS = [
    _StepActionRenderSpec(
        "edelweissfe.stepactions.bodyforce",
        "bodyforce",
        "Apply body forces on element sets.",
        "BodyForceSchema",
    ),
    _StepActionRenderSpec(
        "edelweissfe.stepactions.changematerialproperty",
        "changematerialproperty",
        "Stepaction to change material properties.",
        "ChangeMaterialPropertySchema",
    ),
    _StepActionRenderSpec(
        "edelweissfe.stepactions.dirichlet",
        "dirichlet",
        "Standard Dirichlet boundary condition.",
        "DirichletSchema",
        "updateDirichlet",
        "Update a previously defined dirichlet definition.",
        "UpdateDirichletSchema",
    ),
    _StepActionRenderSpec(
        "edelweissfe.stepactions.distributedload",
        "distributedload",
        "Standard distributed load, applied on a surface set.",
        "DistributedLoadSchema",
        "updatedistributedload",
        "Update a previously defined distributedload definition.",
        "UpdateDistributedloadSchema",
    ),
    _StepActionRenderSpec(
        "edelweissfe.stepactions.geostatic",
        "geostatic",
        "Initialize materials to an geostatic stress state.",
        "GeostaticSchema",
    ),
    _StepActionRenderSpec(
        "edelweissfe.stepactions.indirectcontractioncontrol",
        "indirectcontractioncontrol",
        "Indirect (displacement) controller for the NISTArcLength solver using a ring to control "
        "the contraction, e.g., for tunneling simulations.",
        "IndirectContractionControlSchema",
    ),
    _StepActionRenderSpec(
        "edelweissfe.stepactions.indirectcontrol",
        "indirectcontrol",
        "Indirect (displacement) controller for the NISTArcLength solver using a ring to control "
        "the contraction, e.g., for tunneling simulations.",
        "IndirectControlSchema",
    ),
    _StepActionRenderSpec(
        "edelweissfe.stepactions.initializematerial",
        "initializematerial",
        "Standard distributed load, applied on a surface set.",
        "InitializeMaterialSchema",
    ),
    _StepActionRenderSpec(
        "edelweissfe.stepactions.modelupdate",
        "modelupdate",
        "This step action may be used for updating the model at the beginning of a step.",
        "ModelUpdateSchema",
    ),
    _StepActionRenderSpec(
        "edelweissfe.stepactions.nodeforces",
        "nodeforces",
        "Apply node forces on node sets.",
        "NodeForcesSchema",
        "updateNodeforces",
        "Update a previously defined nodeforces definition.",
        "UpdateNodeforcesSchema",
    ),
    _StepActionRenderSpec(
        "edelweissfe.stepactions.setfield",
        "setfield",
        "Set a field (via fieldOutput) to a predefined value.",
        "SetFieldSchema",
    ),
    _StepActionRenderSpec(
        "edelweissfe.stepactions.setinitialconditions",
        "setinitialconditions",
        "Pass initial conditions to elements.",
        "SetInitialConditionsSchema",
    ),
]
assert len(_U3B_STEP_ACTION_SPECS) == 12


#: The number of step types (registry category "step") every step action's keyword is registered
#: under -- and therefore how many times its ``< name >``/``< updateName >`` block pair repeats in
#: the golden section (``for module in modules: ...`` in every step action's now-frozen ``Module``
#: block). Sourced from the registry, not hand-counted, so a third step type would be caught here
#: rather than silently under-repeating every step action's expected section.
def _stepTypeCount() -> int:
    from edelweissfe.config import registry

    return len(registry.availableNames("step"))


def _stepActionRenderedGrammar(spec: _StepActionRenderSpec) -> str:
    """Build the expected ``< name >``/``< updateName >`` rendering for one step action, repeated
    once per registered step type -- see :func:`_stepTypeCount`."""
    mod = importlib.import_module(spec.modpath)
    schema = getattr(mod, spec.schemaAttr)
    updateSchema = getattr(mod, spec.updateSchemaAttr) if spec.updateSchemaAttr else None

    specs = []
    for _ in range(_stepTypeCount()):
        specs.append(KeywordSurfaceSpec(name=spec.name, description=spec.description, schema=schema))
        if updateSchema is not None:
            specs.append(
                KeywordSurfaceSpec(name=spec.updateName, description=spec.updateDescription, schema=updateSchema)
            )
    return renderSchemaSurface(specs, bracket="<", joiner="\n")


@pytest.mark.parametrize("spec", _U3B_STEP_ACTION_SPECS, ids=[s.modpath for s in _U3B_STEP_ACTION_SPECS])
def test_stepaction_module_section_matches_golden_byte_for_byte(spec):
    """Reconstructing the ``< name > .../< updateName > ...``
    section from each step action's real (registry-backed) schema -- and, where one exists, its
    documentation-only ``Update<Keyword>Schema`` companion -- reproduces the golden section
    byte-for-byte, including the python-docstring-then-grammar assembly and the once-per-step-type
    repetition.
    """
    from edelweissfe.config import registry

    registeredSchema = registry.lookup("stepaction", spec.name)[1]
    mod = importlib.import_module(spec.modpath)
    assert registeredSchema is getattr(mod, spec.schemaAttr), (
        f"{spec.modpath}: the registered 'stepaction' schema is not the same object as "
        f"'{spec.schemaAttr}' -- the test would silently validate the wrong class."
    )

    rendered = _stepActionRenderedGrammar(spec)
    assert _expectedModuleSection(spec.modpath, rendered) == _RAW_MODULE_DOC_GOLDEN_SECTIONS[spec.modpath]


def test_stepaction_render_specs_cover_every_registered_stepaction_except_options():
    """Falsifies :data:`_U3B_STEP_ACTION_SPECS` against drift: it names exactly the built-in
    ``stepaction`` table minus ``options`` (still ``schema=None`` and deliberately deferred -- see
    :data:`_SCHEMA_NONE_WITH_GOLDEN_SECTION`), so a 14th step action or a renamed one would be
    caught here rather than silently missing from the parametrized test above.

    Deliberately reads ``registry._BUILTINS`` rather than ``registry.availableNames("stepaction")``:
    ``test_registry.py::test_isRegistered_sees_all_three_registration_mechanisms`` in-process
    ``register()``s a throwaway ``"inprocessaction"`` stepaction into the shared registry singleton
    and never un-registers it, so ``availableNames`` picks up test pollution once that test has run
    in the same session -- the built-in table itself is unaffected by that, and is what this test
    actually means to enumerate.
    """
    from edelweissfe.config import registry

    builtinNames = {name for (category, name) in registry._BUILTINS if category == "stepaction"}
    specNames = {spec.name for spec in _U3B_STEP_ACTION_SPECS}
    assert specNames == builtinNames - {"options"}


def _meshplotExpectedSection() -> str:
    """Build the expected meshplot section: its real ``__doc__`` then its ``documentation`` dict
    rendered via :func:`~edelweissfe.utils.schemasurface.renderDictDocumentation` -- the dict itself
    is untouched (see that module's own docstring on why it stays a placeholder), so this
    reconstructs the section from the exact same object the legacy renderer already used.
    """
    import edelweissfe.outputmanagers.meshplot as meshplot

    return _expectedModuleSection(
        "edelweissfe.outputmanagers.meshplot", renderDictDocumentation(meshplot.documentation)
    )


def test_meshplot_dict_style_section_matches_golden_byte_for_byte():
    """:func:`renderDictDocumentation` over the module's own,
    untouched ``documentation`` dict reproduces the golden section byte-for-byte."""
    assert _meshplotExpectedSection() == _RAW_MODULE_DOC_GOLDEN_SECTIONS["edelweissfe.outputmanagers.meshplot"]


def test_non_bracket_format_golden_section_is_now_fully_covered():
    """End-to-end assertion mirroring
    ``test_deferred_and_matching_module_sections_partition_every_schema_bearing_entry`` for the
    non-bracket-format shapes: every module the (now empty)
    :data:`_NON_BRACKET_FORMAT_WITH_GOLDEN_SECTION` used to list is covered by one of the two
    dedicated checks above, so every entry really was closed rather than the set having been
    emptied by mistake.
    """
    coveredByStepActionTest = {spec.modpath for spec in _U3B_STEP_ACTION_SPECS}
    coveredByMeshplotTest = {"edelweissfe.outputmanagers.meshplot"}
    formerlyNonBracketFormat = coveredByStepActionTest | coveredByMeshplotTest
    assert len(formerlyNonBracketFormat) == 13
    assert _NON_BRACKET_FORMAT_WITH_GOLDEN_SECTION == frozenset()
