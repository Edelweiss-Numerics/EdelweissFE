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


# --- U2c: the ``Module.__doc__`` "module documentation" sections, over every registry entry that ---
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


def _moduleDocGoldenBodies() -> dict[str, str]:
    """Parse the golden file's "module documentation" sections into one body of text per module,
    keyed by the module's dotted import path -- extracted from the golden file itself, never
    hand-retyped, mirroring :func:`_printKeywordsBlocksByName`.

    A section's raw content (between its own ``===== module documentation: X =====`` marker and
    the next one, or end of file) sometimes carries a leading line or two that is **not** part of
    ``Module.__doc__``'s own rendering at all: ``tests/_inputlanguage_snapshot.py`` additionally
    prints the *Python* module's own docstring (``mod.__doc__.strip()``), when non-empty, directly
    above it -- e.g. ``edelweissfe.generators.findclosestnode``'s golden section starts with its
    ``.py`` file's docstring line before the ``[findclosestnode] ...`` grammar line.
    :func:`renderSchemaSurface` has no access to (and does not reproduce) that Python-level
    docstring -- it only renders from the schema -- so the "body" extracted here starts at the
    first line matching ``^\\[`` (the ``[name] ...`` header ``KeywordSurfaceSpec`` produces),
    discarding everything before it, exactly as the U2c spec's "the lines AFTER the ``[name]``
    header line" phrasing describes. A module documented instead with a ``< name >`` header (the
    step actions) or the legacy dict style (``meshplot``) -- see
    ``_NON_BRACKET_FORMAT_WITH_GOLDEN_SECTION`` -- has no such line, so it is not usable as a golden
    body here at all; those modules are tracked for their (deferred) status only, never compared.
    """
    golden = _GOLDEN_PATH.read_text()
    marker = re.compile(r"===== module documentation: (\S+) =====\n")
    boundaries = list(marker.finditer(golden))
    bodies: dict[str, str] = {}
    for i, m in enumerate(boundaries):
        start = m.end()
        end = boundaries[i + 1].start() if i + 1 < len(boundaries) else len(golden)
        lines = golden[start:end].rstrip("\n").split("\n")
        headerIndex = next((idx for idx, line in enumerate(lines) if re.match(r"^\[\S", line)), None)
        if headerIndex is not None:
            bodies[m.group(1)] = "\n".join(lines[headerIndex + 1 :])
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

#: The 21 module-documentation sections already byte-identical before U2c (measured directly
#: against the golden file, not transcribed from the plan's recon prose -- see the module docstring
#: of ``PLAN_INPUT_SYSTEM_UNIFICATION.md``'s "U2 recon findings + rescope").
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

#: U2c's three closures: the renderer now reproduces ``datalineField`` (closing ``section/plane``
#: and ``section/solid``) and the ``optionsOverrideOnly`` marker excludes ``ensight``'s two
#: ``>>options``-only fields from its module section.
_NEWLY_BYTE_IDENTICAL_MODULES = frozenset(
    {
        "edelweissfe.outputmanagers.ensight",
        "edelweissfe.sections.plane",
        "edelweissfe.sections.solid",
    }
)

_EXPECTED_BYTE_IDENTICAL_MODULES = _PREVIOUSLY_BYTE_IDENTICAL_MODULES | _NEWLY_BYTE_IDENTICAL_MODULES

#: Every module documentation section that HAS a ``[name] ...``-headed golden body (i.e. is a member
#: of :data:`_REGISTRY_SCHEMA_ENTRIES`) but is *not* byte-identical, deliberately deferred to U3 --
#: this is the "documented, deliberately-excluded list" the U2c spec asks for (the 11 constraints and
#: the schema=None modules), so coverage can only grow from here, never shrink silently.
#:
#: The 11 constraints (PLAN_INPUT_SYSTEM_UNIFICATION.md's U2 recon): a structural arg such as
#: `slaveSurface`/`nSet`/`referencePoint` is resolved in `fromConstraintDefinition` (popped from the
#: raw definition) rather than declared on the L2 schema, so the schema under-describes the grammar;
#: 2 of these 11 (`nodetodeformablesurfacepenalty.augmentedLagrange`, `tie.adjust`) additionally show
#: the plan's endorsed `str`->`bool` improvement, whose golden regeneration is bundled with the same
#: U3 commit. `constraints/hangingnode` is the twelfth registered constraint but is NOT in this set
#: -- it is already byte-identical (see `_PREVIOUSLY_BYTE_IDENTICAL_MODULES`).
_DEFERRED_TO_U3 = frozenset(
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

#: The schema=None modules (PLAN_INPUT_SYSTEM_UNIFICATION.md's U2 recon) that additionally have a
#: golden "module documentation" section -- they cannot be rendered at all today, let alone compared,
#: so they are tracked separately from `_DEFERRED_TO_U3` (which is exclusively "has a schema, differs
#: from golden"). Real schemas are added in U3. (`sections/planerandomthickness` and the three
#: `statetransferstrategy` entries are also `schema=None` but have NO golden section at all --
#: they were never `Module`-documented in the legacy grammar -- so they are outside this tracking
#: entirely, not merely deferred.)
_SCHEMA_NONE_WITH_GOLDEN_SECTION = frozenset(
    {
        "edelweissfe.generators.executepythoncode",
        "edelweissfe.stepactions.options",
        "edelweissfe.modelmodifiers.adaptivity.hadaptivity",
    }
)

#: Modules that DO have a real schema and a golden "module documentation:" marker, but whose golden
#: content is not headed by a ``[name] ...`` line at all -- a fundamentally different rendering
#: *shape* than :func:`renderSchemaSurface` produces, discovered while extending this test beyond
#: the categories the U2 recon explicitly measured (``stepaction``, ``outputmanager``). Because
#: :func:`_moduleDocGoldenBodies` only extracts a body for a ``[name] ...``-headed section, none of
#: these ever enter :data:`_REGISTRY_SCHEMA_ENTRIES` in the first place, so they are tracked here
#: rather than in ``_DEFERRED_TO_U3`` (which is exclusively "in `_REGISTRY_SCHEMA_ENTRIES`, differs"):
#:
#: - The 12 step actions (`stepactions/options`, the thirteenth registered one, is `schema=None`
#:   instead -- see above): each golden section documents TWO ``< name >``/``< updateName >``
#:   sub-keyword-style blocks (repeated twice over, a legacy ``documentation = [module, module]``
#:   artifact) rather than one top-level ``[name] ...`` block -- registry-shape/construction-path
#:   rework for U3, not a U2c renderer gap.
#: - ``outputmanagers/meshplot``: golden documents it via the legacy *dict*-style
#:   ``documentation = {...}`` rendering (plain ``key: description`` lines, no ``[``/``<`` header at
#:   all) while its L2 schema declares a scalar ``jobs`` field -- another rendering-shape mismatch.
_NON_BRACKET_FORMAT_WITH_GOLDEN_SECTION = frozenset(
    {
        "edelweissfe.stepactions.bodyforce",
        "edelweissfe.stepactions.changematerialproperty",
        "edelweissfe.stepactions.dirichlet",
        "edelweissfe.stepactions.distributedload",
        "edelweissfe.stepactions.geostatic",
        "edelweissfe.stepactions.indirectcontractioncontrol",
        "edelweissfe.stepactions.indirectcontrol",
        "edelweissfe.stepactions.initializematerial",
        "edelweissfe.stepactions.modelupdate",
        "edelweissfe.stepactions.nodeforces",
        "edelweissfe.stepactions.setfield",
        "edelweissfe.stepactions.setinitialconditions",
        "edelweissfe.outputmanagers.meshplot",
    }
)


def _moduleSectionBody(schema: type) -> str:
    """The lines of :func:`renderSchemaSurface` after its own ``[name] ...`` header line, for a
    single schema -- the "grammar body" the U2c spec's gate compares against the golden body
    extracted by :func:`_moduleDocGoldenBodies`. ``name``/``description`` are placeholders: the
    header line itself is never compared (only the golden extraction's own header line is
    discarded), so what is written here is immaterial.
    """
    rendered = renderSchemaSurface([KeywordSurfaceSpec(name="_", description="_", schema=schema)])
    _, _, body = rendered.partition("\n")
    return body


@pytest.mark.parametrize("modpath", sorted(_EXPECTED_BYTE_IDENTICAL_MODULES))
def test_module_section_matches_golden_byte_for_byte(modpath):
    """U2c's gate (A), extended: every module documentation section not deferred to U3 -- the 21
    already-identical before this phase, plus ``ensight``/``section.plane``/``section.solid`` closed
    by this phase's renderer feature and ``optionsOverrideOnly`` marker -- renders byte-identical to
    its golden grammar body.
    """
    schema = _REGISTRY_SCHEMA_ENTRIES[modpath]
    assert _moduleSectionBody(schema) == _MODULE_DOC_GOLDEN_BODIES[modpath]


def test_module_section_byte_identical_set_is_exactly_previously_21_plus_ensight_plane_solid():
    """The end-to-end U2c assertion the spec's GATE names explicitly: computing byte-identity fresh
    for every qualifying registry entry (not trusting the parametrized list above, which could in
    principle omit an entry) yields exactly ``_EXPECTED_BYTE_IDENTICAL_MODULES`` -- no regression
    among the previous 21, and exactly the three new closures, no more.
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
    with ``schema=None`` today (so U3, not U2c, is where it gains one), and really does have a golden
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
