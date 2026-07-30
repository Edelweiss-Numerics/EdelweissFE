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

"""``renderSchemaSurface``: render the ``.inp`` grammar surface directly from L2 schemas (see
``PLAN_INPUT_SYSTEM_UNIFICATION.md``, U1/§4 gate (A)).

Today, ``tests/golden/inputlanguage_surface.txt`` is produced by
``edelweissfe.utils.inputlanguage.Module.__doc__``/``InputFileKeyword.__doc__``/
``OptionalKeywordArg.__doc__`` (the bracket-and-indent format every ``documentation = [module]``
module renders through) plus ``inputfileparser.printKeywords()`` (a second, unrelated format for
the hand-declared structural/type keywords). This module reproduces the *first* of those two
formats -- the one every ``KeywordBase``-based module will use once ported -- from a schema
directly, with no dependency on :mod:`edelweissfe.utils.inputlanguage` at all.

U1 builds and unit-tests this renderer against hand-built fixtures only; U2 drives it over the
whole grammar and asserts byte-identical output against the committed golden file. Nothing in this
module is wired into the running parser or into any existing module -- see the phase's "additive,
nothing wired" gate.
"""

from __future__ import annotations

import dataclasses
import textwrap
from dataclasses import dataclass
from typing import Iterable

from edelweissfe.utils.schema import SchemaFieldMeta, datalineFieldMeta, schemaFields

#: Matches ``edelweissfe.utils.inputlanguage.indent1``/``indent2`` exactly -- the two indentation
#: levels the legacy renderer uses (a "required/optional arguments|keywords|datalines" section
#: header, and the entries within it).
_INDENT1 = "  "
_INDENT2 = "    "

#: Matches ``edelweissfe.utils.inputfileparser.printKeywords``'s ``kwString``/``kwDataString``
#: exactly -- the format used for the top-level structural/type-dispatch keywords hand-declared
#: directly in ``inputfileparser.py`` (as opposed to the ``Module.__doc__`` format above, used by
#: every keyword ported to a ``documentation = [module]`` block).
_PRINT_KEYWORDS_HEADER = "    {:}    "
_PRINT_KEYWORDS_ARG = "        {:22}{:20}"

#: Matches ``edelweissfe.utils.misc.typeString``'s ``dtypeMapping`` exactly.
_PRINT_KEYWORDS_TYPE_STRING = {str: "string", bool: "boolean", int: "integer", float: "float"}


@dataclass(frozen=True)
class KeywordSurfaceSpec:
    """One renderable keyword, built directly from an L2 schema -- the input :func:`renderSchemaSurface`
    consumes.

    A top-level ``.inp`` keyword (``*ensight``, ``*section``, ...) is described by one of these; a
    ``>>`` sub-keyword block is described by another, nested inside the enclosing schema via
    :func:`~edelweissfe.utils.schema.subKeywordField` -- :func:`renderSchemaSurface` builds the
    nested spec for each such field itself, from the field's own :class:`SchemaFieldMeta`, so a
    caller never constructs one by hand for a sub-keyword.

    Parameters
    ----------
    name
        The keyword's name, as written after ``*``/``>>`` in an ``.inp`` file.
    description
        Human-readable description of the keyword, rendered on the header line. May be empty (as
        for ensight's ``>>configuration``, which has none), in which case the header line carries
        no trailing text.
    schema
        The L2 schema dataclass describing this keyword's own line options, dataline payload, and
        ``>>`` sub-blocks -- typically a :class:`~edelweissfe.keywords.base.keywordbase.KeywordBase`
        subclass's :attr:`~edelweissfe.utils.schema.OptionSchemaProvider.schema`. ``None`` renders
        just the header line (a keyword that declares no schema at all).
    """

    name: str
    description: str
    schema: type | None = None


def specFromKeywordClass(keywordClass: type) -> KeywordSurfaceSpec:
    """Build the :class:`KeywordSurfaceSpec` for a top-level keyword from its
    :class:`~edelweissfe.keywords.base.keywordbase.KeywordBase` subclass.

    The keyword's spelling (:attr:`~edelweissfe.keywords.base.keywordbase.KeywordBase.keywordName`,
    in exact display case), its description, and its schema all come from the class -- the single
    source of truth -- so the grammar surface, the U3 parser, and any test all agree by construction
    rather than re-transcribing the name/description anywhere. Use this in preference to constructing
    a :class:`KeywordSurfaceSpec` by hand for a registered keyword.

    Parameters
    ----------
    keywordClass
        A concrete ``KeywordBase`` subclass.

    Returns
    -------
    KeywordSurfaceSpec
        The spec rendering that keyword's ``printKeywords`` block / module documentation.
    """
    return KeywordSurfaceSpec(
        name=keywordClass.keywordName,
        description=keywordClass.keywordDescription,
        schema=keywordClass.schema,
    )


def renderSchemaSurface(specs: Iterable[KeywordSurfaceSpec]) -> str:
    """Render a sequence of top-level keyword specs in the legacy ``Module.__doc__`` format.

    Parameters
    ----------
    specs
        One :class:`KeywordSurfaceSpec` per top-level keyword to render, in the order they should
        appear.

    Returns
    -------
    str
        The rendered surface: one keyword's block per spec, separated by a blank line -- matching
        how ``tests/_inputlanguage_snapshot.py`` joins per-module documentation blocks today.
    """
    return "\n\n".join(_renderKeywordBlock(spec, bracket="[") for spec in specs)


def _renderKeywordBlock(spec: KeywordSurfaceSpec, bracket: str) -> str:
    """Render one keyword (or sub-keyword) spec, as a single newline-joined string."""
    return "\n".join(_renderKeywordLines(spec, bracket))


def _fieldDefault(field: dataclasses.Field):
    """The value that applies when this (optional) field is not supplied, mirroring
    ``OptionalKeywordArg.documentedDefault`` -- rendered via ``str()``, not ``repr()``, which is why
    a string default such as ``"Monitor"`` prints unquoted in the golden format.
    """
    if field.default is not dataclasses.MISSING:
        return field.default
    if field.default_factory is not dataclasses.MISSING:  # pragma: no cover -- no scalar field uses this today
        return field.default_factory()
    return None


def _renderScalarEntry(optionName: str, meta: SchemaFieldMeta, field: dataclasses.Field | None) -> str:
    """One ``[optionName] description (<class 'dtype'>[, default = X])`` line, matching
    ``KeywordArg.__doc__``/``OptionalKeywordArg.__doc__`` exactly."""
    if meta.required:
        return f"[{optionName}] {meta.description} ({meta.dtype!r})"
    default = _fieldDefault(field) if field is not None else None
    return f"[{optionName}] {meta.description} ({meta.dtype!r}, default = {default})"


def _renderKeywordLines(spec: KeywordSurfaceSpec, bracket: str) -> list[str]:
    """The line-by-line rendering of one keyword/sub-keyword spec, unindented (the caller is
    responsible for indenting every line when nesting this under a parent, exactly as
    ``Module.__doc__``'s ``optional keywords`` branch does with ``indent2``).
    """
    header = f"[{spec.name}]" if bracket == "[" else f"< {spec.name} >"
    lines = [f"{header} {spec.description}" if spec.description else header]

    if spec.schema is None:
        return lines

    fieldsByName = {field.name: field for field in dataclasses.fields(spec.schema)}
    meta = schemaFields(spec.schema)

    scalarRequired: list[tuple[str, str]] = []
    scalarOptional: list[tuple[str, str]] = []
    subRequired: list[tuple[str, str]] = []
    subOptional: list[tuple[str, str]] = []
    for fieldName, fieldMeta in meta.items():
        if fieldMeta.isDataline:
            continue
        optionName = fieldMeta.optionName or fieldName
        if fieldMeta.subSchema is not None:
            (subRequired if fieldMeta.required else subOptional).append((optionName, fieldName))
        elif fieldMeta.optionsOverrideOnly:
            # Reachable only via a later ">>options" override, not part of this keyword's own
            # line/">>"-block grammar -- see SchemaFieldMeta.optionsOverrideOnly. Rendering-only
            # exclusion: scalarOptionNames/optionNames (and therefore registerSchemaOptions) still
            # include it, so ">>options" itself is unaffected.
            continue
        else:
            (scalarRequired if fieldMeta.required else scalarOptional).append((optionName, fieldName))

    if scalarRequired:
        lines.append(_INDENT1 + "required arguments")
        for optionName, fieldName in scalarRequired:
            lines.append(_INDENT2 + _renderScalarEntry(optionName, meta[fieldName], fieldsByName[fieldName]))

    if scalarOptional:
        lines.append(_INDENT1 + "optional arguments")
        for optionName, fieldName in scalarOptional:
            lines.append(_INDENT2 + _renderScalarEntry(optionName, meta[fieldName], fieldsByName[fieldName]))

    if subRequired:
        lines.append(_INDENT1 + "required keywords")
        for optionName, fieldName in subRequired:
            lines.extend(_renderSubKeywordLines(optionName, meta[fieldName]))

    if subOptional:
        lines.append(_INDENT1 + "optional keywords")
        for optionName, fieldName in subOptional:
            lines.extend(_renderSubKeywordLines(optionName, meta[fieldName]))

    datalineMeta = datalineFieldMeta(spec.schema)
    if datalineMeta is not None:
        section = "required datalines" if datalineMeta.required else "optional datalines"
        lines.append(_INDENT1 + section)
        lines.append(_INDENT2 + datalineMeta.description)

    return lines


def _printKeywordsTypeString(dtype: type) -> str:
    """Mirror ``edelweissfe.utils.misc.typeString`` exactly, without importing it: that function
    also accepts a raw string (for a caller passing an already-rendered type), a case this
    renderer's schema-sourced ``dtype`` never needs."""
    return _PRINT_KEYWORDS_TYPE_STRING.get(dtype, str(dtype))


def renderPrintKeywordsBlock(spec: KeywordSurfaceSpec) -> str:
    """Render one top-level keyword in the legacy ``inputfileparser.printKeywords()`` format.

    This is the *second* legacy rendering format (see the module docstring) -- the one used for the
    structural/type-dispatch keywords hand-declared directly in ``inputfileparser.py`` via
    ``inputLanguage.addKeyword(...)``, rather than the ``Module.__doc__`` format
    :func:`renderSchemaSurface` reproduces. It differs in three respects, matching
    ``printKeywords()`` precisely: required arguments are listed before optional ones with no
    section header separating them, no default value is ever shown (even for an optional
    argument), and neither ``>>`` sub-keyword blocks nor the dataline payload are rendered at all
    -- ``printKeywords()`` only ever iterates ``kw.requiredArgs``/``kw.optionalArgs``, which have no
    sub-keyword or dataline concept.

    Uses :mod:`textwrap` with the exact same ``TextWrapper(width=80, replace_whitespace=False)``
    configuration and ``kwString``/``kwDataString`` indent templates as ``printKeywords()``, so a
    multi-line-wrapped description reproduces the original's wrapping byte-for-byte.

    Parameters
    ----------
    spec
        The keyword to render. ``spec.schema`` may be ``None`` (a keyword with no line options at
        all), in which case only the header line is rendered.

    Returns
    -------
    str
        The rendered block: the header line, then one wrapped line (or more) per required and
        optional scalar argument, newline-joined -- **without** the blank-line separator
        ``printKeywords()`` prints between keywords (``print("\\n")``), which is a concern of the
        caller joining multiple blocks, not of rendering one.
    """
    wrapper = textwrap.TextWrapper(width=80, replace_whitespace=False)
    wrapper.initial_indent = _PRINT_KEYWORDS_HEADER.format(spec.name)
    wrapper.subsequent_indent = " " * len(wrapper.initial_indent)
    lines = [wrapper.fill(spec.description)]

    if spec.schema is None:
        return "\n".join(lines)

    meta = schemaFields(spec.schema)
    required: list[tuple[str, SchemaFieldMeta]] = []
    optional: list[tuple[str, SchemaFieldMeta]] = []
    for fieldName, fieldMeta in meta.items():
        if fieldMeta.isDataline or fieldMeta.subSchema is not None:
            continue
        optionName = fieldMeta.optionName or fieldName
        (required if fieldMeta.required else optional).append((optionName, fieldMeta))

    for optionName, fieldMeta in (*required, *optional):
        wrapper.initial_indent = _PRINT_KEYWORDS_ARG.format(optionName, _printKeywordsTypeString(fieldMeta.dtype))
        wrapper.subsequent_indent = " " * len(wrapper.initial_indent)
        lines.append(wrapper.fill(fieldMeta.description))

    return "\n".join(lines)


def _renderSubKeywordLines(optionName: str, fieldMeta: SchemaFieldMeta) -> list[str]:
    """Render one ``>>`` sub-keyword block, indented one level under its ``required``/``optional
    keywords`` section header -- matching ``Module.__doc__``'s ``optional keywords`` branch, which
    prepends ``indent2`` to *every* line of the block's own rendering (including its header line),
    not only to the block as a whole.
    """
    subSpec = KeywordSurfaceSpec(name=optionName, description=fieldMeta.description, schema=fieldMeta.subSchema)
    return [_INDENT2 + line for line in _renderKeywordLines(subSpec, bracket="<")]
