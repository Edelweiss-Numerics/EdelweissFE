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
from dataclasses import dataclass
from typing import Iterable

from edelweissfe.utils.schema import SchemaFieldMeta, datalineFieldMeta, schemaFields

#: Matches ``edelweissfe.utils.inputlanguage.indent1``/``indent2`` exactly -- the two indentation
#: levels the legacy renderer uses (a "required/optional arguments|keywords|datalines" section
#: header, and the entries within it).
_INDENT1 = "  "
_INDENT2 = "    "


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


def _renderSubKeywordLines(optionName: str, fieldMeta: SchemaFieldMeta) -> list[str]:
    """Render one ``>>`` sub-keyword block, indented one level under its ``required``/``optional
    keywords`` section header -- matching ``Module.__doc__``'s ``optional keywords`` branch, which
    prepends ``indent2`` to *every* line of the block's own rendering (including its header line),
    not only to the block as a whole.
    """
    subSpec = KeywordSurfaceSpec(name=optionName, description=fieldMeta.description, schema=fieldMeta.subSchema)
    return [_INDENT2 + line for line in _renderKeywordLines(subSpec, bracket="<")]
