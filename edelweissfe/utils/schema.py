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

"""L2 primitives for the input-language redesign (see ``PLAN_INPUT_SYSTEM.md``, P1).

A "schema" here is an ordinary, immutable (``frozen=True``) :mod:`dataclasses` class that a
module owns for itself: it declares which options that module accepts, their Python types,
whether they are required, and a human-readable description -- everything the future Sphinx
``pprint`` replacement (P5) needs to render documentation, without a parallel dict living
somewhere else and without mutating any other module's state.

This module intentionally has **no** dependency on :mod:`edelweissfe.utils.inputlanguage`,
:mod:`edelweissfe.config.registry`, or any concrete element/material/solver module: it is pure,
stdlib-only (no pydantic, per the plan's rationale: avoid a new dependency and unknown
free-threading behavior) infrastructure that L1 modules and the future L4 adapter both depend on.

Per rule (c) of the target architecture, case-insensitivity and string-to-type coercion belong to
L4 (the input-file front-end), not to the L2 schema objects themselves -- a schema instance is
always exact-case and precisely typed. The coercion helpers in this module (:func:`coerceValue`,
:func:`resolveCaseInsensitiveOptions`, :func:`buildSchemaFromOptions`) are therefore utilities
*available to* L4, not something a schema class does to itself.

The coercion logic mirrors, rather than reinvents, the casting already performed by
``edelweissfe.utils.inputlanguage.KeywordArg.getValueFromKwargs`` and
``castKwargsValuesAndAddDefaults``: a value destined for a ``bool`` field goes through
:func:`edelweissfe.utils.misc.asBool` (tolerant of a real ``bool``, otherwise
``strtobool``-parsed), and any other value is passed through unchanged if it is already an
instance of the target type, otherwise coerced via ``dtype(value)``. Nothing here changes what a
given input produces; it only makes the already-correct-type case explicit rather than relying on
the target type's constructor happening to be idempotent (which is true for ``int``/``float``/
``str`` but was **not** true for ``bool`` -- see the P0 bugfix to ``strtobool``/``asBool``).
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from edelweissfe.utils.misc import asBool, findSimilarString

#: Sentinel re-exported for callers that need to distinguish "no default given" from "default is
#: None" when calling :func:`schemaField` -- identical to :data:`dataclasses.MISSING`.
MISSING = dataclasses.MISSING

_METADATA_KEY = "edelweissfe.schema"


@dataclass(frozen=True)
class SchemaFieldMeta:
    """Documentation-oriented metadata attached to a single field of a schema dataclass.

    Parameters
    ----------
    description
        Human-readable description of the option, used verbatim by the future Sphinx
        documentation generator (P5).
    dtype
        The Python type the option's value must have once coerced (e.g. ``int``, ``float``,
        ``str``, ``bool``).
    required
        Whether the option must be supplied by the caller. This is tracked explicitly (rather
        than being re-derived from "does the dataclass field have a default") so that
        documentation generation does not need to inspect :data:`dataclasses.MISSING` sentinels
        directly.
    """

    description: str
    dtype: type
    required: bool = False


def schemaField(
    *,
    description: str,
    dtype: type,
    default: Any = MISSING,
    default_factory: Any = MISSING,
    required: bool | None = None,
) -> dataclasses.Field:
    """Declare one field of an L2 option schema dataclass.

    This is a thin wrapper around :func:`dataclasses.field` that attaches a
    :class:`SchemaFieldMeta` under a well-known metadata key, instead of maintaining a parallel
    dict of descriptions next to the dataclass. Use it as the value of a field in a class
    decorated with ``@dataclasses.dataclass(frozen=True)``:

    .. code-block:: python

        @dataclass(frozen=True)
        class EnsightConfigurationSchema:
            overwrite: bool = schemaField(description="Overwrite existing output.", dtype=bool, default=False)
            intermediate_save_interval: int = schemaField(
                description="Save an intermediate .case every N increments.", dtype=int, default=10
            )

    Parameters
    ----------
    description
        Human-readable description of the option.
    dtype
        The Python type the option's value must have.
    default
        The default value, if the option is optional. Mutually exclusive with ``default_factory``.
    default_factory
        A zero-argument callable producing the default value, for mutable/composite defaults.
        Mutually exclusive with ``default``.
    required
        Explicitly override whether the option is required. If not given (``None``), it is
        inferred as ``True`` exactly when neither ``default`` nor ``default_factory`` was
        supplied.

    Returns
    -------
    dataclasses.Field
        A field descriptor suitable as a dataclass class-body value.
    """
    if default is not MISSING and default_factory is not MISSING:
        raise ValueError("Specify at most one of 'default' or 'default_factory', not both.")

    if required is None:
        required = default is MISSING and default_factory is MISSING

    meta = SchemaFieldMeta(description=description, dtype=dtype, required=required)

    fieldKwargs: dict[str, Any] = {"metadata": {_METADATA_KEY: meta}}
    if default is not MISSING:
        fieldKwargs["default"] = default
    if default_factory is not MISSING:
        fieldKwargs["default_factory"] = default_factory

    return dataclasses.field(**fieldKwargs)


def fieldSchemaMeta(field: dataclasses.Field) -> SchemaFieldMeta:
    """Retrieve the :class:`SchemaFieldMeta` attached to a dataclass field by :func:`schemaField`.

    Parameters
    ----------
    field
        A field of a schema dataclass, as returned by ``dataclasses.fields(schemaCls)``.

    Returns
    -------
    SchemaFieldMeta
        The metadata describing this field.

    Raises
    ------
    KeyError
        If the field was declared without :func:`schemaField` (i.e. it carries no schema
        metadata).
    """
    try:
        return field.metadata[_METADATA_KEY]
    except KeyError:
        raise KeyError(
            f"Field '{field.name}' was not declared with schemaField(...) and therefore carries " "no SchemaFieldMeta."
        )


def schemaFields(schemaCls: type) -> dict[str, SchemaFieldMeta]:
    """Collect the :class:`SchemaFieldMeta` of every field of a schema dataclass.

    Parameters
    ----------
    schemaCls
        A frozen dataclass whose fields were declared via :func:`schemaField`.

    Returns
    -------
    dict[str, SchemaFieldMeta]
        Mapping of field name to its schema metadata, in declaration order.
    """
    return {field.name: fieldSchemaMeta(field) for field in dataclasses.fields(schemaCls)}


def coerceValue(value: Any, dtype: type) -> Any:
    """Coerce a single value to ``dtype``, tolerating a value that is already correctly typed.

    Mirrors ``edelweissfe.utils.inputlanguage.KeywordArg.getValueFromKwargs``: for ``dtype is
    bool`` this delegates to :func:`edelweissfe.utils.misc.asBool` (which passes a real ``bool``
    through unchanged and otherwise parses a truth-string via ``strtobool``); a real ``strtobool``
    call on an already-``bool`` value raises ``AttributeError`` (it unconditionally calls
    ``.lower()``), which is exactly the P0 bug this module deliberately does not reproduce.

    For every other ``dtype``, a value that is already an instance of ``dtype`` is returned
    unchanged; otherwise ``dtype(value)`` is attempted. This does not change behavior relative to
    the existing (unconditional ``dtype(value)``) casting for ``int``/``float``/``str``, since
    those constructors are idempotent on already-correct-type input -- it only avoids relying on
    that idempotence implicitly.

    Parameters
    ----------
    value
        The value to coerce (typically a string from an ``.inp`` file, or an already-correct
        Python value from a programmatic caller).
    dtype
        The target type.

    Returns
    -------
    Any
        The coerced value.

    Raises
    ------
    ValueError
        If ``value`` cannot be converted to ``dtype``.
    """
    if dtype is bool:
        return asBool(value)

    if isinstance(value, dtype):
        return value

    try:
        return dtype(value)
    except (TypeError, ValueError) as e:
        raise ValueError(f"Cannot convert {value!r} to {dtype}") from e


def resolveCaseInsensitiveOptions(schemaCls: type, options: Mapping[str, Any]) -> dict[str, Any]:
    """Map possibly-mis-cased option keys onto a schema dataclass's exact-case field names.

    This is the one place case-insensitivity is allowed to enter the L2/L4 boundary (rule (c) of
    the target architecture): the schema class itself (``schemaCls``) is always exact-case, but an
    ``.inp`` file (or a careless caller) may spell an option key in any case. This utility performs
    the case-fold lookup once, so schema instances never need to.

    Parameters
    ----------
    schemaCls
        The schema dataclass whose field names are the valid option keys.
    options
        A mapping of (possibly mis-cased) option names to raw values.

    Returns
    -------
    dict[str, Any]
        The same values, keyed by the schema's exact-case field names.

    Raises
    ------
    ValueError
        If a key in ``options`` does not case-insensitively match any field of ``schemaCls``.
    """
    fieldNamesByFold = {field.name.casefold(): field.name for field in dataclasses.fields(schemaCls)}

    resolved: dict[str, Any] = {}
    for key, value in options.items():
        exactName = fieldNamesByFold.get(key.casefold())
        if exactName is None:
            availableNames = sorted(fieldNamesByFold.values())
            hint = ""
            if availableNames:
                try:
                    similar = findSimilarString(key, availableNames)
                    hint = f" Did you mean '{similar}'?"
                except ValueError:
                    pass
            raise ValueError(
                f"'{key}' is not a valid option for {schemaCls.__name__}. "
                f"Available options: {', '.join(availableNames)}.{hint}"
            )
        resolved[exactName] = value

    return resolved


def buildSchemaFromOptions(schemaCls: type, options: Mapping[str, Any]) -> Any:
    """Build an instance of a schema dataclass from a (possibly mis-cased, string-valued) mapping.

    This is the L4-facing counterpart of ``castKwargsValuesAndAddDefaults`` /
    ``caseInsensitiveKwargsChecker`` in ``inputlanguage.py``: resolve case-insensitive keys
    (:func:`resolveCaseInsensitiveOptions`), coerce each present value to its field's declared
    ``dtype`` (:func:`coerceValue`), and let the dataclass constructor fill in defaults for any
    field that was not present in ``options`` -- exactly as ``OptionalKeywordArg.default`` does
    today, just expressed as an ordinary Python default value instead of a side object.

    Parameters
    ----------
    schemaCls
        A frozen dataclass whose fields were declared via :func:`schemaField`.
    options
        A mapping of (possibly mis-cased) option names to raw (typically string) values.

    Returns
    -------
    Any
        An instance of ``schemaCls``.

    Raises
    ------
    ValueError
        If ``options`` contains an unknown key, a value that cannot be coerced, or omits a
        required field.
    """
    fieldsByName = {field.name: field for field in dataclasses.fields(schemaCls)}
    resolvedOptions = resolveCaseInsensitiveOptions(schemaCls, options)

    kwargs: dict[str, Any] = {}
    for name, rawValue in resolvedOptions.items():
        meta = fieldSchemaMeta(fieldsByName[name])
        kwargs[name] = coerceValue(rawValue, meta.dtype)

    missingRequired = [
        name for name, field in fieldsByName.items() if fieldSchemaMeta(field).required and name not in kwargs
    ]
    if missingRequired:
        raise ValueError(f"Missing required option(s) for {schemaCls.__name__}: {', '.join(missingRequired)}.")

    return schemaCls(**kwargs)
