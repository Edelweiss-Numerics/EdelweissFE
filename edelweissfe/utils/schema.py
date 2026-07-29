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
:func:`resolveCaseInsensitiveOptions`, :func:`buildSchemaFromOptions`,
:func:`coercePresentOptions`) are therefore utilities *available to* L4, not something a schema
class does to itself. :func:`coercePresentOptions` is the partial-application sibling of
:func:`buildSchemaFromOptions`, for *overriding* a subset of an already-constructed instance's
fields (via ``dataclasses.replace``) rather than building a fresh, fully-defaulted one.

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
from typing import Any, ClassVar

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
    optionName
        The name this option carries in the input file, when that differs from the dataclass field
        name. ``None`` means they are identical, which is the common case.

        This exists because the input language admits option names that are not valid Python
        identifiers and therefore cannot be dataclass field names at all -- ``f(x)`` (the
        result-transforming expression accepted by several output managers) and ``f_export(x)`` are
        the real cases. Rather than renaming user-facing options, which would break every existing
        ``.inp`` file, a field may declare the spelling it answers to.
    subSchema
        For a field representing a *sub-keyword block* rather than a scalar option: the schema
        dataclass describing one such block. ``None`` (the common case) marks an ordinary scalar
        option. See :func:`subKeywordField`.
    """

    description: str
    dtype: type
    required: bool = False
    optionName: str | None = None
    subSchema: type | None = None


def schemaField(
    *,
    description: str,
    dtype: type,
    default: Any = MISSING,
    default_factory: Any = MISSING,
    required: bool | None = None,
    optionName: str | None = None,
    subSchema: type | None = None,
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
    optionName
        The name this option is spelled with in the input file, if it cannot be the field name --
        e.g. ``optionName="f(x)"`` on a field called ``f_x``. See :class:`SchemaFieldMeta`.
    subSchema
        Marks this field as a repeatable sub-keyword block described by the given schema. Prefer
        :func:`subKeywordField`, which sets the remaining arguments consistently.

    Returns
    -------
    dataclasses.Field
        A field descriptor suitable as a dataclass class-body value.
    """
    if default is not MISSING and default_factory is not MISSING:
        raise ValueError("Specify at most one of 'default' or 'default_factory', not both.")

    if required is None:
        required = default is MISSING and default_factory is MISSING

    meta = SchemaFieldMeta(
        description=description,
        dtype=dtype,
        required=required,
        optionName=optionName,
        subSchema=subSchema,
    )

    fieldKwargs: dict[str, Any] = {"metadata": {_METADATA_KEY: meta}}
    if default is not MISSING:
        fieldKwargs["default"] = default
    if default_factory is not MISSING:
        fieldKwargs["default_factory"] = default_factory

    return dataclasses.field(**fieldKwargs)


def subKeywordField(
    *,
    description: str,
    schema: type,
    optionName: str | None = None,
    required: bool = False,
) -> dataclasses.Field:
    """Declare one field of an L2 schema as a repeatable *sub-keyword block*.

    Several keywords of the input language are not flat lists of options but carry nested,
    repeatable blocks introduced by the module-level keyword identifier ``>>``. ``*output,
    type=ensight`` is the first such case to be ported::

        *output, type=ensight, name=myExport
        >>perNode,    fieldOutput=Displacement
        >>perElement, fieldOutput=Stress
        >>configuration, overwrite=yes

    Each block has its own option set, so the natural L2 representation is a schema per block kind
    plus, on the enclosing schema, one field per kind holding a **tuple** of block instances. The
    tuple (rather than a single instance) is what makes repetition representable, and it is
    immutable, so it does not compromise the enclosing dataclass's ``frozen=True``.

    The field defaults to the empty tuple, i.e. "no block of this kind was given". A block kind
    that must appear at least once can say so via ``required=True``.

    This is deliberately generic rather than special-cased for ensight: roughly two dozen modules
    use ``>>`` sub-keywords today (all of ``sections/``, thirteen ``stepactions/``,
    ``utils/fieldoutput.py``, ``modelmodifiers/adaptivity/hadaptivity.py``), so P4 needs exactly
    this mechanism.

    Parameters
    ----------
    description
        Human-readable description of the block kind.
    schema
        The schema dataclass describing the options of a single block.
    optionName
        The name the sub-keyword is spelled with in the input file, if it differs from the field
        name -- e.g. a field ``configurations`` answering to ``>>configuration``.
    required
        Whether at least one block of this kind must be supplied.

    Returns
    -------
    dataclasses.Field
        A field descriptor suitable as a dataclass class-body value.
    """
    return schemaField(
        description=description,
        dtype=tuple,
        default=(),
        required=required,
        optionName=optionName,
        subSchema=schema,
    )


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


def optionNames(schemaCls: type) -> dict[str, dataclasses.Field]:
    """Map each of a schema's input-file option names to the field that implements it.

    The option name is the field name unless the field declared an explicit
    :attr:`SchemaFieldMeta.optionName` -- see there for why that indirection exists (input-file
    options such as ``f(x)`` are not valid Python identifiers).

    Parameters
    ----------
    schemaCls
        A frozen dataclass whose fields were declared via :func:`schemaField`.

    Returns
    -------
    dict[str, dataclasses.Field]
        Mapping of user-facing option name to its field, in declaration order.
    """
    fields = dataclasses.fields(schemaCls)
    return {(fieldSchemaMeta(field).optionName or field.name): field for field in fields}


def scalarOptionNames(schemaCls: type) -> dict[str, dataclasses.Field]:
    """Like :func:`optionNames`, but restricted to ordinary scalar options.

    Fields declared via :func:`subKeywordField` are excluded: they are filled from nested ``>>``
    blocks, not from the keyword's own option assignments, and must therefore not be reachable as
    a plain ``name=value`` pair (writing ``perNode=1`` on an ensight dataline is an error, not a
    way to populate the ``perNode`` blocks).

    Parameters
    ----------
    schemaCls
        A frozen dataclass whose fields were declared via :func:`schemaField`.

    Returns
    -------
    dict[str, dataclasses.Field]
        Mapping of user-facing option name to its field, in declaration order.
    """
    return {
        optionName: field
        for optionName, field in optionNames(schemaCls).items()
        if fieldSchemaMeta(field).subSchema is None
    }


def subKeywordFieldNames(schemaCls: type) -> dict[str, dataclasses.Field]:
    """The inverse of :func:`scalarOptionNames`: only the sub-keyword block fields.

    Parameters
    ----------
    schemaCls
        A frozen dataclass whose fields were declared via :func:`schemaField`.

    Returns
    -------
    dict[str, dataclasses.Field]
        Mapping of user-facing sub-keyword name to its field, in declaration order.
    """
    return {
        optionName: field
        for optionName, field in optionNames(schemaCls).items()
        if fieldSchemaMeta(field).subSchema is not None
    }


class OptionSchemaProvider:
    """Mixin declaring that a class owns an L2 option schema, exposed as the class attribute
    :attr:`schema`.

    This is the convention by which a schema reaches the L3 registry along the *dotted-string*
    resolution paths -- the built-in table and, crucially, third-party ``importlib.metadata`` entry
    points. ``registry.register(..., schema=...)`` can pass a schema explicitly, but nothing can
    pass an argument alongside a dotted string in a ``pyproject.toml``; without this convention the
    registry would be schema-blind for exactly the external caller it exists to serve (see
    ``PLAN_INPUT_SYSTEM.md`` §4 and the P2 row). Keeping the schema on the class also keeps it next
    to the constructor it describes, so the two cannot drift apart or be registered inconsistently.

    Deriving from this mixin is what makes a class's schema discoverable; the default of ``None``
    means "no schema declared yet", which is the correct answer for every module that has not been
    converted to L1/L2 yet, so a base class can adopt the mixin before its subclasses are ported.

    The declared default is also why :func:`schemaOf` needs no ``getattr``/``hasattr`` probing
    (which this codebase's conventions forbid): the attribute is guaranteed to exist on every
    subclass, so a plain attribute access suffices.
    """

    #: The L2 schema dataclass describing the options this class accepts, or ``None`` if it does
    #: not declare one (yet). Overridden as a plain class attribute by concrete subclasses.
    schema: ClassVar[type | None] = None


class DatalineAggregatingSchema:
    """Base for an L2 schema that is built from *all* datalines of one keyword block at once.

    The ordinary L4 adapter treats each dataline as an independent module instance: it builds one
    schema per dataline and calls the L1 constructor once per dataline. A few legacy modules invert
    that -- a single instance aggregates a *heterogeneous list of jobs*, one per dataline, with the
    kind of job selected by an option value on the line itself (``meshplot``'s
    ``create=perNode|perElement|xyData|meshOnly``, plus its orthogonal ``saveFigure``). Neither a
    flat schema nor :func:`subKeywordField` expresses that: the jobs are not sub-keyword blocks and
    they do not share one option set.

    Rather than teach the generic adapter about tag options, arm tables and presence flags -- a
    third schema pattern, in the framework, for a module shape we do not want to encourage -- a
    schema of this kind takes responsibility for its own dataline interpretation in
    :meth:`fromDatalines`, and the adapter dispatches on this base class (a type check, like
    :func:`schemaOf`, not attribute probing). The generic path stays two lines shorter than the
    special-case it replaces, and the whole pattern disappears with the last module that uses it.

    Prefer a flat schema or :func:`subKeywordField` for anything new.
    """

    @classmethod
    def fromDatalines(cls, datalines: list[Mapping[str, Any]]) -> Any:
        """Build one schema instance from the option mappings of every dataline, in file order.

        Parameters
        ----------
        datalines
            One mapping of raw (string) option name to value per dataline of the keyword block.

        Returns
        -------
        Any
            An instance of the schema.

        Raises
        ------
        ValueError
            If a dataline is not interpretable, with a message naming the offending option.
        """
        raise NotImplementedError  # pragma: no cover -- subclass responsibility


def schemaOf(target: Any) -> type | None:
    """Return the L2 schema declared by ``target``, or ``None`` if it declares none.

    Used by :func:`edelweissfe.config.registry.lookup` to attach a schema to an object it resolved
    from a dotted string.

    ``None`` is returned for anything that is not a class deriving from
    :class:`OptionSchemaProvider`. That covers a registry target that is a plain module-level
    *function* rather than a class (e.g. ``executePythonCode``'s ``Generator``, whose datalines are
    raw code rather than a flat option mapping, so it declares ``schema = None`` deliberately) --
    dispatching on the type here rather than probing for an attribute is what keeps that case an
    explicit, documented ``None`` instead of an ``AttributeError`` at lookup time.

    Parameters
    ----------
    target
        Any object resolved by the registry -- a class, a factory function, anything.

    Returns
    -------
    type | None
        The schema dataclass, or ``None``.
    """
    if isinstance(target, type) and issubclass(target, OptionSchemaProvider):
        return target.schema
    return None


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
    scalarNames = scalarOptionNames(schemaCls)
    fieldNamesByFold = {optionName.casefold(): field.name for optionName, field in scalarNames.items()}
    subKeywordNamesByFold = {name.casefold(): name for name in subKeywordFieldNames(schemaCls)}

    resolved: dict[str, Any] = {}
    for key, value in options.items():
        exactName = fieldNamesByFold.get(key.casefold())
        if exactName is None:
            availableNames = sorted(scalarNames)
            # A sub-keyword name misused as a scalar option is a distinct mistake with a distinct
            # remedy, so say so instead of offering it as a "did you mean" spelling suggestion.
            subKeywordName = subKeywordNamesByFold.get(key.casefold())
            if subKeywordName is not None:
                raise ValueError(
                    f"'{key}' is a sub-keyword of {schemaCls.__name__}, not an option: write it on "
                    f"its own line as '>>{subKeywordName}, ...' rather than as '{key}=...'."
                )
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


def buildSchemaFromOptions(
    schemaCls: type,
    options: Mapping[str, Any],
    subKeywordOptions: Mapping[str, list] | None = None,
) -> Any:
    """Build an instance of a schema dataclass from a (possibly mis-cased, string-valued) mapping.

    This is the L4-facing counterpart of ``castKwargsValuesAndAddDefaults`` /
    ``caseInsensitiveKwargsChecker`` in ``inputlanguage.py``: resolve case-insensitive keys
    (:func:`resolveCaseInsensitiveOptions`), coerce each present value to its field's declared
    ``dtype`` (:func:`coerceValue`), and let the dataclass constructor fill in defaults for any
    field that was not present in ``options`` -- exactly as ``OptionalKeywordArg.default`` does
    today, just expressed as an ordinary Python default value instead of a side object.

    An option whose value is ``None`` is treated as **not supplied**, so the field's default
    applies. This is deliberate and is decided here, once, rather than in each L4 adapter. The
    ``.inp`` parser represents "the user did not specify this optional argument" as a key present
    with the value ``None`` (see e.g. ``inputfilehelpers.py``'s ``definition["name"] is not None``
    test), and coercing that ``None`` would be actively harmful rather than merely useless: for a
    ``str``-typed option, ``coerceValue(None, str)`` yields the *string* ``"None"``, which is
    truthy, so "the user said nothing" would silently become a real value -- an export filename
    literally called ``None``, for instance. Legacy avoided this only by construction, coercing
    exclusively the keys actually present in the kwargs
    (``utils/misc.py``'s ``castKwargsValuesAndAddDefaults``); stating the rule explicitly here
    means an adapter cannot reintroduce the trap by passing definition-level keys straight in.

    Key *names* are still validated before values are examined, so a misspelled option raises even
    if its value happens to be ``None``, and a required field given ``None`` is reported as missing
    rather than being coerced into nonsense.

    Parameters
    ----------
    schemaCls
        A frozen dataclass whose fields were declared via :func:`schemaField`.
    options
        A mapping of (possibly mis-cased) option names to raw (typically string) values.
    subKeywordOptions
        For a schema with :func:`subKeywordField` fields: a mapping of sub-keyword name to the list
        of option mappings for the blocks of that kind, in file order -- exactly the shape the
        ``.inp`` parser produces as ``moduleOptions``. Each block is built recursively into its own
        sub-schema instance. Sub-keyword names are matched case-insensitively, like every other
        name in the input language; the parser stores them as the user wrote them.

    Returns
    -------
    Any
        An instance of ``schemaCls``.

    Raises
    ------
    ValueError
        If ``options`` contains an unknown key, a value that cannot be coerced, or omits a
        required field; or if ``subKeywordOptions`` names a sub-keyword the schema does not declare.
    """
    fieldsByName = {field.name: field for field in dataclasses.fields(schemaCls)}
    resolvedOptions = resolveCaseInsensitiveOptions(schemaCls, options)

    kwargs: dict[str, Any] = {}
    for name, rawValue in resolvedOptions.items():
        if rawValue is None:
            continue
        meta = fieldSchemaMeta(fieldsByName[name])
        kwargs[name] = coerceValue(rawValue, meta.dtype)

    kwargs.update(_buildSubKeywords(schemaCls, subKeywordOptions or {}))

    # Reported by input-file option name, not field name, so a user reading the error sees the
    # spelling they are expected to type (they cannot know that `f(x)` is a field called `f_x`).
    # A required sub-keyword field lands in `kwargs` only when at least one block was supplied,
    # so the same check covers "at least one >>block of this kind is mandatory".
    missingRequired = [
        optionName
        for optionName, field in optionNames(schemaCls).items()
        if fieldSchemaMeta(field).required and field.name not in kwargs
    ]
    if missingRequired:
        raise ValueError(f"Missing required option(s) for {schemaCls.__name__}: {', '.join(missingRequired)}.")

    return schemaCls(**kwargs)


def coercePresentOptions(schemaCls: type, options: Mapping[str, Any]) -> dict[str, Any]:
    """Validate and coerce only the keys actually present in ``options`` against ``schemaCls``,
    without requiring or defaulting anything absent.

    This is the partial-application sibling of :func:`buildSchemaFromOptions`, for overriding a
    handful of fields of an *already-constructed* instance of ``schemaCls`` (via
    ``dataclasses.replace(existingInstance, **coercePresentOptions(type(existingInstance).schema,
    rawOptions))``) rather than building a fresh instance from scratch. There is no missing-required
    check here -- an override is by definition partial, so "required at construction" does not mean
    "must be restated on every override" -- and no field of ``schemaCls`` gets a default value
    inserted for a key that was not given.

    An option whose value is ``None`` is still treated as **not supplied** (see
    :func:`buildSchemaFromOptions`), for the identical reason: a ``None``-valued key is how the
    ``.inp`` parser spells "the user did not write this option", and coercing it would manufacture a
    value out of "nothing was said".

    Parameters
    ----------
    schemaCls
        A frozen dataclass whose fields were declared via :func:`schemaField`.
    options
        A mapping of (possibly mis-cased) option names to raw (typically string) values -- typically
        a small subset of ``schemaCls``'s fields, not all of them.

    Returns
    -------
    dict[str, Any]
        Maps each *present*, non-``None`` option to its coerced value, keyed by the schema's
        exact-case field name. Safe to splat into ``dataclasses.replace``.

    Raises
    ------
    ValueError
        If ``options`` contains a key that is not a valid (scalar) option of ``schemaCls``, or a
        value that cannot be coerced to its field's declared ``dtype``.
    """
    fieldsByName = {field.name: field for field in dataclasses.fields(schemaCls)}
    resolvedOptions = resolveCaseInsensitiveOptions(schemaCls, options)

    coerced: dict[str, Any] = {}
    for name, rawValue in resolvedOptions.items():
        if rawValue is None:
            continue
        meta = fieldSchemaMeta(fieldsByName[name])
        coerced[name] = coerceValue(rawValue, meta.dtype)

    return coerced


def _buildSubKeywords(schemaCls: type, subKeywordOptions: Mapping[str, list]) -> dict[str, tuple]:
    """Build the sub-keyword block tuples of ``schemaCls`` from raw per-block option mappings.

    Returns only the fields for which at least one block was supplied, so that an absent block kind
    falls back to the field's own default (the empty tuple) and is reported as missing if the field
    was declared ``required=True``.
    """
    subKeywordFields = subKeywordFieldNames(schemaCls)
    if not subKeywordOptions:
        return {}

    fieldsByFold = {name.casefold(): (name, field) for name, field in subKeywordFields.items()}

    built: dict[str, tuple] = {}
    for givenName, blocks in subKeywordOptions.items():
        match = fieldsByFold.get(givenName.casefold())
        if match is None:
            available = sorted(subKeywordFields)
            raise ValueError(
                f"'{givenName}' is not a valid sub-keyword for {schemaCls.__name__}. "
                f"Available sub-keywords: {', '.join(available) if available else 'none'}."
            )
        canonicalName, field = match
        if not blocks:
            continue
        subSchema = fieldSchemaMeta(field).subSchema
        try:
            built[field.name] = tuple(buildSchemaFromOptions(subSchema, block) for block in blocks)
        except ValueError as e:
            raise ValueError(f"In sub-keyword '{canonicalName}': " + e.args[0]) from e

    return built
