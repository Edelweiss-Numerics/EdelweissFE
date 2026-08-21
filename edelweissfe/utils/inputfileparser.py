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
#  Paul Hofer Paul.Hofer@uibk.ac.at
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
"""
Inputfileparser for inputfiles employing an Abaqus-like syntax.
"""

import dataclasses
from os.path import dirname, join

from edelweissfe.config import registry
from edelweissfe.config.registry import RegistryConflictError, RegistryLookupError
from edelweissfe.utils.caseinsensitivedict import CaseInsensitiveDict
from edelweissfe.utils.misc import (
    caseInsensitiveKwargsChecker,
    caseInsensitiveRequiredArgsChecker,
    convertAssignmentsToCaseInsensitiveStringDictionary,
    splitLineAtCommas,
    strCaseCmp,
)
from edelweissfe.utils.schema import (
    coerceValue,
    datalineFieldMeta,
    fieldSchemaMeta,
    subKeywordFieldNames,
)

#: The ``.inp`` syntax markers: a top-level keyword line starts with ``*``, a ``>>`` sub-keyword
#: line with ``>>``. The single source of truth for these two markers, imported from here by every
#: other module that needs to spell an error message with the right marker.
keywordIdentifier = "*"
moduleLevelKeywordIdentifier = ">>"

#: Top-level keywords whose grammar is completed by a further name/generator-dispatched target's
#: own schema -- maps the keyword's casefolded name to ``(registry category, the keyword-line
#: option name carrying the dispatch value)``.
#: ``step`` is included only for :func:`_expectsPlainDatalines`'s sake -- its ``>>`` grammar is
#: resolved separately, always via the ``stepaction`` category regardless of the step *type* (see
#: :func:`_resolveModuleKeywordSchema`), because every step action registers identically onto every
#: step type.
_DISPATCH_CATEGORY_BY_KEYWORD = {
    "output": ("outputmanager", "type"),
    "section": ("section", "type"),
    "analyticalfield": ("analyticalfield", "type"),
    "constraint": ("constraint", "type"),
    "modelmodifier": ("modelmodifier", "type"),
    "modelgenerator": ("generator", "generator"),
    "step": ("step", "type"),
}


def _updateStepActionSchema(moduleKeyword: str):
    """Return the dedicated ``update<keyword>`` schema for a step action's partial re-declaration
    in a later step, or ``None`` if this step action has no such companion.

    Only three step actions declare one at all (``dirichlet``, ``distributedload``,
    ``nodeforces``) -- every other step action either has no update path (a re-declaration must
    restate every required argument again) or -- ``options`` -- validates dynamically and never
    reaches this function at all (see the ``isDynamicOptionsKeyword`` branch of
    :func:`parseModuleKeywordLine`). A small explicit table, not a registry category: these
    ``Update<Name>Schema`` classes are documentation/parser-validation companions the step action's
    own constructor never references (see e.g. ``stepactions/dirichlet.py``'s own docstring), so
    they have no home in the ``stepaction`` registry, which maps to the step action classes
    themselves.

    Parameters
    ----------
    moduleKeyword
        The step action's own name, as written in the ``.inp`` file (e.g. ``"dirichlet"``).

    Returns
    -------
    type | None
        The update schema, or ``None``.
    """
    name = moduleKeyword.casefold()
    if name == "dirichlet":
        from edelweissfe.stepactions.dirichlet import UpdateDirichletSchema

        return UpdateDirichletSchema
    if name == "distributedload":
        from edelweissfe.stepactions.distributedload import UpdateDistributedloadSchema

        return UpdateDistributedloadSchema
    if name == "nodeforces":
        from edelweissfe.stepactions.nodeforces import UpdateNodeforcesSchema

        return UpdateNodeforcesSchema
    return None


def _allScalarFields(schemaCls: type | None) -> dict:
    """Map every input-file option name of ``schemaCls``'s own scalar (non-dataline,
    non-sub-keyword) fields to its :class:`dataclasses.Field`.

    Not :func:`~edelweissfe.utils.schema.scalarOptionNames`: that function excludes
    ``structuralOnly`` fields (the adapter pops them before ``buildSchemaFromOptions`` ever sees
    them), but the *parser* must still recognize and require every scalar option a schema declares,
    with no such distinction at all (``structuralOnly``/``optionsOverrideOnly``/``updateOnly`` are
    schema-only rendering/construction concerns, see ``utils/schema.py``'s ``SchemaFieldMeta``).

    Parameters
    ----------
    schemaCls
        A frozen dataclass schema, or ``None`` (a keyword/module declaring no line options at all).

    Returns
    -------
    dict
        Maps option name to field, in declaration order. Empty if ``schemaCls`` is ``None``.
    """
    if schemaCls is None:
        return {}
    fields = {}
    for field in dataclasses.fields(schemaCls):
        meta = fieldSchemaMeta(field)
        if meta.isDataline or meta.subSchema is not None:
            continue
        fields[meta.optionName or field.name] = field
    return fields


def _realScalarFields(schemaCls: type | None) -> dict:
    """Like :func:`_allScalarFields`, but excluding ``optionsOverrideOnly``/``updateOnly`` fields --
    the ones that exist purely for the ``>>options`` override mechanism or the ``update<keyword>``
    grammar and were never part of a keyword's own real line/dataline grammar (whose presence is
    what implies a dispatch target actually expects plain datalines, see
    :func:`_expectsPlainDatalines`).

    Parameters
    ----------
    schemaCls
        A frozen dataclass schema, or ``None``.

    Returns
    -------
    dict
        Maps option name to field, in declaration order. Empty if ``schemaCls`` is ``None``.
    """
    return {
        name: field
        for name, field in _allScalarFields(schemaCls).items()
        if not fieldSchemaMeta(field).optionsOverrideOnly and not fieldSchemaMeta(field).updateOnly
    }


def _requiredAndOptionalNames(schemaCls: type | None) -> tuple[list, list]:
    """Split :func:`_allScalarFields`'s option names into required/optional lists, the shape
    :func:`~edelweissfe.utils.misc.caseInsensitiveKwargsChecker` expects.

    Parameters
    ----------
    schemaCls
        A frozen dataclass schema, or ``None``.

    Returns
    -------
    tuple[list, list]
        ``(requiredNames, optionalNames)``.
    """
    fields = _allScalarFields(schemaCls)
    required = [name for name, field in fields.items() if fieldSchemaMeta(field).required]
    optional = [name for name, field in fields.items() if not fieldSchemaMeta(field).required]
    return required, optional


def _coerceKnownFields(schemaCls: type | None, options: dict) -> CaseInsensitiveDict:
    """Coerce and default the keys of ``options`` that match one of ``schemaCls``'s own scalar
    fields; every other key is left untouched, as a raw string.

    Parameters
    ----------
    schemaCls
        A frozen dataclass schema, or ``None`` (nothing is coerced or defaulted).
    options
        A mapping of (possibly mis-cased) option names to raw, typically string, values.

    Returns
    -------
    CaseInsensitiveDict
        ``options``, with its known keys coerced to their declared type and every missing optional
        field defaulted.
    """
    options = CaseInsensitiveDict(options)
    for optionName, field in _allScalarFields(schemaCls).items():
        meta = fieldSchemaMeta(field)
        if optionName in options:
            options[optionName] = coerceValue(options[optionName], meta.dtype)
        elif not meta.required:
            options[optionName] = field.default if field.default is not dataclasses.MISSING else None
    return options


def _resolveDispatchSchemaForKeywordLine(keyword: str, options: dict) -> type | None:
    """Resolve the schema of the ``type=``/``generator=``-dispatched target hosted by a
    top-level keyword's line, so a value meant for *that* target's own options (e.g. a plane
    section's ``thickness``) written directly on the ``*section`` line is not rejected as unknown.

    Parameters
    ----------
    keyword
        The top-level keyword being parsed.
    options
        Its already-coerced-so-far option mapping (used only to read the dispatch value).

    Returns
    -------
    type | None
        The dispatch target's schema, or ``None`` if this keyword has no such dispatch, the
        dispatch value is missing, or it does not resolve to anything registered.
    """
    dispatch = _DISPATCH_CATEGORY_BY_KEYWORD.get(keyword.casefold())
    if dispatch is None:
        return None
    category, optionName = dispatch
    dispatchValue = options.get(optionName)
    if not dispatchValue:
        return None
    try:
        _, schema = registry.lookup(category, dispatchValue)
    except RegistryConflictError:
        # An unresolvable name conflict, not a "does not exist" -- must not be mistaken for the
        # latter by a caller treating a plain miss as "no dispatch schema".
        raise
    except RegistryLookupError:
        return None
    return schema


def _resolveModuleKeywordSchema(topLevelKeyword: str, topLevelOptions: dict, moduleKeyword: str) -> type | None:
    """Resolve the schema describing one ``>>`` sub-keyword block's own grammar.

    For ``*step``, every step action registers identically onto every step type, so its ``>>``
    grammar is resolved directly against the ``stepaction`` registry category, independent of the
    step's own ``type=``. Every other keyword's ``>>`` grammar is a
    :func:`~edelweissfe.utils.schema.subKeywordField` of either the dispatch target's own schema
    (resolved via ``type=``/``generator=``, e.g. ensight's ``>>perNode``) or, for a keyword with no
    such dispatch (``*fieldOutput``), the keyword's own schema directly.

    Parameters
    ----------
    topLevelKeyword
        The enclosing top-level keyword (e.g. ``"output"``, ``"step"``).
    topLevelOptions
        The enclosing top-level keyword's own (already coerced) option mapping.
    moduleKeyword
        The ``>>`` sub-keyword's own name, as written in the ``.inp`` file.

    Returns
    -------
    type | None
        The sub-keyword's own schema.

    Raises
    ------
    ValueError
        If ``moduleKeyword`` does not name a known ``>>`` sub-keyword in this context.
    """
    if strCaseCmp(topLevelKeyword, "step"):
        try:
            _, schema = registry.lookup("stepaction", moduleKeyword)
        except RegistryLookupError as e:
            raise ValueError(str(e)) from e
        return schema

    _, keywordSchema = registry.lookup("keyword", topLevelKeyword)
    dispatch = _DISPATCH_CATEGORY_BY_KEYWORD.get(topLevelKeyword.casefold())
    if dispatch is None:
        hostSchema = keywordSchema
    else:
        category, optionName = dispatch
        dispatchValue = topLevelOptions.get(optionName)
        try:
            _, hostSchema = registry.lookup(category, dispatchValue)
        except RegistryLookupError as e:
            raise ValueError(str(e)) from e

    subFields = subKeywordFieldNames(hostSchema) if hostSchema is not None else {}
    foldedFields = {name.casefold(): field for name, field in subFields.items()}
    field = foldedFields.get(moduleKeyword.casefold())
    if field is None:
        available = ", ".join(sorted(subFields)) or "none"
        raise ValueError(
            f"'{moduleLevelKeywordIdentifier}{moduleKeyword}' is not a valid module keyword for "
            f"'{keywordIdentifier}{topLevelKeyword}'. Available: {available}."
        )
    return fieldSchemaMeta(field).subSchema


def _dispatchTargetExpectsPlainDatalines(schemaCls: type | None) -> bool:
    """Whether a resolved dispatch target (a section type, output-manager type, generator, ...)
    accepts plain (non-``>>``) datalines: a dispatch target's *own* scalar options, unlike a
    top-level keyword's, are conventionally supplied via datalines rather than on the keyword line
    itself, so a schema with any real scalar field or a dedicated dataline payload of its own both
    count.

    Parameters
    ----------
    schemaCls
        The dispatch target's own schema, or ``None``.

    Returns
    -------
    bool
        Whether a plain dataline is valid here. ``True`` when ``schemaCls`` is ``None``: a dispatch
        target with no schema at all yet is a genuinely raw-datalines case in every instance that
        exists today (``executePythonCode``'s Python source), so this conservatively allows rather
        than rejecting every real ``.inp`` file exercising it.
    """
    if schemaCls is None:
        return True
    return bool(_realScalarFields(schemaCls)) or datalineFieldMeta(schemaCls) is not None


def _expectsPlainDatalines(keyword: str, options: dict) -> bool:
    """Whether a plain (non-``>>``) line following a ``*keyword`` block's own line is a valid
    dataline for it, resolved from the keyword's (or its dispatch target's) own schema.

    Parameters
    ----------
    keyword
        The current top-level keyword.
    options
        Its current (already coerced) option mapping -- used to read the dispatch value
        (``type=``/``generator=``) for a keyword whose plain-dataline grammar belongs to a resolved
        dispatch target rather than to the keyword's own schema (e.g. ``*modelGenerator``'s
        ``x0=...`` datalines belong to the resolved generator, not to ``*modelGenerator`` itself).

    Returns
    -------
    bool
        Whether a plain dataline is valid here.
    """
    dispatch = _DISPATCH_CATEGORY_BY_KEYWORD.get(keyword.casefold())
    if dispatch is None:
        _, keywordSchema = registry.lookup("keyword", keyword)
        return keywordSchema is not None and datalineFieldMeta(keywordSchema) is not None

    if strCaseCmp(keyword, "step"):
        # Every step type shares one schema (`steps.base.stepbase.StepIncrementationSchema`) with
        # only optional fields, so this could dispatch through
        # `dispatchSchema`/`_dispatchTargetExpectsPlainDatalines` like every other keyword -- kept
        # as an unconditional `True` anyway since the outcome is identical for both registered
        # step types and this avoids a resolve-then-recompute round trip on the hottest keyword in
        # a typical input file.
        return True

    category, optionName = dispatch
    dispatchValue = options.get(optionName)
    if not dispatchValue:
        return False
    try:
        _, dispatchSchema = registry.lookup(category, dispatchValue)
    except RegistryConflictError:
        # An unresolvable name conflict, not a "does not exist" -- must not be mistaken for the
        # latter by a caller treating a plain miss as "no plain datalines here".
        raise
    except RegistryLookupError:
        return False
    return _dispatchTargetExpectsPlainDatalines(dispatchSchema)


def parseKeywordLine(line, fileName):
    lineElements = splitLineAtCommas(line.removeprefix(keywordIdentifier))

    keyword = lineElements[0]
    optionAssignments = lineElements[1:]

    try:
        options = convertAssignmentsToCaseInsensitiveStringDictionary(optionAssignments)
    except ValueError as e:
        e.args = (f"Error during parsing of keyword {keywordIdentifier}{keyword}: " + e.args[0],)
        raise e

    try:
        _, keywordSchema = registry.lookup("keyword", keyword)
    except RegistryLookupError as e:
        raise ValueError(str(e)) from e

    options = _coerceKnownFields(keywordSchema, options)

    requiredNames, optionalNames = _requiredAndOptionalNames(keywordSchema)

    @caseInsensitiveKwargsChecker(requiredNames, optionalNames)
    def checkKeywordInput(*args, **kwargs):
        """this is a dummy function needed to apply kwargsChecker"""
        return

    try:
        checkKeywordInput(**options)
    except ValueError as e:
        # in some cases, dispatch-target-specific kwArgs arguments can be given directly on the
        # keyword line (e.g. a plane section's 'thickness') -- see
        # _resolveDispatchSchemaForKeywordLine.
        dispatchSchema = _resolveDispatchSchemaForKeywordLine(keyword, options)

        if dispatchSchema is not None:
            extraRequired, extraOptional = _requiredAndOptionalNames(dispatchSchema)

            @caseInsensitiveKwargsChecker(requiredNames, optionalNames + extraOptional + extraRequired)
            def checkKeywordInput(*args, **kwargs):
                """this is a dummy function needed to apply kwargsChecker"""
                return

            checkKeywordInput(**options)

        else:
            e.args = (f"Error during parsing of keyword {keywordIdentifier}{keyword}: " + e.args[0],)
            raise e

    options["inputFile"] = fileName  # save also the filename of the original inputfile!

    options["datalines"] = []

    return keyword, options


def parseModuleKeywordLine(line, fileName, topLevelKeyword, topLevelOptions, fileDict):
    lineElements = splitLineAtCommas(line.removeprefix(moduleLevelKeywordIdentifier))

    keyword = lineElements[0]
    optionAssignments = lineElements[1:]

    try:
        rawOptions = convertAssignmentsToCaseInsensitiveStringDictionary(optionAssignments)
    except ValueError as e:
        e.args = (f"Error during parsing of keyword {keywordIdentifier}{keyword}: " + e.args[0],)
        raise e

    isDynamicOptionsKeyword = strCaseCmp(topLevelKeyword, "step") and strCaseCmp(keyword, "options")

    if isDynamicOptionsKeyword:
        # This keyword's full option set depends on runtime information the parser does not have
        # (see stepactions/options.py) -- only 'name' is enforced here; every other key is
        # passed through raw for a later stage to validate against whatever it actually resolves to.
        @caseInsensitiveRequiredArgsChecker(["name"])
        def checkKeywordInput(*args, **kwargs):
            """this is a dummy function needed to apply kwargsChecker"""
            return

        checkKeywordInput(**rawOptions)
        options = CaseInsensitiveDict(rawOptions)
    else:
        moduleSchema = _resolveModuleKeywordSchema(topLevelKeyword, topLevelOptions, keyword)
        options = _coerceKnownFields(moduleSchema, rawOptions)

        requiredNames, optionalNames = _requiredAndOptionalNames(moduleSchema)

        @caseInsensitiveKwargsChecker(requiredNames, optionalNames)
        def checkKeywordInput(*args, **kwargs):
            """this is a dummy function needed to apply kwargsChecker"""
            return

        try:
            checkKeywordInput(**options)
        except ValueError as e:
            # A step action can be updated in a subsequent step by repeating the module
            # level keyword with the same name as previously defined. Updates are validated
            # against the dedicated 'update<keyword>' schema, if this step action provides one.
            updateSchema = None
            if (
                strCaseCmp(topLevelKeyword, "step")
                and "name" in rawOptions
                and rawOptions["name"].casefold()
                in [  # check if a step action with the same name already exists
                    item["name"].casefold()
                    for step in fileDict["step"]
                    if keyword in step["moduleOptions"]
                    for item in step["moduleOptions"][keyword]
                    if "name" in item
                ]
            ):
                updateSchema = _updateStepActionSchema(keyword)

            if updateSchema is not None:
                options = _coerceKnownFields(updateSchema, rawOptions)
                updateRequired, updateOptional = _requiredAndOptionalNames(updateSchema)

                @caseInsensitiveKwargsChecker(updateRequired, updateOptional)
                def checkUpdateKeywordInput(*args, **kwargs):
                    """this is a dummy function needed to apply kwargsChecker"""
                    return

                try:
                    checkUpdateKeywordInput(**options)
                except ValueError as e2:
                    e2.args = (
                        f"Error during updating stepaction {moduleLevelKeywordIdentifier}{keyword}, "
                        f"name={options['name']}: " + e2.args[0],
                    )
                    raise e2
            else:
                e.args = (
                    f"Error during parsing of module level keyword {moduleLevelKeywordIdentifier}{keyword}: "
                    + e.args[0],
                )
                raise e

    options["inputFile"] = fileName  # save also the filename of the original inputfile!

    # Unlike a top-level keyword, no `>>` sub-keyword in the whole grammar ever expects its own
    # plain datalines -- a sub-keyword's own scalar options are declared on its schema's regular
    # fields, never on a dataline payload of its own. So a `datalines` key must never be added here
    # for any `>>` block: doing so would break e.g. `>>bodyforce`, whose `fromStepActionDefinition`
    # hands the whole definition dict to `buildSchemaFromOptions` without stripping parser
    # bookkeeping keys first.

    return keyword, options


def parseInputFile(
    fileName: str,
    currentKeyword: str = None,
    existingFileDict: CaseInsensitiveDict = None,
) -> CaseInsensitiveDict:
    """Parse an Abaqus-like input file to generate a dictionary with its content.

    Parameters
    ----------
    fileName
        The name of the file to parse.
    currentKeyword
        If nested parsing is performed by using ``*include``, this option tells which
        keyword is currently active.
    existingFileDict
        An existing dictionary to append. If Nonde, a new dictionary is created.

    Returns
    -------
    CaseInsensitiveDict
        The parsed input file.
    """

    if not existingFileDict:
        fileDict = CaseInsensitiveDict({name: [] for name in registry.availableNames("keyword")})
    else:
        fileDict = existingFileDict

    keyword = currentKeyword
    with open(fileName) as f:
        # filter out empty lines and comments
        lines = (line.strip() for line in f)
        lines = (line for line in lines if line and not line.startswith("**"))

        for line in lines:
            if line.startswith("*"):  # line is keywordline
                lastKeyword = keyword
                keyword, options = parseKeywordLine(line, fileName)
                options["moduleOptions"] = dict()

                # special treatment for *include:
                if strCaseCmp(keyword, "include"):
                    includeFile = options["input"]
                    parseInputFile(
                        join(dirname(fileName), includeFile),
                        currentKeyword=lastKeyword,
                        existingFileDict=fileDict,
                    )
                    keyword = lastKeyword
                else:
                    fileDict[keyword].append(options)

            elif line.startswith(moduleLevelKeywordIdentifier):  # line is a module level keyword line
                moduleKeyword, moduleOptions = parseModuleKeywordLine(line, fileName, keyword, options, fileDict)

                if moduleKeyword in fileDict[keyword][-1]["moduleOptions"]:
                    fileDict[keyword][-1]["moduleOptions"][moduleKeyword].append(moduleOptions)
                else:
                    fileDict[keyword][-1]["moduleOptions"].update({moduleKeyword: [moduleOptions]})

            # else splitLineAtCommas(line)[]  # line is a module level keyword line

            else:  # line is a data line
                if not _expectsPlainDatalines(keyword, options):
                    raise ValueError(
                        f"Error during parsing of keyword {keywordIdentifier}{keyword}: {keywordIdentifier}{keyword} expects no data lines"
                    )
                fileDict[keyword][-1]["datalines"].append(line)

    return fileDict
