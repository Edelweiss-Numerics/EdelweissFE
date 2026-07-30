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
import textwrap
from os.path import dirname, join

from edelweissfe.config import registry
from edelweissfe.config.registry import RegistryLookupError
from edelweissfe.utils.caseinsensitivedict import CaseInsensitiveDict
from edelweissfe.utils.inputlanguage import (
    InputLanguage,
    keywordIdentifier,
    moduleLevelKeywordIdentifier,
)
from edelweissfe.utils.misc import (
    caseInsensitiveKwargsChecker,
    caseInsensitiveRequiredArgsChecker,
    convertAssignmentsToCaseInsensitiveStringDictionary,
    splitLineAtCommas,
    strCaseCmp,
    typeString,
)
from edelweissfe.utils.schema import (
    coerceValue,
    datalineFieldMeta,
    fieldSchemaMeta,
    subKeywordFieldNames,
)

#: Top-level keywords whose grammar is completed by a further name/generator-dispatched target's
#: own schema (see ``PLAN_INPUT_SYSTEM_UNIFICATION.md``, U3d-1) -- maps the keyword's casefolded
#: name to ``(registry category, the keyword-line option name carrying the dispatch value)``.
#: ``step`` is included only for :func:`_expectsPlainDatalines`'s sake -- its ``>>`` grammar is
#: resolved separately, always via the ``stepaction`` category regardless of the step *type* (see
#: :func:`_resolveModuleKeywordSchema`), because every step action registers identically onto every
#: step type in the legacy grammar (``modules = inputLanguage["step"].modules`` in each
#: ``stepactions/*.py``).
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
    in a later step, or ``None`` if this step action has no such companion (U3b,
    ``PLAN_INPUT_SYSTEM_UNIFICATION.md``).

    Only three step actions declare one at all (``dirichlet``, ``distributedload``,
    ``nodeforces``) -- every other step action either has no update path (a re-declaration must
    restate every required argument again) or -- ``options`` -- validates dynamically and never
    reaches this function at all (see the ``isDynamicOptionsKeyword`` branch of
    :func:`parseModuleKeywordLine`). A small explicit table, not a registry category: these
    ``Update<Name>Schema`` classes are documentation/parser-validation companions the L1 step action
    itself never references (see e.g. ``stepactions/dirichlet.py``'s own docstring), so they have no
    home in the ``stepaction`` registry, which maps to the L1 classes.

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

    Deliberately **not** :func:`~edelweissfe.utils.schema.scalarOptionNames`: that function excludes
    ``structuralOnly`` fields (an L4 adapter pops them before ``buildSchemaFromOptions`` ever sees
    them), but the *parser* must still recognize and require them exactly as the legacy
    ``InputFileKeyword``/``Module`` flat ``requiredArgs``/``optionalArgs`` lists did -- those drew no
    such distinction at all (``structuralOnly``/``optionsOverrideOnly``/``updateOnly`` are new,
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
    grammar and were never declared as a real ``Module.addRequiredArg``/``addOptionalArg`` (whose
    presence is what implies a dispatch target actually expects plain datalines, see
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
    """Split :func:`_allScalarFields`'s option names into required/optional lists, mirroring the
    ``[kw.name for kw in kw.requiredArgs]``/``[kw.name for kw in kw.optionalArgs]`` pairs the legacy
    parser passed to :func:`~edelweissfe.utils.misc.caseInsensitiveKwargsChecker`.

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
    fields; every other key is left untouched, as a raw string -- mirroring
    ``castKwargsValuesAndAddDefaults``'s behaviour of only ever touching the keys it knows about.

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
    """Resolve the L2 schema of the ``type=``/``generator=``-dispatched target hosted by a
    top-level keyword's line, so a value meant for *that* target's own options (e.g. a plane
    section's ``thickness``) written directly on the ``*section`` line is not rejected as unknown --
    mirroring the legacy hard-coded ``section``/``plane`` fallback in ``parseKeywordLine``,
    generalized to every dispatch keyword via the registry.

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
    except RegistryLookupError:
        return None
    return schema


def _resolveModuleKeywordSchema(topLevelKeyword: str, topLevelOptions: dict, moduleKeyword: str) -> type | None:
    """Resolve the L2 schema describing one ``>>`` sub-keyword block's own grammar, mirroring the
    legacy ``Module.getKeyword(keyword)`` lookup (U3d-1, ``PLAN_INPUT_SYSTEM_UNIFICATION.md``).

    For ``*step``, every step action registers identically onto every step type
    (``modules = inputLanguage["step"].modules`` in each ``stepactions/*.py``), so its ``>>``
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
    accepts plain (non-``>>``) datalines, mirroring the legacy ``Module``'s
    ``expectsRequiredDatalines``/``expectsOptionalDatalines`` -- which ``Module.addRequiredArg``/
    ``addOptionalArg`` set as a side effect (a module's *own* options, unlike a top-level keyword's,
    are conventionally supplied via datalines) alongside ``addRequiredDatalines``/
    ``addOptionalDatalines`` proper.

    Parameters
    ----------
    schemaCls
        The dispatch target's own schema, or ``None``.

    Returns
    -------
    bool
        Whether a plain dataline is valid here. ``True`` when ``schemaCls`` is ``None``: a dispatch
        target with no L2 schema at all yet is a genuinely raw-datalines case in every instance that
        exists today (``executePythonCode``'s Python source), so this conservatively allows rather
        than rejecting every real ``.inp`` file exercising it.
    """
    if schemaCls is None:
        return True
    return bool(_realScalarFields(schemaCls)) or datalineFieldMeta(schemaCls) is not None


def _expectsPlainDatalines(keyword: str, options: dict) -> bool:
    """Whether a plain (non-``>>``) line following a ``*keyword`` block's own line is a valid
    dataline for it, mirroring ``InputFileKeyword``/``Module``'s
    ``expectsRequiredDatalines``/``expectsOptionalDatalines`` flags (U3d-1,
    ``PLAN_INPUT_SYSTEM_UNIFICATION.md``).

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
        # Neither step type (`adaptive`/`adaptiveForExplicitSimulations`) has an L2 schema yet
        # (PLAN_INPUT_SYSTEM_UNIFICATION.md's U2 recon notes) -- both declare their incrementation
        # options via `Module.addOptionalArg`, which always implies `expectsOptionalDatalines=True`
        # regardless of type, so mirror that directly rather than rejecting every real .inp file
        # that writes step incrementation options (maxInc=..., ...) as plain datalines.
        return True

    category, optionName = dispatch
    dispatchValue = options.get(optionName)
    if not dispatchValue:
        return False
    try:
        _, dispatchSchema = registry.lookup(category, dispatchValue)
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
        # (see stepactions/options.py, U3c) -- only 'name' is enforced here; every other key is
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
    # plain datalines: `Module.addRequiredArg`/`addOptionalArg` (whose side effect is what makes a
    # *module*'s `expectsRequiredDatalines`/`expectsOptionalDatalines` True) is not how a `>>`
    # sub-keyword's own args are declared -- those go through
    # `InputFileKeyword.addRequiredArg`/`addOptionalArg` instead (via `Module.addOptionalKeyword`),
    # which sets no such flag. So legacy never added a `datalines` key here for any `>>` block, and
    # this must not either -- doing so unconditionally regressed e.g. `>>bodyforce`, whose
    # `fromStepActionDefinition` hands the whole definition dict to `buildSchemaFromOptions` without
    # stripping parser bookkeeping keys first.

    return keyword, options


inputLanguage = InputLanguage()

kw = inputLanguage.addKeyword("element", "definition of element(s)")
kw.addRequiredArg("type", "assign one of the types defined in the elementlibrary", str)
kw.addOptionalArg("elSet", "name of elSet to be created", str, None)
kw.addOptionalArg(
    "provider",
    "provider (library) for the element type. Default: Marmot",
    str,
    "Marmot",
)
kw.addRequiredDatalines("Abaqus like element definition lines", "")

kw = inputLanguage.addKeyword("elSet", "definition of an element set")
kw.addRequiredArg("elSet", "name", str)
kw.addOptionalArg(
    "generate",
    "set True to generate from data line 1: start-element, end-element, step",
    bool,
    False,
)
kw.addRequiredDatalines("Abaqus like element set definition lines", "")

kw = inputLanguage.addKeyword("node", "definition of nodes")
kw.addOptionalArg("nSet", "name of nSet to be created", str, None)
kw.addRequiredDatalines("Abaqus like node definition lines: label, x, [y], [z]", "")

kw = inputLanguage.addKeyword("nSet", "definition of an element set")
kw.addRequiredArg("nSet", "name", str)
kw.addOptionalArg(
    "generate",
    "set True to generate from data line 1: start-node, end-node, step",
    bool,
    False,
)
kw.addRequiredDatalines("Abaqus like node set definition lines", "")

kw = inputLanguage.addKeyword("surface", "definition of surface set")
kw.addRequiredArg("name", "name", str)
kw.addRequiredArg("type", "type of surface (currently 'element' only)", str)
kw.addRequiredDatalines("Abaqus like definition. Type 'element': elSet, faceID", "")

"""
*section
"""
kw = inputLanguage.addKeyword("section", "definition of a section")
kw.addRequiredArg("name", "name", str)
kw.addRequiredArg("material", "associated id of defined material", str)
kw.addRequiredArg("type", "type of the section", str)
kw.addRequiredDatalines("list of associated element sets", "")

# kw.addOptionalArg("thickness", "associated element thickness", float, 1.0)
# kw.addOptionalArg("density", "associated element density", float, 1.0)

# isort: off
from edelweissfe.sections.solid import inputLanguage  # noqa: F811,E402
from edelweissfe.sections.plane import inputLanguage  # noqa: F811,E402

# isort: on

"""
*elementproperty
"""
kw = inputLanguage.addKeyword("elementproperty", "define element property values")
kw.addRequiredArg("elSet", "name of element set", str)
kw.addRequiredArg("propertyName", "name of the property", str)
kw.addRequiredDatalines("value(s) of the property", "")

"""
*material
"""
kw = inputLanguage.addKeyword("material", "definition of a material")
kw.addRequiredArg("name", "name of material", str)
kw.addRequiredArg("id", "id of material", str)
kw.addOptionalArg("provider", "material provider", str, "marmotmaterial")
kw.addRequiredDatalines("material properties", "")
# kw.addOptionalArg("statevars", , , None)

"""
*advancedmaterial
"""
kw = inputLanguage.addKeyword("advancedmaterial", "definition of an advanced material")
kw.addRequiredArg("name", "name of material", str)
kw.addRequiredArg("id", "id of material", str)
kw.addOptionalArg("provider", "material provider", str, "marmotmaterial")
kw.addRequiredDatalines("material properties", "")

"""
*fieldOutput
"""
kw = inputLanguage.addKeyword("fieldOutput", "define fieldoutput, which is used by outputmanagers")

# isort: off
from edelweissfe.utils.fieldoutput import inputLanguage  # noqa: F811,E402

# isort: on

"""
*analyticalField
"""
kw = inputLanguage.addKeyword("analyticalField", "define an analytical field")
kw.addRequiredArg("name", "name of analytical field", str)
kw.addRequiredArg("type", "type of analytical field", str)
# kw.addRequiredDatalines("definition lines", "")

# isort: off
from edelweissfe.analyticalfields.randomscalar import inputLanguage  # noqa: F811,E402
from edelweissfe.analyticalfields.fromvtk import inputLanguage  # noqa: F811,E402
from edelweissfe.analyticalfields.scalarexpression import inputLanguage  # noqa: F811,E402

# isort: on

"""
*job
"""
kw = inputLanguage.addKeyword("job", "definition of an analysis job")
kw.addRequiredArg("domain", "define spatial domain: 1d, 2d, 3d", str)
kw.addOptionalArg("startTime", "(optional) start time of job", float, 0.0)
kw.addOptionalArg("name", "Name of job.", str, "defaultJob")
kw.addOptionalArg("solver", "(deprecated) define the solver to be used", str, "NIST")

"""
*solver
"""
kw = inputLanguage.addKeyword("solver", "define a solver")
kw.addRequiredArg("name", "solver name", str)
kw.addRequiredArg("solver", "solver type", str)
kw.addOptionalDatalines("define options which are passed to the respective solver instance.", "")

"""
*step
"""
kw = inputLanguage.addKeyword("step", "define steps")
kw.addRequiredArg("solver", "solver to be used", str)
kw.addOptionalArg("type", "step type", str, "adaptive")

# isort: off
from edelweissfe.steps.adaptivestep import inputLanguage  # noqa: F811,E402
from edelweissfe.steps.adaptivestepforexplicitsimulations import inputLanguage  # noqa: F811,E402

from edelweissfe.stepactions.bodyforce import inputLanguage  # noqa: F811,E402
from edelweissfe.stepactions.changematerialproperty import inputLanguage  # noqa: F811,E402
from edelweissfe.stepactions.dirichlet import inputLanguage  # noqa: F811,E402
from edelweissfe.stepactions.distributedload import inputLanguage  # noqa: F811,E402
from edelweissfe.stepactions.geostatic import inputLanguage  # noqa: F811,E402
from edelweissfe.stepactions.indirectcontractioncontrol import inputLanguage  # noqa: F811,E402
from edelweissfe.stepactions.indirectcontrol import inputLanguage  # noqa: F811,E402
from edelweissfe.stepactions.initializematerial import inputLanguage  # noqa: F811,E402
from edelweissfe.stepactions.modelupdate import inputLanguage  # noqa: F811,E402
from edelweissfe.stepactions.nodeforces import inputLanguage  # noqa: F811,E402
from edelweissfe.stepactions.setfield import inputLanguage  # noqa: F811,E402
from edelweissfe.stepactions.setinitialconditions import inputLanguage  # noqa: F811,E402

from edelweissfe.stepactions.options import _ensureOptionsKeyword, inputLanguage  # noqa: F811,E402

# Guarantees the 'options' keyword is declared on every step type at this exact point in the
# import sequence (every step type + step action module above is already loaded), regardless of
# whether 'edelweissfe.stepactions.options' happened to be imported earlier by some other caller
# before 'step' existed -- in which case its own module-level call above already ran and no-opped,
# and would otherwise never get a second chance (the module is cached in sys.modules by then, so
# the 'from ... import' above does not re-run its body). See _ensureOptionsKeyword's docstring.
_ensureOptionsKeyword()

# isort: on

"""
*output
"""
kw = inputLanguage.addKeyword("output", "define an output module")
kw.addRequiredArg("type", "output module", str)
kw.addOptionalArg("name", "name of output manager", str, None)
# kw.addOptionalDatalines("definition lines", "")

# isort: off
from edelweissfe.outputmanagers.computetimemonitor import inputLanguage  # noqa: F811,E402
from edelweissfe.outputmanagers.conditionalstop import inputLanguage  # noqa: F811,E402
from edelweissfe.outputmanagers.ensight import inputLanguage  # noqa: F811,E402
from edelweissfe.outputmanagers.fractureenergyintegrator import inputLanguage  # noqa: F811,E402
from edelweissfe.outputmanagers.meshdatatofile import inputLanguage  # noqa: F811,E402
from edelweissfe.outputmanagers.meshplot import inputLanguage  # noqa: F811,E402
from edelweissfe.outputmanagers.monitor import inputLanguage  # noqa: F811,E402
from edelweissfe.outputmanagers.plotalongpath import inputLanguage  # noqa: F811,E402
from edelweissfe.outputmanagers.statusfile import inputLanguage  # noqa: F811,E402
from edelweissfe.outputmanagers.timemonitor import inputLanguage  # noqa: F811,E402

# isort: on

"""
*updateConfiguration
"""
kw = inputLanguage.addKeyword("updateConfiguration", "update a configuration")
kw.addRequiredArg("configuration", "name of configuration to be changed", str)
kw.addRequiredDatalines("keyword arguments", "")

"""
*modelGenerator
"""
kw = inputLanguage.addKeyword("modelGenerator", "define a model generator, loaded from a module")
kw.addRequiredArg("name", "name of the generator", str)
kw.addRequiredArg("generator", "name of generator module", str)
kw.addOptionalArg(
    "executeAfterManualGeneration",
    "Delay the execution of the generator after model generation",
    bool,
    False,
)
# kw.addRequiredDatalines("keyword arguments", "")

# isort: off
# from edelweissfe.generators.abqmodelconstructor import inputLanguage  # noqa: F811,E402
from edelweissfe.generators.boxgen import inputLanguage  # noqa: F811,E402
from edelweissfe.generators.cubit import inputLanguage  # noqa: F811,E402
from edelweissfe.generators.executepythoncode import inputLanguage  # noqa: F811,E402
from edelweissfe.generators.findclosestnode import inputLanguage  # noqa: F811,E402
from edelweissfe.generators.pipegen import inputLanguage  # noqa: F811,E402
from edelweissfe.generators.planerectquad import inputLanguage  # noqa: F811,E402
from edelweissfe.generators.cuboidlatticegenerator import inputLanguage  # noqa: F811,E402
from edelweissfe.generators.discreterigidbodygenerator import inputLanguage  # noqa: F811,E402
from edelweissfe.generators.surfaceelementgenerator import inputLanguage  # noqa: F811,E402

# isort: on

"""
*constraint
"""
kw = inputLanguage.addKeyword("constraint", "define a constraint")
kw.addRequiredArg("type", "constraint type", str)
kw.addRequiredDatalines("definition of the constraint", "")
kw.addOptionalArg("name", "name of the constraint", str, None)

# isort: off
from edelweissfe.constraints.equalvaluelagrangian import inputLanguage  # noqa: F811,E402
from edelweissfe.constraints.equalvaluepenalty import inputLanguage  # noqa: F811,E402
from edelweissfe.constraints.linearizedrigidbody import inputLanguage  # noqa: F811,E402
from edelweissfe.constraints.penaltyindirectcontrol import inputLanguage  # noqa: F811,E402
from edelweissfe.constraints.rigidbody import inputLanguage  # noqa: F811,E402
from edelweissfe.constraints.directionalspringpenalty import inputLanguage  # noqa: F811,E402
from edelweissfe.constraints.nodetorigidsurfacepenalty import inputLanguage  # noqa: F811,E402
from edelweissfe.constraints.nodetodiscreterigidbodypenalty import inputLanguage  # noqa: F811,E402
from edelweissfe.constraints.nodetodeformablesurfacepenalty import inputLanguage  # noqa: F811,E402
from edelweissfe.constraints.tie import inputLanguage  # noqa: F811,E402
from edelweissfe.constraints.hangingnode import inputLanguage  # noqa: F811,E402
from edelweissfe.constraints.amrtransparencyprobe import inputLanguage  # noqa: F811,E402

# isort: on

"""
*modelModifier
"""
kw = inputLanguage.addKeyword("modelModifier", "define a model modifier")
kw.addRequiredArg("type", "model modifier type", str)
kw.addRequiredDatalines("definition of the model modifier", "")
kw.addOptionalArg("name", "name of the model modifier", str, None)

# isort: off
from edelweissfe.modelmodifiers.adaptivity.hadaptivity import inputLanguage  # noqa: F811,E402

# isort: on

"""
*configurePlots
"""
kw = inputLanguage.addKeyword("configurePlots", "customize the figures and axes")
kw.addRequiredDatalines("key=value pairs for configuration of figures and axes", "")

"""
*exportPlots
"""
kw = inputLanguage.addKeyword("exportPlots", "export your figures")
kw.addRequiredDatalines("key=value pairs for exporting of figures and axes", "")

"""
*include
"""
kw = inputLanguage.addKeyword("include", "load contents of extra file")
kw.addRequiredArg("input", "path to file (use relative path to current .inp)", str)


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
        fileDict = CaseInsensitiveDict({kw.name: [] for kw in inputLanguage})
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


def printKeywords():
    """Print the input file language set."""

    kwString = "    {:}    "
    kwDataString = "        {:22}{:20}"

    wrapper = textwrap.TextWrapper(width=80, replace_whitespace=False)
    # for kw, (kwDoc, optiondict) in sorted(inputLanguage.items()):
    for kw in inputLanguage:
        wrapper.initial_indent = kwString.format(kw.name)
        wrapper.subsequent_indent = " " * len(wrapper.initial_indent)
        print(wrapper.fill(kw.description))
        # print("")
        for arg in kw.requiredArgs:
            wrapper.initial_indent = kwDataString.format(arg.name, typeString(arg.dtype))
            wrapper.subsequent_indent = " " * len(wrapper.initial_indent)
            print(wrapper.fill(arg.description))
        for arg in kw.optionalArgs:
            wrapper.initial_indent = kwDataString.format(arg.name, typeString(arg.dtype))
            wrapper.subsequent_indent = " " * len(wrapper.initial_indent)
            print(wrapper.fill(arg.description))
        print("\n")
