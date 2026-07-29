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

"""Adjust a solver's or output manager's own options mid-job, by name:

.. code-block:: edelweiss

    *solver, name=mySolver, solver=NIST
    *step, solver=mySolver
    >>options, name=mySolver, extrapolation=linear

``name`` must be the name of an already-declared ``*solver`` or ``*output`` instance -- the same
name that block gave it, not a category or type tag. This step action resolves it directly against
``model.solvers``/``model.outputManagers`` (:func:`_resolveTarget`), validates whichever options are
actually present against *that instance's own* ``type(target).schema`` (:func:`coercePresentOptions`
-- only present keys, no missing-required check, since an override is by definition partial), and
applies them immediately via ``target.applyOptionsOverride(...)``.

This replaces a name-blind, category-string mechanism (``getOptionsOfCategory``, matched against a
solver's ``self.identification`` or an output manager's own name-or-a-hardcoded-fallback) that let a
hand-maintained category tag drift from the object it was supposed to identify -- which is exactly
how ``NISTPArcLength``'s two bespoke options ended up registered, and resolved, under the unrelated
string ``"NISTArcLength"`` rather than its own ``identification``. A name is already unique and
already declared; there is no second tag to keep in sync with it.

The ``>>options`` keyword itself still needs every solver's and output manager's option names
pre-declared on it (:func:`registerSchemaOptions`), because the ``.inp`` grammar is validated
statically against a keyword's own known arguments (see ``utils/inputlanguage.py``) -- a single
``>>options, name=X, ...`` line cannot itself carry the information "only validate the keys ``X``'s
own type declares" at parse time, before ``name`` has even been resolved. What changes is *how* that
declaration happens: one call per *schema* (:func:`registerSchemaOptions`, called once by each
solver/output manager module with its own ``schema`` class) instead of one call per *option*
(the old ``registerOptionsArg``, called by hand once per field) -- so a schema and its grammar
registration cannot independently drift, because there is only one source now.
"""

import dataclasses

from edelweissfe.stepactions.base.stepactionbase import StepActionBase
from edelweissfe.utils.inputlanguage import InputLanguage
from edelweissfe.utils.misc import withoutParserBookkeepingKeys
from edelweissfe.utils.schema import (
    coercePresentOptions,
    fieldSchemaMeta,
    scalarOptionNames,
)

inputLanguage = InputLanguage()

#: The description of the ``options`` keyword. A module-level constant because
#: :func:`_ensureOptionsKeyword` may declare the keyword from either of two call sites, and the two
#: must not be able to disagree -- the rendered grammar surface would then depend on import order.
_OPTIONS_KEYWORD_DESCRIPTION = (
    "Adjust a solver's or output manager's own options mid-job. 'name' must be the name an "
    "already-declared *solver or *output block gave it; every other option is validated against "
    "that specific instance's own type."
)

documentation = []


def _ensureOptionsKeyword():
    """Declare the ``options`` keyword on every registered step type, at most once each.

    Idempotent and safe to call repeatedly, because it has to be called from two places.

    Every other step action declares its keywords in a one-shot ``for module in modules:`` loop at
    import time, which is only correct if the step type modules are already registered by then -- as
    they are when the input file parser does the importing. This module cannot rely on that, because
    it is also imported *indirectly*, by any module calling :func:`registerSchemaOptions` at its own
    import time (``outputmanagers.ensight`` and four solvers do). Reached that way before the step
    types exist, the one-shot loop declared nothing, and -- since the module was then in
    ``sys.modules`` -- the parser's later import did not re-run it, so the keyword stayed undeclared
    and the *next* ``registerSchemaOptions`` call died with
    ``ValueError: options is not a valid argument``. Measured: importing either
    ``edelweissfe.outputmanagers.ensight`` or this module before
    ``edelweissfe.utils.inputfileparser`` broke the parser outright, while the reverse order worked.

    Declaring lazily removes the dependence on hitting that window, rather than papering over it: the
    keyword is declared by whichever of the two call sites first runs at a point where there is a
    step type to declare it on. ``documentation`` is populated here too, so the rendered grammar
    surface cannot differ by import order either.
    """

    if "step" not in inputLanguage:
        return

    for stepModule in inputLanguage["step"].modules:
        if "options" in [keyword.name.casefold() for keyword in stepModule.keywords]:
            continue

        keyword = stepModule.addOptionalKeyword("options", _OPTIONS_KEYWORD_DESCRIPTION)
        keyword.addRequiredArg(
            "name", "The name of the already-declared solver or output manager this block configures.", str
        )

        documentation.append(keyword)


_ensureOptionsKeyword()


def _fieldRuntimeDefault(field: dataclasses.Field):
    """The value a schema field actually defaults to at runtime, for documentation purposes only.

    Parameters
    ----------
    field
        A field of a schema dataclass.

    Returns
    -------
    Any
        The field's default (calling its ``default_factory`` if that is how it is declared), or
        ``None`` if the field is required and therefore has no default to show.
    """
    if field.default is not dataclasses.MISSING:
        return field.default
    if field.default_factory is not dataclasses.MISSING:
        return field.default_factory()
    return None


def registerSchemaOptions(schemaCls: type) -> None:
    """Register every scalar option of an L2 schema onto the shared ``>>options`` keyword.

    This is what lets a solver's or output manager's own options be *written* on an ``>>options``
    block at all: the parser validates a keyword's arguments against a static, pre-declared grammar,
    so every name that may ever appear on this shared keyword must be known before the file is
    parsed -- the same reason the old, deleted ``registerOptionsArg`` existed, one call per option.
    This registers a whole schema in one call instead, so the registration cannot independently drift
    from the schema describing what the resolved target actually accepts (see the module docstring).

    Sub-keyword fields (:func:`~edelweissfe.utils.schema.subKeywordField`) are skipped: they are
    filled from nested ``>>`` blocks of their own module, not from a flat ``name=value`` pair here,
    and are therefore not reachable through ``>>options`` regardless.

    Idempotent per option name: a name already registered (by this schema or, for a subclass
    schema such as ``NISTPArcLengthSchema``, by an ancestor's own registration) is left alone rather
    than appended a second time, since :meth:`~edelweissfe.utils.inputlanguage.Keyword.addOptionalArg`
    has no such guard of its own.

    The runtime default of every registered arg is always ``None`` -- never the schema's own default
    -- which is what lets :meth:`StepAction.fromStepActionDefinition`/
    :meth:`StepAction.updateStepActionFromDefinition` tell "the user wrote this" apart from "some
    other module's option that happens to share this keyword" (see :func:`_writtenOptions`). The
    schema's real default is rendered as ``documentedDefault`` instead, so the generated docs still
    show what actually takes effect.

    Parameters
    ----------
    schemaCls
        A frozen dataclass whose fields were declared via
        :func:`~edelweissfe.utils.schema.schemaField`.
    """

    if "step" not in inputLanguage:
        return

    # The keyword may not exist yet: this module can be imported, indirectly and via this very
    # function, before any step type is registered. See _ensureOptionsKeyword.
    _ensureOptionsKeyword()

    for optionName, field in scalarOptionNames(schemaCls).items():
        meta = fieldSchemaMeta(field)
        for stepModule in inputLanguage["step"].modules:
            keyword = stepModule.getKeyword("options")
            if any(arg.name.casefold() == optionName.casefold() for arg in keyword.optionalArgs):
                continue
            keyword.addOptionalArg(
                optionName, meta.description, meta.dtype, None, documentedDefault=_fieldRuntimeDefault(field)
            )


def _writtenOptions(definition: dict) -> dict:
    """Recover the options a user actually wrote in a parsed ``>>options`` block.

    Every solver's and output manager's options share this one keyword (:func:`registerSchemaOptions`
    registers them all, with a runtime default of ``None``), so a parsed block carries every
    registered name, ``None`` where the user did not write it. Only the non-``None`` entries -- the
    ones the user actually set -- are meaningful to validate against the resolved target's own
    schema; the rest belong to *other* modules and would otherwise fail that validation outright.

    Parameters
    ----------
    definition
        The parsed option mapping of one ``>>options`` block.

    Returns
    -------
    dict
        The options the user actually wrote, without the parser's bookkeeping keys, the ``name``
        this step action is itself identified by, or any ``None`` placeholder contributed by a
        foreign module sharing this keyword.
    """

    written = withoutParserBookkeepingKeys(definition)
    written.pop("name", None)
    return {key: value for key, value in written.items() if value is not None}


def _resolveTarget(name: str, model):
    """Resolve an ``>>options`` block's ``name`` to the solver or output manager it configures.

    Parameters
    ----------
    name
        The name to resolve, exactly as an earlier ``*solver`` or ``*output`` block declared it.
    model
        The model tree, whose ``solvers``/``outputManagers`` attributes (set by the driver before
        steps are generated -- see ``drivers/inputfiledrivensimulation.py``) are searched.

    Returns
    -------
    Any
        The resolved solver or output manager instance.

    Raises
    ------
    ValueError
        If ``name`` resolves to neither, or -- searched deliberately, rather than returning
        whichever is found first -- to both at once.
    """

    solver = model.solvers.get(name)
    outputManager = model.outputManagers.get(name)

    if solver is not None and outputManager is not None:
        raise ValueError(
            f"'{name}' names both a solver and an output manager; '>>options, name={name}' cannot tell "
            "which one is meant. Rename one of them."
        )
    if solver is None and outputManager is None:
        raise ValueError(f"'{name}' is not the name of any declared solver or output manager.")

    target = solver if solver is not None else outputManager
    if type(target).schema is None:
        raise ValueError(
            f"'{name}' ({type(target).__name__}) declares no option schema, so '>>options, name={name}' "
            "has nothing to validate its options against."
        )
    return target


class StepAction(StepActionBase):
    """Adjust a solver's or output manager's own options mid-job.

    Unlike every other ported step action, this one has no scalar schema of its own to declare
    (:attr:`schema` stays the :class:`~edelweissfe.utils.schema.OptionSchemaProvider` default of
    ``None``): it is a dispatcher onto *another* object's schema, resolved by ``name`` at
    :meth:`fromStepActionDefinition` time, not a leaf option consumer -- there is nothing this class
    itself accepts beyond the ``name`` that identifies both the step action and its target.

    The constructor is typed: it takes the resolved target object directly, applying the override
    once immediately (mirroring every other step action's ``__init__`` -> ``updateStepAction``
    convention) rather than only on a later re-declaration.

    Parameters
    ----------
    name
        The name of this step action, identical to the resolved target's own name.
    target
        The solver or output manager this block configures, already resolved by
        :meth:`fromStepActionDefinition`.
    overrides
        The options to apply, coerced and validated against ``type(target).schema``.
    """

    def __init__(self, name: str, target, overrides: dict):
        self.name = name
        self._target = target
        self.updateStepAction(overrides)

    @classmethod
    def fromStepActionDefinition(cls, name, definition, jobInfo, model, fieldOutputController, journal):
        """Build this step action from a parsed ``>>options`` definition. See
        :class:`StepActionBase` for why this is separate from ``__init__``.

        ``name`` (already available as this method's own argument, and identical to the target's own
        name) resolves the target via :func:`_resolveTarget`; the remaining, actually-written options
        (:func:`_writtenOptions`) are then validated against that target's own
        ``type(target).schema``."""

        target = _resolveTarget(name, model)
        overrides = coercePresentOptions(type(target).schema, _writtenOptions(definition))
        return cls(name, target, overrides)

    def updateStepActionFromDefinition(self, definition, jobInfo, model, fieldOutputController, journal):
        """Update from a parsed ``>>options`` definition re-declared in a later step.

        The target was already resolved at construction and cannot change (a step action keeps the
        name it was created with), so this only re-validates and re-applies -- no need to resolve
        ``name`` again."""

        overrides = coercePresentOptions(type(self._target).schema, _writtenOptions(definition))
        self.updateStepAction(overrides)

    def updateStepAction(self, overrides: dict):
        """Apply a validated, partial override onto the resolved target.

        Parameters
        ----------
        overrides
            Maps the target's own schema field name to its new, already-coerced value.
        """

        self._target.applyOptionsOverride(overrides)
