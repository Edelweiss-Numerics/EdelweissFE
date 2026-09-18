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

Resolving directly by name against the target's own schema means there is no separate category tag
that could drift out of sync with the object it is supposed to identify.

The ``>>options`` keyword's own grammar is validated **dynamically**, not against a statically
pre-declared list of every solver's and output manager's option names: at parse time, before
``name`` has even been resolved, the parser's dedicated ``isDynamicOptionsKeyword`` branch (see
``utils/inputfileparser.py::parseModuleKeywordLine``) only enforces that ``name`` itself is present
-- every other ``key=value`` pair is accepted unvalidated and handed on raw. Real validation happens
once ``name`` resolves to a concrete object, against *that object's own* ``type(target).schema``
(:func:`coercePresentOptions`, in :meth:`StepAction.fromStepActionDefinition` below).
"""

from edelweissfe.stepactions.base.stepactionbase import StepActionBase
from edelweissfe.utils.misc import withoutParserBookkeepingKeys
from edelweissfe.utils.schema import coercePresentOptions

#: The description of the ``options`` keyword, quoted by ``tests/_inputlanguage_snapshot.py`` for
#: the rendered grammar surface. A module-level constant so it cannot drift between the two.
_OPTIONS_KEYWORD_DESCRIPTION = (
    "Adjust a solver's or output manager's own options mid-job. 'name' must be the name an "
    "already-declared *solver or *output block gave it; every other option is validated against "
    "that specific instance's own type."
)


def _writtenOptions(definition: dict) -> dict:
    """Recover the options a user actually wrote in a parsed ``>>options`` block.

    The parser accepts any key beyond ``name`` unvalidated, so a parsed block carries exactly the
    keys the user wrote -- plus the parser's own bookkeeping keys and ``name`` itself, both
    stripped here.

    Parameters
    ----------
    definition
        The parsed option mapping of one ``>>options`` block.

    Returns
    -------
    dict
        The options the user actually wrote, without the parser's bookkeeping keys or the ``name``
        this step action is itself identified by. A defensive ``None``-value filter is kept for a
        programmatic caller that hands in an explicit ``None`` to mean "not set" (the convention
        :func:`~edelweissfe.utils.schema.buildSchemaFromOptions`/:func:`~edelweissfe.utils.schema.coercePresentOptions`
        already apply elsewhere), even though the ``.inp`` parser itself never produces one here.
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

    Unlike every other step action, this one has no scalar schema of its own to declare
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
