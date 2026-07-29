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

"""This stepaction serves as a simple case insensitive container for
storing step options for various modules, grouped by a category, e.g.,

.. code-block:: edelweiss

    *step, solver=mySolver
    >>options, category=NISTSolver, extrapolation=linear

Consumers (e.g., solvers and output managers) declare their available options
via :func:`registerOptionsArg`, and retrieve the options defined for their
category via :func:`getOptionsOfCategory`.
"""

from edelweissfe.stepactions.base.stepactionbase import StepActionBase
from edelweissfe.utils.caseinsensitivedict import CaseInsensitiveDict
from edelweissfe.utils.inputlanguage import InputLanguage
from edelweissfe.utils.misc import strCaseCmp, withoutParserBookkeepingKeys

inputLanguage = InputLanguage()

#: The description of the ``options`` keyword. A module-level constant because
#: :func:`_ensureOptionsKeyword` may declare the keyword from either of two call sites, and the two
#: must not be able to disagree -- the rendered grammar surface would then depend on import order.
_OPTIONS_KEYWORD_DESCRIPTION = (
    "This stepaction serves as a case insensitive container for storing step options for various modules."
)

documentation = []


def _ensureOptionsKeyword():
    """Declare the ``options`` keyword on every registered step type, at most once each.

    Idempotent and safe to call repeatedly, because it has to be called from two places.

    Every other step action declares its keywords in a one-shot ``for module in modules:`` loop at
    import time, which is only correct if the step type modules are already registered by then -- as
    they are when the input file parser does the importing. This module cannot rely on that, because
    it is also imported *indirectly*, by any module calling :func:`registerOptionsArg` at its own
    import time (``outputmanagers.ensight`` and four solvers do). Reached that way before the step
    types exist, the one-shot loop declared nothing, and -- since the module was then in
    ``sys.modules`` -- the parser's later import did not re-run it, so the keyword stayed undeclared
    and the *next* ``registerOptionsArg`` call died with
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
        keyword.addRequiredArg("category", "Option category.", str)

        documentation.append(keyword)


_ensureOptionsKeyword()


def registerOptionsArg(name: str, description: str, dataType: type, documentedDefault=None):
    """Register an available option on the ``options`` keyword of all step types.

    The runtime default is always None, which marks the option as not specified by
    the user; :func:`getOptionsOfCategory` strips unspecified options, such that
    consumers receive only the options which were actually defined in the input file.

    This function is the *only* way anything is registered on the shared ``options`` keyword
    (asserted by ``tests/test_stepoptions.py``), which is what turns that "always None" into a
    structural guarantee rather than a convention every future caller has to remember: the
    strip-``None``\\ s rule above is only correct while no option on this keyword carries a real
    default, and a caller cannot supply one here.

    Parameters
    ----------
    name
        The name of the option.
    description
        The description of the option.
    dataType
        The data type of the option.
    documentedDefault
        The default to *show in the generated documentation*, i.e. the value that takes effect when
        the user does not write this option. It lives in the consuming module (a solver's
        ``SolverSpecificOptions``, say), not here, so it cannot be derived -- pass it explicitly, or
        the docs will claim the option defaults to None. Documentation only: it deliberately does not
        touch the registered arg's runtime default.
    """

    if "step" not in inputLanguage:
        return

    # The keyword may not exist yet: this module can be imported, indirectly and via this very
    # function, before any step type is registered. See _ensureOptionsKeyword.
    _ensureOptionsKeyword()

    for stepModule in inputLanguage["step"].modules:
        stepModule.getKeyword("options").addOptionalArg(
            name, description, dataType, None, documentedDefault=documentedDefault
        )


def getOptionsOfCategory(stepActions, category: str) -> CaseInsensitiveDict:
    """Collect the options of a given category defined via ``options`` step actions.

    Parameters
    ----------
    stepActions
        The collection of step actions, grouped by step action module.
    category
        The requested option category.

    Returns
    -------
    CaseInsensitiveDict
        The options specified by the user for this category. Empty if no ``options``
        step action of this category is present.
    """

    matches = [action for action in stepActions["options"].values() if strCaseCmp(action.category, category)]

    if len(matches) > 1:
        raise ValueError(f"Multiple 'options' step action definitions for category {category}.")

    if not matches:
        return CaseInsensitiveDict()

    return CaseInsensitiveDict(matches[0].options)


class StepAction(StepActionBase):
    """A case insensitive container for storing step options of a category.

    The constructor is typed: it takes the category and a mapping of the options that were actually
    *set*. Recovering that mapping from a parsed ``>>options`` block -- discarding the parser's own
    bookkeeping keys, the ``category`` tag, and the ``None`` defaults every foreign module
    contributes to this shared keyword -- is the job of :meth:`fromStepActionDefinition` below.

    That the recovery lives there and not in :func:`getOptionsOfCategory` is the point of the port.
    The strip used to happen at *read* time, once per consumer query, which meant a function the
    solvers call had to know the shape of a parser dict, and a programmatic caller had to hand this
    class a ``None``-filled mapping to be stripped again later. Now ``options`` holds exactly what a
    consumer receives, and nothing downstream of the constructor mentions the input file.

    Two mechanisms for telling the user's entries apart from the defaults contributed by foreign
    modules used to live side by side here: an ``explicitlySetOptions`` set, fed from the parser's
    ``explicitlySetArgs``, and the strip-``None``\\ s. They were developed independently on
    feat/amr-hanging-nodes and on the step management refactor. ``explicitlySetOptions`` is gone: it
    was written on every update and read by nothing, in EdelweissFE or EdelweissMeshfree, so it
    protected no consumer. The single surviving mechanism relies on the invariant that every option
    on the shared keyword defaults to ``None``, which :func:`registerOptionsArg` guarantees by
    construction rather than by convention.

    Parameters
    ----------
    name
        The name of this step action. The parser uses the category for it, since the ``options``
        keyword declares no ``name`` of its own.
    category
        The category the options belong to, e.g. ``"NISTSolver"``.
    options
        The options set for this category. Only entries a consumer should see -- no ``None``
        placeholders.
    """

    def __init__(self, name: str, category: str, options: dict):
        self.name = name
        self.category = category
        self.options = CaseInsensitiveDict(options)

    @classmethod
    def fromStepActionDefinition(cls, name, definition, jobInfo, model, fieldOutputController, journal):
        """Build this container from a parsed ``>>options`` block. See :class:`StepActionBase` for
        why this is separate from ``__init__``."""

        return cls(name, definition["category"], cls._setOptionsFromDefinition(definition))

    def updateStepActionFromDefinition(self, definition, jobInfo, model, fieldOutputController, journal):
        """Update from an ``>>options`` block of the same category in a later step."""

        self.updateStepAction(self._setOptionsFromDefinition(definition))

    def updateStepAction(self, options: dict):
        """Replace the options set for this category.

        Parameters
        ----------
        options
            The options now set for this category. Entries set previously and absent here are
            dropped, i.e. each declaration states the options of its own step in full.

        Notes
        -----
        Replacing rather than merging is what the pre-port code did, although it read as a merge:
        ``self.options.update(...)`` was handed the parser's ``None``-filled mapping, which carries
        *every* declared arg, so every key was overwritten -- an option set in an earlier step and
        omitted here became ``None`` and was then dropped by the read-time strip. Merging the
        stripped mapping instead would silently leak an earlier step's options into this one.
        """

        self.options = CaseInsensitiveDict(options)

    @staticmethod
    def _setOptionsFromDefinition(definition: dict) -> dict:
        """Recover the options a user actually set from a parsed ``>>options`` block.

        Parameters
        ----------
        definition
            The parsed option mapping of one ``>>options`` block.

        Returns
        -------
        dict
            The set options, without the parser's bookkeeping keys, the ``category`` tag, or the
            ``None`` defaults contributed by every module that registered on this shared keyword.
        """

        return {
            key: value
            for key, value in withoutParserBookkeepingKeys(definition).items()
            if value is not None and not strCaseCmp(key, "category")
        }
