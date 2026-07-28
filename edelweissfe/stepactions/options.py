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

# Register this step action for all available step types. This requires the step type
# modules to be imported before the step actions, as done in the input file parser.
modules = inputLanguage["step"].modules if "step" in inputLanguage else []

documentation = []

for module in modules:
    kw = module.addOptionalKeyword(
        "options",
        "This stepaction serves as a case insensitive container for storing step options for various modules.",
    )
    kw.addRequiredArg("category", "Option category.", str)

    documentation.append(kw)


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

    matches = [action for action in stepActions["options"].values() if strCaseCmp(action.options["category"], category)]

    if len(matches) > 1:
        raise ValueError(f"Multiple 'options' step action definitions for category {category}.")

    if not matches:
        return CaseInsensitiveDict()

    return CaseInsensitiveDict(
        {
            key: value
            for key, value in withoutParserBookkeepingKeys(matches[0].options).items()
            if value is not None and not strCaseCmp(key, "category")
        }
    )


class StepAction(StepActionBase):
    """A case insensitive container for storing step options of a category.

    Two mechanisms for telling the user's entries apart from the defaults contributed by foreign
    modules used to live side by side here: an ``explicitlySetOptions`` set, fed from the parser's
    ``explicitlySetArgs``, and :func:`getOptionsOfCategory`'s strip-``None``\\ s. They were developed
    independently on feat/amr-hanging-nodes and on the step management refactor. ``explicitlySetOptions``
    is gone: it was written on every update and read by nothing, in EdelweissFE or EdelweissMeshfree,
    so it protected no consumer. The single surviving mechanism relies on the invariant that every
    option on the shared keyword defaults to ``None``, which :func:`registerOptionsArg` now guarantees
    by construction rather than by convention.
    """

    def __init__(self, name, options, jobInfo, model, fieldOutputController, journal):
        self.name = name
        self.options = CaseInsensitiveDict(options)
        self.updateStepAction(options, jobInfo, model, fieldOutputController, journal)

    def updateStepAction(self, options, jobInfo, model, fieldOutputController, journal):
        self.options.update(options)

    def __contains__(self, *args):
        """wrapper method for CaseInsensitiveDict"""
        return self.options.__contains__(*args)

    def __getitem__(self, *args):
        """wrapper method for CaseInsensitiveDict"""
        return self.options.__getitem__(*args)

    def __setitem__(self, *args):
        """wrapper method for CaseInsensitiveDict"""
        self.options.__setitem__(*args)

    def get(self, *args):
        """wrapper method for CaseInsensitiveDict"""
        return self.options.get(*args)
