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
from edelweissfe.utils.misc import strCaseCmp

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


def registerOptionsArg(name: str, description: str, dataType: type):
    """Register an available option on the ``options`` keyword of all step types.

    The default value is always None, which marks the option as not specified by
    the user; :func:`getOptionsOfCategory` strips unspecified options, such that
    consumers receive only the options which were actually defined in the input file.

    Parameters
    ----------
    name
        The name of the option.
    description
        The description of the option.
    dataType
        The data type of the option.
    """

    if "step" not in inputLanguage:
        return

    for stepModule in inputLanguage["step"].modules:
        stepModule.getKeyword("options").addOptionalArg(name, description, dataType, None)


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
            for key, value in matches[0].options.items()
            if value is not None and not strCaseCmp(key, "category")
        }
    )


class StepAction(StepActionBase):
    """A case insensitive container for storing step options of a category."""

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
