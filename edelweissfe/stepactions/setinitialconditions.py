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
# Created on Mo July 29 10:50:53 2019

# @author: Matthias Neuner
"""
Pass initial conditions to elements.
"""

import numpy as np

from edelweissfe.stepactions.base.stepactionbase import StepActionBase
from edelweissfe.utils.inputlanguage import InputLanguage

inputLanguage = InputLanguage()

# Register this step action for all available step types. This requires the step type
# modules to be imported before the step actions, as done in the input file parser.
modules = inputLanguage["step"].modules if "step" in inputLanguage else []

documentation = []

for module in modules:
    kw = module.addOptionalKeyword("setinitialconditions", "Pass initial conditions to elements.")
    # kw.addRequiredArg("name", "Name of the step action.", str)
    kw.addRequiredArg("property", "The name of the property to be initialized", str)
    kw.addRequiredArg("values", "Comma separated property values.", str)
    kw.addOptionalArg("elSet", "The element set for which the initaliziation is performed", str, "all")

    documentation.append(kw)


class StepAction(StepActionBase):
    """Set initial conditions to elements.

    The constructor is typed: it takes the element set itself, not its name, and the property
    values as a ``np.ndarray`` rather than a comma-separated string. ``propertyName`` stays a plain
    string here -- it is an element-facing identifier (the name ``el.setInitialCondition`` dispatches
    on), not a serialization of some richer object, so there is nothing for
    :meth:`fromStepActionDefinition` to translate it into. Nothing here parses an input file --
    resolving ``elSet=all`` against the model and splitting the comma-separated ``values`` string is
    the job of :meth:`fromStepActionDefinition` below, which is the only part of this module the
    ``.inp`` front-end needs.

    Parameters
    ----------
    name
        The name of this step action.
    elementSet
        The element set for which the initialization is performed.
    propertyName
        The name of the property to be initialized. Named to avoid shadowing the ``property``
        builtin, which this codebase uses as a decorator throughout.
    values
        The property values.
    """

    def __init__(self, name: str, elementSet, propertyName: str, values: np.ndarray):
        self.name = name

        self.theElements = elementSet
        self.theProperty = propertyName
        self.values = values
        self.active = True

    @classmethod
    def fromStepActionDefinition(cls, name, definition, jobInfo, model, fieldOutputController, journal):
        """Build this step action from a parsed ``>>setinitialconditions`` definition. See
        :class:`StepActionBase` for why this is separate from ``__init__``."""

        return cls(
            name,
            model.elementSets[definition["elSet"]],
            definition["property"],
            np.fromstring(definition["values"], dtype=float, sep=","),
        )

    def applyAtStepEnd(self, model, stepMagnitude=None):
        self.active = False

    def applyAtStepStart(self, model):
        if not self.active:
            return

        for el in self.theElements:
            el.setInitialCondition(self.theProperty, self.values)
