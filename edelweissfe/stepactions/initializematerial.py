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
Let materials initialize themselves (e.g., state vars depending on material parameters...) !
"""

from dataclasses import dataclass

import numpy as np

from edelweissfe.stepactions.base.stepactionbase import StepActionBase
from edelweissfe.utils.caseinsensitivedict import CaseInsensitiveDict
from edelweissfe.utils.inputlanguage import InputLanguage
from edelweissfe.utils.misc import withoutParserBookkeepingKeys
from edelweissfe.utils.schema import buildSchemaFromOptions

inputLanguage = InputLanguage()

# Register this step action for all available step types. This requires the step type
# modules to be imported before the step actions, as done in the input file parser.
modules = inputLanguage["step"].modules if "step" in inputLanguage else []

documentation = []

for module in modules:
    kw = module.addOptionalKeyword("initializematerial", "Standard distributed load, applied on a surface set.")
    kw.addRequiredArg("name", "Name of the step action.", str)
    kw.addOptionalArg("elSet", "The element set for application of the boundary condition.", str, "all")

    documentation.append(kw)


@dataclass(frozen=True)
class InitializeMaterialSchema:
    """L2: the scalar options of the ``initializematerial`` keyword, owned by this module and never
    mutated from outside it.

    Declares no fields at all: ``elSet`` is the keyword's only option, and it is *not* a schema
    field -- it names an existing model object, resolved by :meth:`fromStepActionDefinition` before
    the schema is even built, exactly like every other category's structural names. The empty
    schema is still built (rather than skipped) so a misspelled option is rejected the same way as
    for every other module.
    """


class StepAction(StepActionBase):
    """Initializes materials.

    The constructor is typed: it takes the element set itself, not its name. Nothing here parses
    an input file -- resolving ``elSet=all`` against the model is the job of
    :meth:`fromStepActionDefinition` below, which is the only part of this module the ``.inp``
    front-end needs.

    Parameters
    ----------
    name
        The name of this step action.
    elementSet
        The element set whose materials are initialized.
    """

    #: L2 schema declared for the L3 registry, per OptionSchemaProvider.
    schema = InitializeMaterialSchema

    def __init__(self, name: str, elementSet):
        self.name = name

        self.theElements = elementSet
        self.active = True
        self.emptyDef = np.array([0.0])

    @classmethod
    def fromStepActionDefinition(cls, name, definition, jobInfo, model, fieldOutputController, journal):
        """Build this step action from a parsed ``>>initializematerial`` definition. See
        :class:`StepActionBase` for why this is separate from ``__init__``.

        ``name`` and the parser's bookkeeping keys are stripped, and ``elSet`` is structural (it
        names a model object), so both are popped before the (empty) remainder is validated
        against :class:`InitializeMaterialSchema`."""

        definition = CaseInsensitiveDict(withoutParserBookkeepingKeys(definition))
        definition.pop("name", None)
        elSetName = definition.pop("elSet")
        buildSchemaFromOptions(cls.schema, definition)

        return cls(name, model.elementSets[elSetName])

    def applyAtStepEnd(self, model, stepMagnitude=None):
        self.active = False

    def applyAtStepStart(self, model):
        if not self.active:
            return

        for el in self.theElements:
            el.setInitialCondition("initialize material", self.emptyDef)
