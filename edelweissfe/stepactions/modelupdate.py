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
# Created on Mon Jan 23 13:03:09 2017

# @author: Matthias Neuner

from dataclasses import dataclass

from edelweissfe.stepactions.base.stepactionbase import StepActionBase
from edelweissfe.utils.caseinsensitivedict import CaseInsensitiveDict
from edelweissfe.utils.inputlanguage import InputLanguage
from edelweissfe.utils.math import execModelAccessibleExpression
from edelweissfe.utils.misc import withoutParserBookkeepingKeys
from edelweissfe.utils.schema import buildSchemaFromOptions, schemaField

"""This step action may be used for updating something in the model at the beginning
of a step.
"""


inputLanguage = InputLanguage()

# Register this step action for all available step types. This requires the step type
# modules to be imported before the step actions, as done in the input file parser.
modules = inputLanguage["step"].modules if "step" in inputLanguage else []

documentation = []

for module in modules:
    kw = module.addOptionalKeyword(
        "modelupdate", "This step action may be used for updating the model at the beginning of a step."
    )
    # kw.addRequiredArg("name", "Name of the step action.", str)
    kw.addRequiredArg("update", "Model accessible, executable expression", str)

    documentation.append(kw)


@dataclass(frozen=True)
class ModelUpdateSchema:
    """L2: the scalar options of the ``modelupdate`` keyword, owned by this module and never
    mutated from outside it."""

    update: str | None = schemaField(
        description="Model accessible, executable expression", dtype=str, default=None, required=True
    )


class StepAction(StepActionBase):
    """Execute a Python expression against the live model at the beginning of a step.

    The constructor is typed, but its ``updateExpression`` deliberately stays a **string**: the
    expression is executed against a live model by
    :func:`~edelweissfe.utils.math.execModelAccessibleExpression`, so here the string *is* the value
    rather than a serialization of one -- unlike, say, a ``f(t)`` amplitude, which serializes a
    callable. There is consequently nothing for :meth:`fromStepActionDefinition` to translate beyond
    reading the option out of the definition.

    Parameters
    ----------
    name
        The name of this step action.
    updateExpression
        The model accessible, executable Python expression, e.g.
        ``'model.constraints["pc1"].active=False'``.
    model
        The model tree. Accepted for uniformity with the other step actions; the expression is
        evaluated against the model handed to :meth:`updateModel` at execution time, not against
        this one.
    journal
        The journal object for logging. Accepted for uniformity as well; this action logs with the
        journal handed to :meth:`updateModel`.
    """

    #: L2 schema declared for the L3 registry, per OptionSchemaProvider.
    schema = ModelUpdateSchema

    def __init__(self, name, updateExpression: str, model, journal):
        self.name = name

        self.updateStepAction(updateExpression)

    def applyAtStepEnd(self, model):
        """By default, this action is only executed once.

        Parameters
        ----------
        model
            The current state of the model.
        """

        self.active = False

    @classmethod
    def fromStepActionDefinition(cls, name, definition, jobInfo, model, fieldOutputController, journal):
        """Build this step action from a parsed ``>>modelupdate`` definition. See
        :class:`~edelweissfe.stepactions.base.stepactionbase.StepActionBase` for why this is separate
        from ``__init__``.

        Parameters
        ----------
        name
            The name of the step action.
        definition
            The parsed option mapping for this step action.
        jobInfo
            A dictionary containing the information about the job.
        model
            The model tree.
        fieldOutputController
            The field output controlling object.
        journal
            The journal object for logging.

        Returns
        -------
        StepAction
            The constructed step action.
        """

        definition = CaseInsensitiveDict(withoutParserBookkeepingKeys(definition))
        definition.pop("name", None)
        configuration = buildSchemaFromOptions(cls.schema, definition)

        return cls(name, configuration.update, model, journal)

    def updateStepActionFromDefinition(self, definition, jobInfo, model, fieldOutputController, journal):
        """Update from a parsed ``>>modelupdate`` definition re-declared in a later step.

        Parameters
        ----------
        definition
            The parsed option mapping for this step action.
        jobInfo
            A dictionary containing the information about the job.
        model
            The model tree.
        fieldOutputController
            The field output controlling object.
        journal
            The journal object for logging.
        """

        definition = CaseInsensitiveDict(withoutParserBookkeepingKeys(definition))
        definition.pop("name", None)
        configuration = buildSchemaFromOptions(self.schema, definition)

        self.updateStepAction(configuration.update)

    def updateStepAction(self, updateExpression: str):
        """Prescribe a new expression, and set the action active again.

        Parameters
        ----------
        updateExpression
            The model accessible, executable Python expression.
        """

        self.updateExpression = updateExpression
        self.active = True

    def updateModel(self, model, fieldOutputController, journal):
        """Update the model based on an executable provided Python expression.

        Parameters
        ----------
        model
            The current state of the model.
        fieldOutputController
            The field output controlling object, whose field outputs the expression may access.
        journal
            The journal object for logging.

        Returns
        -------
        FEModel
            The updated model.
        """

        if not self.active:
            return model

        journal.message("Updating model: {:}".format(self.updateExpression), self.name)
        execModelAccessibleExpression(
            self.updateExpression,
            model,
            fieldOutputs=fieldOutputController.fieldOutputs,
        )
        return model
