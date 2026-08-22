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
# Created on Tue Feb 9 10:05:41 2021

# @author: Matthias Neuner

from dataclasses import dataclass

import numpy as np

from edelweissfe.analyticalfields.base.analyticalfieldbase import (
    AnalyticalField as AnalyticalFieldBase,
)
from edelweissfe.stepactions.base.stepactionbase import StepActionBase
from edelweissfe.utils.caseinsensitivedict import CaseInsensitiveDict
from edelweissfe.utils.misc import withoutParserBookkeepingKeys
from edelweissfe.utils.schema import buildSchemaFromOptions, schemaField

"""
Set a field (via fieldOutput) to a predefined value.
"""


@dataclass(frozen=True)
class SetFieldSchema:
    """The scalar options of the ``setfield`` keyword, owned by this module and never mutated from
    outside it.

    ``name`` and ``fieldOutput`` are ``structuralOnly`` fields: ``fieldOutput`` names an existing
    model object, resolved by :meth:`fromStepActionDefinition` before the schema is even built,
    exactly like every other category's structural names, and ``name`` is popped even earlier, by
    ``helpers/inputfilehelpers.py``. Both are declared here purely so the rendered grammar surface
    documents them -- :func:`~edelweissfe.utils.schema.buildSchemaFromOptions` never actually sees
    either key. Unlike every other step action, ``name`` is *optional* here (it defaults to
    ``"setfield"``). ``valueType`` is named to avoid shadowing the ``type`` builtin, hence the
    ``optionName`` indirection; ``value`` stays a plain string regardless of what ``valueType``
    says it means -- interpreting it (a numeric vector, or an analytical field name to resolve
    against the model) is :meth:`_valueFromDefinition`'s job, not the schema's.
    """

    name: str | None = schemaField(
        description="Name of the step action.", dtype=str, default="setfield", structuralOnly=True
    )
    fieldOutput: str | None = schemaField(
        description="Field output to be set.", dtype=str, default=None, required=True, structuralOnly=True
    )
    valueType: str | None = schemaField(
        description="Either 'uniform' or 'analyticalField'.", dtype=str, default=None, required=True, optionName="type"
    )
    value: str | None = schemaField(
        description="Scalar value if type 'const'; name of analyticalField if type 'analyticalField'",
        dtype=str,
        default=None,
        required=True,
    )


class StepAction(StepActionBase):
    """Set a field, addressed through a field output, to a prescribed value at the start of a step.

    The declared input file grammar is a tagged union: ``type=uniform`` makes ``value`` a
    comma-separated numeric vector, ``type=analyticalField`` makes it the *name* of an analytical
    field. The tag is collapsed here: the constructor takes the value itself, either as an
    ``np.ndarray`` or as an analytical field object, and :meth:`applyAtStepStart` dispatches on its
    type. That is what turning ``type``/``value`` into typed arguments amounts to -- the tag only
    exists because an ``.inp`` file transports both arms as text, so both it and the name resolution
    belong on :meth:`fromStepActionDefinition`.

    Parameters
    ----------
    name
        The name of this step action.
    fieldOutput
        The field output whose results are overwritten. The resolved object, not its name -- this
        action genuinely needs the field output, not the controller it is registered with.
    value
        Either the value prescribed uniformly for every entry of the field output, as an
        ``np.ndarray``, or an analytical field, which is evaluated at the quadrature points of the
        elements the field output is associated with.
    model
        The model tree.
    journal
        The journal object for logging.
    """

    #: Option schema for this step action, consumed by OptionSchemaProvider's registry.
    schema = SetFieldSchema

    def __init__(self, name, fieldOutput, value, model, journal):
        self.name = name
        self.fieldOutput = fieldOutput
        self.value = value

        self._journal = journal

        self.updateStepAction()

    @classmethod
    def fromStepActionDefinition(cls, name, definition, jobInfo, model, fieldOutputController, journal):
        """Build this step action from a parsed ``>>setfield`` definition. See
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
        fieldOutputName = definition.pop("fieldOutput")
        configuration = buildSchemaFromOptions(cls.schema, definition)

        return cls(
            name,
            fieldOutputController.fieldOutputs[fieldOutputName],
            cls._valueFromDefinition(configuration, model),
            model,
            journal,
        )

    def updateStepActionFromDefinition(self, definition, jobInfo, model, fieldOutputController, journal):
        """Update from a parsed ``>>setfield`` definition re-declared in a later step.

        The re-declared field output and value are deliberately ignored, as they always have been:
        re-declaring this action only arms it again, so that the *same* value is set once more at the
        start of the new step.

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

        self.updateStepAction()

    def applyAtStepEnd(self, model, stepMagnitude=None):
        """Deactivate this step action; the value is set once per step it is declared in.

        Parameters
        ----------
        model
            The current state of the model.
        stepMagnitude
            Unused; present for signature compatibility with the other step actions.
        """

        self.active = False

    def updateStepAction(self):
        """Arm this step action again, so that its value is set at the start of the next step."""

        self.active = True

    def applyAtStepStart(
        self,
        model,
    ):
        """Set the field output's results to the prescribed value, dispatching on the value's type.

        Parameters
        ----------
        model
            The current state of the model.
        """

        if not self.active:
            return

        if isinstance(self.value, AnalyticalFieldBase):
            self._setFromAnalyticalField()
        else:
            self._setUniformly()

    def _setUniformly(self):
        """Set every entry of the field output's results to the prescribed value vector."""

        currentResults = np.zeros_like(self.fieldOutput.getLastResult())
        newResult = self.value

        if currentResults.ndim == 2:
            currentResults = np.expand_dims(currentResults, 0)

        if not currentResults.shape[-1] == newResult.shape[-1]:
            self._journal.errorMessage(
                f"Dimension mismatch. Result '{self.fieldOutput.name}' has length {currentResults.shape[-1]} but value has length {newResult.shape[-1]}",  # noqa: E501
                self.name,
            )
            raise Exception

        currentResults[:] = newResult
        self._journal.message(
            "Setting field {:} to uniform value {:}".format(self.fieldOutput.name, self.value),
            self.name,
        )
        self.fieldOutput.setResults(currentResults)

    def _setFromAnalyticalField(self):
        """Set the field output's results by evaluating the prescribed analytical field at the
        quadrature points of the elements the field output is associated with."""

        currentResults = np.zeros_like(self.fieldOutput.getLastResult())

        if self.value.type == "scalarExpression" and not currentResults.shape[2] == 1:
            self._journal.errorMessage(f"Cannot map scalar value to {currentResults.shape[2]}-dimensional result.")
            raise Exception

        elementList = self.fieldOutput.associatedSet

        for i1, element in enumerate(elementList):
            coordinatesAtQuadraturePoints = element.getCoordinatesAtQuadraturePoints()

            currentResults[i1] = self.value.evaluateAtCoordinates(coordinatesAtQuadraturePoints)

        self.fieldOutput.setResults(currentResults)

    @staticmethod
    def _valueFromDefinition(configuration: "SetFieldSchema", model):
        """Turn a validated ``SetFieldSchema``'s ``valueType``/``value`` pair into the prescribed
        value itself.

        Parameters
        ----------
        configuration
            The validated options of this step action.
        model
            The model tree, against which an analytical field's name is resolved.

        Returns
        -------
        np.ndarray | AnalyticalField
            The uniform value vector for ``type=uniform``, or the analytical field for
            ``type=analyticalField``.
        """

        if configuration.valueType == "uniform":
            return np.fromstring(configuration.value, float, sep=",")

        if configuration.valueType == "analyticalField":
            return model.analyticalFields[configuration.value]

        raise Exception("Invalid type: {}".format(configuration.valueType))
