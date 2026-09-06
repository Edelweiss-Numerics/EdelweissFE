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
# Created on Sat Jul 22 21:26:01 2017

# @author: Matthias Neuner

from dataclasses import dataclass

from edelweissfe.journal.journal import Journal
from edelweissfe.models.femodel import FEModel
from edelweissfe.outputmanagers.base.outputmanagerbase import OutputManagerBase
from edelweissfe.utils.exceptions import ConditionalStop
from edelweissfe.utils.fieldoutput import FieldOutputController
from edelweissfe.utils.math import createModelAccessibleFunction
from edelweissfe.utils.plotter import Plotter
from edelweissfe.utils.schema import schemaField

"""
A conditional stop conditions wenn an expression becomes true.
Useful, e.g., for indirect displacement control.

.. code-block:: edelweiss
    :caption: Example:

    *output, type=conditionalstop, jobName=myJob, name=condStop
        stop='fieldOutputs["damage"]  >= .99'
        stop='fieldOutputs["displacement"]  < -5'
"""


@dataclass(frozen=True)
class ConditionalStopSchema:
    """The options this output manager accepts, owned by this module and never mutated from
    outside it.

    ``stop`` is declared ``required=True`` explicitly, but is still given ``default=None`` so
    that ``ConditionalStopSchema()`` is constructible without an argument at import time. A
    caller going through ``buildSchemaFromOptions`` still must supply ``stop`` (a missing
    required field is rejected regardless of this default); a caller constructing this schema
    directly with no arguments gets a manager that never stops the analysis -- see
    :class:`OutputManager`.
    """

    stop: str | None = schemaField(
        description="Model accessible function describing the stop condition.",
        dtype=str,
        default=None,
        required=True,
    )


class OutputManager(OutputManagerBase):
    identification = "ConditionalStop"
    printTemplate = "{:}, {:}: {:}"

    #: Option schema for this output manager, per OptionSchemaProvider.
    schema = ConditionalStopSchema

    def __init__(
        self,
        name: str,
        model: FEModel,
        fieldOutputController: FieldOutputController,
        journal: Journal,
        plotter: Plotter,
        *,
        configuration: ConditionalStopSchema = ConditionalStopSchema(),
    ):
        """Constructible standalone, with no parser involvement. Options arrive as an
        already-validated, already-typed schema instance.

        Building the stop-condition callable from ``configuration.stop`` turns a user-written
        expression string into a callable that closes over ``model`` and
        ``fieldOutputController.fieldOutputs``, so it happens here where those collaborators are
        available.

        Parameters
        ----------
        name
            The name of this output manager.
        model
            The model tree.
        fieldOutputController
            The field output controller instance.
        journal
            The journal instance for logging.
        plotter
            The plotter instance.
        configuration
            The options this output manager accepts; ``stop`` is required by the input language,
            but defaults to ``None`` here so that no stop condition is ever triggered.
        """
        self.name = name
        self.model = model
        self.journal = journal
        self.monitorJobs = []
        self.fieldOutputController = fieldOutputController

        if configuration.stop is not None:
            stopFunction = createModelAccessibleFunction(
                configuration.stop, model, fieldOutputs=fieldOutputController.fieldOutputs
            )
            self.monitorJobs.append(stopFunction)

    def initializeJob(self):
        pass

    def initializeStep(self, step):
        pass

    def finalizeIncrement(self, **kwargs):
        for nJob in self.monitorJobs:
            if nJob():
                raise ConditionalStop()

    def finalizeFailedIncrement(self, **kwargs):
        pass

    def finalizeStep(self):
        pass

    def finalizeJob(self):
        pass
