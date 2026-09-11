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
from edelweissfe.utils.fieldoutput import FieldOutputController
from edelweissfe.utils.math import createMathExpression
from edelweissfe.utils.plotter import Plotter
from edelweissfe.utils.schema import schemaField

"""
A simple monitor to observe results (fieldOutputs) in the console during analysis.

.. code-block:: edelweiss
    :caption: Example:

    *output, type=monitor, jobName=cpe4job, name=omegaMon
        fieldOutput=omega, f(x)='max(x)'
"""


@dataclass(frozen=True)
class MonitorSchema:
    """The options this output manager accepts, owned by this module and never mutated from
    outside it.

    ``fieldOutput`` is declared ``required=True`` explicitly, but is still given a
    ``default=None`` so that ``MonitorSchema()`` remains constructible without arguments;
    ``buildSchemaFromOptions`` still enforces that an ``.inp`` file supplies it.
    """

    fieldOutput: str | None = schemaField(
        description="Name of the field output to monitor.", dtype=str, default=None, required=True
    )
    label: str = schemaField(description="Name of the output manager.", dtype=str, default="Monitor")
    #: Spelled ``f(x)`` in the input file, which is not a valid Python identifier -- hence the
    #: ``optionName`` indirection, see ``edelweissfe.utils.schema.schemaField``.
    f_x: str | None = schemaField(
        description="Apply a model accessible function on the result.",
        dtype=str,
        default=None,
        optionName="f(x)",
    )


class OutputManager(OutputManagerBase):
    """Simple monitor for nodes, nodeSets, elements and elementSets"""

    identification = "Monitor"
    printTemplate = "{:} ({:}): {:}"

    #: Option schema for this output manager, per OptionSchemaProvider.
    schema = MonitorSchema

    def __init__(
        self,
        name: str,
        model: FEModel,
        fieldOutputController: FieldOutputController,
        journal: Journal,
        plotter: Plotter,
        *,
        configuration: MonitorSchema = MonitorSchema(),
    ):
        """Constructible standalone, with no parser involvement. Options arrive as an
        already-validated, already-typed schema instance.

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
            The options this output manager accepts; defaults to all-defaults.
        """
        # `configuration.label` always overrides the `name` argument, since `label` defaults to
        # "Monitor" and is thus never missing. A per-manager `label` option overriding the name
        # entirely (rather than merely customizing it) looks like a latent bug, but the behavior
        # is preserved here.
        self.name = configuration.label

        self.journal = journal
        self.monitorJobs = []
        self.fieldOutputController = fieldOutputController

        fx = configuration.f_x
        # A *falsy* value -- None (option absent) as well as an explicitly empty string --
        # falls back to the identity expression "x". A schema default of "x" alone would not
        # reproduce this, since an explicitly-empty option would then stay empty instead of
        # falling back.
        if not fx:
            fx = "x"

        entry = dict()
        entry["fieldOutput"] = fieldOutputController.fieldOutputs[configuration.fieldOutput]
        entry["f(x)"] = createMathExpression(fx)

        self.monitorJobs.append(entry)

    def initializeJob(self):
        pass

    def initializeStep(self, step):
        pass

    def finalizeIncrement(self, **kwargs):
        for nJob in self.monitorJobs:
            result = nJob["f(x)"](nJob["fieldOutput"].getLastResult())
            self.journal.message(
                self.printTemplate.format(self.name, nJob["fieldOutput"].name, result),
                self.identification,
            )

    def finalizeFailedIncrement(self, **kwargs):
        pass

    def finalizeStep(
        self,
    ):
        pass

    def finalizeJob(
        self,
    ):
        pass
