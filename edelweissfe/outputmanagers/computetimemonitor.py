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
#  Alexander Dummer alexander.dummer@uibk.ac.at
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


from dataclasses import dataclass

from edelweissfe.journal.journal import Journal
from edelweissfe.models.femodel import FEModel
from edelweissfe.outputmanagers.base.outputmanagerbase import OutputManagerBase
from edelweissfe.utils.fieldoutput import FieldOutputController
from edelweissfe.utils.performancetiming import extractIncrementTimes
from edelweissfe.utils.plotter import Plotter
from edelweissfe.utils.schema import schemaField

"""
Prints the compute times per increment to the screen and writes them into a file (optional).

.. code-block:: console
    :caption: Example:

    *output, type=computetimemonitor, name=mycomputetimes
        export=myComputeTimes
"""


@dataclass(frozen=True)
class ComputeTimeMonitorSchema:
    """L2: the options this output manager accepts, owned by this module and never mutated from
    outside it.
    """

    export: str | None = schemaField(description="Provide a filename to export the results.", dtype=str, default=None)


class OutputManager(OutputManagerBase):
    identification = "ComputeTimeMonitor"

    #: L2 schema declared for the L3 registry, per OptionSchemaProvider.
    schema = ComputeTimeMonitorSchema

    def __init__(
        self,
        name: str,
        model: FEModel,
        fieldOutputController: FieldOutputController,
        journal: Journal,
        plotter: Plotter,
        *,
        configuration: ComputeTimeMonitorSchema = ComputeTimeMonitorSchema(),
    ):
        """L1: constructible standalone, with no parser involvement and
        no ``moduleOptions``. Options arrive as an already-validated, already-typed schema instance,
        so nothing here coerces strings or inspects dictionaries.

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
        self.name = name
        self.journal = journal
        self.stepcounter = 0

        self.exportFile = configuration.export

        if self.exportFile:
            with open(self.exportFile, "w+") as f:
                f.write("# \n# EdelweissFE: computing times per increment\n#\n")

    def updateDefinition(self, **kwargs: dict):
        pass

    def initializeJob(self):
        pass

    def initializeStep(self, step):
        self.stepcounter += 1

    def finalizeIncrement(self, **kwargs):
        self.journal.printPrettyTable(extractIncrementTimes(), self.identification)

    def finalizeFailedIncrement(self, **kwargs):
        self.journal.printPrettyTable(extractIncrementTimes(), self.identification)

    def finalizeStep(
        self,
    ):
        pass

    def finalizeJob(
        self,
    ):
        pass
