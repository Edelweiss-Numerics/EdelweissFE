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
from edelweissfe.utils.plotter import Plotter
from edelweissfe.utils.schema import schemaField

"""
Writes a status file during the analysis.

.. code-block:: console
    :caption: Example:

    *output, type=statusfile, name=status
        filename=myStatus.sta
"""


@dataclass(frozen=True)
class StatusFileSchema:
    """The options this output manager accepts, owned by this module and never mutated from
    outside it.
    """

    filename: str = schemaField(description="Name of the output manager.", dtype=str, default="job.sta")


class OutputManager(OutputManagerBase):
    """Simple status file writer for step, incrementation and iterations"""

    identification = "Statusfile"
    printTemplate = "{:}, {:}: {:}"

    #: Option schema for this output manager, per OptionSchemaProvider.
    schema = StatusFileSchema

    def __init__(
        self,
        name: str,
        model: FEModel,
        fieldOutputController: FieldOutputController,
        journal: Journal,
        plotter: Plotter,
        *,
        configuration: StatusFileSchema = StatusFileSchema(),
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
        self.name = name
        self.journal = journal
        self.filename = configuration.filename
        self.statusFileExists = False

    # def initializeSimulation(self, model):
    #     pass

    def initializeJob(self):
        pass

    def initializeStep(self, step):
        pass

    def finalizeIncrement(self, statusInfoDict: dict = {}, **kwargs):
        if statusInfoDict is None:
            raise ValueError("Status info dictionary is None. Statusfile cannot be used with this solver.")
        self.writeStatusFile(statusInfoDict)

    def finalizeFailedIncrement(self, statusInfoDict: dict = {}, **kwargs):
        if statusInfoDict is None:
            raise ValueError("Status info dictionary is None. Statusfile cannot be used with this solver.")
        self.writeStatusFile(statusInfoDict)

    def finalizeStep(self):
        pass

    def finalizeJob(self):
        pass

    def writeStatusFile(self, statusInfoDict):
        """Write the status to a file.

        Parameters
        ----------
        statusInfoDict
            A dictionary containing information about the simulation status.
        """
        d = statusInfoDict

        if not self.statusFileExists:
            with open(self.filename, "w+") as f:
                f.write("#\n")
                f.write("# This is a status file of EdelweissFE.\n")
                f.write("#\n")
                f.write(
                    "#{: >5}{: >6}{: >6}{: >10}{: >12}{: >12}    {:<}\n".format(
                        "step",
                        "inc",
                        "iters",
                        "converged",
                        "time inc",
                        "time end",
                        "notes",
                    )
                )
                f.write("#\n")
            self.statusFileExists = True

        with open(self.filename, "a") as f:
            f.write(
                "{: >6}{: >6}{: >6}{: >10}{: >12.3e}{: >12.3e}    # {:<s}\n".format(
                    d["step"],
                    d["inc"],
                    d["iters"],
                    d["converged"],
                    d["time inc"],
                    d["time end"],
                    d["notes"],
                )
            )
