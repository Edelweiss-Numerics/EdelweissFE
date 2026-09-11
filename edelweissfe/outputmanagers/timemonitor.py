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
"""
Records the model time at the end of every increment and writes the collected times to a
``.csv`` file when the job finishes.

.. code-block:: console
    :caption: Example:

    *output, type=timemonitor, name=mytimes
        export=myTimes

@author: Matthias Neuner
"""

from dataclasses import dataclass

import numpy as np

from edelweissfe.journal.journal import Journal
from edelweissfe.models.femodel import FEModel
from edelweissfe.outputmanagers.base.outputmanagerbase import OutputManagerBase
from edelweissfe.utils.fieldoutput import FieldOutputController
from edelweissfe.utils.plotter import Plotter
from edelweissfe.utils.schema import schemaField


@dataclass(frozen=True)
class TimeMonitorSchema:
    """The options this output manager accepts, owned by this module and never mutated from
    outside it.

    ``export`` is required, but is still given a default of ``None`` -- with ``required`` forced
    to ``True`` -- so that ``TimeMonitorSchema()`` remains constructible without arguments;
    ``buildSchemaFromOptions`` still enforces that an ``.inp`` file supplies it.
    """

    export: str | None = schemaField(
        description="Provide a filename to export the results.", dtype=str, default=None, required=True
    )


class OutputManager(OutputManagerBase):
    identification = "TimeMonitor"

    #: Option schema for this output manager, per OptionSchemaProvider.
    schema = TimeMonitorSchema

    def __init__(
        self,
        name: str,
        model: FEModel,
        fieldOutputController: FieldOutputController,
        journal: Journal,
        plotter: Plotter,
        *,
        configuration: TimeMonitorSchema = TimeMonitorSchema(),
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
        self.monitorJobs = []
        self.model = model

        self.exportFile = configuration.export
        self.timeVals = []

    def initializeJob(self):
        pass

    def initializeStep(self, step):
        pass

    def finalizeIncrement(self, **kwargs):
        self.timeVals.append(self.model.time)

    def finalizeFailedIncrement(self, **kwargs):
        pass

    def finalizeStep(self):
        pass

    def finalizeJob(self):
        np.savetxt("{:}.csv".format(self.exportFile), np.asarray(self.timeVals).T)
