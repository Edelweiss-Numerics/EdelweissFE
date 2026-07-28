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
Created on Thu Apr 13 14:08:32 2017

@author: Matthias Neuner
"""


from dataclasses import dataclass

import numpy as np

from edelweissfe.journal.journal import Journal
from edelweissfe.models.femodel import FEModel
from edelweissfe.outputmanagers.base.outputmanagerbase import OutputManagerBase
from edelweissfe.utils.fieldoutput import FieldOutputController
from edelweissfe.utils.inputlanguage import InputLanguage, Module
from edelweissfe.utils.plotter import Plotter
from edelweissfe.utils.schema import schemaField

module = Module(
    "computetimemonitor", "A simple monitor to observe results (fieldOutputs) in the console during analysis."
)

inputLanguage = InputLanguage()

keyword = "output"
if keyword in inputLanguage:
    inputLanguage[keyword].addModule(module)

module.addRequiredArg("export", "Provide a filename to export the results.", str)

documentation = [module]

required = [kw.name for kw in module.requiredArgs]
required += [kw.name for kw in module.requiredKeywords]

optional = [kw.name for kw in module.optionalArgs]
optional += [kw.name for kw in module.optionalKeywords]


@dataclass(frozen=True)
class TimeMonitorSchema:
    """L2: the options this output manager accepts, owned by this module and never mutated from
    outside it.

    Mirrors the ``module.addRequiredArg(...)`` declaration above one-for-one. The option itself has
    no default in the ``Module`` declaration (it is required there), but a default of ``None`` is
    given here -- with ``required`` forced to ``True`` -- so that ``TimeMonitorSchema()`` remains
    constructible for the L1 constructor's default argument; the L4 adapter
    (``buildSchemaFromOptions``) still enforces that an ``.inp`` file supplies ``export``, exactly
    as ``caseInsensitiveKwargsChecker`` did against the old ``required`` list.
    """

    export: str | None = schemaField(
        description="Provide a filename to export the results.", dtype=str, default=None, required=True
    )


class OutputManager(OutputManagerBase):
    identification = "TimeMonitor"

    #: L2 schema declared for the L3 registry, per OptionSchemaProvider.
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
        """L1: constructible standalone, with no ``InputLanguage``/``Module``/parser involvement and
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
        self.monitorJobs = []
        self.model = model

        # old factory read the "export" option into a local variable named "filename"; kept as
        # self.exportFile for behavioral parity.
        self.exportFile = configuration.export
        self.timeVals = []

    def initializeJob(self):
        pass

    def initializeStep(self, step):
        pass

    def finalizeIncrement(self, U, P, **kwargs):
        self.timeVals.append(self.model.time)

    def finalizeFailedIncrement(self, **kwargs):
        pass

    def finalizeStep(self, U, P):
        pass

    def finalizeJob(
        self,
        U,
        P,
    ):
        np.savetxt("{:}.csv".format(self.exportFile), np.asarray(self.timeVals).T)
