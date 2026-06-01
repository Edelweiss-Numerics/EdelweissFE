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


from edelweissfe.outputmanagers.base.outputmanagerbase import OutputManagerBase
from edelweissfe.utils.caseinsensitivedict import CaseInsensitiveDict
from edelweissfe.utils.inputlanguage import InputLanguage, Module
from edelweissfe.utils.misc import (
    caseInsensitiveKwargsChecker,
    castKwargsValuesAndAddDefaults,
)
from edelweissfe.utils.performancetiming import extractIncrementTimes

"""
Prints the compute times per increment to the screen and writes them into a file (optional).

.. code-block:: console
    :caption: Example:

    *output, type=computetimemonitor, name=mycomputetimes
        export=myComputeTimes
"""

module = Module(
    "computetimemonitor", "A simple monitor to observe results (fieldOutputs) in the console during analysis."
)

inputLanguage = InputLanguage()

keyword = "output"
if keyword in inputLanguage:
    inputLanguage[keyword].addModule(module)

module.addOptionalArg("export", "Provide a filename to export the results.", str, None)

documentation = [module]

required = [kw.name for kw in module.requiredArgs]
required += [kw.name for kw in module.requiredKeywords]

optional = [kw.name for kw in module.optionalArgs]
optional += [kw.name for kw in module.optionalKeywords]


@caseInsensitiveKwargsChecker(required, optional)
@castKwargsValuesAndAddDefaults(module)
def outputManagerFactory(name, FEModel, fieldOutputController, moduleOptions, journal, plotter, **kwargs):
    kwargs = CaseInsensitiveDict(kwargs)

    filename = kwargs["export"]

    return OutputManager(name, FEModel, fieldOutputController, journal, plotter, filename)


class OutputManager(OutputManagerBase):
    identification = "ComputeTimeMonitor"

    def __init__(self, name, model, fieldOutputController, journal, plotter, filename):
        self.journal = journal
        self.stepcounter = 0

        self.exportFile = filename

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
