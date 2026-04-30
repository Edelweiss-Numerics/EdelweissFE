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

from edelweissfe.outputmanagers.base.outputmanagerbase import OutputManagerBase
from edelweissfe.utils.caseinsensitivedict import CaseInsensitiveDict
from edelweissfe.utils.inputlanguage import InputLanguage
from edelweissfe.utils.math import createMathExpression
from edelweissfe.utils.misc import (
    caseInsensitiveKwargsChecker,
    castKwargsValuesAndAddDefaults,
)

"""
A simple monitor to observe results (fieldOutputs) in the console during analysis.

.. code-block:: edelweiss
    :caption: Example:

    *output, type=monitor, jobName=cpe4job, name=omegaMon
        fieldOutput=omega, f(x)='max(x)'
"""

inputLanguage = InputLanguage()
module = inputLanguage["output"].addModule(
    "monitor",
    "A simple monitor to observe results (fieldOutputs) in the console during analysis.",
)

module.addRequiredArg("fieldOutput", "Name of the field output to monitor.", str)
module.addOptionalArg("label", "Name of the output manager.", str, "Monitor")
module.addOptionalArg("f(x)", "Apply a model accessible function on the result.", str, None)

documentation = [module]

required = [kw.name for kw in module.requiredArgs]
required += [kw.name for kw in module.requiredKeywords]

optional = [kw.name for kw in module.optionalArgs]
optional += [kw.name for kw in module.optionalKeywords]


@caseInsensitiveKwargsChecker(required, optional)
@castKwargsValuesAndAddDefaults(module)
def outputManagerFactory(name, FEModel, fieldOutputController, moduleOptions, journal, plotter, **kwargs):
    kwargs = CaseInsensitiveDict(kwargs)

    fieldOutputName = kwargs["fieldOutput"]
    fx = kwargs["f(x)"]
    if not fx:
        fx = "x"

    name = kwargs["label"]

    return OutputManager(name, FEModel, fieldOutputController, journal, plotter, fieldOutputName, fx)


class OutputManager(OutputManagerBase):
    """Simple monitor for nodes, nodeSets, elements and elementSets"""

    identification = "Monitor"
    printTemplate = "{:} ({:}): {:}"

    def __init__(self, name, model, fieldOutputController, journal, plotter, fieldOutputName, fx):
        self.name = name

        self.journal = journal
        self.monitorJobs = []
        self.fieldOutputController = fieldOutputController

        entry = dict()
        entry["fieldOutput"] = fieldOutputController.fieldOutputs[fieldOutputName]
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
