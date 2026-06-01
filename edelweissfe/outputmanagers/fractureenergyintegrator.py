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
# Created on Tue Dec 17 08:26:01 2019

# @author: Matthias Neuner

import numpy as np

from edelweissfe.outputmanagers.base.outputmanagerbase import OutputManagerBase
from edelweissfe.utils.caseinsensitivedict import CaseInsensitiveDict
from edelweissfe.utils.inputlanguage import InputLanguage, Module
from edelweissfe.utils.math import createMathExpression
from edelweissfe.utils.misc import (
    caseInsensitiveKwargsChecker,
    castKwargsValuesAndAddDefaults,
)

"""
A simple integrator to compute the fracture energy by integrating a load-displacement curve.

.. code-block:: edelweiss
    :caption: Example:

    *output, type=fractureenergyintegrator, jobName=myJob, name=gfi
        forceFieldOutput=RF, displacementFieldOutput=U, fractureArea='100.0*1.0'
"""

module = Module(
    "fractureEnergyIntegrator",
    "A simple integrator to compute the fracture energy by integrating a load-displacement curve.",
)

inputLanguage = InputLanguage()

keyword = "output"
if keyword in inputLanguage:
    inputLanguage[keyword].addModule(module)

module.addRequiredArg("forceFieldOutput", "fieldOutput for force (with time history).", str)
module.addRequiredArg("displacementFieldOutput", "fieldOutput for displacement (with time history).", str)
module.addOptionalArg("f(x)", "Apply a model accessible function on the result.", str, "1")

documentation = [module]

required = [kw.name for kw in module.requiredArgs]
required += [kw.name for kw in module.requiredKeywords]

optional = [kw.name for kw in module.optionalArgs]
optional += [kw.name for kw in module.optionalKeywords]


@caseInsensitiveKwargsChecker(required, optional)
@castKwargsValuesAndAddDefaults(module)
def outputManagerFactory(name, FEModel, fieldOutputController, moduleOptions, journal, plotter, **kwargs):
    kwargs = CaseInsensitiveDict(kwargs)

    forceFieldOutputName = kwargs["forceFieldOutput"]
    displacementFieldOutputName = kwargs["displacementFieldOutput"]
    fractureArea = kwargs["f(x)"]

    if not fractureArea:
        fractureArea = "x"

    return OutputManager(
        name,
        FEModel,
        fieldOutputController,
        journal,
        plotter,
        forceFieldOutputName,
        displacementFieldOutputName,
        fractureArea,
    )


class OutputManager(OutputManagerBase):
    """Simple Integrator for fracture energy"""

    identification = "FEI"
    printTemplate = "{:}, {:}: {:}"

    def __init__(
        self,
        name,
        model,
        fieldOutputController,
        journal,
        plotter,
        forceFieldOutputName,
        displacementFieldOutput,
        fractureArea,
    ):
        self.journal = journal
        self.monitorJobs = []
        self.fieldOutputController = fieldOutputController

        self.fpF = self.fieldOutputController.fieldOutputs[forceFieldOutputName]
        self.fpU = self.fieldOutputController.fieldOutputs[displacementFieldOutput]
        self.A = createMathExpression(fractureArea)(0.0)
        self.fractureEnergy = 0.0

    def initializeJob(self):
        pass

    def initializeStep(self, step):
        pass

    def finalizeIncrement(self, **kwargs):
        pass

    def finalizeFailedIncrement(self, **kwargs):
        pass

    def finalizeStep(
        self,
    ):
        pass

    def finalizeJob(
        self,
    ):
        self.fractureEnergy = np.trapz(self.fpF.getResultHistory(), x=self.fpU.getResultHistory()) / self.A
        self.journal.message(
            "integrated fracture energy: {:3.4f}".format(self.fractureEnergy),
            self.identification,
        )
