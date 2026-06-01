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

import sympy as sp

from edelweissfe.stepactions.base.stepactionbase import StepActionBase
from edelweissfe.steps.adaptivestep import InputLanguage
from edelweissfe.timesteppers.timestep import TimeStep
from edelweissfe.utils.caseinsensitivedict import CaseInsensitiveDict

"""
Stepaction to change material properties.

"""


inputLanguage = InputLanguage()

modules = [
    inputLanguage["step"].getModule("adaptive"),
    inputLanguage["step"].getModule("adaptiveForExplicitSimulations"),
]

documentation = []

for module in modules:
    kw = module.addOptionalKeyword("changematerialproperty", "Stepaction to change material properties.")
    kw.addRequiredArg("name", "Name of the step action.", str)
    kw.addRequiredArg("material", "The id of the material to be changed", str)
    kw.addRequiredArg("index", "The index of the property in the material properties vector", int)
    kw.addOptionalArg("f(t)", "Define an amplitude in the step progress interval [0...1]", str, None)

    documentation.append(kw)


class StepAction(StepActionBase):
    """Action class for changing a material property"""

    def __init__(self, name, action, jobInfo, model, fieldOutputController, journal):
        self.name = name
        self.theIndex = int(action["index"])
        self.theMaterial = CaseInsensitiveDict(model.materials)[action["material"]]

        self.updateStepAction(action, jobInfo, model, fieldOutputController, journal)

        self.journal = journal

    def updateStepAction(self, action, jobInfo, model, fieldOutputController, journal):
        """Update the function describing the material property"""
        self.active = True

        if action["f(t)"] is not None:
            t = sp.symbols("t")
            self.f_t = sp.lambdify(t, sp.sympify(action["f(t)"]), "numpy")

    def applyAtStepEnd(self, model, stepMagnitude=None):
        self.active = False

    def applyAtIncrementStart(self, model, timeStep: TimeStep):
        """Change the actual properties depending on the current step time"""

        if not self.active:
            return

        theCurrentProperty = self.f_t(timeStep.stepTime)
        self.journal.message(
            "Changing property[{:}] of material {:} to {:}".format(
                self.theIndex, self.theMaterial["name"], theCurrentProperty
            ),
            self.name,
        )

        self.theMaterial["properties"][self.theIndex] = theCurrentProperty
        for secname, section in model.sections.items():
            if section.material["name"] == self.theMaterial["name"]:
                for elSet in section.elSets:
                    for el in elSet:
                        if isinstance(section.material, dict):  # for marmotmaterial provider
                            modifiedMaterial = section.material.copy()
                            modifiedMaterial["properties"] = self.theMaterial["properties"]
                        else:  # for edelweissmaterial provider
                            materialType = type(section.material)
                            modifiedProperties = self.theMaterial["properties"]
                            modifiedMaterial = materialType(modifiedProperties)
                            if hasattr(self.material, "_materialEnergy"):  # for autodiff materials
                                modifiedMaterial.setEnergyFunction(self.material._materialEnergy)
                        section.assignSectionPropertiesToElement(el, material=modifiedMaterial)
