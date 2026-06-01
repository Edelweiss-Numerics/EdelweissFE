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
# Created on Tue May  9 19:52:53 2017

# @author: Matthias Neuner

# @ Magdalena
"""
Initialize materials to an geostatic stress state
"""

import numpy as np

from edelweissfe.stepactions.base.stepactionbase import StepActionBase
from edelweissfe.steps.adaptivestep import InputLanguage

inputLanguage = InputLanguage()

modules = [
    inputLanguage["step"].getModule("adaptive"),
    inputLanguage["step"].getModule("adaptiveForExplicitSimulations"),
]

documentation = []

for module in modules:
    kw = module.addOptionalKeyword("geostatic", "Initialize materials to an geostatic stress state.")
    kw.addRequiredArg("name", "Name of the step action.", str)
    kw.addRequiredArg("p1", "sig_x=sig_y=sig_z in first point.", float)
    kw.addOptionalArg("p2", "sig_x=sig_y=sig_z in second point.", float, None)
    kw.addOptionalArg("h1", "y coordinate of first point", float, 1.0)
    kw.addOptionalArg("h2", "y coordinate of second point", float, -1.0)
    kw.addOptionalArg("xLateral", "ratio of sig_x/sig_y, default=1.0", float, 1.0)
    kw.addOptionalArg("zLateral", "ratio of sig_z/sig_y, default=1.0", float, 1.0)
    kw.addOptionalArg("elSet", "The element set for which the initaliziation is performed", str, "all")

    documentation.append(kw)


class StepAction(StepActionBase):
    """Initializes elements of set with an Abaqus-like geostatic stress state.
    Is automatically deactivated at the end of the step."""

    def __init__(self, name, action, jobInfo, model, fieldOutputController, journal):
        self.name = name

        self.geostaticElements = model.elementSets[action["elSet"]]
        self.p1 = action["p1"]
        if action["p2"] is not None:
            self.p2 = action["p2"]
        else:
            self.p2 = action["p1"]
        self.level1 = action["h1"]
        self.level2 = action["h2"]
        self.xLateral = action["xLateral"]
        self.zLateral = action["zLateral"]

        self.geostaticDefinition = np.array(
            [
                self.p1,
                self.level1,
                self.p2,
                self.level2,
                self.xLateral,
                self.zLateral,
            ]
        )

        self.journal = journal

        self.active = True

    def applyAtStepEnd(self, model, stepMagnitude=None):
        if not self.active:
            return

        self.journal.printSeperationLine()
        self.journal.message("End of geostatic step -- displacements are reset", self.name)
        self.journal.printSeperationLine()

        model.nodeFields["displacement"]["U"][:] = 0
        # U[self.theDofManager.indicesOfFieldsInDofVector["displacement"]] = 0.0

        self.active = False

    def applyAtIterationStart(
        self,
    ):
        if not self.active:
            return

        for el in self.geostaticElements:
            el.setInitialCondition("geostatic stress", self.geostaticDefinition)
