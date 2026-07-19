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

from edelweissfe.steps.base.stepbase import (
    StepBase,
    addIncrementationOptionsToModule,
    getModuleArgNames,
)
from edelweissfe.timesteppers.simpletimestepper import SimpleTimeStepper
from edelweissfe.utils.inputlanguage import InputLanguage, Module
from edelweissfe.utils.misc import (
    caseInsensitiveKwargsChecker,
    castKwargsValuesAndAddDefaults,
)

module = Module(
    "adaptiveForExplicitSimulations",
    "An adaptive incremental step for nonlinear simulations with explicit time integration.",
)

inputLanguage = InputLanguage()

keyword = "step"
if keyword in inputLanguage:
    inputLanguage[keyword].addModule(module)

addIncrementationOptionsToModule(module)

required, optional = getModuleArgNames(module)


class AdaptiveStepForExplicitSimulations(StepBase):
    """
    An adaptive incremental step to be used in nonlinear simulations with explicit time integration.
    """

    @caseInsensitiveKwargsChecker(required, optional)
    @castKwargsValuesAndAddDefaults(module)
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _createTimeStepper(self) -> SimpleTimeStepper:
        return SimpleTimeStepper(
            self.model.time,
            self.length,
            self.startIncrementSize,
            self.maxIncrementSize,
            self.minIncrementSize,
            self.maxNumberIncrements,
            self.journal,
        )
