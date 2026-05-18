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

from edelweissfe.journal.journal import Journal
from edelweissfe.models.femodel import FEModel
from edelweissfe.steps.base.stepbase import StepBase
from edelweissfe.timesteppers.adaptivetimestepper import AdaptiveTimeStepper
from edelweissfe.timesteppers.timestep import TimeStep
from edelweissfe.utils.caseinsensitivedict import CaseInsensitiveDict
from edelweissfe.utils.fieldoutput import FieldOutputController
from edelweissfe.utils.inputlanguage import InputLanguage
from edelweissfe.utils.misc import (
    caseInsensitiveKwargsChecker,
    castKwargsValuesAndAddDefaults,
)

inputLanguage = InputLanguage()
module = inputLanguage["step"].addModule(
    "adaptive", "A standard adaptive incremental step to be used in nonlinear simulations."
)

kw = module.addOptionalArg("stepLength", "The durcation of the step.", float, 1.0)
kw = module.addOptionalArg("startInc", "The initial fraction of the step to be computed.", float, 1.0)
kw = module.addOptionalArg("maxInc", "The maximal fraction of the step to be computed.", float, 1.0)
kw = module.addOptionalArg("minInc", "The minimal fraction of the step to be computed.", float, 1e-4)
kw = module.addOptionalArg("maxNumInc", "The maximal number of increments allowed.", int, 1000)
kw = module.addOptionalArg("maxIter", "The maximal number of iterations allowed.", int, 10)
kw = module.addOptionalArg(
    "criticalIter", "The number of critical iterations after which the next increment is reduced.", int, 5
)
kw = module.addOptionalArg("maxGrowIter", "The number of residual growths before the increment is discarded.", int, 10)
kw = module.addOptionalArg(
    "cutbackFactor", "Factor by which the increment size is reduced if no convergence was achieved.", float, 0.25
)

required = [kw.name for kw in module.requiredArgs]
required += [kw.name for kw in module.requiredKeywords]

optional = [kw.name for kw in module.optionalArgs]
optional += [kw.name for kw in module.optionalKeywords]


class AdaptiveStep(StepBase):
    """
    An adaptive incremental step to be used in nonlinear simulations with implicit time integration.
    """

    @caseInsensitiveKwargsChecker(required, optional)
    @castKwargsValuesAndAddDefaults(module)
    def __init__(
        self,
        number: int,
        model: FEModel,
        fieldOutputController: FieldOutputController,
        journal: Journal,
        jobInfo: dict,
        solver,
        outputManagers: list,
        stepActions: dict,
        **kwargs,
    ):
        kwargs = CaseInsensitiveDict(kwargs)
        self.number = number  #: The (unique) number of the step.

        self.model = model
        self.fieldOutputController = fieldOutputController
        self.journal = journal
        self.solver = solver
        self.outputManagers = outputManagers

        self.length = kwargs.get("stepLength", 1.0)  #: The durcation of the step.
        self.startIncrementSize = kwargs["startInc"]
        self.maxIncrementSize = kwargs["maxInc"]
        self.minIncrementSize = kwargs["minInc"]
        self.maxNumberIncrements = kwargs["maxNumInc"]
        self.maxIter = kwargs["maxIter"]
        self.criticalIter = kwargs["criticalIter"]
        self.maxGrowIter = kwargs["maxGrowIter"]
        self.cutbackFactor = kwargs["cutbackFactor"]

        self.incrementGenerator = AdaptiveTimeStepper(
            model.time,
            self.length,
            self.startIncrementSize,
            self.maxIncrementSize,
            self.minIncrementSize,
            self.maxNumberIncrements,
            journal,
        )

        self.actions = stepActions

    def solve(
        self,
    ):
        model = self.model
        fieldOutputController = self.fieldOutputController
        journal = self.journal
        outputManagers = self.outputManagers

        try:
            for modelUpdate in self.actions["modelupdate"].values():
                model = modelUpdate.updateModel(model, fieldOutputController, journal)

            fieldOutputController.initializeStep(self)
            for manager in outputManagers:
                manager.initializeStep(self)

            self.solver.solveStep(self, model, fieldOutputController, outputManagers)

        finally:
            fieldOutputController.finalizeStep()
            for manager in outputManagers:
                manager.finalizeStep()

    def getTimeStep(
        self,
    ) -> TimeStep:
        return self.incrementGenerator.generateTimeStep()

    def discardAndChangeIncrement(self, cutbackFactor: float):
        return self.incrementGenerator.discardAndChangeIncrement(cutbackFactor)

    def preventIncrementIncrease(
        self,
    ):
        return self.incrementGenerator.preventIncrementIncrease()
