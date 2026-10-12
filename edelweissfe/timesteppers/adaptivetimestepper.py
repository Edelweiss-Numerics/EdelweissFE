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
# Created on Sat Jan  21 12:18:10 2017

from edelweissfe.journal.journal import Journal
from edelweissfe.timesteppers.base.timestepperbase import TimeStepperBase
from edelweissfe.timesteppers.timestep import TimeStep
from edelweissfe.utils.exceptions import ReachedMaxIncrements, ReachedMinIncrementSize


class AdaptiveTimeStepper(TimeStepperBase):
    """Divides a step into increments whose size adapts to the convergence of the solver: it grows
    after good increments and is cut back after failed ones.

    Parameters
    ----------
    currentTime
        The start time of the step.
    stepLength
        The total length of the step.
    startIncrement
        The size of the start increment, relative to the step length.
    maxIncrement
        The maximum size of an increment, relative to the step length.
    minIncrement
        The minimum size of an increment, relative to the step length.
    maxNumberIncrements
        The maximum number of allowed increments.
    journal
        The journal instance for logging purposes.
    increaseFactor
        The ratio to increase the increments in case of good convergence.
    goodIncrementsBeforeIncrease
        The number of good increments that must already have passed before an accepted increment lets the next one
        grow: counted from the step start (the zero increment included), or from the last cutback, which itself counts
        as one. From then on, every further good increment lets the next one grow.
    makeZeroIncrementFirst
        If True, the first increment will be zero.
    """

    identification = "AdaptiveTimeStepper"

    checkpointedState = {
        "currentTime": float,
        "incrementCounter": int,
        "nPassedGoodIncrements": int,
        "finishedStepProgress": float,
        "increment": float,
        "allowedToIncreasedNext": bool,
    }

    def __init__(
        self,
        currentTime: float,
        stepLength: float,
        startIncrement: float,
        maxIncrement: float,
        minIncrement: float,
        maxNumberIncrements: int,
        journal: Journal,
        increaseFactor: float = 1.1,
        goodIncrementsBeforeIncrease: int = 3,
        makeZeroIncrementFirst: bool = True,
    ):
        if increaseFactor < 1.0:
            raise ValueError("increaseFactor must be at least 1, got {:}".format(increaseFactor))
        if goodIncrementsBeforeIncrease < 1:
            raise ValueError(
                "goodIncrementsBeforeIncrease must be at least 1, got {:}".format(goodIncrementsBeforeIncrease)
            )

        self.startIncrement = startIncrement
        self.maxIncrement = maxIncrement
        self.minIncrement = minIncrement
        self.maxNumberIncrements = maxNumberIncrements
        self.stepLength = stepLength
        self.journal = journal
        self.increaseFactor = increaseFactor
        self.goodIncrementsBeforeIncrease = goodIncrementsBeforeIncrease
        self.makeZeroIncrementFirst = makeZeroIncrementFirst

        #: The start time of the step.
        self.currentTime = currentTime
        #: The number of the next increment; the zero increment, if any, is number 0.
        self.incrementCounter = 0
        #: Counts the good increments; the increment grows once goodIncrementsBeforeIncrease have passed.
        self.nPassedGoodIncrements = 0
        #: The accepted progress within the step, from 0 to 1.
        self.finishedStepProgress = 0.0
        #: The size of the next increment, relative to the step length.
        self.increment = min(startIncrement, maxIncrement)
        #: False if the solver asked to keep the next increment from growing.
        self.allowedToIncreasedNext = True

    def isFinished(self) -> bool:
        return self.finishedStepProgress >= (1.0 - 1e-15)

    def isAtStepStart(self) -> bool:
        return self.incrementCounter == 0

    def numberOfIncrementsDone(self) -> int:
        return self.incrementCounter

    def proposeTimeStep(self) -> TimeStep:
        if self.incrementCounter > self.maxNumberIncrements:
            self.journal.message("Reached maximum number of increments", self.identification)
            raise ReachedMaxIncrements()

        remainder = 1.0 - self.finishedStepProgress
        if remainder < self.increment:
            self.increment = remainder

        if self.makeZeroIncrementFirst and self.incrementCounter == 0:
            theIncrement = 0.0
        else:
            theIncrement = self.increment

        progress = self.finishedStepProgress + theIncrement
        return TimeStep(
            self.incrementCounter,
            theIncrement,
            progress,
            self.stepLength * theIncrement,
            self.stepLength * progress,
            self.currentTime + self.stepLength * progress,
        )

    def acceptTimeStep(self, timeStep: TimeStep):
        self.finishedStepProgress = timeStep.stepProgress

        if self.nPassedGoodIncrements >= self.goodIncrementsBeforeIncrease and self.allowedToIncreasedNext:
            self.increment = min(self.increment * self.increaseFactor, self.maxIncrement)
        self.allowedToIncreasedNext = True

        self.incrementCounter += 1
        self.nPassedGoodIncrements += 1

    def rejectTimeStep(self, cutbackFactor: float):
        if self.incrementCounter == 0:
            self.journal.errorMessage("Failed zero increment", self.identification)
            raise ReachedMinIncrementSize()

        if self.increment == self.minIncrement:
            self.journal.errorMessage("Cannot reduce increment size", self.identification)
            raise ReachedMinIncrementSize()

        # The retried increment counts as the first good one after the cutback, so the increment
        # grows again once goodIncrementsBeforeIncrease good increments have passed.
        self.nPassedGoodIncrements = 1
        self.allowedToIncreasedNext = True

        self.increment = min(max(self.increment * cutbackFactor, self.minIncrement), self.maxIncrement)

        self.journal.message("Cutback to increment size {:}".format(self.increment), self.identification, 2)

    def preventIncrementIncrease(self):
        self.allowedToIncreasedNext = False

    def changeIncrementSize(self, scaleFactor: float):
        self.increment = min(max(self.increment * scaleFactor, self.minIncrement), self.maxIncrement)

        self.journal.message("New increment size {:}".format(self.increment), self.identification, 2)
