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
# Created on Sat Jan  21 12:18:10 2017

from edelweissfe.journal.journal import Journal
from edelweissfe.timesteppers.base.timestepperbase import TimeStepperBase
from edelweissfe.timesteppers.timestep import TimeStep
from edelweissfe.utils.exceptions import ReachedMaxIncrements, ReachedMinIncrementSize


class SimpleTimeStepper(TimeStepperBase):
    """Divides a step into increments of a fixed size, or of a size enforced by the solver, after one
    zero increment.

    The zero increment comes first, so that an explicit integrator can build its initial state (mass,
    internal force, contact) before taking a step. It is numbered 0; the increments after it are
    numbered from 1. A step resumed from a checkpoint does not repeat it: whether it was done is part
    of the stepper's state.

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
    """

    identification = "SimpleTimeStepper"

    checkpointedState = {
        "currentTime": float,
        "zeroIncrementDone": bool,
        "totalIncrements": int,
        "finishedStepProgress": float,
        "increment": float,
        "enforcedTimeIncrement": float,
    }

    def __init__(
        self,
        currentTime: float,
        stepLength: float,
        startIncrement: float,
        maxIncrement: float,
        minIncrement: float,
        maxNumberIncrements: float,
        journal: Journal,
    ):
        self.startIncrement = startIncrement
        self.maxIncrement = maxIncrement
        self.minIncrement = minIncrement
        self.maxNumberIncrements = maxNumberIncrements
        self.stepLength = stepLength
        self.journal = journal

        #: The start time of the step.
        self.currentTime = currentTime
        #: Whether the zero increment was accepted.
        self.zeroIncrementDone = False
        #: The number of increments accepted after the zero increment.
        self.totalIncrements = 0
        #: The accepted progress within the step, from 0 to 1.
        self.finishedStepProgress = 0.0
        #: The size of the next increment, relative to the step length.
        self.increment = min(startIncrement, maxIncrement)
        #: The time increment the solver enforces, or None.
        self.enforcedTimeIncrement = None

    def isFinished(self) -> bool:
        return self.zeroIncrementDone and self.finishedStepProgress >= (1.0 - 1e-15)

    def isAtStepStart(self) -> bool:
        return not self.zeroIncrementDone

    def numberOfIncrementsDone(self) -> int:
        return self.totalIncrements

    def proposeTimeStep(self) -> TimeStep:
        progress = self.finishedStepProgress

        if not self.zeroIncrementDone:
            return TimeStep(
                0, 0.0, progress, 0.0, self.stepLength * progress, self.currentTime + self.stepLength * progress
            )

        if self.totalIncrements >= self.maxNumberIncrements:
            self.journal.message("Reached maximum number of increments", self.identification)
            raise ReachedMaxIncrements()

        if self.increment > self.maxIncrement:
            self.increment = self.maxIncrement
        if self.enforcedTimeIncrement is not None:
            self.increment = self.enforcedTimeIncrement / self.stepLength
        remainder = 1.0 - progress
        if remainder < self.increment:
            self.increment = remainder

        progress += self.increment
        return TimeStep(
            self.totalIncrements + 1,
            self.increment,
            progress,
            self.stepLength * self.increment,
            self.stepLength * progress,
            self.currentTime + self.stepLength * progress,
        )

    def acceptTimeStep(self, timeStep: TimeStep):
        if not self.zeroIncrementDone:
            self.zeroIncrementDone = True
            return

        self.finishedStepProgress = timeStep.stepProgress
        self.totalIncrements += 1

    def rejectTimeStep(self, cutbackFactor: float):
        if not self.zeroIncrementDone:
            self.journal.errorMessage("Failed zero increment", self.identification)
            raise ReachedMinIncrementSize()

        if self.increment == self.minIncrement:
            self.journal.errorMessage("Cannot reduce increment size", self.identification)
            raise ReachedMinIncrementSize()

        self.increment = min(max(self.increment * cutbackFactor, self.minIncrement), self.maxIncrement)

        if self.enforcedTimeIncrement is not None:
            self.enforcedTimeIncrement = self.increment * self.stepLength

        self.journal.message("Cutback to increment size {:}".format(self.increment), self.identification, 2)

    def enforceTimeIncrement(self, timeIncrement: float):
        self.enforcedTimeIncrement = timeIncrement

    def changeIncrementSize(self, scaleFactor: float):
        if not self.zeroIncrementDone:
            return

        self.increment = min(max(self.increment * scaleFactor, self.minIncrement), self.maxIncrement)

        self.journal.message("New increment size {:}".format(self.increment), self.identification, 2)

    def preventIncrementIncrease(self):
        """This time stepper never increases the increment size automatically, hence this is a no-op."""
