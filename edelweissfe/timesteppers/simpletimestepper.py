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
    identification = "SimpleTimeStepper"

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
        """
        An increment generator for incremental-iterative simulations.

        Implementation as generator class.

        Parameters
        ----------
        currentTime
            The current (start) time.
        stepLength
            The total length of the step.
        startIncrement
            The size of the start increment.
        maxIncrement
            The maximum size of an increment.
        minIncrement
            The minimum size of an increment.
        maxNumberIncrements
            The maximum number of allowed increments.
        journal
            The journal instance for logging purposes.
        """

        self.totalIncrements = int(0)
        self.startIncrement = startIncrement
        self.maxIncrement = maxIncrement
        self.minIncrement = minIncrement
        self.maxNumberIncrements = maxNumberIncrements

        self.finishedStepProgress = 0.0
        self.increment = min(startIncrement, maxIncrement)

        self.currentTime = currentTime
        self.stepLength = stepLength
        self.dT = 0.0
        self.journal = journal
        self.enforcedTimeIncrement = None

    def generateTimeStep(self, enforcedTimeIncrement=None) -> TimeStep:
        """
        Generate the next increment.

        Returns
        -------
        TimeStep
            The current time step.
        """

        # zero increment; return value for first function call
        yield TimeStep(0, 0.0, 0.0, 0.0, 0.0, self.currentTime)
        self.enforcedTimeIncrement = enforcedTimeIncrement

        if self.enforcedTimeIncrement is None:
            while self.finishedStepProgress < (1.0 - 1e-15):
                if self.totalIncrements >= self.maxNumberIncrements:
                    self.journal.message("Reached maximum number of increments", self.identification)
                    raise ReachedMaxIncrements()
                if self.increment > self.maxIncrement:
                    self.increment = self.maxIncrement

                remainder = 1.0 - self.finishedStepProgress
                if remainder < self.increment:
                    self.increment = remainder

                dT = self.stepLength * self.increment
                self.finishedStepProgress += self.increment
                endTimeOfIncrementInStep = self.stepLength * self.finishedStepProgress
                endTimeOfIncrementInTotal = self.currentTime + endTimeOfIncrementInStep

                self.totalIncrements += 1

                yield TimeStep(
                    self.totalIncrements,
                    self.increment,
                    self.finishedStepProgress,
                    dT,
                    endTimeOfIncrementInStep,
                    endTimeOfIncrementInTotal,
                )
        else:
            while self.finishedStepProgress < (1.0 - 1e-15):
                if self.totalIncrements >= self.maxNumberIncrements:
                    self.journal.message("Reached maximum number of increments", self.identification)
                    raise ReachedMaxIncrements()
                if self.increment > self.maxIncrement:
                    self.increment = self.maxIncrement

                # dT = self.enforcedTimeIncrement
                self.increment = self.enforcedTimeIncrement / self.stepLength
                remainder = 1.0 - self.finishedStepProgress
                if remainder < self.increment:
                    self.increment = remainder

                dT = self.stepLength * self.increment
                self.finishedStepProgress += self.increment
                endTimeOfIncrementInStep = self.stepLength * self.finishedStepProgress
                endTimeOfIncrementInTotal = self.currentTime + endTimeOfIncrementInStep

                self.totalIncrements += 1

                yield TimeStep(
                    self.totalIncrements,
                    self.increment,
                    self.finishedStepProgress,
                    dT,
                    endTimeOfIncrementInStep,
                    endTimeOfIncrementInTotal,
                )

    def changeIncrementSize(self, scaleFactor: float):
        """Change increment size between minIncrement and
        maxIncrement by a given scale factor.

        Parameters
        ----------
        scaleFactor
            The factor for scaling based on the previous increment.
        """

        if self.finishedStepProgress == 0.0:
            return

        newIncrement = self.increment * scaleFactor

        if newIncrement > self.maxIncrement:
            self.increment = self.maxIncrement
        elif newIncrement < self.minIncrement:
            self.increment = self.minIncrement
        else:
            self.increment = newIncrement

        self.journal.message(
            "New increment size {:}".format(self.increment),
            self.identification,
            2,
        )

    def discardAndChangeIncrement(self, scaleFactor: float):
        """Change increment size between minIncrement and
        maxIncrement by a given scale factor.

        Parameters
        ----------
        scaleFactor
            The factor for scaling based on the previous increment.
        """

        if self.increment == self.minIncrement:
            self.journal.errorMessage("Cannot reduce increment size", self.identification)
            raise ReachedMinIncrementSize()

        if self.finishedStepProgress == 0.0:
            self.journal.errorMessage("Failed zero increment", self.identification)
            raise ReachedMinIncrementSize()

        self.finishedStepProgress -= self.increment
        newIncrement = self.increment * scaleFactor
        if newIncrement > self.maxIncrement:
            self.increment = self.maxIncrement
        elif newIncrement < self.minIncrement:
            self.increment = self.minIncrement
        else:
            self.increment = newIncrement

        if self.enforcedTimeIncrement is not None:
            # overwrite the enforced time increment with the new increment size
            self.enforcedTimeIncrement = self.increment * self.stepLength

        self.journal.message(
            "Cutback to increment size {:}".format(self.increment),
            self.identification,
            2,
        )
        self.totalIncrements -= 1

    def preventIncrementIncrease(self):
        """This time stepper never increases the increment size automatically,
        hence this is a no-op."""

    def writeRestart(self, restartFile):
        """Write this time stepper's progress within the step to a restart checkpoint.

        Deliberately restricted to the *dynamic* progress state (``currentTime``, ``totalIncrements``,
        ``finishedStepProgress``, ``increment``, ``dT``), not the step's *configuration*
        (``stepLength``, ``startIncrement``, ``maxIncrement``, ``minIncrement``,
        ``maxNumberIncrements``) -- see :meth:`~edelweissfe.timesteppers.adaptivetimestepper.
        AdaptiveTimeStepper.writeRestart`'s docstring for why.

        Parameters
        ----------
        restartFile
            The file to write the restart information to.
        """
        f = restartFile
        f.create_group("timestepper")

        f["timestepper"].attrs["currentTime"] = self.currentTime
        f["timestepper"].attrs["totalIncrements"] = self.totalIncrements
        f["timestepper"].attrs["finishedStepProgress"] = self.finishedStepProgress
        f["timestepper"].attrs["increment"] = self.increment
        f["timestepper"].attrs["dT"] = self.dT

    def readRestart(self, restartFile):
        """Restore this time stepper's progress within the step from a restart checkpoint written
        by :meth:`writeRestart`.

        Parameters
        ----------
        restartFile
            The file to read the restart information from.
        """
        f = restartFile
        self.currentTime = f["timestepper"].attrs["currentTime"]
        self.totalIncrements = f["timestepper"].attrs["totalIncrements"]
        self.finishedStepProgress = f["timestepper"].attrs["finishedStepProgress"]
        self.increment = f["timestepper"].attrs["increment"]
        self.dT = f["timestepper"].attrs["dT"]
