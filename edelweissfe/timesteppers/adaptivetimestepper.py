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
    identification = "AdaptiveTimeStepper"

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
        makeZeroIncrementFirst: bool = True,
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
        increaseFactor
            The ratio to increase the increments in case of good convergence.
        makeZeroIncrementFirst
            If True, the first increment will be zero.
        """

        self.nPassedGoodIncrements = int(0)
        self.incrementCounter = int(0)
        self.startIncrement = startIncrement
        self.maxIncrement = maxIncrement
        self.minIncrement = minIncrement
        self.maxNumberIncrements = maxNumberIncrements

        self.finishedStepProgress = 0.0
        self.increment = min(startIncrement, maxIncrement)
        self.allowedToIncreasedNext = True

        self.currentTime = currentTime
        self.stepLength = stepLength
        self.dT = 0.0
        self.journal = journal
        self.increaseFactor = increaseFactor
        self.makeZeroIncrementFirst = makeZeroIncrementFirst

    def doesZeroIncrement(self):
        """Whether the *next* generated time step will be a zero increment.

        This reports what will actually happen, not merely that zero increments are configured: the
        zero increment is only emitted as the very first step of a step, so after a restart -- which
        restores a non-zero increment counter -- there is none. Callers that require a zero increment
        in order to initialise themselves have to know the difference.

        Returns
        -------
        bool
            True if the next generated time step will have a zero increment.
        """

        return self.makeZeroIncrementFirst and self.incrementCounter == 0

    def generateTimeStep(self, enforcedTimeIncrement: float = None) -> TimeStep:
        """
        Generate the next increment.

        Parameters
        ----------
        enforcedTimeIncrement
            Not supported by this time stepper; a ValueError is raised if it is given.

        Returns
        -------
        TimeStep
            The current time step.
        """

        if enforcedTimeIncrement is not None:
            raise ValueError("AdaptiveTimeStepper does not support enforced time increments")

        while self.finishedStepProgress < (1.0 - 1e-15):

            remainder = 1.0 - self.finishedStepProgress
            if remainder < self.increment:
                self.increment = remainder

            # Force a zero increment only for the first generated step when enabled.
            if self.makeZeroIncrementFirst and (self.incrementCounter == 0):
                theIncrement = 0.0
            else:
                theIncrement = self.increment

            dT = self.stepLength * theIncrement
            # Record the increment actually used, for the same reason as
            # :class:`~edelweissfe.timesteppers.simpletimestepper.SimpleTimeStepper`: writeRestart
            # checkpoints ``self.dT``. The zero increment optionally generated first is deliberately
            # NOT recorded -- it is not a completed increment, and a multi-step integrator resuming
            # from it needs the last real one.
            if theIncrement > 0.0:
                self.dT = dT
            self.finishedStepProgress += theIncrement
            endTimeOfIncrementInStep = self.stepLength * self.finishedStepProgress
            endTimeOfIncrementInTotal = self.currentTime + endTimeOfIncrementInStep

            yield TimeStep(
                self.incrementCounter,
                theIncrement,
                self.finishedStepProgress,
                dT,
                endTimeOfIncrementInStep,
                endTimeOfIncrementInTotal,
            )

            if self.incrementCounter >= self.maxNumberIncrements:
                self.journal.message("Reached maximum number of increments", self.identification)
                raise ReachedMaxIncrements()

            if (self.nPassedGoodIncrements >= 3) and self.allowedToIncreasedNext:
                self.increment *= self.increaseFactor
                if self.increment > self.maxIncrement:
                    self.increment = self.maxIncrement
            self.allowedToIncreasedNext = True

            self.incrementCounter += 1
            self.nPassedGoodIncrements += 1

    def preventIncrementIncrease(
        self,
    ):
        """May be called before an increment is requested, to prevent from
        automatically increasing, e.g. in case of bad convergency."""

        self.allowedToIncreasedNext = False

    def changeIncrementSize(self, scaleFactor: float):
        """Modify the size of the next increment by a given scale factor
        within the bounds of the minimum and maximum increment size.

        Parameters
        ----------
        scaleFactor
            The factor for scaling based on the current increment.
        """

        newIncrement = self.increment * scaleFactor
        self.increment = min(max(newIncrement, self.minIncrement), self.maxIncrement)

        self.journal.message(
            "New increment size {:}".format(self.increment),
            self.identification,
            2,
        )

    def reduceNextIncrement(self, scaleFactor: float):
        """Reduce the increment size for the next increment."""

        if self.increment == self.minIncrement:
            self.journal.errorMessage("Cannot reduce increment size", self.identification)
            raise ReachedMinIncrementSize()

        newIncrement = self.increment * scaleFactor
        if newIncrement > self.maxIncrement:
            self.increment = self.maxIncrement
        elif newIncrement < self.minIncrement:
            self.increment = self.minIncrement
        else:
            self.increment = newIncrement

        self.journal.message(
            "Cutback to increment size {:}".format(self.increment),
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

        if self.incrementCounter == 0:
            self.journal.errorMessage("Failed zero increment", self.identification)
            raise ReachedMinIncrementSize()

        self.finishedStepProgress -= self.increment
        self.incrementCounter -= 1
        self.nPassedGoodIncrements = 0

        self.reduceNextIncrement(scaleFactor)

    def writeRestart(self, restartFile):
        """Write this time stepper's progress within the step to a restart checkpoint.

        Deliberately restricted to the *dynamic* progress state (``currentTime`` -- the step's own
        absolute start time, fixed once the step began -- ``finishedStepProgress``,
        ``incrementCounter``, ``nPassedGoodIncrements``, ``increment``, ``allowedToIncreasedNext``,
        ``dT``), not the step's *configuration* (``stepLength``, ``startIncrement``,
        ``maxIncrement``, ``minIncrement``, ``maxNumberIncrements``): resuming reconstructs the step
        from the (possibly since-edited, e.g. a raised ``maxNumberIncrements`` after a run that hit
        it) ``.inp`` file being used to resume, exactly like :meth:`~edelweissfe.models.femodel.
        FEModel.readRestart` only overwrites converged state and never the model's own structural
        definition. Restoring the configuration fields here would silently re-clobber any such
        edit with whatever was in effect when the checkpoint was written.

        This runs (via the restart output manager's ``finalizeIncrement``) while
        :meth:`generateTimeStep` is paused *at* the ``yield`` for the increment that just
        converged -- i.e. before that generator's own post-yield bookkeeping (the growth-factor
        update and the ``incrementCounter``/``nPassedGoodIncrements`` advance) has run. An
        uninterrupted run never notices, since the same generator object applies that bookkeeping
        itself on its next resume. But a *fresh* generator built for a resumed run has never
        reached that yield point at all, so it would skip the bookkeeping entirely -- repeating
        the just-converged increment's size (mislabeled with its own ``incrementCounter``)
        instead of continuing the growth sequence. So this snapshots the state as it will be once
        that bookkeeping runs, replicating its exact logic, rather than the raw current attributes.

        Parameters
        ----------
        restartFile
            The file to write the restart information to.
        """
        f = restartFile
        f.create_group("timestepper")

        increment = self.increment
        if self.nPassedGoodIncrements >= 3 and self.allowedToIncreasedNext:
            increment = min(increment * self.increaseFactor, self.maxIncrement)

        f["timestepper"].attrs["currentTime"] = self.currentTime
        f["timestepper"].attrs["nPassedGoodIncrements"] = self.nPassedGoodIncrements + 1
        f["timestepper"].attrs["incrementCounter"] = self.incrementCounter + 1
        f["timestepper"].attrs["finishedStepProgress"] = self.finishedStepProgress
        f["timestepper"].attrs["increment"] = increment
        f["timestepper"].attrs["allowedToIncreasedNext"] = True
        f["timestepper"].attrs["dT"] = self.dT

    def readRestart(self, restartFile):
        """Restore this time stepper's progress within the step from a restart checkpoint written
        by :meth:`writeRestart`. See that method's docstring for why the step's configuration
        fields are deliberately left untouched (kept from this instance's own construction).

        Parameters
        ----------
        restartFile
            The file to read the restart information from.
        """
        f = restartFile
        self.currentTime = f["timestepper"].attrs["currentTime"]
        self.nPassedGoodIncrements = f["timestepper"].attrs["nPassedGoodIncrements"]
        self.incrementCounter = f["timestepper"].attrs["incrementCounter"]
        self.finishedStepProgress = f["timestepper"].attrs["finishedStepProgress"]
        self.increment = f["timestepper"].attrs["increment"]
        self.allowedToIncreasedNext = f["timestepper"].attrs["allowedToIncreasedNext"]
        self.dT = f["timestepper"].attrs["dT"]
