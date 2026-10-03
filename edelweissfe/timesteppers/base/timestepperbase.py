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
"""A time stepper divides a simulation step into increments. It is a small state machine:

.. code-block:: text

    while not stepper.isFinished():
        timeStep = stepper.proposeTimeStep()      # the next increment; changes nothing
        try:
            solve the increment
        except failed:
            stepper.rejectTimeStep(cutbackFactor) # a smaller increment is proposed next
            continue
        stepper.acceptTimeStep(timeStep)          # the step advances by this increment
        write output (and restart checkpoints)

All the stepper knows is in its attributes, and after :meth:`acceptTimeStep` they describe exactly
the state the next increment starts from. A restart checkpoint is written after an increment is
accepted, so it holds those attributes as they are, see :meth:`writeRestart`.
"""

from abc import ABC, abstractmethod

import numpy as np

from edelweissfe.timesteppers.timestep import TimeStep


class TimeStepperBase(ABC):
    """Base class for all time steppers.

    It defines the interface which all solvers may rely on for controlling
    the incrementation of a simulation step.
    """

    #: Overridden by every stepper; a default so base-class diagnostics can name their source.
    identification = "TimeStepper"

    #: The attributes that make up the stepper's progress within the step, written to and read from
    #: a restart checkpoint as they are. Not its configuration (step length, increment bounds, the
    #: maximum number of increments): a resumed run takes that from its own input file, so it can be
    #: changed between runs, e.g. to raise the maximum number of increments.
    checkpointedState: tuple[str, ...] = ()

    @abstractmethod
    def isFinished(self) -> bool:
        """Whether the step has reached its end.

        Returns
        -------
        bool
            True if no increment is left.
        """

    @abstractmethod
    def proposeTimeStep(self) -> TimeStep:
        """The next increment, starting from the last accepted one. Changes nothing: proposing twice
        gives the same increment.

        Returns
        -------
        TimeStep
            The proposed increment.

        Raises
        ------
        ReachedMaxIncrements
            If the step needs more increments than it is allowed.
        """

    @abstractmethod
    def acceptTimeStep(self, timeStep: TimeStep):
        """Advance the step by ``timeStep``, the increment just solved, and choose the size of the next.

        Parameters
        ----------
        timeStep
            The increment, as :meth:`proposeTimeStep` returned it.
        """

    @abstractmethod
    def rejectTimeStep(self, cutbackFactor: float):
        """Discard the proposed increment, which could not be solved; the next one proposed is
        smaller by ``cutbackFactor``, within the bounds of the minimum and maximum increment size.

        Parameters
        ----------
        cutbackFactor
            The factor scaling the increment size.

        Raises
        ------
        ReachedMinIncrementSize
            If the increment cannot be reduced any further.
        """

    @abstractmethod
    def changeIncrementSize(self, scaleFactor: float):
        """Modify the size of the next increment by a given scale factor
        within the bounds of the minimum and maximum increment size.

        Parameters
        ----------
        scaleFactor
            The factor for scaling based on the current increment.
        """

    @abstractmethod
    def preventIncrementIncrease(self):
        """Keep the next increment from growing, e.g., in case of bad convergence. Called before
        :meth:`acceptTimeStep`."""

    def enforceTimeIncrement(self, timeIncrement: float):
        """Use ``timeIncrement`` for every following increment, instead of adapting it.

        The caller is an explicit solver, whose stable time increment is a property of the mesh: set
        at the start of the step, and lowered when a refinement shrinks the smallest element.

        Parameters
        ----------
        timeIncrement
            The time increment to use.

        Raises
        ------
        NotImplementedError
            If this stepper cannot run on an enforced time increment.
        """

        raise NotImplementedError(f"{type(self).__name__} cannot run on an enforced time increment.")

    def writeRestart(self, restartFile):
        """Write the attributes named in :attr:`checkpointedState` to a restart checkpoint, as they
        are. An attribute that is None is written as an empty list.

        Parameters
        ----------
        restartFile
            An open, writable :class:`h5py.File` (or group) to write the checkpoint into.
        """

        group = restartFile.create_group("timestepper")
        for name in self.checkpointedState:
            value = self.__dict__[name]
            group.attrs[name] = [] if value is None else value

    def readRestart(self, restartFile):
        """Read the attributes named in :attr:`checkpointedState` back from a restart checkpoint
        written by :meth:`writeRestart`.

        Parameters
        ----------
        restartFile
            An open, readable :class:`h5py.File` (or group) to read the checkpoint from.
        """

        group = restartFile["timestepper"]
        for name in self.checkpointedState:
            value = group.attrs[name]
            self.__dict__[name] = None if np.size(value) == 0 else value

        self._warnIfResumedAtIncrementCap()

    @abstractmethod
    def numberOfIncrementsDone(self) -> int:
        """The number of increments accepted so far, counted against the maximum number of increments.

        Returns
        -------
        int
            The number of increments.
        """

    def _warnIfResumedAtIncrementCap(self):
        """Warn when a checkpoint is resumed at or past the step's increment cap.

        The maximum number of increments counts from the start of the analysis, not from the resume,
        and it is taken from the step's own configuration rather than the checkpoint, so it can be
        raised between runs. The trap: resume without raising it far enough, and the stepper's first
        proposal ends the step, which the solver treats as a step that finished normally -- the job
        reports success having advanced nothing. A diagnostic run resumed at increment 130 000 with
        the cap still at 60 000 was very nearly read as evidence that a crash did not reproduce.

        This does not change the behaviour -- the cap is absolute by design -- only the silence.
        """

        if self.numberOfIncrementsDone() < self.maxNumberIncrements:
            return

        self.journal.message(
            "WARNING: this checkpoint is already at increment {0:}, at or past this step's "
            "maxNumInc of {1:}, so the resumed step will end immediately WITHOUT advancing the "
            "solution -- and the job will then report success. maxNumInc counts increments from "
            "the start of the analysis, not from the resume: raise it above {0:} to continue.".format(
                self.numberOfIncrementsDone(), self.maxNumberIncrements
            ),
            self.identification,
            0,
        )
