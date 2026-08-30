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
"""Time steppers generate the sequence of :class:`~edelweissfe.timesteppers.timestep.TimeStep` s
within a simulation step, and allow the solvers to control the incrementation
(cutbacks, rescaling, freezing the increment size)."""

from abc import ABC, abstractmethod

from edelweissfe.timesteppers.timestep import TimeStep


class TimeStepperBase(ABC):
    """Base class for all time steppers.

    It defines the interface which all solvers may rely on for controlling
    the incrementation of a simulation step.
    """

    @abstractmethod
    def generateTimeStep(self, enforcedTimeIncrement: float = None) -> TimeStep:
        """Generate the (sequence of) time steps.

        Parameters
        ----------
        enforcedTimeIncrement
            If given, enforce this time increment size (e.g., a critical time step
            in explicit simulations). Time steppers which do not support enforced
            increments raise a ValueError if it is given.

        Returns
        -------
        TimeStep
            The generated time steps (generator).
        """

    @abstractmethod
    def discardAndChangeIncrement(self, scaleFactor: float):
        """Discard the current increment, and modify the increment size
        by a given scale factor within the bounds of the minimum and maximum increment size.

        Parameters
        ----------
        scaleFactor
            The factor for scaling based on the discarded increment.
        """

    def enforceTimeIncrement(self, timeIncrement: float):
        """Replace the enforced time increment for the remaining increments of the step.

        Only meaningful for a stepper driven by an externally imposed increment rather than by its own
        adaptation. The caller is an explicit solver whose stable time increment is a property of the
        mesh, so it changes when the mesh does -- an h-adaptivity event mid-step can shrink the
        smallest element and therefore the increment every subsequent increment must use. The
        increment is passed to :meth:`generateTimeStep` once, before the first increment, so there
        would otherwise be no way to revise it.

        Parameters
        ----------
        timeIncrement
            The new enforced time increment.

        Raises
        ------
        NotImplementedError
            If this stepper is not driven by an enforced time increment.
        """

        raise NotImplementedError(
            f"{type(self).__name__} is not driven by an enforced time increment, so it cannot be given " "a new one."
        )

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
        """May be called before an increment is requested, to prevent
        an automatic increase of the increment size, e.g., in case of bad convergence."""

    def restoredTimeIncrement(self) -> float | None:
        """The size of the increment already completed when this time stepper was restored from a
        restart checkpoint.

        A multi-step integrator needs the previous increment size to continue, and on a resumed run
        that increment belongs to the run that wrote the checkpoint. Deliberately NOT abstract: a
        time stepper that does not persist its progress inherits the cold-start answer rather than
        being forced to implement something it has no state for.

        Returns
        -------
        float | None
            The completed increment size, or None if this time stepper is starting cold.
        """

        return None

    @abstractmethod
    def writeRestart(self, restartFile):
        """Write this time stepper's bookkeeping (current time, increment size, progress within
        the step, ...) to a restart checkpoint.

        Parameters
        ----------
        restartFile
            An open, writable :class:`h5py.File` (or group) to write the checkpoint into.
        """

    @abstractmethod
    def readRestart(self, restartFile):
        """Restore this time stepper's bookkeeping from a restart checkpoint written by
        :meth:`writeRestart`.

        Parameters
        ----------
        restartFile
            An open, readable :class:`h5py.File` (or group) to read the checkpoint from.
        """
