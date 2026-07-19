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
