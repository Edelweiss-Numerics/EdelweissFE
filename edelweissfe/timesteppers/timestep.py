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


class TimeStep:
    def __init__(
        self,
        number: int,
        stepProgressIncrement: float,
        stepProgress: float,
        timeIncrement: float,
        stepTime: float,
        totalTime: float,
    ):
        """
        This is the representation of a time step in a simulation.
        It has knowledge about the current time and time increment,
        but also the relative dimensionless progress within a overarching load step
        (:class:`edelweissfe.steps.base.stepbase.StepBase`).

        Parameters
        ----------
        number
            The consecutive number of this TimeStep.
        stepProgressIncrement
            The relative increment within the current loading step.
        stepProgress
            The total relative progress within the current loading step.
        timeIncrement
            The time increment within the total simulation time.
        stepTime
            The current time in the step.
        totalTime
            The total time.
        """

        self.number = number
        self.stepProgressIncrement = stepProgressIncrement
        self.stepProgress = stepProgress
        self.timeIncrement = timeIncrement
        self.stepTime = stepTime
        self.totalTime = totalTime


def writeTimeStep(group, name: str, timeStep: TimeStep | None):
    """Write ``timeStep`` (or its absence) into an HDF5 group of a restart checkpoint.

    Parameters
    ----------
    group
        The group to write into.
    name
        The name of the entry.
    timeStep
        The time step, or None.
    """

    if timeStep is None:
        group.attrs[name] = []
        return
    group.attrs[name] = [
        timeStep.number,
        timeStep.stepProgressIncrement,
        timeStep.stepProgress,
        timeStep.timeIncrement,
        timeStep.stepTime,
        timeStep.totalTime,
    ]


def readTimeStep(group, name: str) -> TimeStep | None:
    """Read back what :func:`writeTimeStep` wrote.

    Parameters
    ----------
    group
        The group to read from.
    name
        The name of the entry.

    Returns
    -------
    TimeStep | None
        The time step, or None if none was written.
    """

    values = [float(value) for value in group.attrs[name]]
    if not values:
        return None
    return TimeStep(int(values[0]), *values[1:])
