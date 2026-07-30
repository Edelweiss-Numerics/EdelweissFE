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
"""One or more Steps form the simulation. Steps are executed in consecutive order,
have a distinct (physical) runtime, and may contain multiple StepActions.
Subsequent Steps inherit StepActions, and they may be updated."""

from abc import ABC, abstractmethod
from dataclasses import dataclass

from edelweissfe.journal.journal import Journal
from edelweissfe.models.femodel import FEModel
from edelweissfe.timesteppers.base.timestepperbase import TimeStepperBase
from edelweissfe.timesteppers.timestep import TimeStep
from edelweissfe.utils.fieldoutput import FieldOutputController
from edelweissfe.utils.schema import buildSchemaFromOptions, schemaField


@dataclass(frozen=True)
class StepIncrementationSchema:
    """L2: the standard incrementation options common to every step type (``*step`` keyword
    datalines), owned by this module and never mutated from outside it.

    Every concrete :class:`StepBase` subclass shares exactly this option set -- there is no
    per-step-type variation -- so it is declared once here rather than once per step type.
    """

    stepLength: float = schemaField(description="The duration of the step.", dtype=float, default=1.0)
    startInc: float = schemaField(
        description="The initial fraction of the step to be computed.", dtype=float, default=1.0
    )
    maxInc: float = schemaField(
        description="The maximal fraction of the step to be computed.", dtype=float, default=1.0
    )
    minInc: float = schemaField(
        description="The minimal fraction of the step to be computed.", dtype=float, default=1e-4
    )
    maxNumInc: int = schemaField(description="The maximal number of increments allowed.", dtype=int, default=1000)
    maxIter: int = schemaField(description="The maximal number of iterations allowed.", dtype=int, default=10)
    criticalIter: int = schemaField(
        description="The number of critical iterations after which the next increment is reduced.",
        dtype=int,
        default=5,
    )
    maxGrowIter: int = schemaField(
        description="The number of residual growths before the increment is discarded.", dtype=int, default=10
    )
    cutbackFactor: float = schemaField(
        description="Factor by which the increment size is reduced if no convergence was achieved.",
        dtype=float,
        default=0.25,
    )


class StepBase(ABC):
    """Base class for simulation steps.

    A step has a specific runtime, holds the StepActions to be executed,
    and delegates the incrementation to a time stepper, which is created by the
    concrete step type.

    Parameters
    ----------
    number
        The (unique) number of this step.
    model
        The current state of the model.
    fieldOutputController
        The FieldOutputController instance for processing results.
    journal
        The Journal instance for logging purposes.
    jobInfo
        Additional information about the job.
    solver
        The solver instance to be used for this step.
    outputManagers
        The OutputManagers used.
    stepActions
        The collection of actions for this step, grouped by action type.
    **kwargs
        The options for this step.
    """

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
        options = buildSchemaFromOptions(StepIncrementationSchema, kwargs)

        self.number = number  #: The (unique) number of the step.
        self.model = model
        self.fieldOutputController = fieldOutputController
        self.journal = journal
        self.solver = solver
        self.outputManagers = outputManagers
        self.actions = stepActions

        self.length = options.stepLength  #: The duration of the step.
        self.startIncrementSize = options.startInc
        self.maxIncrementSize = options.maxInc
        self.minIncrementSize = options.minInc
        self.maxNumberIncrements = options.maxNumInc
        self.maxIter = options.maxIter
        self.criticalIter = options.criticalIter
        self.maxGrowIter = options.maxGrowIter
        self.cutbackFactor = options.cutbackFactor

        self.timeStepper = self._createTimeStepper()

    @abstractmethod
    def _createTimeStepper(self) -> TimeStepperBase:
        """Create the time stepper for this step type.

        Returns
        -------
        TimeStepperBase
            The time stepper controlling the incrementation of this step.
        """

    def solve(self):
        """Let this step be solved by its solver, including the surrounding
        bookkeeping of field outputs and output managers."""

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

    def getTimeStep(self, enforcedTimeIncrement: float = None) -> TimeStep:
        """Generate the sequence of time steps for this step.

        Parameters
        ----------
        enforcedTimeIncrement
            If given, enforce this time increment size (if supported by the time stepper).

        Returns
        -------
        TimeStep
            The generated time steps (generator).
        """

        return self.timeStepper.generateTimeStep(enforcedTimeIncrement=enforcedTimeIncrement)

    def discardAndChangeIncrement(self, cutbackFactor: float):
        """Discard the current increment and modify the increment size by a given scale factor.

        Parameters
        ----------
        cutbackFactor
            The factor for scaling based on the discarded increment.
        """

        return self.timeStepper.discardAndChangeIncrement(cutbackFactor)

    def changeIncrementSize(self, scaleFactor: float):
        """Modify the size of the next increment by a given scale factor.

        Parameters
        ----------
        scaleFactor
            The factor for scaling based on the current increment.
        """

        return self.timeStepper.changeIncrementSize(scaleFactor)

    def preventIncrementIncrease(self):
        """Prevent an automatic increase of the increment size for the next increment."""

        return self.timeStepper.preventIncrementIncrease()
