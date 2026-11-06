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
from edelweissfe.utils import performancetiming
from edelweissfe.utils.checkpoint import ResumeCheckpoint
from edelweissfe.utils.exceptions import (
    ConditionalStop,
    IncrementFailed,
    MeshChangedSinceCheckpoint,
    ReachedMaxIncrements,
    ReachedMinIncrementSize,
    StepFailed,
)
from edelweissfe.utils.fieldoutput import FieldOutputController
from edelweissfe.utils.schema import buildSchemaFromOptions, schemaField


@dataclass(frozen=True)
class StepIncrementationSchema:
    """The standard incrementation options common to every step type (``*step`` keyword
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
        description="Increments needing at least this many iterations keep the next increment from growing.",
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
    increaseFactor: float = schemaField(
        description="Factor by which the increment size grows after good increments (adaptive steps only).",
        dtype=float,
        default=1.1,
    )
    goodIncrementsBeforeIncrease: int = schemaField(
        description=(
            "Number of good increments that must already have passed, since the step start or the last cutback "
            "(which counts as one), before an accepted increment lets the next one grow (adaptive steps only)."
        ),
        dtype=int,
        default=3,
    )
    makeZeroIncrementFirst: bool = schemaField(
        description=(
            "Begin the step with an increment of size zero, which brings the model into equilibrium with the state the "
            "step starts from before time advances (AdaptiveStep only: an explicit step always begins with it, and "
            "refuses False)."
        ),
        dtype=bool,
        default=True,
    )
    maxCheckpointFallbacks: int = schemaField(
        description=(
            "How often the step may fall back to its latest restart checkpoint when an increment fails and cannot be "
            "cut back any further; the step then continues from there with a smaller increment. 0: never."
        ),
        dtype=int,
        default=0,
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
        self.increaseFactor = options.increaseFactor
        self.goodIncrementsBeforeIncrease = options.goodIncrementsBeforeIncrease
        self.makeZeroIncrementFirst = options.makeZeroIncrementFirst
        self.maxCheckpointFallbacks = options.maxCheckpointFallbacks
        if self.maxCheckpointFallbacks < 0:
            raise ValueError("maxCheckpointFallbacks must not be negative, got {:}".format(self.maxCheckpointFallbacks))

        self.timeStepper = self._createTimeStepper()

    @abstractmethod
    def _createTimeStepper(self) -> TimeStepperBase:
        """Create the time stepper for this step type.

        Returns
        -------
        TimeStepperBase
            The time stepper controlling the incrementation of this step.
        """

    def solve(self, resumeFrom=None):
        """Solve this step, increment by increment.

        The increment loop, the same for every solver:

        .. code-block:: text

            begin the step
            while the step is not finished:
                prepare the increment     topology update when due
                propose an increment      time stepper
                attempt it                solver; if it fails: reject it, retry smaller
                                          (at the minimum size: fall back to a checkpoint, if allowed)
                accept it                 solver, then time stepper
                write output              field outputs, output managers, restart checkpoint last
            end the step

        A restart checkpoint is written after an accepted increment, as the last output, so it holds
        exactly the state the next increment starts from -- and a resumed step simply continues
        this loop. It begins like any step, and then takes over the checkpointed state, in this one
        place.

        An increment that fails at the minimum increment size fails the step, unless the step may
        fall back to a checkpoint (``maxCheckpointFallbacks``, see :meth:`fallBackToCheckpoint`):
        it then continues from its latest restart checkpoint, as a resumed step would, with a
        smaller increment.

        Parameters
        ----------
        resumeFrom
            The :class:`~edelweissfe.utils.checkpoint.ResumeCheckpoint` this step continues from, or
            None.
        """

        model = self.model
        solver = self.solver
        timeStepper = self.timeStepper
        fieldOutputController = self.fieldOutputController
        journal = self.journal
        outputManagers = self.outputManagers
        # Restart checkpoints last, so that they hold the bookkeeping of every other output.
        outputManagers = [m for m in outputManagers if not m.writesRestartCheckpoints] + [
            m for m in outputManagers if m.writesRestartCheckpoints
        ]
        restartWriters = [m.name for m in outputManagers if m.writesRestartCheckpoints]
        if self.maxCheckpointFallbacks and len(restartWriters) != 1:
            raise ValueError(
                "step {:}: maxCheckpointFallbacks needs exactly one restart output (*output, type=restart) to fall "
                "back to its checkpoints, found {:}: {:}".format(self.number, len(restartWriters), restartWriters)
            )

        if resumeFrom is not None:
            resumeFrom.restoreTimeStepper(self)

        try:
            # Step-start model updates. A resumed step has none: resuming past one is refused.
            for modelUpdate in self.actions["modelupdate"].values():
                model = modelUpdate.updateModel(model, fieldOutputController, journal)

            fieldOutputController.initializeStep(self)
            for manager in outputManagers:
                manager.initializeStep(self)

            solver.beginStep(self, model, fieldOutputController, outputManagers)
            if resumeFrom is not None:
                resumeFrom.restoreStep(self)
                # Closed here, before the first increment: see ResumeCheckpoint.close.
                resumeFrom.close()
            try:
                fallbacksLeft = self.maxCheckpointFallbacks
                isRetry = False
                while not timeStepper.isFinished():
                    solver.prepareIncrement(self, model, isRetry)
                    timeStep = timeStepper.proposeTimeStep()

                    try:
                        solver.attemptIncrement(self, model, timeStep)
                    except IncrementFailed as e:
                        journal.message(str(e), solver.identification, 1)
                        try:
                            timeStepper.rejectTimeStep(e.cutbackFactor)
                        except ReachedMinIncrementSize:
                            if not fallbacksLeft:
                                raise
                            fallbacksLeft = self.fallBackToCheckpoint(model, outputManagers, fallbacksLeft)
                        for manager in outputManagers:
                            manager.finalizeFailedIncrement(statusInfoDict=solver.incrementStatus)
                        isRetry = True
                        continue

                    solver.acceptIncrement(self, model, timeStep)
                    timeStepper.acceptTimeStep(timeStep)
                    isRetry = False

                    if solver.isOutputIncrement(timeStep):
                        with performancetiming.timeit("output"):
                            with performancetiming.timeit("field output"):
                                fieldOutputController.finalizeIncrement()
                            for manager in outputManagers:
                                # One row per manager, under the name of its *output line.
                                with performancetiming.timeit(manager.name):
                                    manager.finalizeIncrement(statusInfoDict=solver.incrementStatus)

            except ReachedMaxIncrements:
                pass
            except ReachedMinIncrementSize:
                journal.errorMessage("Incrementation failed", solver.identification)
                raise StepFailed()
            except ConditionalStop:
                journal.message("Conditional Stop", solver.identification)
            finally:
                solver.endStep(self, model)

            solver.applyStepActionsAtStepEnd(model, self.actions)

        finally:
            fieldOutputController.finalizeStep()
            for manager in outputManagers:
                manager.finalizeStep()

    def fallBackToCheckpoint(self, model: FEModel, outputManagers: list, fallbacksLeft: int) -> int:
        """Continue the step from its latest restart checkpoint, after an increment failed and could not
        be cut back any further, and cut the increment back from there.

        The increments accepted after the checkpoint are discarded, and the step resumes from the
        checkpoint exactly as a resumed run does (see :class:`~edelweissfe.utils.checkpoint.ResumeCheckpoint`):
        the model, the time stepper, the solver, the step actions, the output managers and the field
        outputs take over the checkpointed state, so that the output of the discarded increments is
        overwritten, not duplicated. The solver ends the abandoned attempt and begins anew from the
        checkpoint. The retried increments then take a different, smaller path through the same
        load history. If the increment the checkpoint holds cannot be cut back either, the step
        falls back further, to the checkpoint before it.

        A checkpoint written after the last increment the step accepted holds the state the failed
        increment started from, from where every increment size down to the minimum has just
        failed: falling back to it would only retry them. The step goes back to the checkpoint
        before it instead. With a checkpoint after every accepted increment (``writeInterval=1``), the
        latest one always is such a checkpoint.

        The checkpoints are those of the step's restart output, of which there is exactly one.
        Every checkpoint serves one fallback at most: falling back again takes the checkpoint before
        it, unless newer ones were written since. The step may fall back to checkpoints of its own
        only, and only as long as its mesh has not changed since (a mesh change cannot be undone).

        Parameters
        ----------
        model
            The model tree.
        outputManagers
            The output managers of the step.
        fallbacksLeft
            How often the step may still fall back.

        Returns
        -------
        int
            How often the step may fall back after this.

        Raises
        ------
        ReachedMinIncrementSize
            If there is no checkpoint of this step left to fall back to, or no fallback is left.
        StepFailed
            If the mesh has changed since the checkpoint, with a message saying so.
        """

        journal = self.journal
        solver = self.solver
        restartWriter = next(m for m in outputManagers if m.writesRestartCheckpoints)

        while fallbacksLeft:
            checkpoint = self.takeLatestCheckpointOfStep(restartWriter)
            if checkpoint is None:
                journal.errorMessage("No restart checkpoint of this step left to fall back to", solver.identification)
                raise ReachedMinIncrementSize()
            if checkpoint.numberOfIncrementsDone == self.timeStepper.numberOfIncrementsDone():
                journal.message(
                    "Restart checkpoint {:} holds the state the failed increment started from: going back to the one "
                    "before it".format(checkpoint.fileName),
                    solver.identification,
                    1,
                )
                checkpoint.close()
                continue
            try:
                try:
                    checkpoint.restoreModel(model, journal)
                except MeshChangedSinceCheckpoint as e:
                    # The checkpoints are taken newest first: all that are left are older still.
                    raise StepFailed(
                        "cannot fall back to restart checkpoint {:} at time {:}: it predates a mesh change (model "
                        "modifier {:} at time {:}), and a mesh change cannot be undone. No usable checkpoint is "
                        "left: every remaining one predates the mesh change as well".format(
                            checkpoint.fileName, checkpoint.time, e.firstChange.modifier, e.firstChange.time
                        )
                    ) from e
                fallbacksLeft -= 1
                journal.message(
                    "Falling back to restart checkpoint {:} at time {:}".format(checkpoint.fileName, model.time),
                    solver.identification,
                    1,
                )
                solver.endStep(self, model)
                checkpoint.restoreTimeStepper(self)
                solver.beginStep(self, model, self.fieldOutputController, outputManagers)
                checkpoint.restoreStep(self)
            finally:
                checkpoint.close()

            try:
                self.timeStepper.rejectTimeStep(self.cutbackFactor)
                return fallbacksLeft
            except ReachedMinIncrementSize:
                continue

        journal.errorMessage("No checkpoint fallback left", solver.identification)
        raise ReachedMinIncrementSize()

    def takeLatestCheckpointOfStep(self, restartWriter) -> ResumeCheckpoint | None:
        """Open the latest restart checkpoint ``restartWriter`` wrote that no fallback has taken yet, if
        it belongs to this step. The writer's checkpoints are taken newest first, so once one belongs
        to an earlier step, all that remain do.

        Parameters
        ----------
        restartWriter
            The output manager writing the restart checkpoints of this step.

        Returns
        -------
        ResumeCheckpoint | None
            The opened checkpoint, or None if no checkpoint of this step is left.
        """

        fileName = restartWriter.takeLatestCheckpoint()
        if fileName is None:
            return None
        checkpoint = ResumeCheckpoint(fileName)
        if checkpoint.stepNumber != self.number:
            checkpoint.close()
            return None
        return checkpoint
