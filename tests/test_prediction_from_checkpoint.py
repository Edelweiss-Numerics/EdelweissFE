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
"""The first increment of a step that continues from a restart checkpoint -- a resumed run, or a step that fell back to
its latest checkpoint (``maxCheckpointFallbacks``) -- extrapolates the checkpoint's last accepted increments only if
its rebuilt discrete system numbers the degrees of freedom as they are numbered. The checkpoint holds the digest of
their numbering with them (``DofManager.dofNumberingDigest``).

In a material point model, the numbering follows the active cells, which may have changed since the checkpoint was
written. Here, the plate's discrete system loses the Lagrange multipliers of a constraint instead: the vectors of the
checkpoint are then longer than those of the rebuilt system, and extrapolating them would end the run with an error.
With as many degrees of freedom, the prediction would be silently mis-numbered.
"""

from pathlib import Path

import h5py
import numpy as np
import pytest

from edelweissfe.drivers.inputfiledrivensimulation import finiteElementSimulation
from edelweissfe.solvers.nonlinearimplicitstatic import NIST
from edelweissfe.steps.base.stepbase import StepBase
from edelweissfe.utils.exceptions import IncrementFailed
from edelweissfe.utils.inputfileparser import parseInputFile

_DECK = """
*material, name=linearelastic, id=linearelastic, provider=edelweiss
210000.0, 0.3

*job, name=job, domain=2d
*solver, solver=NIST, name=theSolver

*modelGenerator, generator=planeRectQuad, name=gen
x0=0, y0=0, l=10, h=10, nX=2, nY=2, elType=CPE4, elProvider=edelweiss

*section, name=section1, thickness=1.0, material=linearelastic, type=plane
gen_all
{constraint}
*output, type=restart, name=restart
writeInterval=1, baseName={directory}/ckpt, numberOfFilesToKeep=100
{resume}
*step, solver=theSolver
stepLength=1, startInc=0.125, maxInc=0.125, minInc=0.03125, cutbackFactor=0.25, maxNumInc=100, maxIter=10, maxCheckpointFallbacks={maxCheckpointFallbacks}
>>options, name=theSolver, extrapolation={extrapolation}
>>dirichlet, name=left, nSet=gen_left, field=displacement, 1=0, 2=0
>>dirichlet, name=right, nSet=gen_right, field=displacement, 1=0.1
"""

#: The top edge keeps one common vertical displacement, enforced by Lagrange multipliers.
_CONSTRAINT = """
*constraint, type=equalValueLagrangian, name=topEqual
nSet=gen_top
field=displacement
component=1
"""

#: The step time of the checkpoint the runs continue from.
CHECKPOINT_TIME = 0.5


def _deck(
    directory: Path,
    withConstraint: bool = True,
    resumeFrom: Path = None,
    maxCheckpointFallbacks: int = 0,
    extrapolation: str = "linear",
) -> str:
    return _DECK.format(
        constraint=_CONSTRAINT if withConstraint else "",
        directory=directory,
        resume="*restart, readFrom={:}".format(resumeFrom) if resumeFrom is not None else "",
        maxCheckpointFallbacks=maxCheckpointFallbacks,
        extrapolation=extrapolation,
    )


def _run(directory: Path, text: str):
    directory.mkdir(exist_ok=True)
    path = directory / "plate.inp"
    path.write_text(text)
    model, _ = finiteElementSimulation(parseInputFile(str(path)), verbose=False, suppressPlots=True)
    return model


def _displacements(model) -> np.ndarray:
    return model.nodeFields["displacement"]["U"].copy()


def _checkpointAt(directory: Path, time: float) -> Path:
    def checkpointTime(path):
        with h5py.File(path, "r") as f:
            return float(f.attrs["time"])

    return next(path for path in directory.glob("ckpt_*.h5") if checkpointTime(path) == pytest.approx(time))


@pytest.fixture
def predictions(monkeypatch) -> list:
    """Records the prediction of every attempt of an increment: (start time of the increment, extrapolated)."""

    record = []
    extrapolateLastIncrement = NIST.extrapolateLastIncrement

    def recordingExtrapolateLastIncrement(self, extrapolation, timeStep, dU, dirichlets, prevTimeStep, model):
        dU, extrapolated = extrapolateLastIncrement(self, extrapolation, timeStep, dU, dirichlets, prevTimeStep, model)
        record.append((timeStep.totalTime - timeStep.timeIncrement, extrapolated))
        return dU, extrapolated

    monkeypatch.setattr(NIST, "extrapolateLastIncrement", recordingExtrapolateLastIncrement)
    return record


@pytest.mark.parametrize("extrapolation", ["linear", "quadratic"])
def test_a_resumed_step_extrapolates_only_increments_numbered_as_its_discrete_system(
    tmp_path, predictions, extrapolation
):
    directory = tmp_path / "uninterrupted"
    uninterrupted = _run(directory, _deck(directory, extrapolation=extrapolation))
    afterTheCheckpoint = [prediction for prediction in predictions if prediction[0] >= CHECKPOINT_TIME]
    checkpoint = _checkpointAt(directory, CHECKPOINT_TIME)
    assert afterTheCheckpoint[0] == (CHECKPOINT_TIME, True), "the deck must extrapolate, or there is nothing to test"

    # Resumed as it was checkpointed: the prediction continues exactly as in the uninterrupted run.
    predictions.clear()
    directory = tmp_path / "resumed"
    resumed = _run(directory, _deck(directory, resumeFrom=checkpoint, extrapolation=extrapolation))
    assert predictions == afterTheCheckpoint
    np.testing.assert_array_equal(_displacements(resumed), _displacements(uninterrupted))

    # Resumed without the constraint: its multipliers leave the discrete system, which numbers the degrees of freedom
    # differently from the checkpoint's last accepted increments. The first increment starts from zero, the following
    # ones extrapolate the increments accepted in the new numbering.
    predictions.clear()
    directory = tmp_path / "renumbered"
    model = _run(directory, _deck(directory, withConstraint=False, resumeFrom=checkpoint, extrapolation=extrapolation))
    assert predictions[0] == (CHECKPOINT_TIME, False)
    assert all(extrapolated for _, extrapolated in predictions[1:])
    assert model.time == pytest.approx(1.0)


@pytest.mark.parametrize("renumbered", [False, True])
def test_a_step_that_fell_back_extrapolates_only_increments_numbered_as_its_discrete_system(
    tmp_path, predictions, monkeypatch, renumbered
):
    """The increment starting at CHECKPOINT_TIME fails at every size until the step has fallen back to a checkpoint.
    Falling back, the constraint is switched off (renumbered), or not: the prediction of the first increment after the
    fallback survives only if the rebuilt discrete system numbers the degrees of freedom as the checkpoint's last
    accepted increments are numbered."""

    fellBack = []
    attemptIncrement = NIST.attemptIncrement

    def failsUntilTheStepFellBack(self, step, model, timeStep):
        if timeStep.totalTime - timeStep.timeIncrement >= CHECKPOINT_TIME - 1e-12 and not fellBack:
            raise IncrementFailed("an increment that cannot be cut back far enough", step.cutbackFactor)
        return attemptIncrement(self, step, model, timeStep)

    fallBackToCheckpoint = StepBase.fallBackToCheckpoint

    def fallBackToCheckpointRenumbered(self, model, outputManagers, fallbacksLeft):
        fellBack.append(len(predictions))
        if renumbered:
            model.constraints["topEqual"].active = False
        return fallBackToCheckpoint(self, model, outputManagers, fallbacksLeft)

    monkeypatch.setattr(NIST, "attemptIncrement", failsUntilTheStepFellBack)
    monkeypatch.setattr(StepBase, "fallBackToCheckpoint", fallBackToCheckpointRenumbered)

    model = _run(tmp_path, _deck(tmp_path, maxCheckpointFallbacks=1))

    assert len(fellBack) == 1
    assert all(extrapolated for _, extrapolated in predictions[2 : fellBack[0]]), "the deck must extrapolate"
    _, extrapolatedAfterTheFallback = predictions[fellBack[0]]
    assert extrapolatedAfterTheFallback == (not renumbered)
    assert model.time == pytest.approx(1.0)
