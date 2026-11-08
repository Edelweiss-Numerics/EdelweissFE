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
"""A step whose increment fails at the minimum increment size falls back to its latest restart
checkpoint (``maxCheckpointFallbacks``) and continues from there with a smaller increment.

The failure is an obstacle put in front of the solver: beyond a point of the load history, an
increment fails as long as the step reached it with fewer than a number of increments -- a load
path that is too coarse fails there, whatever the size of the increment that follows, as in a
softening problem whose state depends on the increments that led to it. Cutting back the failing
increment does not help; going back to a checkpoint and taking a finer path does.
"""

from pathlib import Path

import h5py
import numpy as np
import pytest

# tests/ carries no __init__.py: sibling test modules import by their bare module name.
import test_nid_newmark

from edelweissfe.drivers.inputfiledrivensimulation import finiteElementSimulation
from edelweissfe.journal.journal import Journal
from edelweissfe.solvers.nonlinearimplicitstatic import NIST
from edelweissfe.utils.checkpoint import ResumeCheckpoint
from edelweissfe.utils.exceptions import IncrementFailed, MeshChangedSinceCheckpoint
from edelweissfe.utils.inputfileparser import parseInputFile

_BEAM = """
*material, name=LinearElastic, id=linearelastic
2.0e4, 0.25, 1.0e-3

*modelGenerator, generator=boxGen, name=beam
nX      =6
nY      =2
nZ      =1
lX      =6
lY      =1
lZ      =1
elType  =C3D20

*section, name=beamSection, material=linearelastic, type=solid
beam_all
"""

_AMR = """
*fieldOutput
>>perElement, elSet=beam_all, result=stress, quadraturePoint=0:27, name=stressForAMR, f(x)='np.abs(x)'

*modelModifier, type=hAdaptivity, name=amr
>>marker, type=fieldOutput, fieldOutput=stressForAMR, operator='>', threshold=6.0
refineElSet=beam_all
maxLevel=1
"""

_OUTPUT = """
*fieldOutput
>>perNode, nSet=beam_right, field=displacement, result=U, name=exportU, export={directory}/exportU, f(x)='np.mean(x, axis=0)', saveHistory=True

*output, type=restart, name=restart
writeInterval=1, baseName={directory}/ckpt, numberOfFilesToKeep={numberOfFilesToKeep}
"""

_JOB = """
*job, name=fallback, domain=3d
*solver, solver=NIST, name=theSolver
"""

_STEP = """
*step, solver=theSolver
startInc=0.1, maxInc=0.3, minInc=0.025, cutbackFactor=0.25, increaseFactor=2, goodIncrementsBeforeIncrease=1, maxCheckpointFallbacks={maxCheckpointFallbacks}, stepLength=1
>>dirichlet, name=clamp, nSet=beam_left, field=displacement, 1=0, 2=0, 3=0
>>dirichlet, name=bend, nSet=beam_right, field=displacement, 2=-0.05
"""

#: The increments of the step without the obstacle: the zero increment, then 0.1 (t = 0.1), 0.2 (0.3), 0.3 (0.6),
#: 0.3 (0.9) and 0.1 (1.0). An increment starting at OBSTACLE_TIME = 0.55 or later fails if fewer than
#: INCREMENTS_BEFORE_OBSTACLE increments were accepted, so the increment starting at t = 0.6 fails at every size.
#: The checkpoint at t = 0.6 holds the state that increment started from, so the step goes back to the one at t = 0.3:
#: from there, the cut back increments reach t = 0.375, 0.525 and 0.825, and the step passes the obstacle with 6
#: increments.
OBSTACLE_TIME = 0.55
INCREMENTS_BEFORE_OBSTACLE = 6


def _deck(directory: Path, maxCheckpointFallbacks: int, nSteps: int = 1, writesCheckpoints: bool = True) -> str:
    output = _OUTPUT.format(directory=directory, numberOfFilesToKeep=3)
    if not writesCheckpoints:
        output = output[: output.index("*output, type=restart")]
    return _BEAM + output + _JOB + _STEP.format(maxCheckpointFallbacks=maxCheckpointFallbacks) * nSteps


def _run(directory: Path, text: str):
    path = directory / "fallback.inp"
    path.write_text(text)
    model, _ = finiteElementSimulation(parseInputFile(str(path)), verbose=False, suppressPlots=True)
    return model


@pytest.fixture
def obstacle(monkeypatch):
    """Put the obstacle in front of the implicit solver; ``obstacle(failsAlways, obstacleTime)`` arms it, or
    ``obstacle(failsIf=...)`` arms another one, a predicate of (step, model, start time of the increment). Returns the
    record of the checkpoints fallen back to (their model times)."""

    def arm(failsAlways: bool = False, obstacleTime: float = OBSTACLE_TIME, failsIf=None) -> list:
        attempt = NIST.attemptIncrement

        def failsAtTheObstacle(step, model, startTime):
            return startTime >= obstacleTime - 1e-12 and (
                failsAlways or step.timeStepper.incrementCounter < INCREMENTS_BEFORE_OBSTACLE
            )

        fails = failsIf or failsAtTheObstacle

        def attemptIncrement(self, step, model, timeStep):
            if fails(step, model, timeStep.totalTime - timeStep.timeIncrement):
                raise IncrementFailed("the obstacle", step.cutbackFactor)
            return attempt(self, step, model, timeStep)

        monkeypatch.setattr(NIST, "attemptIncrement", attemptIncrement)

        fallenBackTo = []
        restoreModel = ResumeCheckpoint.restoreModel

        def recordingRestoreModel(self, model, journal):
            restoreModel(self, model, journal)
            fallenBackTo.append(float(model.time))

        monkeypatch.setattr(ResumeCheckpoint, "restoreModel", recordingRestoreModel)
        return fallenBackTo

    return arm


def _exportedTimes(directory: Path) -> np.ndarray:
    return np.loadtxt(directory / "exportU.csv", ndmin=2)[:, 0]


def test_a_step_falls_back_to_an_earlier_checkpoint_and_finishes(tmp_path, obstacle):
    """One fallback suffices: the step does not go back to the checkpoint at t = 0.6, the state the failed increment
    started from, from where every increment size down to the minimum has just failed."""

    reference = _run(tmp_path, _deck(tmp_path, maxCheckpointFallbacks=0))
    referenceU = reference.nodeFields["displacement"]["U"].copy()

    fallbackDirectory = tmp_path / "fallback"
    fallbackDirectory.mkdir()
    fallenBackTo = obstacle()
    model = _run(fallbackDirectory, _deck(fallbackDirectory, maxCheckpointFallbacks=1))

    assert fallenBackTo == pytest.approx([0.3])
    assert model.time == pytest.approx(1.0)
    # linear elastic: the end state does not depend on the load path
    np.testing.assert_allclose(model.nodeFields["displacement"]["U"], referenceU, rtol=1e-10, atol=1e-14)
    # the rows of the discarded increments are replaced, not duplicated: the step start, the zero increment, and
    # the increments of the path taken
    np.testing.assert_allclose(
        _exportedTimes(fallbackDirectory), [0.0, 0.0, 0.1, 0.3, 0.375, 0.525, 0.825, 1.0], atol=1e-12
    )


def test_a_resumed_step_falls_back_to_the_checkpoints_of_the_run_it_resumes(tmp_path, obstacle):
    """A run resumed from the checkpoint at t = 0.6 meets the obstacle at its first increment. The checkpoint it resumed
    from holds the state that increment started from, so the step goes back to the one before it, at t = 0.3, which
    the interrupted run wrote, and passes the obstacle on the finer path."""

    text = (
        _BEAM
        + _OUTPUT.format(directory=tmp_path, numberOfFilesToKeep=10)
        + _JOB
        + _STEP.format(maxCheckpointFallbacks=1)
    )
    interrupted = _run(tmp_path, text)
    referenceU = interrupted.nodeFields["displacement"]["U"].copy()
    checkpoint = next(path for path in tmp_path.glob("ckpt_*.h5") if _modelTime(path) == pytest.approx(0.6))

    fallenBackTo = obstacle()
    resumed = _run(tmp_path, text.replace(_JOB, "\n*restart, readFrom={:}\n".format(checkpoint) + _JOB))

    # the resume itself, then the fallback
    assert fallenBackTo == pytest.approx([0.6, 0.3])
    assert resumed.time == pytest.approx(1.0)
    np.testing.assert_allclose(resumed.nodeFields["displacement"]["U"], referenceU, rtol=1e-10, atol=1e-14)


def test_without_fallbacks_the_step_fails_at_the_minimum_increment(tmp_path, obstacle):
    fallenBackTo = obstacle()
    model = _run(tmp_path, _deck(tmp_path, maxCheckpointFallbacks=0))

    assert fallenBackTo == []
    assert model.time == pytest.approx(0.6)


def test_the_number_of_fallbacks_is_bounded(tmp_path, obstacle):
    """Past the obstacle, nothing helps. The first fallback goes back to t = 0.3, and on the finer path the step
    passes t = 0.55 within one increment, to fail at t = 0.825; the second goes back to t = 0.525, the checkpoint
    before that failed increment's start, and the step fails at t = 0.6, with no fallback left."""

    fallenBackTo = obstacle(failsAlways=True)
    model = _run(tmp_path, _deck(tmp_path, maxCheckpointFallbacks=2))

    assert fallenBackTo == pytest.approx([0.3, 0.525])
    assert model.time == pytest.approx(0.6)


def test_a_step_falls_back_to_checkpoints_of_its_own_only(tmp_path, obstacle):
    """The zero increment of the second step fails: the latest checkpoint is the last one of the first step, which
    the second step cannot go back to."""

    fallenBackTo = obstacle(failsAlways=True, obstacleTime=1.0)
    model = _run(tmp_path, _deck(tmp_path, maxCheckpointFallbacks=3, nSteps=2))

    assert fallenBackTo == []
    assert model.time == pytest.approx(1.0)


def test_a_dynamic_step_that_falls_back_to_the_checkpoint_after_its_zero_increment_computes_its_initial_acceleration(
    tmp_path, obstacle
):
    """A suddenly loaded NID step (the oscillator of ``test_nid_newmark.py``), whose first real increment computes the
    acceleration from equilibrium, a_0 = F / M. Its second real increment fails at every size: the checkpoint after the
    first holds the state that increment started from, so the step goes back to the one after the zero increment, at
    t = 0, where a_0 is still to be computed. From there, the step cut back is the step of an uninterrupted run that
    starts with the cut back increment: a_0 first, then the same increments, to the same end state. Starting from the
    acceleration the checkpoint holds, zero, it would lose dt / 2 a_0 of the velocity for good."""

    dt = 0.02
    increment = dt / test_nid_newmark.STEP_LENGTH

    def deck(startInc: float, maxCheckpointFallbacks: int, extra: str = "") -> str:
        text = test_nid_newmark._deck(dt, extra=extra)
        incrementation = (
            "minInc={!r}, cutbackFactor=0.5, increaseFactor=2, goodIncrementsBeforeIncrease=1, "
            "maxCheckpointFallbacks={:}".format(increment / 4, maxCheckpointFallbacks)
        )
        for old, new in [
            ("startInc={!r}".format(increment), "startInc={!r}".format(startInc)),
            ("minInc=1e-6", incrementation),
        ]:
            assert text.count(old) == 1
            text = text.replace(old, new)
        return text

    # Cut back by 1/2 from the restored increment dt, the step continues with dt / 2, then dt.
    reference, _ = test_nid_newmark._run(tmp_path, "reference", deck(startInc=increment / 2, maxCheckpointFallbacks=0))

    # The increment starting at t = dt fails while the step has accepted fewer than 3 increments (the zero increment
    # included): every size from dt down to the minimum dt / 4 fails after the first real increment, and the path from
    # t = 0 with dt / 2 passes it.
    fallenBackTo = obstacle(
        failsIf=lambda step, model, startTime: startTime >= dt - 1e-12 and step.timeStepper.incrementCounter < 3
    )
    writer = "\n*output, type=restart, name=restart\nwriteInterval=1, baseName={:}, numberOfFilesToKeep=3\n".format(
        tmp_path / "ckpt"
    )
    model, _ = test_nid_newmark._run(
        tmp_path, "fallback", deck(startInc=increment, maxCheckpointFallbacks=1, extra=writer)
    )

    assert fallenBackTo == [0.0]
    assert model.time == pytest.approx(reference.time, abs=1e-12)
    for entry, atol in [("U", 1e-12), ("V", 1e-11), ("A", 1e-10)]:
        np.testing.assert_allclose(
            model.nodeFields["displacement"][entry], reference.nodeFields["displacement"][entry], rtol=0.0, atol=atol
        )


def test_falling_back_needs_restart_checkpoints(tmp_path):
    with pytest.raises(ValueError, match="needs exactly one restart output.*found 0"):
        _run(tmp_path, _deck(tmp_path, maxCheckpointFallbacks=1, writesCheckpoints=False))


def test_falling_back_refuses_several_restart_outputs(tmp_path):
    secondWriter = "\n*output, type=restart, name=restart2\nwriteInterval=2, baseName={:}/other\n".format(tmp_path)
    text = _deck(tmp_path, maxCheckpointFallbacks=1).replace(_JOB, secondWriter + _JOB)
    with pytest.raises(ValueError, match="needs exactly one restart output.*found 2"):
        _run(tmp_path, text)


def test_a_step_whose_mesh_changed_since_its_checkpoints_fails_and_says_why(tmp_path, obstacle, monkeypatch):
    """The increment right after a refinement fails at every size: every checkpoint left predates the refinement."""

    errors = []
    errorMessage = Journal.errorMessage
    monkeypatch.setattr(Journal, "errorMessage", lambda self, m, s: (errors.append(m), errorMessage(self, m, s)))

    def failsAfterARefinement(step, model, startTime):
        history = model.topology.history
        return bool(history) and history[-1].time >= startTime - 1e-12

    fallenBackTo = obstacle(failsIf=failsAfterARefinement)
    text = (
        _BEAM
        + _AMR
        + _OUTPUT.format(directory=tmp_path, numberOfFilesToKeep=3)
        + _JOB
        + _STEP.format(maxCheckpointFallbacks=3)
    )
    model = _run(tmp_path, text)

    assert model.topology.history, "AMR must refine, or there is nothing to test"
    assert fallenBackTo == []
    failure = next(m for m in errors if "cannot fall back" in m)
    assert "predates a mesh change (model modifier amr at time {:}".format(model.topology.history[-1].time) in failure
    assert "a mesh change cannot be undone" in failure
    assert "No usable checkpoint is left" in failure


def _modelTime(checkpoint: Path) -> float:
    with h5py.File(checkpoint, "r") as f:
        return float(f.attrs["time"])


def test_a_refined_mesh_cannot_go_back_to_a_checkpoint_before_the_refinement(tmp_path):
    text = (
        _BEAM
        + _AMR
        + _OUTPUT.format(directory=tmp_path, numberOfFilesToKeep=1000)
        + _JOB
        + _STEP.format(maxCheckpointFallbacks=0)
    )
    model = _run(tmp_path, text)
    assert model.topology.history, "AMR must refine, or there is nothing to test"
    checkpoints = sorted(tmp_path.glob("ckpt_*.h5"), key=lambda p: int(p.stem.split("_")[-1]))
    nElements = len(model.elements)

    beforeRefinement = ResumeCheckpoint(str(checkpoints[0]))
    with pytest.raises(MeshChangedSinceCheckpoint, match="cannot be undone"):
        beforeRefinement.restoreModel(model, None)
    beforeRefinement.close()
    assert model.time == pytest.approx(1.0), "a refused checkpoint must leave the model as it is"

    # the latest checkpoint has the model's mesh: nothing is replayed, the state is restored
    latest = ResumeCheckpoint(str(checkpoints[-1]))
    latest.restoreModel(model, None)
    latest.close()
    assert len(model.elements) == nElements
    assert model.time == pytest.approx(_modelTime(checkpoints[-1]))
