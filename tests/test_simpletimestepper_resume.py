"""The zero increment a step is primed with is done once per step -- a resumed run does not repeat it.

``SimpleTimeStepper`` proposes one zero increment before the first real one, so an explicit
integrator can build its initial state before taking a step. Whether it was done is part of the
stepper's checkpointed state. A resumed run that repeated it did three wrong things: the explicit
solver writes an output (and restart checkpoint) whenever ``timeStep.number`` is a multiple of
``output-frequency``, which 0 always is, so a resumed run wrote one more than the uninterrupted one;
before that was fixed, the increment also reported the step's start and rewound model time,
leaving the Ensight time set non-monotonic (70 of 257 frames of a restarted anchor pry-out run); and
accepting it without evaluating anything committed the contact's freshly constructed trial state
over the restored frictional history.
"""

import h5py
import pytest

from edelweissfe.journal.journal import Journal
from edelweissfe.timesteppers.simpletimestepper import SimpleTimeStepper

STEP_START = 2.0
STEP_LENGTH = 4.0


def _stepper():
    return SimpleTimeStepper(
        currentTime=STEP_START,
        stepLength=STEP_LENGTH,
        startIncrement=0.25,
        maxIncrement=0.25,
        minIncrement=1e-8,
        maxNumberIncrements=100,
        journal=Journal(verbose=False),
    )


def test_a_cold_step_starts_with_a_zero_increment_at_the_step_start():
    first = _stepper().proposeTimeStep()

    assert (first.number, first.timeIncrement, first.stepProgressIncrement) == (0, 0.0, 0.0)
    assert first.stepProgress == 0.0
    assert first.totalTime == STEP_START


def test_the_increments_after_the_zero_increment_are_numbered_from_one():
    stepper = _stepper()
    stepper.acceptTimeStep(stepper.proposeTimeStep())
    second = stepper.proposeTimeStep()

    assert (second.number, second.timeIncrement) == (1, 0.25 * STEP_LENGTH)


def test_a_resumed_step_does_not_repeat_the_zero_increment(tmp_path):
    original = _stepper()
    for _ in range(3):
        original.acceptTimeStep(original.proposeTimeStep())

    with h5py.File(tmp_path / "chk.h5", "w") as f:
        original.writeRestart(f)
    resumed = _stepper()
    with h5py.File(tmp_path / "chk.h5", "r") as f:
        resumed.readRestart(f)

    first = resumed.proposeTimeStep()
    assert first.number == 3
    assert first.timeIncrement > 0.0
    assert vars(first) == vars(original.proposeTimeStep())


def test_time_never_runs_backwards():
    stepper = _stepper()
    times = []
    while not stepper.isFinished():
        timeStep = stepper.proposeTimeStep()
        times.append(timeStep.totalTime)
        stepper.acceptTimeStep(timeStep)
    assert times == sorted(times)
    assert times[-1] == pytest.approx(STEP_START + STEP_LENGTH)
