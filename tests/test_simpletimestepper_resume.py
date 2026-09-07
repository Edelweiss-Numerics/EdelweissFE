"""The zero increment a step is primed with, and what it reports on a resumed run.

``SimpleTimeStepper`` yields one zero increment before the first real one, so an explicit
integrator can build its initial state before taking a step. That increment has to report where
the stepper actually *is*, which on a cold start is the beginning of the step and on a resumed run
is not.

Reporting the beginning of the step unconditionally had two consequences, because the explicit
solver calls ``model.advanceToTime(timeStep.totalTime)`` on every increment and writes an output
whenever ``timeStep.number`` is a multiple of ``output-frequency`` -- which 0 always is. Model time
was rewound to the start of the step for one increment, and an output frame stamped with that
rewound time was written into the middle of an otherwise increasing Ensight time set, leaving it
non-monotonic. That is invalid in the EnSight Gold format, and it makes a reader pair variable
frames with the wrong geometry: measured on a restarted anchor pry-out run, 70 of 257 frames.
"""

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


def test_the_priming_increment_of_a_cold_start_sits_at_the_step_start():
    first = next(_stepper().generateTimeStep())

    assert first.number == 0
    assert first.timeIncrement == 0.0
    assert first.stepProgressIncrement == 0.0
    assert first.stepProgress == pytest.approx(0.0)
    assert first.stepTime == pytest.approx(0.0)
    assert first.totalTime == pytest.approx(STEP_START)


def test_the_priming_increment_of_a_resumed_run_sits_where_the_checkpoint_left_off():
    stepper = _stepper()
    # what readRestart restores: half of this step was already solved before the checkpoint
    stepper.finishedStepProgress = 0.5
    stepper.totalIncrements = 37

    first = next(stepper.generateTimeStep())

    assert first.number == 0
    assert first.timeIncrement == 0.0, "still a zero increment -- it must not advance the solution"
    assert first.stepProgress == pytest.approx(0.5)
    assert first.stepTime == pytest.approx(0.5 * STEP_LENGTH)
    assert first.totalTime == pytest.approx(STEP_START + 0.5 * STEP_LENGTH), (
        "the priming increment reported the step's start, so the solver rewound model.time and "
        "stamped an output frame with it"
    )


def test_the_priming_increment_never_runs_time_backwards():
    """The property that actually matters downstream: time never decreases across the sequence."""
    for progress in (0.0, 0.25, 0.5, 0.75):
        stepper = _stepper()
        stepper.finishedStepProgress = progress
        times = [step.totalTime for step in stepper.generateTimeStep()]
        assert times == sorted(times), "time decreased at progress {:}: {:}".format(progress, times)
        assert times[0] == pytest.approx(STEP_START + progress * STEP_LENGTH)
