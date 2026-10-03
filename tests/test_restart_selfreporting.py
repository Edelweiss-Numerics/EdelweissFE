"""What a resumed run reports about its own state.

Three defects in one theme, each of which cost real diagnosis time on the anchor pry-out campaign:
a resumed run crashed on the checkpoint it had been resumed from, reported an energy balance
computed from an accumulator that had been reset to zero, and could exit "success" having advanced
nothing at all.

The ring-buffer collision is not unit-testable here -- it needs two runs against the same directory,
and is covered by resuming NEDLiveAMR from a slot the ring is due to overwrite.
"""

import h5py
import pytest

from edelweissfe.journal.journal import Journal
from edelweissfe.solvers.nonlinearexplicitdynamic import NED
from edelweissfe.timesteppers.simpletimestepper import SimpleTimeStepper
from edelweissfe.timesteppers.timestep import TimeStep
from edelweissfe.utils.exceptions import RestartError


class _RecordingJournal:
    """Records what would have been reported, so a diagnostic can be asserted on."""

    def __init__(self):
        self.messages = []

    def message(self, text, identification, level=1):
        self.messages.append(text)

    def errorMessage(self, text, identification):
        self.messages.append(text)


def _solver():
    return NED({}, Journal(verbose=False))


def _stepper(journal, maxNumberIncrements=60000):
    return SimpleTimeStepper(
        currentTime=0.0,
        stepLength=1.0,
        startIncrement=0.1,
        maxIncrement=0.1,
        minIncrement=1e-8,
        maxNumberIncrements=maxNumberIncrements,
        journal=journal,
    )


def test_the_explicit_solver_state_survives_a_checkpoint_exactly(tmp_path):
    """The accumulated external work and the last completed increment are written as they are and
    read back as they were -- the state the resumed step continues from."""

    solver = _solver()
    solver._externalWork = -1234.5
    solver.prevTimeStep = TimeStep(7, 0.125, 0.875, 3.5e-7, 2.45e-6, 2.45e-6)

    checkpoint = tmp_path / "chk.h5"
    with h5py.File(checkpoint, "w") as f:
        solver.writeRestart(f)

    resumed = _solver()
    with h5py.File(checkpoint, "r") as f:
        resumed.readRestart(f)

    assert resumed._externalWork == -1234.5
    restored = resumed.prevTimeStep
    assert (restored.number, restored.stepProgressIncrement, restored.stepProgress) == (7, 0.125, 0.875)
    assert (restored.timeIncrement, restored.stepTime, restored.totalTime) == (3.5e-7, 2.45e-6, 2.45e-6)


def test_a_checkpoint_carrying_no_solver_state_is_refused(tmp_path):
    """Written by an older version, before this state was checkpointed: resuming would continue from
    a state the uninterrupted run never had, so it stops instead."""

    checkpoint = tmp_path / "old.h5"
    with h5py.File(checkpoint, "w") as f:
        f.create_group("timestepper")

    with h5py.File(checkpoint, "r") as f:
        with pytest.raises(RestartError):
            _solver().readRestart(f)


def test_resuming_at_or_past_the_increment_cap_is_reported():
    """The trap: maxNumInc counts from the start of the analysis, so a resume that does not raise
    it far enough ends the step on its first check and the job reports success regardless."""
    for alreadyDone in (60000, 130000):
        journal = _RecordingJournal()
        stepper = _stepper(journal)
        stepper.totalIncrements = alreadyDone
        stepper._warnIfResumedAtIncrementCap()

        assert len(journal.messages) == 1, "no warning at {:} of 60000".format(alreadyDone)
        reported = journal.messages[0]
        assert str(alreadyDone) in reported and "60000" in reported
        assert "without advancing" in reported.lower()


def test_resuming_below_the_increment_cap_is_silent():
    journal = _RecordingJournal()
    stepper = _stepper(journal)
    stepper.totalIncrements = 59999
    stepper._warnIfResumedAtIncrementCap()
    assert journal.messages == []
