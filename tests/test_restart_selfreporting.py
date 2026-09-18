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


def _stepper(maxNumberIncrements=60000):
    return SimpleTimeStepper(
        currentTime=0.0,
        stepLength=1.0,
        startIncrement=0.1,
        maxIncrement=0.1,
        minIncrement=1e-8,
        maxNumberIncrements=maxNumberIncrements,
        journal=Journal(verbose=False),
    )


def test_the_accumulated_external_work_survives_a_checkpoint(tmp_path):
    solver = _solver()
    solver._externalWork = -1234.5

    checkpoint = tmp_path / "chk.h5"
    with h5py.File(checkpoint, "w") as f:
        solver.writeRestart(f)

    resumed = _solver()
    with h5py.File(checkpoint, "r") as f:
        resumed.readRestart(f)

    # Staged rather than applied: solveStep resets the live accumulator and necessarily runs after
    # readRestart, so a direct assignment here would be wiped before the first increment.
    assert resumed._resumedExternalWork == pytest.approx(-1234.5)
    assert resumed._externalWork == 0.0


def test_a_checkpoint_carrying_no_solver_state_is_tolerated(tmp_path):
    """Written by a different solver, or before this state was carried at all. A resumed run then
    has the old, wrong energy balance -- which is strictly better than refusing to start."""
    checkpoint = tmp_path / "old.h5"
    with h5py.File(checkpoint, "w") as f:
        f.create_group("timestepper")

    solver = _solver()
    with h5py.File(checkpoint, "r") as f:
        solver.readRestart(f)

    assert solver._resumedExternalWork == 0.0


def test_resuming_at_or_past_the_increment_cap_is_reported():
    """The trap: maxNumInc counts from the start of the analysis, so a resume that does not raise
    it far enough ends the step on its first check and the job reports success regardless."""
    for alreadyDone in (60000, 130000):
        journal = _RecordingJournal()
        _stepper().warnIfResumedAtIncrementCap(alreadyDone, 60000, journal)

        assert len(journal.messages) == 1, "no warning at {:} of 60000".format(alreadyDone)
        reported = journal.messages[0]
        assert str(alreadyDone) in reported and "60000" in reported
        assert "without advancing" in reported.lower()


def test_resuming_below_the_increment_cap_is_silent():
    journal = _RecordingJournal()
    _stepper().warnIfResumedAtIncrementCap(59999, 60000, journal)
    assert journal.messages == []


def test_the_resumed_external_work_is_handed_over_exactly_once(tmp_path):
    """Per-step semantics: the resumed step continues the checkpoint's accumulator, and any later
    step in the same job starts from zero rather than inheriting it again."""
    solver = _solver()
    solver._externalWork = 42.0
    checkpoint = tmp_path / "chk.h5"
    with h5py.File(checkpoint, "w") as f:
        solver.writeRestart(f)

    resumed = _solver()
    with h5py.File(checkpoint, "r") as f:
        resumed.readRestart(f)

    assert resumed._consumeResumedExternalWork() == pytest.approx(42.0)
    assert resumed._consumeResumedExternalWork() == 0.0, "a later step must not inherit it again"


def test_a_cold_start_consumes_nothing():
    assert _solver()._consumeResumedExternalWork() == 0.0
