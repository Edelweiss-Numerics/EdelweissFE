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
"""Regression test for AdaptiveTimeStepper.writeRestart/readRestart, found via a real end-to-end
AnchorPryOut validation run: writeRestart is called (via the restart output manager's
finalizeIncrement) while generateTimeStep is paused *at* the yield for the increment that just
converged -- before that generator's own post-yield bookkeeping (the growth-factor update and the
incrementCounter/nPassedGoodIncrements advance) has run. An uninterrupted run never notices, since
the same generator applies that bookkeeping itself on its next resume. A *fresh* generator built
for a resumed run has never reached that yield point, so without the fix it repeats the
just-converged increment's size (mislabeled with its own incrementCounter) instead of continuing
the growth sequence -- confirmed on the real case by increment sizes matching the *previous*
increment's, shifted back by one, compounding across every subsequent increment."""

import h5py

from edelweissfe.journal.journal import Journal
from edelweissfe.timesteppers.adaptivetimestepper import AdaptiveTimeStepper


def _makeStepper(journal):
    return AdaptiveTimeStepper(
        currentTime=0.0,
        stepLength=1.0,
        startIncrement=0.01,
        maxIncrement=1.0,
        minIncrement=1e-6,
        maxNumberIncrements=1000,
        journal=journal,
        increaseFactor=1.1,
        makeZeroIncrementFirst=False,
    )


def test_restart_snapshot_matches_the_uninterrupted_generators_next_increment(tmp_path):
    journal = Journal(verbose=False)

    original = _makeStepper(journal)
    gen = original.generateTimeStep()

    # Four increments with no cutbacks/preventIncrementIncrease -- by the time increment 3 (0-based)
    # is yielded, nPassedGoodIncrements has reached 3, priming growth for the *next* one.
    for _ in range(4):
        next(gen)

    assert original.nPassedGoodIncrements == 3
    assert original.incrementCounter == 3

    checkpointPath = tmp_path / "restart.h5"
    with h5py.File(checkpointPath, "w") as f:
        original.writeRestart(f)

    # The uninterrupted generator's own next yield -- the ground truth this checkpoint must match.
    expectedNext = next(gen)

    resumed = _makeStepper(journal)
    with h5py.File(checkpointPath, "r") as f:
        resumed.readRestart(f)
    resumedGen = resumed.generateTimeStep()
    actualNext = next(resumedGen)

    assert actualNext.number == expectedNext.number
    assert actualNext.timeIncrement == expectedNext.timeIncrement
    assert actualNext.totalTime == expectedNext.totalTime


def test_restart_snapshot_does_not_grow_when_growth_conditions_are_not_met(tmp_path):
    """Sanity check the other branch: before nPassedGoodIncrements reaches 3, the checkpoint must
    NOT apply the growth factor -- only advance the counters."""

    journal = Journal(verbose=False)

    original = _makeStepper(journal)
    gen = original.generateTimeStep()
    next(gen)  # one increment yielded; its own post-yield bookkeeping hasn't run yet

    assert original.nPassedGoodIncrements == 0

    checkpointPath = tmp_path / "restart.h5"
    with h5py.File(checkpointPath, "w") as f:
        original.writeRestart(f)

    expectedNext = next(gen)

    resumed = _makeStepper(journal)
    with h5py.File(checkpointPath, "r") as f:
        resumed.readRestart(f)
    actualNext = next(resumed.generateTimeStep())

    assert actualNext.timeIncrement == expectedNext.timeIncrement
    assert actualNext.number == expectedNext.number
