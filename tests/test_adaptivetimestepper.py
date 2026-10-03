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
"""The adaptive time stepper as a state machine: a proposal changes nothing, an accepted increment
advances the step, a rejected one is retried smaller from exactly where the step stood."""

import pytest

from edelweissfe.journal.journal import Journal
from edelweissfe.timesteppers.adaptivetimestepper import AdaptiveTimeStepper
from edelweissfe.utils.exceptions import ReachedMaxIncrements, ReachedMinIncrementSize


def _timeStepper(**overrides) -> AdaptiveTimeStepper:
    kwargs = dict(
        currentTime=0.0,
        stepLength=1.0,
        startIncrement=0.1,
        maxIncrement=1.0,
        minIncrement=0.01,
        maxNumberIncrements=100,
        journal=Journal(verbose=False),
    )
    kwargs.update(overrides)
    return AdaptiveTimeStepper(**kwargs)


def test_the_first_increment_is_a_zero_increment():
    timeStepper = _timeStepper()
    first = timeStepper.proposeTimeStep()
    assert (first.number, first.timeIncrement) == (0, 0.0)

    timeStepper.acceptTimeStep(first)
    second = timeStepper.proposeTimeStep()
    assert (second.number, second.timeIncrement) == (1, 0.1)


def test_no_zero_increment_when_disabled():
    first = _timeStepper(makeZeroIncrementFirst=False).proposeTimeStep()
    assert (first.number, first.timeIncrement) == (0, 0.1)


def test_a_proposal_changes_nothing():
    timeStepper = _timeStepper(makeZeroIncrementFirst=False)
    before = dict(vars(timeStepper))
    first, again = timeStepper.proposeTimeStep(), timeStepper.proposeTimeStep()
    assert vars(timeStepper) == before
    assert vars(first) == vars(again)


def test_a_rejected_increment_is_retried_smaller_from_the_same_progress():
    timeStepper = _timeStepper(makeZeroIncrementFirst=False)
    timeStepper.acceptTimeStep(timeStepper.proposeTimeStep())
    progress = timeStepper.finishedStepProgress

    timeStepper.proposeTimeStep()
    timeStepper.rejectTimeStep(0.5)
    retry = timeStepper.proposeTimeStep()

    assert timeStepper.finishedStepProgress == progress
    assert retry.number == 1
    assert retry.stepProgressIncrement == 0.05
    assert retry.stepProgress == progress + 0.05


def test_the_increment_grows_after_three_good_increments_unless_prevented():
    growing, prevented = _timeStepper(makeZeroIncrementFirst=False), _timeStepper(makeZeroIncrementFirst=False)
    for _ in range(4):
        growing.acceptTimeStep(growing.proposeTimeStep())
        prevented.preventIncrementIncrease()
        prevented.acceptTimeStep(prevented.proposeTimeStep())

    assert growing.increment == 0.1 * 1.1
    assert prevented.increment == 0.1


def test_the_step_finishes_exactly_at_its_end():
    timeStepper = _timeStepper(makeZeroIncrementFirst=False, startIncrement=0.3)
    while not timeStepper.isFinished():
        timeStep = timeStepper.proposeTimeStep()
        timeStepper.acceptTimeStep(timeStep)
    assert timeStep.stepProgress == pytest.approx(1.0)


def test_a_failed_zero_increment_cannot_be_cut_back():
    timeStepper = _timeStepper()
    timeStepper.proposeTimeStep()
    with pytest.raises(ReachedMinIncrementSize):
        timeStepper.rejectTimeStep(0.5)


def test_more_increments_than_allowed_are_refused():
    timeStepper = _timeStepper(makeZeroIncrementFirst=False, maxNumberIncrements=2)
    for _ in range(3):
        timeStepper.acceptTimeStep(timeStepper.proposeTimeStep())
    with pytest.raises(ReachedMaxIncrements):
        timeStepper.proposeTimeStep()
