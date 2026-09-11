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
"""Pins ``AdaptiveTimeStepper.doesZeroIncrement()`` to what will actually happen next, not merely
whether zero increments are configured -- it used to return a hardcoded ``True``, so its one
intended use case (a caller that needs to know whether the *next* generated step has zero length,
e.g. to skip initialisation logic that only applies to a genuine zero increment) could never
distinguish "still at the zero increment" from "already past it", most obviously after a restart,
which resumes with a non-zero ``incrementCounter``.
"""

from unittest.mock import MagicMock

from edelweissfe.timesteppers.adaptivetimestepper import AdaptiveTimeStepper


def _timeStepper(**overrides) -> AdaptiveTimeStepper:
    kwargs = dict(
        currentTime=0.0,
        stepLength=1.0,
        startIncrement=0.1,
        maxIncrement=1.0,
        minIncrement=0.01,
        maxNumberIncrements=100,
        journal=MagicMock(),
    )
    kwargs.update(overrides)
    return AdaptiveTimeStepper(**kwargs)


def test_reports_zero_increment_only_before_the_first_increment_is_generated():
    timeStepper = _timeStepper()
    assert timeStepper.doesZeroIncrement()

    timeStepper.incrementCounter = 1
    assert not timeStepper.doesZeroIncrement()


def test_reports_no_zero_increment_when_disabled():
    timeStepper = _timeStepper(makeZeroIncrementFirst=False)
    assert not timeStepper.doesZeroIncrement()


def test_reports_no_zero_increment_after_restart_resumes_a_non_zero_increment_counter():
    """The motivating case: a restarted run's timestepper never sees ``incrementCounter == 0``, so
    it must not claim the next generated step will be a zero increment."""
    timeStepper = _timeStepper()
    timeStepper.incrementCounter = 42
    assert not timeStepper.doesZeroIncrement()
