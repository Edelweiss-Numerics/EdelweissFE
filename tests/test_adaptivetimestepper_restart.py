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
"""A time stepper's checkpoint is its attributes after the last accepted increment: a stepper
restored from it proposes exactly the increments the uninterrupted one proposes -- whatever happened
before (growth, a cutback, a prevented increase)."""

import h5py
import pytest

from edelweissfe.journal.journal import Journal
from edelweissfe.timesteppers.adaptivetimestepper import AdaptiveTimeStepper
from edelweissfe.timesteppers.simpletimestepper import SimpleTimeStepper
from edelweissfe.utils.checkpoint import readRestartDataInto, writeRestartDataOf


def _adaptive():
    return AdaptiveTimeStepper(0.0, 1.0, 0.01, 1.0, 1e-6, 1000, Journal(verbose=False), increaseFactor=1.1)


def _simple():
    return SimpleTimeStepper(2.0, 1.0, 0.1, 0.1, 1e-6, 1000, Journal(verbose=False))


def _accept(stepper, n):
    for _ in range(n):
        stepper.acceptTimeStep(stepper.proposeTimeStep())


def _cutback(stepper):
    stepper.proposeTimeStep()
    stepper.rejectTimeStep(0.25)
    _accept(stepper, 1)


def _prevent(stepper):
    stepper.preventIncrementIncrease()
    _accept(stepper, 1)


def _enforce(stepper):
    stepper.enforceTimeIncrement(0.03)
    _accept(stepper, 1)


def _resumed(stepper, makeStepper, tmp_path):
    with h5py.File(tmp_path / "chk.h5", "w") as f:
        writeRestartDataOf(f, {"timestepper": stepper})
    resumed = makeStepper()
    with h5py.File(tmp_path / "chk.h5", "r") as f:
        readRestartDataInto(f, {"timestepper": resumed})
    return resumed


@pytest.mark.parametrize(
    "makeStepper, history",
    [
        (_adaptive, [lambda s: _accept(s, 1)]),
        (_adaptive, [lambda s: _accept(s, 5)]),
        (_adaptive, [lambda s: _accept(s, 4), _cutback]),
        (_adaptive, [lambda s: _accept(s, 4), _prevent]),
        (_simple, [lambda s: _accept(s, 1)]),
        (_simple, [lambda s: _accept(s, 3)]),
        (_simple, [lambda s: _accept(s, 3), _cutback]),
        (_simple, [lambda s: _accept(s, 2), _enforce]),
    ],
)
def test_a_restored_stepper_continues_exactly(makeStepper, history, tmp_path):
    original = makeStepper()
    for event in history:
        event(original)

    resumed = _resumed(original, makeStepper, tmp_path)

    for _ in range(20):
        if original.isFinished():
            break
        expected, actual = original.proposeTimeStep(), resumed.proposeTimeStep()
        assert vars(actual) == vars(expected)
        original.acceptTimeStep(expected)
        resumed.acceptTimeStep(actual)
    assert resumed.isFinished() == original.isFinished()
