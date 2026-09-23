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
#  ---------------------------------------------------------------------
"""The topology-change conservation checks both dynamic solvers share
(:mod:`edelweissfe.solvers.base.topologychangeconservation`)."""

from types import SimpleNamespace

import numpy as np
import pytest

from edelweissfe.solvers.base.topologychangeconservation import (
    CUMULATIVE_DRIFT_TOLERANCE,
    ConservationLedger,
    linearMomentum,
)


class _RecordingJournal:
    def __init__(self):
        self.messages = []

    def message(self, text, *args, **kwargs):
        self.messages.append(text)


def _fixture(dimensions: dict):
    """A dof manager and a model with one contiguous, node-major slice per field."""

    idcs, start = {}, 0
    for name, (dimension, nNodes) in dimensions.items():
        idcs[name] = slice(start, start + dimension * nNodes)
        start += dimension * nNodes
    dofManager = SimpleNamespace(idcsOfFieldsInDofVector=idcs)
    model = SimpleNamespace(nodeFields={name: SimpleNamespace(dimension=d) for name, (d, _) in dimensions.items()})
    return dofManager, model, start


def test_momentum_is_summed_per_component_and_across_fields():
    dofManager, model, nDof = _fixture({"a": (3, 2), "b": (3, 1)})
    massTimesVelocity = np.arange(1.0, nDof + 1.0)  # node-major: a0=(1,2,3), a1=(4,5,6), b0=(7,8,9)

    momentum = linearMomentum(massTimesVelocity, dofManager, ["a", "b"], model)

    np.testing.assert_array_equal(momentum, [1 + 4 + 7, 2 + 5 + 8, 3 + 6 + 9])
    assert linearMomentum(massTimesVelocity, dofManager, [], model).size == 0


def test_momenta_of_different_dimension_are_not_added():
    dofManager, model, nDof = _fixture({"a": (3, 1), "b": (1, 3)})
    with pytest.raises(ValueError, match="no common components"):
        linearMomentum(np.ones(nDof), dofManager, ["a", "b"], model)


def test_a_violated_total_raises():
    ledger = ConservationLedger(_RecordingJournal(), "test")
    assert ledger.check("mass", 1.0, 1.0 + 1e-9, 1e-6) == pytest.approx(1e-9)
    with pytest.raises(RuntimeError, match="did not conserve the total mass"):
        ledger.check("mass", 1.0, 1.1, 1e-6)


def test_accumulated_drift_is_warned_about_once_per_step_and_reset():
    journal = _RecordingJournal()
    ledger = ConservationLedger(journal, "test")
    perChange = 0.4 * CUMULATIVE_DRIFT_TOLERANCE  # each within a per-change tolerance of 1e-3

    for _ in range(5):
        ledger.check("mass", 1.0, 1.0 + perChange, 1e-3)
    assert len(journal.messages) == 1 and "accumulated relative drift" in journal.messages[0]

    ledger.reset()
    ledger.check("mass", 1.0, 1.0 + perChange, 1e-3)
    assert len(journal.messages) == 1, "the drift of a previous step was carried into the next"
