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
"""Restart equivalence of the Newmark-beta implicit dynamic solver (``NID``).

The same two-invocation shape as ``tests/test_restart_integration.py``: one uninterrupted
reference run, one run truncated mid-step that writes checkpoints, one run resumed from the last
checkpoint -- and the resumed run must land on the reference's final state.

What is specific to a dynamic solver, and why this test exists rather than trusting the generic
node-field checkpoint: the state of a Newmark increment is ``(U, V, A)``, not ``U`` alone. The
velocity and the acceleration are node-field entries the solver publishes after every converged
increment, so they SHOULD travel with the checkpoint through the very same mechanism as ``U`` --
but "should" is exactly what a restart test is for. Both are compared here, not only ``U``: a
resumed run that restored ``U`` and silently started ``V``/``A`` from zero would still produce a
plausible ``U`` curve. Also pinned: a resumed step must NOT recompute its initial acceleration
from equilibrium (a cold step does), because that would replace the checkpointed, scheme-consistent
acceleration by one that agrees only to solver tolerance.
"""

from pathlib import Path

import h5py
import numpy as np

# tests/ carries no __init__.py, so pytest's default (prepend) import mode puts this directory on
# sys.path and sibling test modules import by their bare module name.
from test_nid_newmark import _deck, _run

DT = 0.02
N_INCREMENTS_FULL = 100  # two periods
N_INCREMENTS_TRUNCATED = 37  # mid-way through the first period's second half


def _mostRecentCheckpoint(directory: Path) -> Path:
    checkpoints = list(directory.glob("ckpt_*.h5"))
    assert checkpoints, "the truncated run did not write any restart checkpoint"

    def checkpointTime(path):
        with h5py.File(path, "r") as f:
            return f.attrs["time"]

    return max(checkpoints, key=checkpointTime)


def test_restart_resume_matches_uninterrupted_reference(tmp_path):
    referenceModel, _ = _run(tmp_path, "full", _deck(DT, maxNumInc=N_INCREMENTS_FULL))
    referenceField = referenceModel.nodeFields["displacement"]
    referenceU = referenceField["U"].copy()
    referenceV = referenceField["V"].copy()
    referenceA = referenceField["A"].copy()
    assert referenceModel.time > 1.99

    # the run must actually move: a test on a model at rest passes trivially
    assert np.max(np.abs(referenceV)) > 1e-3
    assert np.max(np.abs(referenceA)) > 1e-2

    # Absolute checkpoint base name: the output manager and the driver's *restart, readFrom= both
    # resolve the file name relative to the process's working directory, not to tmp_path.
    checkpointBaseName = tmp_path / "ckpt"
    restartWriter = (
        "\n*output, type=restart, name=restartwriter\n"
        f"writeInterval=1, baseName={checkpointBaseName}, numberOfFilesToKeep=3\n"
    )

    truncatedModel, _ = _run(tmp_path, "truncated", _deck(DT, maxNumInc=N_INCREMENTS_TRUNCATED, extra=restartWriter))
    assert truncatedModel.time < 1.0, "the step already finished within maxNumInc -- not a real truncation"

    checkpoint = _mostRecentCheckpoint(tmp_path)

    with h5py.File(checkpoint, "r") as f:
        stored = f["nodeFields"]["displacement"]
        assert "V" in stored and "A" in stored, "the checkpoint does not carry the Newmark kinematics"
        assert f["solver"].attrs["newmarkKinematicsCheckpointed"]
        checkpointTime = float(f.attrs["time"])
        # the checkpoint holds the truncated run's own final state, which is what gets resumed
        np.testing.assert_allclose(stored["V"][:], truncatedModel.nodeFields["displacement"]["V"], atol=0.0)
        np.testing.assert_allclose(stored["A"][:], truncatedModel.nodeFields["displacement"]["A"], atol=0.0)
    assert abs(checkpointTime - N_INCREMENTS_TRUNCATED * DT) < 1e-10

    resumedModel, _ = _run(
        tmp_path, "resume", _deck(DT, maxNumInc=N_INCREMENTS_FULL, extra=f"\n*restart, readFrom={checkpoint}\n")
    )
    assert abs(resumedModel.time - referenceModel.time) < 1e-10

    resumedField = resumedModel.nodeFields["displacement"]
    np.testing.assert_allclose(resumedField["U"], referenceU, rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(resumedField["V"], referenceV, rtol=0.0, atol=1e-11)
    np.testing.assert_allclose(resumedField["A"], referenceA, rtol=0.0, atol=1e-10)
