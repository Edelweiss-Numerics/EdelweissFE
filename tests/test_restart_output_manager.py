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
"""The restart writer's ring buffer continues the checkpoints already on disk instead of starting
at index 0 -- otherwise a walltime-limited job chained across several resumes would overwrite
earlier checkpoints, including, with an unchanged ``baseName``, the very one it resumed from.

Which checkpoint is oldest is read from the serial number every checkpoint carries, not from file
times: copying files changes those, and some file systems resolve them coarsely."""

import os

import h5py

from edelweissfe.outputmanagers.restart import _RestartFileRingBuffer


def _writeCheckpointWithSerial(fileName, serial):
    with h5py.File(fileName, "w") as f:
        f.attrs["serial"] = serial


def _readSerial(fileName):
    with h5py.File(fileName, "r") as f:
        return int(f.attrs["serial"])


def test_fresh_directory_starts_at_index_zero(tmp_path):
    buffer = _RestartFileRingBuffer(str(tmp_path / "restart"), maxsize=3)
    assert buffer.nextCheckpoint() == (str(tmp_path / "restart_0.h5"), 0)


def test_resume_continues_after_a_partially_filled_ring(tmp_path):
    """After 2 of 3 slots were ever written, a new process continues at the first never-written
    slot, with the next serial."""

    baseName = str(tmp_path / "restart")
    _writeCheckpointWithSerial("{:}_0.h5".format(baseName), 0)
    _writeCheckpointWithSerial("{:}_1.h5".format(baseName), 1)

    buffer = _RestartFileRingBuffer(baseName, maxsize=3)
    assert buffer.nextCheckpoint() == ("{:}_2.h5".format(baseName), 2)


def test_resume_continues_the_rotation_once_the_ring_is_full(tmp_path):
    """Once every slot was written, the next write replaces the checkpoint with the lowest serial --
    here not index 0."""

    baseName = str(tmp_path / "restart")
    for serial, index in enumerate((2, 0, 1)):
        _writeCheckpointWithSerial("{:}_{:}.h5".format(baseName, index), serial)

    buffer = _RestartFileRingBuffer(baseName, maxsize=3)
    assert buffer.nextCheckpoint() == ("{:}_2.h5".format(baseName), 3)


def test_crash_resume_chain_never_overwrites_a_checkpoint_before_its_slot_is_due(tmp_path):
    """Several process generations (crash -> resume -> ...), each writing a few checkpoints through
    its own ring buffer: none may overwrite a checkpoint before the ring is full, and once it
    wraps, the oldest goes first."""

    baseName = str(tmp_path / "restart")
    maxsize = 3

    def newProcessGeneration(nWrites):
        buffer = _RestartFileRingBuffer(baseName, maxsize=maxsize)
        for _ in range(nWrites):
            fileName, serial = buffer.nextCheckpoint()
            _writeCheckpointWithSerial(fileName, serial)

    newProcessGeneration(2)
    newProcessGeneration(1)
    liveFiles = [str(tmp_path / "restart_{:}.h5".format(i)) for i in range(maxsize)]
    assert {_readSerial(f) for f in liveFiles} == {0, 1, 2}, "a checkpoint was overwritten too early"

    newProcessGeneration(2)
    assert all(os.path.exists(f) for f in liveFiles)
    assert {_readSerial(f) for f in liveFiles} == {2, 3, 4}, "the rotation did not replace the oldest first"
