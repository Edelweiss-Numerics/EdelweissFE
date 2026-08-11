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
"""Regression test for ``_RestartFileRingBuffer``'s resume behavior. Before this fix, every new
``_RestartFileRingBuffer`` (i.e. every new process, including a resumed one that keeps writing
checkpoints) started at index 0 -- a walltime-limited job chained across several resume ->
crash -> resume hops would silently overwrite earlier checkpoints, including, with an unchanged
``baseName``, the very one it had just resumed from.

Exercises the real ``h5py`` files on disk (mtime-based tie-breaking once the ring is full), not a
mock, since that's exactly the state the fix reads back."""

import os
import time

import h5py

from edelweissfe.outputmanagers.restart import _RestartFileRingBuffer


def _touch(fileName):
    with h5py.File(fileName, "w"):
        pass


def test_fresh_directory_starts_at_index_zero(tmp_path):
    buffer = _RestartFileRingBuffer(str(tmp_path / "restart"), maxsize=3)
    assert buffer.nextFileName() == str(tmp_path / "restart_0.h5")


def test_resume_continues_after_a_partially_filled_ring(tmp_path):
    """Simulates a crash after 2 of 3 ring slots were ever written: a fresh
    ``_RestartFileRingBuffer`` (new process) must continue at the first never-written slot, not
    restart at 0 and immediately overwrite the checkpoint it (or a sibling reader) just resumed
    from."""

    baseName = str(tmp_path / "restart")
    _touch("{:}_0.h5".format(baseName))
    _touch("{:}_1.h5".format(baseName))

    buffer = _RestartFileRingBuffer(baseName, maxsize=3)
    assert buffer.nextFileName() == "{:}_2.h5".format(baseName)


def test_resume_continues_the_rotation_once_the_ring_is_full(tmp_path):
    """Once all ``maxsize`` slots have been written at least once, the next write must land on
    the *oldest* surviving checkpoint (by mtime) to continue the rotation correctly, not
    unconditionally on index 0."""

    baseName = str(tmp_path / "restart")
    # Write out of index order but in true chronological order, so index 0 is NOT the oldest --
    # a naive "always resume at 0" fix would silently drop the still-relevant checkpoint at index 1.
    for index in (2, 0, 1):
        _touch("{:}_{:}.h5".format(baseName, index))
        time.sleep(0.01)

    buffer = _RestartFileRingBuffer(baseName, maxsize=3)
    assert buffer.nextFileName() == "{:}_2.h5".format(baseName)


def _writeOrdinal(fileName, ordinal):
    with h5py.File(fileName, "w") as f:
        f.attrs["ordinal"] = ordinal


def _readOrdinal(fileName):
    with h5py.File(fileName, "r") as f:
        return int(f.attrs["ordinal"])


def test_crash_resume_chain_never_overwrites_a_checkpoint_before_its_slot_is_due(tmp_path):
    """End-to-end simulation of the actual use case: several process generations (crash -> resume
    -> crash -> resume -> ...), each writing a handful of checkpoints via its own
    ``_RestartFileRingBuffer`` instance, as a resumed run would. As long as the ring hasn't
    actually filled up yet (total writes across the whole chain <= ``numberOfFilesToKeep``), no
    generation boundary may drop back to index 0 and clobber an earlier, still-relevant
    checkpoint -- the bug this guards against. Once the ring genuinely fills and wraps, the
    correct rotation (oldest-first) must still hold across the boundary."""

    baseName = str(tmp_path / "restart")
    maxsize = 3
    ordinal = 0

    def newProcessGeneration(nWrites):
        nonlocal ordinal
        buffer = _RestartFileRingBuffer(baseName, maxsize=maxsize)
        for _ in range(nWrites):
            fileName = buffer.nextFileName()
            _writeOrdinal(fileName, ordinal)
            ordinal += 1
            time.sleep(0.01)

    # Generation A fills 2 of 3 slots; generation B (a fresh instance, as after a crash+resume)
    # writes just 1 more -- together exactly maxsize writes, so nothing should be lost yet.
    newProcessGeneration(2)
    newProcessGeneration(1)

    liveOrdinals = {_readOrdinal(str(tmp_path / "restart_{:}.h5".format(i))) for i in range(maxsize)}
    assert liveOrdinals == {0, 1, 2}, "a checkpoint was overwritten before the ring ever filled up"

    # Generation C pushes past the ring's capacity; now rotation must recycle the oldest first.
    newProcessGeneration(2)

    liveFiles = [str(tmp_path / "restart_{:}.h5".format(i)) for i in range(maxsize)]
    assert all(os.path.exists(f) for f in liveFiles)
    liveOrdinals = {_readOrdinal(f) for f in liveFiles}
    assert liveOrdinals == {2, 3, 4}
