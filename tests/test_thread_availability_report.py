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

"""A parallel solver must say when the process is pinned to fewer CPUs than it has threads.

Nothing fails when that happens -- the pool is created, the workers run, the answer is right, and
the run is merely several times slower than it reports itself to be -- so the only thing standing
between a pinned run and a week of wondering why the machine is idle is this message.
"""

import os

from edelweissfe.numerics.parallelizationutilities import (
    getNumberOfAvailableCpus,
    reportThreadAvailability,
)


class _RecordingJournal:
    """The face of the Journal this reporting uses: one method that takes a line and a sender."""

    def __init__(self):
        self.messages = []

    def message(self, message: str, senderIdentification: str, level: int = 1):
        self.messages.append(message)


def _pinTo(monkeypatch, nCpus: int):
    """Make the process look as though it were allowed on ``nCpus`` CPUs."""

    monkeypatch.setattr(os, "sched_getaffinity", lambda pid: set(range(nCpus)))


def test_the_thread_count_is_always_reported(monkeypatch):
    _pinTo(monkeypatch, 8)
    journal = _RecordingJournal()

    reportThreadAvailability(8, journal, "theSolver")

    assert journal.messages[0] == "Using 8 threads"


def test_a_pinned_process_is_warned_about(monkeypatch):
    """Fewer CPUs than threads: the run would silently serialise, so it must be named."""

    _pinTo(monkeypatch, 1)
    journal = _RecordingJournal()

    reportThreadAvailability(8, journal, "theSolver")

    warnings = [m for m in journal.messages if m.startswith("WARNING")]
    assert len(warnings) == 1
    # the numbers, so the reader can see how bad it is, and the cause worth checking first
    assert "1 CPUs" in warnings[0]
    assert "8 threads" in warnings[0]
    assert "OMP_PROC_BIND" in warnings[0]
    assert "OMP_PLACES" in warnings[0]


def test_an_unpinned_process_is_not_warned_about(monkeypatch):
    """As many CPUs as threads is the healthy case and must stay quiet."""

    _pinTo(monkeypatch, 8)
    journal = _RecordingJournal()

    reportThreadAvailability(8, journal, "theSolver")

    assert not [m for m in journal.messages if m.startswith("WARNING")]


def test_more_cpus_than_threads_is_not_warned_about(monkeypatch):
    """Asking for fewer threads than the machine allows is a deliberate, ordinary choice."""

    _pinTo(monkeypatch, 64)
    journal = _RecordingJournal()

    reportThreadAvailability(4, journal, "theSolver")

    assert not [m for m in journal.messages if m.startswith("WARNING")]


def test_a_platform_that_cannot_report_its_affinity_stays_quiet(monkeypatch):
    """Without sched_getaffinity there is nothing to compare, and a guess would be worse than
    silence."""

    monkeypatch.delattr(os, "sched_getaffinity", raising=False)
    journal = _RecordingJournal()

    assert getNumberOfAvailableCpus() == 0

    reportThreadAvailability(8, journal, "theSolver")

    assert journal.messages == ["Using 8 threads"]


def test_the_available_cpu_count_is_read_from_the_affinity_mask(monkeypatch):
    """Not from the machine's core count: a cgroup or a taskset is exactly what must be seen."""

    _pinTo(monkeypatch, 3)

    assert getNumberOfAvailableCpus() == 3
