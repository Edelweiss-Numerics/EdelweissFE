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
"""Regression tests for the ``computetimemonitor`` export. Before this fix, the monitor printed each
increment's times but never wrote them to the ``export`` file -- and a naive fix calling
``extractIncrementTimes`` once per consumer would have written zeros, since extracting advances the
snapshot the next delta is measured against. The ``ComputeTimeMonitor`` deck only compares ``U``, so
the file contents are checked here."""

import time

import pytest
from prettytable import PrettyTable

from edelweissfe.outputmanagers.computetimemonitor import (
    ComputeTimeMonitorSchema,
    OutputManager,
)
from edelweissfe.utils import performancetiming
from edelweissfe.utils.performancetiming import (
    extractIncrementTimeRows,
    extractIncrementTimes,
    timeit,
)


class _RecordingJournal:
    def __init__(self):
        self.tables = []

    def printPrettyTable(self, table, identification):
        self.tables.append(table)


@pytest.fixture(autouse=True)
def _resetTimers():
    performancetiming.reset()
    yield
    performancetiming.reset()


def _spendTime():
    with timeit("outer"):
        with timeit("inner"):
            time.sleep(1e-3)
    with timeit("other"):
        time.sleep(1e-3)


def _makeMonitor(exportFile=None):
    journal = _RecordingJournal()
    monitor = OutputManager(
        "timer", None, None, journal, None, configuration=ComputeTimeMonitorSchema(export=exportFile)
    )
    return monitor, journal


def _dataRows(exportFile):
    with open(exportFile) as f:
        return [line.split() for line in f if not line.startswith("#")]


def _parse(row):
    # "inner" and friends contain no whitespace, so the columns split cleanly
    increment, simulationTime, incComputeTime, level, function, t, calls = row
    return increment, simulationTime, float(incComputeTime), int(level), function, float(t), int(calls)


def test_export_writes_the_printed_increment_times(tmp_path):
    exportFile = str(tmp_path / "compTime.txt")
    monitor, journal = _makeMonitor(exportFile)
    monitor.initializeJob()
    monitor.initializeStep(None)

    _spendTime()
    monitor.finalizeIncrement(statusInfoDict={"inc": 3, "time end": 0.25})

    with open(exportFile) as f:
        content = f.read()
    assert "# simulation step 1" in content
    assert "inc compute time" in content

    rows = [_parse(r) for r in _dataRows(exportFile)]
    assert [(level, function, calls) for _, _, _, level, function, _, calls in rows] == [
        (0, "outer", 1),
        (1, "inner", 1),
        (0, "other", 1),
    ]
    for increment, simulationTime, _, _, _, t, _ in rows:
        assert increment == "3"
        assert float(simulationTime) == pytest.approx(0.25)
        # not zeros: the rows were extracted once and shared between journal and file
        assert t > 0.0

    times = {function: t for _, _, _, _, function, t, _ in rows}
    for _, _, incComputeTime, *_ in rows:
        assert incComputeTime == pytest.approx(times["outer"] + times["other"], rel=1e-4)

    assert len(journal.tables) == 1
    assert "outer" in journal.tables[0].get_string()

    monitor.finalizeStep()
    monitor.finalizeJob()


def test_failed_increment_is_exported_too(tmp_path):
    exportFile = str(tmp_path / "compTime.txt")
    monitor, journal = _makeMonitor(exportFile)
    monitor.initializeStep(None)

    _spendTime()
    monitor.finalizeFailedIncrement(statusInfoDict={"inc": 1, "time end": 0.5})

    rows = _dataRows(exportFile)
    assert len(rows) == 3
    assert all(r[0] == "1" for r in rows)
    assert len(journal.tables) == 1


def test_missing_status_info_is_written_as_dashes(tmp_path):
    """The explicit dynamic solver passes ``statusInfoDict=None``."""
    exportFile = str(tmp_path / "compTime.txt")
    monitor, _ = _makeMonitor(exportFile)
    monitor.initializeStep(None)

    _spendTime()
    monitor.finalizeIncrement(statusInfoDict=None)

    rows = _dataRows(exportFile)
    assert len(rows) == 3
    assert all(r[0] == "-" and r[1] == "-" for r in rows)


def test_each_step_gets_its_own_header(tmp_path):
    exportFile = str(tmp_path / "compTime.txt")
    monitor, _ = _makeMonitor(exportFile)
    for _ in range(2):
        monitor.initializeStep(None)
        _spendTime()
        monitor.finalizeIncrement(statusInfoDict={"inc": 1, "time end": 1.0})

    with open(exportFile) as f:
        content = f.read()
    assert "# simulation step 1" in content
    assert "# simulation step 2" in content
    assert len(_dataRows(exportFile)) == 6


def test_without_export_nothing_is_written(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monitor, journal = _makeMonitor()
    monitor.initializeStep(None)

    _spendTime()
    monitor.finalizeIncrement(statusInfoDict={"inc": 1, "time end": 1.0})

    assert len(journal.tables) == 1
    assert list(tmp_path.iterdir()) == []


def test_extracting_is_destructive():
    _spendTime()
    first = extractIncrementTimeRows()
    second = extractIncrementTimeRows()

    assert all(t > 0.0 and calls == 1 for _, _, t, calls in first)
    assert [(level, function) for level, function, _, _ in second] == [
        (level, function) for level, function, _, _ in first
    ]
    assert all(t == 0.0 and calls == 0 for _, _, t, calls in second)


def test_extractIncrementTimes_pretty_table_and_skipUnused():
    _spendTime()
    table = extractIncrementTimes()
    assert isinstance(table, PrettyTable)
    assert len(table.rows) == 3
    assert "outer" in table.get_string()

    # only "other" runs in this interval: the untouched "outer" subtree is dropped
    with timeit("other"):
        pass
    table = extractIncrementTimes(skipUnused=True)
    assert [row[0].strip() for row in table.rows] == ["other"]


def test_maxLevels_limits_the_depth():
    _spendTime()
    rows = extractIncrementTimeRows(maxLevels=0)
    assert [function for _, function, _, _ in rows] == ["outer", "other"]
