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

import inspect
from collections import defaultdict
from time import time

from prettytable import PrettyTable


class _PerformanceTimerBranch(defaultdict):
    def __init__(self):
        self.time = float()  #: the measured time for this branch.
        self.calls = int()
        self._tic = None

        super().__init__(_PerformanceTimerBranch)

    def tic(
        self,
    ):
        """
        Start measuring time.
        """
        self._tic = time()
        self.calls += 1

    def toc(
        self,
    ):
        """
        Stop measuring time.
        """
        self.time += time() - self._tic

    def get_snapshot(self) -> dict:
        """Returns a nested dictionary of the current accumulated times."""
        return {"time": self.time, "calls": self.calls, "children": {k: v.get_snapshot() for k, v in self.items()}}


times = _PerformanceTimerBranch()
"""The global dictionary of measured computations times."""


class timeit:
    """Decorator class for performance timing of functions.
    This decorator has a runtime memory, i.e., it is aware of the stack level
    of nested timed functions.

    Parameters
    ----------
    category
        The category for storing the measured time.
    """

    _currentStackLevel = times

    def __init__(self, category: str):
        self._category = category
        self._parentStackLevel = None

    def __call__(self, theFunction):
        def wrapper(*args, **kwargs):
            self._parentStackLevel = timeit._currentStackLevel
            timer = timeit._currentStackLevel[self._category]
            timeit._currentStackLevel = timer

            timer.tic()
            try:
                return theFunction(*args, **kwargs)
            finally:
                timer.toc()
                timeit._currentStackLevel = self._parentStackLevel

        wrapper.__doc__ = theFunction.__doc__
        wrapper.__module__ = theFunction.__module__
        wrapper.__signature__ = inspect.signature(theFunction)

        return wrapper


def _makeTable(branch: _PerformanceTimerBranch, level: int, maxLevels: int) -> list[tuple]:
    """Recursive function for creating a table of the measured times.

    Parameters
    ----------
    branch
        The current active branch.
    levels
        The current level.
    maxLevels
        The maximum number of stack levels considered in the table.

    Returns
    -------
    list[tuple]
        The table in list format containing columns as tuples."""

    table = []
    for k, v in branch.items():
        table.append((level, k, v.time, v.calls))
        if level < maxLevels and len(v):
            table += _makeTable(v, level + 1, maxLevels)

    return table


def makePrettyTable(maxLevels: int = 4) -> PrettyTable:
    """Create a pretty formatted table of the measured times.

    Parameters
    ----------
    maxLevels
        The maximum number of stack levels considered in the table.

    Returns
    -------
    PrettyTable
        The table in pretty format."""

    theTable = _makeTable(times, 0, maxLevels)

    prettytable = PrettyTable()
    prettytable.field_names = ["function", "acc. runtime", "calls"]
    prettytable.align = "l"

    for level, cat, t, calls in theTable:
        prettytable.add_row(
            (
                "{:}{:}".format(" " * level, cat),
                "{:}{:10.4E}s".format(" " * level, t),
                calls,
            )
        )

    return prettytable


def extractIncrementTimes(maxLevels: int = 4) -> PrettyTable:
    """
    Returns a PrettyTable of the time elapsed since the last time
    this function was called, while keeping global 'times' intact.
    """

    if not hasattr(extractIncrementTimes, "_last_snapshot") or extractIncrementTimes._last_snapshot is None:
        extractIncrementTimes._last_snapshot = None

    current_state = times.get_snapshot()

    def compute_delta(curr, last):
        last_t = last["time"] if last else 0.0
        last_c = last["calls"] if last else 0

        delta_t = curr["time"] - last_t
        delta_c = curr["calls"] - last_c

        children_deltas = []
        for name, child_curr in curr["children"].items():
            child_last = last["children"].get(name) if last else None
            children_deltas.append((name, compute_delta(child_curr, child_last)))

        return {"time": delta_t, "calls": delta_c, "children": children_deltas}

    delta_tree = compute_delta(current_state, extractIncrementTimes._last_snapshot)
    extractIncrementTimes._last_snapshot = current_state

    def flatten_delta(node, level):
        rows = []
        for name, data in node["children"]:
            rows.append((level, name, data["time"], data["calls"]))
            if level < maxLevels and data["children"]:
                rows += flatten_delta(data, level + 1)
        return rows

    delta_rows = flatten_delta(delta_tree, 0)

    prettytable = PrettyTable()
    prettytable.field_names = ["function", "inc. runtime", "calls"]
    prettytable.align = "l"
    for level, cat, t, calls in delta_rows:
        prettytable.add_row([" " * level + cat, "{:}{:10.4E}s".format(" " * level, t), calls])

    return prettytable


def reset():
    """Reset all measured times."""
    global times
    times.clear()
    extractIncrementTimes._last_snapshot = None
