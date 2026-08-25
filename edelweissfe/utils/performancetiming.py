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
import threading
from collections import defaultdict
from time import perf_counter

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
        self._tic = perf_counter()
        self.calls += 1

    def toc(
        self,
    ):
        """
        Stop measuring time.
        """
        self.time += perf_counter() - self._tic

    def get_snapshot(self) -> dict:
        """Returns a nested dictionary of the current accumulated times."""
        return {"time": self.time, "calls": self.calls, "children": {k: v.get_snapshot() for k, v in self.items()}}


class _ThreadLocalState(threading.local):
    """Every thread gets its own independent root branch and its own current-stack-position pointer.

    Under free-threading (PYTHON_GIL=0), :class:`~edelweissfe.solvers.base.parallelelementcomputation`
    genuinely runs Python bytecode from multiple OS threads concurrently (a real thread pool, not just
    native OpenMP workers that never touch Python state) -- and any of those worker threads may enter a
    ``timeit``-decorated/wrapped call. A single *shared* "current stack position" pointer and shared
    per-category accumulator nodes (the previous design) is not just unprotected, it is not a
    thread-safe design at all: two threads swapping the same shared pointer in ``__enter__``/``__exit__``
    can each observe the other's write, corrupting *which* node the surrounding stack frame resumes at,
    and two threads calling ``tic()``/``toc()`` on the very same shared node race on ``self._tic``,
    whichever calls second overwriting the first's start time before its own ``toc()`` reads it back --
    this is what produced the wildly inflated per-category ``acc. runtime`` figures (a "GMRES: 340s"
    reading on a run whose total wall-clock was 80s) recorded as a known gotcha before this fix.

    The correct fix is not a lock around the existing shared state (that would only serialize what is
    supposed to be genuinely parallel element computation) -- it is to give each thread its own,
    entirely independent tree, so no thread ever observes another thread's mutation of ``_tic``/``.time``
    /the stack pointer. Trees are only ever combined by :func:`_mergedSnapshot`, which reads (not
    mutates) each thread's already-accumulated snapshot -- safe because a thread's own tree is only
    ever mutated by that same thread.
    """

    def __init__(self):
        self.root = _PerformanceTimerBranch()
        self.stack = self.root
        self.registered = False


_threadLocalState = _ThreadLocalState()

_allRoots: list[_PerformanceTimerBranch] = []
"""Every thread's own root branch that has ever recorded anything, for :func:`_mergedSnapshot`."""
_allRootsLock = threading.Lock()
"""Guards *only* the append to :data:`_allRoots` -- registration happens once per thread, never on the
per-call tic()/toc()/stack-swap hot path this whole fix exists to keep lock-free."""


def _currentThreadRoot() -> _PerformanceTimerBranch:
    """The calling thread's own root branch, registering it (once) in :data:`_allRoots` first if this
    is that thread's first call into this module."""
    if not _threadLocalState.registered:
        with _allRootsLock:
            _allRoots.append(_threadLocalState.root)
        _threadLocalState.registered = True
    return _threadLocalState.root


def _mergeInto(dst: dict, src: dict) -> None:
    """Recursively accumulate src's time/calls/children into dst, in place."""
    dst["time"] += src["time"]
    dst["calls"] += src["calls"]
    for name, childSnapshot in src["children"].items():
        if name not in dst["children"]:
            dst["children"][name] = {"time": 0.0, "calls": 0, "children": {}}
        _mergeInto(dst["children"][name], childSnapshot)


def _mergedSnapshot() -> dict:
    """The combined snapshot across every thread that has ever recorded a measurement, summed by
    category name at every nesting level -- the multi-threaded equivalent of the old single shared
    ``times`` tree's own :meth:`_PerformanceTimerBranch.get_snapshot`.

    Quiescent-point requirement (found on review, not yet exploitable by any call site in this
    codebase): this reads every registered thread's tree without a lock. Safe as long as no other
    thread is still actively timing something when this is called -- true today, since every call
    site (:func:`makePrettyTable`, :func:`extractIncrementTimes`, :func:`reset`) runs only after
    joining whatever thread pool did the work being reported on, never concurrently with it. Adding
    per-node locking to make this safe under genuinely concurrent reporting too would reintroduce
    exactly the lock contention on the hot tic()/toc() path this module's whole redesign exists to
    avoid, for a usage pattern nothing here currently needs.
    """
    with _allRootsLock:
        roots = list(_allRoots)
    merged = {"time": 0.0, "calls": 0, "children": {}}
    for root in roots:
        _mergeInto(merged, root.get_snapshot())
    return merged


class timeit:
    """Decorator class for performance timing of functions.
    This decorator has a runtime memory, i.e., it is aware of the stack level
    of nested timed functions.

    Parameters
    ----------
    category
        The category for storing the measured time.

    Thread-safety note (found on review): the *decorator* form (``@timeit(...)``) constructs exactly
    one ``timeit`` instance at decoration time, and every call to the decorated function reuses the
    same ``wrapper`` closure over that one instance -- so anything stored on ``self`` here would be
    shared and racy across concurrent or recursive calls to the same decorated function, exactly the
    class of bug this module's thread-local redesign exists to eliminate. ``wrapper`` therefore keeps
    its "what do I restore the stack to" bookkeeping in a plain local variable, not on ``self`` --
    safe by construction, since every call gets its own Python stack frame regardless of which thread
    or how many concurrent/recursive calls are in flight. The *context-manager* form (``with
    timeit(...):``) cannot do the same (``__enter__``/``__exit__`` are separate calls needing to share
    state across them) -- every call site in this codebase already constructs a fresh instance per
    ``with`` block, which is enough in practice, but ``self._parentStackLevel`` is kept in a
    per-instance ``threading.local()`` regardless, so even a ``with`` block built from a *shared,
    reused* ``timeit`` instance stays correct if used across multiple threads.
    """

    def __init__(self, category: str):
        self._category = category
        self._local = threading.local()

    def __call__(self, theFunction):
        category = self._category

        def wrapper(*args, **kwargs):
            _currentThreadRoot()  # idempotent registration; cheap after the first call on this thread
            state = _threadLocalState
            parentStackLevel = state.stack  # a plain local, not self.* -- see the class docstring
            timer = state.stack[category]
            state.stack = timer

            timer.tic()
            try:
                return theFunction(*args, **kwargs)
            finally:
                timer.toc()
                state.stack = parentStackLevel

        wrapper.__doc__ = theFunction.__doc__
        wrapper.__module__ = theFunction.__module__
        wrapper.__signature__ = inspect.signature(theFunction)

        return wrapper

    def __enter__(self):
        _currentThreadRoot()  # idempotent registration; cheap after the first call on this thread
        state = _threadLocalState
        self._local.parentStackLevel = state.stack
        timer = state.stack[self._category]
        state.stack = timer
        timer.tic()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        _threadLocalState.stack.toc()
        _threadLocalState.stack = self._local.parentStackLevel


def _makeTable(branch: dict, level: int, maxLevels: int) -> list[tuple]:
    """Recursive function for creating a table of the measured times.

    Parameters
    ----------
    branch
        The current active branch, as a :meth:`_PerformanceTimerBranch.get_snapshot`-shaped dict
        (``{"time": ..., "calls": ..., "children": {name: dict, ...}}``).
    levels
        The current level.
    maxLevels
        The maximum number of stack levels considered in the table.

    Returns
    -------
    list[tuple]
        The table in list format containing columns as tuples."""

    table = []
    for k, v in branch["children"].items():
        table.append((level, k, v["time"], v["calls"]))
        if level < maxLevels and v["children"]:
            table += _makeTable(v, level + 1, maxLevels)

    return table


def makePrettyTable(maxLevels: int = 4) -> PrettyTable:
    """Create a pretty formatted table of the measured times, merged across every thread that has
    recorded a measurement.

    Parameters
    ----------
    maxLevels
        The maximum number of stack levels considered in the table.

    Returns
    -------
    PrettyTable
        The table in pretty format."""

    theTable = _makeTable(_mergedSnapshot(), 0, maxLevels)

    prettytable = PrettyTable()
    prettytable.field_names = ["function", "acc. runtime", "calls", "time/call"]
    prettytable.align = "l"

    for level, cat, t, calls in theTable:
        t_per_call = t / calls if calls > 0 else 0.0
        prettytable.add_row(
            (
                "{:}{:}".format(" " * level, cat),
                "{:}{:10.5f} s".format(" " * level, t),
                calls,
                "{:10.5f} s".format(t_per_call),
            )
        )

    return prettytable


def extractIncrementTimes(maxLevels: int = 4) -> PrettyTable:
    """
    Returns a PrettyTable of the time elapsed since the last time
    this function was called, while keeping the accumulated totals intact.
    """

    if not hasattr(extractIncrementTimes, "_last_snapshot") or extractIncrementTimes._last_snapshot is None:
        extractIncrementTimes._last_snapshot = None

    current_state = _mergedSnapshot()

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
    prettytable.field_names = ["function", "inc. runtime", "calls", "time/call"]
    prettytable.align = "l"
    for level, cat, t, calls in delta_rows:
        t_per_call = t / calls if calls > 0 else 0.0
        prettytable.add_row(
            [" " * level + cat, "{:}{:10.5f} s".format(" " * level, t), calls, "{:10.5f} s".format(t_per_call)]
        )

    return prettytable


def reset():
    """Reset all measured times, on every thread that has ever recorded one."""
    with _allRootsLock:
        roots = list(_allRoots)
    for root in roots:
        root.clear()
        root.time = 0.0
        root.calls = 0
    extractIncrementTimes._last_snapshot = None
