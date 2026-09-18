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

import concurrent.futures
import itertools
import os
import sys
import threading

_threadPools = {}
_threadPoolsLock = threading.Lock()


def isFreeThreadingSupported() -> bool:
    """Check if free threading is supported in the current build of EdelweissFE.

    Free threading allows for parallel computations using multiple threads.
    This function checks if the current build of EdelweissFE supports this feature.

    Returns:
        bool: True if free threading is supported and the GIL is disabled, False otherwise.
    """
    res = False
    try:
        res = not sys._is_gil_enabled()
    except AttributeError:
        pass

    return res


def getNumberOfThreads() -> int:
    """Get the number of threads available for parallel computations.

    EdelweissFE has a built-in mechanism to determine the number of threads to be used for parallel computations.
    It checks if the environment variable `OMP_NUM_THREADS` is set. If it is
    and can be converted to an integer, that value is used. If not, the function falls back to using a default value of 1.
    The result is clamped to at least 1, since a `ThreadPoolExecutor` requires at least one worker.

    Returns:
        int: Number of threads to be used.
    """

    try:
        env_threads = os.environ.get("OMP_NUM_THREADS")
        num_workers = int(env_threads) if env_threads else 1
    except ValueError:
        num_workers = 1

    return max(1, num_workers)


def getNumberOfAvailableCpus() -> int:
    """Get the number of CPUs this process is actually permitted to run on.

    This is not the machine's core count. A batch scheduler's cgroup, ``taskset``, and an OpenMP
    runtime honouring ``OMP_PROC_BIND`` all narrow it -- the last of them at the moment it is
    loaded, which in this framework is whenever the first compiled extension is imported.

    Returns
    -------
    int
        The number of CPUs available to this process, or 0 where the platform cannot report it.
    """

    try:
        return len(os.sched_getaffinity(0))
    except (AttributeError, OSError):  # pragma: no cover - non-Linux
        return 0


def reportThreadAvailability(numThreads: int, journal, senderIdentification: str):
    """Report how many threads will be used, and warn if the process cannot actually use them.

    A pool of ``numThreads`` workers only runs on ``numThreads`` cores if the process is permitted
    to use that many, and ``OMP_PROC_BIND``/``OMP_PLACES`` are the common reason for it not to be:
    they make the OpenMP runtime pin the thread that loads it to a single place, and every thread
    started afterwards -- the pool's workers included -- inherits that one-core mask. The element
    loop is then serial in fact while still reporting the threads it asked for, and nothing fails,
    so this is worth saying out loud rather than leaving to be discovered by measurement.

    Parameters
    ----------
    numThreads
        The number of threads the solver is about to use.
    journal
        The journal to report to.
    senderIdentification
        The name of the reporting solver.
    """

    journal.message("Using {:} threads".format(numThreads), senderIdentification)

    availableCpus = getNumberOfAvailableCpus()
    if availableCpus and availableCpus < numThreads:
        journal.message(
            "WARNING: this process is allowed to run on {:} CPUs, fewer than the {:} threads it "
            "just asked for, so those threads will take turns on the same cores instead of running "
            "side by side. If OMP_PROC_BIND or OMP_PLACES is set in the environment, unset them "
            "for EdelweissFE: they pin this process to a single place as soon as an OpenMP runtime "
            "is loaded, and every thread started after that inherits the pinning.".format(availableCpus, numThreads),
            senderIdentification,
        )


def getThreadPool(numThreads: int) -> concurrent.futures.ThreadPoolExecutor:
    """Get a persistent thread pool with the requested number of worker threads.

    Pools are created lazily and reused for the lifetime of the process. This avoids
    the cost of spawning and joining worker threads for every parallel computation,
    which the solvers request once per Newton iteration.

    Parameters
    ----------
    numThreads
        The number of worker threads. Clamped to at least 1.

    Returns
    -------
    concurrent.futures.ThreadPoolExecutor
        The persistent thread pool.
    """

    numThreads = max(1, numThreads)

    pool = _threadPools.get(numThreads)
    if pool is None:
        with _threadPoolsLock:
            pool = _threadPools.get(numThreads)
            if pool is None:
                pool = _threadPools[numThreads] = concurrent.futures.ThreadPoolExecutor(max_workers=numThreads)

    return pool


def chunked_iterable(iterable, size):
    """Yield successive n-sized chunks from an iterable."""
    it = iter(iterable)
    while True:
        chunk = tuple(itertools.islice(it, size))
        if not chunk:
            break
        yield chunk
