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


def getNumberOfWorkerThreads() -> int:
    """The number of threads a parallel computation on the thread pool runs with:
    :func:`getNumberOfThreads` if free threading is available, else 1 -- without free threading the
    workers would only take turns on the GIL, so the work runs serially in the calling thread.

    Returns
    -------
    int
        The number of worker threads, at least 1.
    """

    return getNumberOfThreads() if isFreeThreadingSupported() else 1


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


def _mapOnPoolAndWaitForRunningCalls(work, tasks, numThreads: int) -> list:
    """Call ``work(task)`` for every task on the persistent thread pool, and return what the calls returned, in the
    order of the tasks.

    If a call raises, the exception is that of the first failing call in the order of the tasks -- the one a serial
    loop meets first -- raised once the calls before it are done. The calls that have not started are not made, as with
    ``Executor.map``. The calls that are running cannot be stopped: ``Executor.map`` leaves them running, this function
    waits for them. So when it returns or raises, no call is running. Whoever retries after the exception -- the solver
    repeats an increment with a smaller time step when an element raises a ``CutbackRequest`` -- must not have workers
    of the failed attempt still evaluating the same elements and writing into the same arrays as the retry.

    Parameters
    ----------
    work
        Called once per task, with the task.
    tasks
        The tasks, an iterable; it is consumed in the calling thread.
    numThreads
        The number of worker threads of the pool.

    Returns
    -------
    list
        What ``work`` returned for each task, in the order of the tasks.
    """

    pool = getThreadPool(numThreads)
    futures = [pool.submit(work, task) for task in tasks]

    try:
        # In task order, like a serial loop: the first failing call raises, once the calls before it are done.
        return [future.result() for future in futures]
    except BaseException:  # also a KeyboardInterrupt must not leave workers running behind it
        # The calls that have not started are not made, as with Executor.map. The running ones cannot be stopped, so
        # wait for them: cancel() leaves them alone, and a cancelled call counts as done for wait().
        for future in futures:
            future.cancel()
        concurrent.futures.wait(futures)
        raise


def mapChunksInParallel(items, work, chunksPerThread: int = 4) -> list:
    """Call ``work(chunk)`` for consecutive chunks of ``items`` on the persistent thread pool, and return what the calls
    returned, in the order of the chunks.

    Meant for a loop over entities (the elements) whose iterations are independent: each call handles a whole chunk, so
    that the cost of handing out tasks stays negligible against the work, and there are several chunks per thread, so
    that the pool balances chunks of unequal cost. The results come in the order of ``items``: what is assembled from
    them is what the serial loop assembles.

    Runs serially in the calling thread, as one chunk holding all ``items``, if free threading is not available or one
    thread is asked for.

    When it returns or raises, no call is running. If a call raises, the exception is that of the first failing chunk
    in the order of the chunks, the one a serial loop meets first; the chunks that have not started are not made, and
    the running ones are waited for.

    Parameters
    ----------
    items
        The entities, a sized iterable (a list, a dictionary view).
    work
        Called once per chunk, with a tuple of consecutive items; a worker must write only to what belongs to its items.
    chunksPerThread
        How many chunks each thread is handed.

    Returns
    -------
    list
        What ``work`` returned for each chunk, in order.
    """

    numThreads = getNumberOfWorkerThreads()

    if numThreads == 1:
        return [work(items)]

    chunkSize = max(1, len(items) // (numThreads * chunksPerThread))

    return _mapOnPoolAndWaitForRunningCalls(work, chunked_iterable(items, chunkSize), numThreads)


def forEachSliceInParallel(work, length: int, sliceLength: int = 1 << 20) -> list:
    """Call ``work(s)`` for consecutive slices ``s`` that together cover ``range(length)``, on the
    persistent thread pool, and return what the calls returned, in the order of the slices.

    Meant for the simple array operations on arrays as long as the system matrix -- zeroing it,
    adding a scaled mass matrix to it -- which numpy runs on a single thread, and which on a large
    model take seconds per Newton iteration that way. Every call works on its own part of the
    arrays, so the result is bitwise the one of the same operation on the whole array. Short slices
    keep the temporaries of an expression like ``K[s] += a * M[s]`` small.

    Runs serially in the calling thread if free threading is not available or the array fits into
    one slice.

    When it returns or raises, no call is running. If a call raises, the exception is that of the first failing slice
    in the order of the slices, the one a serial loop meets first; the slices that have not started are not made, and
    the running ones are waited for.

    Parameters
    ----------
    work
        Called once per slice, with a :class:`slice` of ``range(length)``; what it returns (e.g. the indices found in
        the slice) is collected.
    length
        The length of the arrays ``work`` operates on.
    sliceLength
        The length of the slices; the last one may be shorter.

    Returns
    -------
    list
        What ``work`` returned for each slice, in order.
    """

    slices = [slice(start, min(start + sliceLength, length)) for start in range(0, length, sliceLength)]
    numThreads = getNumberOfWorkerThreads()

    if numThreads == 1 or len(slices) <= 1:
        return [work(s) for s in slices]

    return _mapOnPoolAndWaitForRunningCalls(work, slices, numThreads)
