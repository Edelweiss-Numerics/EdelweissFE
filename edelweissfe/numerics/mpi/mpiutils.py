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

"""
MPI utility functions and communicator management for EdelweissFE.

This module provides a thin wrapper around mpi4py to handle MPI
communicator access, rank/size queries, and collective operations
used throughout the distributed solver infrastructure.

When MPI is not available (e.g., serial execution), the module
provides fallback stubs that emulate single-rank behavior.
"""

from __future__ import annotations

import numpy as np

try:
    from mpi4py import MPI

    _MPI_AVAILABLE = True
except ImportError:
    _MPI_AVAILABLE = False


def isMPIAvailable() -> bool:
    """Check whether mpi4py is available.

    Returns
    -------
    bool
        True if mpi4py can be imported.
    """
    return _MPI_AVAILABLE


def getComm():
    """Get the default MPI communicator (COMM_WORLD).

    Returns
    -------
    MPI.Comm or None
        The MPI communicator, or None if MPI is not available.
    """
    if _MPI_AVAILABLE:
        return MPI.COMM_WORLD
    return None


def getRank(comm=None) -> int:
    """Get the rank of the current process.

    Parameters
    ----------
    comm
        MPI communicator. If None, uses COMM_WORLD.

    Returns
    -------
    int
        The rank of this process (0 if MPI unavailable).
    """
    if not _MPI_AVAILABLE:
        return 0
    if comm is None:
        comm = MPI.COMM_WORLD
    return comm.Get_rank()


def getSize(comm=None) -> int:
    """Get the total number of MPI processes.

    Parameters
    ----------
    comm
        MPI communicator. If None, uses COMM_WORLD.

    Returns
    -------
    int
        The number of processes (1 if MPI unavailable).
    """
    if not _MPI_AVAILABLE:
        return 1
    if comm is None:
        comm = MPI.COMM_WORLD
    return comm.Get_size()


def isRootRank(comm=None) -> bool:
    """Check if current process is the root rank (rank 0).

    Parameters
    ----------
    comm
        MPI communicator. If None, uses COMM_WORLD.

    Returns
    -------
    bool
        True if this is rank 0.
    """
    return getRank(comm) == 0


def allreduce_sum(local_value: float, comm=None) -> float:
    """Perform an MPI Allreduce with SUM operation.

    Parameters
    ----------
    local_value
        The local contribution from this rank.
    comm
        MPI communicator. If None, uses COMM_WORLD.

    Returns
    -------
    float
        The global sum across all ranks.
    """
    if not _MPI_AVAILABLE:
        return local_value
    if comm is None:
        comm = MPI.COMM_WORLD
    return comm.allreduce(local_value, op=MPI.SUM)


def allreduce_max(local_value: float, comm=None) -> float:
    """Perform an MPI Allreduce with MAX operation.

    Parameters
    ----------
    local_value
        The local contribution from this rank.
    comm
        MPI communicator. If None, uses COMM_WORLD.

    Returns
    -------
    float
        The global maximum across all ranks.
    """
    if not _MPI_AVAILABLE:
        return local_value
    if comm is None:
        comm = MPI.COMM_WORLD
    return comm.allreduce(local_value, op=MPI.MAX)


def allreduce_array_sum(local_array: np.ndarray, comm=None) -> np.ndarray:
    """Perform an MPI Allreduce with SUM on a numpy array.

    Parameters
    ----------
    local_array
        The local numpy array from this rank.
    comm
        MPI communicator. If None, uses COMM_WORLD.

    Returns
    -------
    np.ndarray
        The element-wise global sum across all ranks.
    """
    if not _MPI_AVAILABLE:
        return local_array.copy()
    if comm is None:
        comm = MPI.COMM_WORLD
    global_array = np.zeros_like(local_array)
    comm.Allreduce(local_array, global_array, op=MPI.SUM)
    return global_array


def barrier(comm=None):
    """Synchronization barrier across all MPI ranks.

    Parameters
    ----------
    comm
        MPI communicator. If None, uses COMM_WORLD.
    """
    if not _MPI_AVAILABLE:
        return
    if comm is None:
        comm = MPI.COMM_WORLD
    comm.Barrier()


def broadcast(data, root: int = 0, comm=None):
    """Broadcast data from root to all ranks.

    Parameters
    ----------
    data
        The data to broadcast (only significant on root).
    root
        The root rank.
    comm
        MPI communicator. If None, uses COMM_WORLD.

    Returns
    -------
    object
        The broadcasted data on all ranks.
    """
    if not _MPI_AVAILABLE:
        return data
    if comm is None:
        comm = MPI.COMM_WORLD
    return comm.bcast(data, root=root)
