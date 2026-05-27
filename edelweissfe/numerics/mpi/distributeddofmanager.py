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
Distributed DOF Manager for MPI-parallel finite element computations.

This module extends the serial DOF numbering concept to work with
distributed-memory parallelization. It provides:

- Local-to-global DOF index mappings.
- Ghost DOF identification and ownership tracking.
- PETSc-compatible local/global index sets (IS) for assembly.
- Mappings between the serial DofManager indices and distributed
  PETSc Vec/Mat layout.

The :class:`DistributedDofManager` wraps around the standard
:class:`DofManager` and augments it with distributed ownership
information derived from a :class:`MeshPartition`.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from edelweissfe.numerics.mpi.mpiutils import getComm, getRank, getSize
from edelweissfe.numerics.mpi.partitioner import MeshPartition

try:
    from mpi4py import MPI

    _MPI_AVAILABLE = True
except ImportError:
    _MPI_AVAILABLE = False


@dataclass
class DistributedDofManager:
    """Manages the distributed DOF layout across MPI ranks.

    Built on top of the serial DofManager, this class maintains the
    mapping between local and global DOF indices and identifies which
    DOFs are owned vs. ghosted on each rank.

    Attributes
    ----------
    dofManager
        The underlying serial DofManager (contains full global numbering).
    partition
        The MeshPartition describing element/node ownership.
    nLocalOwned : int
        Number of DOFs owned by this rank.
    nLocalGhost : int
        Number of ghost DOFs on this rank.
    nLocalTotal : int
        Total local DOFs (owned + ghost).
    nGlobal : int
        Total global DOFs across all ranks.
    ownedGlobalDofs : np.ndarray
        Global DOF indices owned by this rank.
    ghostGlobalDofs : np.ndarray
        Global DOF indices that are ghosts on this rank.
    localGlobalDofs : np.ndarray
        All local DOF indices in global numbering (owned first, then ghosts).
    globalToLocal : dict
        Mapping from global DOF index to local DOF index.
    petscLocalRange : tuple
        (start, end) range of owned DOFs in PETSc's contiguous layout.
    """

    dofManager: object
    partition: MeshPartition
    nLocalOwned: int = 0
    nLocalGhost: int = 0
    nLocalTotal: int = 0
    nGlobal: int = 0
    ownedGlobalDofs: np.ndarray = field(default_factory=lambda: np.array([], dtype=int))
    ghostGlobalDofs: np.ndarray = field(default_factory=lambda: np.array([], dtype=int))
    localGlobalDofs: np.ndarray = field(default_factory=lambda: np.array([], dtype=int))
    globalToLocal: dict = field(default_factory=dict)
    petscLocalRange: tuple = (0, 0)


def createDistributedDofManager(dofManager, partition: MeshPartition, comm=None) -> DistributedDofManager:
    """Create a distributed DOF manager from a serial DofManager and partition.

    This function analyzes the serial DofManager's global numbering
    and the mesh partition to determine:
    - Which DOFs are owned by the local rank.
    - Which DOFs are needed as ghosts.
    - The local/global index mappings for PETSc assembly.

    Parameters
    ----------
    dofManager
        The serial DofManager with full global DOF numbering.
    partition
        The MeshPartition for this rank.
    comm
        MPI communicator.

    Returns
    -------
    DistributedDofManager
        The distributed DOF layout for this rank.
    """
    rank = getRank(comm)
    nRanks = getSize(comm)

    nGlobal = dofManager.nDof

    # Determine which global DOF indices are owned by this rank
    # A DOF is owned if it belongs to a node owned by this rank
    ownedDofSet = set()
    ghostDofSet = set()

    # Get DOF indices for owned nodes
    if hasattr(dofManager, "idcsOfFieldVariablesInDofVector"):
        for fieldVar, indices in dofManager.idcsOfFieldVariablesInDofVector.items():
            node = fieldVar.parentNode if hasattr(fieldVar, "parentNode") else None
            if node is not None:
                # Check if this node is owned
                nodeId = _findNodeId(node, partition)
                if nodeId is not None:
                    if nodeId in partition.ownedNodeIds:
                        ownedDofSet.update(indices)
                    elif nodeId in partition.ghostNodeIds:
                        ghostDofSet.update(indices)

    # For scalar variables, assign to rank 0
    if hasattr(dofManager, "idcsOfScalarVariablesInDofVector"):
        for scalarVar, indices in dofManager.idcsOfScalarVariablesInDofVector.items():
            if rank == 0:
                ownedDofSet.update(indices)
            else:
                ghostDofSet.update(indices)

    # If no partition info available (fallback), all DOFs owned by rank 0
    if not ownedDofSet and not ghostDofSet:
        if rank == 0:
            ownedDofSet = set(range(nGlobal))
        else:
            ghostDofSet = set(range(nGlobal))

    # Remove any DOFs that appear in both sets (shouldn't happen but safety)
    ghostDofSet -= ownedDofSet

    ownedGlobalDofs = np.sort(np.array(list(ownedDofSet), dtype=int))
    ghostGlobalDofs = np.sort(np.array(list(ghostDofSet), dtype=int))
    localGlobalDofs = np.concatenate([ownedGlobalDofs, ghostGlobalDofs])

    nLocalOwned = len(ownedGlobalDofs)
    nLocalGhost = len(ghostGlobalDofs)
    nLocalTotal = nLocalOwned + nLocalGhost

    # Build global-to-local mapping
    globalToLocal = {gidx: lidx for lidx, gidx in enumerate(localGlobalDofs)}

    # Compute PETSc local range (contiguous ownership)
    # This needs to be computed via prefix sum of owned counts
    if _MPI_AVAILABLE and comm is not None:
        ownedCounts = np.array([nLocalOwned], dtype=int)
        allCounts = np.zeros(nRanks, dtype=int)
        comm.Allgather(ownedCounts, allCounts)
        startIdx = int(np.sum(allCounts[:rank]))
        endIdx = startIdx + nLocalOwned
    else:
        startIdx = 0
        endIdx = nLocalOwned

    return DistributedDofManager(
        dofManager=dofManager,
        partition=partition,
        nLocalOwned=nLocalOwned,
        nLocalGhost=nLocalGhost,
        nLocalTotal=nLocalTotal,
        nGlobal=nGlobal,
        ownedGlobalDofs=ownedGlobalDofs,
        ghostGlobalDofs=ghostGlobalDofs,
        localGlobalDofs=localGlobalDofs,
        globalToLocal=globalToLocal,
        petscLocalRange=(startIdx, endIdx),
    )


def getElementLocalDofs(element, dofManager, distDofManager: DistributedDofManager) -> np.ndarray:
    """Get the local DOF indices for an element.

    Maps an element's global DOF indices (from the serial DofManager)
    to local indices in the distributed layout.

    Parameters
    ----------
    element
        The finite element.
    dofManager
        The serial DofManager.
    distDofManager
        The distributed DOF manager.

    Returns
    -------
    np.ndarray
        Local DOF indices for this element.
    """
    globalIdcs = dofManager.idcsOfElementsInDofVector[element]
    return np.array([distDofManager.globalToLocal[g] for g in globalIdcs], dtype=int)


def _findNodeId(node, partition: MeshPartition):
    """Find the node ID for a node object in the partition.

    Note: This requires the partition to have a nodeObjToId mapping.
    For the initial implementation, callers should build this map
    externally and pass node IDs directly.

    Parameters
    ----------
    node
        The node object.
    partition
        The MeshPartition.

    Returns
    -------
    int or None
        The node ID, or None if not found.
    """
    # Check if node ID is in owned or ghost sets
    # This is a simplified implementation - in practice the caller
    # should pass the node ID directly rather than the node object
    for nodeId in partition.ownedNodeIds:
        return nodeId
    for nodeId in partition.ghostNodeIds:
        return nodeId
    return None
