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
Mesh and DOF partitioning for distributed-memory parallelization.

This module provides functionality to partition a finite element mesh
across multiple MPI ranks. It supports:

- Element-based partitioning using simple graph bisection or external
  partitioners (when available).
- Node ownership assignment based on element partitioning.
- Ghost node/DOF identification for inter-rank communication.

The partitioner creates a :class:`MeshPartition` object that describes
which elements, nodes, and DOFs belong to the local rank (owned),
and which are ghost copies needed for assembly.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from edelweissfe.numerics.mpi.mpiutils import (
    broadcast,
    getComm,
    getRank,
    getSize,
)


@dataclass
class MeshPartition:
    """Describes the local partition of a mesh on one MPI rank.

    Attributes
    ----------
    rank : int
        The MPI rank this partition belongs to.
    nRanks : int
        Total number of MPI ranks.
    ownedElementIds : list
        Element IDs owned by this rank (responsible for computing).
    ownedNodeIds : set
        Node IDs owned by this rank (DOFs are authoritative here).
    ghostNodeIds : set
        Node IDs that are needed for assembly but owned by another rank.
    elementPartitionMap : dict
        Mapping from element ID to owning rank (for all elements).
    nodeOwnerMap : dict
        Mapping from node ID to owning rank (for all nodes).
    """

    rank: int = 0
    nRanks: int = 1
    ownedElementIds: list = field(default_factory=list)
    ownedNodeIds: set = field(default_factory=set)
    ghostNodeIds: set = field(default_factory=set)
    elementPartitionMap: dict = field(default_factory=dict)
    nodeOwnerMap: dict = field(default_factory=dict)


def partitionMesh(model, comm=None) -> MeshPartition:
    """Partition the mesh across MPI ranks.

    Uses a simple contiguous element partitioning strategy. Each rank
    gets a roughly equal number of elements. Nodes are assigned to the
    rank with the lowest rank number among element owners that reference
    the node. Ghost nodes are identified as nodes referenced by local
    elements but owned by another rank.

    For better load balancing on complex meshes, this function can be
    extended to use graph-based partitioners (e.g., METIS/ParMETIS via
    pymetis or networkx).

    Parameters
    ----------
    model
        The FEModel instance containing nodes, elements, etc.
    comm
        MPI communicator. If None, uses COMM_WORLD.

    Returns
    -------
    MeshPartition
        The partition description for the local rank.
    """

    rank = getRank(comm)
    nRanks = getSize(comm)

    elementIds = list(model.elements.keys())
    nElements = len(elementIds)

    # Partition elements across ranks (contiguous chunks)
    # Root computes the partitioning and broadcasts
    if rank == 0:
        elementPartitionMap = _computeElementPartition(elementIds, nRanks, model)
    else:
        elementPartitionMap = None

    elementPartitionMap = broadcast(elementPartitionMap, root=0, comm=comm)

    # Determine which elements are owned by this rank
    ownedElementIds = [eid for eid, owner in elementPartitionMap.items() if owner == rank]

    # Determine node ownership: node belongs to the lowest-ranked element owner
    nodeOwnerMap = _computeNodeOwnership(model, elementPartitionMap)

    # Determine owned and ghost nodes for this rank
    ownedNodeIds = set()
    ghostNodeIds = set()

    # All nodes referenced by our elements
    localNodeIds = set()
    for eid in ownedElementIds:
        el = model.elements[eid]
        for node in el.nodes:
            localNodeIds.add(id(node))

    for nodeId, nodeObj in model.nodes.items():
        nid = id(nodeObj)
        if nid in localNodeIds:
            if nodeOwnerMap.get(nodeId, rank) == rank:
                ownedNodeIds.add(nodeId)
            else:
                ghostNodeIds.add(nodeId)

    return MeshPartition(
        rank=rank,
        nRanks=nRanks,
        ownedElementIds=ownedElementIds,
        ownedNodeIds=ownedNodeIds,
        ghostNodeIds=ghostNodeIds,
        elementPartitionMap=elementPartitionMap,
        nodeOwnerMap=nodeOwnerMap,
    )


def _computeElementPartition(elementIds: list, nRanks: int, model) -> dict:
    """Compute element-to-rank assignment using graph-based partitioning.

    Uses a simple strategy: build the element adjacency graph (elements
    sharing nodes are neighbors), then partition using recursive coordinate
    bisection based on element centroid coordinates when available,
    falling back to contiguous partitioning.

    Parameters
    ----------
    elementIds
        List of all element IDs.
    nRanks
        Number of MPI ranks.
    model
        The FEModel.

    Returns
    -------
    dict
        Mapping from element ID to owning rank.
    """
    nElements = len(elementIds)

    if nRanks == 1:
        return {eid: 0 for eid in elementIds}

    # Try coordinate-based recursive bisection for better partitioning
    try:
        return _coordinateBisectionPartition(elementIds, nRanks, model)
    except (AttributeError, ValueError, TypeError):
        pass

    # Fallback: simple contiguous partitioning
    chunkSize = (nElements + nRanks - 1) // nRanks
    partitionMap = {}
    for i, eid in enumerate(elementIds):
        partitionMap[eid] = min(i // chunkSize, nRanks - 1)

    return partitionMap


def _coordinateBisectionPartition(elementIds: list, nRanks: int, model) -> dict:
    """Partition elements using recursive coordinate bisection (RCB).

    Computes element centroids and recursively bisects the domain
    along the longest dimension.

    Parameters
    ----------
    elementIds
        List of element IDs to partition.
    nRanks
        Number of target partitions.
    model
        The FEModel with node coordinates.

    Returns
    -------
    dict
        Mapping from element ID to owning rank.
    """
    # Compute centroids
    centroids = np.zeros((len(elementIds), 3))
    for i, eid in enumerate(elementIds):
        el = model.elements[eid]
        coords = np.array([node.coordinates for node in el.nodes])
        centroid = coords.mean(axis=0)
        # Pad to 3D if needed
        centroids[i, : len(centroid)] = centroid

    # Recursive coordinate bisection
    assignments = np.zeros(len(elementIds), dtype=int)
    _rcb_recursive(centroids, np.arange(len(elementIds)), 0, nRanks, assignments)

    return {elementIds[i]: int(assignments[i]) for i in range(len(elementIds))}


def _rcb_recursive(
    centroids: np.ndarray,
    indices: np.ndarray,
    rank_start: int,
    nRanks: int,
    assignments: np.ndarray,
):
    """Recursively bisect elements by coordinate.

    Parameters
    ----------
    centroids
        Array of shape (nElements, 3) with element centroids.
    indices
        Indices into the centroids array for this subgroup.
    rank_start
        Starting rank number for this subgroup.
    nRanks
        Number of ranks to distribute this subgroup across.
    assignments
        Output array to fill with rank assignments.
    """
    if nRanks == 1:
        assignments[indices] = rank_start
        return

    if len(indices) == 0:
        return

    # Find longest dimension
    coords = centroids[indices]
    spans = coords.max(axis=0) - coords.min(axis=0)
    split_dim = np.argmax(spans)

    # Sort by the splitting dimension
    sorted_local = np.argsort(coords[:, split_dim])
    sorted_indices = indices[sorted_local]

    # Split into two halves (proportional to rank count)
    nLeft = nRanks // 2
    nRight = nRanks - nLeft
    splitPoint = int(len(sorted_indices) * nLeft / nRanks)

    left_indices = sorted_indices[:splitPoint]
    right_indices = sorted_indices[splitPoint:]

    _rcb_recursive(centroids, left_indices, rank_start, nLeft, assignments)
    _rcb_recursive(centroids, right_indices, rank_start + nLeft, nRight, assignments)


def _computeNodeOwnership(model, elementPartitionMap: dict) -> dict:
    """Determine node ownership from element partitioning.

    A node is owned by the lowest-ranked process among all element
    owners that reference it.

    Parameters
    ----------
    model
        The FEModel.
    elementPartitionMap
        Mapping from element ID to owning rank.

    Returns
    -------
    dict
        Mapping from node ID to owning rank.
    """
    nodeOwnerMap = {}

    for eid, ownerRank in elementPartitionMap.items():
        el = model.elements[eid]
        for node in el.nodes:
            # Use the node's key in model.nodes
            for nodeId, nodeObj in model.nodes.items():
                if nodeObj is node:
                    if nodeId not in nodeOwnerMap:
                        nodeOwnerMap[nodeId] = ownerRank
                    else:
                        nodeOwnerMap[nodeId] = min(nodeOwnerMap[nodeId], ownerRank)
                    break

    return nodeOwnerMap
