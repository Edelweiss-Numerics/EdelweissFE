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
"""The vertices, edges and faces of the *root* mesh, and where on them a refinement node lies --
decided without looking at a single physical coordinate.

Vocabulary
----------
Root element
    An element of the mesh before any refinement. Every refined element lies inside exactly one.
Reference lattice
    The reference cube of a root element, :math:`[-1, 1]^3`, carries an integer lattice: reference
    coordinate :math:`\\xi` corresponds to the integer :math:`(\\xi + 1)\\,M`, so the cube spans
    :math:`[0, 2M]` per axis. :math:`M` is a power of the split factor, fine enough that every node a
    subdivision can create sits on a lattice point -- so all positions are *exact integers*.
Root entity
    A vertex, edge or face of the root mesh, named by the labels of its corner nodes. Two root elements
    that share a face therefore derive the identical name for it, whatever the orientation of their
    local axes.
Node key
    Where a node sits: the lowest-dimensional root entity containing it, and its exact integer
    position on that entity, measured in a frame that the entity's corner labels fix (see
    :class:`NodeKey`). Two nodes are at one point iff they have one key.

This is the mechanism established AMR libraries use (deal.II, p4est, MFEM): whether two points
coincide and whether a point lies on a face are questions of topology and exact arithmetic, so no
tolerance can make them wrong on a curved or warped mesh.
"""

from typing import NamedTuple


class NodeKey(NamedTuple):
    """Where a node sits: a root entity and the node's exact integer position on it.

    ============  ====================================  ===========================================
    ``kind``      ``entity``                            ``position``
    ============  ====================================  ===========================================
    ``vertex``    the vertex's node label               ``()``
    ``edge``      its two corner labels, ascending      ``(t,)`` from the smaller label, in [0, 2M]
    ``face``      its four corner labels, ascending     ``(u, v)`` in the canonical face frame
    ``interior``  the root element's id                 ``(x, y, z)`` in the root's lattice
    ============  ====================================  ===========================================

    ``component`` is the body the node belongs to: two bodies that touch at a flush interface
    (a tie, a contact pair, a crack plane) share no entity, even where their coordinates coincide.
    """

    kind: str
    component: int
    entity: object
    position: tuple


class RootFace(NamedTuple):
    """A face of the root mesh, named by the ascending labels of its four corner nodes."""

    component: int
    cornerLabels: tuple


class RootEntityTable:
    """The vertices, edges and faces of the root mesh, keyed by root corner labels.

    Parameters
    ----------
    topology
        The :class:`~edelweissfe.adaptivity.topologybase.TopologyBase` of the root elements.
    latticeUnits
        :math:`M`: reference coordinate :math:`\\xi` of a root element is the lattice integer
        :math:`(\\xi + 1)\\,M`.
    """

    def __init__(self, topology, latticeUnits: int):
        self._topology = topology
        self._latticeUnits = latticeUnits
        self._cornerSlots = topology.corner_slots
        #: a corner's position in the root lattice (every coordinate 0 or 2M)
        self._cornerPoint = {
            slot: tuple(int(v) * latticeUnits for v in topology.reference_node_lattice[slot])
            for slot in self._cornerSlots
        }
        cornerSet = set(self._cornerSlots)
        #: per face: its corner slots in loop order; per edge: its two corner slots
        self._faceCorners = [[slot for slot in face if slot in cornerSet] for face in topology.faces]
        self._edgeCorners = [[slot for slot in edge if slot in cornerSet] for edge in topology.edges]
        self._rootCorners = {}  # rootEid -> {corner slot: node label}
        self._rootComponent = {}  # rootEid -> componentId
        self._rootsOfFace = {}  # RootFace -> [rootEid, ...]
        self._rootsOfCorner = {}  # (componentId, corner label) -> [rootEid, ...]
        self._keyOfPoint = {}  # (rootEid, lattice point) -> NodeKey; a node is shared by many elements

    def add_root(self, rootEid: int, connectivity, componentId: int = 0):
        """Register a root element by its node labels (in the topology's local node order).

        Raises
        ------
        ValueError
            For a collapsed root element (a corner label repeated), whose faces and edges cannot be
            named by their corners, and for a face owned by more than two roots (a non-manifold mesh).
        """
        corners = {slot: connectivity[slot] for slot in self._cornerSlots}
        if len(set(corners.values())) != len(corners):
            raise ValueError(
                f"root element {rootEid} repeats a corner node label (a collapsed element); its faces "
                "and edges cannot be named by their corner labels"
            )
        self._rootCorners[rootEid] = corners
        self._rootComponent[rootEid] = componentId
        for label in corners.values():
            self._rootsOfCorner.setdefault((componentId, label), []).append(rootEid)
        for faceCorners in self._faceCorners:
            face = RootFace(componentId, tuple(sorted(corners[s] for s in faceCorners)))
            owners = self._rootsOfFace.setdefault(face, [])
            owners.append(rootEid)
            if len(owners) > 2:
                raise ValueError(
                    f"the face with corner labels {list(face.cornerLabels)} is shared by more than two root "
                    f"elements ({owners}); the root mesh is not a manifold"
                )

    def key(self, rootEid: int, point) -> NodeKey:
        """The :class:`NodeKey` of a lattice point (three integers in [0, 2M]) of root ``rootEid``."""
        cached = self._keyOfPoint.get((rootEid, point))
        if cached is None:
            cached = self._keyOfPoint[(rootEid, point)] = self._computeKey(rootEid, point)
        return cached

    def roots_sharing(self, key: NodeKey) -> list:
        """The root elements whose closure contains the entity of ``key``."""
        if key.kind == "interior":
            return [key.entity]
        if key.kind == "face":
            return list(self._rootsOfFace[RootFace(key.component, key.entity)])
        if key.kind == "vertex":
            return list(self._rootsOfCorner[(key.component, key.entity)])
        a, b = key.entity
        withBoth = set(self._rootsOfCorner[(key.component, a)]) & set(self._rootsOfCorner[(key.component, b)])
        return sorted(root for root in withBoth if self._isEdgeOf(root, key.entity))

    def roots_touching(self, rootEid: int) -> set:
        """The root elements sharing at least one vertex (hence possibly an edge or a face) with
        ``rootEid``, including itself."""
        component = self._rootComponent[rootEid]
        return {
            root for label in self._rootCorners[rootEid].values() for root in self._rootsOfCorner[(component, label)]
        }

    def root_face_position(self, rootEid: int, faceIndex: int, point) -> tuple:
        """The root face ``faceIndex`` (into ``topology.faces``) of ``rootEid`` as a :class:`RootFace`,
        and the position of ``point`` (a lattice point on it, possibly on its boundary) in the face's
        canonical frame -- identical from either root sharing the face."""
        corners = self._rootCorners[rootEid]
        faceCorners = self._faceCorners[faceIndex]
        origin, first, second = self._canonicalFaceFrame(faceCorners, corners)
        face = RootFace(self._rootComponent[rootEid], tuple(sorted(corners[s] for s in faceCorners)))
        return face, self._faceCoordinates(point, origin, first, second)

    def local_point(self, rootEid: int, key: NodeKey) -> tuple:
        """The lattice point, in root element ``rootEid``, of a key whose entity lies in that root's
        closure (see :meth:`roots_sharing`)."""
        corners = self._rootCorners[rootEid]
        slotOf = {label: slot for slot, label in corners.items()}
        if key.kind == "interior":
            return key.position
        if key.kind == "vertex":
            return self._cornerPoint[slotOf[key.entity]]
        if key.kind == "edge":
            a, b = (self._cornerPoint[slotOf[label]] for label in key.entity)
            (t,) = key.position
            return tuple(a[k] + t * (b[k] - a[k]) // (2 * self._latticeUnits) for k in range(3))
        faceCorners = next(fc for fc in self._faceCorners if tuple(sorted(corners[s] for s in fc)) == key.entity)
        origin, first, second = (self._cornerPoint[s] for s in self._canonicalFaceFrame(faceCorners, corners))
        u, v = key.position
        twoM = 2 * self._latticeUnits
        return tuple(origin[k] + (u * (first[k] - origin[k]) + v * (second[k] - origin[k])) // twoM for k in range(3))

    # ---- internals ----
    def _computeKey(self, rootEid: int, point) -> NodeKey:
        corners = self._rootCorners[rootEid]
        component = self._rootComponent[rootEid]
        kind, entity = self._topology.lowest_entity_containing(point, 2 * self._latticeUnits)
        if kind == "vertex":
            return NodeKey("vertex", component, corners[entity], ())
        if kind == "edge":
            a, b = (slot for slot in entity if slot in corners)
            if corners[a] > corners[b]:
                a, b = b, a
            return NodeKey("edge", component, (corners[a], corners[b]), (self._edgeParameter(point, a, b),))
        if kind == "face":
            faceCorners = [slot for slot in entity if slot in corners]
            origin, first, second = self._canonicalFaceFrame(faceCorners, corners)
            labels = tuple(sorted(corners[s] for s in faceCorners))
            return NodeKey("face", component, labels, self._faceCoordinates(point, origin, first, second))
        return NodeKey("interior", component, rootEid, tuple(point))

    def _isEdgeOf(self, rootEid, labels) -> bool:
        corners = self._rootCorners[rootEid]
        return any({corners[a], corners[b]} == set(labels) for a, b in self._edgeCorners)

    def _edgeParameter(self, point, a, b) -> int:
        """Position along the root edge from corner ``a`` (0) to corner ``b`` (2M)."""
        pa, pb = self._cornerPoint[a], self._cornerPoint[b]
        axis = next(k for k in range(3) if pa[k] != pb[k])
        return abs(point[axis] - pa[axis])

    def _canonicalFaceFrame(self, faceCorners, corners):
        """The face frame fixed by the corner labels::

                second ------ .
                  |           |       origin: the corner with the smallest label
                  |           |       first:  its loop neighbour with the smaller label
                origin ---- first     second: its other loop neighbour

        so both roots sharing the face choose the same frame, whatever their local numbering."""
        k = min(range(4), key=lambda i: corners[faceCorners[i]])
        previous, following = faceCorners[(k - 1) % 4], faceCorners[(k + 1) % 4]
        first, second = (previous, following) if corners[previous] < corners[following] else (following, previous)
        return faceCorners[k], first, second

    def _faceCoordinates(self, point, origin, first, second) -> tuple:
        """Position of ``point`` in the face frame: the distances from the origin along the two frame
        edges, which are lattice-axis-aligned, each spanning [0, 2M]."""
        o, e1, e2 = (self._cornerPoint[s] for s in (origin, first, second))
        axisU = next(k for k in range(3) if e1[k] != o[k])
        axisV = next(k for k in range(3) if e2[k] != o[k])
        return (abs(point[axisU] - o[axisU]), abs(point[axisV] - o[axisV]))
