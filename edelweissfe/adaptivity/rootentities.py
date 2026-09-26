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
"""Exact, topological positions of refinement nodes: which entity of the *root* mesh a node lies on,
and where on it -- without looking at a single physical coordinate.

Every element of the octree lies inside exactly one root element and covers an axis-aligned box of
that root's reference cube, obtained by exact subdivision; every node of it sits at an exact rational
reference point of the root. That point lies on a vertex, an edge or a face of the root, or in its
interior. A vertex, an edge and a face of the root mesh are named by the labels of their *root corner
nodes*, so two root elements that share a face derive the identical name for it, whatever the
orientation of their local axes. Together with the point's exact position on the entity, measured in a
frame fixed by those same labels, this gives every node position a unique key.

This is the mechanism established AMR libraries use (deal.II, p4est, MFEM): whether two points
coincide, and whether a point lies on a face, are questions of topology and exact arithmetic, and no
tolerance can make them wrong on a curved or warped mesh.
"""

from fractions import Fraction

import numpy as np


class RootEntityTable:
    """The vertices, edges and faces of the root mesh, keyed by root corner labels.

    Parameters
    ----------
    topology
        The :class:`~edelweissfe.adaptivity.topologybase.TopologyBase` of the root elements.
    """

    def __init__(self, topology):
        reference = np.asarray(topology.reference_node_param(), dtype=float)
        #: local slots of the corner nodes (every reference coordinate at +-1)
        self._cornerSlots = [slot for slot, p in enumerate(reference) if np.all(np.abs(p) == 1.0)]
        self._cornerReference = {slot: tuple(Fraction(int(v)) for v in reference[slot]) for slot in self._cornerSlots}
        cornerSet = set(self._cornerSlots)
        #: per face: its corner slots in loop order
        self._faceCorners = [[slot for slot in face if slot in cornerSet] for face in topology.faces]
        #: per edge: its two corner slots
        self._edgeCorners = [[slot for slot in edge if slot in cornerSet] for edge in topology.edges]
        self._rootCorners = {}  # rootEid -> {corner slot: node label}
        self._rootComponent = {}  # rootEid -> componentId
        self._facesOfRoots = {}  # (componentId, sorted face corner labels) -> [rootEid, ...]
        self._rootsOfCorner = {}  # (componentId, corner label) -> [rootEid, ...]

    def add_root(self, rootEid: int, connectivity, componentId: int = 0):
        """Register a root element by its node labels (in the topology's local node order)."""
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
            owners = self._facesOfRoots.setdefault((componentId, tuple(sorted(corners[s] for s in faceCorners))), [])
            owners.append(rootEid)
            if len(owners) > 2:
                raise ValueError(
                    f"the face with corner labels {sorted(corners[s] for s in faceCorners)} is shared by more "
                    f"than two root elements ({owners}); the root mesh is not a manifold"
                )

    def key(self, rootEid: int, xi) -> tuple:
        """The unique key of the exact reference point ``xi`` (three :class:`~fractions.Fraction` in
        [-1, 1]) of root element ``rootEid``: the root entity it lies on and its position there."""
        corners = self._rootCorners[rootEid]
        component = self._rootComponent[rootEid]
        onBoundary = [axis for axis in range(3) if abs(xi[axis]) == 1]
        if len(onBoundary) == 3:
            slot = next(s for s in self._cornerSlots if self._cornerReference[s] == tuple(xi))
            return ("vertex", component, corners[slot])
        if len(onBoundary) == 2:
            a, b = next((sa, sb) for sa, sb in self._edgeCorners if self._onEntity(xi, (sa, sb)))
            if corners[a] > corners[b]:
                a, b = b, a
            return ("edge", component, (corners[a], corners[b]), self._edgeParameter(xi, a, b))
        if len(onBoundary) == 1:
            faceCorners = next(fc for fc in self._faceCorners if self._onEntity(xi, fc))
            origin, first, second = self._canonicalFaceFrame(faceCorners, corners)
            labels = tuple(sorted(corners[s] for s in faceCorners))
            return ("face", component, labels, self._faceCoordinates(xi, origin, first, second))
        return ("interior", component, rootEid, tuple(xi))

    def roots_sharing(self, key) -> list:
        """The root elements whose closure contains the entity of ``key``."""
        kind = key[0]
        if kind == "interior":
            return [key[2]]
        if kind == "face":
            return list(self._facesOfRoots[(key[1], key[2])])
        if kind == "vertex":
            return list(self._rootsOfCorner[(key[1], key[2])])
        a, b = key[2]
        withBoth = set(self._rootsOfCorner[(key[1], a)]) & set(self._rootsOfCorner[(key[1], b)])
        return sorted(root for root in withBoth if self._isEdgeOf(root, key[2]))

    def local_point(self, rootEid: int, key) -> tuple:
        """The exact reference point, in root element ``rootEid``, of a key whose entity lies in that
        root's closure (see :meth:`roots_sharing`)."""
        corners = self._rootCorners[rootEid]
        slotOf = {label: slot for slot, label in corners.items()}
        kind = key[0]
        if kind == "interior":
            return key[3]
        if kind == "vertex":
            return self._cornerReference[slotOf[key[2]]]
        if kind == "edge":
            a, b = (self._cornerReference[slotOf[label]] for label in key[2])
            t = key[3]
            return tuple(a[k] + t * (b[k] - a[k]) for k in range(3))
        faceCorners = next(fc for fc in self._faceCorners if sorted(corners[s] for s in fc) == list(key[2]))
        origin, first, second = (self._cornerReference[s] for s in self._canonicalFaceFrame(faceCorners, corners))
        u, v = key[3]
        return tuple(origin[k] + u * (first[k] - origin[k]) + v * (second[k] - origin[k]) for k in range(3))

    # ---- internals ----
    def _onEntity(self, xi, cornerSlots) -> bool:
        """True if ``xi`` lies on the reference entity spanned by ``cornerSlots``: on every axis on
        which all of its corners agree, ``xi`` has that same value."""
        refs = [self._cornerReference[s] for s in cornerSlots]
        return all(xi[k] == refs[0][k] for k in range(3) if all(r[k] == refs[0][k] for r in refs))

    def _isEdgeOf(self, rootEid, labels) -> bool:
        corners = self._rootCorners[rootEid]
        return any({corners[a], corners[b]} == set(labels) for a, b in self._edgeCorners)

    def _edgeParameter(self, xi, a, b) -> Fraction:
        """Exact position along the reference edge from corner ``a`` (0) to corner ``b`` (1)."""
        ra, rb = self._cornerReference[a], self._cornerReference[b]
        axis = next(k for k in range(3) if ra[k] != rb[k])
        return (xi[axis] - ra[axis]) / (rb[axis] - ra[axis])

    def _canonicalFaceFrame(self, faceCorners, corners):
        """The face's frame fixed by its labels: origin = the corner with the smallest label, first axis
        towards its loop neighbour with the smaller label, second axis towards the other neighbour."""
        k = min(range(4), key=lambda i: corners[faceCorners[i]])
        previous, following = faceCorners[(k - 1) % 4], faceCorners[(k + 1) % 4]
        first, second = (previous, following) if corners[previous] < corners[following] else (following, previous)
        return faceCorners[k], first, second

    def _faceCoordinates(self, xi, origin, first, second) -> tuple:
        """Exact coordinates of ``xi`` in the face frame; the frame's axes are orthogonal reference
        edges of length 2."""
        o, e1, e2 = (self._cornerReference[s] for s in (origin, first, second))
        d = [xi[k] - o[k] for k in range(3)]
        u = sum(d[k] * (e1[k] - o[k]) for k in range(3)) / 4
        v = sum(d[k] * (e2[k] - o[k]) for k in range(3)) / 4
        return (u, v)
