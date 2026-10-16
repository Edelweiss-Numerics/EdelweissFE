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
"""Node-set inheritance on refinement is decided exactly, from each new node's lattice position in its
parent: a new node joins a node set iff it lies on a parent face or edge whose nodes are all in the set.
No coordinates are compared, so a curved or warped boundary face behaves like a flat one."""

import numpy as np
import pytest
from _adaptivemeshbuilder import AdaptiveMeshBuilder

from edelweissfe.adaptivity.hex20shapefunctions import hex20_box_coords
from edelweissfe.adaptivity.hex20topology import Hex20Topology
from edelweissfe.adaptivity.refinement import AdaptiveMesh

TOPOLOGY = Hex20Topology()


def _singleRoot(splitFactor, warp=False):
    X = hex20_box_coords(0.0, 1.0, 0.0, 1.0, 0.0, 1.0)
    if warp:  # bulge and twist the face x = 1: must not matter
        X[np.isclose(X[:, 0], 1.0), 0] += 0.07 * X[np.isclose(X[:, 0], 1.0), 1]
    builder = AdaptiveMeshBuilder(splitFactor)
    root = builder.addRoot(X)
    return builder.mesh, root, X


def _nodesOnRootEntity(mesh, labels):
    """Active nodes whose exact topological key lies on the root entity spanned by ``labels``."""
    found = set()
    for eid in mesh.active():
        for label, key in zip(mesh.elements[eid]["conn"], mesh.node_keys(eid)):
            spanned = {key.entity} if key.kind == "vertex" else set(key.entity) if key.kind != "interior" else None
            if spanned is not None and spanned <= labels:
                found.add(label)
    return found


@pytest.mark.parametrize("splitFactor", [2, 3])
@pytest.mark.parametrize("warp", [False, True])
def test_aFaceSetGainsExactlyTheNewNodesOnThatFace(splitFactor, warp):
    mesh, root, X = _singleRoot(splitFactor, warp)
    conn = mesh.elements[root]["conn"]
    face = next(
        f for f in TOPOLOGY.faces if all(abs(np.rint(TOPOLOGY.reference_node_param()[i][0]) - 1) == 0 for i in f)
    )
    mesh.define_node_set("face", [conn[i] for i in face])
    mesh.refine(root)
    faceCorners = {conn[i] for i in face[:4]}
    expected = _nodesOnRootEntity(mesh, faceCorners)
    assert mesh.nodeSets["face"] == expected
    n = 2 * splitFactor + 1  # nodes per edge of the refined face
    assert len(expected) == n * n - (splitFactor * splitFactor)  # serendipity: no centre node per sub-face


def test_anEdgeSetGainsOnlyTheNewNodesOnThatEdge():
    mesh, root, _ = _singleRoot(2)
    conn = mesh.elements[root]["conn"]
    edge = TOPOLOGY.edges[0]
    mesh.define_node_set("edge", [conn[i] for i in edge])
    mesh.refine(root)
    assert len(mesh.nodeSets["edge"]) == 5
    assert mesh.nodeSets["edge"] == _nodesOnRootEntity(mesh, {conn[edge[0]], conn[edge[2]]})


def test_anIncompleteFacePassesOnlyItsCompleteEdges():
    """A set holding a face minus one corner node is not a face set: new face-interior nodes must not
    join it; new nodes on the two edges that are still complete must."""
    mesh, root, _ = _singleRoot(2)
    conn = mesh.elements[root]["conn"]
    face = TOPOLOGY.faces[0]
    dropped = face[0]
    mesh.define_node_set("partial", [conn[i] for i in face if i != dropped])
    mesh.refine(root)
    completeEdges = [e for e in TOPOLOGY.edges if set(e) <= set(face) and dropped not in e]
    expected = {conn[i] for i in face if i != dropped}
    for e in completeEdges:
        expected |= _nodesOnRootEntity(mesh, {conn[e[0]], conn[e[2]]})
    assert mesh.nodeSets["partial"] == expected


def test_elementSetsAndSurfacesAreInheritedByTheChildren():
    """Children join every element set of their parent, and an element-based surface on a parent face
    passes to exactly the children tiling that face, with the same face id."""
    builder = AdaptiveMeshBuilder(2)
    mesh = builder.mesh
    root = builder.addRoot(hex20_box_coords(0.0, 1.0, 0.0, 1.0, 0.0, 1.0))
    mesh.define_element_set("body", [root])
    faceId = 1
    mesh.define_surface("top", [(root, faceId)])
    kids = mesh.refine(root)
    assert mesh.elementSets["body"] == {root, *kids}
    tiling = {kids[j] for j in TOPOLOGY.face_child_indices(TOPOLOGY.faceid_to_face[faceId], 2)}
    assert mesh.surfaces["top"] == {(kid, faceId) for kid in tiling}
    assert len(tiling) == 4


def test_refiningAnInactiveElementReturnsItsChildren():
    mesh = AdaptiveMesh(splitFactor=2)  # the default topology is HEX20
    builder = AdaptiveMeshBuilder(2)
    root = builder.addRoot(hex20_box_coords(0.0, 1.0, 0.0, 1.0, 0.0, 1.0))
    kids = builder.mesh.refine(root)
    assert builder.mesh.refine(root) == kids
    assert isinstance(mesh.topology, Hex20Topology)


def test_aNodeSetAwayFromTheRefinedElementIsUnchanged():
    builder = AdaptiveMeshBuilder(2)
    mesh = builder.mesh
    root = builder.addRoot(hex20_box_coords(0.0, 1.0, 0.0, 1.0, 0.0, 1.0))
    other = builder.addRoot(hex20_box_coords(5.0, 6.0, 0.0, 1.0, 0.0, 1.0))
    elsewhere = set(mesh.elements[other]["conn"])
    mesh.define_node_set("elsewhere", elsewhere)
    mesh.refine(root)
    assert mesh.nodeSets["elsewhere"] == elsewhere
