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
"""Hanging nodes on a CURVED and WARPED coarse face -- the normal case in an unstructured (Cubit)
HEX20 mesh -- must all be found and tied to the coarse face's exact quadratic trace.

The face-interior hanging nodes used to be classified with a flat test against the plane through three
of the face's corners (absolute tolerance 1e-8), and their weights came from a bilinear inverse on
the corner quad. On a curved or warped face every face-interior node failed the plane test and was left
unconstrained -- a non-conforming coarse-fine interface -- while nodes on the coarse edges (a quadratic
edge test) were found. Node-set inheritance used the same flat test, so such nodes also never joined a
boundary node set of that face.
"""

import numpy as np
import pytest
from _adaptivemeshbuilder import AdaptiveMeshBuilder

from edelweissfe.adaptivity.hex20shapefunctions import hex20_box_coords, hex20_shape
from edelweissfe.adaptivity.hex20topology import Hex20Topology

TOPOLOGY = Hex20Topology()
_CORNER_SLOTS = set(TOPOLOGY.corner_slots)


def _elementsSharingACurvedWarpedFace():
    """A = [-2, 0] x [0, 2]^2 and B = [0, 2] x [0, 2]^2 share the face x = 0; its 8 nodes are moved in x,
    identically for both elements: the four midside nodes bulge (curved), one corner is pushed out of
    the other three's plane (warped)."""
    A = hex20_box_coords(-2.0, 0.0, 0.0, 2.0, 0.0, 2.0)
    B = hex20_box_coords(0.0, 2.0, 0.0, 2.0, 0.0, 2.0)
    shift = {}
    for x in A:
        if abs(x[0]) < 1e-12:
            key = (round(x[1], 9), round(x[2], 9))
            isCorner = key[0] in (0.0, 2.0) and key[1] in (0.0, 2.0)
            shift[key] = 0.03 if key == (2.0, 2.0) else (0.0 if isCorner else 0.05 + 0.01 * key[0])
    for X in (A, B):
        for x in X:
            if abs(x[0]) < 1e-12:
                x[0] += shift[(round(x[1], 9), round(x[2], 9))]
    return A, B


def _refinedNeighbourMesh():
    A, B = _elementsSharingACurvedWarpedFace()
    builder = AdaptiveMeshBuilder(2)
    mesh = builder.mesh
    coarse = builder.addRoot(A)
    fine = builder.addRoot(B)
    sharedFace = [lab for lab, x in zip(mesh.elements[fine]["conn"], B) if abs(x[0]) < 0.1]
    mesh.define_node_set("sharedFaceOfB", sharedFace)
    mesh.refine(fine)
    return mesh, coarse, fine, A


def _newNodesOnSharedFace(mesh, coarse, A):
    """Nodes of B's children on the shared face that are not nodes of A -- found exactly, from the
    topological node keys: on the root face whose corners are A's (and B's) shared corners, or on one
    of its edges."""
    nodesOfA = mesh.elements[coarse]["conn"]
    sharedCorners = {nodesOfA[i] for i in range(20) if abs(A[i][0]) < 0.1 and i in _CORNER_SLOTS}
    onFace = set()
    for eid in mesh.active():
        if eid == coarse:
            continue
        for label, key in zip(mesh.elements[eid]["conn"], mesh.node_keys(eid)):
            labels = {key.entity} if key.kind == "vertex" else set(key.entity) if key.kind != "interior" else set()
            if label not in nodesOfA and labels and labels <= sharedCorners:
                onFace.add(label)
    return onFace


def test_theSharedFaceIsReallyCurvedAndWarped():
    """Guard: the fixture defeats a flat corner-plane test by far more than its 1e-8 tolerance."""
    A, _ = _elementsSharingACurvedWarpedFace()
    face = next(f for f in TOPOLOGY.faces if all(abs(A[i][0]) < 0.1 for i in f))
    c = A[list(face[:4])]
    n = np.cross(c[1] - c[0], c[3] - c[0])
    n /= np.linalg.norm(n)
    assert abs(np.dot(c[2] - c[0], n)) > 1e-3  # warped: the fourth corner is off the plane
    assert max(abs(np.dot(A[i] - c[0], n)) for i in face[4:]) > 1e-2  # curved: midside nodes off it


def test_weightsReproduceTheSlavesOnTheCurvedCoarseTrace():
    """The weights are the coarse element's own shape functions at the slave: they reproduce its
    position on the curved face exactly (and hence any displacement field of the coarse element)."""
    mesh, coarse, _, A = _refinedNeighbourMesh()
    coords = mesh.registry.coordinates
    records = mesh.hanging_mpc_records()
    assert records
    for slave, masters in records.items():
        reproduced = sum(w * coords[m] for m, w in masters)
        np.testing.assert_allclose(reproduced, coords[slave], atol=1e-12)
        assert sum(w for _, w in masters) == pytest.approx(1.0, abs=1e-12)
    # and a quadratic displacement field of the coarse element is carried over exactly
    u = np.array([[1e-3 * x[1] ** 2 - 2e-3 * x[2], 5e-4 * x[1] * x[2], -1e-3 * x[0] * x[2]] for x in A])
    labelIndex = {lab: i for i, lab in enumerate(mesh.elements[coarse]["conn"])}
    for slave, masters in records.items():
        xi = TOPOLOGY.inverse_map(coords[slave], A)
        np.testing.assert_allclose(sum(w * u[labelIndex[m]] for m, w in masters), hex20_shape(*xi) @ u, atol=1e-14)


def test_newNodesOnACurvedWarpedFaceJoinItsNodeSet():
    mesh, coarse, _, A = _refinedNeighbourMesh()
    onFace = _newNodesOnSharedFace(mesh, coarse, A)
    missing = onFace - mesh.nodeSets["sharedFaceOfB"]
    assert not missing, f"{len(missing)} new face nodes did not join the face's node set"
