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
"""Pins the topological node keys of :class:`~edelweissfe.adaptivity.rootentities.RootEntityTable`
and the exact conformity check :meth:`AdaptiveMesh.check_hanging_completeness` built on them.

Keys: a point on a face (edge, vertex) shared by two root elements gets the identical key from
either root, for every relative orientation of the two elements' local numbering -- all 48 symmetries
of the cube -- because entities are named by corner labels and measured in a frame those labels fix.

Conformity check: exact, no tolerance. It must accept a correct mesh (flat faces, splitFactor 2 and 3,
two-level chains, a curved and warped face) and reject one from which a single slave was dropped --
including exactly the five face-interior slaves the former geometric classifier dropped on a curved
and warped face.
"""

import itertools
from fractions import Fraction

import numpy as np
import pytest

from edelweissfe.adaptivity.hex20shapefunctions import hex20_box_coords
from edelweissfe.adaptivity.hex20topology import Hex20Topology
from edelweissfe.adaptivity.refinement import AdaptiveMesh
from edelweissfe.adaptivity.rootentities import RootEntityTable
from edelweissfe.utils.exceptions import TopologyError

TOPOLOGY = Hex20Topology()
REFERENCE = np.rint(TOPOLOGY.reference_node_param()).astype(int)


def _cubeSymmetries():
    """All 48 signed permutation matrices (rotations and reflections of the reference cube)."""
    for perm in itertools.permutations(range(3)):
        for signs in itertools.product((-1, 1), repeat=3):
            S = np.zeros((3, 3), dtype=int)
            for row, (col, sign) in enumerate(zip(perm, signs)):
                S[row, col] = sign
            yield S


def _twoRootsSharingAFace(S):
    """Root A covers [-1, 1]^3 of a global reference frame, root B covers [1, 3] x [-1, 1]^2 and
    numbers its nodes through the cube symmetry ``S``: its local node p sits at (2, 0, 0) + S p.
    Nodes at the same global point share one label."""
    labelAt = {}

    def label(point):
        return labelAt.setdefault(tuple(point), len(labelAt) + 1)

    connA = [label(p) for p in REFERENCE]
    connB = [label(np.array([2, 0, 0]) + S @ p) for p in REFERENCE]
    table = RootEntityTable(TOPOLOGY)
    table.add_root(1, connA)
    table.add_root(2, connB)
    return table


@pytest.mark.parametrize("S", list(_cubeSymmetries()), ids=lambda S: "".join(map(str, S.flatten())))
def test_bothRootsGiveTheSharedPointsIdenticalKeys(S):
    table = _twoRootsSharingAFace(S)
    Sinv = np.linalg.inv(S).round().astype(int)
    quarters = [Fraction(k, 4) for k in range(-4, 5)]
    for y, z in itertools.product(quarters, quarters):
        inA = (Fraction(1), y, z)
        global_ = np.array([Fraction(1), y, z], dtype=object)
        inB = tuple(Sinv @ (global_ - np.array([2, 0, 0], dtype=object)))
        keyA, keyB = table.key(1, inA), table.key(2, tuple(Fraction(v) for v in inB))
        assert keyA == keyB
        assert keyA[0] == ("face" if abs(y) < 1 and abs(z) < 1 else "edge" if abs(y) < 1 or abs(z) < 1 else "vertex")
        # and the key maps back to the same point in either root
        assert table.local_point(1, keyA) == inA
        assert sorted(table.roots_sharing(keyA)) == [1, 2]


def test_collapsedAndNonManifoldRootsAreRejected():
    table = RootEntityTable(TOPOLOGY)
    conn = list(range(1, 21))
    conn[1] = conn[0]
    with pytest.raises(ValueError, match="collapsed"):
        table.add_root(1, conn)
    # three roots owning the same face (same four corner labels) is not a manifold
    table = RootEntityTable(TOPOLOGY)
    table.add_root(1, list(range(1, 21)))
    face = next(f for f in TOPOLOGY.faces if set(f[:4]) <= set(range(8)))
    shared = {slot for slot in face[:4]}
    table.add_root(2, [slot + 1 if slot in shared else 200 + slot for slot in range(20)])
    with pytest.raises(ValueError, match="manifold"):
        table.add_root(3, [slot + 1 if slot in shared else 300 + slot for slot in range(20)])


# ---- the conformity check ----


def _boxMesh(splitFactor=2, curvedWarped=False, refineTwice=False):
    """A = [-2, 0] x [0, 2]^2 (coarse), B = [0, 2] x [0, 2]^2 (refined); optionally with the shared
    face bulged and warped, and optionally with one child of B refined again (a 2:1 chain)."""
    A = hex20_box_coords(-2.0, 0.0, 0.0, 2.0, 0.0, 2.0)
    B = hex20_box_coords(0.0, 2.0, 0.0, 2.0, 0.0, 2.0)
    if curvedWarped:
        for X in (A, B):
            for x in X:
                if abs(x[0]) < 1e-12:
                    isCorner = x[1] in (0.0, 2.0) and x[2] in (0.0, 2.0)
                    x[0] += 0.03 if (x[1], x[2]) == (2.0, 2.0) else (0.0 if isCorner else 0.05 + 0.01 * x[1])
    mesh = AdaptiveMesh(splitFactor=splitFactor, topology=Hex20Topology())
    mesh.add_root(A, 0)
    kids = mesh.refine(mesh.add_root(B, 0))
    if refineTwice:
        mesh.refine(kids[-1])  # a child away from A: a level-2 / level-1 interface inside B
    return mesh


@pytest.mark.parametrize("splitFactor", [2, 3])
def test_aCorrectMeshPasses(splitFactor):
    mesh = _boxMesh(splitFactor)
    mesh.check_hanging_completeness(mesh.hanging_mpc_records())


def test_aTwoLevelChainPasses():
    mesh = _boxMesh(refineTwice=True)
    mesh.check_hanging_completeness(mesh.hanging_mpc_records())


def test_droppingASingleSlaveIsCaught():
    mesh = _boxMesh()
    records = mesh.hanging_mpc_records()
    dropped = sorted(records)[0]
    with pytest.raises(TopologyError, match="not\\s+conforming"):
        mesh.check_hanging_completeness(set(records) - {dropped})


def test_aCurvedWarpedMeshIsConforming():
    """On the curved and warped face -- where the former flat corner-plane classifier left the five
    face-interior hanging nodes unconstrained -- the topological classifier constrains every one."""
    mesh = _boxMesh(curvedWarped=True)
    records = mesh.hanging_mpc_records()
    mesh.check_hanging_completeness(records)
    assert len(records) == 13


def test_theCheckCatchesTheFormerCurvedFaceDefect():
    """The check needs no geometry: dropping the five face-interior slaves -- exactly what the former
    classifier did on this face -- is caught."""
    mesh = _boxMesh(curvedWarped=True)
    faceSlaves = {h["slave"] for h in mesh.classify_hanging() if h["kind"] == "face"}
    assert len(faceSlaves) == 5
    with pytest.raises(TopologyError, match=r"5 node\(s\) lie on the boundary"):
        mesh.check_hanging_completeness(set(mesh.hanging_mpc_records()) - faceSlaves)
