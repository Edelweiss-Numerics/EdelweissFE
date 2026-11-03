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
"""Pins :class:`~edelweissfe.adaptivity.rootentities.RootEntityTable`: a point on a face, edge or
vertex shared by two root elements gets the identical key from either root, for every relative
orientation of their local numbering -- all 48 symmetries of the cube -- because root entities are
named by corner labels and positions are measured in a frame those labels fix."""

import itertools
from fractions import Fraction

import numpy as np
import pytest

from edelweissfe.adaptivity.hex20topology import Hex20Topology
from edelweissfe.adaptivity.rootentities import RootEntityTable

TOPOLOGY = Hex20Topology()
REFERENCE = TOPOLOGY.reference_node_lattice - 1  # own nodes at reference coordinates -1, 0, 1
M = 4  # lattice units: reference coordinate xi is the lattice integer (xi + 1) * M


def _cubeSymmetries():
    """All 48 signed permutation matrices (rotations and reflections of the reference cube)."""
    for perm in itertools.permutations(range(3)):
        for signs in itertools.product((-1, 1), repeat=3):
            S = np.zeros((3, 3), dtype=int)
            for row, (col, sign) in enumerate(zip(perm, signs)):
                S[row, col] = sign
            yield S


def _toLattice(xi):
    return tuple(int((Fraction(v) + 1) * M) for v in xi)


def _twoRootsSharingAFace(S):
    """Root A covers [-1, 1]^3 of a global reference frame, root B covers [1, 3] x [-1, 1]^2 and
    numbers its nodes through the cube symmetry ``S``: its local node p sits at (2, 0, 0) + S p. Nodes
    at one global point share one label."""
    labelAt = {}

    def label(point):
        return labelAt.setdefault(tuple(point), len(labelAt) + 1)

    table = RootEntityTable(TOPOLOGY, M)
    table.add_root(1, [label(p) for p in REFERENCE])
    table.add_root(2, [label(np.array([2, 0, 0]) + S @ p) for p in REFERENCE])
    return table


@pytest.mark.parametrize("S", list(_cubeSymmetries()), ids=lambda S: "".join(map(str, S.flatten())))
def test_bothRootsGiveTheSharedPointsIdenticalKeys(S):
    table = _twoRootsSharingAFace(S)
    Sinv = np.linalg.inv(S).round().astype(int)
    quarters = [Fraction(k, 4) for k in range(-4, 5)]
    for y, z in itertools.product(quarters, quarters):
        inA = (Fraction(1), y, z)
        inB = tuple(Sinv @ (np.array(inA, dtype=object) - np.array([2, 0, 0], dtype=object)))
        keyA, keyB = table.key(1, _toLattice(inA)), table.key(2, _toLattice(inB))
        assert keyA == keyB
        onEdgeOfFace = abs(y) == 1 or abs(z) == 1
        assert keyA.kind == ("vertex" if abs(y) == 1 and abs(z) == 1 else "edge" if onEdgeOfFace else "face")
        assert table.local_point(1, keyA) == _toLattice(inA)
        assert table.local_point(2, keyA) == _toLattice(inB)
        assert sorted(table.roots_sharing(keyA)) == [1, 2]


def test_collapsedAndNonManifoldRootsAreRejected():
    table = RootEntityTable(TOPOLOGY, M)
    conn = list(range(1, 21))
    conn[1] = conn[0]
    with pytest.raises(ValueError, match="collapsed"):
        table.add_root(1, conn)
    # three roots owning the same face (the same four corner labels) is not a manifold
    table = RootEntityTable(TOPOLOGY, M)
    table.add_root(1, list(range(1, 21)))
    face = TOPOLOGY.faces[0]
    table.add_root(2, [slot + 1 if slot in face else 200 + slot for slot in range(20)])
    with pytest.raises(ValueError, match="manifold"):
        table.add_root(3, [slot + 1 if slot in face else 300 + slot for slot in range(20)])
