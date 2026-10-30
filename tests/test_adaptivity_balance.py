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
"""2:1 balancing decides face adjacency exactly (see ``AdaptiveMesh._facesOnEntities``): on a curved and
warped shared face -- where the former flat-plane test found no shared face and silently skipped
balancing -- a coarse element next to a two-levels-finer neighbour is refined, as on a flat face.
Every balanced mesh also passes the exact conformity check."""

import pytest
from _adaptivemeshbuilder import AdaptiveMeshBuilder

from edelweissfe.adaptivity.hex20shapefunctions import hex20_box_coords


def _twoRoots(curvedWarped, splitFactor=2):
    A = hex20_box_coords(-2.0, 0.0, 0.0, 2.0, 0.0, 2.0)
    B = hex20_box_coords(0.0, 2.0, 0.0, 2.0, 0.0, 2.0)
    if curvedWarped:
        for X in (A, B):
            for x in X:
                if abs(x[0]) < 1e-12:
                    isCorner = x[1] in (0.0, 2.0) and x[2] in (0.0, 2.0)
                    x[0] += 0.03 if (x[1], x[2]) == (2.0, 2.0) else (0.0 if isCorner else 0.05 + 0.01 * x[1])
    builder = AdaptiveMeshBuilder(splitFactor)
    return builder.mesh, builder.addRoot(A), builder.addRoot(B)


def _childOfBTouchingA(mesh, b):
    """A child of B on its face x = 0 (the face shared with A): its box starts at B's low x."""
    kids = mesh.elements[b]["children"]
    return next(
        k for k in kids if mesh.elements[k]["referenceBoxOrigin"][0] == mesh.elements[b]["referenceBoxOrigin"][0]
    )


@pytest.mark.parametrize("curvedWarped", [False, True])
@pytest.mark.parametrize("splitFactor", [2, 3])
def test_aCoarseRootNextToATwoLevelsFinerNeighbourIsRefined(curvedWarped, splitFactor):
    mesh, a, b = _twoRoots(curvedWarped, splitFactor)
    mesh.refine(b)
    mesh.refine(_childOfBTouchingA(mesh, b))  # level 2 now touches the level-0 root A
    assert mesh.balance_2to1() >= 1
    assert not mesh.elements[a]["active"], "A was not refined: its two-levels-finer face neighbour was missed"
    mesh.check_conformity(mesh.hanging_mpc_records())


def test_aNeighbourOnlyOneLevelFinerIsLeftAlone():
    mesh, a, b = _twoRoots(True)
    mesh.refine(b)
    assert mesh.balance_2to1() == 0
    assert mesh.elements[a]["active"]


def test_aTwoLevelJumpInsideOneRootIsBalanced():
    builder = AdaptiveMeshBuilder(2)
    mesh = builder.mesh
    root = builder.addRoot(hex20_box_coords(0.0, 1.0, 0.0, 1.0, 0.0, 1.0))
    kids = mesh.refine(root)
    grandchildren = mesh.refine(kids[0])  # octant (0, 0, 0)
    mesh.refine(grandchildren[-1])  # its octant touching the siblings of kids[0]: level 3 next to level 1
    assert mesh.balance_2to1() >= 1
    mesh.check_conformity(mesh.hanging_mpc_records())
    levels = {mesh.elements[e]["level"] for e in mesh.active()}
    assert levels == {1, 2, 3}
