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
#  This library is free software; you can redistribute it and/or
#  modify it under the terms of the GNU Lesser General Public
#  License as published by the Free Software Foundation; either
#  version 2.1 of the License, or (at your option) any later version.
#
#  The full text of the license can be found in the file LICENSE.md at
#  the top level directory of EdelweissFE.
#  ---------------------------------------------------------------------
"""Tests for ``edelweissfe.numerics.p1topology.buildP1Map`` -- the corner/midside
classification underlying the p-multigrid P1 restriction operator.

Minimal stub objects mirror the real ``Node``/``Element``/``NodeField``/``Model`` interface
(``.nodes``, ``.fields``, ``.nSpatialDimensions``) rather than constructing real Marmot elements,
so this stays pure-Python and Cython/Marmot-free.
"""

import numpy as np
import pytest

from edelweissfe.numerics.p1topology import buildP1Map, classifyElementTopology


class _FakeNode:
    def __init__(self, label, fields):
        self.label = label
        self.fields = fields  # dict: fieldName -> anything (only membership is checked)


class _FakeElement:
    def __init__(self, nodes, nSpatialDimensions=None, fieldNames=("displacement",)):
        self.nodes = nodes
        self.fields = [list(fieldNames) for _ in nodes]
        if nSpatialDimensions is not None:
            self.nSpatialDimensions = nSpatialDimensions
        # else: leave the attribute entirely absent, mirroring a contact facet


class _FakeNodeField:
    def __init__(self, nodes):
        self.nodes = nodes


class _FakeModel:
    def __init__(self, nodeFields, elements):
        self.nodeFields = nodeFields
        self.elements = {i: e for i, e in enumerate(elements)}


def test_quad8_isolated_element():
    # local numbering: 0,1,2,3 corners; 4,5,6,7 midsides of edges (0,1),(1,2),(2,3),(3,0)
    nodes = [_FakeNode(i, {"displacement": True}) for i in range(8)]
    element = _FakeElement(nodes, nSpatialDimensions=2)
    model = _FakeModel({"displacement": _FakeNodeField(nodes)}, [element])

    isCorner, edgeEndpoints, warnings = buildP1Map(model, "displacement")
    assert list(np.nonzero(isCorner)[0]) == [0, 1, 2, 3]
    assert set(edgeEndpoints[4].tolist()) == {0, 1}
    assert set(edgeEndpoints[5].tolist()) == {1, 2}
    assert set(edgeEndpoints[6].tolist()) == {2, 3}
    assert set(edgeEndpoints[7].tolist()) == {3, 0}
    assert (edgeEndpoints[:4] == -1).all()
    assert warnings == []


def test_hexa20_isolated_element():
    nodes = [_FakeNode(i, {"displacement": True}) for i in range(20)]
    element = _FakeElement(nodes, nSpatialDimensions=3)
    model = _FakeModel({"displacement": _FakeNodeField(nodes)}, [element])

    isCorner, edgeEndpoints, warnings = buildP1Map(model, "displacement")
    assert list(np.nonzero(isCorner)[0]) == list(range(8))
    expectedEdges = {
        8: (0, 1),
        9: (1, 2),
        10: (2, 3),
        11: (3, 0),
        12: (4, 5),
        13: (5, 6),
        14: (6, 7),
        15: (7, 4),
        16: (0, 4),
        17: (1, 5),
        18: (2, 6),
        19: (3, 7),
    }
    for mid, (a, b) in expectedEdges.items():
        assert set(edgeEndpoints[mid].tolist()) == {a, b}


def test_amr_corner_wins_over_midside_from_a_different_element():
    """A hanging node can be a midside of a coarse element and a corner of a fine one -- corner
    status must win, regardless of which element is processed first."""
    nodes = [_FakeNode(i, {"displacement": True}) for i in range(8)]
    quad8 = _FakeElement(nodes, nSpatialDimensions=2)

    sharedMidNode = nodes[4]
    linearNodes = [sharedMidNode] + [_FakeNode(100 + i, {"displacement": True}) for i in range(3)]
    linearElement = _FakeElement(linearNodes, nSpatialDimensions=2)  # 4 nodes -> pure-corner

    allNodes = nodes + linearNodes[1:]
    model = _FakeModel({"displacement": _FakeNodeField(allNodes)}, [quad8, linearElement])

    isCorner, edgeEndpoints, warnings = buildP1Map(model, "displacement")
    rowOf4 = allNodes.index(sharedMidNode)
    assert isCorner[rowOf4], "corner status must win over midside status from a different element"
    assert (edgeEndpoints[rowOf4] == -1).all(), "a corner row must not carry edge endpoints"


def test_contact_facet_without_nspatialdimensions_is_pure_corner():
    facetNodes = [_FakeNode(200 + i, {"displacement": True}) for i in range(3)]
    facetElement = _FakeElement(facetNodes, nSpatialDimensions=None)
    model = _FakeModel({"displacement": _FakeNodeField(facetNodes)}, [facetElement])

    isCorner, edgeEndpoints, warnings = buildP1Map(model, "displacement")
    assert isCorner.all()
    assert (edgeEndpoints == -1).all()


def test_unrecognized_topology_raises():
    with pytest.raises(ValueError):
        classifyElementTopology(3, 10)  # e.g. a Tet10 -- not a handled quadratic family

    weirdNodes = [_FakeNode(300 + i, {"displacement": True}) for i in range(10)]
    weirdElement = _FakeElement(weirdNodes, nSpatialDimensions=3)
    model = _FakeModel({"displacement": _FakeNodeField(weirdNodes)}, [weirdElement])
    with pytest.raises(ValueError):
        buildP1Map(model, "displacement")


def test_inconsistent_edge_endpoints_falls_back_to_corner():
    """Two elements disagreeing about a shared midside's edge endpoints is a real pattern found on
    the reference pryout model, at a 2:1 non-conforming AMR refinement boundary -- the
    node is conservatively treated as a corner (always structurally safe for P1) rather than
    hard-erroring or arbitrarily keeping one element's guess, and the fallback is reported via
    the returned `warnings`, not swallowed silently."""
    nodes = [_FakeNode(i, {"displacement": True}) for i in range(8)]
    quad8 = _FakeElement(nodes, nSpatialDimensions=2)  # midside 4 -> edge (nodes[0], nodes[1])

    # a second Quad8 placing the SAME midside node (nodes[4]) at its own local index 4, but with
    # different corners at local 0/1 -- its midside-4 edge now resolves to a different pair.
    otherCorners = [_FakeNode(50 + i, {"displacement": True}) for i in range(4)]
    otherNodes = otherCorners + [nodes[4]] + [_FakeNode(60 + i, {"displacement": True}) for i in range(3)]
    conflicting = _FakeElement(otherNodes, nSpatialDimensions=2)

    allNodes = nodes + otherCorners + otherNodes[5:]
    model = _FakeModel({"displacement": _FakeNodeField(allNodes)}, [quad8, conflicting])

    isCorner, edgeEndpoints, warnings = buildP1Map(model, "displacement")
    rowOf4 = allNodes.index(nodes[4])
    assert isCorner[rowOf4], "a genuine edge-endpoint conflict must fall back to corner"
    assert (edgeEndpoints[rowOf4] == -1).all()
    assert len(warnings) == 1 and "conflicting edge-endpoint" in warnings[0]
