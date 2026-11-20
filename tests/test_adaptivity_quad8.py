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
"""QUAD8 (2D) adaptive refinement, mirroring the HEX20 tests: the shape functions, conformity and
exact hanging-node weights (split factors 2 and 3, a curved shared edge, a multi-level chain), node
identity across differently numbered neighbours, 2:1 balance, node-set and surface inheritance, and
the root-mesh entity keys."""

from fractions import Fraction

import numpy as np
import pytest
from _adaptivemeshbuilder import AdaptiveMeshBuilder

from edelweissfe.adaptivity.hex20topology import Hex20Topology
from edelweissfe.adaptivity.marking import RecoveryErrorMarker
from edelweissfe.adaptivity.quad8topology import Quad8Topology
from edelweissfe.adaptivity.rootentities import RootEntityTable
from edelweissfe.adaptivity.statetransfer.projection import PolynomialProjection
from edelweissfe.elements._quad2dnodeordering import quad8NodeXi
from edelweissfe.utils.exceptions import TopologyError

TOPOLOGY = Quad8Topology()
LOCAL_COORDS = quad8NodeXi.astype(float)
quad8_shape = TOPOLOGY.shape_functions


def quad8_box_coords(x0, x1, y0, y1):
    """The 8 node coordinates of an axis-aligned QUAD8 box, in CPS8 order."""
    return np.array([x0, y0]) + (LOCAL_COORDS + 1) / 2 * np.array([x1 - x0, y1 - y0])


def _builder(splitFactor=2):
    return AdaptiveMeshBuilder(splitFactor, topology=TOPOLOGY)


def _grid(builder, nX, nY, curved=False):
    """An nX x nY grid of unit QUAD8 roots; ``curved`` bulges every vertical interior edge."""
    eids = {}
    for i in range(nX):
        for j in range(nY):
            coords = quad8_box_coords(i, i + 1, j, j + 1)
            if curved:
                # midsides 5 (x = i + 1) and 7 (x = i) of a vertical edge, bulged consistently on both sides
                for slot, x in ((5, i + 1), (7, i)):
                    if 0 < x < nX:
                        coords[slot, 0] += 0.15 * (-1) ** x
            eids[(i, j)] = builder.addRoot(coords)
    return eids


def test_shape_functions_are_the_quad8_serendipity_basis():
    points = LOCAL_COORDS.tolist() + [[0.3, -0.7], [0.0, 0.0], [-0.55, 0.9]]
    assert np.allclose([quad8_shape(*p) for p in LOCAL_COORDS], np.eye(8), atol=1e-14)
    for f in (
        lambda x, y: 1.0,
        lambda x, y: 2 - x + 3 * y,
        lambda x, y: x * y - 2 * x**2 + 0.7 * y**2,
        lambda x, y: x**2 * y - 0.3 * x * y**2,
    ):
        values = np.array([f(*p) for p in LOCAL_COORDS])
        assert max(abs(quad8_shape(*p) @ values - f(*p)) for p in points) < 1e-13
    for p in points:
        exact = TOPOLOGY.shape_functions_exact(*(Fraction(v).limit_denominator(100) for v in p))
        assert np.allclose(
            [float(v) for v in exact], quad8_shape(*(float(Fraction(v).limit_denominator(100)) for v in p))
        )
        N, dN = TOPOLOGY.shape_functions_and_grad(*p)
        h = 1e-7
        fd = np.array(
            [
                (quad8_shape(p[0] + h, p[1]) - quad8_shape(p[0] - h, p[1])) / (2 * h),
                (quad8_shape(p[0], p[1] + h) - quad8_shape(p[0], p[1] - h)) / (2 * h),
            ]
        ).T
        assert np.allclose(N, quad8_shape(*p)) and np.allclose(dN, fd, atol=1e-7)


@pytest.mark.parametrize("n", [2, 3])
def test_children_tile_the_parent_and_its_faces(n):
    children = TOPOLOGY.subdivision_children_param(n)
    assert len(children) == n * n
    area = sum(abs((c[1, 0] - c[0, 0]) * (c[3, 1] - c[0, 1])) for c in children)
    assert area == pytest.approx(4.0)
    for faceIndex, face in enumerate(TOPOLOGY.faces):
        axis = next(k for k in range(2) if np.all(LOCAL_COORDS[face][:, k] == LOCAL_COORDS[face][0, k]))
        value = LOCAL_COORDS[face][0, axis]
        tiles = TOPOLOGY.face_child_indices(faceIndex, n)
        assert len(tiles) == n
        for j in tiles:
            assert np.all(children[j][face][:, axis] == value)


@pytest.mark.parametrize("n", [2, 3])
@pytest.mark.parametrize("curved", [False, True])
def test_a_refined_quad8_mesh_is_conforming_and_its_hanging_nodes_follow_the_coarse_trace(n, curved):
    builder = _builder(n)
    eids = _grid(builder, 3, 3, curved=curved)
    mesh = builder.mesh
    kids = mesh.refine(eids[(1, 1)])
    mesh.refine(kids[0])  # a level-2 cell next to level-1 and level-0 cells
    mesh.balance_2to1()
    hanging = mesh.classify_hanging()
    records = mesh.hanging_mpc_records()
    mesh.check_conformity(records.keys())
    assert hanging and all(h.kind == "edge" and len(h.masters) == 3 for h in hanging)
    # every slave sits exactly on the coarse master edge: its coordinates are the trace of the masters'
    coordinates = mesh.registry.coordinates
    for h in hanging:
        trace = sum(w * coordinates[m] for m, w in zip(h.masters, h.weights))
        assert np.allclose(coordinates[h.slave], trace, atol=1e-12)
    # and the flattened records only name independent masters
    assert not set(records) & {m for masters in records.values() for m, _ in masters}


def test_a_missing_hanging_constraint_is_caught():
    builder = _builder(2)
    eids = _grid(builder, 2, 1)
    mesh = builder.mesh
    mesh.refine(eids[(0, 0)])
    slaves = list(mesh.hanging_mpc_records())
    assert len(slaves) == 2  # the quarter points of the shared edge; its midpoint is the coarse midside node
    mesh.check_conformity(slaves)
    with pytest.raises(TopologyError, match="not conforming"):
        mesh.check_conformity(slaves[1:])


def test_a_neighbour_numbered_differently_shares_every_new_node():
    """The right root's local numbering starts at another corner: the shared edge's new nodes must still
    get one label each, decided from corner labels, not from coordinates."""
    builder = _builder(2)
    left = builder.addRoot(quad8_box_coords(0, 1, 0, 1))
    box = quad8_box_coords(1, 2, 0, 1)
    rotated = np.vstack([np.roll(box[:4], -2, axis=0), np.roll(box[4:], -2, axis=0)])
    right = builder.addRoot(rotated)
    builder.mesh.refine(left)
    builder.mesh.refine(right)
    seen = {}
    for label, coord in builder.mesh.registry.coordinates.items():
        seen.setdefault(tuple(np.round(coord, 12)), []).append(label)
    assert all(len(labels) == 1 for labels in seen.values())
    assert not builder.mesh.classify_hanging()


def test_balance_keeps_face_neighbours_within_one_level():
    builder = _builder(2)
    eids = _grid(builder, 3, 3)
    mesh = builder.mesh
    cell = eids[(1, 1)]
    for _ in range(3):
        cell = mesh.refine(cell)[0]
    assert mesh.balance_2to1() > 0
    faces = {eid: mesh._facesOnEntities(eid) for eid in mesh.active()}
    for a in faces:
        for b in faces:
            if a < b and any(fa.entity == fb.entity and fa.box.overlaps(fb.box) for fa in faces[a] for fb in faces[b]):
                assert abs(mesh.elements[a]["level"] - mesh.elements[b]["level"]) <= 1
    mesh.check_conformity(mesh.hanging_mpc_records().keys())


def test_node_sets_and_surfaces_are_inherited_through_edges_only():
    builder = _builder(2)
    eids = _grid(builder, 2, 1)
    mesh = builder.mesh
    root = eids[(0, 0)]
    bottom = [mesh.elements[root]["conn"][i] for i in TOPOLOGY.faces[0]]
    mesh.define_node_set("bottom", bottom)
    mesh.define_node_set("corner", [mesh.elements[root]["conn"][0]])
    mesh.define_surface("bottomSurface", [(root, 1)])
    kids = mesh.refine(root)
    coordinates = mesh.registry.coordinates
    assert len(mesh.nodeSets["bottom"]) == 5 and all(coordinates[n][1] == 0.0 for n in mesh.nodeSets["bottom"])
    assert len(mesh.nodeSets["corner"]) == 1
    assert mesh.surfaces["bottomSurface"] == {(kids[j], 1) for j in TOPOLOGY.face_child_indices(0, 2)}
    for eid, faceID in mesh.surfaces["bottomSurface"]:
        conn = mesh.elements[eid]["conn"]
        assert all(coordinates[conn[i]][1] == 0.0 for i in TOPOLOGY.faces[TOPOLOGY.faceid_to_face[faceID]])


def test_root_entity_keys_in_2d():
    builder = _builder(2)
    eids = _grid(builder, 2, 1)
    mesh = builder.mesh
    table, M = mesh.rootEntities, mesh.latticeUnits
    left, right = eids[(0, 0)], eids[(1, 0)]
    corner = table.key(left, (2 * M, 0))
    assert corner.kind == "vertex" and sorted(table.roots_sharing(corner)) == [left, right]
    shared = table.key(left, (2 * M, M // 2))
    assert shared.kind == "edge" and sorted(table.roots_sharing(shared)) == [left, right]
    # the same point seen from the right root has the same key
    assert table.key(right, (0, M // 2)) == shared
    assert table.local_point(right, shared) == (0, M // 2)
    assert table.key(left, (M, M)).kind == "interior"


def test_each_edge_of_a_quad8_is_listed_once():
    """In 2D the faces are the edges in another slot order: a point on an edge lies on that one edge."""
    extent = 4
    assert TOPOLOGY.entities_containing((1, 0), extent) == [(0, 4, 1)]
    assert sorted(TOPOLOGY.entities_containing((0, 0), extent)) == [(0, 4, 1), (3, 7, 0)]
    assert TOPOLOGY.entities_containing((1, 1), extent) == []
    assert TOPOLOGY.lowest_entity_containing((4, 3), extent) == ("edge", (1, 5, 2))


@pytest.mark.parametrize("topology", [Quad8Topology(), Hex20Topology()], ids=["QUAD8", "HEX20"])
def test_the_inverse_map_recovers_reference_points_of_a_curved_element(topology):
    """Distorted corners and bulged midside nodes: the map is far from affine, so Newton needs several steps."""
    rng = np.random.default_rng(3)
    reference = np.asarray(topology.reference_node_param(), dtype=float)
    coords = 2.0 * reference + 0.2 * rng.uniform(-1, 1, reference.shape)
    for xi in rng.uniform(-0.95, 0.95, (20, topology.dim)):
        point = topology.shape_functions(*xi) @ coords
        assert np.allclose(topology.inverse_map(point, coords), xi, atol=1e-10)
    # a collapsed element has a singular Jacobian: Newton stops at the center instead of failing
    assert np.array_equal(topology.inverse_map(np.ones(topology.dim), np.zeros_like(coords)), np.zeros(topology.dim))


def _gaussPoints2D(n):
    points, _ = np.polynomial.legendre.leggauss(n)
    return np.array([(x, y) for x in points for y in points])


@pytest.mark.parametrize(
    "nGauss, field",
    [
        (2, lambda x, y: 1.5 - x + 2.0 * y - 3.0 * x * y),
        (3, lambda x, y: -0.5 + x * y**2 - 2.0 * x**2 * y + x**2 * y**2 - y),
    ],
    ids=["bilinear from 4 points", "biquadratic from 9 points"],
)
def test_the_polynomial_projection_in_2d_reproduces_the_tensor_product_fields_of_its_degree(nGauss, field):
    parentPoints = _gaussPoints2D(nGauss)
    # the quadrature points of the lower left child, in the parent's reference square
    childPoints = (_gaussPoints2D(nGauss) - 1) / 2
    parentValues = np.array([[field(*p), 2.0 * field(*p)] for p in parentPoints])
    childValues = PolynomialProjection()._transferColumns(
        parentValues, parentPoints, childPoints, np.zeros((len(childPoints), 2)), np.array([0, 1])
    )
    expected = np.array([[field(*p), 2.0 * field(*p)] for p in childPoints])
    assert np.allclose(childValues, expected, atol=1e-12)


def test_a_non_manifold_2d_root_mesh_is_rejected():
    """Three roots owning one edge (the same two corner labels)."""
    M = 4
    table = RootEntityTable(TOPOLOGY, M)
    table.add_root(1, list(range(1, 9)))
    edge = TOPOLOGY.faces[0]
    table.add_root(2, [slot + 1 if slot in edge else 200 + slot for slot in range(8)])
    with pytest.raises(ValueError, match="manifold"):
        table.add_root(3, [slot + 1 if slot in edge else 300 + slot for slot in range(8)])


def test_the_recovery_error_marker_rejects_quad8_elements():
    marker = RecoveryErrorMarker("nonlocal damage")
    marker.checkTopology(Hex20Topology())
    with pytest.raises(ValueError, match="recoveryError marker on nodeField 'nonlocal damage'.*8-node .* 2D"):
        marker.checkTopology(TOPOLOGY)
