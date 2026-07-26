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

"""HEX20 octree refinement: subdivision, coordinate-based node registry, and hanging-node
classification (WS-A/B).

Geometry-level building blocks that operate on node coordinates and connectivity, independent of
the live FEModel. Subdivision honours curved parents via the parent isoparametric map. Hanging
nodes are classified against the coarse entity (face or edge) they lie on, which yields the master
set + serendipity weights for the exact hanging-node MPC.
"""

from collections import defaultdict

import numpy as np

from edelweissfe.adaptivity.geometry import (
    bilinear_inverse,
    point_in_convex_quad,
    quadratic_edge_parameter,
)
from edelweissfe.adaptivity.hex20topology import (
    EDGES,
    FACEID_TO_FACE,
    FACES,
    face_child_octants,
    hex20_shape,
    quad8_shape,
    subdivision_children_param,
)


def hanging_weights(master_coords, slave_coord):
    """Exact coarse-trace weights of a slave on its master entity (8-node QUAD8 face via bilinear
    inverse, or 3-node quadratic edge). Field-independent (equal-order interpolation)."""
    mc = np.asarray(master_coords, dtype=float)
    p = np.asarray(slave_coord, dtype=float)
    if len(mc) == 8:
        xi, eta = bilinear_inverse(p, mc[0:4])
        return quad8_shape(xi, eta)
    ca, cm, cb = mc[0], mc[1], mc[2]  # edge: [corner, midside, corner]
    t, _ = quadratic_edge_parameter(p, ca, cm, cb)
    return np.array([0.5 * t * (t - 1.0), 1.0 - t**2, 0.5 * t * (t + 1.0)])


class NodeRegistry:
    """Coordinate-keyed node registry that mints unique labels and deduplicates shared nodes."""

    def __init__(self, decimals: int = 8):
        self.decimals = decimals
        self._byKey = {}  # rounded-coord key -> label
        self.coordinates = {}  # label -> np.ndarray(coord)
        self._maxLabel = 0

    def _key(self, coord):
        return tuple(round(float(v), self.decimals) for v in coord)

    def seed(self, label: int, coord):
        """Pre-register an existing (label, coordinate), so a live model's node labels are reused."""
        self._byKey[self._key(coord)] = label
        self.coordinates[label] = np.asarray(coord, dtype=float)
        self._maxLabel = max(self._maxLabel, label)

    def label(self, coord) -> int:
        """Return the label for a coordinate, minting a fresh (max+1) label if unseen."""
        key = self._key(coord)
        lab = self._byKey.get(key)
        if lab is None:
            self._maxLabel += 1
            lab = self._maxLabel
            self._byKey[key] = lab
            self.coordinates[lab] = np.asarray(coord, dtype=float)
        return lab

    def connectivity(self, coords) -> list:
        """Map a list/array of node coordinates to their labels (registering as needed)."""
        return [self.label(c) for c in coords]


def subdivide(parent_coords: np.ndarray, n: int = 2):
    """Subdivide one HEX20 into ``n**3`` children (``n = 2`` = octree bisection, 8 children).

    Parameters
    ----------
    parent_coords
        (20, 3) physical coordinates of the parent HEX20 (C3D20 order).
    n
        Number of equal parts per axis (the split factor).

    Returns
    -------
    list of (20, 3) arrays
        Physical coordinates of the ``n**3`` child HEX20s, in C3D20 order, obtained by evaluating the
        parent isoparametric map at each child node's parent-parametric position.
    """
    parent_coords = np.asarray(parent_coords, dtype=float)
    children = []
    for child_param in subdivision_children_param(n):  # (20, 3) parent-parametric coords
        phys = np.array([hex20_shape(*p) @ parent_coords for p in child_param])
        children.append(phys)
    return children


def _lies_on_segment(p, a, b, tol=1e-8):
    """True if point p lies on segment [a, b] (within tol), returning also the parameter t in [0,1]."""
    ab = b - a
    L2 = float(ab @ ab)
    if L2 == 0.0:
        return False, 0.0
    t = float((p - a) @ ab) / L2
    if t < -tol or t > 1 + tol:
        return False, t
    proj = a + t * ab
    return bool(np.linalg.norm(p - proj) < tol), t


def _box_of(coords):
    coords = np.asarray(coords, dtype=float)
    return coords.min(axis=0), coords.max(axis=0)


def _grid_key(coord, h):
    return (int(np.floor(coord[0] / h)), int(np.floor(coord[1] / h)), int(np.floor(coord[2] / h)))


def _grid_cells_for_box(bMin, bMax, h, pad=1):
    """Yield the grid-cell keys overlapping an axis-aligned box (padded), for a uniform-hash broad
    phase that makes hanging classification and 2:1 balancing local (O(n) instead of O(n^2))."""
    lo = [int(np.floor(bMin[i] / h)) - pad for i in range(3)]
    hi = [int(np.floor(bMax[i] / h)) + pad for i in range(3)]
    for i in range(lo[0], hi[0] + 1):
        for j in range(lo[1], hi[1] + 1):
            for k in range(lo[2], hi[2] + 1):
                yield (i, j, k)


def _boxes_overlap(boxA, boxB, tol=1e-8):
    """Axis-aligned bounding-box overlap test -- a cheap necessary condition (broad phase) used to
    prune the exact face-adjacency test. Correct for any orientation: face-adjacent elements always
    have touching/overlapping AABBs."""
    (aMin, aMax), (bMin, bMax) = boxA, boxB
    return all(aMin[ax] - tol <= bMax[ax] and bMin[ax] - tol <= aMax[ax] for ax in range(3))


def _element_face_corners(coords):
    """The 4 corner coordinates of each of the 6 faces of a HEX20 (in QUAD8 loop order)."""
    coords = np.asarray(coords, dtype=float)
    return [coords[[f[0], f[1], f[2], f[3]]] for f in FACES]


def _elements_share_face(coordsA, coordsB, tol=1e-7):
    """Topological/geometric shared-face neighbour test: do the two hexes have a pair of coplanar,
    overlapping faces? Coordinate-system agnostic (works for arbitrarily oriented, non-parallelogram
    faces) and handles coarse/fine (a fine face nested in a coarse one) via centroid containment."""
    if not _boxes_overlap(_box_of(coordsA), _box_of(coordsB), tol):
        return False
    facesA = _element_face_corners(coordsA)
    facesB = _element_face_corners(coordsB)
    for fa in facesA:
        ca = fa.mean(axis=0)
        na = np.cross(fa[1] - fa[0], fa[3] - fa[0])
        na = na / np.linalg.norm(na)
        for fb in facesB:
            nb = np.cross(fb[1] - fb[0], fb[3] - fb[0])
            nb = nb / np.linalg.norm(nb)
            if abs(abs(na @ nb) - 1.0) > 1e-6:  # planes not parallel
                continue
            cb = fb.mean(axis=0)
            if abs((cb - ca) @ na) > tol:  # planes not coincident
                continue
            if point_in_convex_quad(cb, fa, tol) or point_in_convex_quad(ca, fb, tol):
                return True
    return False


def _point_on_element_surface(p, coords, tol=1e-7):
    """True if point p lies on any of the 6 faces of the hex given by its 20 node coordinates."""
    return any(point_in_convex_quad(p, fc, tol) for fc in _element_face_corners(coords))


class AdaptiveMesh:
    """Octree hierarchy of HEX20 elements: refinement, 2:1 balancing and hanging-node classification.

    Adjacency is computed from axis-aligned bounding boxes, which is exact for a structured
    (axis-aligned) base mesh -- the standard AMR setting. A curved / unstructured base mesh would
    require topological (shared-face) adjacency instead; that is future work.
    """

    def __init__(self, decimals: int = 8, splitFactor: int = 2):
        self.registry = NodeRegistry(decimals)
        self.splitFactor = splitFactor  # n: each refined element is split into n**3 children per axis
        self.elements = {}  # eid -> dict(conn, coords, level, active, parent, children)
        self.elementSets = {}  # name -> set(eid)      (children inherit membership on refine)
        self.nodeSets = {}  # name -> set(node label)
        self.surfaces = {}  # name -> set((eid, faceID))  (element-based, Marmot faceID convention)
        self._next = 1

    # ---- topological containers (WS-K) ----
    def define_element_set(self, name, eids):
        self.elementSets[name] = set(eids)

    def define_node_set(self, name, labels):
        self.nodeSets[name] = set(labels)

    def define_surface(self, name, pairs):
        """pairs: iterable of (eid, faceID) with Marmot faceID (1-6)."""
        self.surfaces[name] = set(pairs)

    def _add(self, coords, level, parent):
        coords = np.asarray(coords, dtype=float)
        eid = self._next
        self._next += 1
        self.elements[eid] = dict(
            conn=self.registry.connectivity(coords),
            coords=coords,
            level=level,
            active=True,
            parent=parent,
            children=[],
        )
        return eid

    def add_root(self, coords) -> int:
        """Add a level-0 element from its 20 node coordinates (C3D20 order)."""
        return self._add(coords, level=0, parent=None)

    def active(self) -> list:
        return [eid for eid, e in self.elements.items() if e["active"]]

    def box(self, eid):
        return _box_of(self.elements[eid]["coords"])

    def find_by_center(self, center, tol=1e-6):
        """Return the active element whose bounding-box center matches (utility for scripting)."""
        center = np.asarray(center, dtype=float)
        for eid in self.active():
            bMin, bMax = self.box(eid)
            if np.linalg.norm((bMin + bMax) / 2 - center) < tol:
                return eid
        return None

    def refine(self, eid) -> list:
        """Subdivide an active element into 8 children (WS-B); deactivate the parent and keep all
        topological containers (element sets, surfaces, node sets) consistent (WS-K).

        Children are returned in octant_children_param order, so kids[j] is octant j.
        """
        e = self.elements[eid]
        if not e["active"]:
            return e["children"]
        parent_conn = e["conn"]
        kids = [self._add(ch, e["level"] + 1, eid) for ch in subdivide(e["coords"], self.splitFactor)]
        e["active"] = False
        e["children"] = kids

        # element sets + section assignment: children inherit every membership of the parent
        for members in self.elementSets.values():
            if eid in members:
                members.update(kids)

        # surfaces: (parent, faceID) -> (child, faceID) for the children tiling that face
        for pairs in self.surfaces.values():
            faceids_here = [fid for (peid, fid) in pairs if peid == eid]
            for fid in faceids_here:
                pairs.discard((eid, fid))
                for j in face_child_octants(FACEID_TO_FACE[fid], self.splitFactor):
                    pairs.add((kids[j], fid))

        # node sets: a new node joins a set if it lies on a parent face/edge fully contained in the set
        new_nodes = {lab for k in kids for lab in self.elements[k]["conn"]} - set(parent_conn)
        coords = self.registry.coordinates
        for S in self.nodeSets.values():
            for f in FACES:
                if all(parent_conn[i] in S for i in f):
                    fcorners = np.array([coords[parent_conn[i]] for i in f[:4]])
                    for nl in new_nodes:
                        if point_in_convex_quad(coords[nl], fcorners):
                            S.add(nl)
            for ed in EDGES:
                if all(parent_conn[i] in S for i in ed):
                    a, b = coords[parent_conn[ed[0]]], coords[parent_conn[ed[2]]]
                    for nl in new_nodes:
                        if _lies_on_segment(coords[nl], a, b)[0]:
                            S.add(nl)
        return kids

    def _cellSize(self, act):
        """A spatial-hash cell size: the smallest active element's largest extent, so a fine element
        spans ~one cell and a one-level-coarser neighbour a few."""
        exts = [float((self.box(eid)[1] - self.box(eid)[0]).max()) for eid in act]
        return max(min(exts), 1e-12) if exts else 1.0

    def balance_2to1(self, tol=1e-8) -> int:
        """Refine coarser elements until no face-adjacent active pair differs by >1 level.

        Uses a uniform spatial hash so each element is only tested against nearby elements (local,
        not O(n^2)). Returns the number of extra elements refined by balancing.
        """
        nExtra = 0
        while True:
            act = self.active()
            lev = {eid: self.elements[eid]["level"] for eid in act}
            crd = {eid: self.elements[eid]["coords"] for eid in act}
            box = {eid: self.box(eid) for eid in act}
            h = self._cellSize(act)
            grid = defaultdict(set)
            for eid in act:
                for cell in _grid_cells_for_box(box[eid][0], box[eid][1], h, pad=0):
                    grid[cell].add(eid)

            to_refine = set()
            for a in act:
                neighbours = set()
                for cell in _grid_cells_for_box(box[a][0], box[a][1], h):
                    neighbours |= grid.get(cell, set())
                for b in neighbours:
                    if a is b or lev[a] > lev[b] - 2:
                        continue  # only test whether the coarser 'a' must be refined
                    if _elements_share_face(crd[a], crd[b], tol):
                        to_refine.add(a)
                        break
            if not to_refine:
                break
            for eid in to_refine:
                self.refine(eid)
                nExtra += 1
        return nExtra

    def classify_hanging(self, tol=1e-8) -> list:
        """Classify all hanging nodes in the current active mesh.

        For each active element treated as a potential coarse master, find nodes lying on its
        boundary that are not its own nodes. Each hanging node is deduplicated across candidate
        masters, preferring the lowest-dimensional entity (edge before face) and, among equals, the
        coarsest (lowest-level) master -- which guarantees global continuity with the coarsest trace.

        Returns
        -------
        list of dict
            {"slave", "kind", "masters"} -- ready to drive one hanging-node MPC per unique master set.
        """
        act = self.active()
        coords = self.registry.coordinates
        used = {lab for eid in act for lab in self.elements[eid]["conn"]}

        # spatial hash of nodes, so each element only tests nearby candidate nodes (local, not O(n*N))
        h_cell = self._cellSize(act)
        nodeGrid = defaultdict(list)
        for lab in used:
            nodeGrid[_grid_key(coords[lab], h_cell)].append(lab)

        # A coarse element hosts a hanging node ONLY where a FINER element abuts it: a same-level
        # conforming neighbour shares the element's own nodes, and a coarser neighbour makes THIS
        # element the slave, not the master. So the master candidates are exactly the active elements
        # that have a strictly finer overlapping neighbour -- the thin "interface shell". Restricting
        # the scan to those makes the per-adaptation cost scale with the refined interface area, not
        # with the total number of active elements (which grows every adaptation).
        lev = {eid: self.elements[eid]["level"] for eid in act}
        box = {eid: self.box(eid) for eid in act}
        elemGrid = defaultdict(set)
        for eid in act:
            for cell in _grid_cells_for_box(box[eid][0], box[eid][1], h_cell, pad=0):
                elemGrid[cell].add(eid)

        def hasFinerNeighbour(eid):
            neighbours = set()
            for cell in _grid_cells_for_box(box[eid][0], box[eid][1], h_cell):
                neighbours |= elemGrid.get(cell, set())
            return any(lev[f] > lev[eid] and _boxes_overlap(box[eid], box[f]) for f in neighbours if f != eid)

        best = {}  # slave -> (dim, level, masters)
        for eid in act:
            if not hasFinerNeighbour(eid):
                continue  # no level jump here -> this element cannot host a hanging node
            E = self.elements[eid]
            Eset = set(E["conn"])
            bMin, bMax = box[eid]
            cands = set()
            for cell in _grid_cells_for_box(bMin, bMax, h_cell):
                for lab in nodeGrid.get(cell, ()):
                    if lab not in Eset:
                        cands.add(lab)
            for h in classify_hanging_on_element(E["conn"], self.registry, cands):
                dim = 1 if h["kind"] == "edge" else 2
                key = (dim, E["level"])
                cur = best.get(h["slave"])
                if cur is None or key < (cur[0], cur[1]):
                    best[h["slave"]] = (dim, E["level"], h["masters"])

        return [{"slave": s, "kind": "edge" if v[0] == 1 else "face", "masters": v[2]} for s, v in best.items()]

    def hanging_mpc_records(self, tol=1e-8) -> dict:
        """Flattened master-slave records for DOF-elimination MPCs (WS-J / surface_tie branch).

        Returns {slaveLabel: [(masterLabel, weight), ...]} where every master is an INDEPENDENT
        (non-hanging) node. Multi-level chains (a master that is itself a slave) are resolved by
        recursive substitution with weight composition, since the DOF-elimination transformation
        does not accept chained records. Weights are field-independent (equal-order).
        """
        coords = self.registry.coordinates
        raw = {}  # slaveLabel -> [(masterLabel, weight)]
        for h in self.classify_hanging(tol):
            mc = [coords[m] for m in h["masters"]]
            w = hanging_weights(mc, coords[h["slave"]])
            raw[h["slave"]] = list(zip(h["masters"], w))

        slaves = set(raw)
        memo = {}

        def resolve(s):
            if s in memo:
                return memo[s]
            acc = defaultdict(float)
            for m, w in raw[s]:
                if m in slaves:  # chained: substitute the master's own (resolved) masters
                    for mm, ww in resolve(m).items():
                        acc[mm] += w * ww
                else:
                    acc[m] += w
            memo[s] = dict(acc)
            return memo[s]

        return {s: sorted(resolve(s).items()) for s in raw}


def classify_hanging_on_element(coarse_conn, registry, candidate_labels, tol=1e-8):
    """Classify which candidate nodes hang on a coarse element's faces/edges.

    Parameters
    ----------
    coarse_conn
        The 20 node labels of the coarse element (C3D20 order).
    registry
        NodeRegistry holding coordinates.
    candidate_labels
        Iterable of node labels to test (typically all nodes not belonging to the coarse element).

    Returns
    -------
    list of dict
        One entry per hanging node: {"slave": label, "kind": "edge"|"face",
        "masters": [labels], "master_coords": (m,3)}. The lowest-dimensional coarse entity wins
        (edge before face), so shared-edge nodes get 3 masters, face-interior nodes get 8.
    """
    coarse_conn = list(coarse_conn)
    coarse_set = set(coarse_conn)
    coords = registry.coordinates
    results = []

    for lab in candidate_labels:
        if lab in coarse_set:
            continue
        p = coords[lab]

        # 1) edge?  (lowest dimension first)
        matched = False
        for edge in EDGES:
            ea, em, eb = (coarse_conn[edge[0]], coarse_conn[edge[1]], coarse_conn[edge[2]])
            _, dist = quadratic_edge_parameter(p, coords[ea], coords[em], coords[eb])
            if dist < tol:
                results.append({"slave": lab, "kind": "edge", "masters": [ea, em, eb]})
                matched = True
                break
        if matched:
            continue

        # 2) face interior?
        for face in FACES:
            fcorners = np.array([coords[coarse_conn[i]] for i in face[:4]])
            if point_in_convex_quad(p, fcorners):
                results.append({"slave": lab, "kind": "face", "masters": [coarse_conn[i] for i in face]})
                break

    return results
