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
classification.

Geometry-level building blocks that operate on node coordinates and connectivity, independent of
the live FEModel. Subdivision honours curved parents via the parent isoparametric map. Hanging
nodes are classified against the coarse entity (face or edge) they lie on, which yields the master
set + serendipity weights for the exact hanging-node MPC.
"""

from collections import defaultdict

import numpy as np

from edelweissfe.adaptivity.geometry import (
    point_in_convex_quad,
    quadratic_edge_parameter,
)
from edelweissfe.utils.performancetiming import timeit


class NodeRegistry:
    """Coordinate-keyed node registry that mints unique labels and deduplicates shared nodes.

    Keys are namespaced per connected component (body): across a flush interface -- a tied surface
    pair, a zero-gap contact pair, a duplicated-node crack plane -- two topologically distinct nodes
    legitimately share one coordinate, and must not be deduplicated into a single label.
    """

    def __init__(self, decimals: int = 8, reserve_labels=None):
        self.decimals = decimals
        self._byKey = {}  # (componentId, rounded-coord) key -> label
        self.coordinates = {}  # label -> np.ndarray(coord)
        self.componentOf = {}  # label -> componentId of the body the node belongs to
        self._maxLabel = 0
        # ``count -> range`` allocator of the model this registry mirrors, i.e.
        # FEModel.reserveNodeNumbers. Given one, the registry stops being a second, independent
        # source of node labels: model and octree then draw from one counter and cannot drift into
        # a collision. Left out (the standalone octree, and its tests, own no model) it falls back
        # to minting max+1 itself, which is only safe while nothing else mints.
        self._reserveLabels = reserve_labels

    def _key(self, coord, componentId: int):
        return (componentId, tuple(round(float(v), self.decimals) for v in coord))

    def seed(self, label: int, coord, componentId: int = 0):
        """Pre-register an existing (label, coordinate) of one body, so a live model's node labels
        are reused.

        Seeding the same node repeatedly (it is shared by several elements of the body) is a no-op.
        A coordinate already claimed by a DIFFERENT label of the same body means two distinct nodes
        occupy the same point, which a coordinate-keyed registry cannot disambiguate -- rejected
        here rather than silently collapsed.
        """
        key = self._key(coord, componentId)
        taken = self._byKey.get(key)
        if taken is not None and taken != label:
            raise ValueError(
                "two distinct nodes ({:d} and {:d}) of the same body occupy the point {:s}; "
                "adaptive refinement identifies nodes by their coordinates and cannot "
                "disambiguate them".format(taken, label, np.array2string(np.asarray(coord, dtype=float)))
            )
        self._byKey[key] = label
        # a copy, never an alias of a live Node's array: an in-place coordinate mutation would
        # otherwise desync the stored coordinate from the key it was registered under
        self.coordinates[label] = np.array(coord, dtype=float)
        self.componentOf[label] = componentId
        self._maxLabel = max(self._maxLabel, label)

    def reserve_labels_up_to(self, label: int):
        """Raise the label high-water mark without registering a coordinate, so freshly minted
        labels cannot collide with existing labels the registry does not track (nodes outside the
        refineable mesh, e.g. those of contact facets)."""
        self._maxLabel = max(self._maxLabel, label)

    def label(self, coord, componentId: int = 0) -> int:
        """Return the label of a coordinate within one body, minting a fresh label if unseen.

        Only the minting branch draws a number; a coordinate that is already known -- the common
        case, since every interior node is shared by several elements -- consumes nothing.
        """
        key = self._key(coord, componentId)
        lab = self._byKey.get(key)
        if lab is None:
            lab = self._mint()
            self._byKey[key] = lab
            self.coordinates[lab] = np.array(coord, dtype=float)
            self.componentOf[lab] = componentId
        return lab

    def _mint(self) -> int:
        """One fresh label, from the model's allocator if this registry was given one."""
        if self._reserveLabels is None:
            label = self._maxLabel + 1
        else:
            (label,) = self._reserveLabels(1)
            if label <= self._maxLabel:
                # The allocator is monotonic and was told about every label the registry knows (see
                # reserve_labels_up_to), so this cannot happen -- unless the two were wired up to
                # different models, which would silently alias two nodes onto one label.
                raise ValueError(
                    "the node allocator handed out label {:d}, which the refinement registry "
                    "already uses; registry and model are out of sync".format(label)
                )
        self._maxLabel = label
        return label

    def connectivity(self, coords, componentId: int = 0) -> list:
        """Map a list/array of node coordinates of one body to their labels (registering as needed)."""
        return [self.label(c, componentId) for c in coords]


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


def _elements_share_face(coordsA, coordsB, topology, tol=1e-7):
    """Topological/geometric shared-face neighbour test: do the two hexes have a pair of coplanar,
    overlapping faces? Coordinate-system agnostic (works for arbitrarily oriented, non-parallelogram
    faces) and handles coarse/fine (a fine face nested in a coarse one) via centroid containment."""
    if not _boxes_overlap(_box_of(coordsA), _box_of(coordsB), tol):
        return False
    facesA = topology.element_face_corners(coordsA)
    facesB = topology.element_face_corners(coordsB)
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


class AdaptiveMesh:
    """Octree hierarchy of HEX20 elements: refinement, 2:1 balancing and hanging-node classification.

    Adjacency is computed from axis-aligned bounding boxes, which is exact for a structured
    (axis-aligned) base mesh -- the standard AMR setting. A curved / unstructured base mesh would
    require topological (shared-face) adjacency instead; that is future work.
    """

    def __init__(self, decimals: int = 8, splitFactor: int = 2, topology=None, reserve_labels=None):
        if topology is None:
            from edelweissfe.adaptivity.hex20topology import Hex20Topology

            topology = Hex20Topology()
        self.topology = topology
        self.registry = NodeRegistry(decimals, reserve_labels=reserve_labels)
        self.splitFactor = splitFactor  # n: each refined element is split into n**3 children per axis
        self.elements = {}  # eid -> dict(conn, coords, level, active, parent, children)
        self.elementSets = {}  # name -> set(eid)      (children inherit membership on refine)
        self.nodeSets = {}  # name -> set(node label)
        self.surfaces = {}  # name -> set((eid, faceID))  (element-based, Marmot faceID convention)
        self._next = 1

    # ---- topological containers ----
    def define_element_set(self, name, eids):
        self.elementSets[name] = set(eids)

    def define_node_set(self, name, labels):
        self.nodeSets[name] = set(labels)

    def define_surface(self, name, pairs):
        """pairs: iterable of (eid, faceID) with Marmot faceID (1-6)."""
        self.surfaces[name] = set(pairs)

    def _add(self, coords, level, parent, componentId: int = 0):
        coords = np.asarray(coords, dtype=float)
        eid = self._next
        self._next += 1
        self.elements[eid] = dict(
            conn=self.registry.connectivity(coords, componentId),
            coords=coords,
            level=level,
            active=True,
            parent=parent,
            children=[],
            componentId=componentId,
        )
        return eid

    def add_root(self, coords, componentId: int = 0) -> int:
        """Add a level-0 element from its 20 node coordinates (C3D20 order).

        ``componentId`` identifies the connected body (mesh component) the element belongs to. Node
        labels are namespaced per body, and hanging nodes are only ever classified within one body,
        so two bodies sharing a flush interface are never welded together by refinement.
        """
        return self._add(coords, level=0, parent=None, componentId=componentId)

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
        """Subdivide an active element into 8 children; deactivate the parent and keep all
        topological containers (element sets, surfaces, node sets) consistent.

        Children are returned in octant_children_param order, so kids[j] is octant j.
        """
        e = self.elements[eid]
        if not e["active"]:
            return e["children"]
        parent_conn = e["conn"]
        kids = [
            self._add(ch, e["level"] + 1, eid, e["componentId"])
            for ch in self.topology.subdivide(e["coords"], self.splitFactor)  # children stay in the parent's body
        ]
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
                for j in self.topology.face_child_indices(self.topology.faceid_to_face[fid], self.splitFactor):
                    pairs.add((kids[j], fid))

        # node sets: a new node joins a set if it lies on a parent face/edge fully contained in the set
        new_nodes = {lab for k in kids for lab in self.elements[k]["conn"]} - set(parent_conn)
        coords = self.registry.coordinates
        for S in self.nodeSets.values():
            for f in self.topology.faces:
                if all(parent_conn[i] in S for i in f):
                    fcorners = np.array([coords[parent_conn[i]] for i in f[:4]])
                    for nl in new_nodes:
                        if point_in_convex_quad(coords[nl], fcorners):
                            S.add(nl)
            for ed in self.topology.edges:
                if all(parent_conn[i] in S for i in ed):
                    a, m, b = coords[parent_conn[ed[0]]], coords[parent_conn[ed[1]]], coords[parent_conn[ed[2]]]
                    for nl in new_nodes:
                        _, dist = quadratic_edge_parameter(coords[nl], a, m, b)
                        if dist < 1e-8:
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
                    if _elements_share_face(crd[a], crd[b], self.topology, tol):
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
        # Timed separately from the scan below: these indices are rebuilt from scratch on every
        # call, over the WHOLE active mesh, whereas the scan itself is restricted to the refined
        # interface shell. If the index build dominates, the cost to attack is incrementality, not
        # the search.
        timerIndex = timeit("hanging: whole-mesh index build")
        timerIndex.__enter__()
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
        # a hanging node is a within-body notion: two bodies meeting at a flush interface (tie,
        # zero-gap contact, crack plane) merely touch, and neither may master the other's nodes
        comp = {eid: self.elements[eid]["componentId"] for eid in act}
        componentOf = self.registry.componentOf
        elemGrid = defaultdict(set)
        # Highest refinement level present in each (grid cell, body). Built in the same pass as the
        # grid itself, for a few dict compares per element, and it is what keeps the scan below from
        # paying for the conforming majority of the mesh -- see hasFinerNeighbour.
        cellMaxLevel = {}
        for eid in act:
            level, componentId = lev[eid], comp[eid]
            for cell in _grid_cells_for_box(box[eid][0], box[eid][1], h_cell, pad=0):
                elemGrid[cell].add(eid)
                key = (cell, componentId)
                if cellMaxLevel.get(key, -1) < level:
                    cellMaxLevel[key] = level

        def hasFinerNeighbour(eid):
            level, componentId = lev[eid], comp[eid]
            cells = _grid_cells_for_box(box[eid][0], box[eid][1], h_cell)

            # Cheap necessary condition first: if no cell this element's box touches holds ANY
            # strictly finer element of the same body, it cannot have a finer neighbour. Away from
            # a refinement front -- i.e. almost everywhere -- this exits before the set unions
            # below, which otherwise ran for every active element and made this scan the second
            # largest per-round cost.
            if not any(cellMaxLevel.get((cell, componentId), -1) > level for cell in cells):
                return False

            # ... and only then the exact test, unchanged: the filter above is necessary, not
            # sufficient (a finer element may sit in a shared cell without its box overlapping).
            neighbours = set()
            for cell in cells:
                neighbours |= elemGrid.get(cell, set())
            return any(
                lev[f] > level and _boxes_overlap(box[eid], box[f])
                for f in neighbours
                if f != eid and comp[f] == componentId
            )

        timerIndex.__exit__(None, None, None)

        best = {}  # slave -> (dim, level, masters)
        timerScan = timeit("hanging: interface-shell scan")
        timerScan.__enter__()
        for eid in act:
            if not hasFinerNeighbour(eid):
                continue  # no level jump here -> this element cannot host a hanging node
            E = self.elements[eid]
            Eset = set(E["conn"])
            bMin, bMax = box[eid]
            compEid = comp[eid]
            cands = set()
            for cell in _grid_cells_for_box(bMin, bMax, h_cell):
                for lab in nodeGrid.get(cell, ()):
                    if lab in Eset or componentOf[lab] != compEid:
                        continue
                    # A hanging node lies on a face/edge of E, hence within E's node-AABB. The padded
                    # broad-phase gather above also pulls in a shell of nodes just
                    # outside E (so nothing on the boundary is missed to grid rounding); those cannot
                    # be hanging on E, so reject them here with a cheap box test before the exact
                    # per-edge/face geometry probes, which otherwise run 12 + 6 tests on each such
                    # node for nothing. Exact: no true hanging node is ever outside this box.
                    c = coords[lab]
                    if (
                        bMin[0] - tol <= c[0] <= bMax[0] + tol
                        and bMin[1] - tol <= c[1] <= bMax[1] + tol
                        and bMin[2] - tol <= c[2] <= bMax[2] + tol
                    ):
                        cands.add(lab)
            for h in self.topology.classify_hanging_on_element(E["conn"], self.registry, cands, tol):
                dim = 1 if h["kind"] == "edge" else 2
                key = (dim, E["level"])
                cur = best.get(h["slave"])
                if cur is None or key < (cur[0], cur[1]):
                    best[h["slave"]] = (dim, E["level"], h["masters"])

        timerScan.__exit__(None, None, None)

        return [{"slave": s, "kind": "edge" if v[0] == 1 else "face", "masters": v[2]} for s, v in best.items()]

    def hanging_mpc_records(self, tol=1e-8) -> dict:
        """Flattened master-slave records for DOF-elimination MPCs.

        Returns {slaveLabel: [(masterLabel, weight), ...]} where every master is an INDEPENDENT
        (non-hanging) node. Multi-level chains (a master that is itself a slave) are resolved here by
        recursive substitution with weight composition; kept as a cheap pre-flattening even though
        :class:`~edelweissfe.numerics.mpctransformation.MultiPointConstraintTransformation` now also
        flattens chains generally (including across other MPCs, e.g. a tie facet referencing a
        hanging slave). Weights are field-independent (equal-order).
        """
        coords = self.registry.coordinates
        raw = {}  # slaveLabel -> [(masterLabel, weight)]
        for h in self.classify_hanging(tol):
            mc = [coords[m] for m in h["masters"]]
            w = self.topology.hanging_weights(mc, coords[h["slave"]], h["kind"])
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
