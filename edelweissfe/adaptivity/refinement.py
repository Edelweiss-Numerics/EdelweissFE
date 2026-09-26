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
from fractions import Fraction
from itertools import product
from math import floor

import numpy as np

from edelweissfe.adaptivity.geometry import (
    point_in_convex_quad,
    quadratic_edge_parameter,
)
from edelweissfe.adaptivity.rootentities import RootEntityTable
from edelweissfe.utils.exceptions import TopologyError


def _latticeOf(params, n: int) -> np.ndarray:
    """Exact integer lattice indices of parametric coordinates on the subdivision grid.

    Every node a subdivision of a serendipity element can produce lies at ``-1 + k / n`` along each
    axis for an integer ``k`` in ``[0, 2n]`` -- the corners of the sub-cells and the midside nodes
    halfway between them. Converting the parametric coordinate to that integer turns a node's
    position within its parent into exact data, which is what lets node identity be decided by
    integer arithmetic instead of by rounding a floating-point coordinate.
    """
    scaled = (np.asarray(params, dtype=float) + 1.0) * n
    lattice = np.rint(scaled).astype(np.int64)
    if np.abs(scaled - lattice).max() > 1e-9:
        raise ValueError(
            "a subdivision produced a node off the 1/n parametric lattice, so its position within "
            "the parent cannot be represented exactly; node identity would have to fall back on "
            "comparing coordinates"
        )
    return lattice


class NodeRegistry:
    """Coordinate-keyed node registry that mints unique labels and deduplicates shared nodes.

    Keys are namespaced per connected component (body): across a flush interface -- a tied surface
    pair, a zero-gap contact pair, a duplicated-node crack plane -- two topologically distinct nodes
    legitimately share one coordinate, and must not be deduplicated into a single label.
    """

    def __init__(self, decimals: int = 8, reserve_labels=None):
        self.decimals = decimals
        self._byKey = {}  # (componentId, rounded-coord) key -> label; seeding and root elements only
        self._byIdentity = {}  # (componentId, topological identity) -> label; everything refinement mints
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

    def label_for_identity(self, identity, coord, componentId: int = 0) -> int:
        """Return the label of a node named by its *topological* identity, minting one if new.

        ``identity`` names the node by which corners of its parent span it and in what exact integer
        proportion, so two elements sharing a face or an edge derive the identical identity for a
        point on it -- without comparing coordinates, and regardless of how their local axes happen
        to be oriented relative to one another.

        This is what :meth:`label` cannot do. Rounding a coordinate to a fixed number of decimals
        splits one node into two whenever the exact value lands on a rounding tie and the two
        parents' last bits disagree, and a mesh written with a fixed number of decimals produces
        such ties systematically rather than by accident: on the anchor pry-out mesh every
        quarter-point of a graded edge landed on one, giving 156 duplicated nodes with nothing
        constraining them together.
        """
        key = (componentId, identity)
        label = self._byIdentity.get(key)
        if label is None:
            label = self._mint()
            self._byIdentity[key] = label
            self.coordinates[label] = np.array(coord, dtype=float)
            self.componentOf[label] = componentId
            # Keep the coordinate index populated so seeding still sees these nodes; it is never
            # consulted to identify a node that has a topological identity.
            self._byKey.setdefault(self._key(coord, componentId), label)
        return label

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
    # math.floor, not np.floor: same IEEE double division and the same floor, so the same integer
    # key, but without a numpy ufunc dispatch per component -- this runs once per node and, via
    # _grid_cells_for_box, several times per active cell on every adaptation.
    return (floor(coord[0] / h), floor(coord[1] / h), floor(coord[2] / h))


def _grid_cells_for_box(bMin, bMax, h, pad=1):
    """Yield the grid-cell keys overlapping an axis-aligned box (padded), for a uniform-hash broad
    phase that makes hanging classification and 2:1 balancing local (O(n) instead of O(n^2))."""
    lo = [floor(bMin[i] / h) - pad for i in range(3)]
    hi = [floor(bMax[i] / h) + pad for i in range(3)]
    # itertools.product, not three nested Python loops: the same keys in the same order (first axis
    # outermost), as a one-shot iterator exactly like the generator it replaces -- callers that
    # iterate the result twice see the same consumption semantics as before.
    return product(range(lo[0], hi[0] + 1), range(lo[1], hi[1] + 1), range(lo[2], hi[2] + 1))


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
        self.elements = {}  # eid -> dict(conn, coords, box, extent, level, active, parent, children)
        # The active cells, kept alongside elements[eid]["active"] so that active() does not scan
        # the whole hierarchy on every call. Insertion-ordered: a child's eid exceeds every existing
        # one, so appending children after removing their parent keeps the ascending-eid order the
        # scan produced -- which balance_2to1 relies on for the order it refines in.
        self._active = {}  # eid -> None
        self.elementSets = {}  # name -> set(eid)      (children inherit membership on refine)
        self.nodeSets = {}  # name -> set(node label)
        self.surfaces = {}  # name -> set((eid, faceID))  (element-based, Marmot faceID convention)
        self._next = 1
        # The subdivision lattice, resolved once: it depends only on the topology and splitFactor.
        self._childLattice = [
            _latticeOf(param, splitFactor) for param in self.topology.subdivision_children_param(splitFactor)
        ]
        ownLattice = _latticeOf(self.topology.reference_node_param(), splitFactor)
        #: lattice position -> local slot of the parent's own nodes, so a child node landing on one
        #: reuses the parent's node instead of minting a second node at the same point
        self._ownNodeAt = {tuple(row): slot for slot, row in enumerate(ownLattice)}
        # The corners are the nodes sitting at an extreme of every axis; the rest are midside nodes,
        # which span no sub-entity of their own and so never appear in an identity.
        isCorner = np.all((ownLattice == 0) | (ownLattice == 2 * splitFactor), axis=1)
        self._cornerSlots = np.flatnonzero(isCorner)
        self._cornerLattice = ownLattice[isCorner]
        #: the root mesh's vertices, edges and faces, named by root corner labels (see rootentities)
        self.rootEntities = RootEntityTable(self.topology)
        #: result of _foreignBoundaryNodes for the current active mesh; reset by every refinement
        self._foreignBoundaryNodesCache = None
        #: the reference coordinates of an element's own nodes, as exact fractions in [-1, 1]
        self._nodeReference = [
            tuple(Fraction(int(v)) for v in p) for p in np.rint(self.topology.reference_node_param()).astype(int)
        ]

    # ---- topological containers ----
    def define_element_set(self, name, eids):
        self.elementSets[name] = set(eids)

    def define_node_set(self, name, labels):
        self.nodeSets[name] = set(labels)

    def define_surface(self, name, pairs):
        """pairs: iterable of (eid, faceID) with Marmot faceID (1-6)."""
        self.surfaces[name] = set(pairs)

    def _childConnectivity(self, parentConn, childIndex, coords, componentId):
        """The 20 node labels of one child, decided topologically rather than geometrically.

        Each child node is either one of the parent's own nodes -- recognised by an exact lattice
        match, and reused -- or a new node spanned by some of the parent's corners. In the latter
        case its identity is the set of spanning corner *labels* paired with exact integer weights.
        A neighbouring element that shares that face or edge names the same corners with the same
        weights, so both arrive at one label for one point.
        """
        extent = 2 * self.splitFactor
        conn = []
        for slot, lattice in enumerate(self._childLattice[childIndex]):
            own = self._ownNodeAt.get(tuple(lattice))
            if own is not None:
                conn.append(parentConn[own])
                continue
            spanning = []
            for cornerSlot, cornerLattice in zip(self._cornerSlots, self._cornerLattice):
                weight = 1
                for axis in range(len(lattice)):
                    weight *= int(lattice[axis]) if cornerLattice[axis] == extent else extent - int(lattice[axis])
                if weight:
                    spanning.append((parentConn[cornerSlot], weight))
            spanning.sort()
            conn.append(self.registry.label_for_identity(tuple(spanning), coords[slot], componentId))
        return conn

    def _add(self, coords, level, parent, componentId: int = 0, parentConn=None, childIndex=None):
        coords = np.asarray(coords, dtype=float)
        eid = self._next
        self._next += 1
        # A cell's coordinates never change after this, so its bounding box and largest extent are
        # computed here, once. Before, box() recomputed them on every call, and the balancing and
        # hanging-node passes call it several times per active cell on every adaptation -- a cost
        # that scales with the whole mesh rather than with what the adaptation changed.
        box = _box_of(coords)
        if parent is None:
            rootEid, referenceLow, referenceSize = eid, (Fraction(-1),) * 3, Fraction(2)
        else:
            # the child's exact box in its root's reference cube: children are ordered
            # ix * n * n + iy * n + iz (see subdivision_children_param)
            parentElement = self.elements[parent]
            n = self.splitFactor
            position = (childIndex // (n * n), (childIndex // n) % n, childIndex % n)
            rootEid = parentElement["rootEid"]
            referenceSize = parentElement["referenceSize"] / n
            referenceLow = tuple(parentElement["referenceLow"][k] + position[k] * referenceSize for k in range(3))
        self.elements[eid] = dict(
            conn=(
                self.registry.connectivity(coords, componentId)
                if parentConn is None
                else self._childConnectivity(parentConn, childIndex, coords, componentId)
            ),
            coords=coords,
            box=box,
            extent=float((box[1] - box[0]).max()),
            level=level,
            active=True,
            parent=parent,
            children=[],
            componentId=componentId,
            rootEid=rootEid,
            referenceLow=referenceLow,
            referenceSize=referenceSize,
        )
        if parent is None:
            self.rootEntities.add_root(eid, self.elements[eid]["conn"], componentId)
        self._active[eid] = None
        self._foreignBoundaryNodesCache = None
        return eid

    def add_root(self, coords, componentId: int = 0) -> int:
        """Add a level-0 element from its 20 node coordinates (C3D20 order).

        ``componentId`` identifies the connected body (mesh component) the element belongs to. Node
        labels are namespaced per body, and hanging nodes are only ever classified within one body,
        so two bodies sharing a flush interface are never welded together by refinement.
        """
        return self._add(coords, level=0, parent=None, componentId=componentId)

    def active(self) -> list:
        """The active (leaf) cells, in ascending eid order."""
        return list(self._active)

    def node_reference_points(self, eid) -> list:
        """The exact reference points, in the element's root, of the element's own nodes (slot order)."""
        e = self.elements[eid]
        low, size = e["referenceLow"], e["referenceSize"]
        return [tuple(low[k] + size * (p[k] + 1) / 2 for k in range(3)) for p in self._nodeReference]

    def node_keys(self, eid) -> list:
        """The topological keys (see :class:`~edelweissfe.adaptivity.rootentities.RootEntityTable`) of
        the element's own nodes, in slot order. Cached: an element's box never changes."""
        e = self.elements[eid]
        if "nodeKeys" not in e:
            e["nodeKeys"] = [self.rootEntities.key(e["rootEid"], xi) for xi in self.node_reference_points(eid)]
        return e["nodeKeys"]

    def _foreignBoundaryNodes(self) -> list:
        """Every node that lies on the boundary of an active element without being one of its nodes,
        as ``(eid, label, zeta)`` with ``zeta`` the node's exact reference coordinates in that element.

        Exact and exhaustive by construction: every node position is an exact point of its root, and
        the root mesh's vertices, edges and faces are named by their corner labels, so a node on an
        element's boundary is found whichever element -- or neighbouring root -- it belongs to. Also
        verifies that no point carries two nodes and no node two points.

        Only a refined root, or a root sharing a vertex with one, can hold such a node (anywhere else
        the mesh is the conforming root mesh itself), so only their active elements are examined.
        Cached until the mesh is next refined.
        """
        if self._foreignBoundaryNodesCache is not None:
            return self._foreignBoundaryNodesCache
        act = self.active()
        refinedRoots = {self.elements[eid]["rootEid"] for eid in act if self.elements[eid]["level"] > 0}
        relevantRoots = set()
        for root in refinedRoots:
            relevantRoots |= self.rootEntities.roots_touching(root)
        relevant = [eid for eid in act if self.elements[eid]["rootEid"] in relevantRoots]
        labelOfKey, keyOfLabel = {}, {}
        for eid in relevant:
            for label, key in zip(self.elements[eid]["conn"], self.node_keys(eid)):
                if labelOfKey.setdefault(key, label) != label or keyOfLabel.setdefault(label, key) != key:
                    raise TopologyError(
                        f"AMR conformity: node {label} and node {labelOfKey[key]} occupy one point {key} "
                        f"(or node {label} sits at two points); refinement split or merged a node"
                    )

        closure = {root: [] for root in relevantRoots}  # root -> [(label, exact point in that root)]
        for key, label in labelOfKey.items():
            for root in self.rootEntities.roots_sharing(key):
                if root in closure:
                    closure[root].append((label, self.rootEntities.local_point(root, key)))

        foreign = []
        for eid in relevant:
            e = self.elements[eid]
            own = set(e["conn"])
            low, size = e["referenceLow"], e["referenceSize"]
            for label, xi in closure[e["rootEid"]]:
                if label in own:
                    continue
                zeta = tuple(2 * (xi[k] - low[k]) / size - 1 for k in range(3))
                if all(abs(z) <= 1 for z in zeta) and any(abs(z) == 1 for z in zeta):
                    foreign.append((eid, label, zeta))
        self._foreignBoundaryNodesCache = foreign
        return foreign

    def check_hanging_completeness(self, slaveLabels):
        """Raise unless the active mesh is conforming up to the given hanging-node slaves.

        Exact and independent of any geometric classification: every node position is known as an
        exact point of its root element, and every root vertex, edge and face by its corner labels.
        Two conditions are checked.

        * One node per point: a topological key maps to one label, and a label to one key. A split
          point would be two unconnected nodes; a merged one would weld distinct points.
        * Every node that lies on a face of an active element is one of that element's own nodes or
          a hanging-node slave. Anything else is a free node on the element's boundary -- the
          interface is not conforming there.

        Parameters
        ----------
        slaveLabels
            The labels of all hanging-node slaves (e.g. the keys of :meth:`hanging_mpc_records`).

        Raises
        ------
        TopologyError
            Naming the offending nodes and elements.
        """
        slaveLabels = set(slaveLabels)
        unconstrained = [(label, eid) for eid, label, _ in self._foreignBoundaryNodes() if label not in slaveLabels]
        if unconstrained:
            nodes = sorted({label for label, _ in unconstrained})
            raise TopologyError(
                f"AMR conformity: {len(nodes)} node(s) lie on the boundary of an active element without "
                f"being one of its nodes or a hanging-node slave -- the coarse-fine interface is not "
                f"conforming there. First (node, element) pairs: {sorted(unconstrained)[:10]}"
            )

    def box(self, eid):
        """The axis-aligned bounding box ``(min, max)`` of a cell; cached at creation, see :meth:`_add`."""
        return self.elements[eid]["box"]

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
            # children stay in the parent's body, and take their node identities from it
            self._add(ch, e["level"] + 1, eid, e["componentId"], parent_conn, childIndex)
            for childIndex, ch in enumerate(self.topology.subdivide(e["coords"], self.splitFactor))
        ]
        e["active"] = False
        del self._active[eid]
        e["children"] = kids
        self._foreignBoundaryNodesCache = None

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
        elements = self.elements
        return max(min(elements[eid]["extent"] for eid in act), 1e-12) if act else 1.0

    def balance_2to1(self, tol=1e-8) -> int:
        """Refine coarser elements until no face-adjacent active pair differs by >1 level.

        Uses a uniform spatial hash so each element is only tested against nearby elements (local,
        not O(n^2)). Returns the number of extra elements refined by balancing.
        """
        nExtra = 0
        while True:
            act = self.active()
            lev = {eid: self.elements[eid]["level"] for eid in act}
            # Only a pair whose levels differ by two or more can violate the 2:1 rule, so only an
            # element at least two levels above the coarsest one can be the finer partner, and only
            # an element at least two levels below the finest one the coarser. A mesh whose active
            # levels span less than that -- every mesh refined by a single level, and every pass but
            # the first of a cascade -- is balanced by construction, and the neighbour search below
            # would only confirm it, at the cost of a grid over every active cell. That is not a
            # small cost: a coarse far-field element spans hundreds of cells at the finest element's
            # size, and on 30k active cells the search took 2-3 s per adaptation for a to_refine set
            # that was always empty. Necessary conditions, not sufficient ones: whatever passes them
            # gets the unchanged exact test, in the unchanged order.
            minLevel, maxLevel = min(lev.values()), max(lev.values())
            if maxLevel - minLevel < 2:
                break
            crd = {eid: self.elements[eid]["coords"] for eid in act}
            box = {eid: self.box(eid) for eid in act}
            h = self._cellSize(act)
            # The grid holds only the candidate finer partners, and cellMaxLevel the finest level
            # among them per cell: an element 'a' whose padded cells hold nothing two levels finer
            # cannot need refinement and is skipped before the neighbour union.
            grid = defaultdict(set)
            cellMaxLevel = {}
            for eid in act:
                level = lev[eid]
                if level < minLevel + 2:
                    continue
                for cell in _grid_cells_for_box(box[eid][0], box[eid][1], h, pad=0):
                    grid[cell].add(eid)
                    if cellMaxLevel.get(cell, -1) < level:
                        cellMaxLevel[cell] = level

            to_refine = set()
            for a in act:
                finerThan = lev[a] + 1
                if finerThan >= maxLevel:
                    continue
                cells = list(_grid_cells_for_box(box[a][0], box[a][1], h))
                if not any(cellMaxLevel.get(cell, -1) > finerThan for cell in cells):
                    continue
                neighbours = set()
                for cell in cells:
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

    def classify_hanging(self) -> list:
        """All hanging nodes of the active mesh, with their masters and exact weights.

        A hanging node is a node on the boundary of an active element that is not one of its own
        nodes (see :meth:`_foreignBoundaryNodes` -- exact, exhaustive, no geometry). Its master
        entity is the element's edge it lies on (two reference coordinates at +-1) or else its face
        (one at +-1); its weights are the element's own shape functions at the node's exact reference
        point, which on that edge or face are non-zero only for the entity's nodes -- the exact coarse
        trace, on curved and warped elements alike. A node hanging on several elements keeps the
        lowest-dimensional entity and, among equals, the coarsest element, which guarantees global
        continuity with the coarsest trace.

        Returns
        -------
        list of dict
            ``{"slave", "kind", "masters", "weights"}``, sorted by slave label.
        """
        reference = np.rint(self.topology.reference_node_param()).astype(int)
        best = {}  # slave -> (dim, level, kind, masters, weights)
        for eid, label, zeta in self._foreignBoundaryNodes():
            E = self.elements[eid]
            onBoundary = [k for k in range(3) if abs(zeta[k]) == 1]
            if len(onBoundary) == 2:
                kind, dim = "edge", 1
                edge = next(
                    ed for ed in self.topology.edges if all(reference[i][k] == zeta[k] for i in ed for k in onBoundary)
                )
                axis = next(k for k in range(3) if k not in onBoundary)
                t = float(zeta[axis] * reference[edge[2]][axis])  # -1 at the edge's first node, +1 at its last
                weights = np.array([0.5 * t * (t - 1.0), 1.0 - t**2, 0.5 * t * (t + 1.0)])
                entity = edge
            elif len(onBoundary) == 1:
                kind, dim = "face", 2
                axis = onBoundary[0]
                entity = next(f for f in self.topology.faces if all(reference[i][axis] == zeta[axis] for i in f))
                weights = self.topology.shape_functions(*(float(z) for z in zeta))[list(entity)]
            else:
                raise TopologyError(f"AMR: node {label} lies on a corner of element {eid} without being its node")
            key = (dim, E["level"])
            current = best.get(label)
            if current is None or key < current[:2]:
                best[label] = (dim, E["level"], kind, [E["conn"][i] for i in entity], weights)
        return [{"slave": s, "kind": v[2], "masters": v[3], "weights": v[4]} for s, v in sorted(best.items())]

    def hanging_mpc_records(self) -> dict:
        """Flattened master-slave records for DOF-elimination MPCs.

        Returns {slaveLabel: [(masterLabel, weight), ...]} where every master is an INDEPENDENT
        (non-hanging) node. Multi-level chains (a master that is itself a slave) are resolved here by
        recursive substitution with weight composition; kept as a cheap pre-flattening even though
        :class:`~edelweissfe.numerics.mpctransformation.MultiPointConstraintTransformation` now also
        flattens chains generally (including across other MPCs, e.g. a tie facet referencing a
        hanging slave). Weights are field-independent (equal-order).
        """
        raw = {}  # slaveLabel -> [(masterLabel, weight)]
        for h in self.classify_hanging():
            raw[h["slave"]] = list(zip(h["masters"], h["weights"]))

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
