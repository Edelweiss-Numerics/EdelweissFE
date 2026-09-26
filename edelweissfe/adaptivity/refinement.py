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


class AdaptiveMesh:
    """Octree hierarchy of HEX20 elements: refinement, 2:1 balancing and hanging-node classification.

    Every decision about how elements and nodes relate -- which nodes coincide, which node lies on
    which face or edge, which elements share a face, which nodes hang and with which weights -- is
    taken topologically and in exact arithmetic, never from physical coordinates: each element knows
    its root and its exact box in the root's reference cube, and the root mesh's vertices, edges and
    faces are named by their corner labels (:class:`~edelweissfe.adaptivity.rootentities.RootEntityTable`).
    This holds on curved, warped and arbitrarily oriented unstructured meshes alike; the only
    requirement is a conforming root mesh. :meth:`check_hanging_completeness` verifies the result
    after every adaptation.
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
        #: every face and edge of an element, as a tuple of node slots, with the lattice coordinates
        #: that are fixed on it (axis, value): a node lies on the entity iff it shares all of them
        self._fixedLatticeOfEntity = {}
        for entity in list(self.topology.faces) + list(self.topology.edges):
            rows = ownLattice[list(entity)]
            fixed = tuple((k, int(rows[0, k])) for k in range(rows.shape[1]) if np.all(rows[:, k] == rows[0, k]))
            self._fixedLatticeOfEntity[tuple(entity)] = fixed
        self._parentEntities = list(self._fixedLatticeOfEntity)
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

    def _add(self, coords, level, parent, componentId: int = 0, parentConn=None, childIndex=None, labels=None):
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
                self._childConnectivity(parentConn, childIndex, coords, componentId)
                if parentConn is not None
                else (
                    self.registry.connectivity(coords, componentId)
                    if labels is None
                    else self._rootConnectivity(labels)
                )
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

    def add_root(self, coords, componentId: int = 0, labels=None) -> int:
        """Add a level-0 element from its 20 node coordinates (C3D20 order).

        ``componentId`` identifies the connected body (mesh component) the element belongs to. Node
        labels are namespaced per body, and hanging nodes are only ever classified within one body,
        so two bodies sharing a flush interface are never welded together by refinement.

        ``labels`` are the element's node labels in the model (same order). A model-backed mesh must
        pass them: the root mesh's connectivity is then the model's own, exactly. Without them (a
        standalone octree) the labels are looked up, or minted, by rounded coordinate.
        """
        return self._add(coords, level=0, parent=None, componentId=componentId, labels=labels)

    def _rootConnectivity(self, labels) -> list:
        """A root element's node labels as given by the model; every one must have been seeded."""
        labels = [int(label) for label in labels]
        unknown = [label for label in labels if label not in self.registry.coordinates]
        if unknown:
            raise ValueError(f"root element node(s) {unknown} were not seeded into the refinement registry")
        return labels

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

        # For each node, descend its root's octree to the active elements whose closed box contains
        # it: O(nodes x depth), not O(elements x nodes) per root.
        foreign = []
        for root, nodes in closure.items():
            for label, xi in nodes:
                for eid in self._activeElementsContaining(root, xi):
                    e = self.elements[eid]
                    if label in e["conn"]:
                        continue
                    low, size = e["referenceLow"], e["referenceSize"]
                    zeta = tuple(2 * (xi[k] - low[k]) / size - 1 for k in range(3))
                    if any(abs(z) == 1 for z in zeta):
                        foreign.append((eid, label, zeta))
        self._foreignBoundaryNodesCache = foreign
        return foreign

    def _activeElementsContaining(self, rootEid, xi) -> list:
        """The active elements of root ``rootEid`` whose closed reference box contains the exact point
        ``xi``, found by descending the octree: at each level the child along each axis follows exactly
        from the point's position, and a point on a split plane belongs to both children there."""
        n = self.splitFactor
        found, stack = [], [rootEid]
        while stack:
            eid = stack.pop()
            e = self.elements[eid]
            if e["active"]:
                found.append(eid)
                continue
            low, size = e["referenceLow"], e["referenceSize"]
            candidates = []
            for k in range(3):
                t = (xi[k] - low[k]) * n / size
                if t.denominator == 1:
                    candidates.append([i for i in (int(t) - 1, int(t)) if 0 <= i < n])
                else:
                    candidates.append([floor(t)])
            for ix, iy, iz in product(*candidates):
                stack.append(e["children"][ix * n * n + iy * n + iz])
        return found

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

        # node sets: a new node joins a set if it lies on a parent face or edge whose nodes are all in
        # the set. Decided exactly from the node's lattice position in the parent (see _entitiesOf).
        onEntities = {}  # new node label -> parent faces and edges (as node-slot tuples) it lies on
        for childIndex, kid in enumerate(kids):
            for label, lattice in zip(self.elements[kid]["conn"], self._childLattice[childIndex]):
                if label not in onEntities and self._ownNodeAt.get(tuple(lattice)) is None:
                    onEntities[label] = self._entitiesOf(lattice)
        for S in self.nodeSets.values():
            inSet = [entity for entity in self._parentEntities if all(parent_conn[i] in S for i in entity)]
            if not inSet:
                continue
            for label, entities in onEntities.items():
                if any(entity in entities for entity in inSet):
                    S.add(label)
        return kids

    def _entitiesOf(self, lattice) -> set:
        """The parent faces and edges (node-slot tuples) that a node at the given lattice position of
        the parent lies on: exactly those whose fixed reference coordinates it shares."""
        return {
            entity for entity, fixed in self._fixedLatticeOfEntity.items() if all(lattice[k] == v for k, v in fixed)
        }

    def _faceDescriptors(self, eid) -> list:
        """The six faces of an active element as ``(entity, rectangle)``: exact and topological.

        A face on the boundary of the element's root lies on a root face, named by its corner labels,
        and its rectangle is measured in that face's canonical frame, so a neighbouring root derives
        the identical entity and coordinates. A face inside the root lies on an internal plane of that
        root, ``(rootEid, axis, value)``, with the rectangle in the other two reference coordinates.
        Two elements share a face iff they have faces on the same entity whose rectangles overlap with
        positive area.
        """
        e = self.elements[eid]
        rootEid, low, size = e["rootEid"], e["referenceLow"], e["referenceSize"]
        faces = []
        for axis in range(3):
            others = [k for k in range(3) if k != axis]
            for value in (low[axis], low[axis] + size):
                if abs(value) == 1:
                    corners = []
                    for a, b in ((low[others[0]], low[others[1]]), (low[others[0]] + size, low[others[1]] + size)):
                        xi = [None] * 3
                        xi[axis], xi[others[0]], xi[others[1]] = value, a, b
                        entity, uv = self.rootEntities.root_face_coordinates(rootEid, axis, value, tuple(xi))
                        corners.append(uv)
                    (u0, v0), (u1, v1) = corners
                    rectangle = (min(u0, u1), max(u0, u1), min(v0, v1), max(v0, v1))
                    faces.append((("root face",) + entity, rectangle))
                else:
                    rectangle = (low[others[0]], low[others[0]] + size, low[others[1]], low[others[1]] + size)
                    faces.append((("plane", rootEid, axis, value), rectangle))
        return faces

    def balance_2to1(self) -> int:
        """Refine coarser elements until no face-adjacent active pair differs by more than one level.

        Face adjacency is exact and topological (see :meth:`_faceDescriptors`), so balancing holds on
        curved and warped meshes as on flat ones. Returns the number of extra elements refined.

        Only *face* neighbours are balanced. Elements touching only along an edge or at a vertex may
        differ by two or more levels; the interface stays conforming (the finer edge nodes hang on the
        coarsest element's edge, through exactly resolved constraint chains), so this is a grading,
        not a correctness, limitation. Edge/vertex balancing is deliberately not implemented: it would
        refine additional elements and change the results of multi-level runs.
        """
        nExtra = 0
        while True:
            act = self.active()
            lev = {eid: self.elements[eid]["level"] for eid in act}
            # Only a pair whose levels differ by two or more can violate the 2:1 rule: a mesh whose
            # active levels span less -- every mesh refined by a single level -- is balanced.
            minLevel, maxLevel = min(lev.values()), max(lev.values())
            if maxLevel - minLevel < 2:
                break
            # the finer partners (at least two levels above the coarsest), indexed by face entity
            finerOnEntity = defaultdict(list)  # entity -> [(level, rectangle)]
            for eid in act:
                if lev[eid] >= minLevel + 2:
                    for entity, rectangle in self._faceDescriptors(eid):
                        finerOnEntity[entity].append((lev[eid], rectangle))
            to_refine = set()
            for a in act:
                if lev[a] + 2 > maxLevel:
                    continue
                for entity, (u0, u1, v0, v1) in self._faceDescriptors(a):
                    if any(
                        level >= lev[a] + 2 and min(u1, b1) > max(u0, b0) and min(v1, c1) > max(v0, c0)
                        for level, (b0, b1, c0, c1) in finerOnEntity.get(entity, ())
                    ):
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
        # 1) the master of every hanging node: the lowest-dimensional entity, then the coarsest element
        best = {}  # slave -> (dim, level, eid, zeta, kind, entity)
        for eid, label, zeta in self._foreignBoundaryNodes():
            onBoundary = [k for k in range(3) if abs(zeta[k]) == 1]
            if len(onBoundary) == 2:
                kind, dim = "edge", 1
                entity = next(
                    ed for ed in self.topology.edges if all(reference[i][k] == zeta[k] for i in ed for k in onBoundary)
                )
            elif len(onBoundary) == 1:
                kind, dim = "face", 2
                axis = onBoundary[0]
                entity = next(f for f in self.topology.faces if all(reference[i][axis] == zeta[axis] for i in f))
            else:
                raise TopologyError(f"AMR: node {label} lies on a corner of element {eid} without being its node")
            candidate = (dim, self.elements[eid]["level"], eid, zeta, kind, entity)
            current = best.get(label)
            if current is None or candidate[:2] < current[:2]:
                best[label] = candidate

        # 2) its weights: the master element's own shape functions at the exact point, which on the
        # entity are non-zero only for the entity's nodes -- the exact trace, verified before use
        hanging = []
        for label, (dim, level, eid, zeta, kind, entity) in sorted(best.items()):
            exactWeights = self.topology.shape_functions_exact(*zeta)
            self._verifyTraceWeights(label, eid, zeta, entity, exactWeights, reference)
            conn = self.elements[eid]["conn"]
            weights = np.array([float(exactWeights[i]) for i in entity])
            hanging.append({"slave": label, "kind": kind, "masters": [conn[i] for i in entity], "weights": weights})
        return hanging

    @staticmethod
    def _verifyTraceWeights(label, eid, zeta, entity, exactWeights, reference):
        """Raise unless the weights of a hanging node on ``entity`` are its exact trace weights.

        Checked in exact rational arithmetic, so no tolerance: the weights of every node off the
        entity vanish, and the entity's weights reproduce the constant, the node's own reference
        coordinates and every quadratic monomial of them -- which catches a wrong entity, a wrong
        master order or a wrong orientation, not just a wrong formula.
        """
        onEntity = set(entity)
        offEntity = [w for i, w in enumerate(exactWeights) if i not in onEntity]
        monomials = [lambda x: 1] + [lambda x, a=a: x[a] for a in range(3)]
        monomials += [lambda x, a=a, b=b: x[a] * x[b] for a in range(3) for b in range(a, 3)]
        reproduced = all(sum(exactWeights[i] * m(reference[i]) for i in entity) == m(zeta) for m in monomials)
        if any(offEntity) or not reproduced:
            raise TopologyError(
                f"AMR: the hanging-node weights of node {label} on element {eid} (reference point {zeta}) "
                "are not the exact trace of that element's shape functions"
            )

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
