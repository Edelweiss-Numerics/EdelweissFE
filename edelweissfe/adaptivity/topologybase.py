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

"""Abstract element topology for adaptive mesh refinement.

:class:`~edelweissfe.adaptivity.refinement.AdaptiveMesh`, the state-transfer strategies
(:mod:`~edelweissfe.adaptivity.statetransfer.base`) and the ``hadaptivity`` model modifier are all
written against this interface rather than against HEX20 directly, so a second refineable element
family only needs its own :class:`TopologyBase` implementation, not changes to the AMR machinery
itself. :class:`~edelweissfe.adaptivity.hex20topology.Hex20Topology` is the reference implementation.
"""

from abc import ABC, abstractmethod
from functools import cached_property
from itertools import product

import numpy as np

from edelweissfe.utils.isoparametricshapes import newtonInverseMapping


class TopologyBase(ABC):
    """Abstract base class for element topologies in adaptive refinement.

    Besides the abstract element data, it provides the exact, integer-based queries that adaptive
    refinement decides everything with: where an element's own nodes sit on the reference lattice,
    which of them are corners, and which faces and edges of the element contain a given lattice point.
    """

    @cached_property
    def dim(self) -> int:
        """The spatial dimension of the reference element (3 for HEX20, 2 for QUAD8)."""
        return int(np.asarray(self.reference_node_param()).shape[1])

    @cached_property
    def n_nodes(self) -> int:
        """The number of nodes of the element (20 for HEX20, 8 for QUAD8)."""
        return len(self.reference_node_param())

    @cached_property
    def reference_node_lattice(self) -> np.ndarray:
        """The element's own nodes on the reference lattice with two units per edge: reference
        coordinate :math:`\\xi \\in \\{-1, 0, 1\\}` becomes the integer :math:`\\xi + 1 \\in \\{0, 1, 2\\}`."""
        return np.rint(np.asarray(self.reference_node_param(), dtype=float)).astype(int) + 1

    @cached_property
    def corner_slots(self) -> list:
        """The local slots of the corner nodes: every lattice coordinate at 0 or 2."""
        return [slot for slot, p in enumerate(self.reference_node_lattice) if all(v != 1 for v in p)]

    @cached_property
    def _fixedCoordinatesOfEntities(self) -> list:
        """Every face and edge as ``(kind, node slots, ((axis, value), ...))``: the lattice coordinates
        that are the same for all of its nodes. A point lies on the entity iff it shares all of them.

        In 2D the faces are the edges, in the other slot order (``[a, b, m]`` instead of ``[a, m, b]``),
        so only the edges are listed: a point on the boundary lies on an ``"edge"`` of a 2D element."""
        entities = []
        kinds = (("face", self.faces), ("edge", self.edges)) if self.dim == 3 else (("edge", self.edges),)
        for kind, collection in kinds:
            for entity in collection:
                rows = self.reference_node_lattice[list(entity)]
                fixed = tuple((k, int(rows[0, k])) for k in range(rows.shape[1]) if np.all(rows[:, k] == rows[0, k]))
                entities.append((kind, tuple(entity), fixed))
        return entities

    def entities_containing(self, point, extent: int) -> list:
        """The faces and edges (as node-slot tuples; in 2D the edges) of an element that contain a lattice point.

        Parameters
        ----------
        point
            Integer coordinates of the point, with the element spanning ``[0, extent]`` per axis.
        extent
            The element's edge length on that lattice (even).
        """
        return [
            entity
            for _, entity, fixed in self._fixedCoordinatesOfEntities
            if all(point[k] * 2 == value * extent for k, value in fixed)
        ]

    def lowest_entity_containing(self, point, extent: int) -> tuple:
        """The lowest-dimensional entity of an element containing a lattice point, as
        ``(kind, entity)``: ``("vertex", corner slot)``, ``("edge", node slots)``, ``("face", node
        slots)`` or ``("interior", None)``. See :meth:`entities_containing` for the arguments."""
        onBoundary = sum(1 for v in point if v == 0 or v == extent)
        if onBoundary == self.dim:
            doubled = tuple(2 * v // extent for v in point)
            return "vertex", next(s for s in self.corner_slots if tuple(self.reference_node_lattice[s]) == doubled)
        if onBoundary == 0:
            return "interior", None
        # named by the entity's own dimension: in 2D, a point on one boundary line lies on an edge
        wanted = "edge" if self.dim - onBoundary == 1 else "face"
        return wanted, next(
            entity
            for kind, entity, fixed in self._fixedCoordinatesOfEntities
            if kind == wanted and all(point[k] * 2 == value * extent for k, value in fixed)
        )

    @property
    @abstractmethod
    def faces(self) -> list:
        """List of faces, each defined by a list of local node indices: the codimension-one boundary
        entities (in 2D, the edges)."""

    @property
    @abstractmethod
    def edges(self) -> list:
        """List of edges, each defined by a list of local node indices."""

    @property
    @abstractmethod
    def faceid_to_face(self) -> dict:
        """Mapping from external FaceID to internal face index."""

    @abstractmethod
    def reference_node_param(self) -> np.ndarray:
        """Parametric coordinates of this element's own nodes, in the same reference cube and the
        same local order that :meth:`subdivision_children_param` uses.

        Refinement needs this to recognise, in exact arithmetic, that a child node has landed on a
        node the parent already owns -- so it reuses that node instead of minting a second one at
        the same place."""

    @abstractmethod
    def shape_functions(self, *params) -> np.ndarray:
        """Evaluate shape functions at parametric coordinates."""

    @abstractmethod
    def shape_functions_exact(self, *params) -> list:
        """The shape functions at an exact rational reference point.

        Parameters
        ----------
        params
            The reference coordinates, as :class:`~fractions.Fraction` (or integers).

        Returns
        -------
        list
            One :class:`~fractions.Fraction` per node, in local node order.
        """

    @abstractmethod
    def shape_functions_and_grad(self, *params) -> tuple:
        """Evaluate shape functions and their analytic gradient w.r.t. parametric coordinates."""

    def subdivision_children_param(self, n: int) -> list:
        """Parametric coordinates (in the parent's reference domain) of the nodes of each of the ``n**dim``
        children of an ``n``-per-axis split, in subdivision order (``ix * n * n + iy * n + iz``, in 2D
        ``ix * n + iy``: x outer). Child ``(i, j, ...)`` is the reference element scaled by ``1 / n`` and
        moved to cell ``(i, j, ...)``, so its nodes are in the topology's own local order."""
        reference = np.asarray(self.reference_node_param(), dtype=float)
        return [(2 * np.asarray(position) + reference + 1) / n - 1 for position in product(range(n), repeat=self.dim)]

    def face_child_indices(self, face_index: int, n: int) -> list:
        """The child indices (in :meth:`subdivision_children_param` order) that tile a given face (in 2D:
        edge) of an ``n``-per-axis split: the children on that face's side of the split. Each covers the
        face with its same local face, and hence the same faceID."""
        rows = self.reference_node_lattice[list(self.faces[face_index])]
        axis = next(k for k in range(self.dim) if np.all(rows[:, k] == rows[0, k]))
        side = n - 1 if rows[0, axis] == 2 else 0
        return [index for index, position in enumerate(product(range(n), repeat=self.dim)) if position[axis] == side]

    def inverse_map(self, point, elementNodeCoords, itmax=30) -> np.ndarray:
        """Map a physical point into the reference domain of an element, by the one Newton inverse mapping of
        EdelweissFE (:func:`edelweissfe.utils.isoparametricshapes.newtonInverseMapping`, on this topology's node
        ordering) from the element center; the result is clipped to ``[-1.2, 1.2]`` per axis if it does not converge.
        The convergence criterion is the dimensionless one of the shared mapping.

        Parameters
        ----------
        point
            The physical coordinates of the point.
        elementNodeCoords
            The element's node coordinates, in local order.
        itmax
            The maximum number of Newton iterations.
        """
        xi, converged = newtonInverseMapping(
            lambda xi_: self.shape_functions_and_grad(*xi_), elementNodeCoords, point, np.zeros(self.dim), itmax
        )
        return xi if converged else np.clip(xi, -1.2, 1.2)

    def subdivide(self, parent_coords: np.ndarray, n: int) -> list:
        """The physical node coordinates of the children of an ``n``-per-axis split of one parent element,
        in :meth:`subdivision_children_param` order: the parent's isoparametric map of each child's nodes."""
        parent_coords = np.asarray(parent_coords, dtype=float)
        return [
            np.array([self.shape_functions(*p) @ parent_coords for p in childParam])
            for childParam in self.subdivision_children_param(n)
        ]
