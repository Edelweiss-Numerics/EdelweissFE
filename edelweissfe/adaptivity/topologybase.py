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

import numpy as np


class TopologyBase(ABC):
    """Abstract base class for element topologies in adaptive refinement.

    Besides the abstract element data, it provides the exact, integer-based queries that adaptive
    refinement decides everything with: where an element's own nodes sit on the reference lattice,
    which of them are corners, and which faces and edges of the element contain a given lattice point.
    """

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
        that are the same for all of its nodes. A point lies on the entity iff it shares all of them."""
        entities = []
        for kind, collection in (("face", self.faces), ("edge", self.edges)):
            for entity in collection:
                rows = self.reference_node_lattice[list(entity)]
                fixed = tuple((k, int(rows[0, k])) for k in range(rows.shape[1]) if np.all(rows[:, k] == rows[0, k]))
                entities.append((kind, tuple(entity), fixed))
        return entities

    def entities_containing(self, point, extent: int) -> list:
        """The faces and edges (as node-slot tuples) of an element that contain a lattice point.

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
        if onBoundary == 3:
            doubled = tuple(2 * v // extent for v in point)
            return "vertex", next(s for s in self.corner_slots if tuple(self.reference_node_lattice[s]) == doubled)
        if onBoundary == 0:
            return "interior", None
        wanted = "edge" if onBoundary == 2 else "face"
        return wanted, next(
            entity
            for kind, entity, fixed in self._fixedCoordinatesOfEntities
            if kind == wanted and all(point[k] * 2 == value * extent for k, value in fixed)
        )

    @property
    @abstractmethod
    def faces(self) -> list:
        """List of faces, each defined by a list of local node indices."""

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
    def subdivision_children_param(self, n: int) -> list:
        """Parametric coordinates (in the parent domain) of the nodes of each child element."""

    @abstractmethod
    def face_child_indices(self, face_index: int, n: int) -> list:
        """The child indices that tile a given face."""

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

    @abstractmethod
    def inverse_map(self, point, elementNodeCoords, tol=1e-11, itmax=30) -> np.ndarray:
        """Map a physical point into the reference domain."""

    @abstractmethod
    def subdivide(self, parent_coords: np.ndarray, n: int) -> list:
        """Subdivide one parent element into children."""
