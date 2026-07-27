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

import numpy as np


class TopologyBase(ABC):
    """Abstract base class for element topologies in adaptive refinement."""

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
    def subdivision_children_param(self, n: int) -> list:
        """Parametric coordinates (in the parent domain) of the nodes of each child element."""

    @abstractmethod
    def face_child_indices(self, face_index: int, n: int) -> list:
        """The child indices that tile a given face."""

    @abstractmethod
    def shape_functions(self, *params) -> np.ndarray:
        """Evaluate shape functions at parametric coordinates."""

    @abstractmethod
    def shape_functions_and_grad(self, *params) -> tuple:
        """Evaluate shape functions and their analytic gradient w.r.t. parametric coordinates."""

    @abstractmethod
    def inverse_map(self, point, elementNodeCoords, tol=1e-11, itmax=30) -> np.ndarray:
        """Map a physical point into the reference domain."""

    @abstractmethod
    def subdivide(self, parent_coords: np.ndarray, n: int) -> list:
        """Subdivide one parent element into children."""

    @abstractmethod
    def element_face_corners(self, coords: np.ndarray) -> list:
        """The 4 corner coordinates of each face of the element."""

    @abstractmethod
    def hanging_weights(self, master_coords, slave_coord, kind: str) -> np.ndarray:
        """Exact coarse-trace weights of a slave on its master entity.

        Parameters
        ----------
        master_coords
            Physical coordinates of the master entity's own nodes.
        slave_coord
            Physical coordinate of the hanging (slave) node.
        kind
            The master entity kind, as classified by :meth:`classify_hanging_on_element`
            (e.g. ``"edge"`` or ``"face"``).
        """

    @abstractmethod
    def classify_hanging_on_element(self, coarse_conn, registry, candidate_labels, tol=1e-8) -> list:
        """Classify which candidate nodes hang on a coarse element's faces/edges."""
