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

"""``TopologyBase`` adapter for the 20-node serendipity hexahedron (HEX20).

All actual math (shape functions, faces/edges, subdivision) lives in
:mod:`~edelweissfe.adaptivity.hex20shapefunctions`; this module only wires that math into the
generic :class:`~edelweissfe.adaptivity.topologybase.TopologyBase` contract that
:class:`~edelweissfe.adaptivity.refinement.AdaptiveMesh` and the state-transfer strategies are
written against. A second element family (e.g. a linear HEX8, or a 2D QUAD8) plugs into AMR by
providing its own such adapter -- this file is the template to copy.
"""

import numpy as np

from edelweissfe.adaptivity.hex20shapefunctions import (
    EDGES,
    FACEID_TO_FACE,
    FACES,
    face_child_octants,
    hex20_box_coords,
    hex20_shape,
    hex20_shape_grad,
    subdivision_children_param,
)
from edelweissfe.adaptivity.topologybase import TopologyBase


class Hex20Topology(TopologyBase):
    """Topology provider for the 20-node serendipity hexahedron (HEX20)."""

    @property
    def faces(self) -> list:
        return FACES

    @property
    def edges(self) -> list:
        return EDGES

    @property
    def faceid_to_face(self) -> dict:
        return FACEID_TO_FACE

    def reference_node_param(self) -> np.ndarray:
        return hex20_box_coords(-1.0, 1.0, -1.0, 1.0, -1.0, 1.0)

    def subdivision_children_param(self, n: int) -> list:
        return subdivision_children_param(n)

    def face_child_indices(self, face_index: int, n: int) -> list:
        return face_child_octants(face_index, n)

    def shape_functions(self, *params) -> np.ndarray:
        return hex20_shape(*params)

    def shape_functions_and_grad(self, *params) -> tuple:
        return hex20_shape_grad(*params)

    def inverse_map(self, point, elementNodeCoords, tol=1e-11, itmax=30) -> np.ndarray:
        coords = np.asarray(elementNodeCoords, dtype=float)
        xi = np.zeros(3)
        for _ in range(itmax):
            N, dN = self.shape_functions_and_grad(*xi)
            x = N @ coords
            residual = x - point
            if float(np.linalg.norm(residual)) < tol:
                return xi
            jac = coords.T @ dN  # jac[i, k] = dx_i / dxi_k
            try:
                dxi = np.linalg.solve(jac, residual)
            except np.linalg.LinAlgError:
                break
            xi -= dxi
            if float(np.linalg.norm(dxi)) < 1e-10:
                break
        return np.clip(xi, -1.2, 1.2)

    def subdivide(self, parent_coords: np.ndarray, n: int) -> list:
        parent_coords = np.asarray(parent_coords, dtype=float)
        children = []
        for child_param in self.subdivision_children_param(n):
            phys = np.array([self.shape_functions(*p) @ parent_coords for p in child_param])
            children.append(phys)
        return children
