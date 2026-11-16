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
"""``TopologyBase`` adapter for the 8-node serendipity quadrilateral (QUAD8), the 2D counterpart of
:class:`~edelweissfe.adaptivity.hex20topology.Hex20Topology`. In 2D, the codimension-one entities
(``faces``: refinement balance, surfaces, root-mesh manifoldness) are the element's edges, so
``faces`` and ``edges`` list the same four edges, in Marmot faceID order and as
``[corner, midside, corner]`` respectively.
"""

import numpy as np

from edelweissfe.adaptivity.quad8shapefunctions import (
    EDGES,
    FACEID_TO_FACE,
    FACES,
    face_child_indices,
    quad8_box_coords,
    quad8_shape,
    quad8_shape_exact,
    quad8_shape_grad,
    subdivision_children_param,
)
from edelweissfe.adaptivity.topologybase import TopologyBase


class Quad8Topology(TopologyBase):
    """Topology provider for the 8-node serendipity quadrilateral (QUAD8)."""

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
        return quad8_box_coords(-1.0, 1.0, -1.0, 1.0)

    def subdivision_children_param(self, n: int) -> list:
        return subdivision_children_param(n)

    def face_child_indices(self, face_index: int, n: int) -> list:
        return face_child_indices(face_index, n)

    def shape_functions(self, *params) -> np.ndarray:
        return quad8_shape(*params)

    def shape_functions_exact(self, *params) -> list:
        return quad8_shape_exact(*params)

    def shape_functions_and_grad(self, *params) -> tuple:
        return quad8_shape_grad(*params)

    def inverse_map(self, point, elementNodeCoords, tol=1e-11, itmax=30) -> np.ndarray:
        coords = np.asarray(elementNodeCoords, dtype=float)
        xi = np.zeros(2)
        for _ in range(itmax):
            N, dN = self.shape_functions_and_grad(*xi)
            residual = N @ coords - point
            if float(np.linalg.norm(residual)) < tol:
                return xi
            try:
                dxi = np.linalg.solve(coords.T @ dN, residual)
            except np.linalg.LinAlgError:
                break
            xi -= dxi
            if float(np.linalg.norm(dxi)) < 1e-10:
                break
        return np.clip(xi, -1.2, 1.2)

    def subdivide(self, parent_coords: np.ndarray, n: int) -> list:
        parent_coords = np.asarray(parent_coords, dtype=float)
        return [
            np.array([self.shape_functions(*p) @ parent_coords for p in childParam])
            for childParam in self.subdivision_children_param(n)
        ]
