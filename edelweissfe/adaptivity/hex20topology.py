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

from edelweissfe.adaptivity.geometry import (
    bilinear_inverse,
    point_in_convex_quad,
    quadratic_edge_parameter,
)
from edelweissfe.adaptivity.hex20shapefunctions import (
    EDGES,
    FACEID_TO_FACE,
    FACES,
    face_child_octants,
    hex20_shape,
    hex20_shape_grad,
    quad8_shape,
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

    def element_face_corners(self, coords: np.ndarray) -> list:
        coords = np.asarray(coords, dtype=float)
        return [coords[[f[0], f[1], f[2], f[3]]] for f in self.faces]

    def hanging_weights(self, master_coords, slave_coord, kind: str) -> np.ndarray:
        mc = np.asarray(master_coords, dtype=float)
        p = np.asarray(slave_coord, dtype=float)
        if kind == "face":
            xi, eta = bilinear_inverse(p, mc[0:4])
            return quad8_shape(xi, eta)
        elif kind == "edge":
            ca, cm, cb = mc[0], mc[1], mc[2]
            t, _ = quadratic_edge_parameter(p, ca, cm, cb)
            return np.array([0.5 * t * (t - 1.0), 1.0 - t**2, 0.5 * t * (t + 1.0)])
        raise ValueError(f"Hex20Topology has no hanging-node entity of kind {kind!r} (expected 'edge' or 'face').")

    def classify_hanging_on_element(self, coarse_conn, registry, candidate_labels, tol=1e-8) -> list:
        coarse_conn = list(coarse_conn)
        coarse_set = set(coarse_conn)
        coords = registry.coordinates
        results = []

        for lab in candidate_labels:
            if lab in coarse_set:
                continue
            p = coords[lab]

            matched = False
            for edge in self.edges:
                ea, em, eb = (coarse_conn[edge[0]], coarse_conn[edge[1]], coarse_conn[edge[2]])
                _, dist = quadratic_edge_parameter(p, coords[ea], coords[em], coords[eb])
                if dist < tol:
                    results.append({"slave": lab, "kind": "edge", "masters": [ea, em, eb]})
                    matched = True
                    break
            if matched:
                continue

            for face in self.faces:
                fcorners = np.array([coords[coarse_conn[i]] for i in face[:4]])
                if point_in_convex_quad(p, fcorners):
                    results.append({"slave": lab, "kind": "face", "masters": [coarse_conn[i] for i in face]})
                    break

        return results
