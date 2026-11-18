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
:class:`~edelweissfe.adaptivity.hex20topology.Hex20Topology`, in the Abaqus/Marmot CPS8 node order of
:mod:`~edelweissfe.elements._quad2dnodeordering` (corners 0-3 counter-clockwise from ``(-1, -1)``, then
the midsides of the edges 0-1, 1-2, 2-3, 3-0), whose shape functions it uses. In 2D the
codimension-one entities (``faces``: 2:1 balance, surfaces, root-mesh manifoldness) are the
element's edges, so ``faces`` and ``edges`` list the same four edges, in Marmot faceID order and as
``[corner, midside, corner]`` respectively.
"""

from fractions import Fraction

import numpy as np

from edelweissfe.adaptivity.topologybase import TopologyBase
from edelweissfe.elements._quad2dnodeordering import quad8DNdXi, quad8N, quad8NodeXi

#: Marmot Quad8 faceID (1-based) -> local slots ``[corner a, corner b, midside]``, from
#: MarmotFiniteElement2D.cpp ``Quad8::getBoundaryElementIndices``.
BOUNDARY_FACEID_INDICES = {1: [0, 1, 4], 2: [1, 2, 5], 3: [2, 3, 6], 4: [3, 0, 7]}
#: The four edges in faceID order, as ``[corner a, corner b, midside]``.
FACES = [BOUNDARY_FACEID_INDICES[faceID] for faceID in sorted(BOUNDARY_FACEID_INDICES)]
#: The same four edges as ``[corner a, midside, corner b]``.
EDGES = [[a, m, b] for a, b, m in FACES]
FACEID_TO_FACE = {faceID: faceID - 1 for faceID in BOUNDARY_FACEID_INDICES}


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
        return quad8NodeXi.astype(float)

    def shape_functions(self, *params) -> np.ndarray:
        return quad8N(*params)

    def shape_functions_exact(self, *params) -> list:
        return list(quad8N(*(Fraction(p) for p in params)))

    def shape_functions_and_grad(self, *params) -> tuple:
        return quad8N(*params), quad8DNdXi(*params).T
