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
written against. Another element family plugs into AMR by implementing the abstract members of
:class:`~edelweissfe.adaptivity.topologybase.TopologyBase` -- its node coordinates, faces and edges,
the faceID table and the shape functions -- as this module and
:mod:`~edelweissfe.adaptivity.quad8topology` do.
"""

import numpy as np

from edelweissfe.adaptivity.hex20shapefunctions import (
    EDGES,
    FACEID_TO_FACE,
    FACES,
    hex20_box_coords,
    hex20_shape,
    hex20_shape_exact,
    hex20_shape_grad,
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

    def shape_functions(self, *params) -> np.ndarray:
        return hex20_shape(*params)

    def shape_functions_exact(self, *params) -> list:
        return hex20_shape_exact(*params)

    def shape_functions_and_grad(self, *params) -> tuple:
        return hex20_shape_grad(*params)
