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
#  ---------------------------------------------------------------------
"""The shape functions of the quadrilaterals and hexahedra, and their node coordinates."""

import numpy as np
import pytest

from edelweissfe.elements._hexa3dnodeordering import (
    hexa8DNdXi,
    hexa8N,
    hexa8NodeXi,
    hexa20DNdXi,
    hexa20N,
    hexa20NodeXi,
)
from edelweissfe.elements._quad2dnodeordering import (
    quad4DNdXi,
    quad4N,
    quad4NodeXi,
    quad8DNdXi,
    quad8N,
    quad8NodeXi,
)

# shape -> (natural coordinates of the nodes, shape functions, their derivatives)
SHAPES = {
    "quad4": (quad4NodeXi, quad4N, quad4DNdXi),
    "quad8": (quad8NodeXi, quad8N, quad8DNdXi),
    "hexa8": (hexa8NodeXi, hexa8N, hexa8DNdXi),
    "hexa20": (hexa20NodeXi, hexa20N, hexa20DNdXi),
}


@pytest.mark.parametrize("shape", SHAPES)
def test_shape_function_j_is_one_at_node_j_and_zero_at_the_others(shape):
    nodeXi, N, _ = SHAPES[shape]
    assert np.array_equal(N(*nodeXi.T), np.eye(len(nodeXi)))


@pytest.mark.parametrize("shape", SHAPES)
def test_derivatives_of_the_shape_functions(shape):
    nodeXi, N, dNdXi = SHAPES[shape]
    xi = np.array([0.3, -0.7, 0.45])[: nodeXi.shape[1]]
    h = 1e-6
    finiteDifferences = [(N(*(xi + h * e)) - N(*(xi - h * e))) / (2 * h) for e in np.eye(len(xi))]
    assert np.allclose(dNdXi(*xi), finiteDifferences, atol=1e-8)
    # a partition of unity
    assert np.isclose(N(*xi).sum(), 1.0)
