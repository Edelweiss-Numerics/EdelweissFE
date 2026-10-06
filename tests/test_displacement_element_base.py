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
"""The shared base of the small-strain and the total-Lagrangian displacement elements."""

import numpy as np
import pytest

from edelweissfe.elements.displacementelement.element import DisplacementElement
from edelweissfe.elements.displacementtlelement.element import DisplacementTLElement
from edelweissfe.points.node import Node

_NODES = {
    "CPE4": [[0, 0], [2, 0], [2, 1], [0, 1]],
    "CPE8": [[0, 0], [2, 0], [2, 1], [0, 1], [1, 0], [2, 0.5], [1, 1], [0, 0.5]],
    "C3D8": [[0, 0, 0], [2, 0, 0], [2, 1, 0], [0, 1, 0], [0, 0, 3], [2, 0, 3], [2, 1, 3], [0, 1, 3]],
}


@pytest.mark.parametrize("elementType", sorted(_NODES))
def test_quadrature_point_coordinates_are_one_row_per_point_inside_the_element(elementType):
    element = DisplacementElement(elementType, 1)
    coordinates = np.array(_NODES[elementType], dtype=float)
    element.setNodes([Node(i + 1, c) for i, c in enumerate(coordinates)])

    qpCoordinates = element.getCoordinatesAtQuadraturePoints()

    assert qpCoordinates.shape == (element.getNumberOfQuadraturePoints(), coordinates.shape[1])
    # every quadrature point of these box elements lies strictly inside the box
    assert np.all(qpCoordinates > coordinates.min(axis=0)) and np.all(qpCoordinates < coordinates.max(axis=0))
    # a symmetric rule on a box: the quadrature points' centroid is the box centre
    assert np.allclose(qpCoordinates.mean(axis=0), coordinates.mean(axis=0))


def test_a_subclass_of_an_element_accepts_its_element_types():
    class CustomDisplacementElement(DisplacementElement):
        pass

    assert CustomDisplacementElement("CPE4", 1).elType == "CPE4"


def test_an_element_type_of_the_other_formulation_is_rejected():
    with pytest.raises(Exception):
        DisplacementElement("CPE4TL", 1)
    with pytest.raises(Exception):
        DisplacementTLElement("CPE4", 1)
