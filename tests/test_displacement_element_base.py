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

from edelweissfe.config.materiallibrary import getMaterialClass
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


# a distorted quadrilateral with straight edges, so that its map from natural coordinates is bilinear
# for both node counts: the Quad8 has its midside nodes at the midpoints of the edges
_CORNERS = np.array([[0.0, 0.0], [2.0, -0.3], [2.6, 1.4], [-0.2, 1.0]])
_DISTORTED = {4: _CORNERS, 8: np.vstack([_CORNERS, 0.5 * (_CORNERS + np.roll(_CORNERS, -1, axis=0))])}

# 2D element types whose rule integrates the bilinear map and its Jacobian exactly
_FULLY_INTEGRATED_2D = ["CPE4", "CPS4", "CPE4E", "CPE8", "CPE8R", "CPS8", "CPE4TL", "CPE8TL", "CPE8RTL"]
_DENSITY = 2.5
_THICKNESS = 0.5


def _distortedElement(elementType: str):
    elementClass = DisplacementTLElement if elementType.endswith("TL") else DisplacementElement
    element = elementClass(elementType, 1)
    coordinates = _DISTORTED[element.nNodes]
    element.setNodes([Node(i + 1, c) for i, c in enumerate(coordinates)])
    element.setProperties(np.array([_THICKNESS]))
    element.setMaterial(getMaterialClass("linearelastic", "edelweiss")(np.array([1000.0, 0.3, _DENSITY])))
    element.initializeElement()
    return element, coordinates


def _bilinearMap(xi: float, eta: float) -> np.ndarray:
    """x(xi, eta) of the distorted quadrilateral, xi from corner 0 to 1, eta from corner 0 to 3."""
    weights = 0.25 * np.array([(1 - xi) * (1 - eta), (1 + xi) * (1 - eta), (1 + xi) * (1 + eta), (1 - xi) * (1 + eta)])
    return weights @ _CORNERS


@pytest.mark.parametrize("elementType", _FULLY_INTEGRATED_2D)
def test_quadrature_points_lie_where_the_jacobian_is_evaluated(elementType):
    """On a distorted element, quadrature point i (shape functions) is the point of the Jacobian J[i]."""

    element, _ = _distortedElement(elementType)
    gaussPoints = np.polynomial.legendre.leggauss(round(np.sqrt(element.getNumberOfQuadraturePoints())))[0]
    candidates = [(xi, eta) for xi in gaussPoints for eta in gaussPoints]
    images = np.array([_bilinearMap(xi, eta) for xi, eta in candidates])

    h = 1e-6
    for i, x in enumerate(element.getCoordinatesAtQuadraturePoints()):
        distances = np.linalg.norm(images - x, axis=1)
        assert distances.min() < 1e-12
        xi, eta = candidates[distances.argmin()]
        # rows: d/dxi, d/deta; columns: x, y
        J = np.array(
            [
                (_bilinearMap(xi + h, eta) - _bilinearMap(xi - h, eta)) / (2 * h),
                (_bilinearMap(xi, eta + h) - _bilinearMap(xi, eta - h)) / (2 * h),
            ]
        )
        assert np.allclose(element.J[i], J, atol=1e-8)


@pytest.mark.parametrize("elementType", _FULLY_INTEGRATED_2D)
def test_mass_matrix_integrates_the_coordinates_of_a_distorted_element(elementType):
    """1^T M x_nodes = rho t (integral of x dA): the shape functions and the Jacobian of each
    quadrature point belong to the same point."""

    element, coordinates = _distortedElement(elementType)
    M = np.zeros((element.nDof, element.nDof))
    element.computeConsistentMassMatrix(M)

    # area and first moments of the polygon (shoelace formula)
    x, y = _CORNERS.T
    x1, y1 = np.roll(x, -1), np.roll(y, -1)
    cross = x * y1 - x1 * y
    area = cross.sum() / 2
    firstMoments = np.array([((x + x1) * cross).sum(), ((y + y1) * cross).sum()]) / 6

    rhoT = _DENSITY * _THICKNESS
    assert np.allclose((M @ coordinates.ravel()).reshape(-1, 2).sum(axis=0), rhoT * firstMoments)
    assert np.isclose(M.sum() / 2, rhoT * area)
