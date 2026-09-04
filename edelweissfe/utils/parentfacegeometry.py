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

import numpy as np

"""
Parametric geometry of a *parent face*: the element face a contact facet was cut from, together
with its own (possibly quadratic) shape functions and the quadrature rules needed to integrate
over the facets tiling it.

This is what separates Gauss-point-to-segment contact from the node-based formulation. A node-based
scheme can only ever deliver a non-negative force to a node, because the force *is* the spring
force there. An integrated scheme evaluates a non-negative pressure at quadrature points and
distributes it with the parent face's own shape functions, so a node whose shape function is
negative over part of the face receives a *tensile* nodal force -- which is exactly the consistent
load of a uniform pressure on a serendipity face (:math:`-pA/12` at the corners) and exactly what a
per-node scheme cannot produce. Without the parent-face basis, quadrature alone changes nothing:
the flat facets' own linear shape functions are non-negative, so the corner nodes would keep
lifting off. See the contact theory documentation.

Canonical parent-face node orderings, and the parametric coordinates they carry:

* ``quad8``: corners ``c0..c3`` counter-cycling the face, then midsides ``m0..m3`` with ``mk``
  between ``ck`` and ``c(k+1)``; corners at :math:`(\\pm 1, \\pm 1)`, midsides at the edge centres.
* ``quad4``: corners ``c0..c3`` only.
* ``line3``: end nodes at :math:`\\xi = \\mp 1`, midside at :math:`\\xi = 0`.
* ``line2``: end nodes at :math:`\\xi = \\mp 1`.

The mapping from an element's local node indices to this canonical order is *derived* from the
surface element generator's own face tables rather than transcribed again -- see
:func:`~edelweissfe.generators.surfaceelementgenerator.canonicalParentFace`.
"""

#: Parametric coordinates of each canonical parent-face node ordering, shape ``(nNodes, nDim - 1)``.
PARENT_FACE_PARAMETRIC_COORDS = {
    "quad8": np.array(
        [
            [-1.0, -1.0],
            [+1.0, -1.0],
            [+1.0, +1.0],
            [-1.0, +1.0],
            [+0.0, -1.0],
            [+1.0, +0.0],
            [+0.0, +1.0],
            [-1.0, +0.0],
        ]
    ),
    "quad4": np.array([[-1.0, -1.0], [+1.0, -1.0], [+1.0, +1.0], [-1.0, +1.0]]),
    "line3": np.array([[-1.0], [+1.0], [0.0]]),
    "line2": np.array([[-1.0], [+1.0]]),
}


def parentFaceShapeFunctions(faceType: str, parametricCoords: np.ndarray) -> np.ndarray:
    """The parent face's shape functions at one parametric point.

    Parameters
    ----------
    faceType
        One of the keys of :data:`PARENT_FACE_PARAMETRIC_COORDS`.
    parametricCoords
        The evaluation point, shape ``(1,)`` for a line face or ``(2,)`` for a quad face.

    Returns
    -------
    numpy.ndarray
        The shape function values, ordered like the canonical node ordering and summing to one. For
        ``quad8`` and ``line3`` these take *negative* values near the corners, which is the whole
        point of this module.
    """

    nodes = PARENT_FACE_PARAMETRIC_COORDS[faceType]

    if faceType == "quad8":
        xi, eta = parametricCoords
        N = np.empty(8)
        for i in range(4):
            a, b = nodes[i]
            N[i] = 0.25 * (1.0 + xi * a) * (1.0 + eta * b) * (xi * a + eta * b - 1.0)
        for i in range(4, 8):
            a, b = nodes[i]
            if a == 0.0:
                N[i] = 0.5 * (1.0 - xi**2) * (1.0 + eta * b)
            else:
                N[i] = 0.5 * (1.0 + xi * a) * (1.0 - eta**2)
        return N

    if faceType == "quad4":
        xi, eta = parametricCoords
        return np.array([0.25 * (1.0 + xi * a) * (1.0 + eta * b) for a, b in nodes])

    if faceType == "line3":
        (xi,) = parametricCoords
        return np.array([0.5 * xi * (xi - 1.0), 0.5 * xi * (xi + 1.0), 1.0 - xi**2])

    if faceType == "line2":
        (xi,) = parametricCoords
        return np.array([0.5 * (1.0 - xi), 0.5 * (1.0 + xi)])

    raise ValueError(f"parentFaceShapeFunctions: unsupported parent face type '{faceType}'.")


#: Barycentric quadrature rules on a triangle, keyed by point count: (points, weights). The weights
#: sum to one, so an integral is ``facetMeasure * sum(w_q * f(b_q))``. 1 point is exact to degree 1,
#: 3 points to degree 2, 6 points to degree 4 (Dunavant).
_TRIANGLE_RULES = {
    1: (np.array([[1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0]]), np.array([1.0])),
    3: (
        np.array(
            [
                [2.0 / 3.0, 1.0 / 6.0, 1.0 / 6.0],
                [1.0 / 6.0, 2.0 / 3.0, 1.0 / 6.0],
                [1.0 / 6.0, 1.0 / 6.0, 2.0 / 3.0],
            ]
        ),
        np.array([1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0]),
    ),
    6: (
        np.array(
            [
                [0.816847572980459, 0.091576213509771, 0.091576213509771],
                [0.091576213509771, 0.816847572980459, 0.091576213509771],
                [0.091576213509771, 0.091576213509771, 0.816847572980459],
                [0.108103018168070, 0.445948490915965, 0.445948490915965],
                [0.445948490915965, 0.108103018168070, 0.445948490915965],
                [0.445948490915965, 0.445948490915965, 0.108103018168070],
            ]
        ),
        np.array(
            [
                0.109951743655322,
                0.109951743655322,
                0.109951743655322,
                0.223381589678011,
                0.223381589678011,
                0.223381589678011,
            ]
        ),
    ),
}

#: Gauss-Legendre rules on a Line2 facet, in the facet's own barycentric coordinates, keyed by point
#: count. Weights sum to one, as for the triangle rules. Exact to degree 2n - 1.
_LINE_RULES = {}
for _n in (1, 2, 3):
    _positions, _weights = np.polynomial.legendre.leggauss(_n)
    _LINE_RULES[_n] = (
        np.column_stack([0.5 * (1.0 - _positions), 0.5 * (1.0 + _positions)]),
        _weights / 2.0,
    )


def facetQuadratureRule(nFacetNodes: int, nPoints: int) -> tuple[np.ndarray, np.ndarray]:
    """The quadrature rule for one contact facet, in the facet's own barycentric coordinates.

    Parameters
    ----------
    nFacetNodes
        3 for a Tria3 facet, 2 for a Line2 facet.
    nPoints
        The number of quadrature points requested.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray]
        The barycentric coordinates, shape ``(nPoints, nFacetNodes)``, and the weights, which sum to
        one -- so an integral over the facet is ``facetMeasure * sum(weights * f(points))``.
    """

    rules = _TRIANGLE_RULES if nFacetNodes == 3 else _LINE_RULES
    if nPoints not in rules:
        raise ValueError(
            f"facetQuadratureRule: no {nPoints}-point rule for a {nFacetNodes}-node facet. "
            f"Available: {sorted(rules.keys())}."
        )
    return rules[nPoints]
