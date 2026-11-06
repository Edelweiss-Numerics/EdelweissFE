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
"""
Shape functions, their parametric derivatives and the inverse isoparametric mapping of the standard element shapes,
identified by their Ensight names (the ``ensightType`` of an element).

The node orderings are the ones of Marmot (and Abaqus):

* ``bar2``: :math:`\\xi = -1, +1`; ``bar3``: :math:`\\xi = -1, +1, 0` (end, end, mid).
* ``quad4``: corners counter-clockwise from :math:`(-1, -1)`; ``quad8``: corners, then the midsides, the midside
  ``4 + k`` between the corners ``k`` and ``k + 1``.
* ``hexa8``: the four corners of the face :math:`\\zeta = -1`, then those of :math:`\\zeta = +1`, each
  counter-clockwise from :math:`(-1, -1)`; ``hexa20``: corners, the midsides of the bottom face, of the top face, and
  of the vertical edges.
* ``tetra4``: :math:`N = (1 - \\xi - \\eta - \\zeta, \\xi, \\eta, \\zeta)`.

Used to embed points (e.g., the nodes or integration points of reinforcement bars) in host elements, see
:mod:`edelweissfe.utils.embedding`.
"""

import numpy as np

_QUAD_CORNERS = np.array([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]])
_QUAD_MIDSIDES = np.array([[0.0, -1.0], [1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])
_HEXA_CORNERS = np.array(
    [
        [-1.0, -1.0, -1.0],
        [1.0, -1.0, -1.0],
        [1.0, 1.0, -1.0],
        [-1.0, 1.0, -1.0],
        [-1.0, -1.0, 1.0],
        [1.0, -1.0, 1.0],
        [1.0, 1.0, 1.0],
        [-1.0, 1.0, 1.0],
    ]
)
_HEXA_MIDSIDES = np.array(
    [
        [0.0, -1.0, -1.0],
        [1.0, 0.0, -1.0],
        [0.0, 1.0, -1.0],
        [-1.0, 0.0, -1.0],
        [0.0, -1.0, 1.0],
        [1.0, 0.0, 1.0],
        [0.0, 1.0, 1.0],
        [-1.0, 0.0, 1.0],
        [-1.0, -1.0, 0.0],
        [1.0, -1.0, 0.0],
        [1.0, 1.0, 0.0],
        [-1.0, 1.0, 0.0],
    ]
)

#: Parametric coordinates of the nodes of each supported shape.
NODE_PARAMETRIC_COORDINATES = {
    "bar2": np.array([[-1.0], [1.0]]),
    "bar3": np.array([[-1.0], [1.0], [0.0]]),
    "quad4": _QUAD_CORNERS,
    "quad8": np.vstack([_QUAD_CORNERS, _QUAD_MIDSIDES]),
    "hexa8": _HEXA_CORNERS,
    "hexa20": np.vstack([_HEXA_CORNERS, _HEXA_MIDSIDES]),
    "tetra4": np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]),
}

#: The parametric dimension of each supported shape.
PARAMETRIC_DIMENSION = {shape: coords.shape[1] for shape, coords in NODE_PARAMETRIC_COORDINATES.items()}


def _lagrangeTensorProductLinear(xi: np.ndarray, nodes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Multilinear shape functions of the corner nodes and their derivatives."""
    factors = 0.5 * (1.0 + nodes * xi)  # (nNodes, dim)
    N = np.prod(factors, axis=1)
    dN = np.empty_like(nodes)
    for d in range(nodes.shape[1]):
        others = np.prod(np.delete(factors, d, axis=1), axis=1)
        dN[:, d] = 0.5 * nodes[:, d] * others
    return N, dN


def _serendipity(xi: np.ndarray, corners: np.ndarray, midsides: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Quadratic serendipity shape functions (quad8, hexa20) and their derivatives."""
    dim = corners.shape[1]
    nC, nM = corners.shape[0], midsides.shape[0]
    N = np.empty(nC + nM)
    dN = np.empty((nC + nM, dim))

    # corners: prod(1 + xi_i xi) / 2^dim * (sum(xi_i xi) - (dim - 1))
    for a, c in enumerate(corners):
        f = 1.0 + c * xi
        p = np.prod(f) / 2**dim
        s = np.dot(c, xi) - (dim - 1)
        N[a] = p * s
        for d in range(dim):
            dp = c[d] * np.prod(np.delete(f, d)) / 2**dim
            dN[a, d] = dp * s + p * c[d]

    # midsides: the coordinate with xi_i = 0 enters as (1 - xi^2), the others as (1 + xi_i xi) / 2
    for m, c in enumerate(midsides):
        a = nC + m
        zero = int(np.flatnonzero(c == 0.0)[0])
        g = np.where(c == 0.0, 1.0 - xi**2, 0.5 * (1.0 + c * xi))
        N[a] = np.prod(g)
        for d in range(dim):
            dg = -2.0 * xi[d] if d == zero else 0.5 * c[d]
            dN[a, d] = dg * np.prod(np.delete(g, d))
    return N, dN


def shapeFunctionsAndDerivatives(shape: str, xi) -> tuple[np.ndarray, np.ndarray]:
    """The shape functions and their derivatives w.r.t. the parametric coordinates.

    Parameters
    ----------
    shape
        The Ensight name of the shape, e.g., ``quad8``.
    xi
        The parametric coordinates.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        The shape functions ``(nNodes,)`` and their derivatives ``(nNodes, parametric dimension)``.
    """
    xi = np.atleast_1d(np.asarray(xi, dtype=float))
    if shape in ("quad4", "hexa8"):
        return _lagrangeTensorProductLinear(xi, NODE_PARAMETRIC_COORDINATES[shape])
    if shape == "quad8":
        return _serendipity(xi, _QUAD_CORNERS, _QUAD_MIDSIDES)
    if shape == "hexa20":
        return _serendipity(xi, _HEXA_CORNERS, _HEXA_MIDSIDES)
    if shape == "bar2":
        return np.array([0.5 * (1 - xi[0]), 0.5 * (1 + xi[0])]), np.array([[-0.5], [0.5]])
    if shape == "bar3":
        x = xi[0]
        return np.array([0.5 * (x - 1) * x, 0.5 * (x + 1) * x, 1 - x * x]), np.array([[x - 0.5], [x + 0.5], [-2 * x]])
    if shape == "tetra4":
        N = np.array([1.0 - xi.sum(), xi[0], xi[1], xi[2]])
        dN = np.vstack([-np.ones(3), np.eye(3)])
        return N, dN
    raise NotImplementedError(f"Shape functions of '{shape}' elements are not implemented.")


def shapeFunctions(shape: str, xi) -> np.ndarray:
    """The shape functions at the parametric coordinates ``xi``, see :func:`shapeFunctionsAndDerivatives`."""
    return shapeFunctionsAndDerivatives(shape, xi)[0]


def isInside(shape: str, xi: np.ndarray, tolerance: float = 1e-8) -> bool:
    """Whether the parametric coordinates ``xi`` lie inside of the parent domain (with a tolerance)."""
    if shape == "tetra4":
        return bool(np.min(xi) >= -tolerance and np.sum(xi) <= 1.0 + tolerance)
    return bool(np.max(np.abs(xi)) <= 1.0 + tolerance)


def parentDomainCenter(shape: str) -> np.ndarray:
    """The parametric coordinates of the center of the parent domain."""
    return NODE_PARAMETRIC_COORDINATES[shape].mean(axis=0)


def inverseMapping(shape: str, nodeCoordinates: np.ndarray, x: np.ndarray, maxIterations: int = 50) -> np.ndarray:
    """The parametric coordinates of a point ``x`` in an element (inverse isoparametric mapping, Newton).

    Parameters
    ----------
    shape
        The Ensight name of the shape; its parametric dimension must equal the spatial dimension.
    nodeCoordinates
        The nodal coordinates ``(nNodes, nDim)``.
    x
        The point.
    maxIterations
        The maximum number of Newton iterations.

    Returns
    -------
    np.ndarray
        The parametric coordinates, or ``None`` if the iteration did not converge (e.g., for a point far outside of a
        distorted element).
    """
    xi = parentDomainCenter(shape)
    scale = np.max(np.ptp(nodeCoordinates, axis=0))
    for _ in range(maxIterations):
        N, dN = shapeFunctionsAndDerivatives(shape, xi)
        residual = N @ nodeCoordinates - x
        J = nodeCoordinates.T @ dN
        try:
            dXi = -np.linalg.solve(J, residual)
        except np.linalg.LinAlgError:
            return None
        xi = xi + dXi
        if np.linalg.norm(dXi) < 1e-13 and np.linalg.norm(residual) < 1e-10 * scale:
            return xi
        if np.max(np.abs(xi)) > 1e3:
            return None
    return None
