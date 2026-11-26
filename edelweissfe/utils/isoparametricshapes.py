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
* ``tria3``, ``tetra4``: the corners at the origin and on the parametric axes, :math:`N = L` with the area
  (volume) coordinates :math:`L = (1 - \\xi - \\eta, \\xi, \\eta)`, :math:`(1 - \\xi - \\eta - \\zeta, \\xi, \\eta, \\zeta)`;
  ``tria6``, ``tetra10``: the corners, then the mid nodes of the edges in the order of Abaqus and Marmot (triangle:
  1-2, 2-3, 3-1; tetrahedron: 1-2, 2-3, 3-1, 1-4, 2-4, 3-4), :math:`N_i = L_i (2 L_i - 1)` and
  :math:`N_{ij} = 4 L_i L_j`.

To add a shape: its node coordinates in :data:`NODE_PARAMETRIC_COORDINATES`, a branch in
:func:`shapeFunctionsAndDerivatives`, the parent domain in :func:`parentDomainExcess`, and a test entry (the tests of
``tests/test_isoparametricshapes_embedding.py`` run over all shapes).

Used to embed points (e.g., the nodes or integration points of reinforcement bars) in host elements, see
:mod:`edelweissfe.utils.embedding`.
"""

from collections.abc import Callable

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

_TRIA_CORNERS = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
_TETRA_CORNERS = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
#: The corner pairs of the mid nodes of the quadratic simplices (0-based), in node order
_TRIA_EDGES = ((0, 1), (1, 2), (2, 0))
_TETRA_EDGES = ((0, 1), (1, 2), (2, 0), (0, 3), (1, 3), (2, 3))


def _midNodes(corners: np.ndarray, edges) -> np.ndarray:
    return np.array([0.5 * (corners[i] + corners[j]) for i, j in edges])


#: Parametric coordinates of the nodes of each supported shape.
NODE_PARAMETRIC_COORDINATES = {
    "bar2": np.array([[-1.0], [1.0]]),
    "bar3": np.array([[-1.0], [1.0], [0.0]]),
    "quad4": _QUAD_CORNERS,
    "quad8": np.vstack([_QUAD_CORNERS, _QUAD_MIDSIDES]),
    "hexa8": _HEXA_CORNERS,
    "hexa20": np.vstack([_HEXA_CORNERS, _HEXA_MIDSIDES]),
    "tria3": _TRIA_CORNERS,
    "tria6": np.vstack([_TRIA_CORNERS, _midNodes(_TRIA_CORNERS, _TRIA_EDGES)]),
    "tetra4": _TETRA_CORNERS,
    "tetra10": np.vstack([_TETRA_CORNERS, _midNodes(_TETRA_CORNERS, _TETRA_EDGES)]),
}

#: The simplex shapes (parent domain: L >= 0 with the area / volume coordinates L).
SIMPLEX_SHAPES = ("tria3", "tria6", "tetra4", "tetra10")

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


def _simplex(xi: np.ndarray, edges) -> tuple[np.ndarray, np.ndarray]:
    """Linear (``edges=None``) or quadratic simplex shape functions from the area / volume coordinates."""
    dim = xi.shape[0]
    L = np.concatenate([[1.0 - xi.sum()], xi])
    dL = np.vstack([-np.ones(dim), np.eye(dim)])  # dL_i / dxi
    if edges is None:
        return L, dL
    N = np.concatenate([L * (2.0 * L - 1.0), [4.0 * L[i] * L[j] for i, j in edges]])
    dN = np.vstack([(4.0 * L[:, None] - 1.0) * dL, np.array([4.0 * (L[i] * dL[j] + L[j] * dL[i]) for i, j in edges])])
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
    if shape in ("tria3", "tetra4"):
        return _simplex(xi, None)
    if shape == "tria6":
        return _simplex(xi, _TRIA_EDGES)
    if shape == "tetra10":
        return _simplex(xi, _TETRA_EDGES)
    raise NotImplementedError(f"Shape functions of '{shape}' elements are not implemented.")


def shapeFunctions(shape: str, xi) -> np.ndarray:
    """The shape functions at the parametric coordinates ``xi``, see :func:`shapeFunctionsAndDerivatives`."""
    return shapeFunctionsAndDerivatives(shape, xi)[0]


#: The tolerance of "inside the parent domain" (in parametric coordinates): a point on a host element's boundary is
#: inside of both neighbours
INSIDE_TOLERANCE = 1e-8


def parentDomainExcess(shape: str, xi: np.ndarray) -> float:
    """By how much the parametric coordinates exceed the parent domain; negative if strictly inside."""
    xi = np.asarray(xi, dtype=float)
    if shape in SIMPLEX_SHAPES:
        return float(max(-np.min(xi), np.sum(xi) - 1.0))
    return float(np.max(np.abs(xi)) - 1.0)


def isInside(shape: str, xi: np.ndarray, tolerance: float = INSIDE_TOLERANCE) -> bool:
    """Whether the parametric coordinates ``xi`` lie inside of the parent domain (with a tolerance)."""
    return parentDomainExcess(shape, xi) <= tolerance


def parentDomainCenter(shape: str) -> np.ndarray:
    """The parametric coordinates of the center of the parent domain."""
    return NODE_PARAMETRIC_COORDINATES[shape].mean(axis=0)


#: Newton converged when the parametric correction is below this (dimensionless: parametric coordinates are O(1))
INVERSE_MAPPING_TOLERANCE = 1e-10


def newtonInverseMapping(
    shapeFunctionsAndDerivativesAt: Callable[[np.ndarray], tuple[np.ndarray, np.ndarray]],
    nodeCoordinates: np.ndarray,
    x: np.ndarray,
    xiStart: np.ndarray,
    maxIterations: int = 50,
) -> tuple[np.ndarray, bool]:
    """The inverse isoparametric mapping by Newton's method: find :math:`\\boldsymbol{\\xi}` with
    :math:`\\boldsymbol{r}(\\boldsymbol{\\xi}) = \\sum_a N_a(\\boldsymbol{\\xi}) \\mathbf{X}_a - \\mathbf{x} = \\mathbf{0}`,

    .. math::

        \\mathbf{J} = \\sum_a \\mathbf{X}_a \\otimes \\frac{\\partial N_a}{\\partial \\boldsymbol{\\xi}}, \\qquad
        \\Delta\\boldsymbol{\\xi} = -\\mathbf{J}^{-1} \\boldsymbol{r}

    (e.g., Hughes, *The Finite Element Method*, ch. 3). The iteration runs on coordinates centred on the element,
    :math:`\\mathbf{X}_a - \\bar{\\mathbf{X}}`, :math:`\\mathbf{x} - \\bar{\\mathbf{X}}`: the round-off of the residual is then
    about :math:`\\varepsilon h` (element size :math:`h`) instead of :math:`\\varepsilon |\\mathbf{x}|`, which for a structure far
    from the origin (e.g., mm coordinates of a model of several metres) exceeds any absolute tolerance. It stops when
    the parametric correction is below :data:`INVERSE_MAPPING_TOLERANCE` (dimensionless).

    Parameters
    ----------
    shapeFunctionsAndDerivativesAt
        ``xi -> (N, dN_dXi)`` of the element.
    nodeCoordinates
        The nodal coordinates ``(nNodes, nDim)``, ordered as the shape functions.
    x
        The point.
    xiStart
        The start, e.g., the centre of the parent domain (inside it, and the unique solution of a convex element is
        reached from it).
    maxIterations
        The maximum number of Newton iterations.

    Returns
    -------
    tuple[np.ndarray, bool]
        The last iterate and whether it converged.
    """
    nodeCoordinates = np.asarray(nodeCoordinates, dtype=float)
    centre = nodeCoordinates.mean(axis=0)
    X = nodeCoordinates - centre
    xLocal = np.asarray(x, dtype=float) - centre
    xi = np.array(xiStart, dtype=float)
    for _ in range(maxIterations):
        N, dN_dXi = shapeFunctionsAndDerivativesAt(xi)
        residual = N @ X - xLocal
        J = X.T @ dN_dXi  # dX/dxi
        try:
            dXi = -np.linalg.solve(J, residual)
        except np.linalg.LinAlgError:
            return xi, False
        xi = xi + dXi
        if np.linalg.norm(dXi) < INVERSE_MAPPING_TOLERANCE:
            return xi, True
        if np.max(np.abs(xi)) > 1e3:  # diverged: far outside of a distorted element
            return xi, False
    return xi, False


def inverseMapping(shape: str, nodeCoordinates: np.ndarray, x: np.ndarray, maxIterations: int = 50) -> np.ndarray:
    """The parametric coordinates of a point ``x`` in an element, see :func:`newtonInverseMapping`.

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
        The parametric coordinates, or ``None`` if the iteration did not converge (a point far outside of a
        distorted element, or a singular Jacobian) -- not converged, which is different from "outside": the point may
        still lie in the element.
    """
    xi, converged = newtonInverseMapping(
        lambda xi_: shapeFunctionsAndDerivatives(shape, xi_),
        nodeCoordinates,
        x,
        parentDomainCenter(shape),
        maxIterations,
    )
    return xi if converged else None
