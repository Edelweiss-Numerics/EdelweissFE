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
#  Daniel Reitmair daniel.reitmair@uibk.ac.at
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

from edelweissfe.elements._hexa3dnodeordering import (
    hexa8DNdXi,
    hexa8N,
    hexa20DNdXi,
    hexa20N,
)
from edelweissfe.elements._quad2dnodeordering import (
    quad4DNdXi,
    quad4N,
    quad8DNdXi,
    quad8N,
)


def computeJacobian(xi: np.ndarray, eta: np.ndarray, z: np.ndarray, x: np.ndarray, nInt: int, nNodes: int, dim: int):
    """Get the Jacobi matrix for the element calculation.

    Parameters
    ----------
    xi
        Local coordinate xi.
    eta
        Local coordinate eta.
    z
        Local coordinate zeta.
    x
        Global coordinates of the element points.
    nInt
        Number of integration points.
    nNodes
        Number of nodes the element has.
    dim
        Dimension the element has.

    Returns
    -------
    np.ndarray
        The requested Jacobian matrix."""

    if dim == 2:
        return _JFromDNdXi(_dNdXi2D(xi, eta, nInt, nNodes), x, nInt)
    elif dim == 3:
        if nNodes == 8:
            return _J3D8(xi, eta, z, x, nInt)
        elif nNodes == 20:
            return _J3D20(xi, eta, z, x, nInt)


def computeBOperator(xi: np.ndarray, eta: np.ndarray, z: np.ndarray, x: np.ndarray, nInt: int, nNodes: int, dim: int):
    """Get the B operator for the element calculation.

    Parameters
    ----------
    xi
        Local coordinate xi.
    eta
        Local coordinate eta.
    z
        Local coordinate zeta.
    x
        Global coordinates of the element points.
    nInt
        Number of integration points.
    nNodes
        Number of nodes the element has.
    dim
        Dimension the element has.

    Returns
    -------
    np.ndarray
        The requested B operator."""

    if dim == 2:
        dNdXi = _dNdXi2D(xi, eta, nInt, nNodes)
        return _B2DGeneric(dNdXi, _JFromDNdXi(dNdXi, x, nInt), nInt, nNodes)
    elif dim == 3:
        if nNodes == 8:
            return _B3D8(xi, eta, z, x, nInt)
        elif nNodes == 20:
            return _B3D20(xi, eta, z, x, nInt)


def computeNOperator(xi: np.ndarray, eta: np.ndarray, z: np.ndarray, nInt: int, nNodes: int, dim: int):
    """Get the N operator containing the shape functions.

    Parameters
    ----------
    xi
        Local coordinates xi for the integration points.
    eta
        Local coordinates eta for the integration points.
    z
        Local coordinates zeta for the integration points.
    nInt
        Number of integration points.
    nNodes
        Number of nodes the element has.
    dim
        Dimension the element has.

    Returns
    -------
    np.ndarray
        The shape functions at the given coordinates."""

    N = np.zeros([nInt, nNodes])
    if dim == 2:
        # Quad4/Quad8 node ordering and natural directions: see edelweissfe.elements._quad2dnodeordering
        if nNodes == 4:  # Quad4
            for i in range(nInt):
                N[i] = quad4N(xi[i], eta[i])
        elif nNodes == 8:  # Quad8
            for i in range(nInt):
                N[i] = quad8N(xi[i], eta[i])
    elif dim == 3:
        # Hexa8/Hexa20 node ordering (see edelweissfe.elements._hexa3dnodeordering) follows the Abaqus
        # C3D8/C3D20 convention (corner ring 0-3 at zeta=-1, ring 4-7 at zeta=+1), as in Marmot.
        if nNodes == 8:  # Hexa8
            for i in range(nInt):
                N[i] = hexa8N(xi[i], eta[i], z[i])
        elif nNodes == 20:  # Hexa20
            for i in range(nInt):
                N[i] = hexa20N(xi[i], eta[i], z[i])
    return N


def _dNdXi3D8(xi: np.ndarray, eta: np.ndarray, z: np.ndarray, nInt: int):
    """Get dN/dxi for a Hexa8 element (matching Marmot's node ordering, see computeNOperator).

    Parameters
    ----------
    xi
        Local coordinates xi for the integration points.
    eta
        Local coordinates eta for the integration points.
    z
        Local coordinates zeta for the integration points.
    nInt
        Number of quadrature points the element has.

    Returns
    -------
    np.ndarray
        Shape ``(nInt, 3, 8)``: row 0/1/2 are d/dxi, d/deta, d/dzeta; columns are nodes."""

    dNdXi = np.zeros([nInt, 3, 8])
    for i in range(nInt):
        dNdXi[i] = hexa8DNdXi(xi[i], eta[i], z[i])
    return dNdXi


def _dNdXi2D(xi: np.ndarray, eta: np.ndarray, nInt: int, nNodes: int):
    """Get dN/dxi for a Quad4 or Quad8 element (see :mod:`edelweissfe.elements._quad2dnodeordering`).

    Parameters
    ----------
    xi
        Local coordinates xi for the integration points.
    eta
        Local coordinates eta for the integration points.
    nInt
        Number of quadrature points the element has.
    nNodes
        Number of nodes the element has, 4 or 8.

    Returns
    -------
    np.ndarray
        Shape ``(nInt, 2, nNodes)``: row 0/1 are d/dxi, d/deta; columns are nodes."""

    dNdXiOfQuad = quad4DNdXi if nNodes == 4 else quad8DNdXi
    dNdXi = np.zeros([nInt, 2, nNodes])
    for i in range(nInt):
        dNdXi[i] = dNdXiOfQuad(xi[i], eta[i])
    return dNdXi


def _JFromDNdXi(dNdXi: np.ndarray, x: np.ndarray, nInt: int):
    """Get the Jacobi matrix for a 2D or 3D solid element from its (already computed) dN/dxi.

    Parameters
    ----------
    dNdXi
        Shape ``(nInt, dim, nNodes)``, as returned by :func:`_dNdXi2D`/:func:`_dNdXi3D8`/:func:`_dNdXi3D20`.
    x
        Coordinates of the element points.
    nInt
        Number of quadrature points the element has.

    Returns
    -------
    np.ndarray
        The requested Jacobian matrix."""

    dim = dNdXi.shape[1]
    J = np.zeros([nInt, dim, dim])
    for i in range(nInt):
        J[i] = dNdXi[i] @ x.T
    return J


def _J3D8(xi: np.ndarray, eta: np.ndarray, z: np.ndarray, x: np.ndarray, nInt: int):
    """Get the Jacobi matrix for a Hexa8 element.

    Parameters
    ----------
    xi
        Local coordinates xi for the integration points.
    eta
        Local coordinates eta for the integration points.
    z
        Local coordinates zeta for the integration points.
    x
        Coordinates of the element points.
    nInt
        Number of quadrature points the element has.

    Returns
    -------
    np.ndarray
        The requested Jacobian matrix."""

    return _JFromDNdXi(_dNdXi3D8(xi, eta, z, nInt), x, nInt)


def _dNdXi3D20(xi: np.ndarray, eta: np.ndarray, z: np.ndarray, nInt: int):
    """Get dN/dxi for a Hexa20 element (matching Marmot's node ordering, see computeNOperator).

    Parameters
    ----------
    xi
        Local coordinates xi for the integration points.
    eta
        Local coordinates eta for the integration points.
    z
        Local coordinates zeta for the integration points.
    nInt
        Number of quadrature points the element has.

    Returns
    -------
    np.ndarray
        Shape ``(nInt, 3, 20)``: row 0/1/2 are d/dxi, d/deta, d/dzeta; columns are nodes."""

    dNdXi = np.zeros([nInt, 3, 20])
    for i in range(nInt):
        dNdXi[i] = hexa20DNdXi(xi[i], eta[i], z[i])
    return dNdXi


def _J3D20(xi: np.ndarray, eta: np.ndarray, z: np.ndarray, x: np.ndarray, nInt: int):
    """Get the Jacobi matrix for a Hexa20 element.

    Parameters
    ----------
    xi
        Local coordinates xi for the integration points.
    eta
        Local coordinates eta for the integration points.
    z
        Local coordinates zeta for the integration points.
    x
        Coordinates of the element points.
    nInt
        Number of quadrature points the element has.

    Returns
    -------
    np.ndarray
        The requested Jacobian matrix."""

    return _JFromDNdXi(_dNdXi3D20(xi, eta, z, nInt), x, nInt)


def _B2DGeneric(dNdXi: np.ndarray, J: np.ndarray, nInt: int, nNodes: int):
    """Assemble the standard isoparametric strain-displacement (B) operator for a 2D solid
    element from its dN/dxi and Jacobian, given a node-major local DOF vector
    ``(ux1,uy1,ux2,uy2,...)``.

    Parameters
    ----------
    dNdXi
        Shape ``(nInt, 2, nNodes)``, as returned by :func:`_dNdXi2D`.
    J
        The Jacobian matrices, shape ``(nInt, 2, 2)``.
    nInt
        Number of quadrature points the element has.
    nNodes
        Number of nodes the element has.

    Returns
    -------
    np.ndarray
        The B operator, shape ``(nInt, 3, 2*nNodes)``, Voigt order ``(exx, eyy, gxy)``."""

    Bi = np.zeros([nInt, 3, 2 * nNodes])
    for i in range(nInt):
        dNdX = np.linalg.solve(J[i], dNdXi[i])  # (2, nNodes): row b = dN/dX_b
        for k in range(nNodes):
            dNkdX, dNkdY = dNdX[:, k]
            Bi[i, :, 2 * k : 2 * k + 2] = np.array([[dNkdX, 0], [0, dNkdY], [dNkdY, dNkdX]])
    return Bi


def _B3DGeneric(dNdXi: np.ndarray, J: np.ndarray, nInt: int, nNodes: int):
    """Assemble the standard isoparametric strain-displacement (B) operator for a 3D solid
    element from its dN/dxi and Jacobian, given a node-major local DOF vector
    ``(ux1,uy1,uz1,ux2,uy2,uz2,...)``.

    Parameters
    ----------
    dNdXi
        Shape ``(nInt, 3, nNodes)``, as returned by :func:`_dNdXi3D8`/:func:`_dNdXi3D20`.
    J
        The Jacobian matrices, shape ``(nInt, 3, 3)``.
    nInt
        Number of quadrature points the element has.
    nNodes
        Number of nodes the element has.

    Returns
    -------
    np.ndarray
        The B operator, shape ``(nInt, 6, 3*nNodes)``, Voigt order
        ``(exx, eyy, ezz, gxy, gxz, gyz)``."""

    Bi = np.zeros([nInt, 6, 3 * nNodes])
    for i in range(nInt):
        dNdX = np.linalg.solve(J[i], dNdXi[i])  # (3, nNodes): row b = dN/dX_b
        for k in range(nNodes):
            dNkdX, dNkdY, dNkdZ = dNdX[:, k]
            Bi[i, :, 3 * k : 3 * k + 3] = np.array(
                [
                    [dNkdX, 0, 0],
                    [0, dNkdY, 0],
                    [0, 0, dNkdZ],
                    [dNkdY, dNkdX, 0],
                    [dNkdZ, 0, dNkdX],
                    [0, dNkdZ, dNkdY],
                ]
            )
    return Bi


def _B3D8(xi: np.ndarray, eta: np.ndarray, z: np.ndarray, x: np.ndarray, nInt: int):
    """Get the B operator for a linear Hexa8 element.

    Parameters
    ----------
    xi
        Local coordinates xi for the integration points.
    eta
        Local coordinates eta for the integration points.
    z
        Local coordinates zeta for the integration points.
    x
        Coordinates of the element points.
    nInt
        Number of quadrature points the element has.

    Returns
    -------
    np.ndarray
        The requested B operator."""

    dNdXi = _dNdXi3D8(xi, eta, z, nInt)
    J = _JFromDNdXi(dNdXi, x, nInt)
    return _B3DGeneric(dNdXi, J, nInt, 8)


def _B3D20(xi: np.ndarray, eta: np.ndarray, z: np.ndarray, x: np.ndarray, nInt: int):
    """Get the B operator for a quadratic Hexa20 element.

    Parameters
    ----------
    xi
        Local coordinates xi for the integration points.
    eta
        Local coordinates eta for the integration points.
    z
        Local coordinates zeta for the integration points.
    x
        Coordinates of the element points.
    nInt
        Number of quadrature points the element has.

    Returns
    -------
    np.ndarray
        The requested B operator."""

    dNdXi = _dNdXi3D20(xi, eta, z, nInt)
    J = _JFromDNdXi(dNdXi, x, nInt)
    return _B3DGeneric(dNdXi, J, nInt, 20)
