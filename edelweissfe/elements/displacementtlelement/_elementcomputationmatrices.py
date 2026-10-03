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
import numpy.linalg as lin

from edelweissfe.elements._hexa3dnodeordering import hexa8DNdXi, hexa20DNdXi


def computeDeformationGradient(U: np.ndarray, nablaN: np.ndarray, nInt: int, nNodes: int, dim: int):
    """Get the deformation gradient for a nonlinear element.

    Parameters
    ----------
    U
        The current displacement vector.
    nablaN
        The derivative of the shape functions w.r.t the actual coordinates.
    nInt
        Number of quadrature points the element has.
    nNodes
        Number of nodes the element has.
    dim
        Dimension of the domain.

    Returns
    -------
    np.ndarray
        The deformation gradient."""

    F = np.zeros([nInt, dim, dim])
    if dim == 2:
        for i in range(nInt):  # for all Gauss points (N in total)
            H = np.zeros([2, 2])
            for j in range(nNodes):  # make deformation gradient
                UI = U[2 * j : 2 * (j + 1)]
                H += np.outer(UI, nablaN[i, :, j])
            F[i] = np.eye(2) + H
    elif dim == 3:
        for i in range(nInt):  # for all Gauss points (N in total)
            H = np.zeros([3, 3])
            for j in range(nNodes):  # for all points/shape functions
                UI = U[3 * j : 3 * (j + 1)]
                H += np.outer(UI, nablaN[i, :, j])
            F[i] = np.eye(3) + H
    return F


def computeBOperator(F: np.ndarray, nablaN: np.ndarray, nInt: int, nNodes: int, dim: int):
    """Get the B operator for the element calculation.

    Parameters
    ----------
    F
        The deformation gradient.
    nablaN
        The derivative of the shape functions w.r.t the actual coordinates.
    nInt
        Number of integration points.
    nNodes
        Number of nodes the element has.
    dim
        Dimension the element has.

    Returns
    -------
    np.ndarray
        The requested B operator for a nonlinear element.
    np.ndarray
        The deformation gradient."""

    if dim == 2:
        return _B02D(F, nablaN, nInt, nNodes)
    elif dim == 3:
        return _B03D(F, nablaN, nInt, nNodes)


def computeNablaN(xi: np.ndarray, eta: np.ndarray, z: np.ndarray, J: np.ndarray, nInt: int, nNodes: int, dim: int):
    """Get the nabla N(i) = dN/dX operator.

    Parameters
    ----------
    xi
        Local coordinates xi for the integration points.
    eta
        Local coordinates eta for the integration points.
    z
        Local coordinates zeta for the integration points.
    J
        The jacobian between the local and global undeformed coordinates.
    nInt
        Number of integration points.
    nNodes
        Number of nodes the element has.
    dim
        Dimension the element has.

    Returns
    -------
    np.ndarray
        The derivative of the shape functions w.r.t the actual coordinates."""

    nablaN = np.zeros([nInt, dim, nNodes])
    for i in range(nInt):  # for all Gauss points (N in total)
        if dim == 2:
            dN = _Ndiff2D(xi[i], eta[i], nNodes)
        else:
            dN = _Ndiff3D(xi[i], eta[i], z[i], nNodes)
        invJ = lin.inv(J[i])
        for j in range(nNodes):  # for all points/shape functions
            nablaN[i, :, j] = dN[:, j] @ invJ.T
    return nablaN


def makeH2D(Hmat, dim):
    """Puts the element material stiffness from 4D into (nDof, nDof).

    Parameters
    ----------
    Hmat
        The element material stiffness matrix.
    dim
        Dimension of the domain.

    Returns
    -------
    np.ndarray
        The element stiffness matrix in 2D form."""

    if dim == 2:
        Hmat = Hmat[:, :dim, :, :dim]
    m, n, _, _ = Hmat.shape
    HmatDim = np.zeros([m * n, m * n])
    for i in range(m):
        for j in range(n):
            HmatDim[dim * i + j] = Hmat[i, j].flatten()
    return HmatDim


def _Ndiff2D(xi: np.ndarray, eta: np.ndarray, nNodes: int):
    """Calculate the differentiated 2D shape functions.

    Parameters
    ----------
    xi
        Local coordinates xi for the integration points.
    eta
        Local coordinates eta for the integration points.
    nNodes
        Number of nodes the element has.

    Returns
    -------
    np.ndarray
        The derivative of the 2D shape functions w.r.t. the local coordinates."""

    if nNodes == 4:  # Quad4
        return np.array(
            [
                [
                    -1 / 4 * (1 - xi),
                    1 / 4 * (1 - xi),
                    1 / 4 * (1 + xi),
                    -1 / 4 * (1 + xi),
                ],
                [
                    -1 / 4 * (1 - eta),
                    -1 / 4 * (1 + eta),
                    1 / 4 * (1 + eta),
                    1 / 4 * (1 - eta),
                ],
            ]
        )
    else:  # Quad8
        return np.array(
            [
                [
                    -1 / 4 * (-1 + xi) * (2 * eta + xi),
                    1 / 4 * (-1 + xi) * (xi - 2 * eta),
                    1 / 4 * (1 + xi) * (2 * eta + xi),
                    -1 / 4 * (1 + xi) * (xi - 2 * eta),
                    eta * (-1 + xi),
                    -1 / 2 * (1 + xi) * (-1 + xi),
                    -eta * (1 + xi),
                    1 / 2 * (1 + xi) * (-1 + xi),
                ],
                [
                    -1 / 4 * (-1 + eta) * (eta + 2 * xi),
                    1 / 4 * (1 + eta) * (2 * xi - eta),
                    1 / 4 * (1 + eta) * (eta + 2 * xi),
                    -1 / 4 * (-1 + eta) * (2 * xi - eta),
                    1 / 2 * (1 + eta) * (-1 + eta),
                    -xi * (1 + eta),
                    -1 / 2 * (1 + eta) * (-1 + eta),
                    xi * (-1 + eta),
                ],
            ]
        )


def _Ndiff3D(xi: np.ndarray, eta: np.ndarray, z: np.ndarray, nNodes: int):
    """Calculate the differentiated 3D shape functions.

    Parameters
    ----------
    xi
        Local coordinates xi for the integration points.
    eta
        Local coordinates eta for the integration points.
    z
        Local coordinates zeta for the integration points.
    nNodes
        Number of nodes the element has.

    Returns
    -------
    np.ndarray
        The derivative of the 3D shape functions w.r.t. the local coordinates."""

    # Node ordering matches computeNOperator (Marmot's/standard Abaqus C3D8/C3D20 convention).
    if nNodes == 8:  # Hexa8
        return hexa8DNdXi(xi, eta, z)
    else:  # Hexa20
        return hexa20DNdXi(xi, eta, z)


# B0 operator
def _B02D(F: np.ndarray, nablaN: np.ndarray, nInt: int, nNodes: int):
    """Get the B operator and the deformation gradient for a nonlinear Quad element.

    Parameters
    ----------
    F
        The deformation gradient.
    nablaN
        The derivative of the shape functions w.r.t the actual coordinates.
    nInt
        Number of quadrature points the element has.
    nNodes
        Number of nodes the element has.

    Returns
    -------
    np.ndarray
        The requested B operator for a nonlinear element.
    np.ndarray
        The deformation gradient."""

    Bi = np.zeros([nInt, 3, nNodes * 2])
    for i in range(nInt):  # for all Gauss points (N in total)
        # [B] for all different xi and eta
        dxdX = F[i]
        for j in range(nNodes):  # construct B operator
            dNdX, dNdY = nablaN[i, :, j]
            Bi[i, :, 2 * j : 2 * j + 2] = np.array(
                [
                    [dNdX * dxdX[0, 0], dNdX * dxdX[1, 0]],
                    [dNdY * dxdX[0, 1], dNdY * dxdX[1, 1]],
                    [dNdX * dxdX[0, 1] + dNdY * dxdX[0, 0], dNdX * dxdX[1, 1] + dNdY * dxdX[1, 0]],
                ]
            )
    return Bi


def _B03D(F: np.ndarray, nablaN: np.ndarray, nInt: int, nNodes: int):
    """Get the B operator and the deformation gradient for a nonlinear Hexa element.

    Parameters
    ----------
    F
        The deformation gradient.
    nablaN
        The derivative of the shape functions w.r.t the actual coordinates.
    nInt
        Number of quadrature points the element has.
    nNodes
        Number of nodes the element has.

    Returns
    -------
    np.ndarray
        The requested B operator for a nonlinear element.
    np.ndarray
        The deformation gradient."""

    Bi = np.zeros([nInt, 6, nNodes * 3])
    for i in range(nInt):  # for all Gauss points (N in total)
        # [B] for all different xi, eta and zeta
        dxdX = F[i]
        for j in range(nNodes):  # construct B operator
            dNdX, dNdY, dNdZ = nablaN[i, :, j]
            Bi[i, :, 3 * j : 3 * j + 3] = np.array(
                [
                    [dNdX * dxdX[0, 0], dNdX * dxdX[1, 0], dNdX * dxdX[2, 0]],
                    [dNdY * dxdX[0, 1], dNdY * dxdX[1, 1], dNdY * dxdX[2, 1]],
                    [dNdZ * dxdX[0, 2], dNdZ * dxdX[1, 2], dNdZ * dxdX[2, 2]],
                    [
                        dNdX * dxdX[0, 1] + dNdY * dxdX[0, 0],
                        dNdX * dxdX[1, 1] + dNdY * dxdX[1, 0],
                        dNdX * dxdX[2, 1] + dNdY * dxdX[2, 0],
                    ],
                    [
                        dNdY * dxdX[0, 2] + dNdZ * dxdX[0, 1],
                        dNdY * dxdX[1, 2] + dNdZ * dxdX[1, 1],
                        dNdY * dxdX[2, 2] + dNdZ * dxdX[2, 1],
                    ],
                    [
                        dNdX * dxdX[0, 2] + dNdZ * dxdX[0, 0],
                        dNdX * dxdX[1, 2] + dNdZ * dxdX[1, 0],
                        dNdX * dxdX[2, 2] + dNdZ * dxdX[2, 0],
                    ],
                ]
            )
    return Bi
