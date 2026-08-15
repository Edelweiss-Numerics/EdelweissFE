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
        if nNodes == 4:
            return _J2D4(xi, eta, x, nInt)
        elif nNodes == 8:
            return _J2D8(xi, eta, x, nInt)
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
        if nNodes == 4:
            return _B2D4(xi, eta, x, nInt)
        elif nNodes == 8:
            return _B2D8(xi, eta, x, nInt)
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
        if nNodes == 4:  # Quad4
            for i in range(nInt):
                N[i] = (
                    1
                    / 4
                    * np.array(
                        [
                            (1 - xi[i]) * (1 - eta[i]),
                            (1 + xi[i]) * (1 - eta[i]),
                            (1 + xi[i]) * (1 + eta[i]),
                            (1 - xi[i]) * (1 + eta[i]),
                        ]
                    )
                )
        elif nNodes == 8:  # Quad8
            for i in range(nInt):
                N[i] = (
                    1
                    / 4
                    * np.array(
                        [
                            (1 - xi[i]) * (1 - eta[i]) * (-xi[i] - eta[i] - 1),
                            (1 + xi[i]) * (1 - eta[i]) * (xi[i] - eta[i] - 1),
                            (1 + xi[i]) * (1 + eta[i]) * (xi[i] + eta[i] - 1),
                            (1 - xi[i]) * (1 + eta[i]) * (-xi[i] + eta[i] - 1),
                            2 * (1 - xi[i] ** 2) * (1 - eta[i]),
                            2 * (1 + xi[i]) * (1 - eta[i] ** 2),
                            2 * (1 - xi[i] ** 2) * (1 + eta[i]),
                            2 * (1 - xi[i]) * (1 - eta[i] ** 2),
                        ]
                    )
                )
    elif dim == 3:
        # Hexa8/Hexa20 node ordering follows the standard Abaqus C3D8/C3D20 convention (corner
        # ring 0-3 at zeta=-1, ring 4-7 at zeta=+1), matching Marmot's element formulation --
        # see edelweissfe.generators.surfaceelementgenerator's module docstring for why this
        # matters for contact-facet generation.
        if nNodes == 8:  # Hexa8
            for i in range(nInt):
                N[i] = (
                    1
                    / 8
                    * np.array(
                        [
                            (1 - xi[i]) * (1 - eta[i]) * (1 - z[i]),
                            (1 + xi[i]) * (1 - eta[i]) * (1 - z[i]),
                            (1 + xi[i]) * (1 + eta[i]) * (1 - z[i]),
                            (1 - xi[i]) * (1 + eta[i]) * (1 - z[i]),
                            (1 - xi[i]) * (1 - eta[i]) * (1 + z[i]),
                            (1 + xi[i]) * (1 - eta[i]) * (1 + z[i]),
                            (1 + xi[i]) * (1 + eta[i]) * (1 + z[i]),
                            (1 - xi[i]) * (1 + eta[i]) * (1 + z[i]),
                        ]
                    )
                )
        elif nNodes == 20:  # Hexa20
            for i in range(nInt):
                N[i] = (
                    1
                    / 8
                    * np.array(
                        [
                            -(1 - xi[i]) * (1 - eta[i]) * (1 - z[i]) * (2 + xi[i] + eta[i] + z[i]),
                            -(1 + xi[i]) * (1 - eta[i]) * (1 - z[i]) * (2 - xi[i] + eta[i] + z[i]),
                            -(1 + xi[i]) * (1 + eta[i]) * (1 - z[i]) * (2 - xi[i] - eta[i] + z[i]),
                            -(1 - xi[i]) * (1 + eta[i]) * (1 - z[i]) * (2 + xi[i] - eta[i] + z[i]),
                            -(1 - xi[i]) * (1 - eta[i]) * (1 + z[i]) * (2 + xi[i] + eta[i] - z[i]),
                            -(1 + xi[i]) * (1 - eta[i]) * (1 + z[i]) * (2 - xi[i] + eta[i] - z[i]),
                            -(1 + xi[i]) * (1 + eta[i]) * (1 + z[i]) * (2 - xi[i] - eta[i] - z[i]),
                            -(1 - xi[i]) * (1 + eta[i]) * (1 + z[i]) * (2 + xi[i] - eta[i] - z[i]),
                            2 * (1 - xi[i] ** 2) * (1 - eta[i]) * (1 - z[i]),
                            2 * (1 - eta[i] ** 2) * (1 + xi[i]) * (1 - z[i]),
                            2 * (1 - xi[i] ** 2) * (1 + eta[i]) * (1 - z[i]),
                            2 * (1 - eta[i] ** 2) * (1 - xi[i]) * (1 - z[i]),
                            2 * (1 - xi[i] ** 2) * (1 - eta[i]) * (1 + z[i]),
                            2 * (1 - eta[i] ** 2) * (1 + xi[i]) * (1 + z[i]),
                            2 * (1 - xi[i] ** 2) * (1 + eta[i]) * (1 + z[i]),
                            2 * (1 - eta[i] ** 2) * (1 - xi[i]) * (1 + z[i]),
                            2 * (1 - xi[i]) * (1 - eta[i]) * (1 - z[i] ** 2),
                            2 * (1 + xi[i]) * (1 - eta[i]) * (1 - z[i] ** 2),
                            2 * (1 + xi[i]) * (1 + eta[i]) * (1 - z[i] ** 2),
                            2 * (1 - xi[i]) * (1 + eta[i]) * (1 - z[i] ** 2),
                        ]
                    )
                )
    return N


def _J2D4(xi: np.ndarray, eta: np.ndarray, x: np.ndarray, nInt: int):
    """Get the Jacobi matrix for a Quad4 element.

    Parameters
    ----------
    xi
        Local coordinates xi for the integration points.
    eta
        Local coordinates eta for the integration points.
    x
        Coordinates of the element points.
    nInt
        Number of quadrature points the element has.

    Returns
    -------
    np.ndarray
        The requested Jacobian matrix."""

    J = np.zeros([nInt, 2, 2])
    # calc all parameters for the X and Y functions (Q4)
    A = np.array([[1, -1, -1, 1], [1, 1, -1, -1], [1, 1, 1, 1], [1, -1, 1, -1]])
    invA = np.linalg.inv(A)
    # calculate parameters
    ax = invA @ np.transpose(x[0])
    ay = invA @ np.transpose(x[1])
    for i in range(0, nInt):  # for all Gauss points (N in total)
        # [J] Jacobi matrix (only Q4)
        J[i] = np.array(
            [
                [ax[1] + ax[3] * xi[i], ay[1] + ay[3] * xi[i]],
                [ax[2] + ax[3] * eta[i], ay[2] + ay[3] * eta[i]],
            ]
        )
    return J


def _J2D8(xi: np.ndarray, eta: np.ndarray, x: np.ndarray, nInt: int):
    """Get the Jacobi matrix for a Quad8 element.

    Parameters
    ----------
    xi
        Local coordinates xi for the integration points.
    eta
        Local coordinates eta for the integration points.
    x
        Coordinates of the element points.
    nInt
        Number of quadrature points the element has.

    Returns
    -------
    np.ndarray
        The requested Jacobian matrix."""

    J = np.zeros([nInt, 2, 2])
    # calc all parameters for the X and Y functions (Q8)
    A = np.array(
        [
            [1, -1, -1, 1, 1, 1, -1, -1],
            [1, 1, -1, -1, 1, 1, -1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1],
            [1, -1, 1, -1, 1, 1, 1, -1],
            [1, 0, -1, 0, 0, 1, 0, 0],
            [1, 1, 0, 0, 1, 0, 0, 0],
            [1, 0, 1, 0, 0, 1, 0, 0],
            [1, -1, 0, 0, 1, 0, 0, 0],
        ]
    )
    invA = np.linalg.inv(A)
    # calculate parameters
    ax = invA @ np.transpose(x[0])
    ay = invA @ np.transpose(x[1])
    for i in range(0, nInt):  # for all Gauss points (N in total)
        # [J] Jacobi matrix for Q8
        J[i] = np.array(
            [
                [
                    ax[1] + ax[3] * xi[i] + 2 * ax[4] * eta[i] + 2 * ax[6] * eta[i] * xi[i] + ax[7] * xi[i] ** 2,
                    ay[1] + ay[3] * xi[i] + 2 * ay[4] * eta[i] + 2 * ay[6] * eta[i] * xi[i] + ay[7] * xi[i] ** 2,
                ],
                [
                    ax[2] + ax[3] * eta[i] + 2 * ax[5] * xi[i] + ax[6] * eta[i] ** 2 + 2 * ax[7] * eta[i] * xi[i],
                    ay[2] + ay[3] * eta[i] + 2 * ay[5] * xi[i] + ay[6] * eta[i] ** 2 + 2 * ay[7] * eta[i] * xi[i],
                ],
            ]
        )
    return J


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
        dNdXi[i] = (
            1
            / 8
            * np.array(
                [
                    [
                        -(1 - eta[i]) * (1 - z[i]),
                        (1 - eta[i]) * (1 - z[i]),
                        (1 + eta[i]) * (1 - z[i]),
                        -(1 + eta[i]) * (1 - z[i]),
                        -(1 - eta[i]) * (1 + z[i]),
                        (1 - eta[i]) * (1 + z[i]),
                        (1 + eta[i]) * (1 + z[i]),
                        -(1 + eta[i]) * (1 + z[i]),
                    ],
                    [
                        -(1 - xi[i]) * (1 - z[i]),
                        -(1 + xi[i]) * (1 - z[i]),
                        (1 + xi[i]) * (1 - z[i]),
                        (1 - xi[i]) * (1 - z[i]),
                        -(1 - xi[i]) * (1 + z[i]),
                        -(1 + xi[i]) * (1 + z[i]),
                        (1 + xi[i]) * (1 + z[i]),
                        (1 - xi[i]) * (1 + z[i]),
                    ],
                    [
                        -(1 - xi[i]) * (1 - eta[i]),
                        -(1 + xi[i]) * (1 - eta[i]),
                        -(1 + xi[i]) * (1 + eta[i]),
                        -(1 - xi[i]) * (1 + eta[i]),
                        (1 - xi[i]) * (1 - eta[i]),
                        (1 + xi[i]) * (1 - eta[i]),
                        (1 + xi[i]) * (1 + eta[i]),
                        (1 - xi[i]) * (1 + eta[i]),
                    ],
                ]
            )
        )
    return dNdXi


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

    dNdXi = _dNdXi3D8(xi, eta, z, nInt)
    J = np.zeros([nInt, 3, 3])
    for i in range(nInt):
        J[i] = dNdXi[i] @ x.T
    return J


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
        xi_, eta_, z_ = xi[i], eta[i], z[i]
        dNdXi[i, 0, :] = [
            0.125 * (1 - eta_) * (1 - z_) * (1 + 2 * xi_ + eta_ + z_),
            -0.125 * (1 - eta_) * (1 - z_) * (1 - 2 * xi_ + eta_ + z_),
            -0.125 * (1 + eta_) * (1 - z_) * (1 - 2 * xi_ - eta_ + z_),
            0.125 * (1 + eta_) * (1 - z_) * (1 + 2 * xi_ - eta_ + z_),
            0.125 * (1 - eta_) * (1 + z_) * (1 + 2 * xi_ + eta_ - z_),
            -0.125 * (1 - eta_) * (1 + z_) * (1 - 2 * xi_ + eta_ - z_),
            -0.125 * (1 + eta_) * (1 + z_) * (1 - 2 * xi_ - eta_ - z_),
            0.125 * (1 + eta_) * (1 + z_) * (1 + 2 * xi_ - eta_ - z_),
            -0.5 * xi_ * (1 - eta_) * (1 - z_),
            0.25 * (1 - eta_ * eta_) * (1 - z_),
            -0.5 * xi_ * (1 + eta_) * (1 - z_),
            -0.25 * (1 - eta_ * eta_) * (1 - z_),
            -0.5 * xi_ * (1 - eta_) * (1 + z_),
            0.25 * (1 - eta_ * eta_) * (1 + z_),
            -0.5 * xi_ * (1 + eta_) * (1 + z_),
            -0.25 * (1 - eta_ * eta_) * (1 + z_),
            -0.25 * (1 - eta_) * (1 - z_ * z_),
            0.25 * (1 - eta_) * (1 - z_ * z_),
            0.25 * (1 + eta_) * (1 - z_ * z_),
            -0.25 * (1 + eta_) * (1 - z_ * z_),
        ]
        dNdXi[i, 1, :] = [
            0.125 * (1 - xi_) * (1 - z_) * (1 + xi_ + 2 * eta_ + z_),
            0.125 * (1 + xi_) * (1 - z_) * (1 - xi_ + 2 * eta_ + z_),
            -0.125 * (1 + xi_) * (1 - z_) * (1 - xi_ - 2 * eta_ + z_),
            -0.125 * (1 - xi_) * (1 - z_) * (1 + xi_ - 2 * eta_ + z_),
            0.125 * (1 - xi_) * (1 + z_) * (1 + xi_ + 2 * eta_ - z_),
            0.125 * (1 + xi_) * (1 + z_) * (1 - xi_ + 2 * eta_ - z_),
            -0.125 * (1 + xi_) * (1 + z_) * (1 - xi_ - 2 * eta_ - z_),
            -0.125 * (1 - xi_) * (1 + z_) * (1 + xi_ - 2 * eta_ - z_),
            -0.25 * (1 - xi_ * xi_) * (1 - z_),
            -0.5 * eta_ * (1 + xi_) * (1 - z_),
            0.25 * (1 - xi_ * xi_) * (1 - z_),
            -0.5 * eta_ * (1 - xi_) * (1 - z_),
            -0.25 * (1 - xi_ * xi_) * (1 + z_),
            -0.5 * eta_ * (1 + xi_) * (1 + z_),
            0.25 * (1 - xi_ * xi_) * (1 + z_),
            -0.5 * eta_ * (1 - xi_) * (1 + z_),
            -0.25 * (1 - xi_) * (1 - z_ * z_),
            -0.25 * (1 + xi_) * (1 - z_ * z_),
            0.25 * (1 + xi_) * (1 - z_ * z_),
            0.25 * (1 - xi_) * (1 - z_ * z_),
        ]
        dNdXi[i, 2, :] = [
            0.125 * (1 - xi_) * (1 - eta_) * (1 + xi_ + eta_ + 2 * z_),
            0.125 * (1 + xi_) * (1 - eta_) * (1 - xi_ + eta_ + 2 * z_),
            0.125 * (1 + xi_) * (1 + eta_) * (1 - xi_ - eta_ + 2 * z_),
            0.125 * (1 - xi_) * (1 + eta_) * (1 + xi_ - eta_ + 2 * z_),
            -0.125 * (1 - xi_) * (1 - eta_) * (1 + xi_ + eta_ - 2 * z_),
            -0.125 * (1 + xi_) * (1 - eta_) * (1 - xi_ + eta_ - 2 * z_),
            -0.125 * (1 + xi_) * (1 + eta_) * (1 - xi_ - eta_ - 2 * z_),
            -0.125 * (1 - xi_) * (1 + eta_) * (1 + xi_ - eta_ - 2 * z_),
            -0.25 * (1 - xi_ * xi_) * (1 - eta_),
            -0.25 * (1 - eta_ * eta_) * (1 + xi_),
            -0.25 * (1 - xi_ * xi_) * (1 + eta_),
            -0.25 * (1 - eta_ * eta_) * (1 - xi_),
            0.25 * (1 - xi_ * xi_) * (1 - eta_),
            0.25 * (1 - eta_ * eta_) * (1 + xi_),
            0.25 * (1 - xi_ * xi_) * (1 + eta_),
            0.25 * (1 - eta_ * eta_) * (1 - xi_),
            -0.5 * (1 - xi_) * (1 - eta_) * z_,
            -0.5 * (1 + xi_) * (1 - eta_) * z_,
            -0.5 * (1 + xi_) * (1 + eta_) * z_,
            -0.5 * (1 - xi_) * (1 + eta_) * z_,
        ]
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

    dNdXi = _dNdXi3D20(xi, eta, z, nInt)
    J = np.zeros([nInt, 3, 3])
    for i in range(nInt):
        J[i] = dNdXi[i] @ x.T
    return J


def _B2D4(xi: np.ndarray, eta: np.ndarray, x: np.ndarray, nInt: int):
    """Get the B operator for a linear Quad4 element.

    Parameters
    ----------
    xi
        Local coordinates xi for the integration points.
    eta
        Local coordinates eta for the integration points.
    x
        Coordinates of the element points.
    nInt
        Number of quadrature points the element has.

    Returns
    -------
    np.ndarray
        The requested B operator."""

    Bi = np.zeros([nInt, 3, 8])
    # [a] matrix that connects strain and displacement derivatives
    a = np.array([[1, 0, 0, 0], [0, 0, 0, 1], [0, 1, 1, 0]])
    J = _J2D4(xi, eta, x, nInt)
    for i in range(0, nInt):  # for all Gauss points (N in total)
        # make inverse of Jacobi
        invJ = np.linalg.inv(J[i])
        # [b] connects displacement derivatives (Q4)
        bi = np.array([[invJ, np.zeros([2, 2])], [np.zeros([2, 2]), invJ]])
        # make [b] what it should actually look like
        b = bi.transpose(0, 2, 1, 3).reshape(4, 4)
        # [h] as a temporary matrix
        h = np.array(
            [
                [
                    -1 / 4 * (1 - xi[i]),
                    0,
                    1 / 4 * (1 - xi[i]),
                    0,
                    1 / 4 * (1 + xi[i]),
                    0,
                    -1 / 4 * (1 + xi[i]),
                ],
                [
                    -1 / 4 * (1 - eta[i]),
                    0,
                    -1 / 4 * (1 + eta[i]),
                    0,
                    1 / 4 * (1 + eta[i]),
                    0,
                    1 / 4 * (1 - eta[i]),
                ],
            ]
        )
        # assemble [c] differentiated shapefunctions (Q4)
        c = np.vstack([np.hstack([h, np.zeros([2, 1])]), np.hstack([np.zeros([2, 1]), h])])
        # [B] for all different s and t
        Bi[i] = a @ b @ c
    return Bi


def _B2D8(xi: np.ndarray, eta: np.ndarray, x: np.ndarray, nInt: int):
    """Get the B operator for a linear Quad8 element.

    Parameters
    ----------
    xi
        Local coordinates xi for the integration points.
    eta
        Local coordinates eta for the integration points.
    x
        Coordinates of the element points.
    nInt
        Number of quadrature points the element has.

    Returns
    -------
    np.ndarray
        The requested B operator."""

    Bi = np.zeros([nInt, 3, 16])
    # [a] matrix that connects strain and displacement derivatives
    a = np.array([[1, 0, 0, 0], [0, 0, 0, 1], [0, 1, 1, 0]])
    J = _J2D8(xi, eta, x, nInt)
    for i in range(0, nInt):  # for all Gauss points (N in total)
        # make inverse of Jacobi
        invJ = np.linalg.inv(J[i])
        # [b] connects displacement derivatives (Q8)
        bi = np.array([[invJ, np.zeros([2, 2])], [np.zeros([2, 2]), invJ]])
        # make [b] what it should actually look like
        b = bi.transpose(0, 2, 1, 3).reshape(4, 4)
        # [h] as a temporary matrix
        h = np.array(
            [
                [
                    -1 / 4 * (-1 + xi[i]) * (2 * eta[i] + xi[i]),
                    0,
                    1 / 4 * (-1 + xi[i]) * (xi[i] - 2 * eta[i]),
                    0,
                    1 / 4 * (1 + xi[i]) * (2 * eta[i] + xi[i]),
                    0,
                    -1 / 4 * (1 + xi[i]) * (xi[i] - 2 * eta[i]),
                    0,
                    eta[i] * (-1 + xi[i]),
                    0,
                    -1 / 2 * (1 + xi[i]) * (-1 + xi[i]),
                    0,
                    -eta[i] * (1 + xi[i]),
                    0,
                    1 / 2 * (1 + xi[i]) * (-1 + xi[i]),
                ],
                [
                    -1 / 4 * (-1 + eta[i]) * (eta[i] + 2 * xi[i]),
                    0,
                    1 / 4 * (1 + eta[i]) * (2 * xi[i] - eta[i]),
                    0,
                    1 / 4 * (1 + eta[i]) * (eta[i] + 2 * xi[i]),
                    0,
                    -1 / 4 * (-1 + eta[i]) * (2 * xi[i] - eta[i]),
                    0,
                    1 / 2 * (1 + eta[i]) * (-1 + eta[i]),
                    0,
                    -xi[i] * (1 + eta[i]),
                    0,
                    -1 / 2 * (1 + eta[i]) * (-1 + eta[i]),
                    0,
                    xi[i] * (-1 + eta[i]),
                ],
            ]
        )
        # assemble [c] differentiated shapefunctions (Q8)
        c = np.vstack([np.hstack([h, np.zeros([2, 1])]), np.hstack([np.zeros([2, 1]), h])])
        # [B] for all different s and t
        Bi[i] = a @ b @ c
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
    J = _J3D8(xi, eta, z, x, nInt)
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
    J = _J3D20(xi, eta, z, x, nInt)
    return _B3DGeneric(dNdXi, J, nInt, 20)
