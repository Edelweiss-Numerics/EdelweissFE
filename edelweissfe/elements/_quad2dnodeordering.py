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
Quad4/Quad8 shape functions, their local derivatives and the natural coordinates of their nodes,
in the standard Abaqus CPE4/CPE8 node ordering (corners 0-3 counter-clockwise, then the midpoints
of the edges 0-1, 1-2, 2-3, 3-0). As for the hexahedra (:mod:`edelweissfe.elements._hexa3dnodeordering`),
``xi`` runs along the first natural direction (from node 0 to node 1) and ``eta`` along the second
(from node 0 to node 3). The single source of truth for the small-strain
(:mod:`edelweissfe.elements.displacementelement`) and total-Lagrangian
(:mod:`edelweissfe.elements.displacementtlelement`) element formulations and the Ensight export.
"""

import numpy as np

#: Natural coordinates ``(xi, eta)`` of the Quad8 nodes: the 4 corners, then the 4 edge midpoints.
quad8NodeXi = np.array(
    [
        [-1, -1], [1, -1], [1, 1], [-1, 1],
        [0, -1], [1, 0], [0, 1], [-1, 0],
    ]
)  # fmt: skip

#: Natural coordinates ``(xi, eta)`` of the Quad4 nodes.
quad4NodeXi = quad8NodeXi[:4]


def quad4N(xi, eta):
    """The Quad4 shape functions at ``(xi, eta)``.

    Parameters
    ----------
    xi, eta
        Local coordinates, either scalars or arrays of the same shape.

    Returns
    -------
    np.ndarray
        Shape ``(4, ...)``: the 4 shape function values (leading dimension), broadcasting over any
        additional shape of the inputs."""

    return 0.25 * np.array(
        [
            (1 - xi) * (1 - eta),
            (1 + xi) * (1 - eta),
            (1 + xi) * (1 + eta),
            (1 - xi) * (1 + eta),
        ]
    )


def quad4DNdXi(xi, eta):
    """dN/d(xi, eta) for Quad4 at ``(xi, eta)``.

    Parameters
    ----------
    xi, eta
        Local coordinates, either scalars or arrays of the same shape.

    Returns
    -------
    np.ndarray
        Shape ``(2, 4, ...)``: row 0/1 are d/dxi, d/deta; the second dimension is the node,
        broadcasting over any additional shape of the inputs."""

    return 0.25 * np.array(
        [
            [-(1 - eta), (1 - eta), (1 + eta), -(1 + eta)],
            [-(1 - xi), -(1 + xi), (1 + xi), (1 - xi)],
        ]
    )


def quad8N(xi, eta):
    """The Quad8 (serendipity) shape functions at ``(xi, eta)``.

    Parameters
    ----------
    xi, eta
        Local coordinates, either scalars or arrays of the same shape.

    Returns
    -------
    np.ndarray
        Shape ``(8, ...)``: the 8 shape function values (leading dimension), broadcasting over any
        additional shape of the inputs."""

    return 0.25 * np.array(
        [
            (1 - xi) * (1 - eta) * (-xi - eta - 1),
            (1 + xi) * (1 - eta) * (xi - eta - 1),
            (1 + xi) * (1 + eta) * (xi + eta - 1),
            (1 - xi) * (1 + eta) * (-xi + eta - 1),
            2 * (1 - xi**2) * (1 - eta),
            2 * (1 + xi) * (1 - eta**2),
            2 * (1 - xi**2) * (1 + eta),
            2 * (1 - xi) * (1 - eta**2),
        ]
    )


def quad8DNdXi(xi, eta):
    """dN/d(xi, eta) for Quad8 at ``(xi, eta)``.

    Parameters
    ----------
    xi, eta
        Local coordinates, either scalars or arrays of the same shape.

    Returns
    -------
    np.ndarray
        Shape ``(2, 8, ...)``: row 0/1 are d/dxi, d/deta; the second dimension is the node,
        broadcasting over any additional shape of the inputs."""

    return 0.25 * np.array(
        [
            [
                (1 - eta) * (2 * xi + eta),
                (1 - eta) * (2 * xi - eta),
                (1 + eta) * (2 * xi + eta),
                (1 + eta) * (2 * xi - eta),
                -4 * xi * (1 - eta),
                2 * (1 - eta**2),
                -4 * xi * (1 + eta),
                -2 * (1 - eta**2),
            ],
            [
                (1 - xi) * (xi + 2 * eta),
                (1 + xi) * (2 * eta - xi),
                (1 + xi) * (xi + 2 * eta),
                (1 - xi) * (2 * eta - xi),
                -2 * (1 - xi**2),
                -4 * eta * (1 + xi),
                2 * (1 - xi**2),
                -4 * eta * (1 - xi),
            ],
        ]
    )
