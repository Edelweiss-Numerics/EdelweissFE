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
Hexa8/Hexa20 shape functions and their local derivatives, in the standard Abaqus C3D8/C3D20 node
ordering (corner ring 0-3 at zeta=-1, ring 4-7 at zeta=+1) -- matching Marmot's element
formulation, so this is the only node-ordering convention across the framework. The single source
of truth for both the small-strain (:mod:`edelweissfe.elements.displacementelement`) and
total-Lagrangian (:mod:`edelweissfe.elements.displacementtlelement`) element formulations, so the
ordering only needs to be correct in one place.
"""

import numpy as np


def hexa8N(xi, eta, z):
    """The Hexa8 shape functions at ``(xi, eta, z)``.

    Parameters
    ----------
    xi, eta, z
        Local coordinates, either scalars or arrays of the same shape.

    Returns
    -------
    np.ndarray
        Shape ``(8, ...)``: the 8 shape function values (leading dimension), broadcasting over
        any additional shape of the inputs."""

    return (
        1
        / 8
        * np.array(
            [
                (1 - xi) * (1 - eta) * (1 - z),
                (1 + xi) * (1 - eta) * (1 - z),
                (1 + xi) * (1 + eta) * (1 - z),
                (1 - xi) * (1 + eta) * (1 - z),
                (1 - xi) * (1 - eta) * (1 + z),
                (1 + xi) * (1 - eta) * (1 + z),
                (1 + xi) * (1 + eta) * (1 + z),
                (1 - xi) * (1 + eta) * (1 + z),
            ]
        )
    )


def hexa8DNdXi(xi, eta, z):
    """dN/d(xi, eta, zeta) for Hexa8 at ``(xi, eta, z)``.

    Parameters
    ----------
    xi, eta, z
        Local coordinates, either scalars or arrays of the same shape.

    Returns
    -------
    np.ndarray
        Shape ``(3, 8, ...)``: row 0/1/2 are d/dxi, d/deta, d/dzeta; the second dimension is the
        node, broadcasting over any additional shape of the inputs."""

    return (
        1
        / 8
        * np.array(
            [
                [
                    -(1 - eta) * (1 - z),
                    (1 - eta) * (1 - z),
                    (1 + eta) * (1 - z),
                    -(1 + eta) * (1 - z),
                    -(1 - eta) * (1 + z),
                    (1 - eta) * (1 + z),
                    (1 + eta) * (1 + z),
                    -(1 + eta) * (1 + z),
                ],
                [
                    -(1 - xi) * (1 - z),
                    -(1 + xi) * (1 - z),
                    (1 + xi) * (1 - z),
                    (1 - xi) * (1 - z),
                    -(1 - xi) * (1 + z),
                    -(1 + xi) * (1 + z),
                    (1 + xi) * (1 + z),
                    (1 - xi) * (1 + z),
                ],
                [
                    -(1 - xi) * (1 - eta),
                    -(1 + xi) * (1 - eta),
                    -(1 + xi) * (1 + eta),
                    -(1 - xi) * (1 + eta),
                    (1 - xi) * (1 - eta),
                    (1 + xi) * (1 - eta),
                    (1 + xi) * (1 + eta),
                    (1 - xi) * (1 + eta),
                ],
            ]
        )
    )


def hexa20N(xi, eta, z):
    """The Hexa20 shape functions at ``(xi, eta, z)``.

    Parameters
    ----------
    xi, eta, z
        Local coordinates, either scalars or arrays of the same shape.

    Returns
    -------
    np.ndarray
        Shape ``(20, ...)``: the 20 shape function values (leading dimension), broadcasting over
        any additional shape of the inputs."""

    return (
        1
        / 8
        * np.array(
            [
                -(1 - xi) * (1 - eta) * (1 - z) * (2 + xi + eta + z),
                -(1 + xi) * (1 - eta) * (1 - z) * (2 - xi + eta + z),
                -(1 + xi) * (1 + eta) * (1 - z) * (2 - xi - eta + z),
                -(1 - xi) * (1 + eta) * (1 - z) * (2 + xi - eta + z),
                -(1 - xi) * (1 - eta) * (1 + z) * (2 + xi + eta - z),
                -(1 + xi) * (1 - eta) * (1 + z) * (2 - xi + eta - z),
                -(1 + xi) * (1 + eta) * (1 + z) * (2 - xi - eta - z),
                -(1 - xi) * (1 + eta) * (1 + z) * (2 + xi - eta - z),
                2 * (1 - xi**2) * (1 - eta) * (1 - z),
                2 * (1 - eta**2) * (1 + xi) * (1 - z),
                2 * (1 - xi**2) * (1 + eta) * (1 - z),
                2 * (1 - eta**2) * (1 - xi) * (1 - z),
                2 * (1 - xi**2) * (1 - eta) * (1 + z),
                2 * (1 - eta**2) * (1 + xi) * (1 + z),
                2 * (1 - xi**2) * (1 + eta) * (1 + z),
                2 * (1 - eta**2) * (1 - xi) * (1 + z),
                2 * (1 - xi) * (1 - eta) * (1 - z**2),
                2 * (1 + xi) * (1 - eta) * (1 - z**2),
                2 * (1 + xi) * (1 + eta) * (1 - z**2),
                2 * (1 - xi) * (1 + eta) * (1 - z**2),
            ]
        )
    )


def hexa20DNdXi(xi, eta, z):
    """dN/d(xi, eta, zeta) for Hexa20 at ``(xi, eta, z)``.

    Parameters
    ----------
    xi, eta, z
        Local coordinates, either scalars or arrays of the same shape.

    Returns
    -------
    np.ndarray
        Shape ``(3, 20, ...)``: row 0/1/2 are d/dxi, d/deta, d/dzeta; the second dimension is the
        node, broadcasting over any additional shape of the inputs."""

    return np.array(
        [
            [
                0.125 * (1 - eta) * (1 - z) * (1 + 2 * xi + eta + z),
                -0.125 * (1 - eta) * (1 - z) * (1 - 2 * xi + eta + z),
                -0.125 * (1 + eta) * (1 - z) * (1 - 2 * xi - eta + z),
                0.125 * (1 + eta) * (1 - z) * (1 + 2 * xi - eta + z),
                0.125 * (1 - eta) * (1 + z) * (1 + 2 * xi + eta - z),
                -0.125 * (1 - eta) * (1 + z) * (1 - 2 * xi + eta - z),
                -0.125 * (1 + eta) * (1 + z) * (1 - 2 * xi - eta - z),
                0.125 * (1 + eta) * (1 + z) * (1 + 2 * xi - eta - z),
                -0.5 * xi * (1 - eta) * (1 - z),
                0.25 * (1 - eta * eta) * (1 - z),
                -0.5 * xi * (1 + eta) * (1 - z),
                -0.25 * (1 - eta * eta) * (1 - z),
                -0.5 * xi * (1 - eta) * (1 + z),
                0.25 * (1 - eta * eta) * (1 + z),
                -0.5 * xi * (1 + eta) * (1 + z),
                -0.25 * (1 - eta * eta) * (1 + z),
                -0.25 * (1 - eta) * (1 - z * z),
                0.25 * (1 - eta) * (1 - z * z),
                0.25 * (1 + eta) * (1 - z * z),
                -0.25 * (1 + eta) * (1 - z * z),
            ],
            [
                0.125 * (1 - xi) * (1 - z) * (1 + xi + 2 * eta + z),
                0.125 * (1 + xi) * (1 - z) * (1 - xi + 2 * eta + z),
                -0.125 * (1 + xi) * (1 - z) * (1 - xi - 2 * eta + z),
                -0.125 * (1 - xi) * (1 - z) * (1 + xi - 2 * eta + z),
                0.125 * (1 - xi) * (1 + z) * (1 + xi + 2 * eta - z),
                0.125 * (1 + xi) * (1 + z) * (1 - xi + 2 * eta - z),
                -0.125 * (1 + xi) * (1 + z) * (1 - xi - 2 * eta - z),
                -0.125 * (1 - xi) * (1 + z) * (1 + xi - 2 * eta - z),
                -0.25 * (1 - xi * xi) * (1 - z),
                -0.5 * eta * (1 + xi) * (1 - z),
                0.25 * (1 - xi * xi) * (1 - z),
                -0.5 * eta * (1 - xi) * (1 - z),
                -0.25 * (1 - xi * xi) * (1 + z),
                -0.5 * eta * (1 + xi) * (1 + z),
                0.25 * (1 - xi * xi) * (1 + z),
                -0.5 * eta * (1 - xi) * (1 + z),
                -0.25 * (1 - xi) * (1 - z * z),
                -0.25 * (1 + xi) * (1 - z * z),
                0.25 * (1 + xi) * (1 - z * z),
                0.25 * (1 - xi) * (1 - z * z),
            ],
            [
                0.125 * (1 - xi) * (1 - eta) * (1 + xi + eta + 2 * z),
                0.125 * (1 + xi) * (1 - eta) * (1 - xi + eta + 2 * z),
                0.125 * (1 + xi) * (1 + eta) * (1 - xi - eta + 2 * z),
                0.125 * (1 - xi) * (1 + eta) * (1 + xi - eta + 2 * z),
                -0.125 * (1 - xi) * (1 - eta) * (1 + xi + eta - 2 * z),
                -0.125 * (1 + xi) * (1 - eta) * (1 - xi + eta - 2 * z),
                -0.125 * (1 + xi) * (1 + eta) * (1 - xi - eta - 2 * z),
                -0.125 * (1 - xi) * (1 + eta) * (1 + xi - eta - 2 * z),
                -0.25 * (1 - xi * xi) * (1 - eta),
                -0.25 * (1 - eta * eta) * (1 + xi),
                -0.25 * (1 - xi * xi) * (1 + eta),
                -0.25 * (1 - eta * eta) * (1 - xi),
                0.25 * (1 - xi * xi) * (1 - eta),
                0.25 * (1 - eta * eta) * (1 + xi),
                0.25 * (1 - xi * xi) * (1 + eta),
                0.25 * (1 - eta * eta) * (1 - xi),
                -0.5 * (1 - xi) * (1 - eta) * z,
                -0.5 * (1 + xi) * (1 - eta) * z,
                -0.5 * (1 + xi) * (1 + eta) * z,
                -0.5 * (1 - xi) * (1 + eta) * z,
            ],
        ]
    )
