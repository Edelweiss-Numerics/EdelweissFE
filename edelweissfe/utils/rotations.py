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
Finite rotations in 3D, as used by rigid bodies and contact.

A rotation is stored as a rotation pseudo-vector :math:`\\boldsymbol{\\theta}` (axis times angle).
The rotation matrix follows from the exponential map, :math:`R = \\exp(\\mathrm{skew}(\\boldsymbol{\\theta}))`,
evaluated with Rodrigues' formula.
"""

import numpy as np


def skewMatrix(v: np.ndarray) -> np.ndarray:
    """The skew-symmetric cross-product matrix of a 3-vector, such that ``skewMatrix(v) @ x == v x x``.

    Parameters
    ----------
    v
        The 3-vector.

    Returns
    -------
    np.ndarray
        The 3x3 skew-symmetric matrix.
    """
    return np.array(
        [
            [0.0, -v[2], v[1]],
            [v[2], 0.0, -v[0]],
            [-v[1], v[0], 0.0],
        ]
    )


def rotationMatrixFromPseudoVector(theta: np.ndarray) -> np.ndarray:
    """Convert a rotation pseudo-vector to a 3x3 rotation matrix via the exponential map (Rodrigues' formula).

    Parameters
    ----------
    theta
        The rotation pseudo-vector (axis times angle).

    Returns
    -------
    np.ndarray
        The 3x3 rotation matrix.
    """
    angle = np.linalg.norm(theta)
    if angle < 1e-12:
        return np.eye(3)
    axis = theta / angle
    K = skewMatrix(axis)
    R = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * np.dot(K, K)
    return R


def rightJacobianSO3(theta: np.ndarray) -> np.ndarray:
    """The right Jacobian of the SO(3) exponential map at the rotation pseudo-vector ``theta``.

    For a fixed body-frame vector :math:`\\bar{\\mathbf{v}}` and
    :math:`R(\\boldsymbol{\\theta}) = \\exp(\\mathrm{skew}(\\boldsymbol{\\theta}))`, this satisfies

    .. math::
        \\frac{\\partial (R(\\boldsymbol{\\theta}) \\bar{\\mathbf{v}})}{\\partial \\boldsymbol{\\theta}}
        = -\\mathrm{skew}(R(\\boldsymbol{\\theta}) \\bar{\\mathbf{v}}) \\, R(\\boldsymbol{\\theta}) \\,
        J_r(\\boldsymbol{\\theta})

    i.e. it maps a perturbation of the *stored, total* pseudo-vector DOF onto the corresponding
    spatial rotation increment -- exact for any accumulated rotation, not just a small-angle
    approximation (which would correspond to ``J_r(theta) = I``).

    Parameters
    ----------
    theta
        The rotation pseudo-vector (axis times angle).

    Returns
    -------
    np.ndarray
        The 3x3 right Jacobian.
    """
    angle = np.linalg.norm(theta)
    K = skewMatrix(theta)
    if angle < 1e-8:
        return np.eye(3) - 0.5 * K + np.dot(K, K) / 6.0
    return np.eye(3) - (1.0 - np.cos(angle)) / angle**2 * K + (angle - np.sin(angle)) / angle**3 * np.dot(K, K)
