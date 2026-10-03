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
"""Unit tests for the finite-rotation helpers in :mod:`edelweissfe.utils.rotations`."""

import unittest

import numpy as np
from scipy.spatial.transform import Rotation

from edelweissfe.utils.rotations import (
    rightJacobianSO3,
    rotationMatrixFromPseudoVector,
    skewMatrix,
)


class TestRotations(unittest.TestCase):
    def setUp(self):
        self.theta = np.array([0.3, -0.7, 1.1])
        self.vector = np.array([0.2, 0.5, -1.3])

    def test_skew_matrix_is_cross_product(self):
        x = np.array([1.0, -2.0, 0.5])
        np.testing.assert_allclose(skewMatrix(self.vector) @ x, np.cross(self.vector, x), rtol=0, atol=1e-15)

    def test_rotation_matrix_matches_rotation_vector(self):
        expected = Rotation.from_rotvec(self.theta).as_matrix()
        np.testing.assert_allclose(rotationMatrixFromPseudoVector(self.theta), expected, rtol=0, atol=1e-14)
        np.testing.assert_array_equal(rotationMatrixFromPseudoVector(np.zeros(3)), np.eye(3))

    def test_right_jacobian_against_finite_differences(self):
        # d(R v)/d(theta) = -skew(R v) R J_r(theta)
        R = rotationMatrixFromPseudoVector(self.theta)
        exact = -skewMatrix(R @ self.vector) @ R @ rightJacobianSO3(self.theta)

        perturbation = 1e-6
        dRotatedVector_dTheta = np.zeros((3, 3))
        for j in range(3):
            delta = np.zeros(3)
            delta[j] = perturbation
            forward = rotationMatrixFromPseudoVector(self.theta + delta) @ self.vector
            backward = rotationMatrixFromPseudoVector(self.theta - delta) @ self.vector
            dRotatedVector_dTheta[:, j] = (forward - backward) / (2 * perturbation)

        np.testing.assert_allclose(exact, dRotatedVector_dTheta, rtol=0, atol=1e-9)

    def test_right_jacobian_small_angle_branch(self):
        np.testing.assert_allclose(rightJacobianSO3(np.full(3, 1e-10)), np.eye(3), rtol=0, atol=1e-9)


if __name__ == "__main__":
    unittest.main()
