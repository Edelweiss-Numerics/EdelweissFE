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
"""Unit tests for the flat-facet projections in :mod:`edelweissfe.utils.facetcontactgeometry`."""

import unittest

import numpy as np

from edelweissfe.utils.facetcontactgeometry import (
    facetNormalAndMeasure,
    line2ClosestPoint,
    line2Projection,
    tria3ClosestPoint,
    tria3Projection,
)


class TestFacetProjections(unittest.TestCase):
    def setUp(self):
        self.triangle = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        self.segment = np.array([[0.0, 0.0], [2.0, 0.0]])

    def test_tria3_projection_inside_agrees_with_closest_point(self):
        point = np.array([0.5, 0.25, 0.7])
        alpha, beta, inside = tria3Projection(point, *self.triangle)
        weights, distance = tria3ClosestPoint(point, *self.triangle)

        self.assertTrue(inside)
        np.testing.assert_allclose(weights, [1.0 - alpha - beta, alpha, beta], rtol=0, atol=1e-15)
        self.assertAlmostEqual(distance, 0.7, places=15)

    def test_tria3_projection_outside_is_not_clamped(self):
        alpha, beta, inside = tria3Projection(np.array([3.0, 0.5, -0.2]), *self.triangle)
        self.assertFalse(inside)
        self.assertAlmostEqual(alpha, 1.5, places=15)
        self.assertAlmostEqual(beta, 0.5, places=15)

    def test_line2_closest_point_is_clamped_projection(self):
        for x in (-1.0, 0.0, 0.7, 2.0, 3.5):
            point = np.array([x, 0.3])
            t, inside = line2Projection(point, *self.segment)
            weights, _ = line2ClosestPoint(point, *self.segment)

            self.assertEqual(inside, 0.0 <= t <= 1.0)
            self.assertEqual(weights[1], np.clip(t, 0.0, 1.0))

    def test_facet_normal_and_measure(self):
        normal, area = facetNormalAndMeasure(self.triangle)
        np.testing.assert_array_equal(normal, [0.0, 0.0, 1.0])
        self.assertEqual(area, 1.0)

        normal, length = facetNormalAndMeasure(self.segment)
        np.testing.assert_array_equal(normal, [0.0, -1.0])
        self.assertEqual(length, 2.0)


if __name__ == "__main__":
    unittest.main()
