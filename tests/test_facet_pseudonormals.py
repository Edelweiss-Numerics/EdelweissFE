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
"""The sign of the distance to a faceted surface that is not convex: the angle-weighted pseudonormals of
:class:`~edelweissfe.utils.facetcontactgeometry.FacetedSurface`.

On a closed surface shaped like a thread tooth on a core -- a polygon in 2D, a prism in 3D, with an acute crest and two
re-entrant roots -- the gap of a point to its closest facet must be negative exactly inside the body and equal in
magnitude to the distance to the surface, at every feature of the closest point: inside a facet, on an edge, at a
vertex. The normal of the closest facet alone gets the sign wrong next to the crest, where the normals of the two
flanks differ by more than a right angle, which the test shows as well. On a surface that ends, the facet's plane goes
on beyond its boundary.
"""

import numpy as np
import pytest

from edelweissfe.utils.facetcontactgeometry import (
    FacetedSurface,
    closestFacets,
    facetNormalAndMeasure,
    rowDot,
)

#: A tooth on a core, counter-clockwise: the core [0, 3] x [0, 1], the tooth's crest (1.5, 3) acute, its roots (2, 1)
#: and (1, 1) re-entrant.
_toothPolygon = np.array([[0.0, 0.0], [3.0, 0.0], [3.0, 1.0], [2.0, 1.0], [1.5, 3.0], [1.0, 1.0], [0.0, 1.0]])

#: The tooth polygon in triangles, counter-clockwise: a fan over the core, and the tooth.
_toothTriangles = ((0, 1, 2), (0, 2, 3), (0, 3, 5), (0, 5, 6), (3, 4, 5))


def _insideTheTooth(points: np.ndarray) -> np.ndarray:
    """Whether points lie inside the tooth polygon: an odd number of its edges crosses the ray to +x."""

    x, y = points[:, 0], points[:, 1]
    inside = np.zeros(len(points), dtype=bool)
    for (xa, ya), (xb, yb) in zip(_toothPolygon, np.roll(_toothPolygon, -1, axis=0)):
        straddles = (ya > y) != (yb > y)
        with np.errstate(divide="ignore", invalid="ignore"):
            crossing = xa + (y - ya) * (xb - xa) / (yb - ya)
        inside ^= straddles & (x < crossing)
    return inside


def _toothPolygonSurface() -> tuple[np.ndarray, np.ndarray]:
    """The tooth as Line2 facets, outward normals: (vertex coordinates, vertices of each facet)."""

    n = len(_toothPolygon)
    return _toothPolygon, np.array([[i, (i + 1) % n] for i in range(n)])


def _toothPrismSurface() -> tuple[np.ndarray, np.ndarray]:
    """The tooth extruded through z = 0..1 as Tria3 facets, outward normals: (vertex coordinates, vertices of each
    facet), the bottom vertices first."""

    n = len(_toothPolygon)
    vertices = np.vstack((np.column_stack((_toothPolygon, np.zeros(n))), np.column_stack((_toothPolygon, np.ones(n)))))
    facets = []
    for i in range(n):
        j = (i + 1) % n
        facets += [[i, j, j + n], [i, j + n, i + n]]
    for a, b, c in _toothTriangles:
        facets += [[a + n, b + n, c + n], [a, c, b]]
    return vertices, np.array(facets)


@pytest.mark.parametrize("surface", (_toothPolygonSurface, _toothPrismSurface))
def test_the_gap_to_the_closest_facet_tells_inside_from_outside_on_a_thread_tooth(surface):
    vertices, nodesOfFacet = surface()
    nDim = vertices.shape[1]
    coordinatesOfFacet = vertices[nodesOfFacet]
    normalOfFacet, _ = facetNormalAndMeasure(coordinatesOfFacet)
    if nDim == 3:
        # The facets are oriented outward: the divergence theorem gives the volume of the prism.
        x1, x2, x3 = np.swapaxes(coordinatesOfFacet, 0, 1)
        volume = np.sum(rowDot(x1, np.cross(x2 - x1, x3 - x1))) / 6.0
        assert volume == pytest.approx(4.0)

    rng = np.random.default_rng(7)
    points = rng.uniform(-0.5, 3.5, size=(6000, nDim))
    if nDim == 3:
        points[:, 2] = rng.uniform(-0.5, 1.5, size=len(points))
    # Densely around the crest and the roots.
    for k, corner in enumerate(((1.5, 3.0), (2.0, 1.0), (1.0, 1.0))):
        points[1000 * k : 1000 * (k + 1), :2] = corner + rng.uniform(-0.4, 0.4, size=(1000, 2))
    inside = _insideTheTooth(points[:, :2])
    if nDim == 3:
        inside &= (points[:, 2] > 0.0) & (points[:, 2] < 1.0)

    allFacets = list(range(len(nodesOfFacet)))
    closestFacet, weights, distance = closestFacets(points, list(coordinatesOfFacet), [allFacets] * len(points))

    surface = FacetedSurface(nodesOfFacet)
    features = surface.features(coordinatesOfFacet, normalOfFacet)
    gap, direction = surface.signedDistances(points, closestFacet, weights, coordinatesOfFacet, normalOfFacet, features)

    clearOfTheSurface = distance > 1e-9
    np.testing.assert_array_equal((gap < 0.0)[clearOfTheSurface], inside[clearOfTheSurface])
    np.testing.assert_allclose(np.abs(gap), distance, rtol=1e-12, atol=1e-15)
    np.testing.assert_allclose(np.linalg.norm(direction, axis=1), 1.0, rtol=1e-12)
    # The direction is the gradient of the gap: moving the point along it changes the gap by as much.
    closestPoints = np.einsum("pa,pad->pd", weights, coordinatesOfFacet[closestFacet])
    np.testing.assert_allclose(rowDot(direction, points - closestPoints), gap, rtol=1e-12, atol=1e-15)

    facetNormalSaysInside = rowDot(normalOfFacet[closestFacet], points - closestPoints) < 0.0
    assert np.any((facetNormalSaysInside != inside)[clearOfTheSurface]), "the facet normal alone should fail here"


def test_a_surface_that_ends_goes_on_in_the_plane_of_its_boundary_facet():
    """Beyond the boundary of an open surface, the gap is the distance to the plane of the closest facet: the surface
    does not say what lies beyond it, and the gap stays continuous where a point crosses that plane."""

    vertices = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
    nodesOfFacet = np.array([[1, 0], [2, 1]])  # the top of a body below y = 0, outward normal +y
    coordinatesOfFacet = vertices[nodesOfFacet]
    normalOfFacet, _ = facetNormalAndMeasure(coordinatesOfFacet)
    np.testing.assert_allclose(normalOfFacet, [[0.0, 1.0], [0.0, 1.0]])

    surface = FacetedSurface(nodesOfFacet)
    features = surface.features(coordinatesOfFacet, normalOfFacet)
    points = np.array([[2.5, -0.1], [2.5, 0.1], [-0.3, -0.2], [1.0, 0.3], [1.0, -0.3]])
    closestFacet, weights, _ = closestFacets(points, list(coordinatesOfFacet), [[0, 1]] * len(points))
    gap, direction = surface.signedDistances(points, closestFacet, weights, coordinatesOfFacet, normalOfFacet, features)

    np.testing.assert_allclose(gap, [-0.1, 0.1, -0.2, 0.3, -0.3], rtol=1e-14)
    np.testing.assert_allclose(direction, np.tile([0.0, 1.0], (5, 1)), atol=1e-15)


def test_outside_a_convex_edge_the_gap_is_the_distance_to_the_edge():
    """A point beside a crest -- beyond the edge of the crest's flat, below its plane, but outside the body -- has a
    positive gap, the true distance to the edge, in the direction from the edge to the point."""

    vertices = np.array([[0.0, 1.0], [1.0, 1.0], [1.0, 0.0]])
    nodesOfFacet = np.array([[1, 0], [2, 1]])  # the top (normal +y) and the side (normal +x) of the block [0,1]^2
    coordinatesOfFacet = vertices[nodesOfFacet]
    normalOfFacet, _ = facetNormalAndMeasure(coordinatesOfFacet)
    surface = FacetedSurface(nodesOfFacet)
    features = surface.features(coordinatesOfFacet, normalOfFacet)

    point = np.array([[1.3, 1.4]])
    gap, direction = surface.signedDistances(
        point, np.array([0]), np.array([[1.0, 0.0]]), coordinatesOfFacet, normalOfFacet, features
    )
    np.testing.assert_allclose(gap, [0.5], rtol=1e-14)
    np.testing.assert_allclose(direction, [[0.6, 0.8]], rtol=1e-14)


def test_on_a_feature_the_pseudonormal_gives_the_direction():
    """A point on a vertex has no direction of its own: the gap is zero, the direction the vertex's pseudonormal."""

    vertices, nodesOfFacet = _toothPolygonSurface()
    coordinatesOfFacet = vertices[nodesOfFacet]
    normalOfFacet, _ = facetNormalAndMeasure(coordinatesOfFacet)
    surface = FacetedSurface(nodesOfFacet)
    features = surface.features(coordinatesOfFacet, normalOfFacet)

    # The root (2, 1) is vertex 3, the end of facet 2 (the top of the core, normal +y) and the start of facet 3 (the
    # flank, normal (2, 0.5) / |(2, 0.5)|).
    gap, direction = surface.signedDistances(
        np.array([[2.0, 1.0]]), np.array([2]), np.array([[0.0, 1.0]]), coordinatesOfFacet, normalOfFacet, features
    )
    np.testing.assert_array_equal(gap, [0.0])
    pseudonormal = np.array([0.0, 1.0]) + np.array([2.0, 0.5]) / np.hypot(2.0, 0.5)
    np.testing.assert_allclose(direction, [pseudonormal / np.linalg.norm(pseudonormal)], rtol=1e-14)
