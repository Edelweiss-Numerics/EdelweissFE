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
"""The closest point of a point on a triangle or segment, as the contact searches compute it.

The contact searches select facets by comparing distances, some of them against tolerances, so the
closest-point evaluation has to be exact, not merely close. It is checked three ways, none of which
depends on how the CPU happens to round:

* On geometry where every operation is exact -- right isosceles triangles and segments whose edge
  lengths squared are powers of two, with points on a quarter grid, so no product, sum or quotient
  rounds -- the result must equal, bit for bit, the true closest point, computed in exact rational
  arithmetic from its definition (the projection onto the plane if it lies inside, else the nearest
  point of the nearest edge). This covers every region: the three vertices, the three edges and the
  interior.
* On arbitrary geometry, including degenerate and single-precision triangles, a pair's result must
  not depend on the other pairs evaluated in the same batch.
* The facet selection must pick the first strict minimum over each point's candidates.
"""

import os
import warnings
from fractions import Fraction

import numpy as np
import pytest
import pyvista as pv

from edelweissfe.utils.facetcontactgeometry import (
    closestFacetCandidates,
    closestFacets,
    line2ClosestPoint,
    line2ClosestPoints,
    tria3ClosestPoint,
    tria3ClosestPoints,
)

_stlFile = os.path.join(
    os.path.dirname(__file__),
    "..",
    "testfiles",
    "edelweiss-only",
    "NEDSurfaceToDiscreteRigidBodyContact",
    "support_box.stl",
)


def _assertBitIdentical(actual, expected):
    """Equal including the sign of zero and the position of NaNs."""

    actual, expected = np.asarray(actual), np.asarray(expected)
    assert actual.shape == expected.shape
    assert np.array_equal(actual, expected, equal_nan=True)
    assert np.array_equal(np.signbit(actual), np.signbit(expected))


# --- The true closest point, in exact arithmetic ------------------------------------------------


def _segmentClosest(p, a, b):
    """The exact closest point on the segment a-b: the clamped projection, as weights on (a, b)."""

    e = [bi - ai for ai, bi in zip(a, b)]
    t = sum((pi - ai) * ei for pi, ai, ei in zip(p, a, e)) / sum(ei * ei for ei in e)
    t = min(max(t, Fraction(0)), Fraction(1))
    return 1 - t, t


def _squaredDistance(p, q):
    return sum((pi - qi) ** 2 for pi, qi in zip(p, q))


def _combine(weights, nodes):
    return [sum(w * node[k] for w, node in zip(weights, nodes)) for k in range(len(nodes[0]))]


def _exactTriangleClosest(p, a, b, c):
    """The exact closest point on the triangle a-b-c, as barycentric weights: the projection onto
    the plane if it lies inside the triangle, otherwise the nearest of the edges' closest points."""

    e1 = [bi - ai for ai, bi in zip(a, b)]
    e2 = [ci - ai for ai, ci in zip(a, c)]
    r = [pi - ai for ai, pi in zip(a, p)]
    dot = lambda u, v: sum(ui * vi for ui, vi in zip(u, v))  # noqa: E731
    g11, g12, g22 = dot(e1, e1), dot(e1, e2), dot(e2, e2)
    determinant = g11 * g22 - g12 * g12
    u = (g22 * dot(e1, r) - g12 * dot(e2, r)) / determinant
    v = (g11 * dot(e2, r) - g12 * dot(e1, r)) / determinant
    if u >= 0 and v >= 0 and u + v <= 1:
        return [1 - u - v, u, v]

    candidates = []
    for i, j in ((0, 1), (0, 2), (1, 2)):
        nodes = (a, b, c)
        wi, wj = _segmentClosest(p, nodes[i], nodes[j])
        weights = [Fraction(0)] * 3
        weights[i], weights[j] = wi, wj
        candidates.append((_squaredDistance(p, _combine(weights, nodes)), weights))
    return min(candidates, key=lambda candidate: candidate[0])[1]


def _exactGeometry(rng, n, dimension):
    """Points on a quarter grid and facets with edge lengths squared a power of two, built so that
    the closest-point evaluation never rounds."""

    quarter = lambda size: rng.integers(-24, 25, size) / 4.0  # noqa: E731
    if dimension == 3:
        legDirections = [
            np.eye(3)[[0, 1]],
            np.eye(3)[[1, 2]],
            np.eye(3)[[2, 0]],
            np.array([[1.0, 1.0, 0.0], [-1.0, 1.0, 0.0]]),
            np.array([[0.0, 1.0, 1.0], [0.0, -1.0, 1.0]]),
        ]
        facets = np.empty((n, 3, 3))
        for k in range(n):
            legs = legDirections[rng.integers(len(legDirections))] * rng.choice([-1.0, 1.0], (2, 1))
            length = 2.0 ** rng.integers(0, 3)
            origin = quarter(3)
            facets[k] = [origin, origin + length * legs[0], origin + length * legs[1]]
    else:
        directions = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [1.0, -1.0]])
        facets = np.empty((n, 2, 2))
        for k in range(n):
            origin = quarter(2)
            length = 2.0 ** rng.integers(0, 3)
            facets[k] = [origin, origin + length * rng.choice([-1.0, 1.0]) * directions[rng.integers(4)]]
    return quarter((n, dimension)), facets


def _fractions(array):
    return [Fraction(float(x)) for x in array]


def test_tria3_equals_the_exact_closest_point():
    rng = np.random.default_rng(0)
    points, triangles = _exactGeometry(rng, 4000, 3)

    weights, distances = tria3ClosestPoints(points, triangles[:, 0], triangles[:, 1], triangles[:, 2])

    regions = set()
    for i in range(len(points)):
        nodes = [_fractions(node) for node in triangles[i]]
        exactWeights = _exactTriangleClosest(_fractions(points[i]), *nodes)
        exactSquaredDistance = _squaredDistance(_fractions(points[i]), _combine(exactWeights, nodes))
        _assertBitIdentical(weights[i], np.array([float(w) for w in exactWeights]))
        _assertBitIdentical(distances[i], np.sqrt(float(exactSquaredDistance)))
        regions.add(tuple(w > 0 for w in exactWeights))

    # every region: three vertices, three edges, the interior
    assert len(regions) == 7


def test_line2_equals_the_exact_closest_point():
    rng = np.random.default_rng(1)
    points, segments = _exactGeometry(rng, 4000, 2)

    weights, distances = line2ClosestPoints(points, segments[:, 0], segments[:, 1])

    for i in range(len(points)):
        a, b = (_fractions(node) for node in segments[i])
        exactWeights = _segmentClosest(_fractions(points[i]), a, b)
        exactSquaredDistance = _squaredDistance(_fractions(points[i]), _combine(exactWeights, (a, b)))
        _assertBitIdentical(weights[i], np.array([float(w) for w in exactWeights]))
        _assertBitIdentical(distances[i], np.sqrt(float(exactSquaredDistance)))


# --- A pair's result does not depend on the batch ------------------------------------------------


def _arbitraryTriangles(rng, n):
    """Points and triangles over many length scales, including degenerate triangles and points
    exactly on vertices and edges."""

    triangles = rng.standard_normal((n, 3, 3)) * rng.uniform(1e-3, 1e3, (n, 1, 1))
    points = rng.standard_normal((n, 3)) * rng.uniform(1e-3, 3e3, (n, 1))
    q = n // 10
    triangles[:q, 2] = triangles[:q, 0] + 0.5 * (triangles[:q, 1] - triangles[:q, 0])  # collinear
    triangles[q : 2 * q, 1] = triangles[q : 2 * q, 0]  # two coincident nodes
    triangles[2 * q : 3 * q] = triangles[2 * q : 3 * q, :1]  # a single point
    points[3 * q : 4 * q] = triangles[3 * q : 4 * q, 2]
    points[4 * q : 5 * q] = 0.5 * (triangles[4 * q : 5 * q, 1] + triangles[4 * q : 5 * q, 2])
    return points, triangles


def _assertBatchIndependent(closestPoints, points, facets, rng):
    """Every pair evaluated in a shuffled batch, and in small batches, gives the bits of the whole."""

    nodes = [facets[:, k] for k in range(facets.shape[1])]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # degenerate facets divide by zero
        weights, distances = closestPoints(points, *nodes)

        order = rng.permutation(len(points))
        shuffledWeights, shuffledDistances = closestPoints(points[order], *(node[order] for node in nodes))
        _assertBitIdentical(shuffledWeights, weights[order])
        _assertBitIdentical(shuffledDistances, distances[order])

        for start in range(0, len(points), 7):
            batch = slice(start, start + 7)
            batchWeights, batchDistances = closestPoints(points[batch], *(node[batch] for node in nodes))
            _assertBitIdentical(batchWeights, weights[batch])
            _assertBitIdentical(batchDistances, distances[batch])
    return weights


@pytest.mark.parametrize("seed", [0, 1])
def test_tria3_result_of_a_pair_does_not_depend_on_the_batch(seed):
    rng = np.random.default_rng(seed)
    points, triangles = _arbitraryTriangles(rng, 5000)
    _assertBatchIndependent(tria3ClosestPoints, points, triangles, rng)


@pytest.mark.parametrize("seed", [0, 1])
def test_line2_result_of_a_pair_does_not_depend_on_the_batch(seed):
    rng = np.random.default_rng(seed)
    n = 5000
    segments = rng.standard_normal((n, 2, 2)) * rng.uniform(1e-3, 1e3, (n, 1, 1))
    points = rng.standard_normal((n, 2)) * rng.uniform(1e-3, 3e3, (n, 1))
    segments[: n // 10, 1] = segments[: n // 10, 0]  # degenerate
    _assertBatchIndependent(line2ClosestPoints, points, segments, rng)


def _stlTriangles():
    """The triangles of a real STL rigid surface, in single precision as the file stores them."""

    surface = pv.read(_stlFile).triangulate()
    return surface.points[surface.regular_faces]


def _pointsAroundAndOn(rng, triangles, n):
    """Points near a surface, plus its vertices, edge midpoints and centroids."""

    lower, upper = triangles.reshape(-1, 3).min(axis=0), triangles.reshape(-1, 3).max(axis=0)
    margin = 0.2 * (upper - lower)
    near = rng.uniform(lower - margin, upper + margin, (n, 3))
    onSurface = np.concatenate(
        [triangles.reshape(-1, 3), 0.5 * (triangles[:, 0] + triangles[:, 1]), triangles.mean(axis=1)]
    )
    return np.concatenate([near, onSurface.astype(float)])


def test_tria3_on_a_single_precision_stl_surface():
    """Single-precision triangles against double-precision points, as in the rigid-body search."""

    rng = np.random.default_rng(3)
    triangles = _stlTriangles()
    assert triangles.dtype == np.float32
    points = _pointsAroundAndOn(rng, triangles, 300)

    pointIndices = np.repeat(np.arange(len(points)), len(triangles))
    pairTriangles = triangles[np.tile(np.arange(len(triangles)), len(points))]
    weights = _assertBatchIndependent(tria3ClosestPoints, points[pointIndices], pairTriangles, rng)
    assert np.all((weights >= 0.0) & (weights <= 1.0))


# --- Facet selection --------------------------------------------------------------------------


def _closestFacetCandidatesScalar(queryPoints, facetCoords, searchDistance):
    """The broadphase with its per-point pruning loop, as it was before it was batched."""

    from scipy.spatial import cKDTree

    stacked = np.asarray(facetCoords, dtype=float)
    centroids = stacked.mean(axis=1)
    radii = np.linalg.norm(stacked - centroids[:, None, :], axis=2).max(axis=1)
    tree = cKDTree(centroids)
    nearestCentroidDistance, _ = tree.query(queryPoints, k=1)
    radius = nearestCentroidDistance + float(radii.max())
    if searchDistance is not None:
        radius = np.minimum(radius, searchDistance + float(radii.max()))
    radius = radius * (1.0 + 8.0 * np.finfo(float).eps)
    tolerance = 1.0 + 8.0 * np.finfo(float).eps

    candidates = []
    for p, ballCandidate in enumerate(tree.query_ball_point(queryPoints, radius)):
        indices = np.fromiter(ballCandidate, dtype=np.intp, count=len(ballCandidate))
        centroidDistance = np.linalg.norm(centroids[indices] - queryPoints[p], axis=1)
        surviving = indices[centroidDistance - radii[indices] <= nearestCentroidDistance[p] * tolerance]
        surviving.sort()
        candidates.append(surviving.tolist())
    return candidates


def _scanForClosest(queryPoints, facetCoords, candidatesPerPoint, closestPoint):
    """The pair-by-pair selection: the first strict minimum over each point's candidates."""

    selection = []
    for p, candidates in enumerate(candidatesPerPoint):
        bestFacet, bestWeights, bestDistance = None, None, np.inf
        for i in candidates:
            weights, distance = closestPoint(queryPoints[p], *facetCoords[i])
            if distance < bestDistance:
                bestFacet, bestWeights, bestDistance = i, weights, distance
        selection.append((bestFacet, bestWeights, bestDistance))
    return selection


def _gradedSurface(rng):
    """A structured, triangulated plane (exact ties between neighbours) next to a random patch of
    facets that differ in size by two orders of magnitude."""

    plane = pv.Plane(i_size=4.0, j_size=4.0, i_resolution=8, j_resolution=8).triangulate()
    structured = plane.points[plane.regular_faces].astype(float)
    centers = rng.uniform(-2.0, 2.0, (200, 3)) + np.array([6.0, 0.0, 0.0])
    random = centers[:, None, :] + rng.standard_normal((200, 3, 3)) * rng.uniform(0.01, 1.0, (200, 1, 1))
    return list(np.concatenate([structured, random]))


@pytest.mark.parametrize("searchDistance", [None, 0.3])
def test_closest_facets_selects_the_first_strict_minimum(searchDistance):
    rng = np.random.default_rng(4)
    facetCoords = _gradedSurface(rng)
    queryPoints = np.concatenate(
        [
            rng.uniform([-3.0, -3.0, -1.0], [9.0, 3.0, 1.0], (2000, 3)),
            np.round(rng.uniform(-2.0, 2.0, (300, 3)) * 2.0) / 2.0,  # on grid lines and nodes
        ]
    )

    candidatesPerPoint = closestFacetCandidates(queryPoints, facetCoords, searchDistance)
    assert candidatesPerPoint == _closestFacetCandidatesScalar(queryPoints, facetCoords, searchDistance)

    closestFacet, closestWeights, closestDistance = closestFacets(queryPoints, facetCoords, candidatesPerPoint)
    expected = _scanForClosest(queryPoints, facetCoords, candidatesPerPoint, tria3ClosestPoint)
    for p, (bestFacet, bestWeights, bestDistance) in enumerate(expected):
        if bestFacet is None:
            assert closestFacet[p] == -1 and closestDistance[p] == np.inf
            continue
        assert closestFacet[p] == bestFacet
        _assertBitIdentical(closestWeights[p], bestWeights)
        _assertBitIdentical(closestDistance[p], bestDistance)


def test_closest_facets_in_2d_and_without_candidates():
    rng = np.random.default_rng(5)
    facetCoords = list(rng.standard_normal((50, 2, 2)))
    queryPoints = rng.uniform(-3.0, 3.0, (500, 2))
    candidatesPerPoint = closestFacetCandidates(queryPoints, facetCoords, None)
    candidatesPerPoint[0] = []

    closestFacet, closestWeights, closestDistance = closestFacets(queryPoints, facetCoords, candidatesPerPoint)
    expected = _scanForClosest(queryPoints, facetCoords, candidatesPerPoint, line2ClosestPoint)
    assert closestFacet[0] == -1 and closestDistance[0] == np.inf
    for p, (bestFacet, bestWeights, bestDistance) in enumerate(expected[1:], start=1):
        assert closestFacet[p] == bestFacet
        _assertBitIdentical(closestWeights[p], bestWeights)
        _assertBitIdentical(closestDistance[p], bestDistance)
