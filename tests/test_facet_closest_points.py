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
"""The batched closest-point evaluation must be bit-identical to a pair-by-pair one.

The contact searches select facets by comparing distances, some of them against tolerances, so a
result that is merely equal within rounding could select a different facet and change a simulation.
The references here are the scalar, branch-by-branch forms the batched functions replaced, kept
verbatim: Ericson's region test with one Python branch per region, and the broadphase's per-point
pruning loop. They are compared bit for bit, on random geometry, on geometry that lands exactly on
region boundaries (vertices, edges, integer grids with exact ties), on degenerate triangles, and on a
real single-precision STL surface, as the rigid-body contact reads it.
"""

import os
import warnings

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


def _tria3ClosestPointScalar(xs, x1, x2, x3):
    """The pair-by-pair region test, as it was before it was batched."""

    e1 = x2 - x1
    e2 = x3 - x1
    r1 = xs - x1

    d1 = e1.dot(r1)
    d2 = e2.dot(r1)
    if d1 <= 0.0 and d2 <= 0.0:
        weights = np.array([1.0, 0.0, 0.0])
    else:
        r2 = xs - x2
        d3 = e1.dot(r2)
        d4 = e2.dot(r2)
        if d3 >= 0.0 and d4 <= d3:
            weights = np.array([0.0, 1.0, 0.0])
        else:
            r3 = xs - x3
            d5 = e1.dot(r3)
            d6 = e2.dot(r3)
            vc = d1 * d4 - d3 * d2
            va = d3 * d6 - d5 * d4
            vb = d5 * d2 - d1 * d6
            if d6 >= 0.0 and d5 <= d6:
                weights = np.array([0.0, 0.0, 1.0])
            elif vc <= 0.0 and d1 >= 0.0 and d3 <= 0.0:
                t = d1 / (d1 - d3)
                weights = np.array([1.0 - t, t, 0.0])
            elif vb <= 0.0 and d2 >= 0.0 and d6 <= 0.0:
                t = d2 / (d2 - d6)
                weights = np.array([1.0 - t, 0.0, t])
            elif va <= 0.0 and (d4 - d3) >= 0.0 and (d5 - d6) >= 0.0:
                t = (d4 - d3) / ((d4 - d3) + (d5 - d6))
                weights = np.array([0.0, 1.0 - t, t])
            else:
                denom = 1.0 / (va + vb + vc)
                beta = vb * denom
                gamma = vc * denom
                weights = np.array([1.0 - beta - gamma, beta, gamma])

    closestPoint = weights[0] * x1 + weights[1] * x2 + weights[2] * x3
    return weights, float(np.linalg.norm(xs - closestPoint))


def _line2ClosestPointScalar(xs, x1, x2):
    """The pair-by-pair segment projection, as it was before it was batched."""

    e = x2 - x1
    t = np.clip((xs - x1).dot(e) / e.dot(e), 0.0, 1.0)
    weights = np.array([1.0 - t, t])
    closestPoint = weights[0] * x1 + weights[1] * x2
    return weights, float(np.linalg.norm(xs - closestPoint))


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


def _assertBitIdentical(actual, expected):
    """Equal including the sign of zero and the position of NaNs."""

    actual, expected = np.asarray(actual), np.asarray(expected)
    assert actual.shape == expected.shape
    assert np.array_equal(actual, expected, equal_nan=True)
    assert np.array_equal(np.signbit(actual), np.signbit(expected))


def _randomPairs(rng, n):
    """Points and triangles over many length scales, with every region boundary hit exactly."""

    triangles = rng.standard_normal((n, 3, 3)) * rng.uniform(1e-3, 1e3, (n, 1, 1))
    points = rng.standard_normal((n, 3)) * rng.uniform(1e-3, 3e3, (n, 1))
    q = n // 10
    # degenerate triangles: collinear, two coincident nodes, a single point
    triangles[:q, 2] = triangles[:q, 0] + 0.5 * (triangles[:q, 1] - triangles[:q, 0])
    triangles[q : 2 * q, 1] = triangles[q : 2 * q, 0]
    triangles[2 * q : 3 * q] = triangles[2 * q : 3 * q, :1]
    # points exactly at a vertex, on an edge, in the plane
    points[3 * q : 4 * q] = triangles[3 * q : 4 * q, 2]
    points[4 * q : 5 * q] = 0.5 * (triangles[4 * q : 5 * q, 1] + triangles[4 * q : 5 * q, 2])
    points[5 * q : 6 * q] = triangles[5 * q : 6 * q].mean(axis=1)
    # integer grid: exact zeros and exact ties between regions
    triangles[6 * q :] = np.round(triangles[6 * q :] / 100.0)
    points[6 * q :] = np.round(points[6 * q :] / 100.0)
    return points, triangles


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


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_tria3_batched_is_bit_identical_to_pairwise(seed):
    rng = np.random.default_rng(seed)
    points, triangles = _randomPairs(rng, 20000)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # degenerate triangles divide by zero
        weights, distances = tria3ClosestPoints(points, triangles[:, 0], triangles[:, 1], triangles[:, 2])
        for i in range(len(points)):
            expectedWeights, expectedDistance = _tria3ClosestPointScalar(points[i], *triangles[i])
            _assertBitIdentical(weights[i], expectedWeights)
            _assertBitIdentical(distances[i], expectedDistance)

            singleWeights, singleDistance = tria3ClosestPoint(points[i], *triangles[i])
            _assertBitIdentical(singleWeights, expectedWeights)
            _assertBitIdentical(singleDistance, expectedDistance)


def test_tria3_batched_covers_every_region():
    """The random pairs above must actually exercise all seven regions."""

    rng = np.random.default_rng(0)
    points, triangles = _randomPairs(rng, 20000)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        weights, _ = tria3ClosestPoints(points, triangles[:, 0], triangles[:, 1], triangles[:, 2])
    nonDegenerate = weights[6000:]  # beyond the degenerate tenths, see _randomPairs
    patterns = {tuple(row) for row in (nonDegenerate > 0.0).astype(int)}
    assert patterns >= {(1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 0), (1, 0, 1), (0, 1, 1), (1, 1, 1)}


def test_tria3_batched_on_single_precision_stl_surface():
    """Single-precision triangles against double-precision points, as in the rigid-body search."""

    rng = np.random.default_rng(3)
    triangles = _stlTriangles()
    assert triangles.dtype == np.float32
    points = _pointsAroundAndOn(rng, triangles, 3000)

    pointIndices = np.repeat(np.arange(len(points)), len(triangles))
    triangleIndices = np.tile(np.arange(len(triangles)), len(points))
    pairTriangles = triangles[triangleIndices]
    weights, distances = tria3ClosestPoints(
        points[pointIndices], pairTriangles[:, 0], pairTriangles[:, 1], pairTriangles[:, 2]
    )
    for k in range(len(pointIndices)):
        expectedWeights, expectedDistance = _tria3ClosestPointScalar(points[pointIndices[k]], *pairTriangles[k])
        _assertBitIdentical(weights[k], expectedWeights)
        _assertBitIdentical(distances[k], expectedDistance)


@pytest.mark.parametrize("seed", [0, 1])
def test_line2_batched_is_bit_identical_to_pairwise(seed):
    rng = np.random.default_rng(seed)
    n = 20000
    segments = rng.standard_normal((n, 2, 2)) * rng.uniform(1e-3, 1e3, (n, 1, 1))
    points = rng.standard_normal((n, 2)) * rng.uniform(1e-3, 3e3, (n, 1))
    segments[: n // 10, 1] = segments[: n // 10, 0]  # degenerate
    points[n // 10 : n // 5] = segments[n // 10 : n // 5, 1]  # at a node
    segments[n // 2 :] = np.round(segments[n // 2 :] / 100.0)
    points[n // 2 :] = np.round(points[n // 2 :] / 100.0)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        weights, distances = line2ClosestPoints(points, segments[:, 0], segments[:, 1])
        for i in range(n):
            expectedWeights, expectedDistance = _line2ClosestPointScalar(points[i], *segments[i])
            _assertBitIdentical(weights[i], expectedWeights)
            _assertBitIdentical(distances[i], expectedDistance)

            singleWeights, singleDistance = line2ClosestPoint(points[i], *segments[i])
            _assertBitIdentical(singleWeights, expectedWeights)
            _assertBitIdentical(singleDistance, expectedDistance)


def _scanForClosest(queryPoints, facetCoords, candidatesPerPoint, closestPoint):
    """The pair-by-pair selection: first strict minimum over each point's candidates."""

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
def test_closest_facets_matches_pairwise_scan(searchDistance):
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
    expected = _scanForClosest(queryPoints, facetCoords, candidatesPerPoint, _tria3ClosestPointScalar)
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
    expected = _scanForClosest(queryPoints, facetCoords, candidatesPerPoint, _line2ClosestPointScalar)
    assert closestFacet[0] == -1 and closestDistance[0] == np.inf
    for p, (bestFacet, bestWeights, bestDistance) in enumerate(expected[1:], start=1):
        assert closestFacet[p] == bestFacet
        _assertBitIdentical(closestWeights[p], bestWeights)
        _assertBitIdentical(closestDistance[p], bestDistance)
