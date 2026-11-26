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
flanks differ by more than a right angle, which the test shows as well; so it does at the tip of a finely faceted cone,
where no two neighbouring facets meet at a crease. On a surface that ends, the facet's plane goes on beyond its
boundary, and so does a crease.

The patch of a facet and the convex creases that a contact point must not slide over are checked on the same tooth.
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


def _toothPrismSurface(closed: bool = True, slopeOfTheEnds: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    """The tooth extruded through z = 0..1 as Tria3 facets, outward normals: (vertex coordinates, vertices of each
    facet), the bottom vertices first; the facets of each flank first, two per flank, then those of the two ends.

    Parameters
    ----------
    closed
        With the two ends; without them, the surface ends at z = 0 and z = 1.
    slopeOfTheEnds
        The ends lie in the planes z = slopeOfTheEnds * x and z = 1 + slopeOfTheEnds * x: cut obliquely, they meet the
        two flanks of the crest at different angles.
    """

    n = len(_toothPolygon)
    bottom = np.column_stack((_toothPolygon, slopeOfTheEnds * _toothPolygon[:, 0]))
    vertices = np.vstack((bottom, bottom + [0.0, 0.0, 1.0]))
    facets = []
    for i in range(n):
        j = (i + 1) % n
        facets += [[i, j, j + n], [i, j + n, i + n]]
    if closed:
        for a, b, c in _toothTriangles:
            facets += [[a + n, b + n, c + n], [a, c, b]]
    return vertices, np.array(facets)


def _pyramidSurface(nSides: int = 12, halfAngle: float = 20.0) -> tuple[np.ndarray, np.ndarray]:
    """A closed pyramid of height 1 over a regular polygon, a finely faceted cone, as Tria3 facets, outward normals:
    (vertex coordinates, vertices of each facet), the apex after the corners of the base, its centre last."""

    radius = np.tan(np.deg2rad(halfAngle))
    angles = 2.0 * np.pi * np.arange(nSides) / nSides
    base = np.column_stack((radius * np.cos(angles), radius * np.sin(angles), np.zeros(nSides)))
    vertices = np.vstack((base, [[0.0, 0.0, 1.0], [0.0, 0.0, 0.0]]))
    apex, centre = nSides, nSides + 1
    sides = [[i, (i + 1) % nSides, apex] for i in range(nSides)]
    bottom = [[(i + 1) % nSides, i, centre] for i in range(nSides)]
    return vertices, np.array(sides + bottom)


def _insideThePyramid(points: np.ndarray, nSides: int = 12, halfAngle: float = 20.0) -> np.ndarray:
    """Whether points lie inside :func:`_pyramidSurface`: above its base, below its apex, and at each height inside the
    polygon of that height, whose inner radius shrinks linearly to zero at the apex."""

    x, y, z = points.T
    sector = 2.0 * np.pi / nSides
    angleFromTheMiddleOfASide = np.mod(np.arctan2(y, x), sector) - 0.5 * sector
    innerRadius = np.tan(np.deg2rad(halfAngle)) * np.cos(0.5 * sector) * (1.0 - z)
    return (z > 0.0) & (z < 1.0) & (np.hypot(x, y) * np.cos(angleFromTheMiddleOfASide) < innerRadius)


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
    gap, dGap_dPoint = surface.gapsAndGradients(
        points, closestFacet, weights, coordinatesOfFacet, normalOfFacet, features
    )

    clearOfTheSurface = distance > 1e-9
    np.testing.assert_array_equal((gap < 0.0)[clearOfTheSurface], inside[clearOfTheSurface])
    np.testing.assert_allclose(np.abs(gap), distance, rtol=1e-12, atol=1e-15)
    np.testing.assert_allclose(np.linalg.norm(dGap_dPoint, axis=1), 1.0, rtol=1e-12)
    # The gap is linear in the point along its gradient: the gradient times the vector from the closest point is the
    # gap.
    closestPoints = np.einsum("pa,pad->pd", weights, coordinatesOfFacet[closestFacet])
    np.testing.assert_allclose(rowDot(dGap_dPoint, points - closestPoints), gap, rtol=1e-12, atol=1e-15)

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
    gap, dGap_dPoint = surface.gapsAndGradients(
        points, closestFacet, weights, coordinatesOfFacet, normalOfFacet, features
    )

    np.testing.assert_allclose(gap, [-0.1, 0.1, -0.2, 0.3, -0.3], rtol=1e-14)
    np.testing.assert_allclose(dGap_dPoint, np.tile([0.0, 1.0], (5, 1)), atol=1e-15)


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
    gap, dGap_dPoint = surface.gapsAndGradients(
        point, np.array([0]), np.array([[1.0, 0.0]]), coordinatesOfFacet, normalOfFacet, features
    )
    np.testing.assert_allclose(gap, [0.5], rtol=1e-14)
    np.testing.assert_allclose(dGap_dPoint, [[0.6, 0.8]], rtol=1e-14)


def test_on_a_feature_the_pseudonormal_gives_the_gradient():
    """A point on a vertex has no direction of its own: the gap is zero, its gradient the vertex's pseudonormal."""

    vertices, nodesOfFacet = _toothPolygonSurface()
    coordinatesOfFacet = vertices[nodesOfFacet]
    normalOfFacet, _ = facetNormalAndMeasure(coordinatesOfFacet)
    surface = FacetedSurface(nodesOfFacet)
    features = surface.features(coordinatesOfFacet, normalOfFacet)

    # The root (2, 1) is vertex 3, the end of facet 2 (the top of the core, normal +y) and the start of facet 3 (the
    # flank, normal (2, 0.5) / |(2, 0.5)|).
    gap, dGap_dPoint = surface.gapsAndGradients(
        np.array([[2.0, 1.0]]), np.array([2]), np.array([[0.0, 1.0]]), coordinatesOfFacet, normalOfFacet, features
    )
    np.testing.assert_array_equal(gap, [0.0])
    pseudonormal = np.array([0.0, 1.0]) + np.array([2.0, 0.5]) / np.hypot(2.0, 0.5)
    np.testing.assert_allclose(dGap_dPoint, [pseudonormal / np.linalg.norm(pseudonormal)], rtol=1e-14)


def test_at_the_tip_of_a_finely_faceted_cone_the_vertex_is_a_crease():
    """At the apex of a 12-sided pyramid of 20 degrees half angle, neighbouring facets differ by less than the limit, so
    that no edge is a crease, but opposite facets differ by 140 degrees. Next to the apex, every facet reaches it at the
    same distance, and the closest facet is whichever comes first: its normal alone gets the side wrong for points on
    the other side. The apex is a crease, and its pseudonormal tells the side."""

    vertices, nodesOfFacet = _pyramidSurface()
    apex = 12
    coordinatesOfFacet = vertices[nodesOfFacet]
    normalOfFacet, _ = facetNormalAndMeasure(coordinatesOfFacet)
    surface = FacetedSurface(nodesOfFacet)
    features = surface.features(coordinatesOfFacet, normalOfFacet)
    edgesAtTheApex = np.any(surface.verticesOfEdge == apex, axis=1)
    assert not np.any(features.edgeIsCrease[edgesAtTheApex])
    assert features.vertexIsCrease[apex]

    points = vertices[apex] + np.random.default_rng(5).uniform(-0.3, 0.3, size=(4000, 3))
    inside = _insideThePyramid(points)
    allFacets = list(range(len(nodesOfFacet)))
    closestFacet, weights, distance = closestFacets(points, list(coordinatesOfFacet), [allFacets] * len(points))
    gap, _ = surface.gapsAndGradients(points, closestFacet, weights, coordinatesOfFacet, normalOfFacet, features)

    clearOfTheSurface = distance > 1e-9
    np.testing.assert_array_equal((gap < 0.0)[clearOfTheSurface], inside[clearOfTheSurface])
    closestPoints = np.einsum("pa,pad->pd", weights, coordinatesOfFacet[closestFacet])
    facetNormalSaysInside = rowDot(normalOfFacet[closestFacet], points - closestPoints) < 0.0
    assert np.any((facetNormalSaysInside != inside)[clearOfTheSurface]), "the facet normal alone should fail here"


def test_a_crease_goes_on_beyond_the_boundary_of_a_surface_that_ends():
    """The tooth prism without its ends: beyond them, the flanks go on in their planes and the creases along their
    edges, so that a point beyond an end has the side of the tooth it would have if the prism went on. A vertex where a
    crease ends has no complete ring of facets; it takes the pseudonormal of the crease edge, which does not depend on
    the angles of the facets at the vertex -- these differ where the end cuts the crest obliquely."""

    vertices, nodesOfFacet = _toothPrismSurface(closed=False)
    coordinatesOfFacet = vertices[nodesOfFacet]
    normalOfFacet, _ = facetNormalAndMeasure(coordinatesOfFacet)
    surface = FacetedSurface(nodesOfFacet)
    features = surface.features(coordinatesOfFacet, normalOfFacet)

    rng = np.random.default_rng(3)
    points = np.empty((6000, 3))
    for k, corner in enumerate(((1.5, 3.0), (2.0, 1.0), (1.0, 1.0))):
        points[2000 * k : 2000 * (k + 1), :2] = corner + rng.uniform(-0.4, 0.4, size=(2000, 2))
    beyond = rng.uniform(0.0, 0.4, size=len(points))
    points[:, 2] = np.where(np.arange(len(points)) % 2 == 0, -beyond, 1.0 + beyond)
    inside = _insideTheTooth(points[:, :2])
    allFacets = list(range(len(nodesOfFacet)))
    closestFacet, weights, distance = closestFacets(points, list(coordinatesOfFacet), [allFacets] * len(points))
    gap, _ = surface.gapsAndGradients(points, closestFacet, weights, coordinatesOfFacet, normalOfFacet, features)
    clearOfTheSurface = distance > 1e-9
    np.testing.assert_array_equal((gap < 0.0)[clearOfTheSurface], inside[clearOfTheSurface])

    # Cut obliquely, the crest (vertices 4 and 11) meets each end at different angles on its two flanks.
    vertices, nodesOfFacet = _toothPrismSurface(closed=False, slopeOfTheEnds=2.5)
    coordinatesOfFacet = vertices[nodesOfFacet]
    normalOfFacet, _ = facetNormalAndMeasure(coordinatesOfFacet)
    surface = FacetedSurface(nodesOfFacet)
    features = surface.features(coordinatesOfFacet, normalOfFacet)
    crestEdge = np.flatnonzero(np.all(surface.verticesOfEdge == [4, 11], axis=1))[0]
    assert features.edgeIsCrease[crestEdge]
    for end in (4, 11):
        assert features.vertexIsCrease[end]
        np.testing.assert_array_equal(features.pseudonormalOfVertex[end], features.pseudonormalOfEdge[crestEdge])


def test_a_patch_is_the_ring_of_facets_around_a_facet():
    """The patch of a facet holds every facet that shares a vertex with it, itself included."""

    for vertices, nodesOfFacet in (_toothPolygonSurface(), _toothPrismSurface()):
        surface = FacetedSurface(nodesOfFacet)
        for facet, itsVertices in enumerate(nodesOfFacet):
            sharingAVertex = [other for other, theirs in enumerate(nodesOfFacet) if set(theirs) & set(itsVertices)]
            np.testing.assert_array_equal(surface.patchOfFacet(facet), sharingAVertex)

    _, nodesOfFacet = _toothPolygonSurface()
    np.testing.assert_array_equal(FacetedSurface(nodesOfFacet).patchOfFacet(3), [2, 3, 4])  # a flank of the crest


@pytest.mark.parametrize("surface", (_toothPolygonSurface, _toothPrismSurface))
def test_going_over_the_crest_crosses_a_convex_crease_and_over_a_root_not(surface):
    """From one flank of the crest to the other, either way, the surface turns away: a convex crease. From the top of
    the core onto a flank, it turns towards: the root, a re-entrant crease. Within one flank, it does not turn."""

    vertices, nodesOfFacet = surface()
    coordinatesOfFacet = vertices[nodesOfFacet]
    normalOfFacet, _ = facetNormalAndMeasure(coordinatesOfFacet)
    # The facets of the top of the core next to the root (2, 1), and of the two flanks of the crest.
    if vertices.shape[1] == 2:
        topOfTheCore, flankA, flankB, alsoFlankA = 2, 3, 4, 3
    else:
        topOfTheCore, flankA, flankB, alsoFlankA = 4, 6, 8, 7

    across = FacetedSurface(nodesOfFacet).acrossAConvexCrease(
        np.array([flankA, flankB, topOfTheCore, flankA]),
        np.array([flankB, flankA, flankA, alsoFlankA]),
        coordinatesOfFacet,
        normalOfFacet,
    )
    np.testing.assert_array_equal(across, [True, True, False, False])


def _windingNumbers(points: np.ndarray, coordinatesOfFacet: np.ndarray) -> np.ndarray:
    """The generalized winding numbers of points with respect to a closed surface: the sum of the signed angles of its
    segments (2D), or of the signed solid angles of its triangles (van Oosterom and Strackee, 3D), over a full turn."""

    toVertices = coordinatesOfFacet[None, :, :, :] - points[:, None, None, :]
    if points.shape[1] == 2:
        a, b = toVertices[:, :, 0], toVertices[:, :, 1]
        angles = np.arctan2(a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0], np.einsum("pfi,pfi->pf", a, b))
        return angles.sum(axis=1) / (2.0 * np.pi)
    a, b, c = toVertices[:, :, 0], toVertices[:, :, 1], toVertices[:, :, 2]
    la, lb, lc = (np.linalg.norm(v, axis=-1) for v in (a, b, c))
    dot = lambda u, v: np.einsum("pfi,pfi->pf", u, v)  # noqa: E731
    halfSolidAngles = np.arctan2(
        dot(a, np.cross(b, c)), la * lb * lc + dot(a, b) * lc + dot(a, c) * lb + dot(b, c) * la
    )
    return halfSolidAngles.sum(axis=1) / (2.0 * np.pi)


def _closestPointsByBruteForce(points: np.ndarray, coordinatesOfFacet: np.ndarray):
    """The closest facet of each point, the weights of its closest point (exact zeros off the feature it lies on) and
    the distance, from all facets: the projection onto the plane of a facet where it falls inside, otherwise the
    closest point of the facet's edges -- not the region test of the module."""

    nPoints, (nFacets, nVerticesPerFacet, _) = len(points), coordinatesOfFacet.shape
    distances = np.full((nPoints, nFacets), np.inf)
    weights = np.zeros((nPoints, nFacets, nVerticesPerFacet))
    for k in range(nVerticesPerFacet):  # the edges (the segment itself in 2D)
        m = (k + 1) % nVerticesPerFacet
        a, b = coordinatesOfFacet[None, :, k], coordinatesOfFacet[None, :, m]
        t = np.clip(np.einsum("pfi,pfi->pf", points[:, None] - a, b - a) / np.einsum("pfi,pfi->pf", b - a, b - a), 0, 1)
        d = np.linalg.norm(points[:, None] - (a + t[..., None] * (b - a)), axis=-1)
        closer = d < distances
        distances = np.where(closer, d, distances)
        onThisEdge = np.zeros_like(weights)
        onThisEdge[..., k], onThisEdge[..., m] = 1.0 - t, t
        weights = np.where(closer[..., None], onThisEdge, weights)
        if nVerticesPerFacet == 2:
            break
    if nVerticesPerFacet == 3:
        x1, x2, x3 = (coordinatesOfFacet[None, :, k] for k in range(3))
        normal = np.cross(x2 - x1, x3 - x1)
        normal /= np.linalg.norm(normal, axis=-1)[..., None]
        height = np.einsum("pfi,pfi->pf", points[:, None] - x1, normal)
        projected = points[:, None] - height[..., None] * normal
        subAreas = np.stack(
            [
                np.einsum("pfi,pfi->pf", np.cross(xb - xa, projected - xa), normal)
                for xa, xb in ((x2, x3), (x3, x1), (x1, x2))
            ],
            axis=-1,
        )
        inside = np.all(subAreas >= 0.0, axis=-1)
        distances = np.where(inside, np.abs(height), distances)
        weights = np.where(inside[..., None], subAreas / subAreas.sum(axis=-1)[..., None], weights)
    closestFacet = np.argmin(distances, axis=1)
    everyPoint = np.arange(nPoints)
    return closestFacet, weights[everyPoint, closestFacet], distances[everyPoint, closestFacet]


def _revolvedThread(nAround: int = 16) -> tuple[np.ndarray, np.ndarray]:
    """A screw-like solid: three saw teeth of 40 degrees on a core, revolved about z in nAround steps, closed at both
    ends, as Tria3 facets with outward normals."""

    profile = [(0.0, 0.0), (1.0, 0.0)]
    for k in range(3):
        profile += [
            (1.0, k + 0.5 - 0.6 * np.tan(np.deg2rad(20.0))),
            (1.6, k + 0.5),
            (1.0, k + 0.5 + 0.6 * np.tan(np.deg2rad(20.0))),
        ]
    profile += [(1.0, 3.0), (0.0, 3.0)]
    angles = 2.0 * np.pi * np.arange(nAround) / nAround
    vertices, indexOf = [], {}
    for j, (r, z) in enumerate(profile):
        for k in [0] if r == 0.0 else range(nAround):
            indexOf[j, k] = len(vertices)
            vertices.append((r * np.cos(angles[k]), r * np.sin(angles[k]), z))
    vertexAt = lambda j, k: indexOf[j, 0] if profile[j][0] == 0.0 else indexOf[j, k % nAround]  # noqa: E731
    facets = []
    for j in range(len(profile) - 1):
        for k in range(nAround):
            a, b, c, d = vertexAt(j, k), vertexAt(j, k + 1), vertexAt(j + 1, k + 1), vertexAt(j + 1, k)
            facets += [f for f in ((a, b, c), (a, c, d)) if len(set(f)) == 3]
    return np.array(vertices), np.array(facets)


def _randomStarPolyhedron(rng, nDirections: int = 60) -> tuple[np.ndarray, np.ndarray]:
    """A random star-shaped polyhedron: random directions, triangulated by their convex hull, at random radii -- convex
    and re-entrant creases, saddles; radial scaling keeps the hull's outward orientation."""

    from scipy.spatial import ConvexHull

    directions = rng.normal(size=(nDirections, 3))
    directions /= np.linalg.norm(directions, axis=1)[:, None]
    facets = ConvexHull(directions).simplices
    x1, x2, x3 = (directions[facets[:, k]] for k in range(3))
    outward = rowDot(np.cross(x2 - x1, x3 - x1), x1) > 0.0
    facets = np.where(outward[:, None], facets, facets[:, ::-1])
    return directions * rng.uniform(0.4, 1.2, size=(nDirections, 1)), facets


@pytest.mark.parametrize(
    "surface",
    (
        lambda: (np.array([[0, 0], [1, 0], [1.0, 0.39], [1.6, 0.5], [1.0, 0.61], [1.0, 1.39], [1.6, 1.5], [1.0, 1.61],
                           [1, 2], [0, 2]], dtype=float), np.array([[i, (i + 1) % 10] for i in range(10)])),  # fmt: skip
        _revolvedThread,
        lambda: _pyramidSurface(nSides=64, halfAngle=5.0),
        lambda: _randomStarPolyhedron(np.random.default_rng(13)),
    ),
    ids=("saw-tooth profile", "revolved saw-tooth thread", "needle cone tip", "random star polyhedron"),
)
def test_the_gap_agrees_with_a_winding_number_and_a_brute_force_distance(surface):
    """Against oracles that share nothing with the module: the sign of the gap is that of the generalized winding
    number of the closed surface, and its magnitude is the distance to the closest facet by brute force, exactly where
    the closest point lies inside a facet or on a crease; elsewhere -- a seam between facets less than 30 degrees apart
    -- the distance along the facet's normal, at least cos(30 degrees) of it. The points lie near random vertices, at
    distances from 1e-6 to 1 of the size of their facets, and in a box around the surface."""

    vertices, nodesOfFacet = surface()
    nDim = vertices.shape[1]
    coordinatesOfFacet = vertices[nodesOfFacet]
    normalOfFacet, measure = facetNormalAndMeasure(coordinatesOfFacet)
    size = np.median(measure ** (1.0 / (nDim - 1)))

    rng = np.random.default_rng(355)
    nearVertices = vertices[rng.integers(0, len(vertices), 1500)]
    directions = rng.normal(size=(1500, nDim))
    directions /= np.linalg.norm(directions, axis=1)[:, None]
    distances = size * np.exp(rng.uniform(np.log(1e-6), 0.0, size=(1500, 1)))
    low, high = vertices.min(axis=0), vertices.max(axis=0)
    points = np.vstack(
        (
            nearVertices + distances * directions,
            rng.uniform(low - 0.2 * (high - low), high + 0.2 * (high - low), (500, nDim)),
        )
    )

    inside = np.abs(_windingNumbers(points, coordinatesOfFacet)) > 0.5
    closestFacet, weights, distance = _closestPointsByBruteForce(points, coordinatesOfFacet)
    surfaceOfFacets = FacetedSurface(nodesOfFacet)
    features = surfaceOfFacets.features(coordinatesOfFacet, normalOfFacet)
    gap, _ = surfaceOfFacets.gapsAndGradients(
        points, closestFacet, weights, coordinatesOfFacet, normalOfFacet, features
    )

    resolved = distance > 1e-8 * size
    np.testing.assert_array_equal((gap < 0.0)[resolved], inside[resolved])

    # The feature of the closest point, from the exact zeros of the brute force: a vertex, an edge (3D), or the facet.
    onTheFeature = weights != 0.0
    atAVertex = onTheFeature.sum(axis=1) == 1
    vertex = nodesOfFacet[closestFacet, np.argmax(onTheFeature, axis=1)]
    onACrease = atAVertex & features.vertexIsCrease[vertex]
    inTheFacet = onTheFeature.all(axis=1)
    if nDim == 3:
        onAnEdge = onTheFeature.sum(axis=1) == 2
        spanning = np.sort(np.where(onTheFeature, nodesOfFacet[closestFacet], -1), axis=1)[:, 1:]
        edgeIndex = {tuple(e): i for i, e in enumerate(surfaceOfFacets.verticesOfEdge.tolist())}
        edge = np.array([edgeIndex.get(tuple(e), 0) for e in spanning.tolist()])
        onACrease |= onAnEdge & features.edgeIsCrease[edge]
    exact = resolved & (onACrease | inTheFacet)
    roundOff = 1e-13 * size
    np.testing.assert_allclose(np.abs(gap[exact]), distance[exact], rtol=1e-9, atol=roundOff)
    assert np.all(np.abs(gap[resolved]) <= distance[resolved] * (1.0 + 1e-9) + roundOff)
    assert np.all(np.abs(gap[resolved]) >= np.cos(np.deg2rad(30.0)) * distance[resolved] * (1.0 - 1e-9) - roundOff)


def test_a_boundary_vertex_where_two_creases_meet_stays_a_crease():
    """On an open surface where two crease edges meet at a boundary vertex (an open 3-face box corner), the vertex is
    still marked as a crease and takes the sum of its incident crease edges' pseudonormals, giving the true signed
    distance outside and inside the corner."""

    vertices = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 1.0, -1.0],
            [0.0, 0.0, -1.0],
        ]
    )
    # Three orthogonal faces meeting at vertex 0: normals +z, -x, -y. Two 90-degree crease edges meet at vertex 0,
    # while the two edges to vertices 3 and 4 are boundary edges.
    nodesOfFacet = np.array([[0, 1, 2], [0, 2, 3], [0, 4, 1]])
    coordinatesOfFacet = vertices[nodesOfFacet]
    normalOfFacet, _ = facetNormalAndMeasure(coordinatesOfFacet)
    surface = FacetedSurface(nodesOfFacet)
    features = surface.features(coordinatesOfFacet, normalOfFacet)

    assert not surface.vertexIsInterior[0]
    assert features.vertexIsCrease[0]
    np.testing.assert_allclose(features.pseudonormalOfVertex[0], [-1.0, -1.0, 2.0], atol=1e-14)

    # Outside the corner: the true distance to vertex 0 is sqrt(0.75) (not the facet-normal projection 0.5).
    outsidePoint = np.array([[-0.5, -0.5, 0.5]])
    closestFacet, weights, distance = closestFacets(outsidePoint, list(coordinatesOfFacet), [[0, 1, 2]])
    gap, dGap_dPoint = surface.gapsAndGradients(
        outsidePoint, closestFacet, weights, coordinatesOfFacet, normalOfFacet, features
    )
    np.testing.assert_allclose(gap, distance, rtol=1e-14)
    np.testing.assert_allclose(dGap_dPoint, outsidePoint / distance[0], rtol=1e-14)

    # Inside the corner, evaluated directly at vertex 0: negative signed distance.
    insidePoint = np.array([[0.5, 0.5, -0.5]])
    gapIn, _ = surface.gapsAndGradients(
        insidePoint, np.array([0]), np.array([[1.0, 0.0, 0.0]]), coordinatesOfFacet, normalOfFacet, features
    )
    np.testing.assert_allclose(gapIn, [-np.linalg.norm(insidePoint)], rtol=1e-14)


def test_a_180_degree_fold_produces_no_nan_in_gaps_or_gradients():
    """A degenerate 180-degree fold whose opposing normals cancel to a zero pseudonormal is flagged as a crease in 2D
    and 3D without producing NaN in gapsAndGradients."""

    # 2D hairpin: two Line2 segments back-to-back with opposing normals (0, +1) and (0, -1).
    vertices2d = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 0.0]])
    nodes2d = np.array([[0, 1], [1, 2]])
    coords2d = vertices2d[nodes2d]
    normals2d, _ = facetNormalAndMeasure(coords2d)
    surf2d = FacetedSurface(nodes2d)
    feats2d = surf2d.features(coords2d, normals2d)
    assert feats2d.vertexIsCrease[1]
    gap2d, grad2d = surf2d.gapsAndGradients(
        np.array([[1.2, 0.1]]), np.array([0]), np.array([[0.0, 1.0]]), coords2d, normals2d, feats2d
    )
    assert np.all(np.isfinite(gap2d)) and np.all(np.isfinite(grad2d))

    # 3D zero-thickness fold: two Tria3 facets sharing edge (0, 1) with opposing normals +z and -z.
    vertices3d = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 1.0, 0.0]])
    nodes3d = np.array([[0, 1, 2], [1, 0, 3]])
    coords3d = vertices3d[nodes3d]
    normals3d, _ = facetNormalAndMeasure(coords3d)
    surf3d = FacetedSurface(nodes3d)
    feats3d = surf3d.features(coords3d, normals3d)
    gap3d, grad3d = surf3d.gapsAndGradients(
        np.array([[0.5, -0.2, 0.1]]), np.array([0]), np.array([[0.5, 0.5, 0.0]]), coords3d, normals3d, feats3d
    )
    assert np.all(np.isfinite(gap3d)) and np.all(np.isfinite(grad3d))
