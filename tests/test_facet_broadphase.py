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
"""The contact broadphase must be a pure accelerator: it may return fewer candidates than an
exhaustive facet sweep would evaluate, never a different winner.

That property is what lets the constraints call it instead of sweeping, and it is not obvious --
the candidate set is pruned twice, once by a ball around the query point and once by each facet's
own radius, and both bounds are exact only in real arithmetic. These tests pin the selection
against a literal exhaustive sweep, on the facet-size spread that motivated the second prune.
"""

import numpy as np
import pytest

from edelweissfe.utils.facetcontactgeometry import (
    closestFacetCandidates,
    tria3ClosestPoint,
)


def _selectExhaustively(queryPoints: np.ndarray, facetCoords: list, searchDistance=None) -> list:
    """The selection the constraints would make without any broadphase at all: every facet
    evaluated, first strict minimum wins."""

    selected = []
    for point in queryPoints:
        bestDistance, bestFacet = np.inf, None
        for i, facet in enumerate(facetCoords):
            _, distance = tria3ClosestPoint(point, *facet)
            if distance < bestDistance:
                bestDistance, bestFacet = distance, i
        if bestFacet is not None and (searchDistance is None or bestDistance <= searchDistance):
            selected.append(bestFacet)
        else:
            selected.append(None)
    return selected


def _selectViaBroadphase(queryPoints: np.ndarray, facetCoords: list, searchDistance=None) -> list:
    """The same selection, restricted to the broadphase's candidates -- i.e. what the constraints
    actually do."""

    candidatesPerPoint = closestFacetCandidates(queryPoints, facetCoords, searchDistance)

    selected = []
    for p, point in enumerate(queryPoints):
        bestDistance, bestFacet = np.inf, None
        for i in candidatesPerPoint[p]:
            _, distance = tria3ClosestPoint(point, *facetCoords[i])
            if distance < bestDistance:
                bestDistance, bestFacet = distance, i
        if bestFacet is not None and (searchDistance is None or bestDistance <= searchDistance):
            selected.append(bestFacet)
        else:
            selected.append(None)
    return selected


def _gradedSurface(fineCells: int, coarseCells: int, fineSize: float, coarseSize: float) -> list:
    """A z = 0 surface of triangles whose size jumps by a large factor across the patch.

    This is the refined-contact-surface case: an h-adaptive run subdivides the loaded region and
    leaves the far field alone, so one surface carries facets differing by more than an order of
    magnitude. A broadphase bounded by the *largest* facet radius degrades to a near-exhaustive
    sweep exactly here.
    """

    facets = []
    for cells, size, x0 in ((fineCells, fineSize, 0.0), (coarseCells, coarseSize, fineCells * fineSize)):
        for i in range(cells):
            for j in range(cells):
                x, y = x0 + i * size, j * size
                a = np.array([x, y, 0.0])
                b = np.array([x + size, y, 0.0])
                c = np.array([x + size, y + size, 0.0])
                d = np.array([x, y + size, 0.0])
                facets.append([a, b, c])
                facets.append([a, c, d])
    return facets


@pytest.fixture
def gradedSurface():
    return _gradedSurface(fineCells=8, coarseCells=4, fineSize=1.0, coarseSize=8.0)


def test_selection_matches_exhaustive_sweep_on_a_graded_surface(gradedSurface):
    """The pruned candidate set picks the facet the exhaustive sweep picks, for every point."""

    rng = np.random.default_rng(20260906)
    queryPoints = np.column_stack(
        (
            rng.uniform(-2.0, 42.0, 400),
            rng.uniform(-2.0, 34.0, 400),
            rng.uniform(-3.0, 3.0, 400),
        )
    )

    assert _selectViaBroadphase(queryPoints, gradedSurface) == _selectExhaustively(queryPoints, gradedSurface)


def test_selection_matches_exhaustive_sweep_with_a_search_distance(gradedSurface):
    """Capping the query radius must not change the outcome either -- including for the points the
    cut-off leaves unassigned."""

    rng = np.random.default_rng(20260907)
    queryPoints = np.column_stack(
        (
            rng.uniform(-6.0, 46.0, 300),
            rng.uniform(-6.0, 38.0, 300),
            rng.uniform(-8.0, 8.0, 300),
        )
    )

    viaBroadphase = _selectViaBroadphase(queryPoints, gradedSurface, searchDistance=2.5)
    exhaustively = _selectExhaustively(queryPoints, gradedSurface, searchDistance=2.5)

    assert viaBroadphase == exhaustively
    assert any(facet is None for facet in exhaustively), "the cut-off never rejected anything -- test is vacuous"


def test_a_tie_keeps_every_tied_facet(gradedSurface):
    """A point equidistant from several facets is the case the ulp margins exist for: the candidate
    set must still contain all of them, so the caller's ascending-index tie-break is what decides."""

    # Directly above a shared vertex of the fine patch: the four triangles meeting there are all at
    # exactly the same distance.
    queryPoints = np.array([[4.0, 4.0, 1.0]])

    candidates = closestFacetCandidates(queryPoints, gradedSurface, None)[0]

    distances = np.array([tria3ClosestPoint(queryPoints[0], *gradedSurface[i])[1] for i in candidates])
    tied = [
        i for i, facet in enumerate(gradedSurface) if abs(tria3ClosestPoint(queryPoints[0], *facet)[1] - 1.0) < 1e-12
    ]

    assert len(tied) > 1, "fixture does not actually produce a tie -- test is vacuous"
    assert set(tied) <= set(candidates)
    assert distances.min() == pytest.approx(1.0)


def test_the_prune_actually_prunes(gradedSurface):
    """The point of the per-facet prune is that it removes most of what the ball admits. Without
    this the exactness tests above would still pass on a broadphase that returned everything."""

    rng = np.random.default_rng(20260908)
    queryPoints = np.column_stack(
        (
            rng.uniform(0.0, 8.0, 100),
            rng.uniform(0.0, 8.0, 100),
            rng.uniform(-1.0, 1.0, 100),
        )
    )

    candidates = closestFacetCandidates(queryPoints, gradedSurface, None)
    meanCandidates = np.mean([len(c) for c in candidates])

    # Over the fine patch, whose facets are 8x smaller than the coarse ones sizing the ball.
    assert meanCandidates < 0.25 * len(gradedSurface)


def test_an_empty_surface_yields_no_candidates():
    assert closestFacetCandidates(np.zeros((3, 3)), [], None) == [[], [], []]
