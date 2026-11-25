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
Exact gap function, gradient, and full Hessian (including the second-derivative term arising
from the cross-product-then-normalize/rotate-then-normalize construction of the facet normal) for
a slave point against a flat contact facet (Tria3 in 3D, Line2 in 2D), expressed directly in terms
of the current nodal coordinates.

Both facets are exactly flat (a plane through 3 points, or a line through 2 points), so the
curvature/second-fundamental-form contribution present in curved-surface contact vanishes
identically -- the only surviving second-derivative term is a pose-dependent nonlinearity of the
normal's own construction from its defining nodes' positions, not a curvature effect.

The closed forms below were derived by hand and cross-verified against exact symbolic
differentiation (SymPy) at many random, non-degenerate configurations before being transcribed
here -- see the derivation/verification scripts referenced in the class docstring of
:mod:`~edelweissfe.constraints.nodetodeformablesurfacepenalty` for the underlying methodology.
Do not hand-edit these formulas without re-verifying them the same way; this kind of
normalize/rotate second-derivative algebra is very easy to get subtly wrong.
"""

import itertools
from typing import NamedTuple

import numpy as np
from scipy.sparse import csr_matrix
from scipy.spatial import cKDTree

from edelweissfe.utils.rotations import skewMatrix

#: The cosine of the largest angle between two normals that still belong to one smooth surface: two adjacent facets
#: whose normals differ more meet at a crease, and a contact point whose normal turns more between two contact searches
#: has slid over one. The node-based contact restarts the frictional history there.
cosOfSmoothNormalChangeLimit = np.cos(np.deg2rad(30.0))


class SurfaceFeatures(NamedTuple):
    """The edges and vertices of a faceted surface in one configuration: their angle-weighted pseudonormals (not
    normalized), and whether they lie on a crease. See :class:`FacetedSurface`."""

    #: The pseudonormal of each edge (3D; none in 2D), shape ``(nEdges, 3)``.
    pseudonormalOfEdge: np.ndarray
    #: The pseudonormal of each vertex, shape ``(nVertices, nDim)``; at a vertex where a crease ends on the boundary of
    #: the surface, that of the crease's last edge.
    pseudonormalOfVertex: np.ndarray
    #: Whether the two facets of an edge meet at a crease.
    edgeIsCrease: np.ndarray
    #: Whether a vertex lies on a crease: one of its edges is one, or one of its facets turns away from its
    #: pseudonormal by more than half the limit; on the boundary of the surface, only where a crease ends.
    vertexIsCrease: np.ndarray


class FacetedSurface:
    """The facets of a surface and where they meet: in 3D the edges and the vertices of its Tria3 facets, in 2D the
    vertices of its Line2 facets. This is what the gap to a surface with corners needs.

    The closest point of a facet to a point lies inside the facet, on one of its edges, or at one of its vertices --
    its *feature*. Inside the facet, the facet's normal tells inside from outside. At a corner it may not: next to an
    acute crest, a point outside the body can lie behind the plane of the flank whose edge is closest to it. There,
    the **angle-weighted pseudonormal** of the feature (Baerentzen and Aanaes, 2005) does: the sum of the normals of
    the facets that meet at an edge, or at a vertex, each weighted with the facet's angle at that vertex. For the
    closest feature of a closed surface, the distance vector :math:`\\boldsymbol{x} - \\boldsymbol{x}_c` has a positive
    component along the pseudonormal outside the body, a negative one inside, at convex and re-entrant corners alike.

    An edge is a **crease** where the normals of its two facets differ by more than
    :data:`cosOfSmoothNormalChangeLimit` allows (a crest, a root). A vertex is one where one of its edges is, or where
    one of its facets turns away from the vertex's pseudonormal by more than half that angle: two of its facets may
    then differ by more than the limit although no two neighbours do, as at the tip of a finely faceted cone. Where
    all its facets stay within half the limit of the pseudonormal, any two of them differ by at most the limit, and the
    normal of each tells the side. Elsewhere the facets tile a smooth surface, and its corners are artefacts of the
    tiling.

    A surface that is not closed has a **boundary**: the edges met by one facet only (in 3D), the vertices met by one
    segment (in 2D). A surface that ends does not say what lies beyond it, and the ring of facets around a vertex on
    its boundary is incomplete, so that the vertex's pseudonormal does not tell the side either. Beyond its boundary,
    the surface is taken to go on as it ends: in the plane of its last facet, and a crease that ends on the boundary
    goes on along its last edge -- that vertex takes the pseudonormal of the edge. No other feature on the boundary is
    a crease.

    Parameters
    ----------
    nodesOfFacet
        The vertices of each facet, as indices into the nodes of the surface, shape ``(nFacets, 3)`` (Tria3) or
        ``(nFacets, 2)`` (Line2); the facets oriented with their outward normals, as the surface element generator
        makes them.
    """

    #: For a Tria3 facet, the local vertices that span the edge opposite local vertex k: a closest point whose
    #: weight k is zero lies on that edge.
    _edgeOppositeLocalVertex = np.array([[1, 2], [0, 2], [0, 1]])

    def __init__(self, nodesOfFacet: np.ndarray):
        self.nodesOfFacet = np.asarray(nodesOfFacet, dtype=np.intp)
        nFacets, nVerticesPerFacet = self.nodesOfFacet.shape
        self.nVertices = int(self.nodesOfFacet.max()) + 1 if nFacets else 0

        if nVerticesPerFacet == 3:
            spanningVertices = np.sort(self.nodesOfFacet[:, self._edgeOppositeLocalVertex], axis=-1)
            edges, edgeOfFacet = np.unique(spanningVertices.reshape((-1, 2)), axis=0, return_inverse=True)
            #: The two vertices of each edge, shape ``(nEdges, 2)``.
            self.verticesOfEdge = edges.reshape((-1, 2))
            #: The edge opposite each local vertex of each facet, shape ``(nFacets, 3)``.
            self.edgeOppositeVertex = edgeOfFacet.reshape((nFacets, 3))
            #: Whether an edge is met by two facets (not on the boundary of the surface).
            self.edgeIsInterior = np.bincount(edgeOfFacet.ravel(), minlength=len(self.verticesOfEdge)) == 2
            onTheBoundary = np.zeros(self.nVertices, dtype=bool)
            onTheBoundary[self.verticesOfEdge[~self.edgeIsInterior].ravel()] = True
            #: Whether a vertex lies inside the surface: on no boundary edge (3D), met by two segments (2D).
            self.vertexIsInterior = ~onTheBoundary
        else:
            self.verticesOfEdge = np.zeros((0, 2), dtype=np.intp)
            self.edgeOppositeVertex = np.zeros((nFacets, 0), dtype=np.intp)
            self.edgeIsInterior = np.zeros(0, dtype=bool)
            self.vertexIsInterior = np.bincount(self.nodesOfFacet.ravel(), minlength=self.nVertices) == 2

        # The facets that share a vertex with a facet, itself included: one ring of facets around it, its patch.
        facetsAtVertex = csr_matrix(
            (
                np.ones(self.nodesOfFacet.size),
                self.nodesOfFacet.ravel(),
                np.arange(0, self.nodesOfFacet.size + 1, nVerticesPerFacet),
            ),
            shape=(nFacets, self.nVertices),
        )
        patches = (facetsAtVertex @ facetsAtVertex.T).tocsr()
        patches.sort_indices()
        self._patchStart, self._facetsOfPatches = patches.indptr, patches.indices

    def patchOfFacet(self, facet: int) -> np.ndarray:
        """The facets that share a vertex with a facet, the facet itself included -- one ring of facets around it, the
        facets a point on it slides onto first.

        Parameters
        ----------
        facet
            The facet.

        Returns
        -------
        numpy.ndarray
            The facets, in ascending order.
        """

        return self._facetsOfPatches[self._patchStart[facet] : self._patchStart[facet + 1]]

    def features(self, coordinatesOfFacet: np.ndarray, normalOfFacet: np.ndarray) -> SurfaceFeatures:
        """The pseudonormals of the edges and the vertices, and which of them lie on a crease, in the configuration
        given.

        The pseudonormal of an edge is the sum of the normals of its two facets, of a vertex the sum of the normals of
        its facets, each weighted with the facet's angle at the vertex; in 2D, the two segments at a vertex count
        alike. They are not normalized: only their direction is used. The normals of the two facets of an edge differ
        by the angle whose cosine is :math:`|\\boldsymbol{n}_A + \\boldsymbol{n}_B|^2 / 2 - 1`.

        Parameters
        ----------
        coordinatesOfFacet
            The node coordinates of each facet, shape ``(nFacets, nVerticesPerFacet, nDim)``.
        normalOfFacet
            The unit outward normal of each facet, shape ``(nFacets, nDim)``.

        Returns
        -------
        SurfaceFeatures
            The pseudonormals and the creases.
        """

        nFacets, nVerticesPerFacet, nDim = coordinatesOfFacet.shape
        pseudonormalOfEdge = np.zeros((len(self.edgeIsInterior), nDim))
        pseudonormalOfVertex = np.zeros((self.nVertices, nDim))
        vertexIsCrease = np.zeros(self.nVertices, dtype=bool)
        if nVerticesPerFacet == 3:
            np.add.at(pseudonormalOfEdge, self.edgeOppositeVertex, np.repeat(normalOfFacet[:, None, :], 3, axis=1))
            toNext = np.roll(coordinatesOfFacet, -1, axis=1) - coordinatesOfFacet
            toPrevious = np.roll(coordinatesOfFacet, 1, axis=1) - coordinatesOfFacet
            angleAtVertex = np.arctan2(
                np.linalg.norm(np.cross(toNext, toPrevious), axis=-1), np.einsum("fad,fad->fa", toNext, toPrevious)
            )
            np.add.at(pseudonormalOfVertex, self.nodesOfFacet, angleAtVertex[:, :, None] * normalOfFacet[:, None, :])
            edgeIsCrease = self.edgeIsInterior & (
                0.5 * rowDot(pseudonormalOfEdge, pseudonormalOfEdge) - 1.0 < cosOfSmoothNormalChangeLimit
            )
            vertexIsCrease[self.verticesOfEdge[edgeIsCrease].ravel()] = True
        else:
            np.add.at(pseudonormalOfVertex, self.nodesOfFacet, np.repeat(normalOfFacet[:, None, :], 2, axis=1))
            edgeIsCrease = np.zeros(0, dtype=bool)

        # A facet that turns away from the pseudonormal of its vertex by more than half the limit: its angle to the
        # pseudonormal has a cosine below that of half the limit, sqrt((1 + cos) / 2).
        cosOfHalfTheLimit = np.sqrt(0.5 * (1.0 + cosOfSmoothNormalChangeLimit))
        atVertexOfFacet = pseudonormalOfVertex[self.nodesOfFacet]
        lengthAtVertexOfFacet = np.sqrt(rowDot(atVertexOfFacet, atVertexOfFacet))
        facetTurnsAway = rowDot(normalOfFacet[:, None, :], atVertexOfFacet) < cosOfHalfTheLimit * lengthAtVertexOfFacet
        np.logical_or.at(vertexIsCrease, self.nodesOfFacet, facetTurnsAway)

        # On the boundary, the ring of facets around a vertex is incomplete, and so is its pseudonormal. Where a crease
        # ends on the boundary, the crease goes on beyond it: the vertex takes the pseudonormal of that crease edge.
        creaseEdges = np.flatnonzero(edgeIsCrease)
        nCreaseEdgesAtVertex = np.bincount(self.verticesOfEdge[creaseEdges].ravel(), minlength=self.nVertices)
        endsACrease = ~self.vertexIsInterior & (nCreaseEdgesAtVertex == 1)
        for end in (0, 1):
            endingHere = creaseEdges[endsACrease[self.verticesOfEdge[creaseEdges, end]]]
            pseudonormalOfVertex[self.verticesOfEdge[endingHere, end]] = pseudonormalOfEdge[endingHere]
        vertexIsCrease &= self.vertexIsInterior | endsACrease

        return SurfaceFeatures(pseudonormalOfEdge, pseudonormalOfVertex, edgeIsCrease, vertexIsCrease)

    def gapsAndGradients(
        self,
        points: np.ndarray,
        facets: np.ndarray,
        weights: np.ndarray,
        coordinatesOfFacet: np.ndarray,
        normalOfFacet: np.ndarray,
        features: SurfaceFeatures,
    ) -> tuple[np.ndarray, np.ndarray]:
        """The gaps of points to the closest points of their facets, negative inside the body, and their gradients
        with respect to the points.

        The gap is the signed distance to the surface where the closest point lies on a crease, and the distance along
        the facet normal elsewhere: the two agree inside a facet, and differ only by the small turn of the normal at a
        seam of a smooth surface.

        - The closest point lies on a **crease** of the surface, on an edge or at a vertex: the distance :math:`\\pm d`,
          the sign that of :math:`\\boldsymbol{x} - \\boldsymbol{x}_c` along the pseudonormal of the feature, and its
          gradient, the unit vector :math:`\\pm(\\boldsymbol{x} - \\boldsymbol{x}_c)/d` from the closest point to the
          point. Outside a convex crease (a crest) the gap so grows with the true distance to the edge in every
          direction, instead of with the distance to the plane of one of its facets, which is zero or negative for a
          point beside the edge; at a re-entrant crease (a root) a point inside is pushed out the shortest way. A point
          closer to the feature than :math:`\\sqrt{\\epsilon}` times the facet's size has no direction to working
          precision: the gradient is the unit pseudonormal.
        - Everywhere else -- inside the facet, on a seam of a smooth surface, on the boundary of a surface that ends
          where no crease ends -- the distance along the facet's normal, and the normal as its gradient: beyond its
          boundary, the surface goes on in the plane of its last facet.

        Parameters
        ----------
        points
            The points, shape ``(nPairs, nDim)``.
        facets
            The facet of each point, shape ``(nPairs,)``.
        weights
            The weights of the closest point of each point in its facet, as :func:`closestFacets` returns them: an
            exact zero marks a closest point on the facet's boundary.
        coordinatesOfFacet
            The node coordinates of all facets, shape ``(nFacets, nVerticesPerFacet, nDim)``.
        normalOfFacet
            The unit outward normals of all facets, shape ``(nFacets, nDim)``.
        features
            The pseudonormals and creases, see :meth:`features`.

        Returns
        -------
        tuple[numpy.ndarray, numpy.ndarray]
            The gaps, shape ``(nPairs,)``, and their gradients with respect to the points, ``dGap_dPoint``, unit
            vectors of shape ``(nPairs, nDim)``.
        """

        toPoint = points - np.einsum("pa,pad->pd", weights, coordinatesOfFacet[facets])
        dGap_dPoint = normalOfFacet[facets]
        gap = rowDot(dGap_dPoint, toPoint)

        onTheFacetBoundary = weights == 0.0
        nZeroWeights = onTheFacetBoundary.sum(axis=1)
        pseudonormal = np.zeros_like(dGap_dPoint)
        onACrease = np.zeros(len(facets), dtype=bool)

        atAVertex = nZeroWeights == weights.shape[1] - 1
        if atAVertex.any():
            vertex = self.nodesOfFacet[facets[atAVertex], np.argmin(onTheFacetBoundary[atAVertex], axis=1)]
            pseudonormal[atAVertex] = features.pseudonormalOfVertex[vertex]
            onACrease[atAVertex] = features.vertexIsCrease[vertex]

        onAnEdge = (nZeroWeights == 1) & ~atAVertex
        if onAnEdge.any():
            edge = self.edgeOppositeVertex[facets[onAnEdge], np.argmax(onTheFacetBoundary[onAnEdge], axis=1)]
            pseudonormal[onAnEdge] = features.pseudonormalOfEdge[edge]
            onACrease[onAnEdge] = features.edgeIsCrease[edge]

        creased = np.flatnonzero(onACrease)
        if len(creased):
            toFeature = toPoint[creased]
            distance = np.sqrt(rowDot(toFeature, toFeature))
            unitPseudonormal = pseudonormal[creased] / np.linalg.norm(pseudonormal[creased], axis=1)[:, None]
            sign = np.where(rowDot(unitPseudonormal, toFeature) < 0.0, -1.0, 1.0)

            _, measure = facetNormalAndMeasure(coordinatesOfFacet[facets[creased]])
            facetSize = measure ** (1.0 / (coordinatesOfFacet.shape[2] - 1))
            resolved = distance > np.sqrt(np.finfo(float).eps) * facetSize

            with np.errstate(divide="ignore", invalid="ignore"):
                dGap_dPoint[creased] = np.where(
                    resolved[:, None], (sign / distance)[:, None] * toFeature, unitPseudonormal
                )
            gap[creased] = np.where(resolved, sign * distance, rowDot(unitPseudonormal, toFeature))

        return gap, dGap_dPoint

    def acrossAConvexCrease(
        self, fromFacets: np.ndarray, toFacets: np.ndarray, coordinatesOfFacet: np.ndarray, normalOfFacet: np.ndarray
    ) -> np.ndarray:
        """Whether going from one facet to another crosses a convex crease of the surface: their normals differ by
        more than :data:`cosOfSmoothNormalChangeLimit` allows, and the second facet lies behind the plane of the
        first (its centroid does) -- the surface turns away, as from the top of a block to its side, or from one side
        of a thin body to the other. At a re-entrant crease, such as the root of a thread, the second facet lies in
        front of the plane of the first.

        Parameters
        ----------
        fromFacets, toFacets
            The facet pairs, shape ``(nPairs,)`` each.
        coordinatesOfFacet
            The node coordinates of all facets, shape ``(nFacets, nVerticesPerFacet, nDim)``.
        normalOfFacet
            The unit outward normals of all facets, shape ``(nFacets, nDim)``.

        Returns
        -------
        numpy.ndarray
            True for the pairs across a convex crease.
        """

        normalOfFrom = normalOfFacet[fromFacets]
        creased = rowDot(normalOfFacet[toFacets], normalOfFrom) < cosOfSmoothNormalChangeLimit
        centroidOfTo = coordinatesOfFacet[toFacets].mean(axis=1)
        behind = rowDot(normalOfFrom, centroidOfTo - coordinatesOfFacet[fromFacets, 0]) < 0.0
        return creased & behind


def tria3GapGradientHessian(
    xs: np.ndarray, x1: np.ndarray, x2: np.ndarray, x3: np.ndarray
) -> tuple[float, np.ndarray, np.ndarray]:
    """Exact gap, gradient, and Hessian of a slave point against a flat Tria3 facet (3D).

    The facet plane is spanned by ``x1, x2, x3`` (in this local order); the outward normal is
    ``cross(x2-x1, x3-x1)``, normalized. The gap is positive outside the facet's half-space,
    negative when penetrating.

    Parameters
    ----------
    xs, x1, x2, x3
        Current coordinates (each shape ``(3,)``) of the slave point and the facet's three nodes,
        in this fixed local order.

    Returns
    -------
    tuple[float, numpy.ndarray, numpy.ndarray]
        The gap ``g``, its gradient ``w`` (shape ``(12,)``, blocks ``[xs, x1, x2, x3]``), and its
        Hessian ``H`` (shape ``(12, 12)``, same block order).
    """

    r = xs - x1
    e1 = x2 - x1
    e2 = x3 - x1
    c = np.cross(e1, e2)
    m = np.linalg.norm(c)
    n = c / m
    g = n.dot(r)

    blocks = ("xs", "x1", "x2", "x3")

    dr_dBlock = {"xs": np.eye(3), "x1": -np.eye(3), "x2": np.zeros((3, 3)), "x3": np.zeros((3, 3))}
    dc_dBlock = {"xs": np.zeros((3, 3)), "x1": -skewMatrix(x2 - x3), "x2": -skewMatrix(e2), "x3": skewMatrix(e1)}

    projectorOntoTangentPlane = np.eye(3) - np.outer(n, n)
    dn_dBlock = {k: (projectorOntoTangentPlane @ dc_dBlock[k]) / m for k in blocks}
    dm_dBlock = {k: n @ dc_dBlock[k] for k in blocks}  # row vector

    w_dBlock = {k: dn_dBlock[k].T @ r + dr_dBlock[k].T @ n for k in blocks}
    w = np.concatenate([w_dBlock[k] for k in blocks])

    # The tangential (in-plane) component of r: normal-projected-out, used below since the
    # curvature-like second-derivative pieces only couple through r's in-plane part.
    rTangential = r - g * n

    # d(dc_dBlock[a])/d(block b) -- dc_dBlock[a] is +/- skew(u_a) for a fixed linear combination
    # u_a of the facet's nodes; this is the constant Jacobian du_a/d(block b) for each a.
    du_dBlock = {
        "x1": {"xs": np.zeros((3, 3)), "x1": np.zeros((3, 3)), "x2": -np.eye(3), "x3": np.eye(3)},  # u = x3-x2
        "x2": {"xs": np.zeros((3, 3)), "x1": -np.eye(3), "x2": np.zeros((3, 3)), "x3": np.eye(3)},  # u = x3-x1
        "x3": {"xs": np.zeros((3, 3)), "x1": -np.eye(3), "x2": np.eye(3), "x3": np.zeros((3, 3))},  # u = x2-x1
    }
    dcSign = {"x1": +1.0, "x2": -1.0, "x3": +1.0}  # dc_dBlock[a] = dcSign[a] * skew(u_a)

    H = np.zeros((12, 12))
    blockSlice = {"xs": slice(0, 3), "x1": slice(3, 6), "x2": slice(6, 9), "x3": slice(9, 12)}

    H[blockSlice["xs"], blockSlice["xs"]] = 0.0
    for b in ("x1", "x2", "x3"):
        H[blockSlice["xs"], blockSlice[b]] = dn_dBlock[b]

    for a in ("x1", "x2", "x3"):
        for b in blocks:
            # d(dn_dBlock[a])/d(block b), contracted with r on the normal's own index -- the
            # exact second derivative of the cross-product-then-normalize construction of n.
            crossNormalizeTerm = -(1.0 / m) * (
                np.outer(dm_dBlock[a], dn_dBlock[b].T @ r) + g * (dc_dBlock[a].T @ dn_dBlock[b])
            )
            skewArgumentTerm = dcSign[a] * (1.0 / m) * (skewMatrix(rTangential) @ du_dBlock[a][b])
            normalizeDenominatorTerm = -(1.0 / m**2) * np.outer(rTangential @ dc_dBlock[a], dm_dBlock[b])

            d2n_a_contractedWithR = crossNormalizeTerm + skewArgumentTerm + normalizeDenominatorTerm
            H[blockSlice[a], blockSlice[b]] = (
                d2n_a_contractedWithR + dn_dBlock[a].T @ dr_dBlock[b] + dr_dBlock[a].T @ dn_dBlock[b]
            )

    return g, w, H


def line2GapGradientHessian(xs: np.ndarray, x1: np.ndarray, x2: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    """Exact gap, gradient, and Hessian of a slave point against a flat Line2 facet (2D).

    The facet edge runs from ``x1`` to ``x2``; the outward normal is the edge direction rotated
    by -90 degrees, normalized. The gap is positive outside the facet's half-plane, negative when
    penetrating.

    Parameters
    ----------
    xs, x1, x2
        Current coordinates (each shape ``(2,)``) of the slave point and the facet's two nodes,
        in this fixed local order.

    Returns
    -------
    tuple[float, numpy.ndarray, numpy.ndarray]
        The gap ``g``, its gradient ``w`` (shape ``(6,)``, blocks ``[xs, x1, x2]``), and its
        Hessian ``H`` (shape ``(6, 6)``, same block order).
    """

    r = xs - x1
    e = x2 - x1
    length = np.linalg.norm(e)
    eHat = e / length
    rotateMinus90 = np.array([[0.0, 1.0], [-1.0, 0.0]])
    n = rotateMinus90 @ eHat
    g = n.dot(r)

    blocks = ("xs", "x1", "x2")

    dr_dBlock = {"xs": np.eye(2), "x1": -np.eye(2), "x2": np.zeros((2, 2))}
    de_dBlock = {"xs": np.zeros((2, 2)), "x1": -np.eye(2), "x2": np.eye(2)}

    projectorOntoNormal = np.eye(2) - np.outer(eHat, eHat)
    dEHat_dBlock = {k: (projectorOntoNormal @ de_dBlock[k]) / length for k in blocks}
    dn_dBlock = {k: rotateMinus90 @ dEHat_dBlock[k] for k in blocks}
    dLength_dBlock = {k: eHat @ de_dBlock[k] for k in blocks}  # row vector

    w_dBlock = {k: dn_dBlock[k].T @ r + dr_dBlock[k].T @ n for k in blocks}
    w = np.concatenate([w_dBlock[k] for k in blocks])

    # Since dn_dBlock = rotateMinus90 @ dEHat_dBlock, contracting r with n's index equals contracting
    # rotateMinus90^T @ r with eHat's index.
    rRotated = rotateMinus90.T @ r
    rRotatedDotEHat = rRotated.dot(eHat)

    H = np.zeros((6, 6))
    blockSlice = {"xs": slice(0, 2), "x1": slice(2, 4), "x2": slice(4, 6)}

    for a in blocks:
        for b in blocks:
            normalizeTerm = -(1.0 / length) * (
                np.outer(eHat @ de_dBlock[a], dEHat_dBlock[b].T @ rRotated)
                + rRotatedDotEHat * (de_dBlock[a].T @ dEHat_dBlock[b])
                + np.outer(dEHat_dBlock[a].T @ rRotated, dLength_dBlock[b])
            )
            H[blockSlice[a], blockSlice[b]] = (
                normalizeTerm + dn_dBlock[a].T @ dr_dBlock[b] + dr_dBlock[a].T @ dn_dBlock[b]
            )

    return g, w, H


def rowDot(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Row-wise dot product over the last axis, accumulated left to right onto +0.0 (so a sum of
    negative zeros is +0.0), with separate multiplications and additions. The order is fixed here,
    not left to a BLAS: ``np.dot`` may fuse multiply and add on one CPU and not on another, and then
    differs in the last bit."""

    dot = 0.0 + a[..., 0] * b[..., 0]
    for k in range(1, a.shape[-1]):
        dot = dot + a[..., k] * b[..., k]
    return dot


def tria3ClosestPoint(xs: np.ndarray, x1: np.ndarray, x2: np.ndarray, x3: np.ndarray) -> tuple[np.ndarray, float]:
    """Closest point on the (closed) triangle (x1, x2, x3) to xs, clamped to the triangle's
    interior/edge/vertex regions (Ericson's real-time-collision-detection region test), as
    barycentric weights (w1, w2, w3) with all w >= 0 and sum(w) == 1, plus the distance.

    The single-pair form of :func:`tria3ClosestPoints`, which it calls, so that the two can never
    disagree."""

    weights, distances = tria3ClosestPoints(xs[None], x1[None], x2[None], x3[None])
    return weights[0], float(distances[0])


def tria3ClosestPoints(xs: np.ndarray, x1: np.ndarray, x2: np.ndarray, x3: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Closest points of many (point, triangle) pairs at once, see :func:`tria3ClosestPoint`.

    Ericson's region test, evaluated for all pairs with array operations instead of one Python call
    per pair. Each pair is assigned the region whose condition, in the order the branch-by-branch
    test checks them, it satisfies first (a NaN, which satisfies no condition, falls through to the
    interior). Every operation acts on each pair alone, so a pair's result does not depend on the
    other pairs in the batch.

    Parameters
    ----------
    xs
        The points, shape ``(nPairs, 3)``.
    x1, x2, x3
        The nodes of each pair's triangle, in its local order, each of shape ``(nPairs, 3)``.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray]
        The barycentric weights of the closest points, shape ``(nPairs, 3)``, and the distances,
        shape ``(nPairs,)``.
    """

    e1 = x2 - x1
    e2 = x3 - x1
    r1 = xs - x1
    r2 = xs - x2
    r3 = xs - x3

    d1 = rowDot(e1, r1)
    d2 = rowDot(e2, r1)
    d3 = rowDot(e1, r2)
    d4 = rowDot(e2, r2)
    d5 = rowDot(e1, r3)
    d6 = rowDot(e2, r3)
    vc = d1 * d4 - d3 * d2
    va = d3 * d6 - d5 * d4
    vb = d5 * d2 - d1 * d6

    weights = np.zeros((len(d1), 3))
    unassigned = np.ones(len(d1), dtype=bool)

    def claim(condition: np.ndarray) -> np.ndarray:
        """The still unassigned pairs that satisfy ``condition``, which are assigned hereby."""
        claimed = unassigned & condition
        unassigned[claimed] = False
        return claimed

    # The regions in the order the scalar test checks them: a pair belongs to the first it satisfies.
    weights[claim((d1 <= 0.0) & (d2 <= 0.0)), 0] = 1.0  # vertex x1
    weights[claim((d3 >= 0.0) & (d4 <= d3)), 1] = 1.0  # vertex x2
    weights[claim((d6 >= 0.0) & (d5 <= d6)), 2] = 1.0  # vertex x3

    # A degenerate triangle divides by zero; as in a pairwise evaluation, the result is inf or NaN.
    with np.errstate(divide="ignore", invalid="ignore"):
        edge = claim((vc <= 0.0) & (d1 >= 0.0) & (d3 <= 0.0))  # edge x1-x2
        t = d1[edge] / (d1[edge] - d3[edge])
        weights[edge, 0] = 1.0 - t
        weights[edge, 1] = t

        edge = claim((vb <= 0.0) & (d2 >= 0.0) & (d6 <= 0.0))  # edge x1-x3
        t = d2[edge] / (d2[edge] - d6[edge])
        weights[edge, 0] = 1.0 - t
        weights[edge, 2] = t

        d4MinusD3 = d4 - d3
        d5MinusD6 = d5 - d6
        edge = claim((va <= 0.0) & (d4MinusD3 >= 0.0) & (d5MinusD6 >= 0.0))  # edge x2-x3
        t = d4MinusD3[edge] / (d4MinusD3[edge] + d5MinusD6[edge])
        weights[edge, 1] = 1.0 - t
        weights[edge, 2] = t

        interior = unassigned
        denom = 1.0 / (va[interior] + vb[interior] + vc[interior])
        beta = vb[interior] * denom
        gamma = vc[interior] * denom
        weights[interior, 0] = 1.0 - beta - gamma
        weights[interior, 1] = beta
        weights[interior, 2] = gamma

    closestPoints = weights[:, 0:1] * x1 + weights[:, 1:2] * x2 + weights[:, 2:3] * x3
    toClosestPoints = xs - closestPoints
    return weights, np.sqrt(rowDot(toClosestPoints, toClosestPoints))


def line2ClosestPoint(xs: np.ndarray, x1: np.ndarray, x2: np.ndarray) -> tuple[np.ndarray, float]:
    """Closest point on the (closed) segment (x1, x2) to xs, as parametric weights (1-t, t) with
    t clamped to [0, 1], plus the distance.

    The single-pair form of :func:`line2ClosestPoints`, which it calls, so that the two can never
    disagree."""

    weights, distances = line2ClosestPoints(xs[None], x1[None], x2[None])
    return weights[0], float(distances[0])


def line2ClosestPoints(xs: np.ndarray, x1: np.ndarray, x2: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Closest points of many (point, segment) pairs at once, see :func:`line2ClosestPoint`.

    Parameters
    ----------
    xs
        The points, shape ``(nPairs, 2)``.
    x1, x2
        The nodes of each pair's segment, in its local order, each of shape ``(nPairs, 2)``.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray]
        The weights of the closest points, shape ``(nPairs, 2)``, and the distances, shape
        ``(nPairs,)``.
    """

    e = x2 - x1
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.clip(rowDot(xs - x1, e) / rowDot(e, e), 0.0, 1.0)
    weights = np.stack([1.0 - t, t], axis=1)
    toClosestPoints = xs - (weights[:, 0:1] * x1 + weights[:, 1:2] * x2)
    return weights, np.sqrt(rowDot(toClosestPoints, toClosestPoints))


def tria3Projection(xs: np.ndarray, x1: np.ndarray, x2: np.ndarray, x3: np.ndarray) -> tuple[float, float, bool]:
    """Barycentric-like in-plane coordinates (alpha, beta) of the projection of xs onto the
    (possibly non-orthogonal) basis spanned by (x2-x1, x3-x1), and whether that projection falls
    inside the triangle.

    Unlike :func:`tria3ClosestPoint`, the projection is not clamped to the triangle."""

    e1 = x2 - x1
    e2 = x3 - x1
    r = xs - x1
    n = np.cross(e1, e2)
    n = n / np.linalg.norm(n)
    rTangential = r - r.dot(n) * n

    A = np.array([[e1.dot(e1), e1.dot(e2)], [e1.dot(e2), e2.dot(e2)]])
    b = np.array([e1.dot(rTangential), e2.dot(rTangential)])
    alpha, beta = np.linalg.solve(A, b)

    inside = alpha >= 0.0 and beta >= 0.0 and (alpha + beta) <= 1.0
    return alpha, beta, inside


def line2Projection(xs: np.ndarray, x1: np.ndarray, x2: np.ndarray) -> tuple[float, bool]:
    """Parametric coordinate t of the projection of xs onto the edge (x1,x2), and whether that
    projection falls inside the segment.

    Unlike :func:`line2ClosestPoint`, the projection is not clamped to the segment."""

    e = x2 - x1
    t = (xs - x1).dot(e) / e.dot(e)
    return t, 0.0 <= t <= 1.0


def facetNormalAndMeasure(coords: np.ndarray) -> tuple[np.ndarray, float | np.ndarray]:
    """The (non-unit-normalized only in intermediate steps) outward normal and measure (area for a
    Tria3 facet, length for a Line2 facet) of a flat facet, as a function of its current node
    coordinates; of many facets at once if their coordinates are stacked.

    Parameters
    ----------
    coords
        Array of shape ``(3, 3)`` (Tria3, 3D) or ``(2, 2)`` (Line2, 2D) with the facet's current
        node coordinates in its fixed local order; or of shape ``(..., 3, 3)`` or ``(..., 2, 2)``, the
        coordinates of many facets.

    Returns
    -------
    tuple[numpy.ndarray, float | numpy.ndarray]
        The outward unit normal, and the facet's measure (area or length); of shapes ``(..., 3)`` (or
        ``(..., 2)``) and ``(...)`` for many facets.
    """

    nNodes, domainSize = coords.shape[-2:]

    if nNodes == 3 and domainSize == 3:
        e1 = coords[..., 1, :] - coords[..., 0, :]
        e2 = coords[..., 2, :] - coords[..., 0, :]
        c = np.cross(e1, e2)
        cNorm = np.linalg.norm(c, axis=-1)
        return c / cNorm[..., np.newaxis], 0.5 * cNorm

    elif nNodes == 2 and domainSize == 2:
        e = coords[..., 1, :] - coords[..., 0, :]
        eNorm = np.linalg.norm(e, axis=-1)
        # Outward normal is e rotated by -90 degrees, consistent with a counter-clockwise
        # (node 1 -> node 2) traversal of the solid's boundary.
        n = np.stack((e[..., 1], -e[..., 0]), axis=-1) / eNorm[..., np.newaxis]
        return n, eNorm

    raise ValueError(f"facetNormalAndMeasure: unsupported facet shape {coords.shape}.")


def closestFacetCandidates(queryPoints: np.ndarray, facetCoords: list, searchDistance: float | None) -> list:
    """For each query point, the facets that could possibly be its closest one.

    Replaces an exhaustive slave x facet sweep with a spatial query, WITHOUT changing which facet
    is selected. The exhaustive form called the clamped closest-point routine once per
    slave-facet pair -- 778,000 calls on the anchor pry-out model, measured at 5.98 s per
    connectivity update, which is why the solver had to be told to search only every few hundred
    increments.

    The bound that makes this exact: let ``F*`` be the truly closest facet, at point-to-closed-
    domain distance ``d*``, with centroid ``C*`` and radius ``r*`` (its largest centroid-to-vertex
    distance). A triangle's centroid lies inside its own closed domain, so the distance to the
    facet owning the nearest centroid is at most ``d0``, the distance to that centroid -- hence
    ``d* <= d0``. And ``|x - C*| <= d* + r* <= d0 + rMax``. So every facet that could win, or
    even tie, has its centroid inside a ball of radius ``d0 + rMax``, and querying that ball
    cannot miss it.

    Candidates are returned in ascending facet index, so the caller's "first strict minimum"
    selection picks exactly the facet the exhaustive sweep would have. The result is bit-identical,
    not merely equivalent.

    Parameters
    ----------
    queryPoints
        Current coordinates of the points to be projected, shape ``(nPoints, nDim)``. Slave nodes
        for the node-based formulation, quadrature points for the integrated one.
    facetCoords
        Current coordinates of each facet's nodes.
    searchDistance
        The caller's cut-off, if any. A facet beyond it is rejected anyway, so the query radius
        is capped accordingly.

    Returns
    -------
    list
        One ascending list of candidate facet indices per query point.
    """

    if not facetCoords:
        return [[] for _ in range(len(queryPoints))]

    stacked = np.asarray(facetCoords, dtype=float)  # (nFacets, nNodesPerFacet, nDim)
    centroids = stacked.mean(axis=1)
    radii = np.linalg.norm(stacked - centroids[:, None, :], axis=2).max(axis=1)
    rMax = float(radii.max())

    tree = cKDTree(centroids)
    nearestCentroidDistance, _ = tree.query(queryPoints, k=1)

    radius = nearestCentroidDistance + rMax
    if searchDistance is not None:
        # The caller rejects facets beyond searchDistance; no facet within it has its centroid beyond
        # searchDistance + rMax.
        radius = np.minimum(radius, searchDistance + rMax)

    # A few ulps of margin: a facet that exactly ties sits ON the radius, and rounding must not drop it.
    radius = radius * (1.0 + 8.0 * np.finfo(float).eps)

    ballCandidates = tree.query_ball_point(queryPoints, radius)

    # The ball uses the global rMax, which is loose on a refined surface. Refine per facet: its closed
    # domain is at least |x - C_i| - r_i away and d* <= d0, so a facet with |x - C_i| - r_i > d0 can
    # neither win nor tie. Same ulp margin as above.
    tolerance = 1.0 + 8.0 * np.finfo(float).eps

    # Evaluated for all (point, ball candidate) pairs at once, with the same arithmetic per pair as
    # one point at a time.
    queryPoints = np.asarray(queryPoints)
    pointIndices, facetIndices = candidatePairs(ballCandidates)
    centroidDistances = np.linalg.norm(centroids[facetIndices] - queryPoints[pointIndices], axis=1)
    surviving = centroidDistances - radii[facetIndices] <= nearestCentroidDistance[pointIndices] * tolerance
    pointIndices, facetIndices = pointIndices[surviving], facetIndices[surviving]

    # Ascending facet index per point; the pairs stay grouped by point.
    order = np.lexsort((facetIndices, pointIndices))
    survivingFacets = facetIndices[order].tolist()
    ends = np.cumsum(np.bincount(pointIndices, minlength=len(queryPoints))).tolist()
    return [survivingFacets[first:end] for first, end in zip([0] + ends[:-1], ends)]


def candidatePairs(candidatesPerPoint: list) -> tuple[np.ndarray, np.ndarray]:
    """Flatten per-point candidate lists into (point, facet) index pairs.

    Parameters
    ----------
    candidatesPerPoint
        One list of candidate facet indices per point, as :func:`closestFacetCandidates` returns.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray]
        The point and the facet index of every pair, ordered by point and, per point, in the order
        of its candidate list.
    """

    counts = np.fromiter(map(len, candidatesPerPoint), dtype=np.intp, count=len(candidatesPerPoint))
    pointIndices = np.repeat(np.arange(len(candidatesPerPoint)), counts)
    facetIndices = np.fromiter(
        itertools.chain.from_iterable(candidatesPerPoint), dtype=np.intp, count=int(counts.sum())
    )
    return pointIndices, facetIndices


def facetClosestPoints(points: np.ndarray, facetCoords: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Closest points of many (point, flat facet) pairs at once: :func:`tria3ClosestPoints` for
    Tria3 facets in 3D, :func:`line2ClosestPoints` for Line2 facets in 2D.

    Parameters
    ----------
    points
        The points, shape ``(nPairs, nDim)``.
    facetCoords
        The node coordinates of each pair's facet, shape ``(nPairs, nNodesPerFacet, nDim)``.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray]
        The weights of the closest points, shape ``(nPairs, nNodesPerFacet)``, and the distances,
        shape ``(nPairs,)``.
    """

    if facetCoords.shape[1:] == (3, 3):
        return tria3ClosestPoints(points, facetCoords[:, 0], facetCoords[:, 1], facetCoords[:, 2])
    if facetCoords.shape[1:] == (2, 2):
        return line2ClosestPoints(points, facetCoords[:, 0], facetCoords[:, 1])
    raise ValueError(f"facetClosestPoints: unsupported facet shape {facetCoords.shape[1:]}.")


def closestFacets(
    queryPoints: np.ndarray, facetCoords: list, candidatesPerPoint: list
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """For each query point, the closest of its candidate facets, the clamped closest point's
    weights in it, and the distance.

    The selection is exactly that of scanning each point's candidates in their order and keeping a
    facet whenever its distance is strictly below the best so far, starting from infinity: the
    first candidate at the minimal distance wins, and a point whose candidates are all at an
    infinite or NaN distance (or that has none) gets no facet. All (point, candidate) pairs are
    evaluated at once, see :func:`facetClosestPoints`, so the weights and distances are
    bit-identical to those of the scan.

    Parameters
    ----------
    queryPoints
        The points, shape ``(nPoints, nDim)``.
    facetCoords
        Current coordinates of each facet's nodes, all of the same shape.
    candidatesPerPoint
        One list of candidate facet indices per point, see :func:`closestFacetCandidates`.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]
        Per point the closest facet's index, -1 if none; its weights, shape
        ``(nPoints, nNodesPerFacet)``, zero if none; and the distance, infinite if none.
    """

    queryPoints = np.asarray(queryPoints, dtype=float)
    nPoints = len(queryPoints)
    closestFacet = np.full(nPoints, -1, dtype=np.intp)
    closestDistance = np.full(nPoints, np.inf)
    if not facetCoords:
        return closestFacet, np.zeros((nPoints, 0)), closestDistance

    stacked = np.asarray(facetCoords, dtype=float)  # (nFacets, nNodesPerFacet, nDim)
    closestWeights = np.zeros((nPoints, stacked.shape[1]))

    pointIndices, facetIndices = candidatePairs(candidatesPerPoint)
    weights, distances = facetClosestPoints(queryPoints[pointIndices], stacked[facetIndices])

    # The scan's strict "<" against a best that starts at infinity never accepts an infinite or a
    # NaN distance, and among equal distances keeps the first.
    acceptable = distances < np.inf
    pairs = np.flatnonzero(acceptable)
    if len(pairs) == 0:
        return closestFacet, closestWeights, closestDistance
    np.minimum.at(closestDistance, pointIndices[pairs], distances[pairs])
    pairs = pairs[distances[pairs] == closestDistance[pointIndices[pairs]]]
    # Pairs are ordered by point, and per point in candidate order: keep each point's first.
    pairs = pairs[np.r_[True, pointIndices[pairs[1:]] != pointIndices[pairs[:-1]]]]

    closestFacet[pointIndices[pairs]] = facetIndices[pairs]
    closestWeights[pointIndices[pairs]] = weights[pairs]
    return closestFacet, closestWeights, closestDistance
