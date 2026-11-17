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
The sub-elements of the quadrature point output of :mod:`edelweissfe.outputmanagers.ensight`: each quadrilateral or
hexahedron integrated by a Gauss rule with n points per direction is split into n^dim linear sub-elements, one per
quadrature point.

**The split.** Along each natural direction, [-1, 1] is split at the breakpoints b_k = -1 + sum_{m<k} w_m
(k = 0, ..., n) of the Gauss weights w_m. Sub-element (i, j) spans [b_i, b_{i+1}] x [b_j, b_{j+1}] in natural
coordinates (and [b_k, b_{k+1}] in 3D), so its natural area is the weight of its Gauss point, which it contains.

**The lattice.** The corners of the sub-elements form a lattice of (n + 1)^dim points in each element, the same for
all elements of a shape (:class:`SubdivisionTemplate`). Each lattice point becomes a node of the part:

- an element node, where one lies (the corners, and for an even n the edge midpoints of a quadratic element);
- inside the element, a new node of that element;
- on an edge or face, a node shared with the neighbors.

New nodes are placed by the element's interpolation, which also maps the per-node results to them.

**Shared points.** Neighbors number a common edge or face differently, so a point on it is identified by a key
independent of that: the corner nodes of the edge (face), sorted, each with the point's lattice distances to it
along the edge (face), sorted. If one of the elements has a node at the point (a Quad8 next to a Quad4), all of
them use it.

All elements of a shape are processed at once, as arrays: a loop over the elements is too slow for large meshes, and
the part is rebuilt after each mesh change.
"""

import functools
from collections import namedtuple
from dataclasses import dataclass

import numpy as np
from scipy.sparse import csr_matrix

from edelweissfe.elements._hexa3dnodeordering import (
    hexa8N,
    hexa8NodeXi,
    hexa20N,
    hexa20NodeXi,
)
from edelweissfe.elements._quad2dnodeordering import (
    quad4N,
    quad4NodeXi,
    quad8N,
    quad8NodeXi,
)
from edelweissfe.points.node import Node

_Shape = namedtuple("_Shape", ["nodeXi", "shapeFunctions", "subShape"])

# the element shapes which can be split, by Ensight type: the sub-elements are linear
_SHAPES = {
    "quad4": _Shape(quad4NodeXi, quad4N, "quad4"),
    "quad8": _Shape(quad8NodeXi, quad8N, "quad4"),
    "hexa8": _Shape(hexa8NodeXi, hexa8N, "hexa8"),
    "hexa20": _Shape(hexa20NodeXi, hexa20N, "hexa8"),
}

# elements per batch when matching quadrature points to cells, which bounds the array of their distances
_MATCHING_BATCH_SIZE = 512


@dataclass(frozen=True)
class SharedPoints:
    """The lattice points on the edges, or on the faces, of an element: those with the same number of natural
    coordinates fixed at -1 or 1."""

    #: (nPoints,) the lattice points
    points: np.ndarray
    #: (nPoints, nCorners) the element corners of the edge or face of each point
    corners: np.ndarray
    #: (nPoints, nCorners, nFree) the lattice distances of each point to these corners along the edge or face, sorted
    distances: np.ndarray


@dataclass(frozen=True)
class SubdivisionTemplate:
    """The split of all elements of a shape with n Gauss points per direction, in natural coordinates."""

    dim: int
    #: the shape of the sub-elements
    subShape: str
    #: (nCells, nNodes) the shape functions at the Gauss point of each cell of the lattice
    shapeFunctionsAtGaussPoints: np.ndarray
    #: (nLattice, nNodes) the shape functions at each lattice point
    shapeFunctionsAtLattice: np.ndarray
    #: the element nodes which lie on the lattice ...
    ownNodes: np.ndarray
    #: ... and their lattice points
    ownPoints: np.ndarray
    #: the lattice points inside the element
    interiorPoints: np.ndarray
    #: the lattice points on edges and faces, by the number of fixed natural coordinates (1 for edges in 2D and faces
    #: in 3D, 2 for edges in 3D)
    sharedPoints: dict[int, SharedPoints]
    #: (nCells, 2**dim) the lattice points at the corners of each cell, in the order of the element corners
    cellCorners: np.ndarray


@dataclass(frozen=True)
class QuadraturePointSubdivision:
    """The sub-elements of an element set."""

    #: the nodes of the element set which the sub-elements use (in their order), then the added nodes
    nodes: list[Node]
    #: the shape of the sub-elements, quad4 or hexa8
    subShape: str
    #: (nElements, nQp, 2**dim) the node indices of the sub-element of each quadrature point of each element
    connectivity: np.ndarray
    #: (len(nodes), len(baseNodes)) maps per-node results of the element set to the nodes
    mapping: csr_matrix


@dataclass
class _ElementsOfShape:
    """The elements of an element set with one shape, processed together."""

    template: SubdivisionTemplate
    #: (nE,) the positions of the elements in the element set
    positions: np.ndarray
    #: (nE, nNodes) the indices of their nodes among the nodes of the element set
    nodes: np.ndarray
    #: (nE, nNodes, nCoordinates) the coordinates of their nodes
    coordinates: np.ndarray
    #: (nE, nQp, nCoordinates) the coordinates of their quadrature points
    quadraturePoints: np.ndarray
    #: (nE, nQp) the lattice cell of each quadrature point
    cellOfQuadraturePoint: np.ndarray = None
    #: (nE, nLattice) the node index of each lattice point, -1 while not assigned
    lattice: np.ndarray = None


@dataclass(frozen=True)
class _Occurrences:
    """Shared lattice points, one entry per element holding them."""

    #: (m, keyLength) the key which identifies the point in all elements holding it
    keys: np.ndarray
    #: (m,) the element node at the point, or -1
    node: np.ndarray
    #: (m,) the index of the element's _ElementsOfShape
    group: np.ndarray
    #: (m,) the element's row in it
    element: np.ndarray
    #: (m,) the lattice point in the element
    point: np.ndarray

    @staticmethod
    def concatenate(occurrences: list) -> "_Occurrences":
        return _Occurrences(
            keys=np.concatenate([o.keys for o in occurrences]),
            node=np.concatenate([o.node for o in occurrences]),
            group=np.concatenate([o.group for o in occurrences]),
            element=np.concatenate([o.element for o in occurrences]),
            point=np.concatenate([o.point for o in occurrences]),
        )


class _AddedNodes:
    """The nodes added to the part, numbered after the nodes of the element set, each placed by the shape functions
    of one element."""

    def __init__(self, nBase: int):
        self.nBase = nBase
        self.count = 0
        self.coordinates = []
        # the interpolation of the per-node results: rows (added node), columns (element node), weights
        self.rows, self.columns, self.weights = [], [], []

    def add(self, shapeFunctions: np.ndarray, elementNodes: np.ndarray, elementCoordinates: np.ndarray) -> np.ndarray:
        """Add one node per row of the shape functions (m, nNodes), of elements with nodes (m, nNodes) at the
        coordinates (m, nNodes, nCoordinates), and return their indices."""

        indices = self.nBase + self.count + np.arange(len(shapeFunctions))
        self.count += len(indices)
        self.coordinates.append(np.einsum("mn,mnd->md", shapeFunctions, elementCoordinates))
        self.rows.append(np.repeat(indices, shapeFunctions.shape[1]))
        self.columns.append(elementNodes.ravel())
        self.weights.append(shapeFunctions.ravel())
        return indices


@functools.lru_cache
def _gaussSubdivision(n: int) -> tuple[np.ndarray, np.ndarray]:
    """The points of the Gauss rule with n points, and the n + 1 breakpoints which split [-1, 1] in proportion to its
    weights: each point lies in the interval of its weight."""

    points, weights = np.polynomial.legendre.leggauss(n)
    breakpoints = np.concatenate([[-1.0], -1.0 + np.cumsum(weights)])
    # symmetric, with the exact values -1, 1 and, for an even n, 0 of the element nodes
    breakpoints = 0.5 * (breakpoints - breakpoints[::-1])
    breakpoints[[0, -1]] = -1.0, 1.0
    return points, breakpoints


def _sharedPoints(lattice: np.ndarray, isFixed: np.ndarray, cornerLattice: np.ndarray, nFixed: int) -> SharedPoints:
    """The lattice points with nFixed natural coordinates fixed at -1 or 1, the corners of their edge or face, and
    their distances to these corners."""

    dim = lattice.shape[1]
    points = np.flatnonzero(isFixed.sum(axis=1) == nFixed)
    corners, distances = [], []
    for point in points:
        fixed = isFixed[point]
        # the corners of the edge or face of the point: those which share its fixed coordinates
        cornersOfPoint = np.flatnonzero(np.all(cornerLattice[:, fixed] == lattice[point, fixed], axis=1))
        corners.append(cornersOfPoint)
        # sorted, since a neighbor may run along the edge or face in other natural directions
        distances.append(np.sort(np.abs(lattice[point] - cornerLattice[cornersOfPoint])[:, ~fixed], axis=1))

    nCorners, nFree = 2 ** (dim - nFixed), dim - nFixed
    return SharedPoints(
        points,
        np.reshape(np.array(corners, dtype=int), (len(points), nCorners)),
        np.reshape(np.array(distances, dtype=int), (len(points), nCorners, nFree)),
    )


@functools.lru_cache
def subdivisionTemplate(ensightType: str, n: int) -> SubdivisionTemplate:
    """The split of the elements of a shape with n Gauss points per direction (cached)."""

    nodeXi, shapeFunctions, subShape = _SHAPES[ensightType]
    dim = nodeXi.shape[1]
    cornerXi = nodeXi[: 2**dim]
    gaussPoints, breakpoints = _gaussSubdivision(n)

    # the multi-indices of the cells and of the lattice points
    latticeShape = dim * (n + 1,)
    cells = np.indices(dim * (n,)).reshape(dim, -1).T
    lattice = np.indices(latticeShape).reshape(dim, -1).T

    # the element nodes on the lattice: the corners, and for an even n the edge midpoints
    latticeIndexOfXi = {-1: 0, 1: n} | ({0: n // 2} if n % 2 == 0 else {})
    ownNodes = [a for a, xi in enumerate(nodeXi) if all(c in latticeIndexOfXi for c in xi)]
    ownPoints = [np.ravel_multi_index([latticeIndexOfXi[c] for c in nodeXi[a]], latticeShape) for a in ownNodes]

    # which natural coordinates of each lattice point are fixed at -1 or 1
    isFixed = (lattice == 0) | (lattice == n)
    cornerLattice = (cornerXi + 1) // 2 * n

    # the corners of each cell, in the order of the element corners
    cellCorners = cells[:, None, :] + (cornerXi[None, :, :] + 1) // 2

    return SubdivisionTemplate(
        dim=dim,
        subShape=subShape,
        shapeFunctionsAtGaussPoints=np.asarray(shapeFunctions(*gaussPoints[cells].T), dtype=float).T,
        shapeFunctionsAtLattice=np.asarray(shapeFunctions(*breakpoints[lattice].T), dtype=float).T,
        ownNodes=np.array(ownNodes, dtype=int),
        ownPoints=np.array(ownPoints, dtype=int),
        interiorPoints=np.flatnonzero(~isFixed.any(axis=1)),
        sharedPoints={nFixed: _sharedPoints(lattice, isFixed, cornerLattice, nFixed) for nFixed in range(1, dim)},
        cellCorners=np.ravel_multi_index(tuple(np.moveaxis(cellCorners, 2, 0)), latticeShape),
    )


def _notAGaussRule(element, nQp: int) -> Exception:
    return Exception(
        "Quadrature point output on sub-elements requires a Gauss rule with the same number of quadrature points in "
        "each direction, element {:} with {:} quadrature points does not use one".format(element.elNumber, nQp)
    )


def _checkElements(setName: str, elements: list) -> tuple[int, int, int]:
    """The number of quadrature points, of Gauss points per direction and of dimensions of the elements; raises if
    they cannot be split."""

    if not elements:
        raise Exception("Quadrature point output on sub-elements requires a non-empty element set")

    nQp = elements[0].getNumberOfQuadraturePoints()
    for element in elements:
        if element.ensightType not in _SHAPES:
            raise Exception(
                "Quadrature point output on sub-elements requires Quad4, Quad8, Hexa8 or Hexa20 elements, "
                "element {:} is {:}".format(element.elNumber, element.ensightType)
            )
        if element.getNumberOfQuadraturePoints() != nQp:
            raise Exception(
                "Quadrature point output on sub-elements requires the same number of quadrature points in all "
                "elements, element set {:} holds elements with {:} and with {:}".format(
                    setName, nQp, element.getNumberOfQuadraturePoints()
                )
            )

    dims = {_SHAPES[element.ensightType].nodeXi.shape[1] for element in elements}
    if len(dims) > 1:
        raise Exception(
            "Quadrature point output on sub-elements requires an element set of quadrilaterals only or of "
            "hexahedra only, element set {:} holds both".format(setName)
        )
    (dim,) = dims

    n = round(nQp ** (1 / dim))
    if n < 1 or n**dim != nQp:
        raise _notAGaussRule(elements[0], nQp)
    return nQp, n, dim


def _groupByShape(elements: list, baseNodes: list[Node], nQp: int, n: int) -> list[_ElementsOfShape]:
    """The elements, grouped by their shape."""

    baseIndex = {node: i for i, node in enumerate(baseNodes)}
    baseCoordinates = np.reshape([node.coordinates for node in baseNodes], (len(baseNodes), -1))
    nCoordinates = baseCoordinates.shape[1]

    positionsOfShape = {}
    for position, element in enumerate(elements):
        positionsOfShape.setdefault(element.ensightType, []).append(position)

    groups = []
    for shape, positions in positionsOfShape.items():
        nodes = np.array([[baseIndex[node] for node in elements[p].visualizationNodes] for p in positions])
        quadraturePoints = []
        for p in positions:
            points = np.asarray(elements[p].getCoordinatesAtQuadraturePoints(), dtype=float)
            if points.size != nQp * nCoordinates:
                raise Exception(
                    "Quadrature point output on sub-elements requires quadrature points of the dimension of the "
                    "nodes ({:}), element {:} has quadrature points of another dimension".format(
                        nCoordinates, elements[p].elNumber
                    )
                )
            quadraturePoints.append(np.reshape(points, (nQp, nCoordinates)))

        groups.append(
            _ElementsOfShape(
                template=subdivisionTemplate(shape, n),
                positions=np.array(positions),
                nodes=nodes,
                coordinates=baseCoordinates[nodes],
                quadraturePoints=np.reshape(quadraturePoints, (len(positions), nQp, nCoordinates)),
            )
        )
    return groups


def _matchQuadraturePointsToCells(group: _ElementsOfShape) -> np.ndarray:
    """Find the cell of each quadrature point, the one of its Gauss point, and return for each element whether its
    quadrature points are the Gauss points, one per cell."""

    nE, nQp, _ = group.quadraturePoints.shape
    gaussPoints = np.einsum("cn,end->ecd", group.template.shapeFunctionsAtGaussPoints, group.coordinates)
    # the rounding error grows with the magnitude of the coordinates, the distance of the Gauss points with the size
    size = np.ptp(group.coordinates, axis=1).max(axis=1)
    magnitude = np.abs(group.coordinates).max(axis=(1, 2))
    tolerance = 1e-8 * np.maximum(size, magnitude)

    group.cellOfQuadraturePoint = np.empty((nE, nQp), dtype=int)
    isGaussRule = np.empty(nE, dtype=bool)
    for start in range(0, nE, _MATCHING_BATCH_SIZE):
        batch = slice(start, start + _MATCHING_BATCH_SIZE)
        # (elements, quadrature points, Gauss points)
        distances = np.linalg.norm(group.quadraturePoints[batch, :, None, :] - gaussPoints[batch, None, :, :], axis=3)
        group.cellOfQuadraturePoint[batch] = distances.argmin(axis=2)
        isGaussRule[batch] = distances.min(axis=2).max(axis=1) <= tolerance[batch]

    # and no two quadrature points at the same Gauss point
    return isGaussRule & np.all(np.sort(group.cellOfQuadraturePoint, axis=1) == np.arange(nQp), axis=1)


def _addInteriorNodes(group: _ElementsOfShape, addedNodes: _AddedNodes):
    """Assign the element nodes to their lattice points, and add a node at each lattice point inside an element."""

    template = group.template
    nE = len(group.positions)
    group.lattice = np.full((nE, len(template.shapeFunctionsAtLattice)), -1, dtype=int)
    group.lattice[:, template.ownPoints] = group.nodes[:, template.ownNodes]

    interior = template.interiorPoints
    nodes = addedNodes.add(
        np.tile(template.shapeFunctionsAtLattice[interior], (nE, 1)),
        np.repeat(group.nodes, len(interior), axis=0),
        np.repeat(group.coordinates, len(interior), axis=0),
    )
    group.lattice[:, interior] = nodes.reshape(nE, len(interior))


def _occurrencesOfSharedPoints(groups: list[_ElementsOfShape], nFixed: int) -> _Occurrences:
    """The lattice points on edges or faces (nFixed natural coordinates fixed) of all elements, with their keys: for
    each corner of the edge or face, in the order of the node indices, its node and the point's distances to it."""

    occurrences = []
    for g, group in enumerate(groups):
        shared = group.template.sharedPoints[nFixed]
        nE, nPoints = len(group.positions), len(shared.points)

        cornerNodes = group.nodes[:, shared.corners]  # (nE, nPoints, nCorners)
        order = np.argsort(cornerNodes, axis=2)
        distances = np.broadcast_to(shared.distances, cornerNodes.shape + shared.distances.shape[2:])
        keys = np.concatenate(
            [
                np.take_along_axis(cornerNodes, order, axis=2)[..., None],
                np.take_along_axis(distances, order[..., None], axis=2),
            ],
            axis=3,
        )

        occurrences.append(
            _Occurrences(
                keys=keys.reshape(nE * nPoints, keys.shape[2] * keys.shape[3]),
                node=group.lattice[:, shared.points].ravel(),
                group=np.full(nE * nPoints, g),
                element=np.repeat(np.arange(nE), nPoints),
                point=np.tile(shared.points, nE),
            )
        )
    return _Occurrences.concatenate(occurrences)


def _shareNodesOnEdgesAndFaces(groups: list[_ElementsOfShape], addedNodes: _AddedNodes, dim: int):
    """Assign one node to each point on edges and faces, the same in all elements holding it."""

    for nFixed in range(1, dim):
        occurrences = _occurrencesOfSharedPoints(groups, nFixed)
        if not len(occurrences.keys):
            continue

        # the distinct points, the first occurrence of each, and the point of each occurrence
        _, firstOccurrence, pointOfOccurrence = np.unique(
            occurrences.keys, axis=0, return_index=True, return_inverse=True
        )
        pointOfOccurrence = pointOfOccurrence.ravel()

        # an element node, if one of the elements holding the point has one there, else a new node placed by the
        # element holding it first
        nodeOfPoint = np.full(len(firstOccurrence), -1, dtype=int)
        np.maximum.at(nodeOfPoint, pointOfOccurrence, occurrences.node)
        isNew = nodeOfPoint < 0
        for g, group in enumerate(groups):
            newHere = np.flatnonzero(isNew & (occurrences.group[firstOccurrence] == g))
            first = firstOccurrence[newHere]
            elements = occurrences.element[first]
            nodeOfPoint[newHere] = addedNodes.add(
                group.template.shapeFunctionsAtLattice[occurrences.point[first]],
                group.nodes[elements],
                group.coordinates[elements],
            )

        for g, group in enumerate(groups):
            here = occurrences.group == g
            group.lattice[occurrences.element[here], occurrences.point[here]] = nodeOfPoint[pointOfOccurrence[here]]


def _connectivity(groups: list[_ElementsOfShape], nElements: int, nQp: int, dim: int) -> np.ndarray:
    """The nodes of the sub-element of each quadrature point: the corners of its cell."""

    connectivity = np.empty((nElements, nQp, 2**dim), dtype=int)
    for group in groups:
        nE = len(group.positions)
        cornerPoints = group.template.cellCorners[group.cellOfQuadraturePoint].reshape(nE, -1)
        connectivity[group.positions] = np.take_along_axis(group.lattice, cornerPoints, axis=1).reshape(nE, nQp, -1)
    return connectivity


def subdivideAtQuadraturePoints(setName: str, elements: list, baseNodes: list[Node]) -> QuadraturePointSubdivision:
    """Split each element of an element set into one linear sub-element per quadrature point, see the module
    documentation. Sub-element g of an element contains its quadrature point g, in whatever order the element
    numbers them.

    Parameters
    ----------
    setName
        The name of the element set.
    elements
        The elements, all quadrilaterals or all hexahedra, with the same Gauss rule.
    baseNodes
        The nodes of the element set.

    Returns
    -------
    QuadraturePointSubdivision
        The nodes and sub-elements.
    """

    elements = list(elements)
    nQp, n, dim = _checkElements(setName, elements)
    groups = _groupByShape(elements, baseNodes, nQp, n)

    isGaussRule = np.ones(len(elements), dtype=bool)
    for group in groups:
        isGaussRule[group.positions] = _matchQuadraturePointsToCells(group)
    if not isGaussRule.all():
        raise _notAGaussRule(elements[np.flatnonzero(~isGaussRule)[0]], nQp)

    addedNodes = _AddedNodes(len(baseNodes))
    for group in groups:
        _addInteriorNodes(group, addedNodes)
    _shareNodesOnEdgesAndFaces(groups, addedNodes, dim)
    connectivity = _connectivity(groups, len(elements), nQp, dim)

    # keep the nodes of the element set which the sub-elements use, e.g. not the edge midpoints for an odd n
    nBase = len(baseNodes)
    isUsed = np.zeros(nBase + addedNodes.count, dtype=bool)
    isUsed[nBase:] = True
    isUsed[connectivity.ravel()] = True
    usedBase = np.flatnonzero(isUsed[:nBase])
    newIndex = np.cumsum(isUsed) - 1

    firstLabel = max(node.label for node in baseNodes) + 1
    coordinates = np.concatenate(addedNodes.coordinates)
    nodes = [baseNodes[i] for i in usedBase] + [Node(firstLabel + i, x) for i, x in enumerate(coordinates)]
    mapping = csr_matrix(
        (
            np.concatenate([np.ones(len(usedBase))] + addedNodes.weights),
            (
                np.concatenate([np.arange(len(usedBase))] + [newIndex[rows] for rows in addedNodes.rows]),
                np.concatenate([usedBase] + addedNodes.columns),
            ),
        ),
        shape=(len(nodes), nBase),
    )
    return QuadraturePointSubdivision(nodes, groups[0].template.subShape, newIndex[connectivity], mapping)
