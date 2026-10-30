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
#  This library is free software; you can redistribute it and/or
#  modify it under the terms of the GNU Lesser General Public
#  License as published by the Free Software Foundation; either
#  version 2.1 of the License, or (at your option) any later version.
#
#  The full text of the license can be found in the file LICENSE.md at
#  the top level directory of EdelweissFE.
#  ---------------------------------------------------------------------
"""Corner/midside node topology for building a P1 (linear) restriction operator over a quadratic
serendipity displacement mesh -- the enabler for p-multigrid
(:class:`~edelweissfe.linsolve.composition.cornernodetwolevelcycle.CornerNodeTwoLevelCycle`).

This module provides the topology only: it classifies every node of a vector field as a corner or an
exclusive midside, with its two edge-endpoint corners if the latter. The prolongation itself is built from it
in C++ (``CornerNodeTwoLevelCycleT`` in ``amgcl-wrapper.hpp``): in physical variables the P1 function
expressed in the serendipity basis, :math:`P_{geo}` -- identity on corner nodes, ½/½ on each midside node
from its two edge-endpoint corners --, and in the scaled variables of :math:`\\hat A = D A D` the scaled
weights :math:`P = D^{-1} P_{geo} D_c`, i.e. :math:`\\tfrac12 s_a / s_m` and :math:`\\tfrac12 s_b / s_m`. That
build also uses what the topology cannot know, the Dirichlet rows: a midside node with a free dof between two
constrained dofs of its edge's corners becomes a coarse node of its own (issue #223).
"""

import numpy as np

#: ``(cornerLocalIndices, [(midsideLocal, cornerALocal, cornerBLocal), ...])`` of the quadratic shapes
_BAR3_TOPOLOGY = ([0, 1], [(2, 0, 1)])
_QUAD8_TOPOLOGY = ([0, 1, 2, 3], [(4, 0, 1), (5, 1, 2), (6, 2, 3), (7, 3, 0)])
_HEXA20_TOPOLOGY = (
    list(range(8)),
    [
        (8, 0, 1),
        (9, 1, 2),
        (10, 2, 3),
        (11, 3, 0),
        (12, 4, 5),
        (13, 5, 6),
        (14, 6, 7),
        (15, 7, 4),
        (16, 0, 4),
        (17, 1, 5),
        (18, 2, 6),
        (19, 3, 7),
    ],
)

#: The corner/midside topology of every element shape this module knows, by the shape an element declares
#: (``element.ensightType``, Ensight Gold notation, whose node order Marmot's and EdelweissFE's elements follow).
_SHAPE_TOPOLOGY = {
    "point": ([0], []),
    "bar2": ([0, 1], []),
    "bar3": _BAR3_TOPOLOGY,
    "tria3": ([0, 1, 2], []),
    "quad4": ([0, 1, 2, 3], []),
    "quad8": _QUAD8_TOPOLOGY,
    "tetra4": ([0, 1, 2, 3], []),
    "penta6": (list(range(6)), []),
    "hexa8": (list(range(8)), []),
    "hexa20": _HEXA20_TOPOLOGY,
}


class UnsupportedElementTopology(ValueError):
    """An element whose corner/midside topology this module does not know (e.g. a Tet10), or whose nodes do not fit
    the shape it declares: its field has no midside corners to build a corner-node interpolation from."""


def classifyElementTopology(element):
    """The corner/midside topology of one element, from the shape it declares (``element.ensightType``).

    An element may have more nodes than its shape: a bond element has the nodes of its bar (its shape), then those of
    the host element it couples the bar to. The nodes after its shape's are nodes it couples to, which
    :func:`buildMidsideCorners` requires to be nodes of the shape of another element.

    Parameters
    ----------
    element
        The element.

    Returns
    -------
    tuple
        ``cornerLocalIndices`` (list of int), ``edgeEndpointsLocal`` (list of
        ``(midsideLocalIndex, cornerALocalIndex, cornerBLocalIndex)``, empty for a pure-corner shape) and the number
        of nodes of the shape.

    Raises
    ------
    UnsupportedElementTopology
        If the shape is not one of :data:`_SHAPE_TOPOLOGY`, or the element has fewer nodes than its shape.
    """
    shape = element.ensightType
    if shape not in _SHAPE_TOPOLOGY:
        raise UnsupportedElementTopology(
            "element {:} ({:}) has the shape '{:}', whose corner/midside topology the corner-node interpolation does "
            "not know (known: {:}). Switch the corner-node interpolation off for its field (blockamg: "
            "cornerNodeInterpolation=False), or add the topology of '{:}' to edelweissfe.numerics.p1topology.".format(
                element.elNumber, element.elType, shape, ", ".join(_SHAPE_TOPOLOGY), shape
            )
        )
    cornersLocal, edgesLocal = _SHAPE_TOPOLOGY[shape]
    nShapeNodes = len(cornersLocal) + len(edgesLocal)
    if len(element.nodes) < nShapeNodes:
        raise UnsupportedElementTopology(
            "element {:} ({:}) has {:} nodes, fewer than its shape '{:}' ({:})".format(
                element.elNumber, element.elType, len(element.nodes), shape, nShapeNodes
            )
        )
    return cornersLocal, edgesLocal, nShapeNodes


def buildMidsideCorners(model, fieldName: str):
    """Classify every node of ``fieldName`` (a vector field, e.g. ``"displacement"``) as a corner
    or an exclusive midside, in the same order as ``model.nodeFields[fieldName].nodes`` (the
    field's own DOF-vector row order, see ``DofManager._reserveSpaceForNodeFields``).

    Every element of the field is classified by the shape it declares (:func:`classifyElementTopology`); an element of
    a shape this module does not know is an error, never skipped. A node is a corner iff it is a corner node of *at
    least one* element's shape, if it belongs to no element at all (e.g. the reference point of a rigid body, which
    carries the field through a constraint), **or** if two elements disagree about its edge endpoints (see below) --
    both cases are load-bearing under AMR, and every element's own contribution only ever sets ``isCorner`` to
    ``True``, never clears it. A node an element only couples to (one after its shape's nodes, e.g. a host node of a
    bond element) must be a node of the shape of another element; else it is an error.

    **Why a disagreement falls back to "corner", rather than erroring.** A node classified as a
    midside by the coarse side of a 2:1-balanced non-conforming refinement boundary can
    legitimately coincide, in raw 3D coordinates, with a node the AMR module's coordinate-based
    node registry (``edelweissfe.adaptivity.refinement.NodeRegistry``) also assigns to an
    unrelated fine element on the other side of that boundary -- found live on a real reference
    model: two genuine, currently-active ``GC3D20R`` elements, both individually geometry-verified
    as internally self-consistent, disagreeing about one shared node's edge endpoints. Silently
    keeping only the first element's guess would be an arbitrary, unauditable
    choice; hard-erroring would block every model exhibiting this (evidently not rare) AMR
    interface pattern. Treating the node as a corner instead is always *structurally* safe for a
    P1 restriction operator -- a corner is identity in ``P`` regardless of why it was classified
    that way, so this can only ever make the P1 space a little larger (a few extra corners) than
    the theoretical minimum, never wrong. Every such fallback is returned (``conflictingRows``)
    rather than passed over silently.

    Parameters
    ----------
    model
        The model tree.
    fieldName
        The vector field to classify (e.g. ``"displacement"``).

    Returns
    -------
    isCorner : np.ndarray
        Boolean, shape ``(nNodes,)``.
    edgeEndpoints : np.ndarray
        Int, shape ``(nNodes, 2)``, ``-1`` for corner rows; for an exclusive midside row, the two
        edge-endpoint corner rows (both guaranteed corners themselves, by construction).
    conflictingRows : np.ndarray
        Int, sorted: the rows to which two elements give different edge endpoints; they are corners
        (empty on an ordinary, conforming mesh).

    Raises
    ------
    UnsupportedElementTopology
        If an element of the field has a shape this module does not know or does not fit its shape, or a node an
        element couples to is a node of no element's shape.
    """
    field = model.nodeFields[fieldName]
    nodeRows = {node: i for i, node in enumerate(field.nodes)}
    nNodes = len(field.nodes)
    isCorner = np.zeros(nNodes, dtype=bool)
    # Provisional: a midside of one element may be a corner of a later one (AMR), so candidates are
    # collected over all elements first and resolved afterward.
    provisionalEdges: dict[int, frozenset] = {}
    conflictingRows: set[int] = set()
    inSomeShape = np.zeros(nNodes, dtype=bool)
    coupledBy: dict[int, object] = {}

    for element in model.elements.values():
        if not any(fieldName in nodeFields for nodeFields in element.fields):
            continue

        cornersLocal, edgesLocal, nShapeNodes = classifyElementTopology(element)
        elRows = [nodeRows.get(node) for node in element.nodes]
        inSomeShape[[row for row in elRows[:nShapeNodes] if row is not None]] = True
        for row in elRows[nShapeNodes:]:
            if row is not None:
                coupledBy[row] = element

        for c in cornersLocal:
            row = elRows[c]
            if row is not None:
                isCorner[row] = True

        for midLocal, cALocal, cBLocal in edgesLocal:
            row = elRows[midLocal]
            if row is None:
                continue
            candidate = frozenset((elRows[cALocal], elRows[cBLocal]))
            existing = provisionalEdges.get(row)
            if existing is not None and existing != candidate:
                conflictingRows.add(row)
                continue
            provisionalEdges[row] = candidate

    for row, element in coupledBy.items():
        if not inSomeShape[row]:
            raise UnsupportedElementTopology(
                "element {:} ({:}, shape '{:}') couples to node {:} of the field '{:}', which is a node of no "
                "element's shape: its corner/midside topology is unknown".format(
                    element.elNumber, element.elType, element.ensightType, field.nodes[row].label, fieldName
                )
            )

    conflictingRows = np.array(sorted(conflictingRows), dtype=int)
    isCorner[conflictingRows] = True

    # a node of no element (e.g. the reference point of a rigid body) is a corner: identity in P1
    isCorner[~inSomeShape] = True
    edgeEndpoints = -np.ones((nNodes, 2), dtype=int)
    for row in np.nonzero(~isCorner)[0]:
        edgeEndpoints[row] = tuple(sorted(provisionalEdges[row]))

    return isCorner, edgeEndpoints, conflictingRows


def restrictMidsideCorners(isCorner: np.ndarray, edgeEndpoints: np.ndarray, remains: np.ndarray) -> tuple:
    """A field's midside corners (see :func:`buildMidsideCorners`) restricted to some of its nodes, ``remains`` (e.g.
    those a Schur complement does not condense): the rows are renumbered, and a midside node whose edge endpoint is not
    among them becomes a corner.
    That is always safe: the P1 restriction takes a corner as it is, so a promoted node only enlarges the coarse space
    of the corner-node two-level cycle, it never makes it wrong.

    Parameters
    ----------
    isCorner, edgeEndpoints
        The field's midside corners, see :func:`buildMidsideCorners`.
    remains
        Per node of the field, whether it remains.

    Returns
    -------
    tuple
        ``(isCorner, edgeEndpoints)`` of the remaining nodes.
    """
    newRow = np.cumsum(remains) - 1
    isCorner, edgeEndpoints = isCorner[remains].copy(), edgeEndpoints[remains].copy()
    isMidside = ~isCorner
    endpointRemains = np.ones_like(isMidside)
    endpointRemains[isMidside] = remains[edgeEndpoints[isMidside]].all(axis=1)
    becomesCorner = isMidside & ~endpointRemains
    isCorner[becomesCorner] = True
    edgeEndpoints[becomesCorner] = -1
    stillMidside = ~isCorner
    edgeEndpoints[stillMidside] = newRow[edgeEndpoints[stillMidside]]
    return isCorner, edgeEndpoints
