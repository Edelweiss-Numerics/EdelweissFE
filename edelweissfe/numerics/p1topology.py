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
(:class:`~edelweissfe.linsolve.blockamg.ptwogrid.PTwoGridPreconditioner`).

The projection :math:`P` is purely topological: identity on corner nodes, ½/½ on each exclusive
midside node from its two edge-endpoint corners (the P1 function expressed in the serendipity
basis) -- so building it only requires classifying every node of a vector field as a corner or an
exclusive midside, with its two edge-endpoint corners if the latter.
"""

import numpy as np

#: ``(nSpatialDimensions, nNodes) -> (cornerLocalIndices, [(midsideLocal, cornerALocal, cornerBLocal), ...])``
#: the only two quadratic-serendipity element families in this codebase that carry exclusive
#: midside nodes (verified against the actual shape functions, both the pure-Python
#: ``edelweissfe.elements.displacementelement`` and Marmot's ``DisplacementFiniteElement``, and
#: cross-checked against ``edelweissfe.adaptivity.hex20shapefunctions.EDGES`` for Hexa20).
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
_QUADRATIC_TOPOLOGY = {(2, 8): _QUAD8_TOPOLOGY, (3, 20): _HEXA20_TOPOLOGY}

#: node counts verified, against this repo's full element-type inventory (every ``*element,
#: type=...`` across ``testfiles/``), to carry no midside nodes at all whenever the
#: ``(nSpatialDimensions, nNodes)`` pair is not one of the two quadratic families above -- e.g.
#: ``C3D8``/``GC3D8*`` (3, 8), ``GCPS4`` (2, 4), ``T2D2`` (2, 2). A contact facet has no
#: ``nSpatialDimensions`` of its own at all and is always pure-corner regardless of node count,
#: since a facet element cannot itself carry a midside concept.
_KNOWN_LINEAR_NODE_COUNTS = {1, 2, 3, 4, 6, 8}


def classifyElementTopology(nSpatialDimensions, nNodes: int):
    """Return ``(cornerLocalIndices, edgeEndpointsLocal)`` for one element's local node numbering.

    Parameters
    ----------
    nSpatialDimensions
        The element's spatial dimension, or ``None`` if unavailable (e.g. a contact facet, always
        treated as pure-corner regardless of node count).
    nNodes
        The element's node count.

    Returns
    -------
    tuple
        ``cornerLocalIndices`` (list of int) and ``edgeEndpointsLocal`` (list of
        ``(midsideLocalIndex, cornerALocalIndex, cornerBLocalIndex)``, empty for a pure-corner
        element).

    Raises
    ------
    ValueError
        If ``nNodes`` is neither a recognized quadratic-serendipity count for
        ``nSpatialDimensions`` nor among the verified pure-corner node counts -- silently
        misclassifying an unrecognized element's midside nodes as corners would quietly degrade
        the resulting P1 operator, so this fails loudly instead.
    """
    key = (nSpatialDimensions, nNodes)
    if key in _QUADRATIC_TOPOLOGY:
        return _QUADRATIC_TOPOLOGY[key]
    if nSpatialDimensions is None or nNodes in _KNOWN_LINEAR_NODE_COUNTS:
        return list(range(nNodes)), []
    raise ValueError(
        "p1topology: unrecognized element topology (nSpatialDimensions={:}, nNodes={:}) -- neither "
        "a known quadratic-serendipity family ((2, 8) Quad8, (3, 20) Hexa20) nor a verified linear "
        "node count. Add its corner/midside topology before including it in a P1 map.".format(
            nSpatialDimensions, nNodes
        )
    )


def buildP1Map(model, fieldName: str):
    """Classify every node of ``fieldName`` (a vector field, e.g. ``"displacement"``) as a corner
    or an exclusive midside, in the same order as ``model.nodeFields[fieldName].nodes`` (the
    field's own DOF-vector row order, see ``DofManager._reserveSpaceForNodeFields``).

    A node is a corner iff it is a corner node of *at least one* element it belongs to, **or** if
    two elements disagree about its edge endpoints (see below) -- both cases are load-bearing
    under AMR, and every element's own contribution only ever sets ``isCorner`` to ``True``, never
    clears it.

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
    the theoretical minimum, never wrong. Every such fallback is recorded in the returned
    ``warnings`` list rather than passed over silently.

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
    warnings : list[str]
        One entry per node demoted to a corner because of a genuine edge-endpoint disagreement
        between two elements (empty on an ordinary, conforming mesh).
    """
    field = model.nodeFields[fieldName]
    nodeRows = {node: i for i, node in enumerate(field.nodes)}
    nNodes = len(field.nodes)
    isCorner = np.zeros(nNodes, dtype=bool)
    # provisional -- a node classified as a midside by the element seen first may turn out to be a
    # corner of a *different* element processed later (the AMR corner-wins case), so isCorner and
    # the edge-endpoint candidates are collected in one pass over every element first, and only
    # resolved into the final per-row result afterward -- writing straight into a shared
    # `edgeEndpoints` array during this loop would leave a stale entry for exactly that case.
    provisionalEdges: dict[int, frozenset] = {}
    conflictingRows: set[int] = set()

    for element in model.elements.values():
        if not any(fieldName in nodeFields for nodeFields in element.fields):
            continue

        try:
            dim = element.nSpatialDimensions
        except AttributeError:
            dim = None
        elNodes = element.nodes
        cornersLocal, edgesLocal = classifyElementTopology(dim, len(elNodes))
        elRows = [nodeRows.get(node) for node in elNodes]

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

    warnings = []
    for row in conflictingRows:
        isCorner[row] = True
        warnings.append(
            "p1topology: node (row {:}, field '{:}') had conflicting edge-endpoint candidates "
            "from different elements -- treated as a corner (safe: identity in P1 regardless), "
            "not the exclusive midside a conforming mesh would make it.".format(row, fieldName)
        )

    edgeEndpoints = -np.ones((nNodes, 2), dtype=int)
    for row in range(nNodes):
        if isCorner[row]:
            continue
        candidate = provisionalEdges.get(row)
        if candidate is None:
            raise AssertionError(
                "p1topology: node {:} of field '{:}' is neither a corner of any element nor has "
                "recorded edge endpoints -- every non-corner node must be an exclusive midside of "
                "at least one element.".format(row, fieldName)
            )
        edgeEndpoints[row] = tuple(candidate)

    return isCorner, edgeEndpoints, warnings
