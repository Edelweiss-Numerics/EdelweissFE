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
Generates flat, geometry-only "contact facet" elements (:class:`~edelweissfe.elements.
contactsurfaceelement.Tria3ContactFacet` / ``Line2ContactFacet``) from an existing ``*surface``
definition, for use as the slave or master side of node-to-deformable-surface penalty contact.
Quad faces of 3D solids are split into two Tria3 facets via a fixed diagonal. Higher-order
element faces are either reduced to their linear corner-node subset (``triangulation=corner``,
exact for straight-edged meshes) or triangulated including their midside nodes
(``triangulation=midside``, strictly more accurate for curved faces). The facets carry
face-consistent per-node tributary area shares for the pressure-weighted contact formulation;
``nodalWeights`` selects between the facet tiling's own consistent shares and the non-negative
weighting that best approximates a *serendipity* face's consistent nodal loads. See the
:doc:`contact theory documentation </documentation/contacttheory>` for the full background.

The underlying :func:`buildContactFacets` is idempotent and re-runnable, so a
:class:`~edelweissfe.models.meshdependent.MeshDependent` consumer of these facets (e.g.
:mod:`~edelweissfe.constraints.nodetodeformablesurfacepenalty`) can regenerate them from the
current ``*surface`` definition after the source solid elements change underneath it (e.g. an AMR
refinement) as well as at setup time; the recipe is recorded in ``model.contactFacetRecipes``.

.. code-block:: edelweiss
    :caption: Example

    *modelGenerator, generator=surfaceElementGenerator, name=gen
        surface = mySurface
        name    = myContactSurface
"""

from dataclasses import dataclass

import numpy as np

from edelweissfe.elements.contactsurfaceelement import (
    Line2ContactFacet,
    Tria3ContactFacet,
)
from edelweissfe.generators.base.generatorbase import GeneratorBase
from edelweissfe.journal.journal import Journal
from edelweissfe.models.femodel import FEModel
from edelweissfe.sets.elementset import ElementSet
from edelweissfe.sets.nodeset import NodeSet
from edelweissfe.utils.parentfacegeometry import PARENT_FACE_PARAMETRIC_COORDS
from edelweissfe.utils.schema import schemaField

# Face-node-ordering tables, 0-indexed, reduced to each element type's linear corner nodes. Each
# face maps to a tuple of node-index groups: a 3-tuple is a Tria3 facet, a 2-tuple is a Line2 facet.
# Quad faces of 3D solids are split into two Tria3 facets via a diagonal, preserving the source
# face's winding (and thus outward orientation).
#
# hexa8/hexa20 face numbers/node groups are transcribed from Marmot's own face definitions
# (MarmotFiniteElement3D.cpp, Hexa8::getBoundaryElementIndices/Hexa20::getBoundaryElementIndices),
# fan-triangulated from each face's first listed corner. Since edelweissfe's own Hexa8/Hexa20
# elements use the same node ordering as Marmot (see edelweissfe.elements.displacementelement.
# _elementcomputationmatrices), this is also the genuine Abaqus S1..S6 face-numbering convention --
# unlike the codebase's previous, non-standard convention, a *surface keyword's S<n> face numbers
# now mean the same thing here as they do in Marmot or a real Abaqus model. boxGen/pipeGen's own
# face-number registration (1=Ymin, 2=Ymax, 3=Xmin, 4=Zmax, 5=Xmax, 6=Zmin) already matches this
# numbering -- verified by applying Marmot's face definitions directly to boxgen's own node layout
# (local node k <-> (ix, iy, iz) offsets) and checking that each triangle's cross-product normal
# points in the expected outward direction.
_FACE_TABLES = {
    "quad4": {
        1: ((0, 1),),
        2: ((1, 2),),
        3: ((2, 3),),
        4: ((3, 0),),
    },
    "quad8": {
        1: ((0, 1),),
        2: ((1, 2),),
        3: ((2, 3),),
        4: ((3, 0),),
    },
    "hexa8": {
        1: ((3, 2, 1), (3, 1, 0)),  # Ymin
        2: ((4, 5, 6), (4, 6, 7)),  # Ymax
        3: ((0, 1, 5), (0, 5, 4)),  # Xmin
        4: ((6, 5, 1), (6, 1, 2)),  # Zmax
        5: ((7, 6, 2), (7, 2, 3)),  # Xmax
        6: ((4, 7, 3), (4, 3, 0)),  # Zmin
    },
    "hexa20": {
        1: ((3, 2, 1), (3, 1, 0)),  # Ymin
        2: ((4, 5, 6), (4, 6, 7)),  # Ymax
        3: ((0, 1, 5), (0, 5, 4)),  # Xmin
        4: ((6, 5, 1), (6, 1, 2)),  # Zmax
        5: ((7, 6, 2), (7, 2, 3)),  # Xmax
        6: ((4, 7, 3), (4, 3, 0)),  # Zmin
    },
}

# Midside-triangulation tables for higher-order element faces: the full 8-node face boundary
# polygon (c1, m1, c2, m2, c3, m3, c4, m4) is split into 4 corner triangles (m_prev, c_i, m_i)
# plus the central midside quad (m1, m2, m3, m4) split into 2 triangles -- 6 flat Tria3 facets
# using only real nodes (2D quad8 edges: split at the midside node into 2 Line2 facets). NOTE: a
# naive fan from a *corner* would instead contain the boundary triangles (c1, m1, c2)/(c1, c4, m4)
# which are exactly degenerate (zero area, collinear nodes) for straight-edged meshes -- the
# midside-quad split has no such degenerate members. Identical coverage to the corner reduction
# for straight-edged meshes, strictly more accurate for curved faces (offset midside nodes).
# hexa20 corner cycles and midside indices are transcribed from Marmot's Hexa20::
# getBoundaryElementIndices (4 corners + 4 edge-midside nodes per face, same corner cycle as the
# corner table above); midside local indices 8-19 follow edelweissfe's/Marmot's own Hexa20 edge
# numbering (8-19 on edges 0-1, 1-2, 2-3, 3-0, 4-5, 5-6, 6-7, 7-4, 0-4, 1-5, 2-6, 3-7). All verified
# numerically against boxgen's actual node construction (face-plane membership, outward
# cross-product normals, non-degeneracy, area tiling, midside-between-corners positions).
# Weight map of the 'serendipityOptimal' option for a midside-triangulated quadratic face's
# corner triangles, whose table order is (m_prev, c, m_next): it reassigns the corner's (local
# index 1) interpolation weight in equal halves to its two adjacent midside nodes, which for this
# triangulation are exactly the facet's own other two nodes -- so the map never has to reach
# outside the facet. Column-stochastic, hence force-preserving. This is the facet-local form of
# the modified serendipity shape functions Ntilde_mid = N_mid + 1/2 (N_c1 + N_c2),
# Ntilde_corner = 0; see _applyModifiedSerendipityShares for the rationale.
_MODIFIED_CORNER_TRIANGLE_TRANSFORM = np.array(
    [
        [1.0, 0.5, 0.0],
        [0.0, 0.0, 0.0],
        [0.0, 0.5, 1.0],
    ]
)

_MIDSIDE_FACE_TABLES = {
    "quad8": {
        1: ((0, 4), (4, 1)),
        2: ((1, 5), (5, 2)),
        3: ((2, 6), (6, 3)),
        4: ((3, 7), (7, 0)),
    },
    "hexa20": {
        1: ((11, 3, 10), (10, 2, 9), (9, 1, 8), (8, 0, 11), (10, 9, 8), (10, 8, 11)),  # Ymin
        2: ((15, 4, 12), (12, 5, 13), (13, 6, 14), (14, 7, 15), (12, 13, 14), (12, 14, 15)),  # Ymax
        3: ((16, 0, 8), (8, 1, 17), (17, 5, 12), (12, 4, 16), (8, 17, 12), (8, 12, 16)),  # Xmin
        4: ((18, 6, 13), (13, 5, 17), (17, 1, 9), (9, 2, 18), (13, 17, 9), (13, 9, 18)),  # Zmax
        5: ((19, 7, 14), (14, 6, 18), (18, 2, 10), (10, 3, 19), (14, 18, 10), (14, 10, 19)),  # Xmax
        6: ((16, 4, 15), (15, 7, 19), (19, 3, 11), (11, 0, 16), (15, 19, 11), (15, 11, 16)),  # Zmin
    },
}


def canonicalParentFace(ensightType: str, faceNumber: int) -> tuple[str, tuple]:
    """The parent face of ``faceNumber``, as its canonical type and its element-local node indices
    in the canonical ordering of :mod:`~edelweissfe.utils.parentfacegeometry`.

    Derived from the face tables above rather than transcribed a third time, which is the point: a
    separate table of parent-face orderings would be one more chance to get a node ordering subtly
    wrong. The midside table lists every corner triangle as ``(m_prev, c, m_next)``, so the corner
    cycle is its middle entries and the midside between ``ck`` and ``c(k+1)`` is its last -- and the
    linear table's fan ``(a, b, c), (a, c, d)`` gives the corner cycle directly.

    The parent face is a property of the *element*, not of how the face was tiled: a hexa20 face is
    a quad8 parent face even under ``triangulation=corner``, where the facets happen to use only its
    corner nodes.

    Parameters
    ----------
    ensightType
        The source element's ensight type, e.g. ``"hexa20"``.
    faceNumber
        The face number, as used by the face tables.

    Returns
    -------
    tuple[str, tuple]
        The canonical face type and the parent face's element-local node indices in that ordering.
    """

    midsideTable = _MIDSIDE_FACE_TABLES.get(ensightType)
    if midsideTable is not None:
        groups = midsideTable.get(faceNumber)
        if groups is None:
            raise ValueError(
                f"surfaceElementGenerator: face {faceNumber} is not defined for element type " f"'{ensightType}'."
            )
        if len(groups[0]) == 3:
            corners = tuple(group[1] for group in groups[:4])
            midsides = tuple(group[2] for group in groups[:4])
            return "quad8", corners + midsides
        # a 2D higher-order element edge, split at its midside node into two Line2 facets
        return "line3", (groups[0][0], groups[1][1], groups[0][1])

    linearTable = _FACE_TABLES.get(ensightType)
    if linearTable is None:
        raise ValueError(
            f"surfaceElementGenerator: no face-node-ordering table available for element type " f"'{ensightType}'."
        )
    groups = linearTable.get(faceNumber)
    if groups is None:
        raise ValueError(
            f"surfaceElementGenerator: face {faceNumber} is not defined for element type " f"'{ensightType}'."
        )
    if len(groups[0]) == 3:
        return "quad4", (groups[0][0], groups[0][1], groups[0][2], groups[1][2])
    return "line2", tuple(groups[0])


def _stampParentFace(facet, sourceElement, faceType: str, canonicalIndices: tuple):
    """Record the parent face on one facet, mapping the facet's own nodes onto the parent face's
    canonical parametric coordinates.

    Parameters
    ----------
    facet
        The contact facet element to stamp.
    sourceElement
        The solid element the face belongs to.
    faceType
        The canonical parent face type, from :func:`canonicalParentFace`.
    canonicalIndices
        The parent face's element-local node indices in canonical order, from the same.
    """

    parametricCoords = PARENT_FACE_PARAMETRIC_COORDS[faceType]
    positionOfNode = {sourceElement.nodes[localIndex]: k for k, localIndex in enumerate(canonicalIndices)}
    try:
        vertexParametricCoords = np.array([parametricCoords[positionOfNode[node]] for node in facet.nodes])
    except KeyError as exception:
        raise ValueError(
            f"surfaceElementGenerator: facet {facet.elNumber} has a node that is not on the parent "
            f"face it was cut from (element {sourceElement.elNumber}, face type '{faceType}') -- the "
            "face tables and the canonical parent-face derivation disagree."
        ) from exception

    facet.setParentFace(faceType, [sourceElement.nodes[i] for i in canonicalIndices], vertexParametricCoords)


def _assignQuadConsistentShares(quadFacets: list):
    """Assign the consistent lumping of a uniform pressure on a bilinear quad (quad area / 4 per
    node) to the two Tria3 facets triangulating it, distributing each node's quarter evenly over
    the triangles of this quad containing it -- removing the diagonal-position dependence of the
    equal per-triangle split.

    Known limitation (not yet addressed, candidate for future investigation): the equal quad
    area / 4 split is the consistent lumping of a *uniform* pressure only for an affine
    (parallelogram) quad; for a general distorted bilinear quad, the consistent nodal shares of a
    uniform pressure are not exactly equal quarters, so this is an approximation whose accuracy
    degrades with facet distortion. No test currently quantifies this error.

    Parameters
    ----------
    quadFacets
        The two Tria3 facet elements triangulating one quad.
    """

    quadArea = sum(f.nodalAreaShares.sum() for f in quadFacets)
    facetsOfNode = {}
    for facet in quadFacets:
        for node in facet.nodes:
            facetsOfNode[node] = facetsOfNode.get(node, 0) + 1
    for facet in quadFacets:
        facet.setNodalAreaShares([quadArea / 4.0 / facetsOfNode[node] for node in facet.nodes])


def _applyModifiedSerendipityShares(cornerTriangles: list):
    """Reassign each corner node's tributary share to its two adjacent midside nodes, yielding the
    modified serendipity shape functions ``Ntilde_mid = N_mid + 1/2 (N_c1 + N_c2)``,
    ``Ntilde_corner = 0`` of Puso, Laursen & Solberg (CMAME 197, 2008).

    Why this particular weighting: for a quad8 face of area :math:`A` the consistent nodal loads of
    a uniform pressure are :math:`-pA/12` per corner and :math:`+pA/3` per midside node. A
    unilateral per-node penalty spring cannot carry the negative corner load -- a negative
    tributary area would turn that spring into an attractor -- so no admissible weighting is exact,
    and the mismatch against the consistent loads is a self-equilibrated, zero-moment corner/
    midside alternating load pattern that excites the face's quadratic surface modes (visible as
    surface undulation under contact, and not reduced by mesh refinement: it is a fixed fraction of
    the transmitted traction).

    Among all resultant-exact, non-negative weightings the mismatch per node is
    :math:`w_{\text{corner}} + A/12`, monotone in the corner share, so a *vanishing* corner share
    is the optimum. That is this weighting: it reduces the mismatch from :math:`A/8` per node (the
    default ``facetConsistent`` shares of the midside triangulation) to :math:`A/12`, i.e. by a
    third, and it is provably the best a per-node weighting can do. Removing the remaining
    :math:`A/12` requires integrating a pressure *field* against the parent face's own quadratic
    shape functions (segment-to-segment/mortar), which this formulation does not do.

    Applied per facet: the midside table lists every corner triangle as ``(m_prev, c, m_next)``, so
    local index 1 is the corner and its two adjacent midside nodes on the parent face are exactly
    local indices 0 and 2. The redistribution therefore never leaves the facet, and stays correct
    for distorted and curved faces, whose four corner triangles have unequal areas.

    Side effect: the corner nodes of the affected faces end up with a zero tributary area, i.e.
    they are no longer contact points at all. This is intended (it is what removes the corner
    over-constraint), but it also means they carry no penetration guard.

    IMPORTANT, measured: no benefit has been demonstrated for this weighting. Wherever a quadratic
    face carries a near-uniform pressure, the unilateral constraint resolves the impossible tensile
    corner load by *opening the corner gaps* rather than by carrying a mis-distributed load -- in
    the matched hexa20 patch test all nine corner nodes lift off under both weightings, so their
    tributary area is already irrelevant and this reassignment changes the interface force by
    0.007% and nothing else. It can only matter where corner nodes are genuinely pressed into the
    master (convex slave corners, strongly non-matching interfaces), which no test here covers. See
    :ref:`serendipity-liftoff` in the contact theory documentation before reaching for this option.

    Parameters
    ----------
    cornerTriangles
        The four corner triangles of one midside-triangulated quadratic face, in table order.
    """

    for facet in cornerTriangles:
        shares = facet.nodalAreaShares.copy()
        shares[0] += 0.5 * shares[1]
        shares[2] += 0.5 * shares[1]
        shares[1] = 0.0
        facet.setNodalAreaShares(shares)
        facet.setWeightTransform(_MODIFIED_CORNER_TRIANGLE_TRANSFORM)


def buildContactFacets(
    model: FEModel, surfaceName: str, prefix: str, triangulation: str, nodalWeights: str, journal
) -> tuple[str, str]:
    """(Re)generate the flat contact facet elements tiling ``surfaceName`` under ``prefix``.

    Idempotent: any facets a previous call under the same ``prefix`` created are removed first, so
    this can be re-run after ``surfaceName`` changes underneath it (e.g. an AMR refinement of the
    solid elements it tiles -- the model's ``surfaces`` entry is kept in sync with the refined child
    faces by the modifier, see :mod:`~edelweissfe.modelmodifiers.adaptivity.hadaptivity`) as well as
    at setup time. The recipe (``surfaceName``, ``prefix``, ``triangulation``) is recorded in
    ``model.contactFacetRecipes`` keyed by the generated facet element set name, so a
    :class:`~edelweissfe.models.meshdependent.MeshDependent` consumer of these facets can find its
    way back to the source surface it needs to watch via
    :meth:`~edelweissfe.models.femodel.FEModel.changesSince`.

    Parameters
    ----------
    model
        The model tree.
    surfaceName
        The name of an existing ``*surface`` definition.
    prefix
        The prefix for the generated element/node sets.
    triangulation
        The facet triangulation of higher-order element faces: 'corner' or 'midside'.
    nodalWeights
        The per-node contact weighting: 'facetConsistent' or 'serendipityOptimal'.
    journal
        The journal instance.

    Returns
    -------
    tuple[str, str]
        The generated ``(facetsSetName, nodesSetName)``.
    """

    triangulation = triangulation.lower()
    if triangulation not in ("corner", "midside"):
        raise ValueError(
            f"surfaceElementGenerator: triangulation '{triangulation}' is not supported. Use 'corner' or 'midside'."
        )

    if nodalWeights not in ("facetConsistent", "serendipityOptimal"):
        raise ValueError(
            f"surfaceElementGenerator: nodalWeights '{nodalWeights}' is not supported. Use "
            "'facetConsistent' or 'serendipityOptimal'."
        )

    if surfaceName not in model.surfaces:
        raise ValueError(f"surfaceElementGenerator: surface '{surfaceName}' is not defined.")

    if nodalWeights == "serendipityOptimal" and triangulation != "midside":
        # The corner reduction keeps no midside nodes in the facets, so there would be nothing to
        # reassign the corner shares to: every share of a quadratic face would be zeroed and the
        # surface would silently stop making contact. Refuse rather than produce that.
        offending = sorted(
            {
                sourceElement.ensightType
                for elementSet in model.surfaces[surfaceName].values()
                for sourceElement in elementSet
                if sourceElement.ensightType in _MIDSIDE_FACE_TABLES
            }
        )
        if offending:
            raise ValueError(
                f"surfaceElementGenerator: nodalWeights='serendipityOptimal' requires "
                f"triangulation='midside', but surface '{surfaceName}' has higher-order source "
                f"elements ({', '.join(offending)}) and triangulation='{triangulation}'."
            )

    if nodalWeights == "serendipityOptimal":
        # The redistribution is applied to the four corner *triangles* of a midside-triangulated
        # quadratic face, so it only ever fires for a face that tiles into six Tria3 facets. A 2D
        # quadratic element edge tiles into two Line2 facets instead and reaches neither branch, so
        # without this the option would pass validation and then do nothing at all. Refuse: a
        # silent no-op is the worst of the three possible behaviours. It is also not merely
        # unimplemented -- a quadratic edge's consistent weights are Simpson's (L/6, 2L/3, L/6),
        # all non-negative, so there the mismatch is removable exactly and a corner-share
        # reassignment is the wrong instrument for it in the first place.
        unsupported = set()
        for elementSet in model.surfaces[surfaceName].values():
            for sourceElement in elementSet:
                midsideTable = _MIDSIDE_FACE_TABLES.get(sourceElement.ensightType)
                if midsideTable is None:
                    continue
                anyFaceGroups = next(iter(midsideTable.values()))
                if len(anyFaceGroups[0]) == 2:
                    unsupported.add(sourceElement.ensightType)
        unsupported = sorted(unsupported)
        if unsupported:
            raise ValueError(
                f"surfaceElementGenerator: nodalWeights='serendipityOptimal' is not available for "
                f"2D higher-order element edges, but surface '{surfaceName}' has source elements "
                f"({', '.join(unsupported)}) whose faces are Line2 facets. Use the default "
                "nodalWeights='facetConsistent'."
            )

    facetsSetName = f"{prefix}_facets"
    nodesSetName = f"{prefix}_nodes"

    # remove any facets a previous call under this prefix created, so re-running is idempotent
    for staleFacet in model.elementSets.get(facetsSetName, []):
        if staleFacet.elNumber in model.elements:
            model.removeElement(staleFacet.elNumber)

    surfaceDef = model.surfaces[surfaceName]

    # Facet numbers come from the model's monotonic allocator, NOT from max(model.elements)+1. The
    # old expression read the maximum *after* the stale facets above were deleted, so a rebuild
    # handed the dead facets' numbers straight back out -- making element numbering a function of
    # the deletion history, which a restart replay cannot reproduce. See
    # FEModel.reserveElementNumbers.
    newElements = {}

    for faceNumber, elementSet in surfaceDef.items():
        for sourceElement in elementSet:
            faceTable = None
            if triangulation == "midside":
                faceTable = _MIDSIDE_FACE_TABLES.get(sourceElement.ensightType)
            if faceTable is None:
                faceTable = _FACE_TABLES.get(sourceElement.ensightType)
            if faceTable is None:
                raise ValueError(
                    f"surfaceElementGenerator: no face-node-ordering table available for element "
                    f"type '{sourceElement.ensightType}' (element {sourceElement.elNumber})."
                )

            faceNodeGroups = faceTable.get(faceNumber)
            if faceNodeGroups is None:
                raise ValueError(
                    f"surfaceElementGenerator: face {faceNumber} is not defined for element type "
                    f"'{sourceElement.ensightType}' (element {sourceElement.elNumber})."
                )

            parentFaceType, parentFaceIndices = canonicalParentFace(sourceElement.ensightType, faceNumber)

            faceFacets = []
            for localIndices in faceNodeGroups:
                facetNodes = [sourceElement.nodes[i] for i in localIndices]

                if len(localIndices) == 3:
                    facetElementType, facetClass = "Tria3ContactFacet", Tria3ContactFacet
                elif len(localIndices) == 2:
                    facetElementType, facetClass = "Line2ContactFacet", Line2ContactFacet
                else:
                    raise ValueError(
                        f"surfaceElementGenerator: unsupported face-node-group size "
                        f"{len(localIndices)} for element type '{sourceElement.ensightType}'."
                    )

                (elNumber,) = model.reserveElementNumbers(1)
                facetElement = facetClass(facetElementType, elNumber)
                facetElement.setNodes(facetNodes)
                facetElement.initializeElement()
                _stampParentFace(facetElement, sourceElement, parentFaceType, parentFaceIndices)

                faceFacets.append(facetElement)
                newElements[elNumber] = facetElement

            if len(faceFacets) == 2 and all(len(f.nodes) == 3 for f in faceFacets):
                # Two Tria3 from a linear quad face (fixed diagonal split): the per-triangle
                # equal split (measure/3 each) would give diagonal-position-dependent nodal
                # tributary areas, inconsistent with the unique lumping of a uniform pressure on
                # a bilinear quad face (face area / 4 per corner) -- the resulting force-vs-area
                # mismatch shows up as spurious contact pressure oscillation in an otherwise
                # exact patch test. Override: distribute each corner's area/4 evenly over the
                # triangles of THIS face containing it.
                _assignQuadConsistentShares(faceFacets)

            elif len(faceFacets) == 6 and all(len(f.nodes) == 3 for f in faceFacets):
                # Midside triangulation of a quadratic face: 4 corner triangles followed by the
                # 2 triangles of the central midside quad (table order). The central quad's fixed
                # diagonal would give its two diagonal midside nodes more incident triangles than
                # the other two -- asymmetric tributary areas on a symmetric face. Apply the same
                # quad-consistent lumping (area/4 per node) to the central quad; the corner
                # triangles keep their equal per-triangle split. NOTE: exact pointwise pressure
                # consistency is fundamentally unattainable for serendipity faces regardless of
                # the weights -- the consistent nodal forces of a uniform pressure on a quad8
                # face are NEGATIVE at the corners, which no unilateral per-node spring scheme
                # can reproduce (see the constraint documentation). nodalWeights=
                # 'serendipityOptimal' minimises, but cannot remove, that mismatch.
                _assignQuadConsistentShares(faceFacets[4:])
                if nodalWeights == "serendipityOptimal":
                    _applyModifiedSerendipityShares(faceFacets[:4])

    for facetElement in newElements.values():
        model.createElement(facetElement)

    # this function is the one that mutates model.elements outside the mesh modifier (removing the
    # stale facets above and inserting newElements here), so model.elementSets["all"] must be
    # resynced here to mirror model.elements -- otherwise "all" keeps dangling references to the
    # popped stale facets and misses the new ones for the rest of the refinement window.
    if "all" in model.elementSets:
        model.elementSets["all"].replaceMembers(list(model.elements.values()))

    # stable identity across rebuilds (mutate in place rather than replace under the same key), like
    # every other AMR-mutated topological container -- a consumer that merely caches
    # model.elementSets[facetsSetName]/model.nodeSets[nodesSetName] (e.g. a fromExpression
    # FieldOutput reading a contact constraint's per-facet-node result) would otherwise keep
    # referencing the pre-rebuild object and silently go stale/size-mismatched on the next rebuild.
    if facetsSetName in model.elementSets:
        model.elementSets[facetsSetName].replaceMembers(list(newElements.values()))
    else:
        model.elementSets[facetsSetName] = ElementSet(facetsSetName, list(newElements.values()))

    seenNodes = set()
    facetNodesInOrder = []
    for facetElement in newElements.values():
        for node in facetElement.nodes:
            if node not in seenNodes:
                seenNodes.add(node)
                facetNodesInOrder.append(node)

    if nodesSetName in model.nodeSets:
        model.nodeSets[nodesSetName].replaceMembers(facetNodesInOrder)
    else:
        model.nodeSets[nodesSetName] = NodeSet(nodesSetName, facetNodesInOrder)
    model.contactFacetRecipes[facetsSetName] = (surfaceName, prefix, triangulation, nodalWeights)

    journal.message(
        f"generated {len(newElements)} contact facet element(s) from surface '{surfaceName}' "
        f"into element set '{facetsSetName}'",
        "surfaceElementGenerator",
        1,
    )

    return facetsSetName, nodesSetName


@dataclass(frozen=True)
class SurfaceElementGeneratorSchema:
    """The options this generator accepts, owned by this module and never mutated from outside
    it.

    Unlike every other generator, ``name`` here is a *required scalar option* (the facet-set name
    prefix), independent of the ``*modelGenerator`` keyword's own top-line ``name`` argument (which
    this generator, uniquely among the built-ins, never reads).
    """

    surface: str | None = schemaField(
        description="The name of an existing *surface definition.", dtype=str, default=None, required=True
    )
    name: str | None = schemaField(
        description="The prefix for the generated element/node sets.", dtype=str, default=None, required=True
    )
    triangulation: str = schemaField(
        description="The facet triangulation of higher-order element faces: 'corner' (linear "
        "corner-node subset only; exact for straight-edged meshes) or 'midside' (triangulation of "
        "the full face boundary including midside nodes; strictly more accurate for curved "
        "faces). Linear element faces are unaffected by this option.",
        dtype=str,
        default="corner",
    )
    nodalWeights: str = schemaField(
        description="The per-node contact weighting of the generated facets: 'facetConsistent' "
        "(the consistent nodal loads of the flat facet tiling itself) or 'serendipityOptimal' "
        "(the resultant-exact, non-negative weighting that minimises the mismatch against a "
        "quadratic face's own consistent nodal loads, by reassigning each corner node's share to "
        "its two adjacent midside nodes -- the modified shape functions of Puso, Laursen & "
        "Solberg, CMAME 2008). Reduces the spurious corner/midside load pattern on hexa20 "
        "contact surfaces by a third, at the price of corner nodes ceasing to be contact points -- "
        "but NO measured benefit has been demonstrated, because a quadratic face under near-uniform "
        "pressure lifts its corner nodes off anyway, which makes their weight moot; see the contact "
        "theory documentation. Requires triangulation='midside'; linear element faces are "
        "unaffected by this option.",
        dtype=str,
        default="facetConsistent",
    )


class Generator(GeneratorBase):
    """Generates flat, geometry-only "contact facet" elements
    (:class:`~edelweissfe.elements.contactsurfaceelement.Tria3ContactFacet` /
    ``Line2ContactFacet``) from an existing ``*surface`` definition. See the module docstring for
    the full background.
    """

    #: Option schema for this generator, per OptionSchemaProvider.
    schema = SurfaceElementGeneratorSchema

    def __init__(
        self,
        name: str,
        model: FEModel,
        journal: Journal,
        *,
        configuration: SurfaceElementGeneratorSchema = SurfaceElementGeneratorSchema(),
    ):
        """Constructible standalone, with no parser involvement.
        Populates ``model`` directly (via :func:`buildContactFacets`); construction *is* the
        generation.

        Parameters
        ----------
        name
            Unused: this generator's set-name prefix is ``configuration.name``, a required scalar
            option distinct from the ``*modelGenerator`` keyword's own top-line ``name``.
        model
            The model tree to populate. Mutated in place.
        journal
            The journal instance.
        configuration
            The options this generator accepts; ``surface``/``name`` are still required, see
            :class:`SurfaceElementGeneratorSchema`.
        """
        buildContactFacets(
            model,
            configuration.surface,
            configuration.name,
            configuration.triangulation,
            configuration.nodalWeights,
            journal,
        )
