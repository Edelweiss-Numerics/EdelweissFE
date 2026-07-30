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
face-consistent per-node tributary area shares for the pressure-weighted contact formulation.
See the :doc:`contact theory documentation </documentation/contacttheory>` for the full
background.

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

from edelweissfe.elements.contactsurfaceelement import (
    Line2ContactFacet,
    Tria3ContactFacet,
)
from edelweissfe.generators.base.generatorbase import GeneratorBase
from edelweissfe.journal.journal import Journal
from edelweissfe.models.femodel import FEModel
from edelweissfe.sets.elementset import ElementSet
from edelweissfe.sets.nodeset import NodeSet
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


def buildContactFacets(model: FEModel, surfaceName: str, prefix: str, triangulation: str, journal) -> tuple[str, str]:
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

    if surfaceName not in model.surfaces:
        raise ValueError(f"surfaceElementGenerator: surface '{surfaceName}' is not defined.")

    facetsSetName = f"{prefix}_facets"
    nodesSetName = f"{prefix}_nodes"

    # remove any facets a previous call under this prefix created, so re-running is idempotent
    for staleFacet in model.elementSets.get(facetsSetName, []):
        model.elements.pop(staleFacet.elNumber, None)

    surfaceDef = model.surfaces[surfaceName]

    nextElNumber = max(model.elements.keys(), default=0) + 1
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

                facetElement = facetClass(facetElementType, nextElNumber)
                facetElement.setNodes(facetNodes)
                facetElement.initializeElement()

                faceFacets.append(facetElement)
                newElements[nextElNumber] = facetElement
                nextElNumber += 1

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
                # can reproduce (see the constraint documentation).
                _assignQuadConsistentShares(faceFacets[4:])

    model.elements.update(newElements)

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
    model.contactFacetRecipes[facetsSetName] = (surfaceName, prefix, triangulation)

    journal.message(
        f"generated {len(newElements)} contact facet element(s) from surface '{surfaceName}' "
        f"into element set '{facetsSetName}'",
        "surfaceElementGenerator",
        1,
    )

    return facetsSetName, nodesSetName


@dataclass(frozen=True)
class SurfaceElementGeneratorSchema:
    """L2: the options this generator accepts, owned by this module and never mutated from
    outside it.

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


class Generator(GeneratorBase):
    """Generates flat, geometry-only "contact facet" elements
    (:class:`~edelweissfe.elements.contactsurfaceelement.Tria3ContactFacet` /
    ``Line2ContactFacet``) from an existing ``*surface`` definition. See the module docstring for
    the full background.
    """

    #: L2 schema declared for the L3 registry, per OptionSchemaProvider.
    schema = SurfaceElementGeneratorSchema

    def __init__(
        self,
        name: str,
        model: FEModel,
        journal: Journal,
        *,
        configuration: SurfaceElementGeneratorSchema = SurfaceElementGeneratorSchema(),
    ):
        """L1: constructible standalone, with no parser involvement.
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
        buildContactFacets(model, configuration.surface, configuration.name, configuration.triangulation, journal)
