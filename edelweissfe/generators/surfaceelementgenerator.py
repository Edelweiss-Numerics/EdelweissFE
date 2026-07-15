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
definition, for use as the master side of node-to-deformable-surface penalty contact. Quad faces
of 3D solids are split into two Tria3 facets via a fixed diagonal; higher-order element faces are
reduced to their linear corner-node subset -- both are standard, accepted contact-mechanics
simplifications, not approximations specific to this generator.

.. code-block:: edelweiss
    :caption: Example

    *modelGenerator, generator=surfaceElementGenerator, name=gen
        surface = mySurface
        name    = myContactSurface
"""

from edelweissfe.elements.contactsurfaceelement import (
    Line2ContactFacet,
    Tria3ContactFacet,
)
from edelweissfe.models.femodel import FEModel
from edelweissfe.sets.elementset import ElementSet
from edelweissfe.sets.nodeset import NodeSet
from edelweissfe.utils.caseinsensitivedict import CaseInsensitiveDict
from edelweissfe.utils.inputlanguage import InputLanguage, Module
from edelweissfe.utils.misc import (
    caseInsensitiveKwargsChecker,
    castKwargsValuesAndAddDefaults,
)

module = Module(
    "surfaceElementGenerator",
    "Generates flat contact facet elements (Tria3ContactFacet/Line2ContactFacet) from an "
    "existing *surface definition.",
)

inputLanguage = InputLanguage()

keyword = "modelGenerator"
if keyword in inputLanguage:
    inputLanguage[keyword].addModule(module)

module.addRequiredArg("surface", "The name of an existing *surface definition.", str)
module.addRequiredArg("name", "The prefix for the generated element/node sets.", str)

documentation = [module]

# Abaqus-style face-node-ordering tables, 0-indexed, reduced to each element type's linear corner
# nodes. Each face maps to a tuple of node-index groups: a 3-tuple is a Tria3 facet, a 2-tuple is
# a Line2 facet. Quad faces of 3D solids are split into two Tria3 facets via the (0,2) diagonal,
# preserving the source face's winding (and thus outward orientation).
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
        1: ((0, 1, 2), (0, 2, 3)),
        2: ((4, 7, 6), (4, 6, 5)),
        3: ((0, 4, 5), (0, 5, 1)),
        4: ((1, 5, 6), (1, 6, 2)),
        5: ((2, 6, 7), (2, 7, 3)),
        6: ((3, 7, 4), (3, 4, 0)),
    },
    "hexa20": {
        1: ((0, 1, 2), (0, 2, 3)),
        2: ((4, 7, 6), (4, 6, 5)),
        3: ((0, 4, 5), (0, 5, 1)),
        4: ((1, 5, 6), (1, 6, 2)),
        5: ((2, 6, 7), (2, 7, 3)),
        6: ((3, 7, 4), (3, 4, 0)),
    },
}


@caseInsensitiveKwargsChecker([kw.name for kw in module.requiredArgs], [kw.name for kw in module.optionalArgs])
@castKwargsValuesAndAddDefaults(module)
def generateModelData(generatorDefinition: dict, model: FEModel, journal, *args, **kwargs) -> FEModel:
    """Generate contact facet elements from an existing ``*surface`` definition.

    Parameters
    ----------
    generatorDefinition
        The generator definition dict.
    model
        The model tree.
    journal
        The journal instance.

    Returns
    -------
    FEModel
        The updated model tree.
    """

    kwargs = CaseInsensitiveDict(kwargs)

    surfaceName = kwargs["surface"]
    prefix = kwargs["name"]

    if surfaceName not in model.surfaces:
        raise ValueError(f"surfaceElementGenerator: surface '{surfaceName}' is not defined.")

    surfaceDef = model.surfaces[surfaceName]

    nextElNumber = max(model.elements.keys(), default=0) + 1
    newElements = {}

    for faceNumber, elementSet in surfaceDef.items():
        for sourceElement in elementSet:
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

                newElements[nextElNumber] = facetElement
                nextElNumber += 1

    model.elements.update(newElements)

    facetsSetName = f"{prefix}_facets"
    model.elementSets[facetsSetName] = ElementSet(facetsSetName, list(newElements.values()))

    seenNodes = set()
    facetNodesInOrder = []
    for facetElement in newElements.values():
        for node in facetElement.nodes:
            if node not in seenNodes:
                seenNodes.add(node)
                facetNodesInOrder.append(node)

    nodesSetName = f"{prefix}_nodes"
    model.nodeSets[nodesSetName] = NodeSet(nodesSetName, facetNodesInOrder)

    journal.message(
        f"generated {len(newElements)} contact facet element(s) from surface '{surfaceName}' "
        f"into element set '{facetsSetName}'",
        "surfaceElementGenerator",
        1,
    )

    return model
