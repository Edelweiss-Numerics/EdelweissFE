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

import numpy as np

from edelweissfe.constraints.base.multipointconstraintbase import (
    MultiPointConstraintBase,
)
from edelweissfe.models.femodel import FEModel
from edelweissfe.utils.caseinsensitivedict import CaseInsensitiveDict
from edelweissfe.utils.facetcontactgeometry import line2ClosestPoint, tria3ClosestPoint
from edelweissfe.utils.inputlanguage import InputLanguage, Module
from edelweissfe.utils.misc import (
    caseInsensitiveKwargsChecker,
    castKwargsValuesAndAddDefaults,
    strtobool,
)

"""
An Abaqus-style surface-to-surface tie constraint, bonding the nodes of a slave surface rigidly to
a deformable master surface via master-slave DOF elimination (multi-point constraint
condensation). Both surfaces are represented by flat contact facet elements
(:mod:`~edelweissfe.elements.contactsurfaceelement`, typically created via
:mod:`~edelweissfe.generators.surfaceelementgenerator`). Each slave node is projected onto its
closest master facet in the reference configuration; the clamped closest-point weights are frozen
and every displacement component of the slave node is constrained to the identically-weighted
master interpolation. The constraint is enforced exactly -- no penalty parameter, no Lagrange
multiplier DOFs -- and adds zero stiffness, which in particular leaves the critical time step of
explicit dynamics untouched.
"""

module = Module(
    "tie",
    "An Abaqus-style tie constraint, bonding a slave surface rigidly to a deformable master "
    "surface via master-slave DOF elimination.",
)

inputLanguage = InputLanguage()

keyword = "constraint"
if keyword in inputLanguage:
    inputLanguage[keyword].addModule(module)

module.addRequiredArg(
    "slaveSurface",
    "The element set of contact facet elements (Tria3ContactFacet/Line2ContactFacet) forming the "
    "slave surface; its nodes are tied. For quadratic (hexa20/quad8) faces, generate the facets "
    "with triangulation=midside on BOTH surfaces -- the corner triangulation excludes the midside "
    "nodes from the facet node list entirely, leaving them untied.",
    str,
)
module.addRequiredArg(
    "masterSurface",
    "The element set of contact facet elements (Tria3ContactFacet/Line2ContactFacet) forming the " "master surface.",
    str,
)
module.addOptionalArg(
    "positionTolerance",
    "Slave nodes whose reference-configuration closest-point distance to the master surface "
    "exceeds this tolerance are left untied (recorded in the constraint's untiedSlaveNodes), "
    "matching Abaqus' *TIE default behavior of silently dropping out-of-range slave nodes. If not "
    "given (the default), a tolerance is computed per slave node as positionToleranceFactor times "
    "the characteristic edge length of that node's closest master facet -- see "
    "positionToleranceFactor. Set this explicitly to an absolute distance to override that.",
    float,
    None,
)
module.addOptionalArg(
    "positionToleranceFactor",
    "Used only when positionTolerance is not given: the default tolerance for a slave node is this "
    "fraction of its closest master facet's own characteristic (mean) edge length, so it scales "
    "with local mesh density instead of being one fixed number for the whole surface. 0.25 "
    "comfortably exceeds the sub-percent gaps expected between two compatible discretizations of "
    "the same surface (mismatched density, curvature/interpolation error), while remaining well "
    "below the facet-size-or-larger gaps that indicate the surfaces don't actually correspond (e.g. "
    "a slave surface extending beyond the master surface's actual extent -- a partial-bond-length "
    "or otherwise partially-overlapping pair of surfaces).",
    float,
    0.25,
)
module.addOptionalArg(
    "adjust",
    "Move each tied slave node onto its closest master point at construction (Abaqus-like "
    "default). If False, any initial geometric gap between the surfaces is preserved rigidly "
    "(the displacements are tied, not the positions). Note that adjusting modifies the nodal "
    "coordinates before the element geometry is initialized; avoid adjusting nodes that also "
    "belong to an already-generated contact surface of another constraint.",
    str,
    "True",
)

documentation = [module]


def _facetCharacteristicSize(facetCoords: np.ndarray) -> float:
    """Mean edge length of a facet (Line2: its one edge; Tria3: its three edges) -- the local
    length scale used for the default position tolerance when none is given explicitly.

    Parameters
    ----------
    facetCoords
        The facet's node coordinates, shape (2, nDim) for a Line2 or (3, nDim) for a Tria3.

    Returns
    -------
    float
        The facet's mean edge length.
    """

    nNodes = facetCoords.shape[0]
    edgeLengths = [np.linalg.norm(facetCoords[i] - facetCoords[(i + 1) % nNodes]) for i in range(nNodes)]
    return np.mean(edgeLengths)


class Constraint(MultiPointConstraintBase):
    """
    An Abaqus-style surface-to-surface tie constraint via master-slave DOF elimination.

    Theoretical background
    -----------------------
    Each slave node is projected onto its closest master facet once, in the reference
    configuration, using the clamped closest-point search shared with the small-sliding contact
    formulation. The resulting non-negative facet weights :math:`N_a` are frozen, and every
    displacement component of the slave node is constrained linearly to the master facet nodes:

    .. math::
        u_s = \\sum_a N_a \\, u_{m_a}.

    The constraint records are collected by the solver and enforced by condensing the slave DOFs
    out of the equation system (see
    :class:`~edelweissfe.numerics.mpctransformation.MultiPointConstraintTransformation`) -- the
    identical mechanism serves the implicit solvers (system matrix transformation) and explicit
    dynamics (mass/force folding onto the masters, direct kinematic slaving). The enforcement is
    exact for arbitrary meshes; the interpolation across a master facet is the facet's own linear
    one, so a linear displacement field is transferred exactly (patch test) for matching and
    non-matching meshes alike, on straight-edged hexa20 faces included.

    Currently only the 'displacement' field is tied. Available for spatialdomain = 3D (Tria3
    facets) and 2D (Line2 facets).
    """

    @caseInsensitiveKwargsChecker([kw.name for kw in module.requiredArgs], [kw.name for kw in module.optionalArgs])
    @castKwargsValuesAndAddDefaults(module)
    def __init__(self, name: str, model: FEModel, *args, **kwargs):
        kwargs = CaseInsensitiveDict(kwargs)

        self.name = name
        self.nDim = model.domainSize

        slaveFacetElements = list(model.elementSets[kwargs["slaveSurface"]])
        masterFacetElements = list(model.elementSets[kwargs["masterSurface"]])

        # the tied points are the unique nodes of the slave surface, in first-seen order
        slaveNodes = list(dict.fromkeys(node for el in slaveFacetElements for node in el.nodes))

        masterNodes = {node for el in masterFacetElements for node in el.nodes}
        if not masterNodes.isdisjoint(slaveNodes):
            raise ValueError(
                f"Constraint '{name}': slave surface '{kwargs['slaveSurface']}' and master surface "
                f"'{kwargs['masterSurface']}' share nodes -- a node cannot be tied to itself."
            )

        positionTolerance = kwargs["positionTolerance"]
        positionToleranceFactor = kwargs["positionToleranceFactor"]
        adjust = strtobool(kwargs["adjust"])

        closestPointFunction = tria3ClosestPoint if self.nDim == 3 else line2ClosestPoint
        masterFacetCoords = [np.array([n.coordinates for n in el.nodes]) for el in masterFacetElements]
        masterFacetCharacteristicSizes = [_facetCharacteristicSize(coords) for coords in masterFacetCoords]

        #: Tied records: (slaveNode, masterNodes of the assigned facet, frozen weights).
        self.tiedRecords = []
        #: Slave nodes beyond positionTolerance (explicit or computed default), left untied.
        self.untiedSlaveNodes = []

        for slaveNode in slaveNodes:
            bestWeights = None
            bestFacetIdx = None
            bestDistance = np.inf

            for facetIdx, facetCoords in enumerate(masterFacetCoords):
                weights, distance = closestPointFunction(slaveNode.coordinates, *facetCoords)
                if distance < bestDistance:
                    bestDistance = distance
                    bestWeights = weights
                    bestFacetIdx = facetIdx

            # Abaqus' *TIE always enforces some tolerance (an explicit POSITION TOLERANCE, or an
            # internally computed default) and silently drops slave nodes outside it -- it never
            # ties unconditionally regardless of distance. Mirror that: fall back to a tolerance
            # scaled by the winning facet's own size when none is given explicitly, instead of
            # skipping the check entirely.
            effectiveTolerance = (
                positionTolerance
                if positionTolerance is not None
                else positionToleranceFactor * masterFacetCharacteristicSizes[bestFacetIdx]
            )
            if bestDistance > effectiveTolerance:
                self.untiedSlaveNodes.append(slaveNode)
                continue

            if adjust and bestDistance > 0.0:
                slaveNode.coordinates[:] = bestWeights @ masterFacetCoords[bestFacetIdx]

            self.tiedRecords.append((slaveNode, masterFacetElements[bestFacetIdx].nodes, bestWeights))

    def getMultiPointConstraints(self, dofManager) -> list[tuple[int, list[tuple[int, float]]]]:
        fieldVariableIndices = dofManager.idcsOfFieldVariablesInDofVector

        records = []
        for slaveNode, masterNodes, weights in self.tiedRecords:
            slaveDofIndices = fieldVariableIndices[slaveNode.fields["displacement"]]
            masterDofIndices = [fieldVariableIndices[node.fields["displacement"]] for node in masterNodes]

            for component in range(self.nDim):
                records.append(
                    (
                        slaveDofIndices[component],
                        [
                            (masterDofIdcs[component], weight)
                            for masterDofIdcs, weight in zip(masterDofIndices, weights)
                            if weight != 0.0
                        ],
                    )
                )

        return records
