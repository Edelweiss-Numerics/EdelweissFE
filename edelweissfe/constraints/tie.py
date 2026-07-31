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
from edelweissfe.sets.nodeset import NodeSet
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
    "Whether a slave node is tied at all: nodes whose reference-configuration closest-point "
    "distance to the master surface exceeds this tolerance are left untied (recorded in the "
    "constraint's untiedSlaveNodes), matching Abaqus' *TIE default behavior of silently dropping "
    "out-of-range slave nodes. Applies independently of adjust -- whether a tied node's "
    "coordinates additionally get snapped onto the master is a separate decision, see "
    "adjust/adjustTolerance. If not given (the default), a tolerance is computed as "
    "positionToleranceFactor times the master surface's characteristic facet size -- see "
    "positionToleranceFactor. Set this explicitly to an absolute distance to override that.",
    float,
    None,
)
module.addOptionalArg(
    "positionToleranceFactor",
    "Used only when positionTolerance is not given: the default tolerance is this fraction of the "
    "master surface's characteristic (mean, over all its facets) edge length, computed once and "
    "used for every slave node. 0.25 comfortably exceeds the sub-percent gaps expected between two "
    "compatible discretizations of the same surface (mismatched density, curvature/interpolation "
    "error), while remaining well below the facet-size-or-larger gaps that indicate the surfaces "
    "don't actually correspond (e.g. a slave surface extending beyond the master surface's actual "
    "extent -- a partial-bond-length or otherwise partially-overlapping pair of surfaces).",
    float,
    0.25,
)
module.addOptionalArg(
    "adjust",
    "Whether a TIED node's coordinates additionally get snapped onto its projected master point "
    "at construction (Abaqus-like default) -- a separate decision from whether the node is tied "
    "at all (see positionTolerance). If False, no tied node is ever snapped: any initial geometric "
    "gap is preserved rigidly (the displacements are tied, not the positions), regardless of size. "
    "If True, a tied node is snapped only if its distance is also within adjustTolerance (default: "
    "unconditionally, i.e. every tied node is snapped, matching plain Abaqus ADJUST=YES) -- see "
    "adjustTolerance to snap away only small, effectively-numerical gaps while still tying "
    "(without snapping) across larger, deliberate ones. Note that adjusting modifies the nodal "
    "coordinates before the element geometry is initialized; avoid adjusting nodes that also "
    "belong to an already-generated contact surface of another constraint.",
    str,
    "True",
)
module.addOptionalArg(
    "adjustTolerance",
    "Used only when adjust=True: a tied node's coordinates are snapped onto the master only if "
    "its closest-point distance is also within this tolerance; beyond it, the node stays tied "
    "(kinematically) but its position is left as found, preserving the gap. Independent of "
    "positionTolerance -- a node can be tied across a fairly generous distance while only "
    "genuinely small (e.g. sub-percent, mesh-discretization-scale) gaps within that get snapped "
    "away. If not given (the default), any tied node is snapped, matching plain Abaqus ADJUST=YES "
    "and this constraint's behavior before this option existed.",
    float,
    None,
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
        adjustTolerance = kwargs["adjustTolerance"]

        closestPointFunction = tria3ClosestPoint if self.nDim == 3 else line2ClosestPoint
        masterFacetCoords = [np.array([n.coordinates for n in el.nodes]) for el in masterFacetElements]

        # Tie MEMBERSHIP (is this node tied at all) is independent of adjust (does a tied node get
        # snapped) -- these are separate decisions, see the adjust arg's docstring. A single
        # tolerance, computed once from the whole master surface's characteristic facet size, is
        # used for every slave node (matching Abaqus' *TIE, which always enforces some tolerance --
        # explicit or internally computed -- and never ties unconditionally regardless of distance).
        membershipTolerance = (
            positionTolerance
            if positionTolerance is not None
            else positionToleranceFactor * np.mean([_facetCharacteristicSize(c) for c in masterFacetCoords])
        )

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

            if bestDistance > membershipTolerance:
                self.untiedSlaveNodes.append(slaveNode)
                continue

            # Snapping is the separate, independent decision: a tied node is only snapped if adjust
            # is requested AND (when given) its distance is also within adjustTolerance -- a node
            # beyond adjustTolerance stays tied, just not snapped, preserving its gap exactly.
            withinAdjustTolerance = adjustTolerance is None or bestDistance <= adjustTolerance
            if adjust and withinAdjustTolerance and bestDistance > 0.0:
                slaveNode.coordinates[:] = bestWeights @ masterFacetCoords[bestFacetIdx]

            self.tiedRecords.append((slaveNode, masterFacetElements[bestFacetIdx].nodes, bestWeights))

        # Exposed as ordinary node sets so the tied/untied split is directly inspectable -- e.g. via
        # *fieldOutput, or for free in Ensight, which automatically creates a part for every node
        # set in the model (edelweissfe/outputmanagers/ensight.py's _createGeometryParts), no
        # fieldOutput required just to see where these nodes are.
        model.nodeSets[f"{name}_tied"] = NodeSet(f"{name}_tied", [record[0] for record in self.tiedRecords])
        model.nodeSets[f"{name}_untied"] = NodeSet(f"{name}_untied", self.untiedSlaveNodes)

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
