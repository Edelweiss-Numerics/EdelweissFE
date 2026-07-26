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
from edelweissfe.generators.surfaceelementgenerator import buildContactFacets
from edelweissfe.journal.journal import Journal
from edelweissfe.models.femodel import FEModel
from edelweissfe.models.meshdependent import MeshDependent
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

This constraint is a :class:`~edelweissfe.models.meshdependent.MeshDependent`: if either surface's
source solid elements are refined mid-run (e.g. by :mod:`~edelweissfe.modelmodifiers.adaptivity.
hadaptivity`), it regenerates that side's facets and re-projects the tied records -- no separate
wiring needed. Unlike the penalty contact constraint, a tie has no per-increment tick of its own
that runs *before* the DofManager/system matrix is rebuilt (its only hook,
:meth:`getMultiPointConstraints`, is called from inside that rebuild -- too late to safely swap in
freshly regenerated facet elements), so it reconciles via the model's push notification instead,
synchronously as part of the mesh-mutating modifier's own call. The re-projection never re-adjusts
slave node coordinates regardless of the ``adjust`` setting: that snap is a setup-time convenience
for removing an initial geometric gap, not something to repeat on an already-loaded, already-tied
node.
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
    "If given, slave nodes whose reference-configuration closest-point distance to the master "
    "surface exceeds this tolerance are left untied (recorded in the constraint's "
    "untiedSlaveNodes). If not given, every slave node is tied unconditionally.",
    float,
    None,
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


class Constraint(MultiPointConstraintBase, MeshDependent):
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
    def __init__(self, name: str, model: FEModel, **kwargs):
        kwargs = CaseInsensitiveDict(kwargs)

        self.name = name
        self.nDim = model.domainSize

        self._journal = Journal()
        self._slaveSurfaceSetName = kwargs["slaveSurface"]
        self._masterSurfaceSetName = kwargs["masterSurface"]
        self._positionTolerance = kwargs["positionTolerance"]

        slaveFacetElements = list(model.elementSets[self._slaveSurfaceSetName])
        masterFacetElements = list(model.elementSets[self._masterSurfaceSetName])

        masterNodes = {node for el in masterFacetElements for node in el.nodes}
        slaveNodesForCheck = {node for el in slaveFacetElements for node in el.nodes}
        if not masterNodes.isdisjoint(slaveNodesForCheck):
            raise ValueError(
                f"Constraint '{name}': slave surface '{kwargs['slaveSurface']}' and master surface "
                f"'{kwargs['masterSurface']}' share nodes -- a node cannot be tied to itself."
            )

        self.tiedRecords, self.untiedSlaveNodes = self._buildTiedRecords(
            slaveFacetElements, masterFacetElements, adjust=strtobool(kwargs["adjust"])
        )

        # A tie has no per-increment tick of its own that runs before the DofManager/VIJSystemMatrix
        # rebuild -- getMultiPointConstraints() is only called from inside that rebuild, too late to
        # safely swap in newly regenerated facet elements (the just-built system wouldn't know about
        # them). So, unlike nodeToDeformableSurfacePenalty, a tie reconciles via the push escape
        # hatch: model.notifyModelChanged() calls onModelChanged() synchronously, from inside the
        # model modifier's own updateModel(), strictly before the rebuild decision is even made.
        model.registerObserver(self)

    def onModelChanged(self, model: FEModel, changeType, change) -> None:
        if change is not None:
            self.reconcile(model, change)

    def _buildTiedRecords(self, slaveFacetElements, masterFacetElements, adjust: bool):
        """Project every unique slave-surface node onto its closest master facet (reference
        configuration) and freeze the resulting clamped weights. With ``adjust``, additionally snap
        each tied node onto its projected point, removing any initial geometric gap -- a setup-time
        convenience, never applied on a reconcile-triggered re-projection."""

        slaveNodes = list(dict.fromkeys(node for el in slaveFacetElements for node in el.nodes))

        closestPointFunction = tria3ClosestPoint if self.nDim == 3 else line2ClosestPoint
        masterFacetCoords = [np.array([n.coordinates for n in el.nodes]) for el in masterFacetElements]

        tiedRecords = []
        untiedSlaveNodes = []

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

            if self._positionTolerance is not None and bestDistance > self._positionTolerance:
                untiedSlaveNodes.append(slaveNode)
                continue

            if adjust and bestDistance > 0.0:
                slaveNode.coordinates[:] = bestWeights @ masterFacetCoords[bestFacetIdx]

            tiedRecords.append((slaveNode, masterFacetElements[bestFacetIdx].nodes, bestWeights))

        return tiedRecords, untiedSlaveNodes

    def reconcile(self, model: FEModel, change) -> bool:
        """Regenerate whichever side's facets were affected by ``change`` (via its recorded
        :attr:`~edelweissfe.models.femodel.FEModel.contactFacetRecipes`) and re-project the tied
        records from scratch against the rebuilt surfaces -- never adjusting coordinates, since the
        tied nodes are already loaded mid-run."""

        slaveRecipe = model.contactFacetRecipes.get(self._slaveSurfaceSetName)
        masterRecipe = model.contactFacetRecipes.get(self._masterSurfaceSetName)
        touchedSlave = slaveRecipe is not None and change.touchesSurface(slaveRecipe[0])
        touchedMaster = masterRecipe is not None and change.touchesSurface(masterRecipe[0])
        if not (touchedSlave or touchedMaster):
            return False

        if touchedSlave:
            buildContactFacets(model, *slaveRecipe, self._journal)
        if touchedMaster:
            buildContactFacets(model, *masterRecipe, self._journal)

        slaveFacetElements = list(model.elementSets[self._slaveSurfaceSetName])
        masterFacetElements = list(model.elementSets[self._masterSurfaceSetName])
        self.tiedRecords, self.untiedSlaveNodes = self._buildTiedRecords(
            slaveFacetElements, masterFacetElements, adjust=False
        )
        return True

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
