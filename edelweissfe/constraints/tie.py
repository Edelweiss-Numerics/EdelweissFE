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

from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree

from edelweissfe.constraints.base.multipointconstraintbase import (
    MultiPointConstraintBase,
)
from edelweissfe.generators.surfaceelementgenerator import buildContactFacets
from edelweissfe.journal.journal import Journal
from edelweissfe.models.femodel import FEModel
from edelweissfe.models.meshdependent import MeshDependent
from edelweissfe.sets.elementset import ElementSet
from edelweissfe.sets.nodeset import NodeSet
from edelweissfe.utils.facetcontactgeometry import line2ClosestPoint, tria3ClosestPoint
from edelweissfe.utils.inputlanguage import InputLanguage, Module
from edelweissfe.utils.schema import buildSchemaFromOptions, schemaField

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
    bool,
    True,
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


@dataclass(frozen=True)
class TieSchema:
    """L2: the options this constraint accepts, owned by this module and never mutated from
    outside it.

    Mirrors the ``module.addOptionalArg(...)`` declarations above one-for-one. The two
    declarations coexist while the migration is in progress; the ``Module`` one goes away with the
    ``InputLanguage`` singleton in P5. ``adjust`` is a real ``bool`` field rather than the
    hand-``strtobool``-cast ``str`` the legacy grammar declared it as (see
    :func:`edelweissfe.utils.schema.coerceValue`).
    """

    slaveSurface: str | None = schemaField(
        description="The element set of contact facet elements (Tria3ContactFacet/Line2ContactFacet) "
        "forming the slave surface; its nodes are tied. For quadratic (hexa20/quad8) faces, generate "
        "the facets with triangulation=midside on BOTH surfaces -- the corner triangulation excludes "
        "the midside nodes from the facet node list entirely, leaving them untied.",
        dtype=str,
        default=None,
        required=True,
    )
    masterSurface: str | None = schemaField(
        description="The element set of contact facet elements (Tria3ContactFacet/Line2ContactFacet) "
        "forming the master surface.",
        dtype=str,
        default=None,
        required=True,
    )
    positionTolerance: float | None = schemaField(
        description="Whether a slave node is tied at all: nodes whose reference-configuration "
        "closest-point distance to the master surface exceeds this tolerance are left untied "
        "(recorded in the constraint's untiedSlaveNodes), matching Abaqus' *TIE default behavior "
        "of silently dropping out-of-range slave nodes. Applies independently of adjust -- whether "
        "a tied node's coordinates additionally get snapped onto the master is a separate "
        "decision, see adjust/adjustTolerance. If not given (the default), a tolerance is computed "
        "as positionToleranceFactor times the master surface's characteristic facet size -- see "
        "positionToleranceFactor. Set this explicitly to an absolute distance to override that.",
        dtype=float,
        default=None,
    )
    positionToleranceFactor: float = schemaField(
        description="Used only when positionTolerance is not given: the default tolerance is this "
        "fraction of the master surface's characteristic (mean, over all its facets) edge length, "
        "computed once and used for every slave node. 0.25 comfortably exceeds the sub-percent gaps "
        "expected between two compatible discretizations of the same surface (mismatched density, "
        "curvature/interpolation error), while remaining well below the facet-size-or-larger gaps "
        "that indicate the surfaces don't actually correspond (e.g. a slave surface extending "
        "beyond the master surface's actual extent -- a partial-bond-length or otherwise "
        "partially-overlapping pair of surfaces).",
        dtype=float,
        default=0.25,
    )
    adjust: bool = schemaField(
        description="Whether a TIED node's coordinates additionally get snapped onto its projected "
        "master point at construction (Abaqus-like default) -- a separate decision from whether "
        "the node is tied at all (see positionTolerance). If False, no tied node is ever snapped: "
        "any initial geometric gap is preserved rigidly (the displacements are tied, not the "
        "positions), regardless of size. If True, a tied node is snapped only if its distance is "
        "also within adjustTolerance (default: unconditionally, i.e. every tied node is snapped, "
        "matching plain Abaqus ADJUST=YES) -- see adjustTolerance to snap away only small, "
        "effectively-numerical gaps while still tying (without snapping) across larger, deliberate "
        "ones. Note that adjusting modifies the nodal coordinates before the element geometry is "
        "initialized; avoid adjusting nodes that also belong to an already-generated contact "
        "surface of another constraint.",
        dtype=bool,
        default=True,
    )
    adjustTolerance: float | None = schemaField(
        description="Used only when adjust=True: a tied node's coordinates are snapped onto the "
        "master only if its closest-point distance is also within this tolerance; beyond it, the "
        "node stays tied (kinematically) but its position is left as found, preserving the gap. "
        "Independent of positionTolerance -- a node can be tied across a fairly generous distance "
        "while only genuinely small (e.g. sub-percent, mesh-discretization-scale) gaps within that "
        "get snapped away. If not given (the default), any tied node is snapped, matching plain "
        "Abaqus ADJUST=YES and this constraint's behavior before this option existed.",
        dtype=float,
        default=None,
    )


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

    #: L2 schema declared for the L3 registry, per OptionSchemaProvider.
    schema = TieSchema

    def __init__(
        self,
        name: str,
        model: FEModel,
        slaveSurface: ElementSet,
        masterSurface: ElementSet,
        journal: Journal,
        *,
        configuration: TieSchema = TieSchema(),
    ):
        self.name = name
        self.nDim = model.domainSize

        self._journal = journal
        self._slaveSurfaceSetName = slaveSurface.name
        self._masterSurfaceSetName = masterSurface.name
        self._positionTolerance = configuration.positionTolerance
        self._positionToleranceFactor = configuration.positionToleranceFactor
        self._adjustTolerance = configuration.adjustTolerance
        #: References to the published tied/untied NodeSets, so a later reconcile() updates their
        #: membership in place (via replaceMembers) instead of colliding with its own earlier publish.
        self._tiedNodeSet = None
        self._untiedNodeSet = None

        slaveFacetElements = list(slaveSurface)
        masterFacetElements = list(masterSurface)

        if not masterFacetElements:
            raise ValueError(
                f"Constraint '{name}': master surface '{self._masterSurfaceSetName}' contains no facet elements."
            )
        if not slaveFacetElements:
            raise ValueError(
                f"Constraint '{name}': slave surface '{self._slaveSurfaceSetName}' contains no facet elements."
            )

        masterNodes = {node for el in masterFacetElements for node in el.nodes}
        slaveNodesForCheck = {node for el in slaveFacetElements for node in el.nodes}
        if not masterNodes.isdisjoint(slaveNodesForCheck):
            raise ValueError(
                f"Constraint '{name}': slave surface '{self._slaveSurfaceSetName}' and master "
                f"surface '{self._masterSurfaceSetName}' share nodes -- a node cannot be tied to "
                "itself."
            )

        # Frozen ONCE from the INITIAL (pre-any-AMR) master surface, not recomputed on every
        # reconcile(): computing it fresh from whatever the master surface's mean facet size happens
        # to be at reconcile time is unsafe under AMR -- a node evaluated after an unrelated
        # refinement has already shrunk that mean would get an artificially tight tolerance unrelated
        # to its actual (unchanged) gap.
        if self._positionTolerance is not None:
            self._membershipTolerance = self._positionTolerance
        else:
            initialMasterFacetCoords = [np.array([n.coordinates for n in el.nodes]) for el in masterFacetElements]
            self._membershipTolerance = self._positionToleranceFactor * np.mean(
                [_facetCharacteristicSize(c) for c in initialMasterFacetCoords]
            )

        self.tiedRecords, self.untiedSlaveNodes = self._buildTiedRecords(
            model, slaveFacetElements, masterFacetElements, adjust=configuration.adjust
        )
        self._publishTiedUntiedNodeSets(model)

        # A tie has no per-increment tick of its own that runs before the DofManager/VIJSystemMatrix
        # rebuild -- getMultiPointConstraints() is only called from inside that rebuild, too late to
        # safely swap in newly regenerated facet elements (the just-built system wouldn't know about
        # them). So, unlike nodeToDeformableSurfacePenalty, a tie reconciles via the push escape
        # hatch: model.notifyModelChanged() calls onModelChanged() synchronously, from inside the
        # model modifier's own updateModel(), strictly before the rebuild decision is even made.
        model.registerObserver(self)

    @classmethod
    def fromConstraintDefinition(
        cls, name: str, definition: dict, model: FEModel, journal: Journal = None
    ) -> "Constraint":
        """Build this constraint from a parsed ``*constraint`` definition. See
        :class:`~edelweissfe.constraints.base.multipointconstraintbase.MultiPointConstraintBase`
        for why this is separate from ``__init__``."""
        configuration = buildSchemaFromOptions(cls.schema, definition)
        return cls(
            name,
            model,
            model.elementSets[configuration.slaveSurface],
            model.elementSets[configuration.masterSurface],
            journal,
            configuration=configuration,
        )

    def onModelChanged(self, model: FEModel, changeType, change) -> None:
        if change is not None:
            self.reconcile(model, change)

    def _buildTiedRecords(self, model: FEModel, slaveFacetElements, masterFacetElements, adjust: bool):
        """Project every unique slave-surface node onto its closest master facet (reference
        configuration) and freeze the resulting clamped weights. With ``adjust``, additionally snap
        each tied node onto its projected point (only if also within ``adjustTolerance``), removing
        any initial geometric gap -- a setup-time convenience, never applied on a reconcile-triggered
        re-projection.

        Tie MEMBERSHIP (is this node tied at all) is independent of adjust (does a tied node get
        snapped) -- these are separate decisions, see the ``adjust`` option's docstring. A single
        tolerance, frozen once at construction from the INITIAL (pre-any-AMR) master surface's
        characteristic facet size unless given explicitly, is used for every slave node regardless of
        when it is evaluated (matching Abaqus' *TIE, which always enforces some tolerance -- explicit
        or internally computed -- and never ties unconditionally regardless of distance). It is
        deliberately NOT recomputed from ``masterFacetElements`` on a later reconcile() call: an
        unrelated AMR refinement elsewhere on the master surface would otherwise shrink the mean facet
        size and retroactively tighten the tolerance for nodes evaluated afterwards, for a gap that
        never changed.

        Slave nodes already claimed as slaves by another multi-point constraint of the model are
        skipped: a DOF may be condensed out only once, and a second record for it would be rejected
        by the condensation operator. Dropping the tie record is exact, not a silencer -- the typical
        case is a hanging node created by adaptive refinement of the slave surface, whose masters are
        the coarse-trace nodes of that very surface and are themselves tied, so its interpolated
        motion already is the tied motion and the tie equation is redundant."""

        slaveNodes = list(dict.fromkeys(node for el in slaveFacetElements for node in el.nodes))

        alreadyClaimedNodes = set()
        for constraint in model.multiPointConstraints.values():
            if constraint is not self:
                alreadyClaimedNodes |= constraint.claimedSlaveNodes()

        closestPointFunction = tria3ClosestPoint if self.nDim == 3 else line2ClosestPoint
        masterFacetCoords = [np.array([n.coordinates for n in el.nodes]) for el in masterFacetElements]

        # Frozen at construction (see __init__) -- NOT recomputed from the current (possibly
        # AMR-refined, and therefore shrunk) masterFacetCoords passed in here on a reconcile() call.
        membershipTolerance = self._membershipTolerance

        # Projecting every slave node onto its closest master facet by brute force is O(nSlave *
        # nMaster) and, on a large tied surface re-projected after every AMR refinement, dominates
        # the whole refinement step. Index the master facets by centroid in a k-d tree instead and,
        # per slave node, run the exact closest-point test only on the facets that could possibly win.
        # The candidate set is a *superset* of the brute-force winner (see maxFacetRadius below), and
        # iterating it in ascending facet order with a strict "<" reproduces the brute-force choice
        # exactly (same first-of-equals tie-break) -- so this is an acceleration, not an approximation.
        if masterFacetCoords:
            centroids = np.array([coords.mean(axis=0) for coords in masterFacetCoords])
            facetTree = cKDTree(centroids)
            # The farthest any facet vertex sits from that facet's own centroid. If a facet's true
            # closest point beats a known distance d, its centroid lies within d + maxFacetRadius of
            # the slave node (triangle inequality), so a ball query of that radius, seeded with the
            # exact distance to the nearest-centroid facet (an upper bound on the global minimum),
            # is guaranteed to include the brute-force winner.
            maxFacetRadius = max(
                float(np.linalg.norm(coords - coords.mean(axis=0), axis=1).max()) for coords in masterFacetCoords
            )

        tiedRecords = []
        untiedSlaveNodes = []
        nSkippedClaimedNodes = 0

        for slaveNode in slaveNodes:
            if slaveNode in alreadyClaimedNodes:
                nSkippedClaimedNodes += 1
                continue

            bestWeights = None
            bestFacetIdx = None
            bestDistance = np.inf

            if masterFacetCoords:
                xs = slaveNode.coordinates
                _, seedIdx = facetTree.query(xs, k=1)
                _, seedDistance = closestPointFunction(xs, *masterFacetCoords[int(seedIdx)])
                candidates = set(facetTree.query_ball_point(xs, seedDistance + maxFacetRadius))
                candidates.add(int(seedIdx))
                for facetIdx in sorted(candidates):
                    weights, distance = closestPointFunction(xs, *masterFacetCoords[facetIdx])
                    if distance < bestDistance:
                        bestDistance = distance
                        bestWeights = weights
                        bestFacetIdx = facetIdx

            if bestFacetIdx is None or bestDistance > membershipTolerance:
                untiedSlaveNodes.append(slaveNode)
                continue

            # Snapping is the separate, independent decision: a tied node is only snapped if adjust
            # is requested AND (when given) its distance is also within adjustTolerance -- a node
            # beyond adjustTolerance stays tied, just not snapped, preserving its gap exactly.
            withinAdjustTolerance = self._adjustTolerance is None or bestDistance <= self._adjustTolerance
            if adjust and withinAdjustTolerance and bestDistance > 0.0:
                slaveNode.coordinates[:] = bestWeights @ masterFacetCoords[bestFacetIdx]

            tiedRecords.append((slaveNode, masterFacetElements[bestFacetIdx].nodes, bestWeights))

        if nSkippedClaimedNodes:
            self._journal.message(
                "{:} slave node(s) are already slaves of another multi-point constraint "
                "(e.g. hanging nodes); their redundant tie records were dropped".format(nSkippedClaimedNodes),
                self.name,
            )

        return tiedRecords, untiedSlaveNodes

    def _publishTiedUntiedNodeSets(self, model: FEModel):
        """Expose the tied/untied slave nodes as ordinary node sets -- e.g. via *fieldOutput, or for
        free in Ensight, which automatically creates a part for every node set in the model
        (edelweissfe/outputmanagers/ensight.py's _createGeometryParts), no fieldOutput required just
        to see where these nodes are. A set with no members is left unpublished rather than published
        empty: Ensight's export is unconditional over every node set, so a tie whose untied side is
        (as is typical) always empty would otherwise get its own empty, useless part in every export.

        Called again after every :meth:`reconcile` (AMR may change which nodes are tied/untied): an
        already-published set is updated in place via :meth:`~edelweissfe.sets.orderedset.OrderedSet.
        replaceMembers` rather than recreated, so a reference elsewhere in the model keeps seeing
        current membership, and the "does this name already exist" collision check below only ever
        fires against a genuinely foreign set."""

        tiedNodes = [record[0] for record in self.tiedRecords]
        if tiedNodes:
            if self._tiedNodeSet is not None:
                self._tiedNodeSet.replaceMembers(tiedNodes)
            else:
                tiedSetName = f"{self.name}_tied"
                if tiedSetName in model.nodeSets:
                    raise ValueError(f"Constraint '{self.name}': node set '{tiedSetName}' already exists in the model.")
                self._tiedNodeSet = model.nodeSets[tiedSetName] = NodeSet(tiedSetName, tiedNodes)
        if self.untiedSlaveNodes:
            if self._untiedNodeSet is not None:
                self._untiedNodeSet.replaceMembers(self.untiedSlaveNodes)
            else:
                untiedSetName = f"{self.name}_untied"
                if untiedSetName in model.nodeSets:
                    raise ValueError(
                        f"Constraint '{self.name}': node set '{untiedSetName}' already exists in the model."
                    )
                self._untiedNodeSet = model.nodeSets[untiedSetName] = NodeSet(untiedSetName, self.untiedSlaveNodes)

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
            model, slaveFacetElements, masterFacetElements, adjust=False
        )
        self._publishTiedUntiedNodeSets(model)
        return True

    def claimedSlaveNodes(self) -> set:
        """The tied slave nodes of this constraint. Overrides the base implementation, since a tie
        keeps its records in :attr:`tiedRecords` rather than in the base class' ``_records``."""

        return {slaveNode for slaveNode, _, _ in self.tiedRecords}

    def getMultiPointConstraints(self, dofManager) -> list[tuple[int, list[tuple[int, float]]]]:
        fieldVariableIndices = dofManager.idcsOfFieldVariablesInDofVector

        def getDofs(node):
            if "displacement" not in node.fields:
                raise KeyError(f"Constraint '{self.name}': node {node.label} has no 'displacement' field defined.")
            fieldVar = node.fields["displacement"]
            if fieldVar not in fieldVariableIndices:
                raise KeyError(
                    f"Constraint '{self.name}': displacement field of node {node.label} is not registered in the DofManager (no active degrees of freedom)."
                )
            return fieldVariableIndices[fieldVar]

        records = []
        for slaveNode, masterNodes, weights in self.tiedRecords:
            slaveDofIndices = getDofs(slaveNode)
            masterDofIndices = [getDofs(node) for node in masterNodes]

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
