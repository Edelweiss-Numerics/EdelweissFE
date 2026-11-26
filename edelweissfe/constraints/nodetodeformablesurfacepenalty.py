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

from dataclasses import dataclass, replace

import numpy as np

from edelweissfe.constraints.base.constraintbase import ConstraintBase
from edelweissfe.constraints.base.contactpointsatslavenodes import (
    tributaryAreasOfSlaveNodes,
)
from edelweissfe.constraints.base.forcesonlyexplicitevaluation import (
    ForcesOnlyExplicitEvaluation,
)
from edelweissfe.constraints.base.frozencontactsearch import (
    FrozenContactSearch,
    packAssignment,
    packPerPointArrays,
    unpackAssignment,
    unpackPerPointArrays,
)
from edelweissfe.constraints.base.penaltylaw import penaltyLaw
from edelweissfe.constraints.base.penaltysprings import (
    SpringGradientOnOwnNodes,
    addBoundOfAbsoluteStiffnessOfSprings,
    smallestGapsWithin,
)
from edelweissfe.journal.journal import Journal
from edelweissfe.models.femodel import FEModel
from edelweissfe.models.meshdependent import MeshDependent
from edelweissfe.numerics.systemmatrixcontributor import localEntriesOfSubBlocks
from edelweissfe.sets.elementset import ElementSet
from edelweissfe.solvers.base.criticaltimestep import NextIncrement
from edelweissfe.timesteppers.timestep import TimeStep
from edelweissfe.utils.facetcontactgeometry import (
    closestFacetCandidates,
    closestFacets,
    cosOfSmoothNormalChangeLimit,
    facetNormalAndMeasure,
    line2GapGradientHessian,
    line2Projection,
    tria3GapGradientHessian,
    tria3Projection,
)
from edelweissfe.utils.meshtools import currentNodeCoordinates
from edelweissfe.utils.schema import buildSchemaFromOptions, schemaField

"""
A penalty based unilateral contact constraint between a deformable slave surface and a deformable
master surface, both represented by flat contact facet elements (:mod:`~edelweissfe.elements.
contactsurfaceelement`, typically created via :mod:`~edelweissfe.generators.surfaceelementgenerator`).
The slave surface's nodes act as the contact points; each slave node's penalty force is weighted by
its tributary area (the sum of ``measure / nFacetNodes`` over its incident slave facets, evaluated
in the reference configuration), so ``penalty`` has the meaning of an interface stiffness modulus
per unit area, and the contact response converges under slave surface refinement.

This constraint is a :class:`~edelweissfe.models.meshdependent.MeshDependent`: if either surface's
source solid elements are refined mid-run (e.g. by :mod:`~edelweissfe.modelmodifiers.adaptivity.
hadaptivity`), it regenerates that side's facets from the current (refined) ``*surface`` definition
and rebinds its cached facet list/reference coordinates at its own next :meth:`updateConnectivity`
tick -- no separate wiring needed. A retained slave node (same across the rebuild, by ``Node``
identity) keeps its frictional history and augmented-Lagrange multiplier; a newly exposed one
starts at rest.
"""


@dataclass(frozen=True)
class _ContactOfSlave:
    """The contact of a slave node with its facet, see :meth:`Constraint._contactOfSlave`."""

    #: The gap, negative when penetrating.
    gap: float
    #: Its gradient :math:`\boldsymbol{a}_p = [c, -N^m] \otimes \bar{n}` over the slave node and the nodes of its
    #: facet.
    gradient: np.ndarray
    #: Its Hessian with finite sliding; None with small sliding, where it vanishes.
    hessian: np.ndarray | None
    #: :math:`[c, -N^m] = [1, -N_1, \ldots, -N_k]`: how the gap moves with the slave node (:math:`c = [1]`) and with
    #: the nodes of its facet (their shape functions at the projection).
    nodeCoefficients: np.ndarray
    #: :math:`\bar{n}`, the unit normal of the facet.
    normal: np.ndarray
    #: The normal force (the law's and the multiplier's), negative in contact; None for an open gap without a
    #: multiplier: no force and no spring.
    normalForce: float | None
    #: The stiffness of the normal force, :math:`\mathrm{d}f/\mathrm{d}g`.
    stiffness: float


def _tangentDirections(normal: np.ndarray, first: np.ndarray | None = None) -> np.ndarray:
    """An orthonormal basis of the plane normal to a unit normal: one direction in 2D, two in 3D.

    Parameters
    ----------
    normal
        The unit normal.
    first
        A unit direction in the plane to start the basis with (3D); None for any.

    Returns
    -------
    numpy.ndarray
        The directions, one per row.
    """

    if len(normal) == 2:
        return np.array([[-normal[1], normal[0]]])
    if first is None:
        first = np.cross(normal, np.eye(3)[np.argmin(np.abs(normal))])
        first /= np.linalg.norm(first)
    return np.array([first, np.cross(normal, first)])


@dataclass(frozen=True)
class NodeToDeformableSurfacePenaltySchema:
    """The options this constraint accepts, owned by this module and never mutated from
    outside it.

    The update-type option is spelled ``type`` in the input file but the field is named
    ``contactType`` here -- a dataclass field literally called ``type`` would shadow the builtin,
    which this project's conventions avoid. ``penalty`` is declared ``required=True``, but is
    still given a ``default=None`` so the schema remains constructible on its own;
    ``buildSchemaFromOptions`` still enforces that an ``.inp`` file supplies it.
    ``augmentedLagrange`` is a real ``bool`` field (see :func:`edelweissfe.utils.schema.coerceValue`
    for how string option values are coerced to it).
    """

    slaveSurface: str | None = schemaField(
        description="The element set of contact facet elements (Tria3ContactFacet/Line2ContactFacet) "
        "forming the slave surface; its nodes act as the contact points, penalty-weighted by their "
        "tributary areas.",
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
    penalty: float | None = schemaField(
        description="The numerical penalty value, an interface stiffness modulus per unit slave " "surface area.",
        dtype=float,
        default=None,
        required=True,
    )
    contactType: str = schemaField(
        description="The formulation type: 'linear' (linear force, constant stiffness with jump) "
        "or 'quadratic' (quadratic force, linear stiffness).",
        dtype=str,
        default="linear",
        optionName="type",
    )
    searchDistance: float | None = schemaField(
        description="An optional broadphase distance for the per-increment candidate-facet "
        "search. If not given, every slave is always assigned its single closest facet, without a "
        "distance gate.",
        dtype=float,
        default=None,
    )
    sliding: str = schemaField(
        description="The kinematic treatment of the contact geometry: 'finite' (gap, gradient and "
        "exact Hessian recomputed from the current Newton iterate every iteration) or 'small' "
        "(Abaqus-style small sliding: the closest-point projection -- facet, clamped local "
        "coordinates, and normal -- is frozen once per increment from the last converged "
        "configuration, making the gap linear in the displacement DOFs).",
        dtype=str,
        default="finite",
    )
    mu: float = schemaField(
        description="The Coulomb friction coefficient. Requires sliding=small; mu=0 disables "
        "friction. Strongly recommended in combination with type=quadratic: the quadratic law's "
        "contact stiffness vanishes continuously at gap activation, keeping the frictional "
        "tangent continuous for slave nodes lifting off/touching down -- with type=linear, the "
        "activation stiffness jump scaled by mu makes Newton prone to limit-cycling at such "
        "events.",
        dtype=float,
        default=0.0,
    )
    tangentPenalty: float | None = schemaField(
        description="The tangential penalty stiffness per unit slave surface area for frictional "
        "stick. Defaults to the normal penalty.",
        dtype=float,
        default=None,
    )
    augmentedLagrange: bool = schemaField(
        description="Augment the penalty force with a per-slave normal traction multiplier, "
        "updated once per increment on acceptance (incremental Uzawa: lambda <- min(0, lambda + "
        "penalty * A * g)). The multiplier is constant within an increment (zero tangent "
        "contribution), drives the penetration toward zero over the increments at a fixed "
        "penalty, and sharpens the friction cone mu * N. Requires sliding=small.",
        dtype=bool,
        default=False,
    )


class DeformableSurfaceContactStiffnessView:
    """Provides structured 2-D sub-views for the sparse stiffness matrix slice of
    :class:`Constraint`.

    Each currently-active slave couples only to its own self-block and its currently-assigned
    facet's self-block and slave-facet coupling blocks -- there is no coupling between different
    slave nodes, nor between a slave and any facet it is not currently assigned to.

    Attributes
    ----------
    K_pp : list[numpy.ndarray]
        List of per-slave views of shape ``(nDim, nDim)``, the self-block of each slave node.
    K_ff : list[numpy.ndarray]
        List of per-slave views of shape ``(m, m)``, the self-block of the assigned facet
        (``m = nFacetNodes * nDim``).
    K_pf : list[numpy.ndarray]
        List of per-slave views of shape ``(nDim, m)``, slave-to-facet coupling.
    K_fp : list[numpy.ndarray]
        List of per-slave views of shape ``(m, nDim)``, facet-to-slave coupling (transpose of
        ``K_pf``).
    """

    def __init__(self, flat_array: np.ndarray, nDim: int, facetNodeCounts: list[int]):
        self.K_pp = []
        self.K_ff = []
        self.K_pf = []
        self.K_fp = []

        offset = 0
        for nFacetNodes in facetNodeCounts:
            m = nFacetNodes * nDim

            pp = flat_array[offset : offset + nDim * nDim].reshape((nDim, nDim))
            offset += nDim * nDim

            ff = flat_array[offset : offset + m * m].reshape((m, m))
            offset += m * m

            pf = flat_array[offset : offset + nDim * m].reshape((nDim, m))
            offset += nDim * m

            fp = flat_array[offset : offset + m * nDim].reshape((m, nDim))
            offset += m * nDim

            self.K_pp.append(pp)
            self.K_ff.append(ff)
            self.K_pf.append(pf)
            self.K_fp.append(fp)


class Constraint(FrozenContactSearch, ForcesOnlyExplicitEvaluation, ConstraintBase, MeshDependent):
    """
    Penalty based unilateral contact between the tributary-area-weighted nodes of a deformable
    slave surface and a deformable master surface, both represented by flat (Tria3/Line2) contact
    facet elements.

    Theoretical background
    -----------------------
    Each facet is exactly flat (a plane through 3 nodes, or a line through 2 nodes), so its normal
    has exactly zero curvature over its own domain -- mirroring why the discrete rigid body
    contact's triangulated master surface didn't need a curvature term either. Unlike that rigid
    case, though, each facet's nodes are ordinary displacement DOFs of a deformable body, and
    different facets have disjoint DOFs -- so the set of candidate master facets must be kept in
    sync with the actual discrete system rather than fixed once for the whole analysis.

    This constraint implements :meth:`updateConnectivity`, called once per increment (before the
    discrete system is (re)built, see :class:`~edelweissfe.solvers.nonlinearimplicitstatic.NIST`),
    re-assigning each slave node to its single closest facet (within ``searchDistance``, if given)
    based on the last converged configuration -- mirroring the pattern already used by
    EdelweissMeshfree's ``NonlinearQuasistaticSolver``/``DiscreteRigidBodyPenaltyContact`` for
    dynamic contact-pair connectivity. Within :meth:`computeKernels`, the gap, its exact gradient,
    and its exact Hessian (see :mod:`~edelweissfe.utils.facetcontactgeometry`, including the
    second-derivative term from the facet normal's own pose-dependence -- not curvature, since the
    facet is flat) are recomputed fresh from the *current Newton iterate* every iteration -- no
    geometry is frozen across iterations within an increment.

    Each slave is assigned at most *one* active facet at a time (reassigned each increment). If
    the slave's assigned facet ever fails its exact in-facet containment test mid-Newton
    (the true contact point has moved onto a neighboring facet within the same increment), no
    contact contribution is assembled for that slave until the next connectivity update -- an
    accepted non-smoothness at facet boundaries.

    The slave side is itself a contact facet surface: the constraint's contact points are the
    unique nodes of the ``slaveSurface`` element set, and each node's penalty force is weighted by
    its tributary area (the sum of ``measure / nFacetNodes`` over its incident slave facets,
    evaluated in the reference configuration). ``penalty`` is thus an interface stiffness modulus
    per unit area, the assembled forces approximate a contact *pressure* distribution, and the
    contact response is insensitive to slave surface refinement.

    Both the slave-side area shares and the master-side distribution of the transferred force are
    set by the surface element generator, whose ``nodalWeights`` option selects between them. Under
    ``nodalWeights=serendipityOptimal`` the corner nodes of quadratic faces carry a *zero*
    tributary area -- deliberately: it is the resultant-exact, non-negative weighting closest to
    the parent face's own consistent nodal loads, whose corner entries are negative and hence
    unreachable for a unilateral spring. Such a node stays in the slave node list (so the ordering
    of :meth:`getNormalPressures` and of the generated ``<prefix>_nodes`` set is unaffected) but
    contributes no force and is reported at zero pressure. It therefore also carries no
    penetration guard. Note that no measured benefit has been demonstrated for that option: a
    quadratic face under near-uniform pressure opens its corner gaps regardless, which makes the
    corner weight moot. See the :doc:`contact theory documentation
    </theory/contact>`.

    With ``sliding=small`` (Abaqus-style small sliding), the closest-point projection of each
    slave onto the master surface -- assigned facet, *clamped* local coordinates (closed-domain
    closest point: interior, edge, or vertex; no dead zone at facet seams), and unit normal -- is
    frozen once per increment from the last converged configuration. The gap is then *linear* in
    the displacement DOFs: the gradient is constant, the geometric Hessian term vanishes, and
    both non-smoothness sources of the finite-sliding formulation (facet-normal snap at seams,
    mid-Newton containment loss) disappear; only the gap-sign activation switch remains. This is
    the appropriate formulation for small-deformation applications, and the required basis for
    friction.

    Coulomb friction (``mu > 0``, requires ``sliding=small``) uses an elastic-predictor/
    radial-return corrector in the frozen tangent frame: the tangential force history (promoted
    on increment acceptance via :meth:`acceptLastState`, rotated into the new tangent plane on
    reassignment) plus the tangential penalty stiffness times the incremental relative slip forms
    the stick predictor, capped at ``mu * N`` on the friction cone. The consistent tangent is
    symmetric in stick and nonsymmetric on slip (normal-tangential coupling). Combine with
    ``type=quadratic`` (see the ``mu`` option documentation). Rotating the history is only
    admissible as long as the frozen tangent plane changes continuously between the increments;
    a slave sliding over a genuine crease of the master surface (normal change beyond
    :data:`~edelweissfe.utils.facetcontactgeometry.cosOfSmoothNormalChangeLimit`) restarts its
    frictional history instead, keeping its normal traction multiplier.

    With ``augmentedLagrange=True`` (requires ``sliding=small``), a per-slave normal traction
    multiplier augments the penalty force. It is constant within an increment (zero tangent
    contribution -- it cannot destabilize Newton) and is updated on increment acceptance by the
    converged *penalty force part* (incremental Uzawa; note that the textbook ``penalty * A * g``
    update is correct for the linear law only and overshoots grossly for the quadratic one),
    clamped at zero from above, released at open gaps. Penetration is driven toward zero over
    the increments at a fixed -- hence freely reducible -- penalty, and the friction cone
    ``mu * N`` uses the sharper multiplier-augmented normal force.

    Per-slave results (normal pressure, tangential traction, gap) are exposed via
    :meth:`getNormalPressures`/:meth:`getTangentialTractions`/:meth:`getGaps`, ordered like the
    ``<prefix>_nodes`` node set created by the surface element generator (both enumerate the
    unique slave nodes in facet-creation/first-seen order), so they can be requested directly as
    a ``fromExpression`` field output associated with that node set::

        *fieldOutput
        >>fromExpression, name=pN, nSet=slaveSurf_nodes,
        expression='model.constraints["contact"].getNormalPressures()'

    Currently only available for spatialdomain = 3D (Tria3 facets) or 2D (Line2 facets), matching
    whichever facet type populates the given surface element sets.
    """

    #: Option schema for this constraint, per OptionSchemaProvider.
    schema = NodeToDeformableSurfacePenaltySchema

    #: The contact searches for the surface it touches.
    connectivityChangesWithMotion = True

    def __init__(
        self,
        name: str,
        model: FEModel,
        slaveSurface: ElementSet,
        masterSurface: ElementSet,
        journal: Journal,
        *,
        configuration: NodeToDeformableSurfacePenaltySchema = NodeToDeformableSurfacePenaltySchema(),
    ):
        super().__init__(name, model)

        self.name = name
        self.journal = journal
        self._lastSeenTopologyVersion = model.topology.version
        model.topology.registerMeshDependent(self)

        self._slaveSurfaceSetName = slaveSurface.name
        self._masterSurfaceSetName = masterSurface.name
        self.slaveFacetElements = list(slaveSurface)
        self.facetElements = list(masterSurface)

        # The contact points are the unique slave surface nodes, each weighted by its tributary area.
        self.slaveNodes, self.tributaryAreas = tributaryAreasOfSlaveNodes(self.slaveFacetElements)
        self.nSlaves = len(self.slaveNodes)

        masterNodes = {node for el in self.facetElements for node in el.nodes}
        if not masterNodes.isdisjoint(self.slaveNodes):
            raise ValueError(
                f"Constraint '{name}': slave surface '{self._slaveSurfaceSetName}' and master "
                f"surface '{self._masterSurfaceSetName}' share nodes -- self-contact is not "
                "supported."
            )

        self.penalty = configuration.penalty
        if self.penalty <= 0.0:
            raise ValueError("The penalty must be positive: a non-positive penalty silently disables contact.")
        self.law = penaltyLaw(configuration.contactType)
        self.searchDistance = configuration.searchDistance

        self.sliding = configuration.sliding.lower()
        if self.sliding not in ["finite", "small"]:
            raise ValueError(f"Constraint sliding '{self.sliding}' is not supported. Use 'finite' or 'small'.")
        self._validateMasterWeightTransforms()

        self.mu = configuration.mu
        if self.mu < 0.0:
            raise ValueError("The friction coefficient mu must be non-negative.")
        if self.mu > 0.0 and self.sliding != "small":
            raise ValueError(
                "Coulomb friction (mu > 0) requires sliding=small: the frictional predictor/"
                "corrector operates in the frozen tangent frame of the small-sliding formulation."
            )
        self.tangentPenalty = configuration.tangentPenalty if configuration.tangentPenalty is not None else self.penalty
        if self.mu > 0.0 and self.tangentPenalty <= 0.0:
            raise ValueError(
                "The tangential penalty must be positive when mu > 0: a non-positive tangentPenalty "
                "silently disables the frictional stick response."
            )

        self.augmentedLagrange = configuration.augmentedLagrange
        if self.augmentedLagrange and self.sliding != "small":
            raise ValueError(
                "augmentedLagrange requires sliding=small: the multiplier force acts along the "
                "frozen gap gradient of the small-sliding formulation."
            )

        self.nDim = model.domainSize

        self._referenceCoordsSlaves = np.array([n.coordinates for n in self.slaveNodes])
        self._referenceCoordsFacets = [np.array([n.coordinates for n in el.nodes]) for el in self.facetElements]

        self._assignedFacetIdx = [None] * self.nSlaves

        # Small-sliding frozen projection data, refreshed once per increment in updateConnectivity:
        # clamped closest-point weights on the assigned facet, and the facet's unit normal.
        self._frozenWeights = [None] * self.nSlaves
        self._frozenNormals = [None] * self.nSlaves

        # Frictional history: tangential force on each slave node, converged and at the current Newton
        # iterate (scratch, promoted by acceptLastState()).
        self._tangentialForceConverged = np.zeros((self.nSlaves, self.nDim))
        self._tangentialForceCurrent = np.zeros((self.nSlaves, self.nDim))

        # Augmented-Lagrange state: per-slave normal traction multiplier (a force, <= 0 in contact), and
        # the gap of the current Newton iterate for the Uzawa update on increment acceptance.
        self._lambdaN = np.zeros(self.nSlaves)
        self._gapCurrent = np.zeros(self.nSlaves)

        # Per-slave normal force of the current Newton iterate, for result output.
        self._normalForceCurrent = np.zeros(self.nSlaves)

        self._nodes = []
        self._fields = []
        self._nDof = 0

        self.totalNormalForce = 0.0

        self.journal.message(
            f"{self.nSlaves} slave nodes, {len(self.facetElements)} master facets, "
            f"sliding={self.sliding}, type={self.law.name}, friction={'on' if self.mu > 0.0 else 'off'}, "
            f"augmentedLagrange={self.augmentedLagrange}",
            name,
        )

    @classmethod
    def fromConstraintDefinition(cls, name: str, definition: dict, model: FEModel, journal: Journal) -> "Constraint":
        """Build this constraint from a parsed ``*constraint`` definition. See
        :class:`~edelweissfe.constraints.base.constraintbase.ConstraintBase` for why this is
        separate from ``__init__``."""
        configuration = buildSchemaFromOptions(cls.schema, definition)
        return cls(
            name,
            model,
            model.elementSets[configuration.slaveSurface],
            model.elementSets[configuration.masterSurface],
            journal,
            configuration=configuration,
        )

    @property
    def nodes(self) -> list:
        return self._nodes

    @property
    def fields(self) -> list:
        return self._fields

    @property
    def nDof(self) -> int:
        return self._nDof

    def updateConnectivity(self, model: FEModel) -> bool:
        """Re-assign each slave node to its single closest facet, based on the last converged
        configuration. Called once per increment by the solver, before the discrete system is
        (re)built. A mesh mutation (e.g. an AMR refinement of either surface's source solid
        elements) has been reconciled before, by the topology update -- see
        :class:`~edelweissfe.models.meshdependent.MeshDependent`."""

        slaveCoords = currentNodeCoordinates(self.slaveNodes, model, self._referenceCoordsSlaves)
        facetCoords = [
            currentNodeCoordinates(el.nodes, model, self._referenceCoordsFacets[i])
            for i, el in enumerate(self.facetElements)
        ]

        newAssignment = [None] * self.nSlaves

        if self.sliding == "small":
            # Clamped closest-point search (no dead zone at facet seams, non-negative weights). Facet,
            # weights, and normal are frozen for the whole increment: the gap is linear in the DOFs.
            nSlavesWithDiscardedHistory = 0
            candidatesPerSlave = closestFacetCandidates(slaveCoords, facetCoords, self.searchDistance)
            closestFacet, closestWeights, closestDistance = closestFacets(slaveCoords, facetCoords, candidatesPerSlave)
            for s in range(self.nSlaves):
                bestFacet, bestWeights, bestDistance = int(closestFacet[s]), closestWeights[s], closestDistance[s]

                if bestFacet >= 0 and (self.searchDistance is None or bestDistance <= self.searchDistance):
                    newAssignment[s] = bestFacet
                    # Only the stored weights are transformed, not those of the search;
                    # see _validateMasterWeightTransforms.
                    weightTransform = self.facetElements[bestFacet].weightTransform
                    self._frozenWeights[s] = bestWeights if weightTransform is None else weightTransform @ bestWeights
                    normal, _ = facetNormalAndMeasure(facetCoords[bestFacet])
                    previousNormal = self._frozenNormals[s]
                    self._frozenNormals[s] = normal

                    if previousNormal is None or previousNormal.dot(normal) >= cosOfSmoothNormalChangeLimit:
                        # Rotate the frictional history into the new frozen tangent plane (nothing to
                        # rotate without a previous normal: the history is zero then).
                        projectorOntoTangentPlane = np.eye(self.nDim) - np.outer(normal, normal)
                        self._tangentialForceConverged[s] = (
                            projectorOntoTangentPlane @ self._tangentialForceConverged[s]
                        )
                    else:
                        # The slave slid over a crease: discard the frictional history (the slave
                        # re-enters like a fresh contact on the new face), but keep the normal traction
                        # multiplier, which is frame-independent.
                        self._tangentialForceConverged[s] = 0.0
                        nSlavesWithDiscardedHistory += 1
                else:
                    self._frozenWeights[s] = None
                    self._frozenNormals[s] = None
                    self._tangentialForceConverged[s] = 0.0
                    self._lambdaN[s] = 0.0

            if self.mu > 0.0 and nSlavesWithDiscardedHistory > 0:
                self.journal.message(
                    f"{nSlavesWithDiscardedHistory} slave nodes crossed a crease of the master "
                    "surface; their frictional history was discarded",
                    self.name,
                    level=2,
                )
        else:
            facetCentroids = np.array([np.mean(c, axis=0) for c in facetCoords])
            for s in range(self.nSlaves):
                distances = np.linalg.norm(facetCentroids - slaveCoords[s], axis=1)
                closest = int(np.argmin(distances))
                if self.searchDistance is None or distances[closest] <= self.searchDistance:
                    newAssignment[s] = closest

        return self._assignMasterFacets(newAssignment)

    def _assignMasterFacets(self, newAssignment: list) -> bool:
        """Assign each slave node its master facet (searched or restored) and update the coupled DOFs."""

        hasChanged = newAssignment != self._assignedFacetIdx
        self._assignedFacetIdx = newAssignment

        newNodes = []
        newFields = []
        for s in range(self.nSlaves):
            newNodes.append(self.slaveNodes[s])
            newFields.append(["displacement"])
            if newAssignment[s] is not None:
                facetNodes = self.facetElements[newAssignment[s]].nodes
                newNodes.extend(facetNodes)
                newFields.extend([["displacement"]] * len(facetNodes))

        if newNodes != self._nodes:
            hasChanged = True

        self._nodes = newNodes
        self._fields = newFields
        self._nDof = sum(self.nDim for _ in newNodes)

        return hasChanged

    def refresh(self, model: FEModel, change) -> bool:
        """Regenerate whichever side's facets were affected by ``change`` (via its recorded
        :attr:`~edelweissfe.models.femodel.FEModel.contactFacetRecipes`) and rebind the cached
        per-slave/per-facet arrays to match. A currently-tracked slave node keeps its frictional
        history and augmented-Lagrange multiplier (keyed on ``Node`` identity, since AMR reuses the
        same ``Node`` objects for retained nodes); a newly exposed slave starts at rest. Per-slave
        facet assignment, frozen weights and normals are simply re-sized to the new slave count --
        the next :meth:`updateConnectivity` (called right before this, every increment) recomputes
        them from the current geometry regardless, so there is no stale-index window."""

        slaveRecipe = model.contactFacetRecipes.get(self._slaveSurfaceSetName)
        masterRecipe = model.contactFacetRecipes.get(self._masterSurfaceSetName)
        touchedSlave = slaveRecipe is not None and change.touchesSurface(slaveRecipe[0])
        touchedMaster = masterRecipe is not None and change.touchesSurface(masterRecipe[0])
        if not (touchedSlave or touchedMaster):
            return False

        # Facets are regenerated in the topology-update phase by the implicit surfaceFacets
        # modifier; this constraint only rebinds to whichever side changed.
        if touchedSlave:
            self._rebindSlave(model)
        if touchedMaster:
            self._rebindMaster(model)
        return True

    def _validateMasterWeightTransforms(self) -> None:
        """Refuse a master surface whose facets carry a weight transform unless sliding is small.

        A facet's weight transform redistributes the contact point's interpolation weights among
        the facet's nodes (see the surface element generator's ``nodalWeights`` option). Under
        ``sliding=small`` that is variationally harmless: the facet is flat, the weights are a
        partition of unity, and the frozen normal is normal to the facet plane at the increment's
        start, so ``g = nBar . (xs - sum_a w_a x_a)`` takes the same value for *any* partition of
        unity over the -- coplanar -- facet nodes. The gap, its gradient ``w = kron(c, nBar)`` and
        the symmetry of ``stiffness * outer(w, w)`` are all preserved; only the distribution of the
        transferred force among the facet's nodes changes, which is the entire point. The
        substitution perturbs the gap only as the facet tilts away from the frozen normal within
        the increment, i.e. at exactly the order the small-sliding formulation already discards by
        freezing the normal and dropping the Hessian.

        Under ``sliding=finite`` none of that holds: the gap is measured to the exact closest point
        with a live normal, so substituting the weights would make the force distribution differ
        from the transpose of the gap gradient (a non-symmetric, Petrov--Galerkin operator) and
        would leave the geometric term ``f_n * H`` inconsistent with it. Rather than half-support
        that, refuse it.
        """

        if self.sliding == "small":
            return

        offending = [el.elNumber for el in self.facetElements if el.weightTransform is not None]
        if offending:
            raise ValueError(
                f"Constraint '{self.name}': master surface '{self._masterSurfaceSetName}' was "
                f"generated with nodalWeights='serendipityOptimal' (facet {offending[0]} and "
                f"{len(offending) - 1} more carry a weight transform), which requires "
                "sliding=small. Either set sliding=small on this constraint, or generate the "
                "master facets with the default nodalWeights='facetConsistent'."
            )

    def _rebindSlave(self, model: FEModel) -> None:
        """Rebuild the slave-side node list/tributary areas from the regenerated facet set,
        preserving frictional history and the AL multiplier of retained slave nodes by identity."""

        oldTangential = dict(zip(self.slaveNodes, self._tangentialForceConverged))
        oldLambda = dict(zip(self.slaveNodes, self._lambdaN))

        self.slaveFacetElements = list(model.elementSets[self._slaveSurfaceSetName])
        self.slaveNodes, self.tributaryAreas = tributaryAreasOfSlaveNodes(self.slaveFacetElements)
        self.nSlaves = len(self.slaveNodes)
        self._referenceCoordsSlaves = np.array([n.coordinates for n in self.slaveNodes])

        self._tangentialForceConverged = np.array([oldTangential.get(n, np.zeros(self.nDim)) for n in self.slaveNodes])
        self._tangentialForceCurrent = np.zeros((self.nSlaves, self.nDim))
        self._lambdaN = np.array([oldLambda.get(n, 0.0) for n in self.slaveNodes])
        self._gapCurrent = np.zeros(self.nSlaves)
        self._normalForceCurrent = np.zeros(self.nSlaves)

        self._assignedFacetIdx = [None] * self.nSlaves
        self._frozenWeights = [None] * self.nSlaves
        self._frozenNormals = [None] * self.nSlaves

    def _rebindMaster(self, model: FEModel) -> None:
        """Rebuild the master facet list/reference coordinates from the regenerated facet set. Any
        per-slave facet assignment is invalidated, since it indexes into this list."""

        self.facetElements = list(model.elementSets[self._masterSurfaceSetName])
        self._referenceCoordsFacets = [np.array([n.coordinates for n in el.nodes]) for el in self.facetElements]
        self._validateMasterWeightTransforms()

        self._assignedFacetIdx = [None] * self.nSlaves
        self._frozenWeights = [None] * self.nSlaves
        self._frozenNormals = [None] * self.nSlaves

    def shapeBlock(self, values: np.ndarray) -> DeformableSurfaceContactStiffnessView:
        facetNodeCounts = [
            len(self.facetElements[self._assignedFacetIdx[s]].nodes)
            for s in range(self.nSlaves)
            if self._assignedFacetIdx[s] is not None
        ]
        return DeformableSurfaceContactStiffnessView(values, nDim=self.nDim, facetNodeCounts=facetNodeCounts)

    def localEntriesOfBlock(self) -> tuple[np.ndarray, np.ndarray]:
        """Per assigned slave node, the blocks slave-slave, facet-facet, slave-facet and facet-slave, each row by row;
        a slave node without a facet has no entries."""
        subBlocks = []
        localOffset = 0
        for s in range(self.nSlaves):
            slaveIndices = np.arange(localOffset, localOffset + self.nDim)
            localOffset += self.nDim

            if self._assignedFacetIdx[s] is None:
                continue

            m = len(self.facetElements[self._assignedFacetIdx[s]].nodes) * self.nDim
            facetIndices = np.arange(localOffset, localOffset + m)
            localOffset += m

            subBlocks += [
                (slaveIndices, slaveIndices),
                (facetIndices, facetIndices),
                (slaveIndices, facetIndices),
                (facetIndices, slaveIndices),
            ]
        return localEntriesOfSubBlocks(subBlocks)

    def _slavesWithAFacet(self):
        """The slave nodes with a facet assigned, in the order of the block, with where each sits in it.

        Yields
        ------
        tuple[int, list[int], list[int]]
            The slave, its degrees of freedom in the block, and those of the nodes of its facet.
        """

        localOffset = 0
        for s in range(self.nSlaves):
            pStart = localOffset
            localOffset += self.nDim
            if self._assignedFacetIdx[s] is None:
                continue
            m = len(self.facetElements[self._assignedFacetIdx[s]].nodes) * self.nDim
            fStart = localOffset
            localOffset += m
            yield s, list(range(pStart, pStart + self.nDim)), list(range(fStart, fStart + m))

    def _contactOfSlave(self, s: int, U_np: np.ndarray, pIdcs: list, fIdcs: list) -> "_ContactOfSlave | None":
        """The contact of a slave node with its facet: the gap, its gradient and Hessian, and the normal force and the
        stiffness the law (and the augmented-Lagrange multiplier) gives it. The force (:meth:`computeKernels`) and the
        bound of the stiffness (:meth:`addBoundOfAbsoluteStiffnessTimes`) are made from it.

        Parameters
        ----------
        s
            The slave.
        U_np
            The solution of the constraint's block.
        pIdcs
            The degrees of freedom of the slave in the block.
        fIdcs
            Those of the nodes of its facet.

        Returns
        -------
        _ContactOfSlave | None
            The contact; None if, with finite sliding, the projection has left the facet.
        """

        facetElement = self.facetElements[self._assignedFacetIdx[s]]
        nFacetNodes = len(facetElement.nodes)

        xs = self._referenceCoordsSlaves[s] + U_np[pIdcs]
        facetU = U_np[fIdcs].reshape((nFacetNodes, self.nDim))
        facetCoords = self._referenceCoordsFacets[self._assignedFacetIdx[s]] + facetU

        if self.sliding == "small":
            # Frozen projection: gap is linear in the DOFs, gradient w is constant, and the
            # geometric (Hessian) term vanishes identically. The gradient has the same block
            # structure as the finite case: w = [nBar, -NBar_1 nBar, ..., -NBar_k nBar],
            # compactly w = kron(c, nBar) with c = [1, -NBar_1, ..., -NBar_k].
            weights = self._frozenWeights[s]
            nBar = self._frozenNormals[s]
            g = nBar.dot(xs - weights @ facetCoords)
            c = np.concatenate(([1.0], -weights))
            w = np.kron(c, nBar)
            H = None
        else:
            # The gap of a point against a flat facet: its gradient is again w = kron(c, n), with the facet's shape
            # functions at the projection in c and the facet normal n, its slave block.
            if nFacetNodes == 3:
                alpha, beta, inside = tria3Projection(xs, *facetCoords)
                if not inside:
                    return None
                g, w, H = tria3GapGradientHessian(xs, *facetCoords)
                c = np.array([1.0, -(1.0 - alpha - beta), -alpha, -beta])
            else:
                t, inside = line2Projection(xs, *facetCoords)
                if not inside:
                    return None
                g, w, H = line2GapGradientHessian(xs, *facetCoords)
                c = np.array([1.0, -(1.0 - t), -t])
            nBar = w[: self.nDim]

        return _ContactOfSlave(g, w, H, c, nBar, *self._normalForceOfSlave(s, g))

    def _normalForceOfSlave(self, s: int, g: float) -> tuple[float | None, float]:
        """The normal force and its stiffness the law and the augmented-Lagrange multiplier give a slave node at a gap.

        Parameters
        ----------
        s
            The slave.
        g
            The gap.

        Returns
        -------
        tuple[float | None, float]
            The normal force, negative in contact, None for an open gap without a multiplier; and its stiffness
            :math:`\\mathrm{d}f/\\mathrm{d}g`.
        """

        lambdaForce = self._lambdaN[s] if self.augmentedLagrange else 0.0

        if g >= 0.0 and lambdaForce == 0.0:
            return None, 0.0

        if g >= 0.0:
            # Open gap, but a not-yet-released multiplier: apply its constant force only.
            return lambdaForce, 0.0

        # f_n carries the sign of g (negative in contact), so that the contact force on the nodes,
        # -f_n * w, pushes the slave outward; see edelweissfe.constraints.base.penaltylaw.
        penaltyForce, stiffness = self.law.normalForce(self.penalty * self.tributaryAreas[s], g)
        return lambdaForce + penaltyForce, stiffness

    def computeKernels(
        self,
        U_np: np.ndarray,
        dU: np.ndarray,
        P: np.ndarray,
        K: DeformableSurfaceContactStiffnessView | None,
        timeStep: TimeStep,
    ):
        """Add the residual contribution of the contact forces, and the tangent unless ``K`` is None."""

        self.totalNormalForce = 0.0
        self._tangentialForceCurrent[:] = 0.0
        self._normalForceCurrent[:] = 0.0

        for activeIdx, (s, pIdcs, fIdcs) in enumerate(self._slavesWithAFacet()):
            contact = self._contactOfSlave(s, U_np, pIdcs, fIdcs)
            if contact is None:
                continue

            if self.sliding == "small":
                self._gapCurrent[s] = contact.gap

            if contact.normalForce is None:
                continue

            f_n, stiffness, w, H = contact.normalForce, contact.stiffness, contact.gradient, contact.hessian

            # the residual contribution is the negative of the contact force on the nodes
            PLocal = f_n * w

            # K is None in explicit runs -- see ForcesOnlyExplicitEvaluation.
            KLocal = None
            if K is not None:
                KLocal = stiffness * np.outer(w, w)
                if H is not None:
                    KLocal += f_n * H

            if self.mu > 0.0:
                PFriction, KFriction = self._computeFriction(
                    s,
                    contact.nodeCoefficients,
                    contact.normal,
                    f_n,
                    stiffness,
                    w,
                    dU,
                    pIdcs,
                    fIdcs,
                    computeTangent=(KLocal is not None),
                )
                PLocal += PFriction
                if KLocal is not None:
                    KLocal += KFriction

            globalIdcs = pIdcs + fIdcs
            P[globalIdcs] += PLocal

            if KLocal is not None:
                K.K_pp[activeIdx] += KLocal[: self.nDim, : self.nDim]
                K.K_ff[activeIdx] += KLocal[self.nDim :, self.nDim :]
                K.K_pf[activeIdx] += KLocal[: self.nDim, self.nDim :]
                K.K_fp[activeIdx] += KLocal[self.nDim :, : self.nDim]

            self._normalForceCurrent[s] = f_n
            self.totalNormalForce += f_n

    def _contactAtTheSmallestGapWithinTheNextIncrement(
        self,
        s: int,
        contact: "_ContactOfSlave",
        U_np: np.ndarray,
        dU: np.ndarray,
        pIdcs: list,
        fIdcs: list,
        nextIncrement: NextIncrement,
        timeStep: TimeStep,
    ) -> "_ContactOfSlave":
        """The contact of a slave node at its smallest gap within the next increment
        (:func:`~edelweissfe.constraints.base.penaltysprings.smallestGapsWithin`), for the bound of the
        stiffness over it: the normal force and the stiffness the law gives that gap, the gradient and the Hessian of
        now. With finite sliding, a part of the increment that moves the projection off the facet is not counted (a
        neighbouring facet is not seen), and the normal turns within the increment, so the gap is affine in it, as
        ``smallestGapsWithin`` takes it, only to first order.

        Parameters
        ----------
        s
            The slave.
        contact
            Its contact now (:meth:`_contactOfSlave`).
        U_np
            The solution of the constraint's block.
        dU
            The increment of the constraint's block.
        pIdcs
            The degrees of freedom of the slave in the block.
        fIdcs
            Those of the nodes of its facet.
        nextIncrement
            The next increment.
        timeStep
            The time step now.

        Returns
        -------
        _ContactOfSlave
            The contact at the smallest gap.
        """

        def gapAt(U, dU, timeStep):
            contactThere = self._contactOfSlave(s, U, pIdcs, fIdcs)
            return np.inf if contactThere is None else contactThere.gap

        gap = smallestGapsWithin(nextIncrement, gapAt, U_np, dU, timeStep)
        if not (gap < contact.gap and gap < 0.0):
            return contact
        normalForce, stiffness = self._normalForceOfSlave(s, gap)
        return replace(contact, gap=gap, normalForce=normalForce, stiffness=stiffness)

    def addBoundOfAbsoluteStiffnessTimes(
        self,
        U_np: np.ndarray,
        dU: np.ndarray,
        weights: np.ndarray,
        bound: np.ndarray,
        timeStep: TimeStep,
        nextIncrement: NextIncrement | None = None,
    ):
        """Add the row sums of the absolute value of the stiffness of the contact, weighted by column, from the springs
        of its force; see
        :meth:`~edelweissfe.constraints.base.constraintbase.ConstraintBase.addBoundOfAbsoluteStiffnessTimes`.

        Every slave node in contact (:meth:`_contactOfSlave`, as for the force) has a spring along the normal: it acts
        with the vector :math:`\\boldsymbol{a}_p = [c, -N^m] \\otimes \\bar{n}`, the gradient of its gap over the slave
        node and the nodes of its facet, with the stiffness :math:`k_p = \\mathrm{d}f/\\mathrm{d}g` the law gives it.
        With friction, it has a spring of the tangential penalty :math:`k_T` along every direction :math:`t` of the
        tangent plane, :math:`\\boldsymbol{a}_p = [c, -N^m] \\otimes t`: together, the tangent of sticking. The bound
        of the springs is the one of every penalty spring
        (:func:`~edelweissfe.constraints.base.penaltysprings.addBoundOfAbsoluteStiffnessOfSprings`). Two parts of the
        tangent are not springs; they are added by their absolute row sums, from what the force computes:

        * the geometric stiffness of finite sliding, :math:`f_n \\boldsymbol{H}`, the normal force times the second
          derivative of the gap (against the spring, of the order of the penetration over the size of a facet);
        * while the node slips in the direction :math:`\\boldsymbol{e}_s`, the coupling of the tangential to the normal
          force, :math:`\\mu k_p\\, ([c, -N^m] \\otimes \\boldsymbol{e}_s) \\otimes \\boldsymbol{a}_p`, which is not
          symmetric: in every row the larger of its absolute row sums and of their mean with its absolute column sums,
          as the default bound of a tangent (the elements' argument needs the symmetric part). The rest of the
          tangent of slipping is a spring along the direction :math:`q` of the tangent plane normal to
          :math:`\\boldsymbol{e}_s`, softer than :math:`k_T`: the directions :math:`t` start with
          :math:`\\boldsymbol{e}_s` then, so that it lies under the spring along :math:`q`.

        At the state, the bound holds what the default bound of the tangent gives, since the absolute value of a sum is
        at most the sum of the absolute values: it is at least that, and above it where
        :math:`k_p \\boldsymbol{a}_p \\otimes \\boldsymbol{a}_p`, :math:`f_n \\boldsymbol{H}` and the coupling cancel in
        part, and by the spring along :math:`\\boldsymbol{e}_s` of a slipping node, which its tangent does not have;
        without friction and with small sliding, it is them. Over the next increment, every slave node in contact
        within it enters as it is where its gap is the smallest
        (:meth:`_contactAtTheSmallestGapWithinTheNextIncrement`), with the gradient, the Hessian and the slip of now (a
        node open now enters sticking).

        Each slave node acts on its own copy of the nodes of its facet in the block (:meth:`localEntriesOfBlock`), and
        ``NED`` adds the copies after the bound, so the tangent, too, would bound it over the next increment. The bound
        is taken from the springs so that there is one way to bound a contact: from the same code as its force
        (:meth:`_contactOfSlave`), as for every contact, and without forming a tangent only for the bound in a
        constraint that evaluates its explicit forces without one
        (:class:`~edelweissfe.constraints.base.forcesonlyexplicitevaluation.ForcesOnlyExplicitEvaluation`).
        """

        nodeCoefficients, directions, stiffness, dofs = [], [], [], []
        for s, pIdcs, fIdcs in self._slavesWithAFacet():
            contact = self._contactOfSlave(s, U_np, pIdcs, fIdcs)
            if contact is None:
                continue
            slipDirection = None
            if self.mu > 0.0 and contact.normalForce is not None:
                _, slipDirection = self._stickPredictor(s, contact.normal, contact.normalForce, dU, pIdcs, fIdcs)
            if nextIncrement is not None:
                contact = self._contactAtTheSmallestGapWithinTheNextIncrement(
                    s, contact, U_np, dU, pIdcs, fIdcs, nextIncrement, timeStep
                )
            if contact.normalForce is None:
                continue

            rows = pIdcs + fIdcs
            springs = [(contact.normal, contact.stiffness)]
            if self.mu > 0.0:
                kTangent = self.tangentPenalty * self.tributaryAreas[s]
                springs += [(t, kTangent) for t in _tangentDirections(contact.normal, slipDirection)]
            for direction, k in springs:
                nodeCoefficients.append(contact.nodeCoefficients)
                directions.append(direction)
                stiffness.append(k)
                dofs.append(np.reshape(rows, (-1, self.nDim)))

            if contact.hessian is not None:
                bound[rows] += np.abs(contact.normalForce * contact.hessian) @ weights[rows]
            if slipDirection is not None:
                # The coupling mu k u (x) a_p is not symmetric: in every row the larger of the row sums of its absolute
                # value and of their mean with its column sums, as the default bound of a tangent.
                coupling = self.mu * abs(contact.stiffness)
                absoluteU = np.abs(np.kron(contact.nodeCoefficients, slipDirection))
                absoluteGradient = np.abs(contact.gradient)
                rowsOfTheCoupling = coupling * absoluteU * (absoluteGradient @ weights[rows])
                columnsOfTheCoupling = coupling * absoluteGradient * (absoluteU @ weights[rows])
                bound[rows] += np.maximum(rowsOfTheCoupling, 0.5 * (rowsOfTheCoupling + columnsOfTheCoupling))

        if stiffness:
            gradient = [SpringGradientOnOwnNodes(np.array(nodeCoefficients), np.array(directions), np.array(dofs))]
            addBoundOfAbsoluteStiffnessOfSprings(bound, weights, np.array(stiffness), gradient)

    def getNormalPressures(self) -> np.ndarray:
        """The current per-slave normal contact pressures (positive in compression), ordered like
        the surface element generator's ``<prefix>_nodes`` node set of the slave surface. 0 for a
        slave node with a degenerate (zero-area) tributary area, rather than NaN."""

        return np.divide(
            -self._normalForceCurrent,
            self.tributaryAreas,
            out=np.zeros_like(self.tributaryAreas),
            where=self.tributaryAreas > 0,
        )

    def getTangentialTractions(self) -> np.ndarray:
        """The current per-slave tangential (frictional) traction magnitudes, ordered like the
        surface element generator's ``<prefix>_nodes`` node set of the slave surface. 0 for a
        slave node with a degenerate (zero-area) tributary area, rather than NaN."""

        return np.divide(
            np.linalg.norm(self._tangentialForceCurrent, axis=1),
            self.tributaryAreas,
            out=np.zeros_like(self.tributaryAreas),
            where=self.tributaryAreas > 0,
        )

    def getGaps(self) -> np.ndarray:
        """The current per-slave gaps (negative when penetrating; only meaningful for
        sliding=small), ordered like the surface element generator's ``<prefix>_nodes`` node set
        of the slave surface."""

        return self._gapCurrent.copy()

    def _computeFriction(
        self,
        s: int,
        c: np.ndarray,
        nBar: np.ndarray,
        f_n: float,
        stiffness: float,
        w: np.ndarray,
        dU: np.ndarray,
        pIdcs: list,
        fIdcs: list,
        computeTangent: bool = True,
    ) -> tuple[np.ndarray, np.ndarray | None]:
        """Coulomb friction in the frozen small-sliding frame: elastic (stick) predictor from the
        converged tangential force and the incremental tangential relative displacement, radial
        return onto the friction cone ``|f_T| <= mu * N`` on slip.

        Returns the local residual contribution ``P = -G.T f_T``, the negative of the friction force on
        the nodes, and its tangent ``K = dP_dU`` (nonsymmetric on slip), in the same
        ``[slave, facet nodes]`` block layout as the normal contact contribution.

        The incremental relative displacement maps through the constant matrix
        ``G = kron(c, I)`` (``u_rel = xs - sum_a NBar_a xa``), so ``G.T M G = kron(outer(c, c), M)``
        and ``G.T v = kron(c, v)`` -- the same structure that gives ``w = kron(c, nBar)``.
        """

        nDim = self.nDim
        projectorOntoTangentPlane = np.eye(nDim) - np.outer(nBar, nBar)
        kTangent = self.tangentPenalty * self.tributaryAreas[s]
        slipLimit = self.mu * -f_n
        stickForce, slipDirection = self._stickPredictor(s, nBar, f_n, dU, pIdcs, fIdcs)

        if slipDirection is None:
            tangentialForce = stickForce
            # d(stickForce)_d(dURelative) = -kTangent * projector; K = -G.T dfT_dU G
            KFriction = np.kron(np.outer(c, c), kTangent * projectorOntoTangentPlane) if computeTangent else None
        else:
            stickForceMagnitude = np.linalg.norm(stickForce)
            tangentialForce = slipLimit * slipDirection
            if computeTangent:
                # dfT_dU = slipDirection * mu * dN_dU + slipLimit * dSlipDirection_dU, with
                # dN_dU = -stiffness * w and dSlipDirection_dU built from the stick predictor;
                # the first term makes KFriction nonsymmetric (normal-tangential coupling).
                KFriction = self.mu * stiffness * np.outer(np.kron(c, slipDirection), w)
                KFriction += np.kron(
                    np.outer(c, c),
                    (slipLimit * kTangent / stickForceMagnitude)
                    * ((np.eye(nDim) - np.outer(slipDirection, slipDirection)) @ projectorOntoTangentPlane),
                )
            else:
                KFriction = None

        self._tangentialForceCurrent[s] = tangentialForce
        # the friction force on the slave is f_T and on the facet nodes -NBar_a f_T, i.e. G.T f_T
        PFriction = -np.kron(c, tangentialForce)
        return PFriction, KFriction

    def _stickPredictor(
        self, s: int, nBar: np.ndarray, f_n: float, dU: np.ndarray, pIdcs: list, fIdcs: list
    ) -> tuple[np.ndarray, np.ndarray | None]:
        """The tangential force of a slave node in contact if it sticks (the elastic predictor from the converged
        tangential force and the increment of the relative displacement), and the direction it slips in.

        Parameters
        ----------
        s
            The slave.
        nBar
            The unit normal of its facet.
        f_n
            The normal force, negative in contact.
        dU
            The increment of the constraint's block.
        pIdcs
            The degrees of freedom of the slave in the block.
        fIdcs
            Those of the nodes of its facet.

        Returns
        -------
        tuple[numpy.ndarray, numpy.ndarray | None]
            The stick predictor; and the unit direction of the slip, None while the predictor lies within the friction
            cone :math:`|f_T| \\le \\mu N` (the node sticks).
        """

        nDim = self.nDim
        weights = self._frozenWeights[s]
        projectorOntoTangentPlane = np.eye(nDim) - np.outer(nBar, nBar)
        kTangent = self.tangentPenalty * self.tributaryAreas[s]

        dURelative = dU[pIdcs] - weights @ dU[fIdcs].reshape((len(weights), nDim))
        stickForce = self._tangentialForceConverged[s] - kTangent * (projectorOntoTangentPlane @ dURelative)

        stickForceMagnitude = np.linalg.norm(stickForce)
        if stickForceMagnitude <= self.mu * -f_n:
            return stickForce, None
        return stickForce, stickForce / stickForceMagnitude

    def acceptLastState(self):
        """Promote the tangential forces of the last (converged) Newton iterate to the frictional
        history, and perform the incremental Uzawa update of the normal traction multipliers.
        Called by :meth:`~edelweissfe.models.femodel.FEModel.acceptTheStateOfTheConstraints` upon increment
        acceptance."""

        self._tangentialForceConverged[:] = self._tangentialForceCurrent

        if self.augmentedLagrange:
            for s in range(self.nSlaves):
                if self._assignedFacetIdx[s] is None:
                    continue

                g = self._gapCurrent[s]
                penaltyTimesArea = self.penalty * self.tributaryAreas[s]

                # Augment by the converged penalty force (law-dependent); the textbook penalty*A*g holds
                # for the linear law only and overshoots grossly for the quadratic one. At an open gap,
                # release toward zero with the linear measure.
                if g >= 0.0:
                    penaltyForcePart = penaltyTimesArea * g
                else:
                    penaltyForcePart, _ = self.law.normalForce(penaltyTimesArea, g)

                self._lambdaN[s] = min(0.0, self._lambdaN[s] + penaltyForcePart)

    def _searchLayout(self) -> dict[str, np.ndarray]:
        return {
            "slaveNodes": np.array([node.label for node in self.slaveNodes], dtype=np.int64),
            "masterFacets": np.array([facet.elNumber for facet in self.facetElements], dtype=np.int64),
        }

    def _frozenProjection(self) -> dict[str, np.ndarray]:
        weightCounts, weights = packPerPointArrays(self._frozenWeights)
        normalCounts, normals = packPerPointArrays(self._frozenNormals)
        return {
            "assignedFacetIdx": packAssignment(self._assignedFacetIdx),
            "weightCounts": weightCounts,
            "weights": weights,
            "normalCounts": normalCounts,
            "normals": normals,
        }

    def _adoptFrozenProjection(self, projection: dict[str, np.ndarray]) -> bool:
        # The frictional history was already rotated into this tangent plane by the search that froze it.
        self._frozenWeights = unpackPerPointArrays(projection["weightCounts"], projection["weights"])
        self._frozenNormals = unpackPerPointArrays(projection["normalCounts"], projection["normals"])
        return self._assignMasterFacets(unpackAssignment(projection["assignedFacetIdx"]))

    def getRestartData(self) -> dict[str, np.ndarray]:
        """Return the frozen projection and the frictional and augmented-Lagrange history. Overridden
        rather than declared: the frozen projection is packed by :class:`FrozenContactSearch`."""

        return super().getRestartData() | {
            "tangentialForceConverged": self._tangentialForceConverged,
            "lambdaN": self._lambdaN,
        }

    def setRestartData(self, data: dict[str, np.ndarray]):
        """Restore the frozen projection and the frictional and augmented-Lagrange history."""

        super().setRestartData(data)
        self._tangentialForceConverged[:] = data["tangentialForceConverged"]
        self._lambdaN[:] = data["lambdaN"]
