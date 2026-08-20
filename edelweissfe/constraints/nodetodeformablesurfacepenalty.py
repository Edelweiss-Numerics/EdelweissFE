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

from edelweissfe.constraints.base.constraintbase import ConstraintBase
from edelweissfe.elements.contactsurfaceelement import facetNormalAndMeasure
from edelweissfe.generators.surfaceelementgenerator import buildContactFacets
from edelweissfe.journal.journal import Journal
from edelweissfe.models.femodel import FEModel
from edelweissfe.models.meshdependent import MeshDependent
from edelweissfe.sets.elementset import ElementSet
from edelweissfe.timesteppers.timestep import TimeStep
from edelweissfe.utils.facetcontactgeometry import (
    line2ClosestPoint,
    line2GapGradientHessian,
    tria3ClosestPoint,
    tria3GapGradientHessian,
)
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

# Angular threshold separating a *continuous* drift of the frozen contact frame between two
# increments (the slave staying on its facet, or crossing onto a smoothly adjacent one of a curved
# but adequately resolved master surface) from a genuine *crease* of the master surface, used in
# updateConnectivity to decide whether the frictional history may still be rotated into the newly
# frozen tangent plane. 30 degrees is the customary smooth-versus-sharp feature angle of surface
# geometry processing and of contact surface smoothing, and it separates the two regimes with a
# comfortable margin on both sides: a smooth surface of curvature radius R facetted with facet size
# h changes its normal by about h / R between neighboring facets, i.e. by less than 10 degrees for
# any surface resolved finely enough for a meaningful contact pressure distribution, while the
# creases of engineering geometries (45 degree chamfers, 90 degree steps, notch flanks) change it
# by 45 degrees and more.
_cosOfSmoothNormalChangeLimit = np.cos(np.deg2rad(30.0))


@dataclass(frozen=True)
class NodeToDeformableSurfacePenaltySchema:
    """L2: the options this constraint accepts, owned by this module and never mutated from
    outside it.

    The update-type option is spelled ``type`` in the input file but the field is named
    ``contactType`` here -- a dataclass field literally called ``type`` would shadow the builtin,
    which this project's conventions avoid. ``penalty`` is declared ``required=True`` explicitly,
    but is still given a ``default=None`` so the schema remains constructible for the L1
    constructor's default argument; the L4 adapter (``buildSchemaFromOptions``) still enforces
    that an ``.inp`` file supplies it. ``augmentedLagrange`` is a real ``bool`` field rather than
    the hand-``strtobool``-cast ``str`` the legacy grammar declared it as (see
    :func:`edelweissfe.utils.schema.coerceValue`).
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


def _tria3Containment(xs: np.ndarray, x1: np.ndarray, x2: np.ndarray, x3: np.ndarray) -> tuple[float, float, bool]:
    """Barycentric-like in-plane coordinates (alpha, beta) of the projection of xs onto the
    (possibly non-orthogonal) basis spanned by (x2-x1, x3-x1), and whether that projection falls
    inside the triangle."""

    e1 = x2 - x1
    e2 = x3 - x1
    r = xs - x1
    n = np.cross(e1, e2)
    n = n / np.linalg.norm(n)
    rTangential = r - r.dot(n) * n

    A = np.array([[e1.dot(e1), e1.dot(e2)], [e1.dot(e2), e2.dot(e2)]])
    b = np.array([e1.dot(rTangential), e2.dot(rTangential)])
    alpha, beta = np.linalg.solve(A, b)

    inside = alpha >= 0.0 and beta >= 0.0 and (alpha + beta) <= 1.0
    return alpha, beta, inside


def _line2Containment(xs: np.ndarray, x1: np.ndarray, x2: np.ndarray) -> tuple[float, bool]:
    """Parametric coordinate t of the projection of xs onto the edge (x1,x2), and whether that
    projection falls inside the segment."""

    e = x2 - x1
    t = (xs - x1).dot(e) / e.dot(e)
    return t, 0.0 <= t <= 1.0


class Constraint(ConstraintBase, MeshDependent):
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
    sync with the actual equation system rather than fixed once for the whole analysis.

    This constraint implements :meth:`updateConnectivity`, called once per increment (before the
    equation system is (re)built, see :class:`~edelweissfe.solvers.nonlinearimplicitstatic.NIST`),
    re-assigning each slave node to its single closest facet (within ``searchDistance``, if given)
    based on the last converged configuration -- mirroring the pattern already used by
    EdelweissMeshfree's ``NonlinearQuasistaticSolver``/``DiscreteRigidBodyPenaltyContact`` for
    dynamic contact-pair connectivity. Within :meth:`applyConstraint`, the gap, its exact gradient,
    and its exact Hessian (see :mod:`~edelweissfe.utils.facetcontactgeometry`, including the
    second-derivative term from the facet normal's own pose-dependence -- not curvature, since the
    facet is flat) are recomputed fresh from the *current Newton iterate* every iteration, exactly
    as :mod:`~edelweissfe.constraints.nodetodiscreterigidbodypenalty` does for rigid bodies -- no
    geometry is frozen across iterations within an increment.

    Each slave is assigned at most *one* active facet at a time (reassigned each increment); this
    is a deliberate simplification relative to a multi-candidate-per-slave design -- see the
    project plan this was built from for the more elaborate alternative and why it was not needed
    here. If the slave's assigned facet ever fails its exact in-facet containment test mid-Newton
    (the true contact point has moved onto a neighboring facet within the same increment), no
    contact contribution is assembled for that slave until the next connectivity update -- the
    same accepted non-smoothness at facet boundaries as the rigid-body case's mesh edges.

    The slave side is itself a contact facet surface: the constraint's contact points are the
    unique nodes of the ``slaveSurface`` element set, and each node's penalty force is weighted by
    its tributary area (the sum of ``measure / nFacetNodes`` over its incident slave facets,
    evaluated in the reference configuration). ``penalty`` is thus an interface stiffness modulus
    per unit area, the assembled forces approximate a contact *pressure* distribution, and the
    contact response is insensitive to slave surface refinement.

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
    ``_cosOfSmoothNormalChangeLimit``) restarts its frictional history instead, keeping its normal
    traction multiplier.

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

    #: L2 schema declared for the L3 registry, per OptionSchemaProvider.
    schema = NodeToDeformableSurfacePenaltySchema

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
        self._lastSeenTopologyVersion = model.topologyVersion

        self._slaveSurfaceSetName = slaveSurface.name
        self._masterSurfaceSetName = masterSurface.name
        self.slaveFacetElements = list(slaveSurface)
        self.facetElements = list(masterSurface)

        # The contact points are the unique nodes of the slave surface, each weighted by its
        # tributary area: the sum of its area shares over its incident slave facets (assigned by
        # the surface element generator; consistent with a uniform pressure on the source faces),
        # evaluated in the reference configuration (consistent with the small-deformation
        # setting).
        tributaryAreaOfSlaveNode = {}
        for slaveFacet in self.slaveFacetElements:
            for node, share in zip(slaveFacet.nodes, slaveFacet.nodalAreaShares):
                tributaryAreaOfSlaveNode[node] = tributaryAreaOfSlaveNode.get(node, 0.0) + share

        self.slaveNodes = list(tributaryAreaOfSlaveNode.keys())
        self.tributaryAreas = np.array(list(tributaryAreaOfSlaveNode.values()))
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
        self.type = configuration.contactType.lower()
        if self.type not in ["linear", "quadratic"]:
            raise ValueError(f"Constraint type '{self.type}' is not supported. Use 'linear' or 'quadratic'.")
        self.searchDistance = configuration.searchDistance

        self.sliding = configuration.sliding.lower()
        if self.sliding not in ["finite", "small"]:
            raise ValueError(f"Constraint sliding '{self.sliding}' is not supported. Use 'finite' or 'small'.")

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

        # Frictional history: tangential force exerted on each slave node, at the last converged
        # state and at the current Newton iterate (the latter is only a scratch value, promoted to
        # converged by acceptLastState()).
        self._tangentialForceConverged = np.zeros((self.nSlaves, self.nDim))
        self._tangentialForceCurrent = np.zeros((self.nSlaves, self.nDim))

        # Augmented-Lagrange state: per-slave normal traction multiplier (a force, <= 0 in
        # contact), plus the gap of the current Newton iterate as a scratch value for the Uzawa
        # update on increment acceptance.
        self._lambdaN = np.zeros(self.nSlaves)
        self._gapCurrent = np.zeros(self.nSlaves)

        # Per-slave normal force of the current Newton iterate, for result output.
        self._normalForceCurrent = np.zeros(self.nSlaves)

        self._nodes = []
        self._fieldsOnNodes = []
        self._nDof = 0

        self.totalNormalForce = 0.0

        self.journal.message(
            f"{self.nSlaves} slave nodes, {len(self.facetElements)} master facets, "
            f"sliding={self.sliding}, type={self.type}, friction={'on' if self.mu > 0.0 else 'off'}, "
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
    def fieldsOnNodes(self) -> list:
        return self._fieldsOnNodes

    @property
    def nDof(self) -> int:
        return self._nDof

    def _currentCoordinates(self, nodes: list, model: FEModel, referenceCoords: np.ndarray) -> np.ndarray:
        dispField = model.nodeFields.get("displacement")
        if dispField is None or "U" not in dispField:
            return referenceCoords.copy()
        idcs = dispField._indicesOfNodesInArray
        u = np.array([dispField["U"][idcs[n]] if n in idcs else np.zeros(self.nDim) for n in nodes])
        return referenceCoords + u

    def updateConnectivity(self, model: FEModel) -> bool:
        """Re-assign each slave node to its single closest facet, based on the last converged
        configuration. Called once per increment by the solver, before the equation system is
        (re)built. Reconciles against a mesh mutation (e.g. an AMR refinement of either surface's
        source solid elements) first, at this natural per-increment tick -- see
        :class:`~edelweissfe.models.meshdependent.MeshDependent`."""

        self.reconcileIfChanged(model)

        slaveCoords = self._currentCoordinates(self.slaveNodes, model, self._referenceCoordsSlaves)
        facetCoords = [
            self._currentCoordinates(el.nodes, model, self._referenceCoordsFacets[i])
            for i, el in enumerate(self.facetElements)
        ]

        newAssignment = [None] * self.nSlaves

        if self.sliding == "small":
            # Clamped closest-point search: the assigned facet is the one whose closed domain
            # (interior/edges/vertices) is truly closest -- no dead zone at facet seams, and the
            # clamped weights are non-negative by construction. Facet, weights, and normal are
            # frozen for the whole increment, making the gap linear in the displacement DOFs.
            closestPointFunction = tria3ClosestPoint if self.nDim == 3 else line2ClosestPoint
            nSlavesWithDiscardedHistory = 0
            for s in range(self.nSlaves):
                bestDistance = np.inf
                bestFacet = None
                bestWeights = None
                for i in range(len(self.facetElements)):
                    weights, distance = closestPointFunction(slaveCoords[s], *facetCoords[i])
                    if distance < bestDistance:
                        bestDistance, bestFacet, bestWeights = distance, i, weights

                if bestFacet is not None and (self.searchDistance is None or bestDistance <= self.searchDistance):
                    newAssignment[s] = bestFacet
                    self._frozenWeights[s] = bestWeights
                    normal, _ = facetNormalAndMeasure(facetCoords[bestFacet])
                    previousNormal = self._frozenNormals[s]
                    self._frozenNormals[s] = normal

                    if previousNormal is None or previousNormal.dot(normal) >= _cosOfSmoothNormalChangeLimit:
                        # Rotate the frictional history into the new frozen tangent plane; the frame
                        # changes only slightly per increment in a small-deformation setting. Nothing
                        # to rotate if the slave was out of contact in the last increment (no
                        # previous normal): its history has been zeroed then, respectively at
                        # construction.
                        projectorOntoTangentPlane = np.eye(self.nDim) - np.outer(normal, normal)
                        self._tangentialForceConverged[s] = (
                            projectorOntoTangentPlane @ self._tangentialForceConverged[s]
                        )
                    else:
                        # The new frozen tangent plane is *not* a small perturbation of the one the
                        # frictional history was accumulated in: the slave has slid over a genuine
                        # crease of the master surface (or its facet has rotated far beyond what the
                        # small-sliding formulation assumes). Rotating the history here would
                        # silently discard the -- possibly dominant -- part of it that now points
                        # along the new normal, leaving a stick predictor unrelated to the actual
                        # frictional state, so the history is discarded instead and the slave enters
                        # the increment like a fresh contact on the new face. This is cheap where it
                        # is wrong and right where it matters: a slave in genuine stick does not
                        # move, hence cannot cross a crease in the first place, while a *slipping*
                        # slave is returned onto the cone |f_T| = mu * N within the same increment,
                        # independently of its history. The normal traction multiplier is retained:
                        # it is a frame-independent magnitude acting along the freshly frozen gap
                        # gradient, and the contact pressure does not vanish just because the slave
                        # slid over a corner -- releasing it (as done below on loss of contact)
                        # would only undo the converged Uzawa progress and let the penetration
                        # spike again.
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

        hasChanged = newAssignment != self._assignedFacetIdx
        self._assignedFacetIdx = newAssignment

        newNodes = []
        newFieldsOnNodes = []
        for s in range(self.nSlaves):
            newNodes.append(self.slaveNodes[s])
            newFieldsOnNodes.append(["displacement"])
            if newAssignment[s] is not None:
                facetNodes = self.facetElements[newAssignment[s]].nodes
                newNodes.extend(facetNodes)
                newFieldsOnNodes.extend([["displacement"]] * len(facetNodes))

        if newNodes != self._nodes:
            hasChanged = True

        self._nodes = newNodes
        self._fieldsOnNodes = newFieldsOnNodes
        self._nDof = sum(self.nDim for _ in newNodes)

        return hasChanged

    def reconcile(self, model: FEModel, change) -> bool:
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

        if touchedSlave:
            buildContactFacets(model, *slaveRecipe, self.journal)
            self._rebindSlave(model)
        if touchedMaster:
            buildContactFacets(model, *masterRecipe, self.journal)
            self._rebindMaster(model)
        return True

    def _rebindSlave(self, model: FEModel) -> None:
        """Rebuild the slave-side node list/tributary areas from the regenerated facet set,
        preserving frictional history and the AL multiplier of retained slave nodes by identity."""

        oldTangential = dict(zip(self.slaveNodes, self._tangentialForceConverged))
        oldLambda = dict(zip(self.slaveNodes, self._lambdaN))

        self.slaveFacetElements = list(model.elementSets[self._slaveSurfaceSetName])
        tributaryAreaOfSlaveNode = {}
        for slaveFacet in self.slaveFacetElements:
            for node, share in zip(slaveFacet.nodes, slaveFacet.nodalAreaShares):
                tributaryAreaOfSlaveNode[node] = tributaryAreaOfSlaveNode.get(node, 0.0) + share

        self.slaveNodes = list(tributaryAreaOfSlaveNode.keys())
        self.tributaryAreas = np.array(list(tributaryAreaOfSlaveNode.values()))
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

        self._assignedFacetIdx = [None] * self.nSlaves
        self._frozenWeights = [None] * self.nSlaves
        self._frozenNormals = [None] * self.nSlaves

    def getVIJContributionSize(self) -> int:
        size = 0
        for s in range(self.nSlaves):
            if self._assignedFacetIdx[s] is None:
                continue
            m = len(self.facetElements[self._assignedFacetIdx[s]].nodes) * self.nDim
            size += self.nDim**2 + m * m + 2 * self.nDim * m
        return size

    def shapeVIJContribution(self, flat_view: np.ndarray) -> DeformableSurfaceContactStiffnessView:
        facetNodeCounts = [
            len(self.facetElements[self._assignedFacetIdx[s]].nodes)
            for s in range(self.nSlaves)
            if self._assignedFacetIdx[s] is not None
        ]
        return DeformableSurfaceContactStiffnessView(flat_view, nDim=self.nDim, facetNodeCounts=facetNodeCounts)

    def initializeVIJContribution(self, idcs: np.ndarray, I_: np.ndarray, J_: np.ndarray, offset: int) -> None:
        k = offset
        localOffset = 0
        for s in range(self.nSlaves):
            pIdcs = [idcs[localOffset + i] for i in range(self.nDim)]
            localOffset += self.nDim

            if self._assignedFacetIdx[s] is None:
                continue

            nFacetNodes = len(self.facetElements[self._assignedFacetIdx[s]].nodes)
            m = nFacetNodes * self.nDim
            fIdcs = [idcs[localOffset + i] for i in range(m)]
            localOffset += m

            for i in range(self.nDim):
                for j in range(self.nDim):
                    I_[k] = pIdcs[i]
                    J_[k] = pIdcs[j]
                    k += 1

            for i in range(m):
                for j in range(m):
                    I_[k] = fIdcs[i]
                    J_[k] = fIdcs[j]
                    k += 1

            for i in range(self.nDim):
                for j in range(m):
                    I_[k] = pIdcs[i]
                    J_[k] = fIdcs[j]
                    k += 1

            for i in range(m):
                for j in range(self.nDim):
                    I_[k] = fIdcs[i]
                    J_[k] = pIdcs[j]
                    k += 1

    def applyConstraint(
        self,
        U_np: np.ndarray,
        dU: np.ndarray,
        PExt: np.ndarray,
        K: DeformableSurfaceContactStiffnessView,
        timeStep: TimeStep,
    ):
        self.totalNormalForce = 0.0

        localOffset = 0
        activeIdx = 0
        for s in range(self.nSlaves):
            pStart = localOffset
            localOffset += self.nDim

            self._tangentialForceCurrent[s] = 0.0
            self._normalForceCurrent[s] = 0.0

            if self._assignedFacetIdx[s] is None:
                continue

            facetElement = self.facetElements[self._assignedFacetIdx[s]]
            nFacetNodes = len(facetElement.nodes)
            m = nFacetNodes * self.nDim
            fStart = localOffset
            localOffset += m

            pIdcs = list(range(pStart, pStart + self.nDim))
            fIdcs = list(range(fStart, fStart + m))

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
                if nFacetNodes == 3:
                    alpha, beta, inside = _tria3Containment(xs, *facetCoords)
                    if not inside:
                        activeIdx += 1
                        continue
                    g, w, H = tria3GapGradientHessian(xs, *facetCoords)
                else:
                    t, inside = _line2Containment(xs, *facetCoords)
                    if not inside:
                        activeIdx += 1
                        continue
                    g, w, H = line2GapGradientHessian(xs, *facetCoords)

            if self.sliding == "small":
                self._gapCurrent[s] = g

            lambdaForce = self._lambdaN[s] if self.augmentedLagrange else 0.0

            if g >= 0.0 and lambdaForce == 0.0:
                activeIdx += 1
                continue

            penaltyTimesArea = self.penalty * self.tributaryAreas[s]

            if g >= 0.0:
                # Open gap, but a not-yet-released multiplier from the last Uzawa update: apply
                # its constant force only (no penalty part, no tangent contribution). The
                # multiplier decays to zero within a few increments after separation.
                f_n = lambdaForce
                stiffness = 0.0
            elif self.type == "linear":
                f_n = lambdaForce + penaltyTimesArea * g
                stiffness = penaltyTimesArea
            else:
                # Repulsive force growing quadratically with penetration: f_n must carry the sign
                # of g (negative in contact) so that PExt -= f_n * w pushes the slave outward,
                # matching the linear branch; stiffness = df_n/dg is then positive for g < 0.
                f_n = lambdaForce - 0.5 * penaltyTimesArea * g**2
                stiffness = -penaltyTimesArea * g

            PLocal = -f_n * w
            KLocal = stiffness * np.outer(w, w)
            if H is not None:
                KLocal += f_n * H

            if self.mu > 0.0:
                PFriction, KFriction = self._computeFriction(s, c, nBar, f_n, stiffness, w, dU, pIdcs, fIdcs)
                PLocal += PFriction
                KLocal += KFriction

            globalIdcs = pIdcs + fIdcs
            PExt[globalIdcs] += PLocal

            K.K_pp[activeIdx] += KLocal[: self.nDim, : self.nDim]
            K.K_ff[activeIdx] += KLocal[self.nDim :, self.nDim :]
            K.K_pf[activeIdx] += KLocal[: self.nDim, self.nDim :]
            K.K_fp[activeIdx] += KLocal[self.nDim :, : self.nDim]

            self._normalForceCurrent[s] = f_n
            self.totalNormalForce += f_n
            activeIdx += 1

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
    ) -> tuple[np.ndarray, np.ndarray]:
        """Coulomb friction in the frozen small-sliding frame: elastic (stick) predictor from the
        converged tangential force and the incremental tangential relative displacement, radial
        return onto the friction cone ``|f_T| <= mu * N`` on slip.

        Returns the local external-force contribution and the local stiffness contribution
        ``K = -dPExt_dU`` (nonsymmetric on slip), in the same ``[slave, facet nodes]`` block
        layout as the normal contact contribution.

        The incremental relative displacement maps through the constant matrix
        ``G = kron(c, I)`` (``u_rel = xs - sum_a NBar_a xa``), so ``G.T M G = kron(outer(c, c), M)``
        and ``G.T v = kron(c, v)`` -- the same structure that gives ``w = kron(c, nBar)``.
        """

        nDim = self.nDim
        weights = self._frozenWeights[s]

        projectorOntoTangentPlane = np.eye(nDim) - np.outer(nBar, nBar)
        kTangent = self.tangentPenalty * self.tributaryAreas[s]

        dURelative = dU[pIdcs] - weights @ dU[fIdcs].reshape((len(weights), nDim))
        stickForce = self._tangentialForceConverged[s] - kTangent * (projectorOntoTangentPlane @ dURelative)

        normalForceMagnitude = -f_n
        slipLimit = self.mu * normalForceMagnitude
        stickForceMagnitude = np.linalg.norm(stickForce)

        if stickForceMagnitude <= slipLimit:
            tangentialForce = stickForce
            # d(stickForce)_d(dURelative) = -kTangent * projector; K = -G.T dfT_dU G
            KFriction = np.kron(np.outer(c, c), kTangent * projectorOntoTangentPlane)
        else:
            slipDirection = stickForce / stickForceMagnitude
            tangentialForce = slipLimit * slipDirection
            # dfT_dU = slipDirection * mu * dN_dU + slipLimit * dSlipDirection_dU, with
            # dN_dU = -stiffness * w and dSlipDirection_dU built from the stick predictor;
            # the first term makes KFriction nonsymmetric (normal-tangential coupling).
            KFriction = self.mu * stiffness * np.outer(np.kron(c, slipDirection), w)
            KFriction += np.kron(
                np.outer(c, c),
                (slipLimit * kTangent / stickForceMagnitude)
                * ((np.eye(nDim) - np.outer(slipDirection, slipDirection)) @ projectorOntoTangentPlane),
            )

        self._tangentialForceCurrent[s] = tangentialForce
        PFriction = np.kron(c, tangentialForce)
        return PFriction, KFriction

    def acceptLastState(self):
        """Promote the tangential forces of the last (converged) Newton iterate to the frictional
        history, and perform the incremental Uzawa update of the normal traction multipliers.
        Called by :meth:`~edelweissfe.models.femodel.FEModel.advanceToTime` upon increment
        acceptance."""

        self._tangentialForceConverged[:] = self._tangentialForceCurrent

        if self.augmentedLagrange:
            for s in range(self.nSlaves):
                if self._assignedFacetIdx[s] is None:
                    continue

                g = self._gapCurrent[s]
                penaltyTimesArea = self.penalty * self.tributaryAreas[s]

                # Augment by the converged *penalty force part* (law-dependent), so the multiplier
                # absorbs exactly the force the penalty spring was carrying -- one-step transfer
                # for both laws. Augmenting by penalty*A*g irrespective of the law is the textbook
                # rule for the linear law only; for the quadratic law the converged gap scales as
                # sqrt(2N/(penalty*A)), so penalty*A*g overshoots the required traction by orders
                # of magnitude and destabilizes the increments. At an open gap, release toward
                # zero with the linear measure (there is no penalty force to transfer).
                if g >= 0.0:
                    penaltyForcePart = penaltyTimesArea * g
                elif self.type == "linear":
                    penaltyForcePart = penaltyTimesArea * g
                else:
                    penaltyForcePart = -0.5 * penaltyTimesArea * g**2

                self._lambdaN[s] = min(0.0, self._lambdaN[s] + penaltyForcePart)
