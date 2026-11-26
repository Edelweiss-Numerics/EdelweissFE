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
from edelweissfe.constraints.base.contactpoints import (
    ContactPoints,
    refusePointsMadeForAnotherKindOfModel,
)
from edelweissfe.constraints.base.forcesonlyexplicitevaluation import (
    ForcesOnlyExplicitEvaluation,
)
from edelweissfe.constraints.base.frozencontactsearch import FrozenContactSearch
from edelweissfe.constraints.base.penaltylaw import penaltyLaw
from edelweissfe.constraints.base.penaltysprings import (
    PenaltySprings,
    SpringGradientOnOwnNodes,
    SpringGradientOnSharedNodes,
    addBoundOfAbsoluteStiffnessOfSprings,
    smallestGapsWithin,
)
from edelweissfe.constraints.base.rigidbodycontactstiffness import (
    RigidBodyContactStiffnessView,
    rigidBodyContactLocalEntries,
)
from edelweissfe.constraints.base.surfacecontactpenaltyschema import (
    SurfaceContactPenaltySchema,
    contactPointsOfTheSlaveSurface,
)
from edelweissfe.journal.journal import Journal
from edelweissfe.models.femodel import FEModel
from edelweissfe.models.meshdependent import MeshDependent
from edelweissfe.rigidbodies.discreterigidbody import DiscreteRigidBody
from edelweissfe.solvers.base.criticaltimestep import NextIncrement
from edelweissfe.timesteppers.timestep import TimeStep
from edelweissfe.utils.facetcontactgeometry import (
    candidatePairs,
    closestFacetCandidates,
    facetClosestPoints,
    rowDot,
)
from edelweissfe.utils.rotations import skewMatrix
from edelweissfe.utils.schema import buildSchemaFromOptions, schemaField

"""
A penalty based unilateral contact constraint between the contact points of a slave body and a
:class:`~edelweissfe.rigidbodies.discreterigidbody.DiscreteRigidBody` -- for every kind of contact point: the quadrature
points of the faces of finite elements and the nodes of finite elements. What a kind of point contributes -- how it
moves, which nodes its force goes to, the area it stands for -- is its
:class:`~edelweissfe.constraints.base.contactpoints.ContactPoints`, as in the contact with a deformable master
(:mod:`~edelweissfe.constraints.surfacetodeformablesurfacepenalty`); only the master side differs.

This is the rigid-body counterpart of :mod:`~edelweissfe.constraints.surfacetodeformablesurfacepenalty`. Its default
contact points are quadrature points for the same reason: nodes cannot transmit a correct pressure across a serendipity
(quad8/hexa20) face, whose consistent corner loads are tensile. Integrating the pressure at quadrature points and
distributing it with the parent-face shape functions lets a corner node carry a tensile force from a non-negative
pressure field. With ``contactPoints=nodes`` it is the contact of nodes against a rigid body, with a penalty per unit
area (times the tributary area of a node) and a small-sliding gap.
"""


@dataclass(frozen=True)
class SurfaceToDiscreteRigidBodyPenaltySchema(SurfaceContactPenaltySchema):
    """The options this constraint accepts: the shared slave-side and penalty options of
    :class:`~edelweissfe.constraints.base.surfacecontactpenaltyschema.SurfaceContactPenaltySchema`,
    plus the rigid body. With a ``searchDistance``, a contact point is also never assigned a rigid
    triangle whose plane it lies behind by more than that distance -- which keeps a point that
    penetrated a thin rigid plate from being pushed out through the opposite face.
    """

    rigidBody: str | None = schemaField(
        description="The name of the discrete rigid body (as registered in model.rigidBodies). Its "
        "surface must be closed and consist of triangles.",
        dtype=str,
        default=None,
        required=True,
    )


class Constraint(FrozenContactSearch, ForcesOnlyExplicitEvaluation, ConstraintBase, MeshDependent):
    """
    Penalty based unilateral contact between the contact points of a slave body and a discrete rigid body.

    Contact points
    --------------
    A contact point :math:`p` moves with an interpolation vector :math:`\\boldsymbol{c}` over the nodes it acts on, and
    stands for an area :math:`A_p` (see :mod:`~edelweissfe.constraints.base.contactpoints`): a quadrature point of a
    slave facet, positioned on the curved slave parent face, :math:`\\boldsymbol{x}_p = N^s(\\xi_q) \\cdot
    \\boldsymbol{x}^s` (:math:`\\boldsymbol{c} = N^s(\\xi_q)`, :math:`A_p = w_q J_q`), or a slave node
    (:math:`\\boldsymbol{c} = [1]`, its tributary area). Each acts on nodes of its own in the block of the constraint: a
    quadrature point on a copy of its own of the nodes of its slave face, a slave node on itself; only the RP of the
    rigid body is shared by all. An input file chooses the quadrature points or the nodes of a slave surface (option
    ``contactPoints``); a kind of point of another library is given to the constructor in code.

    Rigid body motion
    -----------------
    The rigid body moves with its reference point (RP): current position
    :math:`\\boldsymbol{x}_{RP} = \\boldsymbol{X}_{RP} + \\boldsymbol{u}_{RP}` and rotation
    :math:`R = \\exp(\\mathrm{skew}(\\boldsymbol{\\theta}))`. A body-frame point :math:`\\boldsymbol{Y}` is
    at :math:`\\boldsymbol{y} = \\boldsymbol{x}_{RP} + R\\,(\\boldsymbol{Y} - \\boldsymbol{X}_{RP})`.

    Contact search (small sliding)
    ------------------------------
    At every contact search (:meth:`updateConnectivity`), each contact point is pulled back into the body
    frame, and its closest rigid triangle :math:`t` is found. Frozen until the next search are the
    triangle's outward body-frame normal :math:`\\boldsymbol{N}_p` and its plane offset
    :math:`d_p = \\boldsymbol{N}_p \\cdot (\\boldsymbol{Y}_t - \\boldsymbol{X}_{RP})`, with
    :math:`\\boldsymbol{Y}_t` any vertex of the triangle. The point then sees the triangle's *plane*,
    moving rigidly with the body. The points that have a triangle are *located* in the block of the constraint
    (:meth:`~edelweissfe.constraints.base.contactpoints.ContactPoints.locatePoints`); a point farther than
    ``searchDistance`` from the rigid surface has none and takes no part until a later search finds it within reach.

    Gap and its derivatives
    -----------------------
    With the current normal :math:`\\boldsymbol{n}_p = R\\,\\boldsymbol{N}_p` and the lever arm
    :math:`\\boldsymbol{r}_p = \\boldsymbol{x}_p - \\boldsymbol{x}_{RP}`, the gap (negative when penetrating) is

    .. math::
        g_p = \\boldsymbol{n}_p \\cdot \\boldsymbol{r}_p - d_p .

    Its gradient with respect to the nodes :math:`a` the point acts on, the RP displacement and the RP rotation
    is

    .. math::
        \\frac{\\partial g_p}{\\partial \\boldsymbol{x}_a} = c_a \\, \\boldsymbol{n}_p , \\qquad
        \\frac{\\partial g_p}{\\partial \\boldsymbol{u}_{RP}} = -\\boldsymbol{n}_p , \\qquad
        \\frac{\\partial g_p}{\\partial \\boldsymbol{\\theta}} = -S^T (\\boldsymbol{r}_p \\times \\boldsymbol{n}_p) ,

    with ``dPhysicalSpin_dTheta`` :math:`S = R\\,J_r(\\boldsymbol{\\theta})`, see
    :meth:`~edelweissfe.rigidbodies.discreterigidbody.DiscreteRigidBody.poseFromDofs`. Since the normal
    rotates with the body, :math:`\\partial \\boldsymbol{n}_p / \\partial \\boldsymbol{\\theta} =
    -\\mathrm{skew}(\\boldsymbol{n}_p)\\,S`, the non-zero second derivatives are

    .. math::
        \\frac{\\partial^2 g_p}{\\partial \\boldsymbol{x}_a \\partial \\boldsymbol{\\theta}}
        = -c_a \\, \\mathrm{skew}(\\boldsymbol{n}_p)\\,S , \\qquad
        \\frac{\\partial^2 g_p}{\\partial \\boldsymbol{u}_{RP} \\partial \\boldsymbol{\\theta}}
        = \\mathrm{skew}(\\boldsymbol{n}_p)\\,S ,

    and their transposes. The :math:`\\boldsymbol{\\theta}\\boldsymbol{\\theta}` block is omitted: it needs the
    second derivative of the exponential map. It affects only the convergence rate of implicit runs with a rotating
    RP, not the residual; with a fixed RP, or in explicit runs, it plays no role.

    Forces and tangent
    ------------------
    With the normal force :math:`f_n` and :math:`\\mathrm{d}f_n/\\mathrm{d}g` from
    :meth:`~edelweissfe.constraints.base.penaltylaw.ContactPenaltyLaw.normalForce`, evaluated for
    :math:`k\\,A_p`, with :math:`k` the ``penalty``, each closed point exerts the force
    :math:`-f_n \\, \\partial g_p / \\partial \\boldsymbol{q}` on the nodes, so it adds the residual
    contribution and its tangent

    .. math::
        P \\mathrel{+{=}} f_n \\, \\frac{\\partial g_p}{\\partial \\boldsymbol{q}} , \\qquad
        K \\mathrel{+{=}} \\frac{\\mathrm{d}f_n}{\\mathrm{d}g} \\,
        \\frac{\\partial g_p}{\\partial \\boldsymbol{q}} \\otimes \\frac{\\partial g_p}{\\partial \\boldsymbol{q}}
        + f_n \\, \\frac{\\partial^2 g_p}{\\partial \\boldsymbol{q}^2} .

    The contact points distribute their part with their interpolation vectors
    (:meth:`~edelweissfe.constraints.base.contactpoints.ContactPoints.addForceResidual`), as in the contact with a
    deformable master.

    Degrees of freedom
    ------------------
    The constraint acts on the nodes of the located contact points (the slave part of its block), followed by the RP
    (displacement and rotation). The stiffness is one block per located point plus the RP block they all share, see
    :mod:`~edelweissfe.constraints.base.rigidbodycontactstiffness`. A search changes the block only if it locates other
    points (or, for a kind of point whose nodes follow from where it is, if the located points act on other nodes). A
    fixed rigid body is simply one whose RP carries Dirichlet conditions; the RP reaction is then the support force.

    The force evaluation is vectorized over the located points, which requires a slave surface of finite elements
    to consist of faces of one type (a single element type).

    Parameters
    ----------
    name
        The name of this constraint.
    model
        The model tree.
    contactPoints
        The contact points of the slave side.
    rigidBody
        The discrete rigid body.
    journal
        The journal instance.
    configuration
        The options this constraint accepts, see :class:`SurfaceToDiscreteRigidBodyPenaltySchema`. Its
        ``slaveSurface``, ``nQuadraturePoints`` and ``contactPoints`` say which contact points an input file builds
        (:meth:`fromConstraintDefinition`); here they are given.
    """

    #: Option schema for this constraint, per OptionSchemaProvider.
    schema = SurfaceToDiscreteRigidBodyPenaltySchema

    #: The contact searches for the closest triangles as the bodies move.
    connectivityChangesWithMotion = True

    #: Journal sender tag.
    identification = "RigidSurfaceContact"

    def __init__(
        self,
        name: str,
        model: FEModel,
        contactPoints: ContactPoints,
        rigidBody: DiscreteRigidBody,
        journal: Journal,
        *,
        configuration: SurfaceToDiscreteRigidBodyPenaltySchema = SurfaceToDiscreteRigidBodyPenaltySchema(),
    ):
        super().__init__(name, model)

        if model.domainSize != 3:
            raise ValueError("surfaceToDiscreteRigidBodyPenalty is only implemented for 3D models.")

        self.name = name
        self.journal = journal
        self.rigidBody = rigidBody
        self.rpNode = rigidBody.rpNode
        self._lastSeenTopologyVersion = model.topology.version
        model.topology.registerMeshDependent(self)

        self.nDim = 3
        self.rpDofCount = 6

        self.penalty = configuration.penalty
        if self.penalty <= 0.0:
            raise ValueError("The penalty must be positive: a non-positive penalty silently disables contact.")
        self.law = penaltyLaw(configuration.contactType)
        if configuration.sliding.lower() != "small":
            raise ValueError(
                f"Constraint sliding '{configuration.sliding}' is not supported by "
                "surfaceToDiscreteRigidBodyPenalty: only 'small' is implemented."
            )
        self.searchDistance = configuration.searchDistance

        # Validates the rigid surface (closed, triangles only, no zero-area triangles).
        self._rigidTriangles, self._rigidTriangleNormals = rigidBody.referenceTriangles()

        refusePointsMadeForAnotherKindOfModel(contactPoints, model, name)
        self.slave = contactPoints
        self._nodes = [self.rpNode]
        self._resetContactPoints()

        self.journal.message(
            f"contact '{self.name}': {self.slave.describe()} against rigid body '{rigidBody.name}' "
            f"({len(self._rigidTriangles)} triangles), type={self.law.name}",
            self.identification,
        )

    @classmethod
    def fromConstraintDefinition(cls, name: str, definition: dict, model: FEModel, journal: Journal) -> "Constraint":
        """Build this constraint from a parsed ``*constraint`` definition, with the contact points the option
        ``contactPoints`` chooses on the slave surface. See
        :class:`~edelweissfe.constraints.base.constraintbase.ConstraintBase` for why this is
        separate from ``__init__``."""
        configuration = buildSchemaFromOptions(cls.schema, definition)
        return cls(
            name,
            model,
            contactPointsOfTheSlaveSurface(configuration, model, name),
            model.rigidBodies[configuration.rigidBody],
            journal,
            configuration=configuration,
        )

    def _resetContactPoints(self):
        """Size the per-point state for the current contact points, which start without a rigid triangle."""

        nPoints = self.slave.nPoints

        # The frozen contact search result of each point. A point without a triangle keeps a zero normal and a zero
        # offset, and is not located.
        self._frozenBodyNormals = np.zeros((nPoints, self.nDim))
        self._frozenPlaneOffsets = np.zeros(nPoints)

        # Per-point results of the current evaluation, for output.
        self._gapCurrent = np.zeros(nPoints)
        self._normalForceCurrent = np.zeros(nPoints)
        self.totalNormalForce = 0.0

        self._locatePointsWithATriangle()

    @property
    def nodes(self) -> list:
        return self._nodes

    @property
    def fields(self) -> list:
        return [["displacement"]] * (len(self._nodes) - 1) + [["displacement", "rotation"]]

    @property
    def nDof(self) -> int:
        return self.nSlaveDof + self.rpDofCount

    @property
    def nPoints(self) -> int:
        """The number of contact points."""
        return self.slave.nPoints

    @property
    def slaveSurfaceNodes(self) -> list:
        """The nodes of an FE slave surface, see
        :attr:`~edelweissfe.constraints.base.contactpoints.ContactPoints.surfaceNodes`."""
        return self.slave.surfaceNodes

    def updateConnectivity(self, model: FEModel) -> bool:
        """The contact search: assign each contact point its closest rigid triangle, in the configuration of the
        model, freeze that triangle's normal and plane offset, and locate the points that have one.

        Called at the start of every increment by implicit solvers, and every
        ``contact-update-frequency`` increments by the explicit solver.

        Returns
        -------
        bool
            True if the nodes of the constraint changed.
        """

        self._searchClosestRigidTriangles(model)
        return self._locatePointsWithATriangle()

    def _locatePointsWithATriangle(self) -> bool:
        """Locate the contact points that have a rigid triangle in the block: the nodes they act on (the slave part),
        then the RP. Their frozen normals and offsets are gathered into arrays, once per search, so that an evaluation
        is a fixed handful of array operations rather than a loop over the points.

        Returns
        -------
        bool
            True if the nodes of the constraint changed.
        """

        located = np.flatnonzero(np.any(self._frozenBodyNormals != 0.0, axis=1))
        self._locatedPoints = located

        previousNodes = self._nodes
        slaveNodes = self.slave.locatePoints(located)
        self._nodes = slaveNodes + [self.rpNode]
        self.nSlaveDof = self.nDim * len(slaveNodes)

        self._bodyNormalsOfLocated = self._frozenBodyNormals[located]
        self._planeOffsetsOfLocated = self._frozenPlaneOffsets[located]
        self._areasOfLocated = self.slave.areas[located]
        self._slaveDofsOfLocated = [self.slave.interpolationOfLocatedPoint(k)[0] for k in range(len(located))]

        # A point that lost its triangle has no gap and no force: reset them, so that getGaps() and the pressures do
        # not keep their last values.
        withoutATriangle = np.ones(self.nPoints, dtype=bool)
        withoutATriangle[located] = False
        self._gapCurrent[withoutATriangle] = 0.0
        self._normalForceCurrent[withoutATriangle] = 0.0

        return self._nodes != previousNodes

    def _searchClosestRigidTriangles(self, model: FEModel):
        """Assign each contact point its closest rigid triangle, see :meth:`updateConnectivity`."""

        currentPoints = self.slave.currentPointCoordinates(model)

        # Pull the contact points back into the body frame: X = X_RP + R^T (x - c). For row vectors,
        # (x - c) @ R computes R^T (x - c).
        rpDisplacement, rotationMatrix, rpReferencePosition = self.rigidBody.getCurrentKinematics()
        rpCurrentPosition = rpReferencePosition + rpDisplacement
        bodyFramePoints = rpReferencePosition + (currentPoints - rpCurrentPosition) @ rotationMatrix

        candidatesPerPoint = closestFacetCandidates(bodyFramePoints, list(self._rigidTriangles), self.searchDistance)

        # Beyond an edge or a vertex, several triangles share the same closest point; the one whose plane
        # the point lies farthest in front of measures the gap correctly. Tolerance: STL is single precision.
        allVertices = self._rigidTriangles.reshape(-1, 3)
        equalDistanceTolerance = 1e-6 * np.linalg.norm(allVertices.max(axis=0) - allVertices.min(axis=0))

        # All (point, candidate) pairs are evaluated at once; the selection below scans the candidate ranks,
        # for all points at once, in each point's candidate order.
        pointIndices, triangleIndices = candidatePairs(candidatesPerPoint)
        pairPoints = bodyFramePoints[pointIndices]
        pairTriangles = self._rigidTriangles[triangleIndices]
        signedDistancesToPlane = rowDot(self._rigidTriangleNormals[triangleIndices], pairPoints - pairTriangles[:, 0])
        if self.searchDistance is not None:
            inRange = ~(signedDistancesToPlane < -self.searchDistance)
            pointIndices, triangleIndices = pointIndices[inRange], triangleIndices[inRange]
            pairPoints, pairTriangles = pairPoints[inRange], pairTriangles[inRange]
            signedDistancesToPlane = signedDistancesToPlane[inRange]
        _, distances = facetClosestPoints(pairPoints, pairTriangles)

        nPoints = len(bodyFramePoints)
        closestTriangle = np.full(nPoints, -1, dtype=np.intp)
        closestDistance = np.full(nPoints, np.inf)
        closestSignedDistanceToPlane = np.full(nPoints, -np.inf)

        # The rank of each pair among its point's candidates; pairs are ordered by point, then rank.
        pairsPerPoint = np.bincount(pointIndices, minlength=nPoints)
        firstPairOfPoint = np.cumsum(pairsPerPoint) - pairsPerPoint
        ranks = np.arange(len(pointIndices)) - firstPairOfPoint[pointIndices]
        pairsByRank = np.argsort(ranks, kind="stable")
        rankBoundaries = np.searchsorted(ranks[pairsByRank], np.arange(ranks.max(initial=-1) + 2))

        for first, end in zip(rankBoundaries[:-1], rankBoundaries[1:]):
            pairs = pairsByRank[first:end]  # at most one per point
            p = pointIndices[pairs]
            distance = distances[pairs]
            signedDistanceToPlane = signedDistancesToPlane[pairs]
            isCloser = distance < closestDistance[p] - equalDistanceTolerance
            isEquallyCloseButInFront = (distance <= closestDistance[p] + equalDistanceTolerance) & (
                signedDistanceToPlane > closestSignedDistanceToPlane[p]
            )
            accepted = isCloser | isEquallyCloseButInFront
            p, pairs = p[accepted], pairs[accepted]
            closestTriangle[p] = triangleIndices[pairs]
            closestDistance[p] = distances[pairs]
            closestSignedDistanceToPlane[p] = signedDistancesToPlane[pairs]

        inContactRange = closestTriangle >= 0
        if self.searchDistance is not None:
            inContactRange &= ~(closestDistance > self.searchDistance)

        self._frozenBodyNormals[:] = 0.0
        self._frozenPlaneOffsets[:] = 0.0
        normals = self._rigidTriangleNormals[closestTriangle[inContactRange]]
        self._frozenBodyNormals[inContactRange] = normals
        self._frozenPlaneOffsets[inContactRange] = rowDot(
            normals, self._rigidTriangles[closestTriangle[inContactRange], 0] - rpReferencePosition
        )

    def _searchLayout(self) -> dict[str, np.ndarray]:
        # The rigid body's triangles are fixed by its geometry file, so their number identifies them.
        return {
            **self.slave.searchLayout(),
            "nRigidTriangles": np.array([len(self._rigidTriangles)], dtype=np.int64),
        }

    def _frozenProjection(self) -> dict[str, np.ndarray]:
        return {"bodyNormals": self._frozenBodyNormals, "planeOffsets": self._frozenPlaneOffsets}

    def _adoptFrozenProjection(self, projection: dict[str, np.ndarray]) -> bool:
        self._frozenBodyNormals[:] = np.reshape(projection["bodyNormals"], self._frozenBodyNormals.shape)
        self._frozenPlaneOffsets[:] = projection["planeOffsets"]
        if self.slave.locatedOnlyWhenConnected:
            # Located at the next update of the connectivity, once the entities of the points are connected.
            return False
        return self._locatePointsWithATriangle()

    def refresh(self, model: FEModel, change) -> bool:
        """Regenerate the contact points if ``change`` touched them (e.g. a refinement of the elements of the slave
        surface), and search the rigid triangles again for the new contact points."""

        if not self.slave.isTouchedBy(model, change):
            return False

        self.slave = self.slave.regenerated(model)
        refusePointsMadeForAnotherKindOfModel(self.slave, model, self.name)
        self._resetContactPoints()
        self._searchClosestRigidTriangles(model)
        self._locatePointsWithATriangle()
        return True

    def localEntriesOfBlock(self) -> tuple[np.ndarray, np.ndarray]:
        """The RP block, then per located contact point its block over its slave degrees of freedom and their coupling
        with the RP, see :mod:`~edelweissfe.constraints.base.rigidbodycontactstiffness`."""
        return rigidBodyContactLocalEntries(np.arange(self.nSlaveDof, self.nDof), self._slaveDofsOfLocated)

    def shapeBlock(self, values: np.ndarray) -> RigidBodyContactStiffnessView:
        return RigidBodyContactStiffnessView(values, self.rpDofCount, [len(dofs) for dofs in self._slaveDofsOfLocated])

    def computeKernels(
        self,
        U_np: np.ndarray,
        dU: np.ndarray,
        P: np.ndarray,
        K: RigidBodyContactStiffnessView | None,
        timeStep: TimeStep,
    ):
        """Add the residual contribution of the contact, the negative of its forces on the nodes of the contact points
        and on the rigid body, and the tangent unless ``K`` is None (explicit runs)."""

        self.totalNormalForce = 0.0
        self._normalForceCurrent[:] = 0.0
        located = self._locatedPoints
        if not len(located):
            return

        gaps, normals, leverArms, dPhysicalSpin_dTheta = self._locatedPointsAgainstTheRigidBody(U_np, dU)
        self._gapCurrent[located] = gaps

        springs = self._springsOfClosedPoints(gaps, normals)
        if springs is None:
            return
        closed, normalForces = springs.points, springs.forces

        self._normalForceCurrent[located[closed]] = normalForces
        self.totalNormalForce = float(normalForces.sum())

        # The force on the nodes of a point is -f_n c n: the contact points distribute it.
        self.slave.addForceResidual(P[: self.nSlaveDof], closed, normalForces, springs.directions)

        # The force on the RP is -f_n dg/du_RP = f_n n, and the force conjugate to its rotation DOF is
        # -f_n dg/dtheta = f_n S^T (r x n).
        forceOnRigidBody = normalForces @ springs.directions
        momentOnRigidBody = normalForces @ np.cross(leverArms[closed], springs.directions)
        P[self.nSlaveDof : self.nSlaveDof + self.nDim] -= forceOnRigidBody
        P[self.nSlaveDof + self.nDim :] -= dPhysicalSpin_dTheta.T @ momentOnRigidBody

        if K is not None:
            self._addTangent(K, closed, normalForces, springs.stiffness, normals, leverArms, dPhysicalSpin_dTheta)

    def _locatedPointsAgainstTheRigidBody(
        self, U_np: np.ndarray, dU: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Where the located contact points are against the rigid body: their gaps, the current normals of their
        triangles, their lever arms about the RP, and :math:`S`, see the class docstring.

        Parameters
        ----------
        U_np
            The solution of the constraint's block.
        dU
            The increment of the constraint's block.

        Returns
        -------
        tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray]
            The gaps (nLocated), the normals and the lever arms (nLocated, nDim), and ``dPhysicalSpin_dTheta``.
        """

        nSlaveDof, nDim = self.nSlaveDof, self.nDim

        # Where the located contact points are, as their kind of point says.
        pointCoordinates = self.slave.coordinatesOfLocatedPoints(U_np[:nSlaveDof], dU[:nSlaveDof])

        # The rigid body pose, from this constraint's own trial solution (not the converged NodeFields).
        rpDisplacement = U_np[nSlaveDof : nSlaveDof + nDim]
        rpRotation = U_np[nSlaveDof + nDim :]
        rpCurrentPosition, rotationMatrix, dPhysicalSpin_dTheta = self.rigidBody.poseFromDofs(
            rpDisplacement, rpRotation
        )

        # n = R N, for row vectors N @ R^T; the lever arm r = x - c; the gap g = n.r - d.
        normals = self._bodyNormalsOfLocated @ rotationMatrix.T
        leverArms = pointCoordinates - rpCurrentPosition
        gaps = np.einsum("pj,pj->p", normals, leverArms) - self._planeOffsetsOfLocated
        return gaps, normals, leverArms, dPhysicalSpin_dTheta

    def _springsOfClosedPoints(self, gaps: np.ndarray, normals: np.ndarray) -> PenaltySprings | None:
        """The penalty springs of the contact law (:meth:`~edelweissfe.constraints.base.penaltylaw.PenaltyLaw.springs`):
        one along the current normal at every closed point, with the force and the stiffness the law gives it for
        :math:`k\\, A_p`, with :math:`k` the ``penalty``. The force (:meth:`computeKernels`) and the bound of the
        stiffness (:meth:`addBoundOfAbsoluteStiffnessTimes`) go through these springs. The constraint measures gaps
        only and takes contact laws only, whose springs are these, if any.

        Parameters
        ----------
        gaps
            The gaps of the located points.
        normals
            The current normals of their triangles.

        Returns
        -------
        PenaltySprings or None
            The springs; None if no point is closed.
        """

        springs = self.law.springs(self.penalty * self._areasOfLocated, gaps, normals)
        return springs[0] if springs else None

    def addBoundOfAbsoluteStiffnessTimes(
        self,
        U_np: np.ndarray,
        dU: np.ndarray,
        weights: np.ndarray,
        bound: np.ndarray,
        timeStep: TimeStep,
        nextIncrement: NextIncrement | None = None,
    ):
        """Add the row sums of the absolute value of the stiffness of the springs, weighted by column; see
        :meth:`~edelweissfe.constraints.base.constraintbase.ConstraintBase.addBoundOfAbsoluteStiffnessTimes`.

        The springs are those of the force (:meth:`_springsOfClosedPoints`), over the next increment at the smallest
        gap of each point within it (:func:`~edelweissfe.constraints.base.penaltysprings.smallestGapsWithin`).
        A spring :math:`p` acts with the vector :math:`\\boldsymbol{a}_p`, the gradient of its gap now (see the class
        docstring): :math:`\\boldsymbol{c} \\otimes \\boldsymbol{n}_p` on the nodes of the contact point, each point on
        its own (with the interpolation matrix of the contact points), :math:`-\\boldsymbol{n}_p` on the displacement of
        the RP and :math:`-S^T (\\boldsymbol{r}_p \\times \\boldsymbol{n}_p)` on its rotation, with the stiffness
        :math:`k_p = \\mathrm{d}f/\\mathrm{d}g` its law gives it; the bound is the one of every penalty spring
        (:func:`~edelweissfe.constraints.base.penaltysprings.addBoundOfAbsoluteStiffnessOfSprings`). Every spring acts
        on the RP, so the tangent's largest entries over the states of the next increment would miss the sum of
        springs that close in different states there; the springs' bound sums them. At the RP's rows it is at least
        the row sums of the springs' tangent, and above them where the moments of points on either side of the RP
        cancel in the tangent, not in the bound -- safe, and blunt where a light free rigid body governs the critical
        time step. Not bounded: the second derivatives
        of the gap, scaled by the normal force (the normal rotates with the body); against the spring, of the order of
        the penetration over the lever arm.
        """

        located = self._locatedPoints
        if not len(located):
            return

        gaps, normals, leverArms, dPhysicalSpin_dTheta = self._locatedPointsAgainstTheRigidBody(U_np, dU)
        if nextIncrement is not None:
            gaps = smallestGapsWithin(
                nextIncrement,
                lambda U, dU, timeStep: self._locatedPointsAgainstTheRigidBody(U, dU)[0],
                U_np,
                dU,
                timeStep,
            )
        springs = self._springsOfClosedPoints(gaps, normals)
        if springs is None:
            return
        closed, n = springs.points, springs.directions

        nSlaveDof, nDim = self.nSlaveDof, self.nDim
        oneNodeEach = -np.ones((len(closed), 1))
        dofsOfTheRp = nSlaveDof + np.arange(self.rpDofCount)
        gradient = [
            SpringGradientOnSharedNodes(
                self.slave.interpolationMatrixOfLocatedPoints()[closed], n, np.arange(nSlaveDof).reshape((-1, nDim))
            ),
            SpringGradientOnOwnNodes(oneNodeEach, n, np.broadcast_to(dofsOfTheRp[:nDim], (len(closed), 1, nDim))),
            SpringGradientOnOwnNodes(
                oneNodeEach,
                np.cross(leverArms[closed], n) @ dPhysicalSpin_dTheta,
                np.broadcast_to(dofsOfTheRp[nDim:], (len(closed), 1, self.rpDofCount - nDim)),
            ),
        ]
        addBoundOfAbsoluteStiffnessOfSprings(bound, weights, springs.stiffness, gradient)

    def _addTangent(
        self,
        K: RigidBodyContactStiffnessView,
        closedPoints: np.ndarray,
        normalForces: np.ndarray,
        dNormalForce_dGap: np.ndarray,
        normals: np.ndarray,
        leverArms: np.ndarray,
        dPhysicalSpin_dTheta: np.ndarray,
    ):
        """Add the tangent of the closed contact points, see the class docstring.

        Parameters
        ----------
        K
            The tangent of the constraint; augmented.
        closedPoints
            The closed points (indices into the located points).
        normalForces
            Their normal forces.
        dNormalForce_dGap
            Their stiffness.
        normals
            The current normals of the located points.
        leverArms
            The lever arms of the located points about the RP.
        dPhysicalSpin_dTheta
            :math:`S`, see the class docstring.
        """

        nDim = self.nDim
        for i, k in enumerate(closedPoints):
            interpolation = self.slave.interpolationOfLocatedPoint(k)[1]
            n = normals[k]

            dGap_dSlave = np.kron(interpolation, n)
            dGap_dRp = np.concatenate((-n, -dPhysicalSpin_dTheta.T @ np.cross(leverArms[k], n)))

            K.K_ss[k] += dNormalForce_dGap[i] * np.outer(dGap_dSlave, dGap_dSlave)
            K.K_srp[k] += dNormalForce_dGap[i] * np.outer(dGap_dSlave, dGap_dRp)
            K.K_rps[k] += dNormalForce_dGap[i] * np.outer(dGap_dRp, dGap_dSlave)
            K.K_rprp += dNormalForce_dGap[i] * np.outer(dGap_dRp, dGap_dRp)

            # The normal rotates with the rigid body: dn/dtheta = -skew(n) S.
            dNormal_dTheta = -skewMatrix(n) @ dPhysicalSpin_dTheta
            d2Gap_dSlave_dTheta = np.kron(interpolation[:, None], dNormal_dTheta)
            d2Gap_dRpDisplacement_dTheta = -dNormal_dTheta

            K.K_srp[k][:, nDim:] += normalForces[i] * d2Gap_dSlave_dTheta
            K.K_rps[k][nDim:, :] += normalForces[i] * d2Gap_dSlave_dTheta.T
            K.K_rprp[:nDim, nDim:] += normalForces[i] * d2Gap_dRpDisplacement_dTheta
            K.K_rprp[nDim:, :nDim] += normalForces[i] * d2Gap_dRpDisplacement_dTheta.T

    def getNormalPressures(self) -> np.ndarray:
        """The current normal contact pressures (positive in compression), one per contact point, ordered like the
        contact points (for quadrature points: by slave facet and quadrature point)."""

        return self.slave.normalPressures(self._normalForceCurrent)

    def getGaps(self) -> np.ndarray:
        """The current gap at each contact point (negative when penetrating); zero for a point
        without an assigned rigid triangle."""

        return self._gapCurrent.copy()

    def getSlaveNodalNormalForces(self) -> np.ndarray:
        """The normal contact force at each node of an FE slave surface; negative means pulled, see
        :meth:`~edelweissfe.constraints.base.contactpoints.ContactPoints.nodalNormalForces`.
        """

        return self.slave.nodalNormalForces(self._normalForceCurrent)
