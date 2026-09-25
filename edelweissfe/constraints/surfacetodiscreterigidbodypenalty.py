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
from edelweissfe.constraints.base.contactpointsonslavesurface import (
    ContactPointsOnSlaveSurface,
)
from edelweissfe.constraints.base.forcesonlyexplicitevaluation import (
    ForcesOnlyExplicitEvaluation,
)
from edelweissfe.constraints.base.penaltylaw import (
    normalPenaltyForce,
    validatedContactType,
)
from edelweissfe.constraints.base.rigidbodycontactstiffness import (
    RigidBodyContactStiffnessView,
    fillRigidBodyContactIndices,
    rigidBodyContactContributionSize,
)
from edelweissfe.constraints.base.surfacecontactpenaltyschema import (
    SurfaceContactPenaltySchema,
)
from edelweissfe.journal.journal import Journal
from edelweissfe.models.femodel import FEModel
from edelweissfe.models.meshdependent import MeshDependent
from edelweissfe.rigidbodies.discreterigidbody import DiscreteRigidBody
from edelweissfe.sets.elementset import ElementSet
from edelweissfe.timesteppers.timestep import TimeStep
from edelweissfe.utils.facetcontactgeometry import (
    closestFacetCandidates,
    tria3ClosestPoint,
)
from edelweissfe.utils.rotations import skewMatrix
from edelweissfe.utils.schema import buildSchemaFromOptions, schemaField

"""
A penalty based unilateral contact constraint between a deformable slave surface and a
:class:`~edelweissfe.rigidbodies.discreterigidbody.DiscreteRigidBody`, integrated over the slave
surface at quadrature points and distributed with the slave *parent element face* shape functions.

This is the rigid-body counterpart of :mod:`~edelweissfe.constraints.surfacetodeformablesurfacepenalty`,
and exists for the same reason: a node-based scheme such as
:mod:`~edelweissfe.constraints.nodetodiscreterigidbodypenalty` cannot transmit a correct pressure
across a serendipity (quad8/hexa20) face, whose consistent corner loads are tensile. Integrating the
pressure at quadrature points and distributing it with the parent-face shape functions lets a corner
node carry a tensile force from a non-negative pressure field.
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


class Constraint(ForcesOnlyExplicitEvaluation, ConstraintBase, MeshDependent):
    """
    Penalty based unilateral contact between a slave surface and a discrete rigid body, integrated
    at quadrature points over the slave facets.

    Contact points
    --------------
    The contact points are the quadrature points of the slave facets, positioned on the curved
    slave parent faces, :math:`\\boldsymbol{x}_q = N^s(\\xi_q) \\cdot \\boldsymbol{x}^s`, with
    integration weights :math:`J_q`; see
    :class:`~edelweissfe.constraints.base.contactpointsonslavesurface.ContactPointsOnSlaveSurface`.

    Rigid body motion
    -----------------
    The rigid body moves with its reference point (RP): current position
    :math:`\\boldsymbol{c} = \\boldsymbol{X}_{RP} + \\boldsymbol{u}_{RP}` and rotation
    :math:`R = \\exp(\\mathrm{skew}(\\boldsymbol{\\theta}))`. A body-frame point :math:`\\boldsymbol{Y}` is
    at :math:`\\boldsymbol{y} = \\boldsymbol{c} + R\\,(\\boldsymbol{Y} - \\boldsymbol{X}_{RP})`.

    Contact search (small sliding)
    ------------------------------
    At every contact search (:meth:`updateConnectivity`), each contact point is pulled back into the
    body frame, and its closest rigid triangle :math:`t` is found. Frozen until the next search are the
    triangle's outward body-frame normal :math:`\\boldsymbol{N}_q` and its plane offset
    :math:`d_q = \\boldsymbol{N}_q \\cdot (\\boldsymbol{Y}_t - \\boldsymbol{X}_{RP})`, with
    :math:`\\boldsymbol{Y}_t` any vertex of the triangle. The point then sees the triangle's *plane*,
    moving rigidly with the body.

    Gap and its derivatives
    -----------------------
    With the current normal :math:`\\boldsymbol{n}_q = R\\,\\boldsymbol{N}_q` and the lever arm
    :math:`\\boldsymbol{r}_q = \\boldsymbol{x}_q - \\boldsymbol{c}`, the gap (negative when penetrating) is

    .. math::
        g_q = \\boldsymbol{n}_q \\cdot \\boldsymbol{r}_q - d_q .

    Its gradient with respect to the slave parent-face nodes, the RP displacement and the RP rotation
    is

    .. math::
        \\frac{\\partial g_q}{\\partial \\boldsymbol{x}^s_a} = N^s_a \\, \\boldsymbol{n}_q , \\qquad
        \\frac{\\partial g_q}{\\partial \\boldsymbol{u}_{RP}} = -\\boldsymbol{n}_q , \\qquad
        \\frac{\\partial g_q}{\\partial \\boldsymbol{\\theta}} = -S^T (\\boldsymbol{r}_q \\times \\boldsymbol{n}_q) ,

    with ``dPhysicalSpin_dTheta`` :math:`S = R\\,J_r(\\boldsymbol{\\theta})`, see
    :meth:`~edelweissfe.rigidbodies.discreterigidbody.DiscreteRigidBody.poseFromDofs`. Since the normal
    rotates with the body, :math:`\\partial \\boldsymbol{n}_q / \\partial \\boldsymbol{\\theta} =
    -\\mathrm{skew}(\\boldsymbol{n}_q)\\,S`, the non-zero second derivatives are

    .. math::
        \\frac{\\partial^2 g_q}{\\partial \\boldsymbol{x}^s_a \\partial \\boldsymbol{\\theta}}
        = -N^s_a \\, \\mathrm{skew}(\\boldsymbol{n}_q)\\,S , \\qquad
        \\frac{\\partial^2 g_q}{\\partial \\boldsymbol{u}_{RP} \\partial \\boldsymbol{\\theta}}
        = \\mathrm{skew}(\\boldsymbol{n}_q)\\,S ,

    and their transposes. The :math:`\\boldsymbol{\\theta}\\boldsymbol{\\theta}` block is omitted, as in
    :mod:`~edelweissfe.constraints.nodetodiscreterigidbodypenalty`: it needs the second derivative of
    the exponential map. It affects only the convergence rate of implicit runs with a rotating RP, not
    the residual; with a fixed RP, or in explicit runs, it plays no role.

    Forces and tangent
    ------------------
    With the normal force :math:`f_n` and :math:`\\mathrm{d}f_n/\\mathrm{d}g` from
    :func:`~edelweissfe.constraints.base.penaltylaw.normalPenaltyForce`, evaluated for
    :math:`p\\,J_q`, each closed point contributes

    .. math::
        P_{ext} \\mathrel{-{=}} f_n \\, \\frac{\\partial g_q}{\\partial \\boldsymbol{q}} , \\qquad
        K \\mathrel{+{=}} \\frac{\\mathrm{d}f_n}{\\mathrm{d}g} \\,
        \\frac{\\partial g_q}{\\partial \\boldsymbol{q}} \\otimes \\frac{\\partial g_q}{\\partial \\boldsymbol{q}}
        + f_n \\, \\frac{\\partial^2 g_q}{\\partial \\boldsymbol{q}^2} .

    Degrees of freedom
    ------------------
    The constraint acts on the parent-face nodes of every slave facet (displacement), followed by the
    RP (displacement and rotation). This footprint does not depend on the contact search, so a search
    never changes the equation system. The stiffness is one block per slave facet plus the shared RP
    block, see :mod:`~edelweissfe.constraints.base.rigidbodycontactstiffness`. A fixed rigid body is
    simply one whose RP carries Dirichlet conditions; the RP reaction is then the support force.

    The force evaluation is vectorized over all contact points, which requires all slave parent faces
    to have the same number of nodes (a slave surface of a single element type).

    Parameters
    ----------
    name
        The name of this constraint.
    model
        The model tree.
    slaveSurface
        The element set of contact facet elements forming the slave surface.
    rigidBody
        The discrete rigid body.
    journal
        The journal instance.
    configuration
        The options this constraint accepts, see :class:`SurfaceToDiscreteRigidBodyPenaltySchema`.
    """

    #: Option schema for this constraint, per OptionSchemaProvider.
    schema = SurfaceToDiscreteRigidBodyPenaltySchema

    #: Journal sender tag.
    identification = "RigidSurfaceContact"

    def __init__(
        self,
        name: str,
        model: FEModel,
        slaveSurface: ElementSet,
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
        self._slaveSurfaceSetName = slaveSurface.name
        self._lastSeenTopologyVersion = model.topologyVersion
        model.registerMeshDependent(self)

        self.nDim = 3
        self.rpDofCount = 6

        self.penalty = configuration.penalty
        if self.penalty <= 0.0:
            raise ValueError("The penalty must be positive: a non-positive penalty silently disables contact.")
        self.type = validatedContactType(configuration.contactType)
        if configuration.sliding.lower() != "small":
            raise ValueError(
                f"Constraint sliding '{configuration.sliding}' is not supported by "
                "surfaceToDiscreteRigidBodyPenalty: only 'small' is implemented."
            )
        self.nQuadraturePoints = configuration.nQuadraturePoints
        self.searchDistance = configuration.searchDistance

        # Validates the rigid surface (closed, triangles only, no zero-area triangles) right away,
        # rather than at the first contact search.
        self._rigidTriangles, self._rigidTriangleNormals = rigidBody.referenceTriangles()

        self._buildFromSlaveSurface(slaveSurface)

        self.journal.message(
            f"contact '{self.name}': {self.nPoints} points ({len(self.slave.facets)} slave facets x "
            f"{self.nQuadraturePoints} quadrature points) against rigid body '{rigidBody.name}' "
            f"({len(self._rigidTriangles)} triangles), type={self.type}",
            self.identification,
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
            model.rigidBodies[configuration.rigidBody],
            journal,
            configuration=configuration,
        )

    def _buildFromSlaveSurface(self, slaveSurface: ElementSet):
        """Build (or rebuild, after an AMR refinement) everything that depends on the slave surface.

        Parameters
        ----------
        slaveSurface
            The element set of slave contact facets.
        """

        self.slave = ContactPointsOnSlaveSurface(slaveSurface, self.nQuadraturePoints, self.name)

        parentNodeCounts = {len(nodes) for nodes in self.slave.parentNodes}
        if len(parentNodeCounts) > 1:
            raise ValueError(
                f"Constraint '{self.name}': the slave surface '{self._slaveSurfaceSetName}' mixes parent "
                f"faces with {sorted(parentNodeCounts)} nodes; a single element type is required."
            )
        self.nParentNodes = parentNodeCounts.pop() if parentNodeCounts else 0

        # Degrees of freedom: all parent-face nodes of each slave facet, facet by facet, then the RP.
        nFacets = len(self.slave.facets)
        self._nodes = [node for nodes in self.slave.parentNodes for node in nodes] + [self.rpNode]
        self._fieldsOnNodes = [["displacement"]] * (len(self._nodes) - 1) + [["displacement", "rotation"]]
        self.nSlaveDof = nFacets * self.nParentNodes * self.nDim
        self._nDof = self.nSlaveDof + self.rpDofCount
        self._slaveBlockSizes = [self.nParentNodes * self.nDim] * nFacets

        # All contact points stacked, for the vectorized evaluation.
        self._shapeFunctions = (
            np.concatenate(self.slave.shapeFunctions) if nFacets else np.zeros((0, self.nParentNodes))
        )
        self._integrationWeights = np.concatenate(self.slave.integrationWeights) if nFacets else np.zeros(0)
        self._parentReferenceCoordinates = np.array(self.slave.parentReferenceCoordinates).reshape(
            (nFacets, self.nParentNodes, self.nDim)
        )

        # The frozen contact search result of each point. An unassigned point keeps a zero normal and
        # a zero offset, so that its gap is exactly zero and it never closes.
        self._frozenBodyNormals = np.zeros((self.nPoints, self.nDim))
        self._frozenPlaneOffsets = np.zeros(self.nPoints)

        self._gapCurrent = np.zeros(self.nPoints)
        self._normalForceCurrent = np.zeros(self.nPoints)
        self.totalNormalForce = 0.0

    @property
    def nodes(self) -> list:
        return self._nodes

    @property
    def fieldsOnNodes(self) -> list:
        return self._fieldsOnNodes

    @property
    def nDof(self) -> int:
        return self._nDof

    @property
    def nPoints(self) -> int:
        """The number of contact points, see :class:`ContactPointsOnSlaveSurface`."""
        return self.slave.nPoints

    @property
    def slaveSurfaceNodes(self) -> list:
        """The unique slave facet nodes, see :attr:`ContactPointsOnSlaveSurface.surfaceNodes`."""
        return self.slave.surfaceNodes

    def updateConnectivity(self, model: FEModel) -> bool:
        """The contact search: assign each contact point its closest rigid triangle, in the last
        converged configuration, and freeze that triangle's normal and plane offset.

        Called at the start of every increment by implicit solvers, and every
        ``contact-update-frequency`` increments by the explicit solver.

        Returns
        -------
        bool
            Always False: the degrees of freedom of this constraint do not depend on the search.
        """

        self._searchClosestRigidTriangles(model)
        return False

    def _searchClosestRigidTriangles(self, model: FEModel):
        """Assign each contact point its closest rigid triangle, see :meth:`updateConnectivity`."""

        currentPoints = self.slave.currentPointCoordinates(model)

        # Pull the contact points back into the body frame: X = X_RP + R^T (x - c). For row vectors,
        # (x - c) @ R computes R^T (x - c).
        rpDisplacement, rotationMatrix, rpReferencePosition = self.rigidBody.getCurrentKinematics()
        rpCurrentPosition = rpReferencePosition + rpDisplacement
        bodyFramePoints = rpReferencePosition + (currentPoints - rpCurrentPosition) @ rotationMatrix

        candidatesPerPoint = closestFacetCandidates(bodyFramePoints, list(self._rigidTriangles), self.searchDistance)

        # Beyond an edge or a vertex, several triangles share the same (clamped) closest point. Of
        # those, the one whose plane the point lies farthest in front of measures the gap correctly;
        # any other would support the point by its infinite plane where the body is not. STL
        # coordinates are single precision, hence the tolerance for "the same distance".
        allVertices = self._rigidTriangles.reshape(-1, 3)
        equalDistanceTolerance = 1e-6 * np.linalg.norm(allVertices.max(axis=0) - allVertices.min(axis=0))

        self._frozenBodyNormals[:] = 0.0
        self._frozenPlaneOffsets[:] = 0.0
        for p, candidates in enumerate(candidatesPerPoint):
            closestTriangle = None
            closestDistance = np.inf
            closestSignedDistanceToPlane = -np.inf
            for t in candidates:
                triangle = self._rigidTriangles[t]
                signedDistanceToPlane = self._rigidTriangleNormals[t] @ (bodyFramePoints[p] - triangle[0])
                if self.searchDistance is not None and signedDistanceToPlane < -self.searchDistance:
                    continue
                _, distance = tria3ClosestPoint(bodyFramePoints[p], *triangle)
                isCloser = distance < closestDistance - equalDistanceTolerance
                isEquallyCloseButInFront = (
                    distance <= closestDistance + equalDistanceTolerance
                    and signedDistanceToPlane > closestSignedDistanceToPlane
                )
                if isCloser or isEquallyCloseButInFront:
                    closestTriangle, closestDistance = t, distance
                    closestSignedDistanceToPlane = signedDistanceToPlane

            if closestTriangle is None:
                continue
            if self.searchDistance is not None and closestDistance > self.searchDistance:
                continue

            normal = self._rigidTriangleNormals[closestTriangle]
            self._frozenBodyNormals[p] = normal
            self._frozenPlaneOffsets[p] = normal @ (self._rigidTriangles[closestTriangle][0] - rpReferencePosition)

    def refresh(self, model: FEModel, change) -> bool:
        """Rebuild the slave side from the regenerated facet set if ``change`` touched its source
        elements, and search the rigid triangles again for the new contact points."""

        if not self.slave.isTouchedBy(model, change):
            return False

        self._buildFromSlaveSurface(model.elementSets[self._slaveSurfaceSetName])
        self._searchClosestRigidTriangles(model)
        return True

    def getVIJContributionSize(self) -> int:
        return rigidBodyContactContributionSize(self.rpDofCount, self._slaveBlockSizes)

    def shapeVIJContribution(self, flat_view: np.ndarray) -> RigidBodyContactStiffnessView:
        return RigidBodyContactStiffnessView(flat_view, self.rpDofCount, self._slaveBlockSizes)

    def initializeVIJContribution(self, idcs: np.ndarray, I_: np.ndarray, J_: np.ndarray, offset: int) -> None:
        slaveBlockGlobalIndices = [idcs[first : first + m] for first, m in self._slaveBlockOffsets()]
        fillRigidBodyContactIndices(idcs[self.nSlaveDof :], slaveBlockGlobalIndices, I_, J_, offset)

    def _slaveBlockOffsets(self) -> list[tuple[int, int]]:
        """The first local index and the size of each slave facet's block."""
        m = self.nParentNodes * self.nDim
        return [(f * m, m) for f in range(len(self.slave.facets))]

    def applyConstraint(
        self,
        U_np: np.ndarray,
        dU: np.ndarray,
        PExt: np.ndarray,
        K: RigidBodyContactStiffnessView | None,
        timeStep: TimeStep,
    ):
        """Evaluate the contact forces, and the tangent unless ``K`` is None (explicit runs)."""

        self.totalNormalForce = 0.0
        self._normalForceCurrent[:] = 0.0
        if self.nPoints == 0:
            return

        nFacets, nQuadraturePoints, nDim = len(self.slave.facets), self.nQuadraturePoints, self.nDim

        # Current positions of the contact points on the curved slave parent faces.
        slaveDisplacements = U_np[: self.nSlaveDof].reshape((nFacets, self.nParentNodes, nDim))
        parentCoordinates = self._parentReferenceCoordinates + slaveDisplacements
        pointCoordinates = np.einsum("pa,paj->pj", self._shapeFunctions, parentCoordinates[self.slave.pointFacet])

        # The rigid body pose, from this constraint's own trial solution (not the converged NodeFields).
        rpDisplacement = U_np[self.nSlaveDof : self.nSlaveDof + nDim]
        rpRotation = U_np[self.nSlaveDof + nDim :]
        rpCurrentPosition, rotationMatrix, dPhysicalSpin_dTheta = self.rigidBody.poseFromDofs(
            rpDisplacement, rpRotation
        )

        # n = R N, for row vectors N @ R^T; the lever arm r = x - c; the gap g = n.r - d.
        normals = self._frozenBodyNormals @ rotationMatrix.T
        leverArms = pointCoordinates - rpCurrentPosition
        gaps = np.einsum("pj,pj->p", normals, leverArms) - self._frozenPlaneOffsets
        self._gapCurrent[:] = gaps

        closed = gaps < 0.0
        if not closed.any():
            return

        normalForces, dNormalForce_dGap = normalPenaltyForce(
            self.type, self.penalty * self._integrationWeights[closed], gaps[closed]
        )
        self._normalForceCurrent[closed] = normalForces
        self.totalNormalForce = float(normalForces.sum())

        # Slave nodes: PExt -= f_n N_a n, summed over the quadrature points of each facet.
        slaveForces = -self._normalForceCurrent[:, None, None] * self._shapeFunctions[:, :, None] * normals[:, None, :]
        PExt[: self.nSlaveDof] += slaveForces.reshape((nFacets, nQuadraturePoints, -1)).sum(axis=1).ravel()

        # RP: PExt -= f_n dg/du_RP = f_n n, and PExt -= f_n dg/dtheta = f_n S^T (r x n).
        forceOnRigidBody = normalForces @ normals[closed]
        momentOnRigidBody = normalForces @ np.cross(leverArms[closed], normals[closed])
        PExt[self.nSlaveDof : self.nSlaveDof + nDim] += forceOnRigidBody
        PExt[self.nSlaveDof + nDim :] += dPhysicalSpin_dTheta.T @ momentOnRigidBody

        if K is not None:
            self._addTangent(
                K,
                np.flatnonzero(closed),
                normalForces,
                dNormalForce_dGap,
                normals,
                leverArms,
                dPhysicalSpin_dTheta,
            )

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
        """Add the tangent of the closed contact points, see the class docstring."""

        nDim = self.nDim
        for i, p in enumerate(closedPoints):
            f = self.slave.pointFacet[p]
            shapeFunctions = self._shapeFunctions[p]
            n = normals[p]

            dGap_dSlave = np.kron(shapeFunctions, n)
            dGap_dRp = np.concatenate((-n, -dPhysicalSpin_dTheta.T @ np.cross(leverArms[p], n)))

            K.K_ss[f] += dNormalForce_dGap[i] * np.outer(dGap_dSlave, dGap_dSlave)
            K.K_srp[f] += dNormalForce_dGap[i] * np.outer(dGap_dSlave, dGap_dRp)
            K.K_rps[f] += dNormalForce_dGap[i] * np.outer(dGap_dRp, dGap_dSlave)
            K.K_rprp += dNormalForce_dGap[i] * np.outer(dGap_dRp, dGap_dRp)

            # The normal rotates with the rigid body: dn/dtheta = -skew(n) S.
            dNormal_dTheta = -skewMatrix(n) @ dPhysicalSpin_dTheta
            d2Gap_dSlave_dTheta = np.kron(shapeFunctions[:, None], dNormal_dTheta)
            d2Gap_dRpDisplacement_dTheta = -dNormal_dTheta

            K.K_srp[f][:, nDim:] += normalForces[i] * d2Gap_dSlave_dTheta
            K.K_rps[f][nDim:, :] += normalForces[i] * d2Gap_dSlave_dTheta.T
            K.K_rprp[:nDim, nDim:] += normalForces[i] * d2Gap_dRpDisplacement_dTheta
            K.K_rprp[nDim:, :nDim] += normalForces[i] * d2Gap_dRpDisplacement_dTheta.T

    def getNormalPressures(self) -> np.ndarray:
        """The current normal contact pressures (positive in compression), one per contact point,
        ordered by slave facet and quadrature point."""

        return self.slave.normalPressures(self._normalForceCurrent)

    def getGaps(self) -> np.ndarray:
        """The current gap at each contact point (negative when penetrating); zero for a point
        without an assigned rigid triangle."""

        return self._gapCurrent.copy()

    def getSlaveNodalNormalForces(self) -> np.ndarray:
        """The normal contact force at each slave surface node; negative means pulled, see
        :meth:`~edelweissfe.constraints.base.contactpointsonslavesurface.ContactPointsOnSlaveSurface.nodalNormalForces`.
        """

        return self.slave.nodalNormalForces(self._normalForceCurrent)
