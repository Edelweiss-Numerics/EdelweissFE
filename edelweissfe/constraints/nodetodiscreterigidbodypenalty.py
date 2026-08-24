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
from edelweissfe.journal.journal import Journal
from edelweissfe.models.femodel import FEModel
from edelweissfe.models.meshdependent import MeshDependent
from edelweissfe.rigidbodies.discreterigidbody import DiscreteRigidBody
from edelweissfe.sets.nodeset import NodeSet
from edelweissfe.timesteppers.timestep import TimeStep
from edelweissfe.utils.schema import buildSchemaFromOptions, schemaField

"""
A penalty based unilateral contact constraint between a node set of ordinary FE nodes and the
surface of a :class:`~edelweissfe.rigidbodies.discreterigidbody.DiscreteRigidBody`.

This constraint is a :class:`~edelweissfe.models.meshdependent.MeshDependent`: if an AMR refinement
(e.g. :mod:`~edelweissfe.modelmodifiers.adaptivity.hadaptivity`) adds nodes to the watched slave
``nSet`` (a boundary reaching into a refined region), the new nodes are picked up and protected from
penetrating the rigid body at the constraint's own next :meth:`updateConnectivity` tick -- no
separate wiring needed. There is no per-slave history to preserve across the rebuild (unlike the
deformable-surface/tie facet constraints): every quantity is recomputed fresh from the current
geometry every Newton iteration.
"""


@dataclass(frozen=True)
class NodeToDiscreteRigidBodyPenaltySchema:
    """The options this constraint accepts, owned by this module and never mutated from
    outside it.

    The update-type option is spelled ``type`` in the input file but the field is named
    ``contactType`` here -- a dataclass field literally called ``type`` would shadow the builtin,
    which this project's conventions avoid. ``penalty`` is declared ``required=True``, but is
    still given a ``default=None`` so the schema remains constructible on its own;
    ``buildSchemaFromOptions`` still enforces that an ``.inp`` file supplies it.
    """

    nSet: str | None = schemaField(
        description="The (slave) node set to be protected from penetrating the rigid body.",
        dtype=str,
        default=None,
        required=True,
    )
    rigidBody: str | None = schemaField(
        description="The name of the discrete rigid body (as registered in model.rigidBodies).",
        dtype=str,
        default=None,
        required=True,
    )
    penalty: float | None = schemaField(
        description="The numerical penalty value.", dtype=float, default=None, required=True
    )
    contactType: str = schemaField(
        description="The formulation type: 'linear' (linear force, constant stiffness with jump) "
        "or 'quadratic' (quadratic force, linear stiffness).",
        dtype=str,
        default="linear",
        optionName="type",
    )
    searchDistance: float | None = schemaField(
        description="An optional broadphase distance (passed on to the rigid body's surface "
        "query) for culling nodes far away from the rigid body. If not given, every slave node is "
        "queried exactly every iteration.",
        dtype=float,
        default=None,
    )


def _skew(v: np.ndarray) -> np.ndarray:
    """The skew-symmetric cross-product matrix of a 3-vector, such that ``_skew(v) @ x == v x x``."""
    return np.array(
        [
            [0.0, -v[2], v[1]],
            [v[2], 0.0, -v[0]],
            [-v[1], v[0], 0.0],
        ]
    )


def _rightJacobianSO3(theta: np.ndarray) -> np.ndarray:
    """The right Jacobian of the SO(3) exponential map at the rotation pseudo-vector ``theta``.

    For a fixed body-frame vector :math:`\\bar{\\mathbf{v}}` and
    :math:`R(\\boldsymbol{\\theta}) = \\exp(\\mathrm{skew}(\\boldsymbol{\\theta}))`, this satisfies

    .. math::
        \\frac{\\partial (R(\\boldsymbol{\\theta}) \\bar{\\mathbf{v}})}{\\partial \\boldsymbol{\\theta}}
        = -\\mathrm{skew}(R(\\boldsymbol{\\theta}) \\bar{\\mathbf{v}}) \\, R(\\boldsymbol{\\theta}) \\,
        J_r(\\boldsymbol{\\theta})

    i.e. it maps a perturbation of the *stored, total* pseudo-vector DOF onto the corresponding
    spatial rotation increment -- exact for any accumulated rotation, not just a small-angle
    approximation (which would correspond to ``J_r(theta) = I``).
    """
    angle = np.linalg.norm(theta)
    K = _skew(theta)
    if angle < 1e-8:
        return np.eye(3) - 0.5 * K + np.dot(K, K) / 6.0
    return np.eye(3) - (1.0 - np.cos(angle)) / angle**2 * K + (angle - np.sin(angle)) / angle**3 * np.dot(K, K)


class DiscreteRigidBodyContactStiffnessView:
    """Provides structured 2-D sub-views for the sparse stiffness matrix slice of
    :class:`Constraint`.

    Only the reference point (RP) self-block, and the per-slave self-block and slave-RP coupling
    blocks are populated -- there is no coupling between different slave nodes.

    Attributes
    ----------
    K_rprp : numpy.ndarray
        2-D view of shape ``(rprp_dof, rprp_dof)`` for the RP translation+rotation self-block,
        shared (and accumulated in-place) across all slave nodes.
    K_pp : list[numpy.ndarray]
        List of ``nSlaves`` views of shape ``(nDim, nDim)``, the self-block of each slave node.
    K_prp : list[numpy.ndarray]
        List of ``nSlaves`` views of shape ``(nDim, rprp_dof)``, slave-to-RP coupling.
    K_rpp : list[numpy.ndarray]
        List of ``nSlaves`` views of shape ``(rprp_dof, nDim)``, RP-to-slave coupling (transpose of
        ``K_prp``).
    """

    def __init__(self, flat_array: np.ndarray, nDim: int, nRot: int, nSlaves: int):
        rprpDof = nDim + nRot
        kRprpSize = rprpDof**2

        self.K_rprp = flat_array[0:kRprpSize].reshape((rprpDof, rprpDof))

        self.K_pp = []
        self.K_prp = []
        self.K_rpp = []

        offset = kRprpSize
        for _ in range(nSlaves):
            pp = flat_array[offset : offset + nDim * nDim].reshape((nDim, nDim))
            offset += nDim * nDim

            prp = flat_array[offset : offset + nDim * rprpDof].reshape((nDim, rprpDof))
            offset += nDim * rprpDof

            rpp = flat_array[offset : offset + rprpDof * nDim].reshape((rprpDof, nDim))
            offset += rprpDof * nDim

            self.K_pp.append(pp)
            self.K_prp.append(prp)
            self.K_rpp.append(rpp)


class Constraint(ConstraintBase, MeshDependent):
    """
    Penalty based unilateral contact between a slave node set and a discrete rigid body.

    Notes
    -----
    For a slave node :math:`s` at current position :math:`\\mathbf{x}_s`, the rigid body's surface
    query returns the signed distance :math:`d_s` (negative when penetrating) and outward unit
    normal :math:`\\mathbf{n}_s` of the closest surface point. The contact is active whenever
    :math:`d_s < 0`, with gap :math:`g_s = -d_s`.

    With :math:`\\mathbf{r}_s = \\mathbf{x}_s - \\mathbf{x}_{RP}` the moment arm of the contact point
    about the rigid body's current reference point (RP) position, the residual is assembled from

    .. math::
        \\mathbf{w}_s = \\begin{bmatrix} -\\mathbf{n}_s & \\mathbf{n}_s &
        \\text{dPhysicalSpin\\_dTheta}^T (\\mathbf{r}_s \\times \\mathbf{n}_s) \\end{bmatrix}
        \\, , \\qquad \\text{dPhysicalSpin\\_dTheta} = R(\\boldsymbol{\\theta}_{RP}) \\,
        J_r(\\boldsymbol{\\theta}_{RP})

    where :math:`J_r` is the right Jacobian of the SO(3) exponential map (see
    :func:`_rightJacobianSO3`), and ``dPhysicalSpin_dTheta`` maps a perturbation of the *stored*
    rotation pseudo-vector DOF onto the physical (spatial) infinitesimal rotation it actually
    produces (see :meth:`Constraint.applyConstraint`). The plain physical moment
    :math:`\\mathbf{r}_s \\times \\mathbf{n}_s` is only the exact gap gradient w.r.t. the RP
    rotation DOF when the RP has zero *total accumulated* rotation (where
    ``dPhysicalSpin_dTheta = I``); in general the ``dPhysicalSpin_dTheta``:math:`^T` factor is
    required because the rotation DOF stores the raw total pseudo-vector (fed directly through
    :meth:`~edelweissfe.rigidbodies.discreterigidbody.DiscreteRigidBody.rotationMatrixFromPseudoVector`
    every iteration, not a per-increment-reset relative rotation) -- incrementing that raw
    coordinate by :math:`\\delta\\boldsymbol{\\theta}` does not correspond to a physical spin of
    :math:`\\delta\\boldsymbol{\\theta}` unless the RP is currently unrotated, since the exponential
    map does not compose additively (:math:`\\exp(\\boldsymbol{\\theta}+\\delta\\boldsymbol{\\theta})
    \\neq \\exp(\\boldsymbol{\\theta})\\exp(\\delta\\boldsymbol{\\theta})`). Without this factor, this
    term is a few % off at ~0.1 rad accumulated rotation and ~15% off at ~0.4 rad (verified against
    finite differences of the underlying closed-form gap function).

    The penalty normal force is :math:`f_n = k \\, g_s` (``type=linear``, constant tangent :math:`k`)
    or :math:`f_n = \\tfrac{1}{2} k \\, g_s^2` (``type=quadratic``, tangent :math:`k \\, g_s`), and is
    assembled -- exactly like :mod:`~edelweissfe.constraints.nodetorigidsurfacepenalty` -- as

    .. math::
        P_{ext} \\mathrel{-{=}} f_n \\, \\mathbf{w}_s \\, , \\qquad
        K \\mathrel{+{=}} k \\, (\\mathbf{w}_s \\otimes \\mathbf{w}_s)

    Both :math:`\\mathbf{n}_s` and :math:`\\mathbf{r}_s` are recomputed from the current, total
    solution every Newton iteration (no per-increment caching). Since the rigid body's surface is a
    triangulated (piecewise-planar) mesh, :math:`\\mathbf{n}_s` is exactly constant within a facet --
    the curvature contribution to the tangent vanishes identically there -- but it still rotates
    rigidly with the RP. On top of the secant term above, the tangent includes the *exact*
    linearization of :math:`\\mathbf{w}_s` with respect to all of
    :math:`(\\mathbf{u}_s, \\mathbf{u}_{RP}, \\boldsymbol{\\theta}_{RP})`, valid for any accumulated
    rotation, not just a small-angle approximation (verified against finite differences); see the
    inline comments in :meth:`Constraint.applyConstraint`. Because :math:`\\mathbf{w}_s` is now
    exactly the gap gradient (no residual approximation left), this tangent is the true, symmetric
    Hessian of the gap function restricted to the implemented blocks. Not included: the
    :math:`\\boldsymbol{\\theta}_{RP}\\boldsymbol{\\theta}_{RP}` self-block -- symmetric on its own
    (verified numerically) but requiring the second derivative of the SO(3) exponential map, which
    is not implemented. This is still only a convergence-rate issue, not a correctness bug (the
    residual itself is exact), but the omitted block is not negligible: measured by central finite
    differences, it is 40-60% of the largest entry in the assembled tangent across accumulated
    rotations from 0 to 0.4 rad. Facet-boundary (edge/vertex) normal discontinuities are also not
    smoothed, so no consistent tangent exists exactly there regardless.

    Currently only available for spatialdomain = 3D.
    """

    #: Option schema for this constraint, per OptionSchemaProvider.
    schema = NodeToDiscreteRigidBodyPenaltySchema

    def __init__(
        self,
        name: str,
        model: FEModel,
        nSet: NodeSet,
        rigidBody: DiscreteRigidBody,
        *,
        configuration: NodeToDiscreteRigidBodyPenaltySchema = NodeToDiscreteRigidBodyPenaltySchema(),
    ):
        super().__init__(name, model)

        if model.domainSize != 3:
            raise ValueError("nodeToDiscreteRigidBodyPenalty is currently only implemented for 3D models.")

        self.rigidBody = rigidBody
        self.rpNode = self.rigidBody.rpNode

        rigidBodyNodes = set(self.rigidBody.surfaceNodes) | {self.rpNode}
        if any(node in rigidBodyNodes for node in nSet):
            raise ValueError(
                f"nSet '{nSet.name}' for constraint '{name}' contains nodes belonging to rigid body "
                f"'{self.rigidBody.name}' (e.g. its surface or reference-point nodes) -- these carry no "
                "field variables and cannot be slave nodes. Exclude them from the node set."
            )

        self.penalty = configuration.penalty
        self.type = configuration.contactType.lower()
        if self.type not in ["linear", "quadratic"]:
            raise ValueError(f"Constraint type '{self.type}' is not supported. Use 'linear' or 'quadratic'.")
        self.searchDistance = configuration.searchDistance

        self.nDim = model.domainSize
        self.nRot = 3
        self.rprpDof = self.nDim + self.nRot

        self._nSetName = nSet.name
        self._lastSeenTopologyVersion = model.topologyVersion
        self.slaveNodes = list(nSet)
        self._rebuildFromSlaveNodes()

        self.totalNormalForce = 0.0

    @classmethod
    def fromConstraintDefinition(
        cls, name: str, definition: dict, model: FEModel, journal: "Journal" = None
    ) -> "Constraint":
        """Build this constraint from a parsed ``*constraint`` definition. See
        :class:`~edelweissfe.constraints.base.constraintbase.ConstraintBase` for why this is
        separate from ``__init__``."""
        configuration = buildSchemaFromOptions(cls.schema, definition)
        return cls(
            name,
            model,
            model.nodeSets[configuration.nSet],
            model.rigidBodies[configuration.rigidBody],
            configuration=configuration,
        )

    def _rebuildFromSlaveNodes(self) -> None:
        """(Re)derive every quantity that depends on the slave node list/count."""

        self.nSlaves = len(self.slaveNodes)
        self._referenceCoords = np.array([n.coordinates for n in self.slaveNodes])

        self._nodes = self.slaveNodes + [self.rpNode]
        self._fieldsOnNodes = [["displacement"]] * self.nSlaves + [["displacement", "rotation"]]
        self._nDof = self.nSlaves * self.nDim + self.rprpDof

        # Local DOF index blocks, in the order [slave_0, slave_1, ..., RP displacement, RP rotation].
        self._indicesOfSlaveInLocal = [list(range(s * self.nDim, (s + 1) * self.nDim)) for s in range(self.nSlaves)]
        self._indicesOfRPDispInLocal = list(range(self.nSlaves * self.nDim, self.nSlaves * self.nDim + self.nDim))
        self._indicesOfRPRotInLocal = list(
            range(self.nSlaves * self.nDim + self.nDim, self.nSlaves * self.nDim + self.rprpDof)
        )
        self._indicesOfRPInLocal = self._indicesOfRPDispInLocal + self._indicesOfRPRotInLocal

    def reconcile(self, model: FEModel, change) -> bool:
        """Refresh the slave node list from the (possibly grown) watched ``nSet``."""

        if not change.touchesNodeSet(self._nSetName):
            return False
        self.slaveNodes = list(model.nodeSets[self._nSetName])
        self._rebuildFromSlaveNodes()
        return True

    def updateConnectivity(self, model: FEModel) -> bool:
        return self.reconcileIfChanged(model)

    @property
    def nodes(self) -> list:
        return self._nodes

    @property
    def fieldsOnNodes(self) -> list:
        return self._fieldsOnNodes

    @property
    def nDof(self) -> int:
        return self._nDof

    def getVIJContributionSize(self) -> int:
        """No coupling between different slave nodes: one shared RP self-block, plus per-slave
        self-block and slave-RP coupling blocks."""
        return self.rprpDof**2 + self.nSlaves * (self.nDim**2 + 2 * self.nDim * self.rprpDof)

    def shapeVIJContribution(self, flat_view: np.ndarray) -> DiscreteRigidBodyContactStiffnessView:
        return DiscreteRigidBodyContactStiffnessView(flat_view, nDim=self.nDim, nRot=self.nRot, nSlaves=self.nSlaves)

    def initializeVIJContribution(self, idcs: np.ndarray, I_: np.ndarray, J_: np.ndarray, offset: int) -> None:
        rprpDof = self.rprpDof
        k = offset

        rpIdcs = [idcs[i] for i in self._indicesOfRPInLocal]
        for i in range(rprpDof):
            for j in range(rprpDof):
                I_[k] = rpIdcs[i]
                J_[k] = rpIdcs[j]
                k += 1

        for s in range(self.nSlaves):
            pIdcs = [idcs[i] for i in self._indicesOfSlaveInLocal[s]]

            for i in range(self.nDim):
                for j in range(self.nDim):
                    I_[k] = pIdcs[i]
                    J_[k] = pIdcs[j]
                    k += 1

            for i in range(self.nDim):
                for j in range(rprpDof):
                    I_[k] = pIdcs[i]
                    J_[k] = rpIdcs[j]
                    k += 1

            for i in range(rprpDof):
                for j in range(self.nDim):
                    I_[k] = rpIdcs[i]
                    J_[k] = pIdcs[j]
                    k += 1

    def applyConstraint(
        self,
        U_np: np.ndarray,
        dU: np.ndarray,
        PExt: np.ndarray,
        K: DiscreteRigidBodyContactStiffnessView,
        timeStep: TimeStep,
    ):
        self.totalNormalForce = 0.0

        uSlaves = np.array([U_np[idcs] for idcs in self._indicesOfSlaveInLocal])
        coords = self._referenceCoords + uSlaves

        # The rigid body's own querySurface() would otherwise fall back to
        # getCurrentKinematics(), which reads the RP's pose from NodeFields --
        # only refreshed once per *converged* increment. Mid-Newton-iteration,
        # that pose would be one increment stale relative to U_np (used above for
        # the slave coordinates), decoupling the RP DOFs from the contact residual.
        # Compute the current RP kinematics directly from this constraint's own
        # local U_np slice instead, so slave and RP kinematics are always consistent.
        u_rp = U_np[self._indicesOfRPDispInLocal]
        theta_rp = U_np[self._indicesOfRPRotInLocal]
        R = DiscreteRigidBody.rotationMatrixFromPseudoVector(theta_rp)
        kinematics = (u_rp, R, self.rpNode.coordinates)

        # dPhysicalSpin_dTheta maps a perturbation of the *stored* pseudo-vector DOF theta_rp onto the
        # physical (spatial) infinitesimal rotation it actually produces -- the identity only at
        # theta_rp = 0; see _rightJacobianSO3 and the class docstring.
        dPhysicalSpin_dTheta = R @ _rightJacobianSO3(theta_rp)

        dists, normals = self.rigidBody.querySurface(
            coords, proximityDistance=self.searchDistance, kinematics=kinematics
        )

        activeMask = dists < 0.0
        if not np.any(activeMask):
            return

        rpCurrent = self.rpNode.coordinates + u_rp

        for s in np.where(activeMask)[0]:
            n_s = normals[s]
            r_s = coords[s] - rpCurrent
            g = -dists[s]

            if self.type == "linear":
                f_n = self.penalty * g
                stiffness = self.penalty
            else:
                f_n = 0.5 * self.penalty * g**2
                stiffness = self.penalty * g

            w_p = -n_s
            # dPhysicalSpin_dTheta^T maps the physical moment r_s x n_s onto the generalized force
            # conjugate to the *stored* rotation pseudo-vector DOF -- exact for any accumulated RP
            # rotation, not just theta_rp=0 (where dPhysicalSpin_dTheta=I and this reduces to the naive
            # r_s x n_s). See class docstring.
            w_rp = np.concatenate((n_s, dPhysicalSpin_dTheta.T @ np.cross(r_s, n_s)))

            pIdcs = self._indicesOfSlaveInLocal[s]
            rpIdcs = self._indicesOfRPInLocal

            PExt[pIdcs] -= f_n * w_p
            PExt[rpIdcs] -= f_n * w_rp

            K.K_pp[s] += stiffness * np.outer(w_p, w_p)
            K.K_prp[s] += stiffness * np.outer(w_p, w_rp)
            K.K_rpp[s] += stiffness * np.outer(w_rp, w_p)
            K.K_rprp += stiffness * np.outer(w_rp, w_rp)

            # Geometric stiffness from the rigid rotation of the (locally flat) facet normal, and from
            # dPhysicalSpin_dTheta(theta_rp) itself, with the RP rotation. Each block below is named
            # after the derivative it is (see the class docstring for the theta_rp-theta_rp self-block,
            # which is not included here):
            #   dn_dTheta    = d(n_s)/d(theta_rp)     = -skew(n_s) @ dPhysicalSpin_dTheta
            #   dMoment_dUs  = d(r_s x n_s)/d(u_s)     = -skew(n_s)
            #   dMoment_dUrp = d(r_s x n_s)/d(u_rp)    = skew(n_s)
            # so that:
            #   d(w_p)/d(theta_rp)       = -dn_dTheta
            #   d(w_rp,disp)/d(theta_rp) =  dn_dTheta
            #   d(w_rp,rot)/d(u_s)       =  dPhysicalSpin_dTheta^T @ dMoment_dUs
            #   d(w_rp,rot)/d(u_rp)      =  dPhysicalSpin_dTheta^T @ dMoment_dUrp
            dn_dTheta = -_skew(n_s) @ dPhysicalSpin_dTheta
            dMoment_dUs = -_skew(n_s)
            dMoment_dUrp = _skew(n_s)

            K.K_prp[s][:, self.nDim :] += f_n * (-dn_dTheta)
            K.K_rpp[s][self.nDim :, :] += f_n * (dPhysicalSpin_dTheta.T @ dMoment_dUs)
            K.K_rprp[0 : self.nDim, self.nDim :] += f_n * dn_dTheta
            K.K_rprp[self.nDim :, 0 : self.nDim] += f_n * (dPhysicalSpin_dTheta.T @ dMoment_dUrp)

            self.totalNormalForce += f_n
