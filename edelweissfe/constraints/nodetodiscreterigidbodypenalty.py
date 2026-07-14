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

from edelweissfe.constraints.base.constraintbase import ConstraintBase
from edelweissfe.models.femodel import FEModel
from edelweissfe.timesteppers.timestep import TimeStep
from edelweissfe.utils.caseinsensitivedict import CaseInsensitiveDict
from edelweissfe.utils.inputlanguage import InputLanguage, Module
from edelweissfe.utils.misc import (
    caseInsensitiveKwargsChecker,
    castKwargsValuesAndAddDefaults,
)

"""
A penalty based unilateral contact constraint between a node set of ordinary FE nodes and the
surface of a :class:`~edelweissfe.rigidbodies.discreterigidbody.DiscreteRigidBody`.
"""

module = Module(
    "nodeToDiscreteRigidBodyPenalty",
    "A penalty based unilateral contact constraint preventing nodes of a node set from penetrating "
    "the surface of a discrete rigid body.",
)

inputLanguage = InputLanguage()

keyword = "constraint"
if keyword in inputLanguage:
    inputLanguage[keyword].addModule(module)

module.addRequiredArg("nSet", "The (slave) node set to be protected from penetrating the rigid body.", str)
module.addRequiredArg("rigidBody", "The name of the discrete rigid body (as registered in model.rigidBodies).", str)
module.addRequiredArg("penalty", "The numerical penalty value.", float)

module.addOptionalArg(
    "type",
    "The formulation type: 'linear' (linear force, constant stiffness with jump) or 'quadratic' (quadratic "
    "force, linear stiffness).",
    str,
    "linear",
)
module.addOptionalArg(
    "searchDistance",
    "An optional broadphase distance (passed on to the rigid body's surface query) for culling nodes far "
    "away from the rigid body. If not given, every slave node is queried exactly every iteration.",
    float,
    None,
)

documentation = [module]


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


class Constraint(ConstraintBase):
    """
    Penalty based unilateral contact between a slave node set and a discrete rigid body.

    Theoretical background
    -----------------------
    For a slave node :math:`s` at current position :math:`\\mathbf{x}_s`, the rigid body's surface
    query returns the signed distance :math:`d_s` (negative when penetrating) and outward unit
    normal :math:`\\mathbf{n}_s` of the closest surface point. The contact is active whenever
    :math:`d_s < 0`, with gap :math:`g_s = -d_s`.

    With :math:`\\mathbf{r}_s = \\mathbf{x}_s - \\mathbf{x}_{RP}` the moment arm of the contact point
    about the rigid body's current reference point (RP) position, the gradient of the gap with
    respect to the coupled degrees of freedom (slave displacement, RP displacement, RP rotation) is

    .. math::
        \\mathbf{w}_s = \\begin{bmatrix} -\\mathbf{n}_s & \\mathbf{n}_s & \\mathbf{r}_s \\times \\mathbf{n}_s \\end{bmatrix}

    The penalty normal force is :math:`f_n = k \\, g_s` (``type=linear``, constant tangent :math:`k`)
    or :math:`f_n = \\tfrac{1}{2} k \\, g_s^2` (``type=quadratic``, tangent :math:`k \\, g_s`), and is
    assembled -- exactly like :mod:`~edelweissfe.constraints.nodetorigidsurfacepenalty` -- as

    .. math::
        P_{ext} \\mathrel{-{=}} f_n \\, \\mathbf{w}_s \\, , \\qquad
        K \\mathrel{+{=}} k \\, (\\mathbf{w}_s \\otimes \\mathbf{w}_s)

    Both :math:`\\mathbf{n}_s` and :math:`\\mathbf{r}_s` are recomputed from the current, total
    solution every Newton iteration (no per-increment caching), but treated as locally constant when
    forming the tangent -- i.e., the geometric stiffness contribution from the curvature of the rigid
    surface and from the rotation of :math:`\\mathbf{r}_s` is neglected, the same simplification used
    by EdelweissMeshfree's analogous ``DiscreteRigidBodyPenaltyContact``.

    Currently only available for spatialdomain = 3D.
    """

    @caseInsensitiveKwargsChecker([kw.name for kw in module.requiredArgs], [kw.name for kw in module.optionalArgs])
    @castKwargsValuesAndAddDefaults(module)
    def __init__(self, name: str, model: FEModel, *args, **kwargs):
        super().__init__(name, model, *args, **kwargs)

        if model.domainSize != 3:
            raise ValueError("nodeToDiscreteRigidBodyPenalty is currently only implemented for 3D models.")

        kwargs = CaseInsensitiveDict(kwargs)

        self.rigidBody = model.rigidBodies[kwargs["rigidBody"]]
        self.rpNode = self.rigidBody.rpNode

        self.slaveNodes = [node for node in model.nodeSets[kwargs["nSet"]] if node is not self.rpNode]
        self.nSlaves = len(self.slaveNodes)

        self.penalty = kwargs["penalty"]
        self.type = kwargs["type"].lower()
        if self.type not in ["linear", "quadratic"]:
            raise ValueError(f"Constraint type '{self.type}' is not supported. Use 'linear' or 'quadratic'.")
        self.searchDistance = kwargs["searchDistance"]

        self.nDim = model.domainSize
        self.nRot = 3
        self.rprpDof = self.nDim + self.nRot

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

        dists, normals = self.rigidBody.querySurface(coords, proximityDistance=self.searchDistance)

        activeMask = dists < 0.0
        if not np.any(activeMask):
            return

        rpCurrent = self.rpNode.coordinates + U_np[self._indicesOfRPDispInLocal]

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
            w_rp = np.concatenate((n_s, np.cross(r_s, n_s)))

            pIdcs = self._indicesOfSlaveInLocal[s]
            rpIdcs = self._indicesOfRPInLocal

            PExt[pIdcs] -= f_n * w_p
            PExt[rpIdcs] -= f_n * w_rp

            K.K_pp[s] += stiffness * np.outer(w_p, w_p)
            K.K_prp[s] += stiffness * np.outer(w_p, w_rp)
            K.K_rpp[s] += stiffness * np.outer(w_rp, w_p)
            K.K_rprp += stiffness * np.outer(w_rp, w_rp)

            self.totalNormalForce += f_n
