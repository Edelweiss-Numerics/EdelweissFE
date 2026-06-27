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
# Created on Tue Dec 11 11:21:39 2018

# @author: Matthias Neuner

import numpy as np

from edelweissfe.constraints.base.constraintbase import ConstraintBase
from edelweissfe.utils.caseinsensitivedict import CaseInsensitiveDict
from edelweissfe.utils.inputlanguage import InputLanguage, Module
from edelweissfe.utils.misc import (
    caseInsensitiveKwargsChecker,
    castKwargsValuesAndAddDefaults,
)

module = Module("rigidbody", "A rigid body constraint tying nodes to a reference point.")

inputLanguage = InputLanguage()

keyword = "constraint"
if keyword in inputLanguage:
    inputLanguage[keyword].addModule(module)

module.addRequiredArg("nSet", "Node set to tie.", str)
module.addRequiredArg("referencePoint", "Node set containing only the reference point.", str)

documentation = [module]


class RigidBodyStiffnessView:
    """Provides structured 2-D sub-views for the sparse rigid body stiffness matrix slice.

    Theoretical Background
    ----------------------
    In a geometrically exact rigid body constraint, a set of slave nodes is tied to a reference point (RP).
    For each slave node :math:`s`, we define:
      - Nodal displacement DOFs: :math:`\\mathbf{u}_s` (size ``nDim``)
      - RP displacement DOFs: :math:`\\mathbf{u}_{RP}` (size ``nDim``)
      - RP rotation DOFs: :math:`\\boldsymbol{\\phi}_{RP}` (size ``nRot``)
      - Lagrange multipliers: :math:`\\boldsymbol{\\lambda}_s` (size ``nDim``)

    The constraint equation enforcing the rigid body distance :math:`\\mathbf{d}_0^s` is nonlinear:
    .. math::
       \\mathbf{g}_s(\\mathbf{u}_s, \\mathbf{u}_{RP}, \\boldsymbol{\\phi}_{RP}) =
       -\\mathbf{d}_0^s - (\\mathbf{u}_s - \\mathbf{u}_{RP}) + \\mathbf{T}(\\boldsymbol{\\phi}_{RP}) \\mathbf{d}_0^s = \\mathbf{0}

    where :math:`\\mathbf{T}` is the rotation matrix of the RP.

    The gradient of the constraint with respect to the coupled DOFs (ordered as slave displacement,
    RP displacement, and RP rotation) is:
    .. math::
       \\mathbf{G}_s = \\frac{\\partial \\mathbf{g}_s}{\\partial \\mathbf{U}} =
       \\begin{bmatrix}
         -\\mathbf{I}_{\\text{nDim}} &
         +\\mathbf{I}_{\\text{nDim}} &
         \\frac{\\partial \\mathbf{T}}{\\partial \\boldsymbol{\\phi}_{RP}} \\mathbf{d}_0^s
       \\end{bmatrix}

    which is of size ``nDim × nUc``, where ``nUc = nDim + nDim + nRot``.

    The contribution of the constraint to the tangent stiffness matrix arises from:
      1. The first derivative of the constraint equations, which gives the off-diagonal coupling blocks:
         :math:`\\mathbf{K}_{UL}^s = \\mathbf{G}_s^T` (size ``nUc × nDim``) and
         :math:`\\mathbf{K}_{LU}^s = \\mathbf{G}_s` (size ``nDim × nUc``).
      2. The second derivative of the nonlinear constraint with respect to the rotation DOFs, scaled by the
         Lagrange multipliers, which gives the RP rotation-rotation stiffness block:
         :math:`\\mathbf{K}_{UU}^s = \\sum_s \\boldsymbol{\\lambda}_s^T \\frac{\\partial^2 \\mathbf{T}}{\\partial \\boldsymbol{\\phi}_{RP}^2} \\mathbf{d}_0^s` (size ``nRot × nRot``).

    Implementation & Sparse Layout
    ------------------------------
    To avoid constructing high-dimensional empty matrices or performing expensive index lookups,
    the global sparse VIJ system matrix allocates a contiguous 1-D slice for each constraint's contributions.
    For this rigid body constraint, the 1-D slice (of size ``nRot² + nSlaves × 2 × nUc × nDim``) has the layout:
      - ``[0 : nRot²]``: Contiguous entries for the :math:`\\mathbf{K}_{UU}` block.
      - For each slave node :math:`s`:
        - ``K_UL`` block of size ``nUc × nDim``.
        - ``K_LU`` block of size ``nDim × nUc``.

    This class wraps the flat 1-D numpy array and exposes reshaped 2-D views (or lists of views)
    that share the same memory. Any additions or writes to ``K_UU``, ``K_UL[s]``, or ``K_LU[s]``
    directly modify the global VIJ system matrix in-place.

    Attributes
    ----------
    K_UU : np.ndarray
        2-D view of shape ``(nRot, nRot)`` representing the accumulated reference point rotation-rotation
        tangent stiffness contribution.
    K_UL : list[np.ndarray]
        List of ``nSlaves`` 2-D views, each of shape ``(nUc, nDim)``, representing the gradient coupling between
        the coupled DOFs (slave displacement, RP displacement, RP rotation) and the Lagrange multipliers.
    K_LU : list[np.ndarray]
        List of ``nSlaves`` 2-D views, each of shape ``(nDim, nUc)``, representing the transpose of the gradient coupling.
    """

    def __init__(self, flat_array: np.ndarray, nRot: int, nUc: int, nDim: int, nSlaves: int):
        self._flat_array = flat_array
        self._nRot = nRot
        self._nUc = nUc
        self._nDim = nDim
        self._nSlaves = nSlaves

        kuu_size = nRot**2
        entries_per_slave = 2 * nUc * nDim

        # 2-D view of RP rotation block
        self.K_UU = flat_array[0:kuu_size].reshape((nRot, nRot))

        # Lists of 2-D views for coupling blocks of each slave node
        self.K_UL = [
            flat_array[kuu_size + s * entries_per_slave : kuu_size + s * entries_per_slave + nUc * nDim].reshape(
                (nUc, nDim)
            )
            for s in range(nSlaves)
        ]
        self.K_LU = [
            flat_array[kuu_size + s * entries_per_slave + nUc * nDim : kuu_size + (s + 1) * entries_per_slave].reshape(
                (nDim, nUc)
            )
            for s in range(nSlaves)
        ]


class Constraint(ConstraintBase):
    """
    Geometrically exact rigid body constraint: Constrains a nodeset to a reference point.
    Currently only available for spatialdomain = 3D.
    """

    @caseInsensitiveKwargsChecker([kw.name for kw in module.requiredArgs], [kw.name for kw in module.optionalArgs])
    @castKwargsValuesAndAddDefaults(module)
    def __init__(self, name, model, *args, **kwargs):
        super().__init__(name, model, *args, **kwargs)

        # self.name = name

        kwargs = CaseInsensitiveDict(kwargs)

        self.nDim = model.domainSize
        nDim = self.nDim

        if nDim == 2:
            raise Exception("rigid body constraint not yet implemented for 2D")

        rbNset = kwargs["nSet"]
        nodeSets = model.nodeSets

        if len(nodeSets[kwargs["referencePoint"]]) > 1:
            raise Exception(
                "node set for reference point '{:}' contains more than one node".format(kwargs["referencePoint"])
            )

        self.referencePoint = nodeSets[kwargs["referencePoint"]][0]

        slaveNodeSet = nodeSets[rbNset]  # slave node set may contain the reference point

        # reference point is removed (if present) and node set is converted to list
        self.slaveNodes = [node for node in slaveNodeSet if not node == self.referencePoint]

        nRot = 3
        nSlaves = len(self.slaveNodes)

        self.indicesOfSlaveNodesInP = [[i * nDim + j for j in range(nDim)] for i in range(nSlaves)]
        self.indicesOfRPUinP = [nSlaves * nDim + j for j in range(nDim)]
        self.indicesOfRPPhiInP = [nSlaves * nDim + nDim + j for j in range(nRot)]

        # all nodes

        slaveNodeSet = nodeSets[rbNset]  # slave node set may contain the reference point

        # reference point is removed (if present) and node set is converted to list
        self.slaveNodes = [node for node in slaveNodeSet if not node == self.referencePoint]

        # list of all nodes including RP at end
        self._nodes = self.slaveNodes + [self.referencePoint]

        self.slaveNodesFields = [["displacement"]] * nSlaves
        self.referencePointFields = [["displacement", "rotation"]]
        self._fieldsOnNodes = self.slaveNodesFields + self.referencePointFields

        nConstraints = nSlaves * nDim
        nRp = 1

        self.nDofsOnNodes = nDim * (nSlaves + nRp) + nRot

        self.distancesSlaveNodeRP = [s.coordinates - self.referencePoint.coordinates for s in self.slaveNodes]

        self._nDof = self.nDofsOnNodes + nConstraints
        self.sizeStiffness = self._nDof * self._nDof

        self.nConstraints = nConstraints

        self.nRot = 3

        # Number of U-type DOFs that appear in each per-slave constraint equation:
        # slave_i displacement (nDim) + RP displacement (nDim) + RP rotation (nRot).
        self._nUCoupledPerSlave = nDim + nDim + nRot

        self._reactions = np.zeros(self.nRot + self.nDim)

    @property
    def nodes(self) -> list:
        return self._nodes

    @property
    def fieldsOnNodes(self) -> list:
        return self._fieldsOnNodes

    @property
    def nDof(self) -> int:
        return self._nDof

    def update(self, options):
        """No updates are possible for this constraint."""

    def getNumberOfAdditionalNeededScalarVariables(self):
        return self.nConstraints

    def getVIJContributionSize(self) -> int:
        """Return the actual (sparse) number of VIJ entries for this constraint.

        Each slave i contributes a (nDim+nDim+nRot) × nDim block for K_UL and its
        transpose for K_LU.  Additionally there is one shared nRot × nRot block in K_UU
        for the reference-point rotation DOFs (accumulated over all slaves).

        Total = nRot² + nSlaves × 2 × nUCoupledPerSlave × nDim
              = 9    + nSlaves × 54  (in 3-D)
        """
        return self.nRot**2 + len(self.slaveNodes) * 2 * self._nUCoupledPerSlave * self.nDim

    def shapeVIJContribution(self, flat_view: np.ndarray) -> RigidBodyStiffnessView:
        """Shape the flat VIJ values slice for this constraint using RigidBodyStiffnessView."""
        return RigidBodyStiffnessView(
            flat_view,
            nRot=self.nRot,
            nUc=self._nUCoupledPerSlave,
            nDim=self.nDim,
            nSlaves=len(self.slaveNodes),
        )

    def initializeVIJContribution(self, idcs: np.ndarray, I_: np.ndarray, J_: np.ndarray, offset: int) -> None:
        """Fill the VIJ index arrays with the sparse pattern of this constraint.

        Layout (starting at ``offset``):

        * [0 : nRot²)          – K_UU rotation block
          (rows / cols = RP rotation DOFs in global index space)
        * for each slave *s*:
          * [9 + s*2*nUc*nDim  : 9 + s*2*nUc*nDim + nUc*nDim)
            K_UL block: nUc rows (slave_s ∪ RP_u ∪ RP_phi) × nDim cols (Lambda_s)
          * [9 + s*2*nUc*nDim + nUc*nDim : 9 + (s+1)*2*nUc*nDim)
            K_LU block: nDim rows (Lambda_s) × nUc cols (slave_s ∪ RP_u ∪ RP_phi)

        where nUc = nDim + nDim + nRot = 9 (in 3-D).
        """
        nDim = self.nDim
        nRot = self.nRot
        nSlaves = len(self.slaveNodes)
        nU = self.nDofsOnNodes
        nUc = self._nUCoupledPerSlave  # 9

        k = offset

        # K_UU: rotation-rotation block of the reference point
        for ri in range(nRot):
            for rj in range(nRot):
                I_[k] = idcs[nU - nRot + ri]
                J_[k] = idcs[nU - nRot + rj]
                k += 1

        # Per-slave K_UL and K_LU blocks
        for s in range(nSlaves):
            indcsU_s = self.indicesOfSlaveNodesInP[s] + self.indicesOfRPUinP + self.indicesOfRPPhiInP
            L0_local = nU + s * nDim  # start of Lambda_s in the local DOF vector

            # K_UL: nUc × nDim
            for iu in range(nUc):
                for il in range(nDim):
                    I_[k] = idcs[indcsU_s[iu]]
                    J_[k] = idcs[L0_local + il]
                    k += 1

            # K_LU: nDim × nUc (transpose of K_UL)
            for il in range(nDim):
                for iu in range(nUc):
                    I_[k] = idcs[L0_local + il]
                    J_[k] = idcs[indcsU_s[iu]]
                    k += 1

    def Rz_2D(self, phi, derivative):
        phi = phi + np.pi / 2 * derivative
        return np.array(
            [
                [np.cos(phi), -np.sin(phi)],
                [np.sin(phi), +np.cos(phi)],
            ]
        )

    def Rx_3D(self, phi, derivative):
        phi = phi + np.pi / 2 * derivative
        i = 0.0 if derivative > 0 else 1.0
        return np.array(
            [
                [
                    i,
                    0,
                    0,
                ],
                [0, np.cos(phi), -np.sin(phi)],
                [0, np.sin(phi), +np.cos(phi)],
            ]
        )

    def Ry_3D(self, phi, derivative):
        phi = phi + np.pi / 2 * derivative
        i = 0.0 if derivative > 0 else 1.0
        return np.array(
            [
                [
                    np.cos(phi),
                    0,
                    +np.sin(phi),
                ],
                [0, i, 0],
                [-np.sin(phi), 0, +np.cos(phi)],
            ]
        )

    def Rz_3D(self, phi, derivative):
        phi = phi + np.pi / 2 * derivative
        i = 0.0 if derivative > 0 else 1.0
        return np.array(
            [
                [np.cos(phi), -np.sin(phi), 0],
                [np.sin(phi), +np.cos(phi), 0],
                [
                    0,
                    0,
                    i,
                ],
            ]
        )

    def applyConstraint(self, U_np, dU, PExt, K, timeStep):
        """Apply the rigid body constraint.

        ``K`` is received as a RigidBodyStiffnessView object.
        """
        nConstraints = self.nConstraints
        nDim = self.nDim
        nRot = self.nRot
        nUc = self._nUCoupledPerSlave  # 9 in 3-D

        nU = self._nDof - nConstraints  # nDofs (disp., rot.) without Lagrangian multipliers
        nSlaves = len(self.slaveNodes)

        URp = U_np[self.indicesOfRPUinP]
        PhiRp = U_np[self.indicesOfRPPhiInP]
        Lambdas = U_np[nU:].reshape((nDim, -1), order="F")

        G = np.zeros((nDim, nUc))
        H = np.zeros((nDim, nUc, nUc))

        # dg/dU_Node and dg/dU_RP (displacement parts are constant)
        G[:, 0:nDim] = -np.identity(nDim)
        G[:, nDim : 2 * nDim] = +np.identity(nDim)

        if nDim == 3:
            Rx, Ry, Rz = self.Rx_3D, self.Ry_3D, self.Rz_3D

            RotationMatricesAndDerivatives = [
                [R(phi, derivative) for derivative in range(3)] for R, phi in zip((Rx, Ry, Rz), PhiRp)
            ]
            Rx = RotationMatricesAndDerivatives[0]
            Ry = RotationMatricesAndDerivatives[1]
            Rz = RotationMatricesAndDerivatives[2]

            T = Rz[0] @ Ry[0] @ Rx[0]

            RDerivativeProductsI = (
                Rz[0] @ Ry[0] @ Rx[1],
                Rz[0] @ Ry[1] @ Rx[0],
                Rz[1] @ Ry[0] @ Rx[0],
            )

            RDerivativeProductsII = (
                (
                    Rz[0] @ Ry[0] @ Rx[2],
                    Rz[0] @ Ry[1] @ Rx[1],
                    Rz[1] @ Ry[0] @ Rx[1],
                ),
                (
                    Rz[0] @ Ry[1] @ Rx[1],
                    Rz[0] @ Ry[2] @ Rx[0],
                    Rz[1] @ Ry[1] @ Rx[0],
                ),
                (
                    Rz[1] @ Ry[0] @ Rx[1],
                    Rz[1] @ Ry[1] @ Rx[0],
                    Rz[2] @ Ry[0] @ Rx[0],
                ),
            )

        self._reactions.fill(0.0)

        for s in range(nSlaves):
            d0 = self.distancesSlaveNodeRP[s]
            indcsUNode = self.indicesOfSlaveNodesInP[s]
            U_n = U_np[indcsUNode]
            Lambda = Lambdas[:, s]

            g = -d0 - (U_n - URp) + T @ d0

            for j in range(nRot):
                G[:, 2 * nDim + j] = RDerivativeProductsI[j] @ d0
                for k in range(nRot):
                    # only the rotation block is nonzero in H
                    H[:, 2 * nDim + j, 2 * nDim + k] = RDerivativeProductsII[j][k] @ d0

            indcsU = np.array(indcsUNode + self.indicesOfRPUinP + self.indicesOfRPPhiInP, dtype=int)

            L0 = nU + s * nDim  # local start of Lambda_s in PExt / U_np

            PExt[indcsU] -= Lambda.T @ G
            PExt[L0 : L0 + nDim] -= g

            # ---- Write to the sparse structured K view ----
            K.K_UU += np.einsum("i,ijk->jk", Lambda, H[:, -nRot:, -nRot:])
            K.K_UL[s] += G.T
            K.K_LU[s] += G

            self._reactions[0 : self.nDim] += Lambda
            self._reactions[self.nDim :] += np.cross(T @ d0, Lambda)
