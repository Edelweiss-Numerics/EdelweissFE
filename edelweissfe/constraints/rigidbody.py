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

        ``K`` is received as a **1-D** array whose entries correspond to the sparse
        VIJ pattern laid out by :meth:`initializeVIJContribution`:

        * ``K[0 : nRot²]``
          – K_UU rotation block (accumulated across all slaves, row-major).
        * for slave *s* (s = 0, …, nSlaves-1):

          * ``K[nRot² + s * 2*nUc*nDim  :  nRot² + s * 2*nUc*nDim + nUc*nDim]``
            – K_UL block for slave *s*, stored as ``G.T.flatten()``
            (row-major with *nUc* rows and *nDim* cols).
          * ``K[nRot² + s * 2*nUc*nDim + nUc*nDim  :  nRot² + (s+1) * 2*nUc*nDim]``
            – K_LU block for slave *s*, stored as ``G.flatten()``
            (row-major with *nDim* rows and *nUc* cols).

        where ``nUc = nDim + nDim + nRot`` (= 9 in 3-D).
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

        # Per-slave offset constants for the sparse K layout.
        kuu_size = nRot**2
        entries_per_slave = 2 * nUc * nDim

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

            # ---- Write to the sparse 1-D K view ----

            # K_UU: only the Phi_RP × Phi_RP block is nonzero (accumulated over slaves).
            # Layout: K[ri*nRot + rj] ↔ (RP_phi[ri], RP_phi[rj]).
            # np.einsum gives shape (nRot, nRot); .flatten() is row-major = ri*nRot+rj order.
            K[0:kuu_size] += np.einsum("i,ijk->jk", Lambda, H[:, -nRot:, -nRot:]).flatten()

            kul_offset = kuu_size + s * entries_per_slave
            klu_offset = kul_offset + nUc * nDim

            # K_UL: G.T has shape (nUc, nDim); .flatten() is row-major = iu*nDim+il order.
            # Layout: K[kul_offset + iu*nDim + il] ↔ (indcsU[iu], Lambda_s[il]).
            K[kul_offset : kul_offset + nUc * nDim] += G.T.flatten()

            # K_LU: G has shape (nDim, nUc); .flatten() is row-major = il*nUc+iu order.
            # Layout: K[klu_offset + il*nUc + iu] ↔ (Lambda_s[il], indcsU[iu]).
            K[klu_offset : klu_offset + nDim * nUc] += G.flatten()

            self._reactions[0 : self.nDim] += Lambda
            self._reactions[self.nDim :] += np.cross(T @ d0, Lambda)
