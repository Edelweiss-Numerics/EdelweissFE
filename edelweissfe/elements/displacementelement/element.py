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
#  Daniel Reitmair daniel.reitmair@uibk.ac.at
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
import numpy.linalg as lin

from edelweissfe.elements.base.displacementelementbase import DisplacementElementBase
from edelweissfe.elements.displacementelement._elementcomputationmatrices import (
    computeBOperator,
    computeJacobian,
)


class DisplacementElement(DisplacementElementBase):
    """This element can be used for EdelweissFE.
    The element currently only allows calculations with node forces and given displacements.

    Parameters
    ----------
    elementType
        A string identifying the requested element formulation as shown below.
    elNumber
        A unique integer label used for all kinds of purposes.

    Notes
    -----
    The following types of elements and attributes are currently possible (elementType):

    **Elements**

    - CPE4: quadrilateral element with 4 nodes and plane strain.
    - CPE8: quadrilateral element with 8 nodes and plane strain.
    - CPS4: quadrilateral element with 4 nodes and plane stress.
    - CPS8: quadrilateral element with 8 nodes and plane stress.
    - C3D8: hexahedron element with 8 nodes.
    - C3D20: hexahedron element with 20 nodes.

    **Optional Parameters**

    The following attributes are also included in the elType definition:

    - R: reduced integration for element, at the end of elType.
    - E: extended integration for element, at the end of elType.
    - N: (optional) regular integration for element, at the end of elType.

    If R or E is not given by the user, regular integration is assumed."""

    def setMaterial(self, material: type):
        """Assign a material and allocate the small-strain material tangent dStress/dStrain.

        Parameters
        ----------
        material
            An initalized instance of a material.
        """

        super().setMaterial(material)
        self._dStressdStrain = np.zeros([self._nInt, self._matrixSize, self._matrixSize])

    def initializeElement(
        self,
    ):
        """Initalize the element to be ready for computing."""

        # initialize the matrices
        self.J = computeJacobian(
            self._xi, self._eta, self._zeta, self._nodesCoordinates, self._nInt, self.nNodes, self.nSpatialDimensions
        )
        self.detJ = np.array([lin.det(self.J[i]) for i in range(self._nInt)])
        self.B = computeBOperator(
            self._xi, self._eta, self._zeta, self._nodesCoordinates, self._nInt, self.nNodes, self.nSpatialDimensions
        )

    def computeKernels(
        self,
        K_: np.ndarray,
        P: np.ndarray,
        U: np.ndarray,
        dU: np.ndarray,
        time: float,
        dTime: float,
    ):
        """Evaluate the residual and stiffness matrix for given time, field, and field increment due to a displacement or load.

        Parameters
        ----------
        P
            The external load vector gets calculated.
        K
            The stiffness matrix gets calculated.
        U
            The current solution vector.
        dU
            The current solution vector increment.
        time
            The current time.
        dTime
            The time increment.
        """

        # assume it's plain strain if it's not given by user
        K = K_ if K_.ndim == 2 else np.reshape(K_, (self._nDof, self._nDof))

        # copy all elements
        self._stateVarsTemp = [self._stateVarsRef[i].copy() for i in range(self._nInt)].copy()
        # strain increment
        self._dStrain[:, self._activeVoigtIndices] = np.array([self.B[i] @ dU for i in range(self._nInt)])
        for i in range(self._nInt):
            # get stress and strain
            stress = self._stateVarsTemp[i][0:6]
            self.material.assignCurrentStateVars(self._stateVarsTemp[i][12:])
            if not self._isHyperelastic:
                # use 3D for 2D planeStrain
                if not self.planeStrain and self.nSpatialDimensions == 2:
                    self.material.computePlaneStress(stress, self._dStressdStrain[i], self._dStrain[i], time, dTime)
                else:
                    self.material.computeStress(stress, self._dStressdStrain[i], self._dStrain[i], time, dTime)
            elif self._isHyperelastic:
                raise Exception("Please use the nonlinear element (displacementtlelement) for hyperelastic materials.")
            # C material tangent
            C = self._dStressdStrain[i][self._matrixVoigtIndices][:, self._matrixVoigtIndices]
            # B operator
            B = self.B[i]
            # Jacobi determinant
            detJ = lin.det(self.J[i])
            # get stiffness matrix for element j in point i
            K += B.T @ C @ B * detJ * self._t * self._weight[i]
            # calculate P
            P += B.T @ stress[self._activeVoigtIndices] * detJ * self._weight[i] * self._t
            # update strain in stateVars
            self._stateVarsTemp[i][6:12] += self._dStrain[i]

    def computeKernelsExplicit(
        self,
        P: np.ndarray,
        U: np.ndarray,
        dU: np.ndarray,
        time: float,
        dTime: float,
    ):
        """Evaluate the residual for given time, field, and field increment due to a displacement or load.

        Parameters
        ----------
        P
            The internal load vector gets calculated.
        U
            The current solution vector.
        dU
            The current solution vector increment.
        time
            The current time.
        dTime
            The time increment.
        """
        # copy all elements
        self._stateVarsTemp = [self._stateVarsRef[i].copy() for i in range(self._nInt)].copy()
        # strain increment
        self._dStrain[:, self._activeVoigtIndices] = np.array([self.B[i] @ dU for i in range(self._nInt)])
        for i in range(self._nInt):
            # get stress and strain
            stress = self._stateVarsTemp[i][0:6]
            self.material.assignCurrentStateVars(self._stateVarsTemp[i][12:])
            # use 3D for 2D planeStrain
            if not self.planeStrain and self.nSpatialDimensions == 2:
                self.material.computePlaneStress(stress, self._dStressdStrain[i], self._dStrain[i], time, dTime)
            else:
                self.material.computeStress(stress, self._dStressdStrain[i], self._dStrain[i], time, dTime)
            # B operator
            B = self.B[i]
            # Jacobi determinant
            detJ = lin.det(self.J[i])
            # calculate P
            P += B.T @ stress[self._activeVoigtIndices] * detJ * self._weight[i] * self._t
            # update strain in stateVars
            self._stateVarsTemp[i][6:12] += self._dStrain[i]

    def computeCriticalTimeStepForExplicitDynamics(self, Q: np.ndarray):

        dt = np.inf

        rho = self.material.getDensity()

        dEps = np.zeros(6)
        dEps += 1e-6  # small strain increment to compute the tangent stiffness
        _stateVarsTemp = [self._stateVarsRef[i].copy() for i in range(self._nInt)].copy()

        for i in range(self._nInt):
            # get characteristic element length
            l_ = self.getCharacteristicElementLength(i)

            tangent = np.zeros((6, 6))
            self.material.assignCurrentStateVars(_stateVarsTemp[i][12:])
            self.material.computeStress(_stateVarsTemp[i][0:6], tangent, dEps, np.array([0, 0]), 1)

            # get the maximum diagonal element
            maxCii = max(np.diag(tangent))
            # compute wave speed
            c = np.sqrt(maxCii / rho)

            dt = min(dt, l_ / c)

        return dt

    def computeInternalEnergy(self) -> float:
        """Evaluate the internal energy of the element.

        Returns
        -------
        float
            The internal energy.
        """
        energy = 0
        for i in range(self._nInt):
            stress = self._stateVarsTemp[i][0:6]
            strain = self._stateVarsTemp[i][6:12]
            energy += (
                0.5
                * np.dot(stress[self._activeVoigtIndices], strain[self._activeVoigtIndices])
                * lin.det(self.J[i])
                * self._t
                * self._weight[i]
            )

        return energy
