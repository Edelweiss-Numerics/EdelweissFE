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
    computeJacobian,
)
from edelweissfe.elements.displacementtlelement._elementcomputationmatrices import (
    computeBOperator,
    computeDeformationGradient,
    computeNablaN,
    makeH2D,
)
from edelweissfe.utils.voigtnotation import doVoigtStrain, undoVoigtStress


def Hgeo(nablaN, S, dim):
    """Computes the unintegrated geometric stiffness matrix for one quadrature point.

    Parameters
    ----------
    nablaN
        The derivative of the shape functions w.r.t the actual coordinates.
    S
        The second Piola-Kirchhoff stress tensor in Voigt notation.
    dim
        Dimension of the domain.

    Returns
    -------
    np.ndarray
        The unintegrated geometric stiffness matrix."""

    S = undoVoigtStress(dim, S)
    Ie = np.eye(dim)
    Hsub = nablaN.T @ S @ nablaN  # [nNodes x nNodes]
    nDof = dim * len(nablaN[0])
    H = np.zeros([nDof, nDof])  # [nDof x nDof]
    for i in range(len(Hsub)):
        for j in range(len(Hsub[0])):
            H[dim * i : dim * (i + 1), dim * j : dim * (j + 1)] = Ie * Hsub[i, j]
    return H


def makeDeformationGradient3D(F, dim):
    """Make the deformation gradient 3D [3 x 3].

    Parameters
    ----------
    F
        The deformation gradient.
    dim
        Dimension of the domain.

    Returns
    -------
    np.ndarray
        The deformation gradient in [3 x 3]."""

    return np.array([[F[0, 0], F[0, 1], 0], [F[1, 0], F[1, 1], 0], [0, 0, 1]]) if dim == 2 else F


class DisplacementTLElement(DisplacementElementBase):
    """This total lagrangian element can be used for EdelweissFE.
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
    - N
            (optional) regular integration for element, at the end of elType.

    If R or E is not given by the user, we assume regular increment."""

    def __init__(self, elementType: str, elNumber: int):
        super().__init__(elementType, elNumber)
        self._strain = np.zeros([self._nInt, 6])
        # Green-Lagrange strain at the converged (old) and the current state, and the deformation
        # gradient, per quadrature point
        self._Eold = np.zeros([self._nInt, self.nSpatialDimensions, self.nSpatialDimensions])
        self._E = np.zeros([self._nInt, self.nSpatialDimensions, self.nSpatialDimensions])
        self._F = np.stack(self._nInt * [np.eye(self.nSpatialDimensions)])

    def initializeElement(
        self,
    ):
        """Initalize the element to be ready for computing."""

        self.J = computeJacobian(
            self._xi, self._eta, self._zeta, self._nodesCoordinates, self._nInt, self._nNodes, self.nSpatialDimensions
        )
        self.nablaN = computeNablaN(
            self._xi, self._eta, self._zeta, self.J, self._nInt, self._nNodes, self.nSpatialDimensions
        )

    def setMaterial(self, material: type):
        """Assign a material and allocate its material tangent: dStress/dDeformationGradient for a
        hyperelastic material, dStress/dStrain otherwise.

        Parameters
        ----------
        material
            An initalized instance of a material.
        """

        super().setMaterial(material)
        self._materialProperties = self.material.materialProperties
        if self._isHyperelastic:
            self._dStress_dDeformationGradient = np.zeros([self._nInt, 3, 3, 3, 3])
        else:
            self._dStress_dStrain = np.zeros([self._nInt, self._matrixSize, self._matrixSize])

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

        dim = self.nSpatialDimensions
        # assume it's plain strain if it's not given by user
        K = K_ if K_.ndim == 2 else np.reshape(K_, (self._nDof, self._nDof))

        # get current state Vars
        self._stateVarsTemp = [self._stateVarsRef[i].copy() for i in range(self._nInt)].copy()
        # compute the deformation gradient
        self._F = computeDeformationGradient(U, self.nablaN, self._nInt, self._nNodes, dim)
        if not self._isHyperelastic:
            B = computeBOperator(self._F, self.nablaN, self._nInt, self._nNodes, dim)
        for i in range(self._nInt):
            detJ = lin.det(self.J[i])
            # get stress (PK2) and strain
            stress = self._stateVarsTemp[i][0:6]
            self.material.assignCurrentStateVars(self._stateVarsTemp[i][12:])
            H = self._F[i] - np.eye(dim)
            self._E[i] = 1 / 2 * (H + H.T + H.T @ H)
            invF = lin.inv(self._F[i])
            F = makeDeformationGradient3D(self._F[i], dim)
            if self._isHyperelastic:
                NAi = np.zeros([self._nNodes, 3])
                _nablaN = np.zeros([3, self._nNodes])
                NAi[:, :dim] = self.nablaN[i].T @ invF
                _nablaN[:dim] = self.nablaN[i]
                self._strain[i, self._activeVoigtIndices] = doVoigtStrain(dim, self._E[i])
                # use 3D for 2D planeStrain
                if not self.planeStrain and dim == 2:
                    self.material.computePlaneKirchhoff(stress, self._dStress_dDeformationGradient[i], F, time, dTime)
                    T = undoVoigtStress(2, stress)
                else:
                    invF = makeDeformationGradient3D(invF, dim)
                    self.material.computeKirchhoff(stress, self._dStress_dDeformationGradient[i], F, time, dTime)
                    T = undoVoigtStress(3, stress)
                PK1 = invF @ T
                # update strain in stateVars
                self._stateVarsTemp[i][6:12] = self._strain[i]
                Hk = makeH2D(
                    np.einsum("ai,ijkl,bl->ajbk", NAi, self._dStress_dDeformationGradient[i], _nablaN.T)
                    - np.einsum("ak,bi,ij->ajbk", NAi, NAi, T),
                    dim,
                )
                # compute inner forces
                P += (self.nablaN[i].T @ PK1[:dim, :dim]).flatten() * detJ * self._t * self._weight[i]
            else:  # for non-hyperelastic materials
                self._dStrain[i, self._activeVoigtIndices] = doVoigtStrain(dim, self._E[i] - self._Eold[i])
                if not self.planeStrain and dim == 2:
                    self.material.computePlaneStress(stress, self._dStress_dStrain[i], self._dStrain[i], time, dTime)
                else:
                    self.material.computeStress(stress, self._dStress_dStrain[i], self._dStrain[i], time, dTime)
                self._stateVarsTemp[i][6:12] += self._dStrain[i]
                Cm = self._dStress_dStrain[i][self._matrixVoigtIndices][:, self._matrixVoigtIndices]
                Hk = B[i].T @ Cm @ B[i] + Hgeo(self.nablaN[i], stress, dim)
                # compute inner forces
                P += B[i].T @ stress[self._matrixVoigtIndices] * detJ * self._weight[i] * self._t
            # calculate complete stiffness matrix
            K += Hk * detJ * self._t * self._weight[i]

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

        dim = self.nSpatialDimensions
        # get current state Vars
        self._stateVarsTemp = [self._stateVarsRef[i].copy() for i in range(self._nInt)].copy()
        # compute the deformation gradient
        self._F = computeDeformationGradient(U, self.nablaN, self._nInt, self._nNodes, dim)
        if not self._isHyperelastic:
            B = computeBOperator(self._F, self.nablaN, self._nInt, self._nNodes, dim)
        for i in range(self._nInt):
            detJ = lin.det(self.J[i])
            # get stress (PK2) and strain
            stress = self._stateVarsTemp[i][0:6]
            self.material.assignCurrentStateVars(self._stateVarsTemp[i][12:])
            H = self._F[i] - np.eye(dim)
            self._E[i] = 1 / 2 * (H + H.T + H.T @ H)
            invF = lin.inv(self._F[i])
            F = makeDeformationGradient3D(self._F[i], dim)
            if self._isHyperelastic:
                NAi = np.zeros([self._nNodes, 3])
                _nablaN = np.zeros([3, self._nNodes])
                NAi[:, :dim] = self.nablaN[i].T @ invF
                _nablaN[:dim] = self.nablaN[i]
                self._strain[i, self._activeVoigtIndices] = doVoigtStrain(dim, self._E[i])
                # use 3D for 2D planeStrain
                if not self.planeStrain and dim == 2:
                    self.material.computePlaneKirchhoff(stress, self._dStress_dDeformationGradient[i], F, time, dTime)
                    T = undoVoigtStress(2, stress)
                else:
                    invF = makeDeformationGradient3D(invF, dim)
                    self.material.computeKirchhoff(stress, self._dStress_dDeformationGradient[i], F, time, dTime)
                    T = undoVoigtStress(3, stress)
                PK1 = invF @ T
                # update strain in stateVars
                self._stateVarsTemp[i][6:12] = self._strain[i]
                # compute inner forces
                P += (self.nablaN[i].T @ PK1[:dim, :dim]).flatten() * detJ * self._t * self._weight[i]
            else:  # for non-hyperelastic materials
                self._dStrain[i, self._activeVoigtIndices] = doVoigtStrain(dim, self._E[i] - self._Eold[i])
                if not self.planeStrain and dim == 2:
                    self.material.computePlaneStress(stress, self._dStress_dStrain[i], self._dStrain[i], time, dTime)
                else:
                    self.material.computeStress(stress, self._dStress_dStrain[i], self._dStrain[i], time, dTime)
                self._stateVarsTemp[i][6:12] += self._dStrain[i]
                # compute inner forces
                P += B[i].T @ stress[self._matrixVoigtIndices] * detJ * self._weight[i] * self._t

    def computeCriticalTimeStepForExplicitDynamics(self, Q: np.ndarray):
        raise NotImplementedError(
            "Critical time step computation for explicit dynamics is not implemented "
            "for this total-Lagrangian displacement element."
        )

    def computeInternalEnergy(self) -> float:
        """Compute the internal energy of the element.

        Returns
        -------
        energy
            The internal energy of the element.
        """

        energy = 0.0
        for i in range(self._nInt):
            detJ = lin.det(self.J[i])
            energy += self._stateVarsTemp[i][0:6] @ self._stateVarsTemp[i][6:12] * detJ * self._t * self._weight[i]

        return energy

    def acceptLastState(
        self,
    ):
        """Accept the computed state (in nonlinear iteration schemes), including the Green-Lagrange
        strain."""

        super().acceptLastState()
        self._Eold = self._E.copy()

    def getStateVars(self) -> np.ndarray:
        """Return a copy of the converged quadrature-point state-variable buffer, including the
        converged Green-Lagrange strain (``_Eold``) needed to resume the total-Lagrangian
        incremental-strain computation."""

        return np.concatenate([self._stateVarsRef.reshape(-1), self._Eold.reshape(-1)]).copy()

    def setStateVars(self, values: np.ndarray):
        """Overwrite the converged quadrature-point state-variable buffer and ``_Eold`` in place.

        The working buffers are seeded from them as well. :meth:`acceptLastState` copies
        ``_stateVarsTemp`` over ``_stateVarsRef`` *and* ``_E`` over ``_Eold``, and it can run
        before any element computation has populated either: both explicit solvers process a zero
        increment first, for which :meth:`computeYourself` is never called, and then accept it. On
        a resumed run that copied freshly allocated buffers of zeros over the state just restored,
        discarding both the quadrature-point history and the converged Green-Lagrange strain the
        total-Lagrangian incremental-strain computation resumes from. A cold start never noticed,
        because there both buffers are zero anyway.
        """

        values = np.asarray(values)
        nStateVars = self._stateVarsRef.size
        self._stateVarsRef[:] = values[:nStateVars].reshape(self._stateVarsRef.shape)
        self._Eold[:] = values[nStateVars:].reshape(self._Eold.shape)
        self._stateVarsTemp = self._stateVarsRef.copy()
        self._E = self._Eold.copy()
