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
#  Alexander Dummer alexander.dummer@uibk.ac.at
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
"""
A point-wise interface to hypoelastic materials of
`Marmot <https://github.com/MAteRialMOdelingToolbox/Marmot/>`_.

In contrast to :mod:`edelweissfe.elements.marmotelement`, no element is involved: a
:class:`MarmotHypoElasticMaterial` is a material point, and it honors the very same
interface as the native Python materials, i.e.,
:class:`~edelweissfe.materials.base.basehypoelasticmaterial.BaseHypoElasticMaterial`.
Marmot materials and EdelweissFE materials are therefore interchangeable in any
consumer, be it a finite element or a finite difference stencil.

Following the convention of the native materials, the caller owns the stress
(and, if desired, the strain) storage and passes the stress in and out of
:meth:`~MarmotHypoElasticMaterial.computeStress`, while
:meth:`~MarmotHypoElasticMaterial.getNumberOfRequiredStateVars` reports only the state
variables Marmot itself requires.
"""

import numpy as np

cimport numpy as np
from libcpp.string cimport string

from edelweissfe.materials.marmot._marmotmaterials cimport (
    MarmotMaterialHypoElastic,
    MarmotMaterialHypoElasticFactory,
    Matrix3d,
    Matrix6d,
    StateView,
    Vector3d,
    Vector6d,
    stateViewAsArray,
)

from edelweissfe.materials.base.basehypoelasticmaterial import BaseHypoElasticMaterial
from edelweissfe.utils.exceptions import CutbackRequest

#: The factor by which the time increment is reduced if a Marmot material fails.
cutbackFactorOnMaterialFailure = 0.25

#: The Voigt indices of the 2D (plane) subspace within the 3D Voigt vector.
planeVoigtIndices = (0, 1, 3)


cdef class MarmotHypoElasticMaterial:
    """A hypoelastic material provided by Marmot, evaluated at a single material point.

    Parameters
    ----------
    materialName
        The name Marmot registered the material under, e.g., ``LinearElastic``.
    materialProperties
        The numpy array containing the material properties.
    materialNumber
        A label passed on to Marmot, only relevant for material specific messages.
    """

    cdef MarmotMaterialHypoElastic* _material

    # Marmot keeps the *pointer* to the material properties without taking ownership,
    # so this contiguous copy must outlive the material itself.
    cdef double[::1] _materialProperties

    cdef double[::1] _stateVars

    cdef readonly str materialName

    def __cinit__(self, materialName: str, materialProperties: np.ndarray, int materialNumber = 1):

        self._material = NULL

        self._materialProperties = np.ascontiguousarray(materialProperties, dtype=float)

        cdef string materialName_ = materialName.encode("UTF-8")

        self._material = MarmotMaterialHypoElasticFactory.createMaterial(
            materialName_,
            &self._materialProperties[0],
            self._materialProperties.shape[0],
            materialNumber,
        )

        if self._material == NULL:
            raise ValueError(
                "Marmot does not provide a hypoelastic material '{:}'".format(materialName)
            )

        self.materialName = materialName

    def __dealloc__(self):
        del self._material

    @property
    def materialProperties(self) -> np.ndarray:
        """The properties the material has."""

        return np.asarray(self._materialProperties)

    def getNumberOfRequiredStateVars(self) -> int:
        """The number of state variables Marmot requires for this material.

        Returns
        -------
        int
            The number of required state variables.
        """

        return self._material.getNumberOfRequiredStateVars()

    def assignCurrentStateVars(self, currentStateVars: np.ndarray):
        """Assign the state variable storage this material should operate on.

        Parameters
        ----------
        currentStateVars
            The contiguous array holding the material state variables.
        """

        self._stateVars = currentStateVars

    def initializeYourself(self):
        """Let Marmot initialize the assigned state variables."""

        self._material.initializeYourself(&self._stateVars[0], self._stateVars.shape[0])

    def setCharacteristicElementLength(self, double length):
        """Communicate the characteristic length of the discretization to the material.

        Parameters
        ----------
        length
            The characteristic length.
        """

        self._material.setCharacteristicElementLength(length)

    def getDensity(self) -> float:
        """Determine the density of the material.

        Returns
        -------
        float
            The density of the material.
        """

        return self._material.getDensity(&self._stateVars[0])

    def computeStress(
        self,
        double[::1] stress,
        dStress_dStrain: np.ndarray,
        double[::1] dStrain,
        double time,
        double dTime,
    ):
        """Compute the stress for a 3D material / 2D material with plane strain.

        Parameters
        ----------
        stress
            The 6 component Voigt stress at the beginning of the increment, updated in place.
        dStress_dStrain
            The 6x6 matrix receiving dStress/dStrain.
        dStrain
            The 6 component Voigt strain increment.
        time
            The total time at the end of the increment.
        dTime
            The time increment.
        """

        cdef double[36] tangentBuffer
        cdef int i

        for i in range(36):
            tangentBuffer[i] = 0.0

        cdef Matrix6d dStress_dStrain_ = Matrix6d(&tangentBuffer[0])
        cdef Vector6d dStrain_ = Vector6d(&dStrain[0])

        cdef MarmotMaterialHypoElastic.state3D state
        state.stress = Vector6d(&stress[0])
        state.elasticEnergyDensity = 0.0
        state.dissipation = 0.0
        state.stateVars = &self._stateVars[0]

        cdef MarmotMaterialHypoElastic.timeInfo timeInfo
        timeInfo.time = time
        timeInfo.dT = dTime

        try:
            self._material.computeStress(state, dStress_dStrain_, dStrain_, timeInfo)
        except (ValueError, RuntimeError) as e:
            raise CutbackRequest(str(e), cutbackFactorOnMaterialFailure)

        cdef double[::1] stressView = <double[:6]> (&state.stress(0))
        cdef double[::1] tangentView = <double[:36]> (&dStress_dStrain_(0, 0))

        stress[:] = stressView

        # Eigen stores fixed size matrices column wise by default.
        dStress_dStrain[:, :] = np.reshape(np.asarray(tangentView), (6, 6), order="F")

    def computePlaneStress(
        self,
        double[::1] stress,
        dStress_dStrain: np.ndarray,
        double[::1] dStrain,
        double time,
        double dTime,
    ):
        """Compute the stress for a 2D material with plane stress.

        In line with the native materials, the stress is exchanged as a 6 component Voigt
        vector of which the plane components (11, 22, 12) are used, whereas the tangent is
        the reduced 3x3 matrix. Marmot's own plane stress algorithm is used, i.e., the
        out-of-plane strain is condensed out by the material itself.

        Parameters
        ----------
        stress
            The 6 component Voigt stress at the beginning of the increment, updated in place.
        dStress_dStrain
            The 3x3 matrix receiving dStress/dStrain of the plane components.
        dStrain
            The 6 component Voigt strain increment.
        time
            The total time at the end of the increment.
        dTime
            The time increment.
        """

        cdef double[3] stressBuffer
        cdef double[3] dStrainBuffer
        cdef double[9] tangentBuffer
        cdef int[3] planeIdx
        cdef int i

        planeIdx[0] = planeVoigtIndices[0]
        planeIdx[1] = planeVoigtIndices[1]
        planeIdx[2] = planeVoigtIndices[2]

        for i in range(3):
            stressBuffer[i] = stress[planeIdx[i]]
            dStrainBuffer[i] = dStrain[planeIdx[i]]

        for i in range(9):
            tangentBuffer[i] = 0.0

        cdef Matrix3d dStress_dStrain_ = Matrix3d(&tangentBuffer[0])
        cdef Vector3d dStrain_ = Vector3d(&dStrainBuffer[0])

        cdef MarmotMaterialHypoElastic.state2D state
        state.stress = Vector3d(&stressBuffer[0])
        state.elasticEnergyDensity = 0.0
        state.dissipation = 0.0
        state.stateVars = &self._stateVars[0]

        cdef MarmotMaterialHypoElastic.timeInfo timeInfo
        timeInfo.time = time
        timeInfo.dT = dTime

        try:
            self._material.computePlaneStress(state, dStress_dStrain_, dStrain_, timeInfo)
        except (ValueError, RuntimeError) as e:
            raise CutbackRequest(str(e), cutbackFactorOnMaterialFailure)

        for i in range(3):
            stress[planeIdx[i]] = state.stress(i)

        cdef double[::1] tangentView = <double[:9]> (&dStress_dStrain_(0, 0))

        dStress_dStrain[:, :] = np.reshape(np.asarray(tangentView), (3, 3), order="F")

    def computeUniaxialStress(
        self,
        double[::1] stress,
        dStress_dStrain: np.ndarray,
        double[::1] dStrain,
        double time,
        double dTime,
    ):
        """Compute the stress for a uniaxial stress state.

        The stress is exchanged as a 6 component Voigt vector of which only the axial
        component is used, whereas the tangent is the reduced 1x1 matrix. Marmot condenses
        out the two lateral strains itself.

        Parameters
        ----------
        stress
            The 6 component Voigt stress at the beginning of the increment, updated in place.
        dStress_dStrain
            The 1x1 matrix receiving dStress/dStrain of the axial component.
        dStrain
            The 6 component Voigt strain increment.
        time
            The total time at the end of the increment.
        dTime
            The time increment.
        """

        cdef double dStress_dStrain_ = 0.0

        cdef MarmotMaterialHypoElastic.state1D state
        state.stress = stress[0]
        state.elasticEnergyDensity = 0.0
        state.dissipation = 0.0
        state.stateVars = &self._stateVars[0]

        cdef MarmotMaterialHypoElastic.timeInfo timeInfo
        timeInfo.time = time
        timeInfo.dT = dTime

        try:
            self._material.computeUniaxialStress(state, dStress_dStrain_, dStrain[0], timeInfo)
        except (ValueError, RuntimeError) as e:
            raise CutbackRequest(str(e), cutbackFactorOnMaterialFailure)

        stress[0] = state.stress

        dStress_dStrain[0, 0] = dStress_dStrain_

    def getResult(self, result: str) -> np.ndarray:
        """Get a result of the material as a persistent view into the state variables.

        Parameters
        ----------
        result
            The name Marmot registered the result under.

        Returns
        -------
        np.ndarray
            The persistent view onto the result.
        """

        cdef string result_ = result.encode("UTF-8")
        cdef StateView res = self._material.getStateView(result_, &self._stateVars[0])

        return stateViewAsArray(res)


BaseHypoElasticMaterial.register(MarmotHypoElasticMaterial)
