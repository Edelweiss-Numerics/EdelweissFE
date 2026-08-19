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
A point-wise interface to the gradient plasticity materials of hypoelastic type of
`Marmot <https://github.com/MAteRialMOdelingToolbox/Marmot/>`_, i.e. materials whose yield
condition depends on the Laplacian of the plastic multiplier.

Registered Marmot materials of this kind are ``GRADIENTVONMISES`` and
``GRADIENTLINEARELASTIC``.

The material honors
:class:`~edelweissfe.materials.base.basegradientplasticityhypoelasticmaterial.BaseGradientPlasticityHypoElasticMaterial`.
Marmot's templated base class is reached through the C++ shim
``_gradientplasticityshim.h``, since Cython cannot express non-type template parameters.
"""

import numpy as np

cimport numpy as np
from libcpp.string cimport string

from edelweissfe.materials.marmot._gradientplasticity cimport (
    GradientPlasticityHypoElasticShim1,
)
from edelweissfe.materials.marmot._marmotmaterials cimport StateView, stateViewAsArray

from edelweissfe.materials.base.basegradientplasticityhypoelasticmaterial import (
    BaseGradientPlasticityHypoElasticMaterial,
)
from edelweissfe.utils.exceptions import CutbackRequest

#: The factor by which the time increment is reduced if a Marmot material fails.
cutbackFactorOnMaterialFailure = 0.25


cdef class MarmotGradientPlasticityHypoElasticMaterial:
    """A gradient plasticity material of hypoelastic type provided by Marmot, evaluated at a
    single material point. This implementation covers materials with a single yield surface.

    Parameters
    ----------
    materialName
        The name Marmot registered the material under, e.g., ``GradientVonMises``.
    materialProperties
        The numpy array containing the material properties.
    materialNumber
        A label passed on to Marmot, only relevant for material specific messages.
    """

    cdef GradientPlasticityHypoElasticShim1* _material

    cdef double[::1] _materialProperties
    cdef double[::1] _stateVars

    cdef readonly str materialName

    def __cinit__(self, materialName: str, materialProperties: np.ndarray, int materialNumber = 1):

        self._material = NULL

        self._materialProperties = np.ascontiguousarray(materialProperties, dtype=float)

        cdef string materialName_ = materialName.encode("UTF-8")

        self._material = new GradientPlasticityHypoElasticShim1(
            materialName_,
            &self._materialProperties[0],
            self._materialProperties.shape[0],
            materialNumber,
        )

        self.materialName = materialName

    def __dealloc__(self):
        del self._material

    @property
    def nYieldSurfaces(self) -> int:
        """The number of yield surfaces this material has."""

        return GradientPlasticityHypoElasticShim1.getNumberOfYieldSurfaces()

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
        response,
        tangents,
        increment,
        double time,
        double dTime,
    ):
        """Compute the material response and the algorithmic tangents for a 3D material /
        2D material with plane strain.

        The stress held by ``response`` is the stress at the beginning of the increment on
        input and the updated stress on output, mirroring Marmot's response struct.

        Parameters
        ----------
        response
            The container to be filled with the stress and the yield function values.
        tangents
            The container to be filled with the six tangent blocks.
        increment
            The container describing the strain increment, the increment of the plastic
            multipliers and its Laplacian.
        time
            The total time at the end of the increment.
        dTime
            The time increment.
        """

        self._computeStress(response, tangents, increment, time, dTime, False)

    def computePlaneStress(
        self,
        response,
        tangents,
        increment,
        double time,
        double dTime,
    ):
        """Compute the material response and the algorithmic tangents for a 2D material
        with plane stress.

        Marmot's own plane stress algorithm is used, i.e. the out-of-plane strain is
        condensed out by the material itself.

        Parameters
        ----------
        response
            The container to be filled with the stress and the yield function values.
        tangents
            The container to be filled with the six tangent blocks.
        increment
            The container describing the strain increment, the increment of the plastic
            multipliers and its Laplacian.
        time
            The total time at the end of the increment.
        dTime
            The time increment.
        """

        self._computeStress(response, tangents, increment, time, dTime, True)

    def _computeStress(
        self,
        response,
        tangents,
        increment,
        double time,
        double dTime,
        bint planeStress,
    ):
        """Marshal the containers and call Marmot.

        Parameters
        ----------
        response
            The container to be filled with the stress and the yield function values.
        tangents
            The container to be filled with the six tangent blocks.
        increment
            The container describing the increment.
        time
            The total time at the end of the increment.
        dTime
            The time increment.
        planeStress
            Whether Marmot's plane stress algorithm should be used.
        """

        cdef double[::1] stress = response.stress
        cdef double[::1] f = response.f

        # Typed 2D views enforce C contiguity, so the shim writes the row-major tangents
        # straight into the caller's arrays instead of into a silent temporary copy.
        cdef double[:, ::1] dStress_dStrain = tangents.dStress_dStrain
        cdef double[:, ::1] dStress_dLambda = tangents.dStress_dLambda
        cdef double[:, ::1] dStress_dLaplacian = tangents.dStress_dLaplacian
        cdef double[:, ::1] dF_dStrain = tangents.dF_dStrain
        cdef double[:, ::1] dF_dLambda = tangents.dF_dLambda
        cdef double[:, ::1] dF_dLaplacian = tangents.dF_dLaplacian

        cdef const double[::1] dStrain = increment.dStrain
        cdef const double[::1] dLambda = increment.dLambda
        cdef const double[::1] laplaceDLambda = increment.laplaceDLambda

        cdef double elasticEnergyDensity = 0.0
        cdef double dissipation = 0.0

        try:
            self._material.computeStress(
                &stress[0],
                &f[0],
                &elasticEnergyDensity,
                &dissipation,
                &dStress_dStrain[0, 0],
                &dStress_dLambda[0, 0],
                &dStress_dLaplacian[0, 0],
                &dF_dStrain[0, 0],
                &dF_dLambda[0, 0],
                &dF_dLaplacian[0, 0],
                &dStrain[0],
                &dLambda[0],
                &laplaceDLambda[0],
                &self._stateVars[0],
                time,
                dTime,
                planeStress,
            )
        except (ValueError, RuntimeError) as e:
            raise CutbackRequest(str(e), cutbackFactorOnMaterialFailure)

        response.elasticEnergyDensity = elasticEnergyDensity
        response.dissipation = dissipation

    def getResult(self, result: str) -> np.ndarray:
        """Get a result of the material as a persistent view into the state variables.

        Parameters
        ----------
        result
            The name Marmot registered the result under, e.g. ``kappa``.

        Returns
        -------
        np.ndarray
            The persistent view onto the result.
        """

        cdef string result_ = result.encode("UTF-8")
        cdef StateView res = self._material.getStateView(result_, &self._stateVars[0])

        return stateViewAsArray(res)


BaseGradientPlasticityHypoElasticMaterial.register(MarmotGradientPlasticityHypoElasticMaterial)
