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
Material drivers for the single quadrature point element.

A driver translates a point-wise material of :mod:`edelweissfe.materials.marmot` into the
residual and tangent of a zero dimensional 'element'. It owns the layout of the state
variables and knows which fields the material couples.

The residual and the tangent follow the convention of the solvers, i.e., the residual is
the internal flux in positive sense and the tangent is its derivative with respect to the
solution vector.
"""

from abc import ABC, abstractmethod

import numpy as np

from edelweissfe.materials.base.basegradientenhancedhypoelasticmaterial import (
    GradientEnhancedIncrement,
    GradientEnhancedResponse,
    GradientEnhancedTangents,
)
from edelweissfe.materials.base.basegradientplasticityhypoelasticmaterial import (
    GradientPlasticityIncrement,
    GradientPlasticityResponse,
    GradientPlasticityTangents,
)


class BaseMaterialDriver(ABC):
    """The interface a material driver of the single quadrature point element has to
    fulfill."""

    @property
    @abstractmethod
    def fields(self) -> list[str]:
        """The fields the driven material couples on the (dummy) node."""

    @property
    @abstractmethod
    def nDof(self) -> int:
        """The number of degrees of freedom resulting from the coupled fields."""

    @abstractmethod
    def createMaterial(self, materialName: str, materialProperties: np.ndarray):
        """Create the underlying Marmot material.

        Parameters
        ----------
        materialName
            The name Marmot registered the material under.
        materialProperties
            The material properties.
        """

    @abstractmethod
    def getNumberOfRequiredStateVars(self) -> int:
        """The total number of state variables, i.e., the driver's own bookkeeping plus
        those of the material."""

    @abstractmethod
    def assignStateVars(self, stateVars: np.ndarray):
        """Assign the state variable storage and create the persistent views into it.

        Parameters
        ----------
        stateVars
            The array holding all state variables.
        """

    @abstractmethod
    def computeKernels(
        self,
        Ke: np.ndarray,
        Pe: np.ndarray,
        U: np.ndarray,
        dU: np.ndarray,
        time: float,
        dTime: float,
    ):
        """Evaluate the material and assemble residual and tangent.

        Parameters
        ----------
        Ke
            The tangent to be defined, shape ``(nDof, nDof)``.
        Pe
            The residual to be defined, shape ``(nDof,)``.
        U
            The current solution vector.
        dU
            The current solution vector increment.
        time
            The total time at the end of the increment.
        dTime
            The time increment.
        """

    def initializeYourself(self):
        """Let the material initialize the assigned state variables."""

        self._material.initializeYourself()

    def setCharacteristicElementLength(self, length: float):
        """Communicate the characteristic length of the discretization to the material.

        Parameters
        ----------
        length
            The characteristic length.
        """

    @abstractmethod
    def getResultArray(self, result: str, getPersistentView: bool = True) -> np.ndarray:
        """Get a result of the driver or of the material.

        Parameters
        ----------
        result
            The name of the result.
        getPersistentView
            Whether a view or a copy should be returned.
        """


class MarmotMaterialHypoElasticDriver(BaseMaterialDriver):
    """Drives a hypoelastic Marmot material. The single node carries the strain, the
    residual is the stress and the tangent is dStress/dStrain."""

    #: Number of state variables the driver keeps in front of the material's own ones.
    nStateVarsOverhead = 6 + 6 + 36

    def __init__(self):
        self._fields = ["strain symmetric"]
        self._nDof = 6
        self._material = None

    @property
    def fields(self) -> list[str]:
        return self._fields

    @property
    def nDof(self) -> int:
        return self._nDof

    def createMaterial(self, materialName: str, materialProperties: np.ndarray):
        from edelweissfe.materials.marmot.marmothypoelastic import (
            MarmotHypoElasticMaterial,
        )

        self._material = MarmotHypoElasticMaterial(materialName, materialProperties)

    def getNumberOfRequiredStateVars(self) -> int:
        return self.nStateVarsOverhead + self._material.getNumberOfRequiredStateVars()

    def assignStateVars(self, stateVars: np.ndarray):
        self._stress = stateVars[0:6]
        self._strain = stateVars[6:12]
        self._dStress_dStrain = stateVars[12:48]

        self._material.assignCurrentStateVars(stateVars[48:])

    def computeKernels(
        self,
        Ke: np.ndarray,
        Pe: np.ndarray,
        U: np.ndarray,
        dU: np.ndarray,
        time: float,
        dTime: float,
    ):
        dStrain = np.ascontiguousarray(dU)
        tangent = np.zeros((6, 6))

        self._material.computeStress(self._stress, tangent, dStrain, time, dTime)

        self._strain += dStrain

        # stored column wise, so that a reshape with order='F' recovers the matrix
        self._dStress_dStrain[:] = tangent.flatten(order="F")

        Pe[:] = self._stress
        Ke[:, :] = tangent

    def setCharacteristicElementLength(self, length: float):
        self._material.setCharacteristicElementLength(length)

    def getResultArray(self, result: str, getPersistentView: bool = True) -> np.ndarray:
        if result == "stress":
            return np.array(self._stress, copy=not getPersistentView)
        if result == "strain":
            return np.array(self._strain, copy=not getPersistentView)
        if result == "dStress_dStrain":
            return np.array(self._dStress_dStrain, copy=not getPersistentView)

        return np.array(self._material.getResult(result), copy=not getPersistentView)


class MarmotMaterialGradientEnhancedHypoElasticDriver(BaseMaterialDriver):
    """Drives a general gradient-enhanced hypoelastic Marmot material with a single
    nonlocal variable.

    The single node carries the strain and the nonlocal variable. Since a material point
    has no spatial extent, the gradient term of the nonlocal balance equation drops out
    and the second residual reduces to the local part

    .. math::
        \\bar\\kappa - \\kappa(\\boldsymbol \\varepsilon,\\, \\bar\\kappa)
    """

    def __init__(self):
        self._fields = ["strain symmetric", "nonlocal damage"]
        self._nDof = 7
        self._material = None

        self._response = GradientEnhancedResponse.createZero(1)
        self._tangents = GradientEnhancedTangents.createZero(1)
        self._increment = GradientEnhancedIncrement.createZero(1)

    @property
    def fields(self) -> list[str]:
        return self._fields

    @property
    def nDof(self) -> int:
        return self._nDof

    @property
    def nStateVarsOverhead(self) -> int:
        """Number of state variables the driver keeps in front of the material's own ones."""

        return 2 * self._nDof + self._nDof**2

    def createMaterial(self, materialName: str, materialProperties: np.ndarray):
        from edelweissfe.materials.marmot.marmotgradientenhancedhypoelastic import (
            MarmotGradientEnhancedHypoElasticMaterial,
        )

        self._material = MarmotGradientEnhancedHypoElasticMaterial(materialName, materialProperties)

        if self._material.nNonlocalVariables != 1:
            raise ValueError("This driver supports materials with a single nonlocal variable only.")

    def getNumberOfRequiredStateVars(self) -> int:
        return self.nStateVarsOverhead + self._material.getNumberOfRequiredStateVars()

    def assignStateVars(self, stateVars: np.ndarray):
        n = self._nDof

        # the stress like quantities are the stress and the local driving variable,
        # the strain like quantities are the strain and the nonlocal variable
        self._stressLike = stateVars[0:n]
        self._strainLike = stateVars[n : 2 * n]
        self._algorithmicTangent = stateVars[2 * n : 2 * n + n**2]

        self._material.assignCurrentStateVars(stateVars[2 * n + n**2 :])

    def computeKernels(
        self,
        Ke: np.ndarray,
        Pe: np.ndarray,
        U: np.ndarray,
        dU: np.ndarray,
        time: float,
        dTime: float,
    ):
        response = self._response
        tangents = self._tangents
        increment = self._increment

        increment.dStrain[:] = dU[0:6]
        increment.K[:] = U[6:7]
        increment.dK[:] = dU[6:7]

        # the stress enters the rate form material as the stress of the last increment
        response.stress[:] = self._stressLike[0:6]
        response.KLocal[:] = self._stressLike[6:7]
        tangents.zero()

        self._material.computeStress(response, tangents, increment, time, dTime)

        self._strainLike[0:6] += increment.dStrain
        self._strainLike[6:7] = increment.K

        self._stressLike[0:6] = response.stress
        self._stressLike[6:7] = response.KLocal

        Pe[0:6] = response.stress
        Pe[6] = increment.K[0] - response.KLocal[0]

        Ke[0:6, 0:6] = tangents.dStress_dStrain
        Ke[0:6, 6] = tangents.dStress_dK[:, 0]
        Ke[6, 0:6] = -tangents.dKLocal_dStrain[0, :]
        Ke[6, 6] = 1.0 - tangents.dKLocal_dK[0, 0]

        # stored column wise, so that a reshape with order='F' recovers the matrix
        self._algorithmicTangent[:] = np.asarray(Ke).flatten(order="F")

    def getResultArray(self, result: str, getPersistentView: bool = True) -> np.ndarray:
        if result == "stress":
            return np.array(self._stressLike[0:6], copy=not getPersistentView)
        if result == "strain":
            return np.array(self._strainLike[0:6], copy=not getPersistentView)
        if result == "nonlocal damage":
            return np.array(self._strainLike[6:7], copy=not getPersistentView)
        if result == "local damage driving variable":
            return np.array(self._stressLike[6:7], copy=not getPersistentView)
        if result == "algorithmicTangent":
            return np.array(self._algorithmicTangent, copy=not getPersistentView)

        return np.array(self._material.getResult(result), copy=not getPersistentView)


class MarmotMaterialGradientPlasticityHypoElasticDriver(BaseMaterialDriver):
    """Drives a gradient plasticity Marmot material of hypoelastic type with a single yield
    surface.

    The single node carries the strain and the plastic multiplier. Since a material point has
    no spatial extent, there are no neighbours to form a Laplacian from, so the driver always
    passes a vanishing Laplacian of the plastic multiplier increment to the material: the yield
    condition reduces to the local one degree of freedom the point owns, exactly as the local
    part of :class:`MarmotMaterialGradientEnhancedHypoElasticDriver`'s second residual does.
    """

    def __init__(self):
        self._fields = ["strain symmetric", "plastic multiplier"]
        self._nDof = 7
        self._material = None

        self._response = GradientPlasticityResponse.createZero(1)
        self._tangents = GradientPlasticityTangents.createZero(1)
        self._increment = GradientPlasticityIncrement.createZero(1)

    @property
    def fields(self) -> list[str]:
        return self._fields

    @property
    def nDof(self) -> int:
        return self._nDof

    @property
    def nStateVarsOverhead(self) -> int:
        """Number of state variables the driver keeps in front of the material's own ones."""

        nYieldSurfaces = self._nDof - 6

        return 6 + 6 + nYieldSurfaces + self._nDof**2

    def createMaterial(self, materialName: str, materialProperties: np.ndarray):
        from edelweissfe.materials.marmot.marmotgradientplasticityhypoelastic import (
            MarmotGradientPlasticityHypoElasticMaterial,
        )

        self._material = MarmotGradientPlasticityHypoElasticMaterial(materialName, materialProperties)

        if self._material.nYieldSurfaces != 1:
            raise ValueError("This driver supports materials with a single yield surface only.")

    def getNumberOfRequiredStateVars(self) -> int:
        return self.nStateVarsOverhead + self._material.getNumberOfRequiredStateVars()

    def assignStateVars(self, stateVars: np.ndarray):
        self._stress = stateVars[0:6]
        self._strain = stateVars[6:12]
        self._lambda = stateVars[12:13]
        self._algorithmicTangent = stateVars[13 : 13 + self._nDof**2]

        self._material.assignCurrentStateVars(stateVars[13 + self._nDof**2 :])

    def computeKernels(
        self,
        Ke: np.ndarray,
        Pe: np.ndarray,
        U: np.ndarray,
        dU: np.ndarray,
        time: float,
        dTime: float,
    ):
        response = self._response
        tangents = self._tangents
        increment = self._increment

        increment.dStrain[:] = dU[0:6]
        increment.dLambda[:] = dU[6:7]
        increment.laplaceDLambda[:] = 0.0

        # the stress enters the rate form material as the stress of the last increment
        response.stress[:] = self._stress
        tangents.zero()

        self._material.computeStress(response, tangents, increment, time, dTime)

        self._strain += increment.dStrain
        self._lambda += increment.dLambda

        self._stress[:] = response.stress

        # the yield function value is already the residual of the additional balance
        # equation, cf. GradientPlasticityResponse
        Pe[0:6] = response.stress
        Pe[6] = response.f[0]

        Ke[0:6, 0:6] = tangents.dStress_dStrain
        Ke[0:6, 6] = tangents.dStress_dLambda[:, 0]
        Ke[6, 0:6] = tangents.dF_dStrain[0, :]
        Ke[6, 6] = tangents.dF_dLambda[0, 0]

        # stored column wise, so that a reshape with order='F' recovers the matrix
        self._algorithmicTangent[:] = np.asarray(Ke).flatten(order="F")

    def getResultArray(self, result: str, getPersistentView: bool = True) -> np.ndarray:
        if result == "stress":
            return np.array(self._stress, copy=not getPersistentView)
        if result == "strain":
            return np.array(self._strain, copy=not getPersistentView)
        if result == "plastic multiplier":
            return np.array(self._lambda, copy=not getPersistentView)
        if result == "algorithmicTangent":
            return np.array(self._algorithmicTangent, copy=not getPersistentView)

        return np.array(self._material.getResult(result), copy=not getPersistentView)


#: The available material drivers, keyed by the name of the Marmot material base class.
materialDrivers = {
    "MarmotMaterialHypoElastic": MarmotMaterialHypoElasticDriver,
    "MarmotMaterialGradientEnhancedHypoElastic": MarmotMaterialGradientEnhancedHypoElasticDriver,
    "MarmotMaterialGradientPlasticityHypoElastic": MarmotMaterialGradientPlasticityHypoElasticDriver,
}


def materialDriverForMaterialType(materialType: str) -> type:
    """Get the material driver for the requested Marmot material base class.

    Parameters
    ----------
    materialType
        The name of the Marmot material base class.

    Returns
    -------
    type
        The material driver class.
    """

    try:
        return materialDrivers[materialType]
    except KeyError:
        raise KeyError(
            "No material driver for Marmot material base class '{:}'; available are {:}".format(
                materialType, ", ".join(materialDrivers)
            )
        )
