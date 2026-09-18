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

from edelweissfe.materials.base.basehypoelasticmaterial import BaseHypoElasticMaterial


class LinearElasticMaterial(BaseHypoElasticMaterial):
    """Linear elastic material for 2D and 3D cases.

    Parameters
    ----------
    materialProperties
        The numpy array containing the material properties.

    Notes
    ----
    **Material Properties**

    - E: Elasticity module (Young's modulus).
    - v: Poisson's ratio."""

    def getNumberOfRequiredStateVars(self) -> int:
        """Returns number of needed material state Variables per integration point in the material.

        Returns
        -------
        int
            Number of needed material state Vars."""

        return 0

    def __init__(self, materialProperties: np.ndarray):
        self._E = materialProperties[0]  # set E
        self._v = materialProperties[1]  # set v
        if len(materialProperties) > 2:
            self._density = materialProperties[2]
        # E and v are immutable for the lifetime of this instance (ChangeMaterialProperty
        # constructs a new material instance rather than mutating an existing one), so the
        # elasticity matrices only need to be assembled once.
        self._EiPlaneStress = (
            self._E / (1 - self._v**2) * np.array([[1, self._v, 0], [self._v, 1, 0], [0, 0, (1 - self._v) / 2]])
        )
        self._Ei = (
            self._E
            / ((1 + self._v) * (1 - 2 * self._v))
            * np.array(
                [
                    [(1 - self._v), self._v, self._v, 0, 0, 0],
                    [self._v, (1 - self._v), self._v, 0, 0, 0],
                    [self._v, self._v, (1 - self._v), 0, 0, 0],
                    [0, 0, 0, (1 - 2 * self._v) / 2, 0, 0],
                    [0, 0, 0, 0, (1 - 2 * self._v) / 2, 0],
                    [0, 0, 0, 0, 0, (1 - 2 * self._v) / 2],
                ]
            )
        )

    def elasticityMatrixPlaneStress(self) -> np.ndarray:
        """The 2D plane stress material elasticity matrix.

        Returns
        -------
        np.ndarray
            The elasticity matrix."""

        return self._EiPlaneStress

    def elasticityMatrix(self) -> np.ndarray:
        """The 3D material elasticity matrix.

        Returns
        -------
        np.ndarray
            The elasticity matrix."""

        return self._Ei

    def assignCurrentStateVars(self, currentStateVars: np.ndarray):
        """Assign new current state vars.

        Parameters
        ----------
        currentStateVars
            Array containing the material state vars."""

    def computePlaneStress(
        self,
        stress: np.ndarray,
        dStress_dStrain: np.ndarray,
        dStrain: np.ndarray,
        time: float,
        dTime: float,
    ):
        """Computes the stresses for a 2D material with plane stress.

        Parameters
        ----------
        stress
            Vector containing the stresses.
        dStress_dStrain
            Matrix containing dStress/dStrain.
        dStrain
            Strain vector increment at time step t to t+dTime.
        time
            Array of step time and total time.
        dTime
            Current time step size."""

        Ei = self._EiPlaneStress
        index = [0, 1, 3]
        dStress_dStrain[:] = Ei
        stress[index] += Ei @ dStrain[index]
        dStrain[2] = -self._v / (1 - self._v) * (dStrain[0] + dStrain[1])  # dStrain33 for plane stress

    def computeStress(
        self,
        stress: np.ndarray,
        dStress_dStrain: np.ndarray,
        dStrain: np.ndarray,
        time: float,
        dTime: float,
    ):
        """Computes the stresses for a 3D material.

        Parameters
        ----------
        stress
            Vector containing the stresses.
        dStress_dStrain
            Matrix containing dStress/dStrain.
        dStrain
            Strain vector increment at time step t to t+dTime.
        time
            Array of step time and total time.
        dTime
            Current time step size."""

        Ei = self._Ei
        dStress_dStrain[:] = Ei
        stress += Ei @ dStrain

    def getResult(self, result: str) -> float:
        """Get the result, as a persistent view which is continiously
        updated by the material.

        Parameters
        ----------
        result
            The name of the result.

        Returns
        -------
        float
            The result.
        """

        raise Exception("This result doesn't exist for the current material.")
