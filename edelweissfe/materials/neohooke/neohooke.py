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

from edelweissfe.materials.base.basehyperelasticmaterial import BaseHyperElasticMaterial
from edelweissfe.utils.voigtnotation import doVoigtStress


def _kirchhoffWa(mu, K, F):
    """Energy, Kirchhoff stress and tangent for W_a (I1, J) = mu/2*(I1-3) + (K/2-mu/3)(J-1)^2 - mu*ln(J)."""

    B = F @ F.T  # left Cauchy-Green tensor
    invF = lin.inv(F)
    J = lin.det(F)
    I1 = np.trace(B)
    lambdaBar = (K - 2 / 3 * mu) * (J**2 - J) - mu
    muBar = (K - 2 / 3 * mu) * (2 * J**2 - J)
    T = mu * B + lambdaBar * np.eye(3)
    CTauF = mu * (np.einsum("ik,jl->ijkl", np.eye(3), F) + np.einsum("il,jk->ijkl", F, np.eye(3))) + muBar * np.einsum(
        "ij,lk->ijkl", np.eye(3), invF
    )
    energy = mu / 2 * (I1 - 3) + (K / 2 - mu / 3) * (J - 1) ** 2 - mu * np.log(J)
    return (energy, T, CTauF)


def _kirchhoffWb(mu, K, F):
    """Energy, Kirchhoff stress and tangent for W_b (I1, J) = mu/2*(I1/J^(2/3)-3) + K/8*(J^2+1/J^2-2)."""

    B = F @ F.T
    invF = lin.inv(F)
    J = lin.det(F)
    I1 = np.trace(B)
    lambdaHat = K / 2 * (J**2 + 1 / J**2)
    muBar = mu / (3 * J ** (2 / 3))
    lambdaBar = K / 4 * (J**2 - 1 / J**2) - muBar * I1
    T = mu / J ** (2 / 3) * B + lambdaBar * np.eye(3)
    CTauF = (
        3 * muBar * (np.einsum("ik,jl->ijkl", np.eye(3), F) + np.einsum("il,jk->ijkl", F, np.eye(3)))
        - 2 * muBar * np.einsum("ij,lk->ijkl", B, invF)
        + (lambdaHat + 2 / 3 * I1 * muBar) * np.einsum("ij,lk->ijkl", np.eye(3), invF)
        - 2 * muBar * np.einsum("ij,kl->ijkl", np.eye(3), F)
    )
    energy = mu / 2 * (I1 / J ** (2 / 3) - 3) + K / 8 * (J**2 + 1 / J**2 - 2)
    return (energy, T, CTauF)


def _kirchhoffWc(mu, K, F):
    """Energy, Kirchhoff stress and tangent for W_c (I1, J) = mu/2*(I1-3) + 3mu^2/(3K-2mu)*(J^(2/3-K/mu)-1)."""

    B = F @ F.T
    invF = lin.inv(F)
    J = lin.det(F)
    # NOTE: kept as trace(F) to preserve existing behavior; formulations Wa/Wb use trace(B) for I1 instead.
    I1 = np.trace(F)
    muBar = mu * J ** (2 / 3 - K / mu)
    lambdaBar = (K / mu - 2 / 3) * muBar
    T = mu * B - muBar * np.eye(3)
    CTauF = mu * (
        np.einsum("ik,jl->ijkl", np.eye(3), F) + np.einsum("il,jk->ijkl", F, np.eye(3))
    ) + lambdaBar * np.einsum("ij,lk->ijkl", np.eye(3), invF)
    energy = mu / 2 * (I1 - 3) + 3 * mu**2 / (3 * K - 2 * mu) * (J ** (2 / 3 - K / mu) - 1)
    return (energy, T, CTauF)


_FORMULATIONS = {1: _kirchhoffWa, 2: _kirchhoffWb, 3: _kirchhoffWc}


class NeoHookeanMaterial(BaseHyperElasticMaterial):
    """Compressible neo-Hookean material, Pence-Gou formulations Wa/Wb/Wc according to [1].

    [1] Pence, T. J., & Gou, K. (2015). On compressible versions of the incompressible neo-Hookean material.
        Mathematics and Mechanics of Solids, 20(2), 157–182. https://doi.org/10.1177/1081286514544258

    Parameters
    ----------
    materialProperties
        The numpy array containing the material properties for the requested material, in order:
        formulation selector (1=Wa, 2=Wb, 3=Wc), mu, K, and optionally the density."""

    def getNumberOfRequiredStateVars(self) -> int:
        """Returns number of needed material state Variables per integration point in the material.

        Returns
        -------
        int
            Number of needed material state Vars."""

        return 1

    def __init__(self, materialProperties: np.ndarray):
        self._materialProperties = materialProperties
        formulation = int(materialProperties[0])
        if formulation not in _FORMULATIONS:
            raise Exception(
                "Unknown Neo-Hookean Pence-Gou formulation selector {:}; must be 1 (Wa), 2 (Wb), or 3 (Wc).".format(
                    formulation
                )
            )
        self._computeKirchhoffImpl = _FORMULATIONS[formulation]
        self._mu = materialProperties[1]
        self._K = materialProperties[2]
        if len(materialProperties) > 3:
            self._density = materialProperties[3]

    def assignCurrentStateVars(self, currentStateVars: np.ndarray):
        """Assign new current state vars.

        Parameters
        ----------
        currentStateVars
            Array containing the material state vars."""

        self._energy = currentStateVars

    def computeKirchhoff(
        self,
        stress: np.ndarray,
        dStress_dDeformationGradient: np.ndarray,
        deformationGradient: np.ndarray,
        time: float,
        dTime: float,
    ):
        """Computes the stresses for a 3D material/2D material with plane strain.

        Parameters
        ----------
        stress
            Vector containing the stresses.
        dStress_dDeformationGradient
            Matrix containing dStress/dStrain.
        deformationGradient
            The deformation gradient at time step t.
        time
            Array of step time and total time.
        dTime
            Current time step size."""

        energy, T, CTauF = self._computeKirchhoffImpl(self._mu, self._K, deformationGradient)
        stress[:] = doVoigtStress(3, T)
        dStress_dDeformationGradient[:] = CTauF
        self._energy[0] = energy

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

        if result == "energy":
            return self._energy
        else:
            raise Exception("This result doesn't exist for the current material.")
