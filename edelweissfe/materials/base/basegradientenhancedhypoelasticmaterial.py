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
This module defines the interface for general gradient-enhanced hypoelastic materials,
i.e., materials which, in addition to the balance of linear momentum, introduce one
balance equation per nonlocal variable :math:`\\bar\\kappa_i`

.. math::
    \\bar\\kappa_i - \\nabla \\cdot \\left( c(\\bar\\kappa_i)\\, \\nabla \\bar\\kappa_i \\right)
    = \\kappa_i(\\boldsymbol \\varepsilon,\\, \\bar\\kappa_i)

with the nonlocal interaction parameter :math:`c(\\bar\\kappa_i)` and the local driving
variable :math:`\\kappa_i`.

The interface deliberately mirrors ``MarmotMaterialGeneralGradientEnhancedHypoElastic``
of `Marmot <https://github.com/MAteRialMOdelingToolbox/Marmot/>`_: the quantities are
grouped into an :class:`GradientEnhancedIncrement` (the input), an
:class:`GradientEnhancedResponse` and a set of :class:`GradientEnhancedTangents`, such
that both discretizations (finite elements and finite differences) and both material
providers (Marmot and EdelweissFE) speak the same language.

All containers hold preallocated arrays and are meant to be created once per material
point and reused in every iteration.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np


@dataclass
class GradientEnhancedIncrement:
    """The input of a gradient-enhanced material evaluation.

    Parameters
    ----------
    dStrain
        The strain increment in Voigt notation, shape ``(6,)``.
    K
        The nonlocal variables at the end of the increment, shape ``(nNonlocalVariables,)``.
    dK
        The increment of the nonlocal variables, shape ``(nNonlocalVariables,)``.
    """

    dStrain: np.ndarray
    K: np.ndarray
    dK: np.ndarray

    @classmethod
    def createZero(cls, nNonlocalVariables: int):
        """Create an instance with all entries set to zero.

        Parameters
        ----------
        nNonlocalVariables
            The number of nonlocal variables of the material.

        Returns
        -------
        GradientEnhancedIncrement
            The zero initialized instance.
        """

        return cls(
            dStrain=np.zeros(6),
            K=np.zeros(nNonlocalVariables),
            dK=np.zeros(nNonlocalVariables),
        )

    def zero(self):
        """Reset all entries to zero."""

        self.dStrain[:] = 0.0
        self.K[:] = 0.0
        self.dK[:] = 0.0


@dataclass
class GradientEnhancedResponse:
    """The response of a gradient-enhanced material evaluation.

    Parameters
    ----------
    stress
        The stress in Voigt notation, shape ``(6,)``.
    KLocal
        The local driving variables, shape ``(nNonlocalVariables,)``.
    c
        The nonlocal interaction parameters, shape ``(nNonlocalVariables,)``.
    elasticEnergyDensity
        The elastic strain energy density.
    dissipation
        The dissipation, if applicable.
    """

    stress: np.ndarray
    KLocal: np.ndarray
    c: np.ndarray
    elasticEnergyDensity: float = 0.0
    dissipation: float = 0.0

    @classmethod
    def createZero(cls, nNonlocalVariables: int):
        """Create an instance with all entries set to zero.

        Parameters
        ----------
        nNonlocalVariables
            The number of nonlocal variables of the material.

        Returns
        -------
        GradientEnhancedResponse
            The zero initialized instance.
        """

        return cls(
            stress=np.zeros(6),
            KLocal=np.zeros(nNonlocalVariables),
            c=np.zeros(nNonlocalVariables),
        )

    def zero(self):
        """Reset all entries to zero."""

        self.stress[:] = 0.0
        self.KLocal[:] = 0.0
        self.c[:] = 0.0
        self.elasticEnergyDensity = 0.0
        self.dissipation = 0.0


@dataclass
class GradientEnhancedTangents:
    """The algorithmic tangents of a gradient-enhanced material evaluation.

    Parameters
    ----------
    dStress_dStrain
        The tangent relating the stress to the strain, shape ``(6, 6)``.
    dStress_dK
        The tangent relating the stress to the nonlocal variables, shape ``(6, n)``.
    dKLocal_dStrain
        The tangent relating the local driving variables to the strain, shape ``(n, 6)``.
    dKLocal_dK
        The tangent relating the local driving variables to the nonlocal variables, shape ``(n, n)``.
    dc_dK
        The first derivative of the nonlocal interaction parameters with respect to the
        nonlocal variables, shape ``(n, n)``.
    d2c_dK2
        The second derivative of the nonlocal interaction parameters with respect to the
        nonlocal variables, shape ``(n, n)``.
    """

    dStress_dStrain: np.ndarray
    dStress_dK: np.ndarray
    dKLocal_dStrain: np.ndarray
    dKLocal_dK: np.ndarray
    dc_dK: np.ndarray
    d2c_dK2: np.ndarray

    #: The names of all tangent entries, in the order in which Marmot declares them.
    entryNames: tuple = field(
        default=(
            "dStress_dStrain",
            "dStress_dK",
            "dKLocal_dStrain",
            "dKLocal_dK",
            "dc_dK",
            "d2c_dK2",
        ),
        repr=False,
    )

    @classmethod
    def createZero(cls, nNonlocalVariables: int):
        """Create an instance with all entries set to zero.

        Parameters
        ----------
        nNonlocalVariables
            The number of nonlocal variables of the material.

        Returns
        -------
        GradientEnhancedTangents
            The zero initialized instance.
        """

        n = nNonlocalVariables

        shapes = ((6, 6), (6, n), (n, 6), (n, n), (n, n), (n, n))

        # One contiguous block with the entries as views into it, so that resetting is a single
        # fill rather than one per entry. The entries are small -- a six by six and five smaller
        # ones -- so for a material point evaluated hundreds of thousands of times in a
        # simulation, the per call overhead of six numpy operations is what dominates, not the
        # arithmetic, cf. GradientPlasticityTangents.createZero.
        block = np.zeros(sum(rows * columns for rows, columns in shapes))

        views, offset = [], 0
        for rows, columns in shapes:
            views.append(block[offset : offset + rows * columns].reshape(rows, columns))
            offset += rows * columns

        instance = cls(*views)
        instance._block = block

        return instance

    def zero(self):
        """Reset all entries to zero.

        A single fill of the block the entries are views into, see :meth:`createZero`. Falls back
        to entry by entry for an instance that was built by hand rather than by ``createZero``, so
        that the class stays usable either way.
        """

        block = getattr(self, "_block", None)

        if block is not None:
            block.fill(0.0)
            return

        for name in self.entryNames:
            getattr(self, name)[:] = 0.0


class BaseGradientEnhancedHypoElasticMaterial(ABC):
    """Base material class for a general gradient-enhanced hypoelastic material.

    Parameters
    ----------
    materialProperties
        The numpy array containing the material properties for the requested material."""

    @property
    @abstractmethod
    def nNonlocalVariables(self) -> int:
        """The number of nonlocal variables this material introduces."""

    @property
    def materialProperties(self) -> np.ndarray:
        """The properties the material has."""

    @abstractmethod
    def __init__(self, materialProperties: np.ndarray):
        """Initialize."""

    @abstractmethod
    def getNumberOfRequiredStateVars(self) -> int:
        """Returns number of needed material state Variables per integration point in the material.

        Returns
        -------
        int
            Number of needed material state Vars."""

    @abstractmethod
    def assignCurrentStateVars(self, currentStateVars: np.ndarray):
        """Assign new current state vars.

        Parameters
        ----------
        currentStateVars
            Array containing the material state vars."""

    @abstractmethod
    def computeStress(
        self,
        response: GradientEnhancedResponse,
        tangents: GradientEnhancedTangents,
        increment: GradientEnhancedIncrement,
        time: float,
        dTime: float,
    ):
        """Compute the material response and the algorithmic tangents for a 3D
        material / 2D material with plane strain.

        Parameters
        ----------
        response
            The response container to be filled.
        tangents
            The tangents container to be filled.
        increment
            The increment describing the strain increment and the nonlocal variables.
        time
            The total time at the end of the increment.
        dTime
            Current time step size."""

    def computePlaneStress(
        self,
        response: GradientEnhancedResponse,
        tangents: GradientEnhancedTangents,
        increment: GradientEnhancedIncrement,
        time: float,
        dTime: float,
    ):
        """Compute the material response and the algorithmic tangents for a 2D material
        with plane stress, i.e. with the out-of-plane strain condensed out.

        Parameters
        ----------
        response
            The response container to be filled.
        tangents
            The tangents container to be filled.
        increment
            The increment describing the strain increment and the nonlocal variables.
        time
            The total time at the end of the increment.
        dTime
            Current time step size."""

        raise NotImplementedError("This material does not provide a plane stress state.")

    @abstractmethod
    def getDensity(self) -> float:
        """Determines the density of the material.

        Returns
        -------
        float
            The density of the material."""

    @abstractmethod
    def getResult(self, result: str) -> np.ndarray:
        """Get the result, as a persistent view which is continiously
        updated by the material.

        Parameters
        ----------
        result
            The name of the result.

        Returns
        -------
        np.ndarray
            The result.
        """
