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
This module defines the interface for gradient plasticity materials of hypoelastic type,
i.e., materials whose yield condition depends not only on the plastic multiplier
:math:`\\lambda` but also on its Laplacian,

.. math::
    f\\left( \\boldsymbol \\sigma,\\, \\lambda,\\, \\nabla^2 \\lambda \\right) = 0

The additional balance equation solved alongside the balance of linear momentum is that
yield condition itself, one per yield surface, which is what distinguishes this family from
the gradient *enhanced* materials of
:mod:`~edelweissfe.materials.base.basegradientenhancedhypoelasticmaterial`: there the extra
equation is a screened Poisson equation supplied by the discretization, here the material
provides the equation and merely asks to be told the Laplacian.

The interface mirrors ``MarmotMaterialGradientPlasticityHypoElastic`` of
`Marmot <https://github.com/MAteRialMOdelingToolbox/Marmot/>`_: the quantities are grouped
into a :class:`GradientPlasticityIncrement` (the input), a
:class:`GradientPlasticityResponse` and a set of :class:`GradientPlasticityTangents`.

All containers hold preallocated arrays and are meant to be created once per material point
and reused in every iteration.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np


@dataclass
class GradientPlasticityIncrement:
    """The input of a gradient plasticity material evaluation.

    Parameters
    ----------
    dStrain
        The strain increment in Voigt notation, shape ``(6,)``.
    dLambda
        The increment of the plastic multipliers, shape ``(nYieldSurfaces,)``.
    laplaceDLambda
        The Laplacian of the increment of the plastic multipliers, shape ``(nYieldSurfaces,)``.
    """

    dStrain: np.ndarray
    dLambda: np.ndarray
    laplaceDLambda: np.ndarray

    @classmethod
    def createZero(cls, nYieldSurfaces: int):
        """Create an instance with all entries set to zero.

        Parameters
        ----------
        nYieldSurfaces
            The number of yield surfaces of the material.

        Returns
        -------
        GradientPlasticityIncrement
            The zero initialized instance.
        """

        return cls(
            dStrain=np.zeros(6),
            dLambda=np.zeros(nYieldSurfaces),
            laplaceDLambda=np.zeros(nYieldSurfaces),
        )

    def zero(self):
        """Reset all entries to zero."""

        self.dStrain[:] = 0.0
        self.dLambda[:] = 0.0
        self.laplaceDLambda[:] = 0.0


@dataclass
class GradientPlasticityResponse:
    """The response of a gradient plasticity material evaluation.

    Parameters
    ----------
    stress
        The stress in Voigt notation, shape ``(6,)``.
    f
        The value of the yield function per yield surface, shape ``(nYieldSurfaces,)``. It is
        the residual of the additional balance equation and has to vanish in equilibrium.
    elasticEnergyDensity
        The elastic strain energy density.
    dissipation
        The dissipation, if applicable.
    """

    stress: np.ndarray
    f: np.ndarray
    elasticEnergyDensity: float = 0.0
    dissipation: float = 0.0

    @classmethod
    def createZero(cls, nYieldSurfaces: int):
        """Create an instance with all entries set to zero.

        Parameters
        ----------
        nYieldSurfaces
            The number of yield surfaces of the material.

        Returns
        -------
        GradientPlasticityResponse
            The zero initialized instance.
        """

        return cls(stress=np.zeros(6), f=np.zeros(nYieldSurfaces))

    def zero(self):
        """Reset all entries to zero."""

        self.stress[:] = 0.0
        self.f[:] = 0.0
        self.elasticEnergyDensity = 0.0
        self.dissipation = 0.0


@dataclass
class GradientPlasticityTangents:
    """The algorithmic tangents of a gradient plasticity material evaluation.

    Parameters
    ----------
    dStress_dStrain
        The tangent relating the stress to the strain, shape ``(6, 6)``.
    dStress_dLambda
        The tangent relating the stress to the plastic multipliers, shape ``(6, n)``.
    dStress_dLaplacian
        The tangent relating the stress to the Laplacian of the plastic multipliers,
        shape ``(6, n)``.
    dF_dStrain
        The tangent relating the yield function values to the strain, shape ``(n, 6)``.
    dF_dLambda
        The tangent relating the yield function values to the plastic multipliers,
        shape ``(n, n)``.
    dF_dLaplacian
        The tangent relating the yield function values to the Laplacian of the plastic
        multipliers, shape ``(n, n)``.
    """

    dStress_dStrain: np.ndarray
    dStress_dLambda: np.ndarray
    dStress_dLaplacian: np.ndarray
    dF_dStrain: np.ndarray
    dF_dLambda: np.ndarray
    dF_dLaplacian: np.ndarray

    #: The names of all tangent entries, in the order in which Marmot declares them.
    entryNames: tuple = field(
        default=(
            "dStress_dStrain",
            "dStress_dLambda",
            "dStress_dLaplacian",
            "dF_dStrain",
            "dF_dLambda",
            "dF_dLaplacian",
        ),
        repr=False,
    )

    @classmethod
    def createZero(cls, nYieldSurfaces: int):
        """Create an instance with all entries set to zero.

        Parameters
        ----------
        nYieldSurfaces
            The number of yield surfaces of the material.

        Returns
        -------
        GradientPlasticityTangents
            The zero initialized instance.
        """

        n = nYieldSurfaces

        shapes = ((6, 6), (6, n), (6, n), (n, 6), (n, n), (n, n))

        # One contiguous block with the entries as views into it, so that resetting is a single
        # fill rather than one per entry. The entries are small -- a six by six and five smaller
        # ones -- so for a material point evaluated hundreds of thousands of times in a
        # simulation, the per call overhead of six numpy operations is what dominates, not the
        # arithmetic. Measured on a two dimensional gradient plasticity stencil, zeroing went from
        # 2.4 to 0.3 microseconds, out of 40 microseconds per material point.
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


class BaseGradientPlasticityHypoElasticMaterial(ABC):
    """Base material class for a gradient plasticity material of hypoelastic type.

    Parameters
    ----------
    materialProperties
        The numpy array containing the material properties for the requested material."""

    @property
    @abstractmethod
    def nYieldSurfaces(self) -> int:
        """The number of yield surfaces, i.e. of plastic multipliers, this material has."""

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
        response: GradientPlasticityResponse,
        tangents: GradientPlasticityTangents,
        increment: GradientPlasticityIncrement,
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
            The increment describing the strain increment, the plastic multipliers and their
            Laplacian.
        time
            The total time at the end of the increment.
        dTime
            Current time step size."""

    def computePlaneStress(
        self,
        response: GradientPlasticityResponse,
        tangents: GradientPlasticityTangents,
        increment: GradientPlasticityIncrement,
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
            The increment describing the strain increment, the plastic multipliers and their
            Laplacian.
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
