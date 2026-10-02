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
"""Pins ``ReconstructStressFromStrain``: an elastic child quadrature point gets exactly the strain of
the parent's displacement field at its location (Marmot's Voigt order, engineering shear) and
``C : strain`` as its stress; a point with non-virgin material state keeps the fallback transfer."""

import itertools

import numpy as np
import pytest

from edelweissfe.adaptivity.hex20shapefunctions import hex20_box_coords
from edelweissfe.adaptivity.hex20topology import Hex20Topology
from edelweissfe.adaptivity.statetransfer import (
    NearestQuadraturePointCopy,
    ReconstructStressFromStrain,
)
from edelweissfe.materials.linearelastic.linearelastic import LinearElasticMaterial
from edelweissfe.modelmodifiers.adaptivity.hadaptivity import (
    _buildStateTransferStrategy,
)

E, NU = 30600.0, 0.2
C = LinearElasticMaterial(np.array([E, NU])).elasticityMatrix()
GAUSS = np.array(list(itertools.product((-1 / np.sqrt(3), 1 / np.sqrt(3)), repeat=3)))
# per-QP block as Marmot's displacement elements lay it out: stress, strain, two energies, material
SLICES = {"stress": (0, 6), "strain": (6, 6), "begin of material state": (14, 0), "omega": (15, 1)}
BLOCK = 16  # ... + two material state variables
TOPOLOGY = Hex20Topology()


class _Node:
    def __init__(self, coordinates):
        self.coordinates = np.asarray(coordinates, dtype=float)


class _Element:
    """The element interface the state transfer uses, with Marmot's per-QP block layout."""

    def __init__(self, nodeCoords, stateVars=None):
        self.nodes = [_Node(x) for x in nodeCoords]
        self._state = np.zeros(8 * BLOCK) if stateVars is None else np.asarray(stateVars, dtype=float).copy()

    def getNumberOfQuadraturePoints(self):
        return 8

    def getCoordinatesAtQuadraturePoints(self):
        coords = np.array([n.coordinates for n in self.nodes])
        return np.array([TOPOLOGY.shape_functions(*xi) @ coords for xi in GAUSS])

    def getStateVars(self):
        return self._state

    def setStateVars(self, values):
        self._state = np.asarray(values, dtype=float).copy()

    def getStateVarSlice(self, name):
        return SLICES[name]


PARENT_COORDS = hex20_box_coords(0.0, 8.0, -8.0, 0.0, -8.0, 0.0)


def voigtStrain(gradient):
    """Small strain in Marmot's Voigt order with engineering shear, from du_i/dx_j."""
    g = gradient
    return np.array([g[0, 0], g[1, 1], g[2, 2], g[0, 1] + g[1, 0], g[0, 2] + g[2, 0], g[1, 2] + g[2, 1]])


def parentWithDisplacement(displacementOf):
    """An elastic parent whose QP stress/strain are consistent with the nodal displacement field."""
    parent = _Element(PARENT_COORDS)
    U = np.array([displacementOf(x) for x in PARENT_COORDS])
    return parent, U


def children():
    return [_Element(c) for c in TOPOLOGY.subdivide(PARENT_COORDS, 2)]


def test_linearFieldGivesExactStressAndStrain():
    gradient = np.array([[1.0e-4, 2.0e-5, -3.0e-5], [4.0e-5, -2.0e-4, 1.0e-5], [-5.0e-5, 6.0e-5, 3.0e-4]])
    parent, U = parentWithDisplacement(lambda x: gradient @ x + np.array([0.1, -0.2, 0.3]))
    kids = children()
    strategy = ReconstructStressFromStrain(NearestQuadraturePointCopy(), E, NU)
    for child in kids:
        strategy.transferState(parent, [child], TOPOLOGY, U)
        block = child.getStateVars().reshape(8, BLOCK)
        np.testing.assert_allclose(block[:, 6:12], np.tile(voigtStrain(gradient), (8, 1)), rtol=1e-11, atol=1e-18)
        np.testing.assert_allclose(block[:, 0:6], np.tile(C @ voigtStrain(gradient), (8, 1)), rtol=1e-12, atol=1e-12)
    assert "at 64 child" in strategy.reportAndResetTransferStatistics()
    assert strategy.reportAndResetTransferStatistics() is None


def test_engineeringShearConvention():
    """u_1 = k x_2: Marmot's B-operator gives gamma_12 = k (engineering shear), not k/2."""
    k = 1.0e-3
    parent, U = parentWithDisplacement(lambda x: np.array([k * x[1], 0.0, 0.0]))
    child = children()[0]
    ReconstructStressFromStrain(NearestQuadraturePointCopy(), E, NU).transferState(parent, [child], TOPOLOGY, U)
    block = child.getStateVars().reshape(8, BLOCK)
    np.testing.assert_allclose(block[:, 6:12], np.tile([0, 0, 0, k, 0, 0], (8, 1)), atol=1e-15)
    np.testing.assert_allclose(block[:, 3], E / (2 * (1 + NU)) * k, rtol=1e-12)


def test_quadraticFieldStrainIsTheParentFieldsStrainAtTheChildPoints():
    def u(x):
        return np.array([1e-5 * x[0] ** 2 + 2e-5 * x[1] * x[2], -3e-5 * x[1] ** 2, 4e-5 * x[0] * x[1]])

    def du_dX(x):
        return np.array([[2e-5 * x[0], 2e-5 * x[2], 2e-5 * x[1]], [0, -6e-5 * x[1], 0], [4e-5 * x[1], 4e-5 * x[0], 0]])

    parent, U = parentWithDisplacement(u)
    for child in children():
        ReconstructStressFromStrain(NearestQuadraturePointCopy(), E, NU).transferState(parent, [child], TOPOLOGY, U)
        block = child.getStateVars().reshape(8, BLOCK)
        expected = np.array([voigtStrain(du_dX(x)) for x in child.getCoordinatesAtQuadraturePoints()])
        np.testing.assert_allclose(block[:, 6:12], expected, rtol=1e-11, atol=1e-18)


def test_nonVirginMaterialStateKeepsTheFallback():
    parentState = np.zeros((8, BLOCK))
    parentState[:, 0:6] = 1.0  # a stress unrelated to the displacement
    parentState[7, 14] = 0.4  # material state (e.g. a hardening variable) active at one parent QP
    parent = _Element(PARENT_COORDS, parentState.reshape(-1))
    U = np.zeros((20, 3))
    strategy = ReconstructStressFromStrain(NearestQuadraturePointCopy(), E, NU)
    kept = 0
    for child in children():
        strategy.transferState(parent, [child], TOPOLOGY, U)
        block = child.getStateVars().reshape(8, BLOCK)
        plastic = block[:, 14] > 0
        np.testing.assert_array_equal(block[plastic, 0:6], 1.0)  # copied by the fallback
        np.testing.assert_array_equal(block[~plastic, 0:6], 0.0)  # rebuilt: zero displacement
        kept += int(plastic.sum())
    assert kept == 8
    assert "at 56 child" in strategy.reportAndResetTransferStatistics()


def test_builderNeedsElasticConstantsAndRejectsPerVariableUse():
    with pytest.raises(ValueError, match="elasticModulusForStressReconstruction"):
        _buildStateTransferStrategy("reconstructStressFromStrain", None)
    with pytest.raises(ValueError, match="cannot be routed"):
        _buildStateTransferStrategy("nearestQp", "stress:reconstructStressFromStrain")
    strategy = _buildStateTransferStrategy("reconstructStressFromStrain", "alphaP:limitedProjection", E, NU)
    assert isinstance(strategy, ReconstructStressFromStrain)


def test_namedDamageScalesTheRebuiltStressAndDoesNotForceTheFallback():
    """Damage without plastic flow (a nonlocal damage field reaching an elastic point): with the
    damage variable named, the point is rebuilt as (1 - omega) C : strain; unnamed, it falls back."""
    gradient = np.array([[1.0e-4, 0.0, 0.0], [0.0, -2.0e-5, 3.0e-5], [0.0, 3.0e-5, 5.0e-5]])
    parentState = np.zeros((8, BLOCK))
    parentState[:, 15] = 0.3
    parent = _Element(PARENT_COORDS, parentState.reshape(-1))
    U = np.array([gradient @ x for x in PARENT_COORDS])

    named = ReconstructStressFromStrain(NearestQuadraturePointCopy(), E, NU, damageStateVarName="omega")
    for child in children():
        named.transferState(parent, [child], TOPOLOGY, U)
        block = child.getStateVars().reshape(8, BLOCK)
        np.testing.assert_allclose(block[:, 15], 0.3)
        np.testing.assert_allclose(
            block[:, 0:6], np.tile(0.7 * C @ voigtStrain(gradient), (8, 1)), rtol=1e-12, atol=1e-12
        )
    assert "at 64 child" in named.reportAndResetTransferStatistics()

    unnamed = ReconstructStressFromStrain(NearestQuadraturePointCopy(), E, NU)
    for child in children():
        unnamed.transferState(parent, [child], TOPOLOGY, U)
    assert "at 0 child" in unnamed.reportAndResetTransferStatistics()
