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
"""The Marmot heat conduction elements (DC*T) through the element wrapper: field, loads and their residual and tangent
convention (see ``test_residual_tangent_convention.py``)."""

import numpy as np
import pytest

from edelweissfe.config.phenomena import getFieldSize, phenomena
from edelweissfe.points.node import Node
from edelweissfe.stepactions.base.nodalloadbase import NodalLoadBase
from edelweissfe.timesteppers.timestep import TimeStep


def _heatElement(elType="DC2D4T", corners=((0, 0), (2, 0), (2, 1), (0, 1)), thickness=3.0):
    elementModule = pytest.importorskip("edelweissfe.elements.marmotelement.element")
    try:
        element = elementModule.MarmotElementWrapper(elType, 1)
    except NotImplementedError:
        pytest.skip(f"this Marmot build has no {elType}")
    element.setNodes([Node(i, np.array(x, dtype=float)) for i, x in enumerate(corners)])
    element.setProperties(np.array([thickness]))
    element.initializeElement()
    element.setMaterial("HEATCONDUCTION", np.array([2.0, 3.0, 1.5, 0.01, 0.0, 20.0]))
    return element


def test_temperature_is_a_scalar_field():
    assert phenomena["temperature"] == "scalar"
    assert getFieldSize("temperature", 3) == 1


def test_heat_element_has_one_temperature_per_node():
    element = _heatElement()
    assert element.fields == [["temperature"]] * 4
    assert element.nDof == 4


def test_heat_flux_is_the_negative_external_load():
    """A heat flux q into the body on the top face (length 2, thickness 3) puts half of 6 q on each of its nodes."""
    element = _heatElement()
    P = np.zeros(4)
    K = np.zeros(16)
    element.computeDistributedLoad("heat flux", P, K, 3, np.array([5.0]), np.zeros(4), 0.0, 1.0)
    expected = np.array([0.0, 0.0, -15.0, -15.0])
    np.testing.assert_allclose(P, expected, rtol=1e-13, atol=1e-14)
    np.testing.assert_array_equal(K, 0.0)


@pytest.mark.parametrize(
    "loadType, load", [("heat film", [7.0, 290.0]), ("heat radiation", [0.8 * 5.67e-8, 300.0, -273.15])]
)
def test_heat_surface_load_tangent_is_the_derivative_of_its_residual_contribution(loadType, load):
    """Film and radiation depend on the temperature: K = dP/dT of the residual contribution, by central differences,
    and a body warmer than its surroundings loses heat (positive residual contribution)."""
    element = _heatElement()
    nDof = element.nDof
    T = 400.0 + 30.0 * np.sin(np.arange(nDof))

    def contribution(faceID, T):
        P = np.zeros(nDof)
        K = np.zeros(nDof * nDof)
        element.computeDistributedLoad(loadType, P, K, faceID, np.array(load), T, 0.0, 1.0)
        return P, K.reshape((nDof, nDof), order="F")

    for faceID in range(1, 5):
        P, K = contribution(faceID, T)
        h = 1e-4
        KFD = np.column_stack(
            [(contribution(faceID, T + h * e)[0] - contribution(faceID, T - h * e)[0]) / (2 * h) for e in np.eye(nDof)]
        )
        np.testing.assert_allclose(K, KFD, rtol=1e-7, atol=1e-9 * np.abs(K).max())
        assert P.sum() > 0.0


def test_heat_load_with_too_few_values_is_rejected():
    element = _heatElement()
    with pytest.raises(ValueError):
        element.computeDistributedLoad(
            "heat radiation", np.zeros(4), np.zeros(16), 1, np.array([1.0, 2.0]), np.zeros(4), 0.0, 1.0
        )


def test_zero_time_increment_lets_no_heat_flow():
    """Nothing flows in a zero time increment (e.g. the zero increment at the beginning of a step): an element at a
    nonuniform but unchanged temperature has no residual, and the loads vanish."""
    element = _heatElement()
    T = np.array([10.0, 50.0, 80.0, 20.0])
    P = np.zeros(4)
    K = np.zeros(16)
    element.computeKernels(K, P, T, np.zeros(4), 0.0, 0.0)
    np.testing.assert_array_equal(P, 0.0)
    element.computeKernels(K, P, T, np.zeros(4), 0.0, 1.0)
    assert np.abs(P).max() > 0.0
    PLoad = np.zeros(4)
    element.computeDistributedLoad("heat film", PLoad, np.zeros(16), 1, np.array([7.0, 20.0]), T, 0.0, 0.0)
    np.testing.assert_array_equal(PLoad, 0.0)


class _PointLoad(NodalLoadBase):
    """A load of 2 on the single degree of freedom of one node of the given field."""

    def __init__(self, field):
        self._field = field

    @property
    def field(self):
        return self._field

    @property
    def nodeSet(self):
        return None

    def getCurrentLoad(self, timeStep):
        return np.array([[2.0]])

    def applyAtStepEnd(self, model, loadParameter=None):
        pass


class _OneDofNumbering:
    @staticmethod
    def locateFieldOnNodeSet(field, nodeSet):
        return np.array([0]), np.array([0])


class _OneDofLocations:
    numbering = _OneDofNumbering


@pytest.mark.parametrize(
    "field, timeIncrement, expected",
    [("temperature", 0.0, 0.0), ("temperature", 0.5, -2.0), ("displacement", 0.0, -2.0), ("displacement", 0.5, -2.0)],
)
def test_point_heat_source_supplies_nothing_in_a_zero_time_increment(field, timeIncrement, expected):
    """A point load on a field whose fluxes are rates (a heat flow in W) supplies nothing while no time passes, like
    the distributed and body loads of the transport elements: the zero increment at the beginning of a step would
    otherwise balance it against the capacity alone and add the heat of one unit of time. Forces are not affected."""
    P = np.zeros(1)
    timeStep = TimeStep(0, timeIncrement, timeIncrement, timeIncrement, timeIncrement, timeIncrement)
    _PointLoad(field).assembleLoad(P, None, None, None, timeStep, _OneDofLocations())
    assert P[0] == expected
