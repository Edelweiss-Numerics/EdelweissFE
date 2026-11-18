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
"""The Marmot moisture diffusion elements (DC*H) through the element wrapper: field, loads and their residual and
tangent convention (see ``test_heat_transport.py``)."""

import numpy as np
import pytest

from edelweissfe.config.phenomena import getFieldSize, phenomena
from edelweissfe.points.node import Node

# Bazant-Najjar D(h) (D1, alpha0, hc, n), BSB isotherm (type 1, C, k, Vm), self-desiccation (dwInf, tauS)
_material = np.array([2.0e-3, 0.05, 0.8, 6.0, 1.0, 18.0, 0.73, 20.0, 5.0, 30.0])


def _moistureElement(elType="DC2D4H", corners=((0, 0), (2, 0), (2, 1), (0, 1)), thickness=3.0):
    elementModule = pytest.importorskip("edelweissfe.elements.marmotelement.element")
    try:
        element = elementModule.MarmotElementWrapper(elType, 1)
    except NotImplementedError:
        pytest.skip(f"this Marmot build has no {elType}")
    element.setNodes([Node(i, np.array(x, dtype=float)) for i, x in enumerate(corners)])
    element.setProperties(np.array([thickness]))
    element.initializeElement()
    element.setMaterial("MOISTUREDIFFUSION", _material)
    return element


def test_relative_humidity_is_a_scalar_field():
    assert phenomena["relative humidity"] == "scalar"
    assert getFieldSize("relative humidity", 3) == 1


def test_moisture_element_has_one_relative_humidity_per_node():
    element = _moistureElement()
    assert element.fields == [["relative humidity"]] * 4
    assert element.nDof == 4


def test_moisture_flux_is_the_negative_external_load():
    """A moisture flux q into the body on the top face (length 2, thickness 3) puts half of 6 q on each of its nodes."""
    element = _moistureElement()
    P = np.zeros(4)
    K = np.zeros(16)
    element.computeDistributedLoad("moisture flux", P, K, 3, np.array([5.0]), np.full(4, 0.9), 1.0, 1.0)
    np.testing.assert_allclose(P, [0.0, 0.0, -15.0, -15.0], rtol=1e-13, atol=1e-14)
    np.testing.assert_array_equal(K, 0.0)


def test_moisture_film_tangent_is_the_derivative_of_its_residual_contribution():
    """The film depends on h: K = dP/dh of the residual contribution, by central differences, and a body wetter than
    its environment dries (positive residual contribution)."""
    element = _moistureElement()
    nDof = element.nDof
    h0 = 0.8 + 0.1 * np.sin(np.arange(nDof))
    load = np.array([3.0e-2, 0.5])

    def contribution(faceID, h):
        P = np.zeros(nDof)
        K = np.zeros(nDof * nDof)
        element.computeDistributedLoad("moisture film", P, K, faceID, load, h, 1.0, 1.0)
        return P, K.reshape((nDof, nDof), order="F")

    for faceID in range(1, 5):
        P, K = contribution(faceID, h0)
        d = 1e-6
        KFD = np.column_stack(
            [
                (contribution(faceID, h0 + d * e)[0] - contribution(faceID, h0 - d * e)[0]) / (2 * d)
                for e in np.eye(nDof)
            ]
        )
        np.testing.assert_allclose(K, KFD, rtol=1e-7, atol=1e-9 * np.abs(K).max())
        assert P.sum() > 0.0


def test_moisture_and_heat_loads_are_checked():
    """A moisture film needs two values, and a heat load cannot act on an element without temperature."""
    element = _moistureElement()
    with pytest.raises(ValueError):
        element.computeDistributedLoad(
            "moisture film", np.zeros(4), np.zeros(16), 1, np.array([1.0]), np.zeros(4), 0.0, 1.0
        )
    with pytest.raises(Exception):
        element.computeDistributedLoad(
            "heat film", np.zeros(4), np.zeros(16), 1, np.array([1.0, 2.0]), np.zeros(4), 0.0, 1.0
        )


def test_element_tangent_matches_central_differences():
    """Nonlinear D(h), BSB isotherm and the self-desiccation sink: K = dP/dh."""
    element = _moistureElement()
    hOld = np.array([0.95, 0.9, 0.85, 0.92])
    h = np.array([0.7, 0.85, 0.78, 0.6])

    def residual(h):
        P = np.zeros(4)
        K = np.zeros(16)
        element.computeKernels(K, P, h, h - hOld, 5.0, 0.5)
        return P, K.reshape((4, 4), order="F")

    P, K = residual(h)
    d = 1e-6
    KFD = np.column_stack([(residual(h + d * e)[0] - residual(h - d * e)[0]) / (2 * d) for e in np.eye(4)])
    np.testing.assert_allclose(K, KFD, rtol=1e-6, atol=1e-8 * np.abs(K).max())


def test_zero_time_increment_lets_no_moisture_flow():
    """Nothing flows or is consumed in a zero time increment: an element at a nonuniform but unchanged relative humidity
    has no residual, and the loads vanish."""
    element = _moistureElement()
    h = np.array([0.5, 0.7, 0.9, 0.6])
    P = np.zeros(4)
    K = np.zeros(16)
    element.computeKernels(K, P, h, np.zeros(4), 1.0, 0.0)
    np.testing.assert_array_equal(P, 0.0)
    element.computeKernels(K, P, h, np.zeros(4), 1.0, 1.0)
    assert np.abs(P).max() > 0.0
    PLoad = np.zeros(4)
    element.computeDistributedLoad("moisture film", PLoad, np.zeros(16), 1, np.array([7.0, 0.5]), h, 1.0, 0.0)
    np.testing.assert_array_equal(PLoad, 0.0)
