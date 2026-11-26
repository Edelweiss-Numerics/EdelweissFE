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
"""The Marmot coupled heat and moisture transport elements (DC*TH) through the element wrapper: two fields per node,
the latent heat film and its coupling tangent, the coupled element tangent (all four blocks), and the uncoupled limit
against the single field elements DC*T and DC*H (see ``test_heat_transport.py`` and ``test_moisture_transport.py``)."""

import numpy as np
import pytest

from edelweissfe.points.node import Node

_heat = [2.4, 0.9, 1.8, 1.6, 1.7, 0.1, 0.05, -0.08, 0.002, 0.001, 20.0]
_moisture = [2.0e-3, 0.05, 0.8, 6.0, 1.0, 18.0, 0.73, 20.0, 5.0, 30.0]
# T_0, Q_D/R, m, kappa, chi, c_w, beta_kh, ell, E_h/R, Q_inf, tau_q: all couplings on
_couplings = [-273.15, 4700.0, 1.0, 0.01, 1.0, 4.186e-3, 0.3, 0.05, 4000.0, 60.0, 2.0]
_corners = ((0, 0), (2, 0), (2, 1), (0, 1))


def _element(elType, material, props, thickness=3.0):
    elementModule = pytest.importorskip("edelweissfe.elements.marmotelement.element")
    try:
        element = elementModule.MarmotElementWrapper(elType, 1)
    except NotImplementedError:
        pytest.skip(f"this Marmot build has no {elType}")
    element.setNodes([Node(i, np.array(x, dtype=float)) for i, x in enumerate(_corners)])
    element.setProperties(np.array([thickness]))
    element.initializeElement()
    element.setMaterial(material, np.array(props))
    return element


def _thermoHygro(props=_heat + _moisture + _couplings):
    return _element("DC2D4TH", "THERMOHYGRO", props)


def _fields(T, h):
    U = np.empty(2 * len(T))
    U[0::2], U[1::2] = T, h
    return U


def test_thermo_hygro_element_has_temperature_and_relative_humidity_per_node():
    element = _thermoHygro()
    assert element.fields == [["temperature", "relative humidity"]] * 4
    assert element.nDof == 8


def test_latent_heat_film_acts_on_the_temperature_and_depends_on_the_relative_humidity():
    """q_n = c (h - h_inf) on the temperature rows; its tangent sits in the relative humidity columns."""
    element = _thermoHygro()
    U0 = _fields([40.0, 45.0, 50.0, 42.0], [0.8, 0.85, 0.7, 0.9])
    load = np.array([25.0, 0.5])

    def contribution(faceID, U):
        P = np.zeros(8)
        K = np.zeros(64)
        element.computeDistributedLoad("latent heat film", P, K, faceID, load, U, 1.0, 1.0)
        return P, K.reshape((8, 8), order="F")

    for faceID in range(1, 5):
        P, K = contribution(faceID, U0)
        d = 1e-6
        KFD = np.column_stack(
            [(contribution(faceID, U0 + d * e)[0] - contribution(faceID, U0 - d * e)[0]) / (2 * d) for e in np.eye(8)]
        )
        np.testing.assert_allclose(K, KFD, rtol=1e-7, atol=1e-9 * np.abs(K).max())
        np.testing.assert_array_equal(P[1::2], 0.0)
        np.testing.assert_array_equal(K[:, 0::2], 0.0)
        assert P.sum() > 0.0  # the drying surface loses heat


def test_latent_heat_film_needs_two_values_and_a_relative_humidity():
    element = _thermoHygro()
    with pytest.raises(ValueError):
        element.computeDistributedLoad(
            "latent heat film", np.zeros(8), np.zeros(64), 1, np.array([1.0]), np.zeros(8), 0.0, 1.0
        )
    heatOnly = _element("DC2D4T", "HEATCONDUCTION", _heat)
    with pytest.raises(Exception):
        heatOnly.computeDistributedLoad(
            "latent heat film", np.zeros(4), np.zeros(16), 1, np.array([1.0, 0.5]), np.full(4, 20.0), 0.0, 1.0
        )


def test_coupled_element_tangent_matches_central_differences():
    """All couplings on: K = dP/dU, all four blocks."""
    element = _thermoHygro()
    UOld = _fields([30.0, 32.0, 28.0, 35.0], [0.95, 0.9, 0.85, 0.92])
    U = _fields([45.0, 60.0, 38.0, 52.0], [0.7, 0.85, 0.78, 0.6])

    def residual(U):
        P = np.zeros(8)
        K = np.zeros(64)
        element.computeKernels(K, P, U, U - UOld, 5.0, 0.5)
        return P, K.reshape((8, 8), order="F")

    P, K = residual(U)
    steps = np.tile([1e-4, 1e-7], 4)
    KFD = np.column_stack(
        [(residual(U + s * e)[0] - residual(U - s * e)[0]) / (2 * s) for s, e in zip(steps, np.eye(8))]
    )
    for a in range(2):
        for b in range(2):
            Kab, KFDab = K[a::2, b::2], KFD[a::2, b::2]
            assert np.abs(KFDab).max() > 0.0
            np.testing.assert_allclose(Kab, KFDab, rtol=1e-5, atol=1e-6 * np.abs(KFDab).max())


def test_uncoupled_limit_reproduces_the_single_field_elements():
    """THERMOHYGRO with 21 properties (no coupling) == HEATCONDUCTION in DC2D4T + MOISTUREDIFFUSION in DC2D4H."""
    th = _thermoHygro(_heat + _moisture)
    t = _element("DC2D4T", "HEATCONDUCTION", _heat)
    hy = _element("DC2D4H", "MOISTUREDIFFUSION", _moisture)
    T, TOld = np.array([45.0, 60.0, 38.0, 52.0]), np.array([30.0, 32.0, 28.0, 35.0])
    h, hOld = np.array([0.7, 0.85, 0.78, 0.6]), np.array([0.95, 0.9, 0.85, 0.92])
    for dT in (0.5, 0.0):
        P, K = np.zeros(8), np.zeros(64)
        th.computeKernels(K, P, _fields(T, h), _fields(T - TOld, h - hOld), 5.0, dT)
        PT, KT = np.zeros(4), np.zeros(16)
        t.computeKernels(KT, PT, T, T - TOld, 5.0, dT)
        PH, KH = np.zeros(4), np.zeros(16)
        hy.computeKernels(KH, PH, h, h - hOld, 5.0, dT)
        K = K.reshape((8, 8), order="F")
        np.testing.assert_allclose(P[0::2], PT, rtol=1e-14, atol=1e-14 * np.abs(PT).max())
        np.testing.assert_allclose(P[1::2], PH, rtol=1e-14, atol=1e-14 * np.abs(PH).max())
        np.testing.assert_allclose(
            K[0::2, 0::2], KT.reshape((4, 4), order="F"), rtol=1e-14, atol=1e-14 * np.abs(KT).max()
        )
        np.testing.assert_allclose(
            K[1::2, 1::2], KH.reshape((4, 4), order="F"), rtol=1e-14, atol=1e-14 * np.abs(KH).max()
        )
        np.testing.assert_array_equal(K[0::2, 1::2], 0.0)
        np.testing.assert_array_equal(K[1::2, 0::2], 0.0)


def test_zero_time_increment_lets_nothing_flow_and_no_latent_heat_leave():
    element = _thermoHygro()
    U = _fields([40.0, 45.0, 50.0, 42.0], [0.8, 0.85, 0.7, 0.9])
    P, K = np.zeros(8), np.zeros(64)
    element.computeKernels(K, P, U, np.zeros(8), 1.0, 0.0)
    np.testing.assert_array_equal(P, 0.0)
    PLoad = np.zeros(8)
    element.computeDistributedLoad("latent heat film", PLoad, np.zeros(64), 1, np.array([25.0, 0.5]), U, 1.0, 0.0)
    np.testing.assert_array_equal(PLoad, 0.0)
