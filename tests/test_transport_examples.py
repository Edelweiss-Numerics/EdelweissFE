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
"""The fast examples of the Marmot transport elements (``examples/HeatTransfer``, ``examples/MoistureDiffusion``,
``examples/ThermoHygro``) run as committed, with their documented results."""

import math
import pathlib
import shutil

import numpy as np
import pytest

from edelweissfe.drivers.inputfiledrivensimulation import finiteElementSimulation
from edelweissfe.utils.inputfileparser import parseInputFile

EXAMPLES = pathlib.Path(__file__).resolve().parent.parent / "examples"


def _requireMarmotElement(elType):
    elementModule = pytest.importorskip("edelweissfe.elements.marmotelement.element")
    try:
        elementModule.MarmotElementWrapper(elType, 1)
    except NotImplementedError:
        pytest.skip(f"this Marmot build has no {elType}")


def _runExample(tmp_path, monkeypatch, relativePath):
    """Run an example deck in a scratch directory (its Ensight output goes there)."""
    source = EXAMPLES / relativePath
    if not source.exists():
        pytest.skip(f"{relativePath} is not part of this checkout")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("OMP_NUM_THREADS", "1")
    shutil.copy(source, tmp_path / source.name)
    model, _ = finiteElementSimulation(parseInputFile(source.name), verbose=False, suppressPlots=True)
    return model


def _nodal(model, fieldName, result="U", nodeSet=None):
    """Coordinates and values of a nodal field (on a node set, or everywhere), ordered by node label."""
    field = model.nodeFields[fieldName]
    index = field._indicesOfNodesInArray
    nodes = sorted(model.nodeSets[nodeSet] if nodeSet else index.keys(), key=lambda n: n.label)
    X = np.array([n.coordinates for n in nodes])
    values = np.array([field[result][index[n]] for n in nodes]).reshape(len(nodes), -1)[:, 0]
    return X, values


def test_heated_pipe_reaches_the_closed_form_steady_state(tmp_path, monkeypatch):
    """examples/HeatTransfer/heated_pipe_axisymmetric.inp: after 10 days the wall temperature and the heat flow at the
    bore are those of steady conduction through a cylindrical wall with convection outside."""
    _requireMarmotElement("DCAX8T")
    model = _runExample(tmp_path, monkeypatch, "HeatTransfer/heated_pipe_axisymmetric.inp")

    ri, ro, k, h, Ti, Tinf, H = 0.5, 0.8, 1.6, 10.0, 80.0, 0.0, 0.1
    denominator = math.log(ro / ri) + k / (h * ro)
    X, T = _nodal(model, "temperature")
    exact = Ti - (Ti - Tinf) * np.log(X[:, 0] / ri) / denominator
    np.testing.assert_allclose(T, exact, rtol=0, atol=1e-4)

    _, P = _nodal(model, "temperature", result="P", nodeSet="gen_left")
    Q = 2 * math.pi * H * k * (Ti - Tinf) / denominator
    assert abs(abs(P.sum()) - Q) < 1e-4 * Q


def test_curing_slab_dries_at_the_top_and_self_desiccates_at_the_sealed_bottom(tmp_path, monkeypatch):
    """examples/MoistureDiffusion/curing_slab_self_desiccation.inp: after 1 year (28 days of evaporation through the
    top, then sealed) the humidity has dropped everywhere by self-desiccation, more at the top than at the bottom."""
    _requireMarmotElement("DC3D20H")
    model = _runExample(tmp_path, monkeypatch, "MoistureDiffusion/curing_slab_self_desiccation.inp")

    _, top = _nodal(model, "relative humidity", nodeSet="gen_front")
    _, bottom = _nodal(model, "relative humidity", nodeSet="gen_back")
    np.testing.assert_allclose(top, 0.9673, atol=2e-4)
    np.testing.assert_allclose(bottom, 0.9733, atol=2e-4)
