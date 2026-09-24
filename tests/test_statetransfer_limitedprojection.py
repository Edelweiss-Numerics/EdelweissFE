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
"""Pins the bound- and mean-preservation of ``LimitedPolynomialProjection`` for an octree split of a
2x2x2 Gauss rule (the GC3D20R case), where the child quadrature points at +-0.789 lie outside the
parent Gauss points at +-0.577 and plain projection extrapolates past the parent's value range."""

import itertools

import numpy as np
import pytest

from edelweissfe.adaptivity.statetransfer import (
    LimitedPolynomialProjection,
    PolynomialProjection,
)
from edelweissfe.config.statetransferstrategies import getStateTransferStrategyClass

GAUSS = 1.0 / np.sqrt(3.0)
PARENT_QPS = np.array(list(itertools.product((-GAUSS, GAUSS), repeat=3)))


def childQuadraturePoints(octant):
    """Quadrature points of one octree child, in the parent reference cube."""
    return np.asarray(octant) * 0.5 + 0.5 * PARENT_QPS


OCTANTS = list(itertools.product((-1.0, 1.0), repeat=3))


def transferToAllChildren(strategy, parentValues):
    columns = np.arange(parentValues.shape[1])
    return [
        strategy._transferColumns(parentValues, PARENT_QPS, childQuadraturePoints(o), None, columns) for o in OCTANTS
    ]


@pytest.fixture
def steepField():
    """A localized, steep field (one hot parent point) plus a linear gradient, two columns."""
    hot = np.zeros(8)
    hot[-1] = 1.0
    linear = 2.0 + PARENT_QPS @ np.array([1.0, -0.5, 0.25])
    return np.column_stack([hot, linear])


def test_plainProjectionOvershoots(steepField):
    """Premise of the limiter: unlimited projection leaves the parent range at the outer child points."""
    children = np.vstack(transferToAllChildren(PolynomialProjection(), steepField))
    assert children[:, 1].max() > steepField[:, 1].max() + 1e-8


def test_boundsPreserved(steepField):
    children = np.vstack(transferToAllChildren(LimitedPolynomialProjection(), steepField))
    assert np.all(children >= steepField.min(axis=0) - 1e-14)
    assert np.all(children <= steepField.max(axis=0) + 1e-14)


def test_integralPreserved(steepField):
    """Equal weights on parent and children: the parent QP sum equals the sum over all child QPs / 8."""
    children = np.vstack(transferToAllChildren(LimitedPolynomialProjection(), steepField))
    np.testing.assert_allclose(children.sum(axis=0) / 8.0, steepField.sum(axis=0), rtol=1e-13)


def test_constantFieldExact():
    parent = np.full((8, 1), 0.73)
    for child in transferToAllChildren(LimitedPolynomialProjection(), parent):
        np.testing.assert_allclose(child, 0.73, rtol=1e-14)


def test_registered():
    assert getStateTransferStrategyClass("limitedProjection") is LimitedPolynomialProjection
