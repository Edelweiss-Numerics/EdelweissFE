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
"""The bound of the stiffness of penalty springs (:mod:`edelweissfe.constraints.base.penaltysprings`)."""

import numpy as np
from scipy.sparse import csr_matrix

from edelweissfe.constraints.base.penaltysprings import (
    SpringGradientOnOwnNodes,
    SpringGradientOnSharedNodes,
    addBoundOfAbsoluteStiffnessOfSprings,
)


def test_the_bound_is_the_sum_over_the_springs_of_the_absolute_gradients():
    """:math:`b = \\sum_p |k_p|\\, |w_p|\\, (|w_p| \\cdot \\boldsymbol{w})`, with the gradients given in parts on shared
    nodes (a sparse and a dense interpolation) and on nodes of each spring's own (some named by several springs),
    against the dense sum."""

    rng = np.random.default_rng(0)
    nDof, nSprings, nDim = 30, 7, 3
    stiffness = rng.normal(size=nSprings)
    weights = rng.uniform(0.5, 2.0, nDof)

    shared = rng.normal(size=(nSprings, 4)) * (rng.uniform(size=(nSprings, 4)) < 0.6)
    sharedDirections = rng.normal(size=(nSprings, nDim))
    dofsOfSharedNodes = np.arange(12).reshape((4, nDim))
    own = rng.normal(size=(nSprings, 2))
    ownDirections = rng.normal(size=(nSprings, nDim))
    # two nodes of each spring's own, distinct within a spring, some shared between springs
    ownNodes = np.array([rng.choice(6, size=2, replace=False) for _ in range(nSprings)])
    ownDofs = 12 + ownNodes[:, :, None] * nDim + np.arange(nDim)

    W = np.zeros((nSprings, nDof))
    for p in range(nSprings):
        W[p, dofsOfSharedNodes] += np.outer(shared[p], sharedDirections[p])
        np.add.at(W[p], ownDofs[p], np.outer(own[p], ownDirections[p]))
    expected = (np.abs(stiffness)[:, None] * np.abs(W)).T @ (np.abs(W) @ weights)

    for interpolation in (shared, csr_matrix(shared)):
        bound = np.zeros(nDof)
        gradient = [
            SpringGradientOnSharedNodes(interpolation, sharedDirections, dofsOfSharedNodes),
            SpringGradientOnOwnNodes(own, ownDirections, ownDofs),
        ]
        addBoundOfAbsoluteStiffnessOfSprings(bound, weights, stiffness, gradient)
        np.testing.assert_allclose(bound, expected, rtol=1e-13)
