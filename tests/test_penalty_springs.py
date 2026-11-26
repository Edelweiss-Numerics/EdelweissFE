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
    SpringGradientOnSharedNodes,
    addBoundOfAbsoluteStiffnessOfSprings,
)


def test_the_bound_is_the_sum_over_the_springs_of_the_absolute_gradients():
    """:math:`b_i = \\sum_p |k_p|\\, |a_{p,i}|\\, (|\\boldsymbol{a}_p| \\cdot \\boldsymbol{w})`, with the gradients
    :math:`\\boldsymbol{a}_p` on nodes the springs share, with a dense and a sparse matrix of coefficients, against the
    dense sum."""

    rng = np.random.default_rng(0)
    nDof, nSprings, nNodes, nDim = 15, 7, 5, 3
    stiffness = rng.normal(size=nSprings)
    weights = rng.uniform(0.5, 2.0, nDof)

    coefficients = rng.normal(size=(nSprings, nNodes)) * (rng.uniform(size=(nSprings, nNodes)) < 0.6)
    directions = rng.normal(size=(nSprings, nDim))
    dofsOfNodes = np.arange(nDof).reshape((nNodes, nDim))

    A = np.zeros((nSprings, nDof))
    for p in range(nSprings):
        A[p, dofsOfNodes] += np.outer(coefficients[p], directions[p])
    expected = (np.abs(stiffness)[:, None] * np.abs(A)).T @ (np.abs(A) @ weights)

    for interpolation in (coefficients, csr_matrix(coefficients)):
        bound = np.zeros(nDof)
        gradient = [SpringGradientOnSharedNodes(interpolation, directions, dofsOfNodes)]
        addBoundOfAbsoluteStiffnessOfSprings(bound, weights, stiffness, gradient)
        np.testing.assert_allclose(bound, expected, rtol=1e-13)
