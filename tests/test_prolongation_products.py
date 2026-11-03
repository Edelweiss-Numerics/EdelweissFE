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
#  This library is free software; you can redistribute it and/or
#  modify it under the terms of the GNU Lesser General Public
#  License as published by the Free Software Foundation; either
#  version 2.1 of the License, or (at your option) any later version.
#
#  The full text of the license can be found in the file LICENSE.md at
#  the top level directory of EdelweissFE.
#  ---------------------------------------------------------------------
"""Tests for the threaded products of :class:`~edelweissfe.linsolve.composition.combinators.CoarseCorrection`
(``Prolongation`` in ``amgcl-wrapper.hpp``, :class:`~edelweissfe.linsolve.amgcl.amgcl.PyAMGCLProlongation`): the
restriction, the prolongation and the Galerkin coarse operator are SciPy's, up to the order of the sums.
"""

import numpy as np
import pytest
import scipy.sparse as sp

pytest.importorskip("edelweissfe.linsolve.amgcl.amgcl")

from edelweissfe.linsolve.amgcl.amgcl import PyAMGCLProlongation  # noqa: E402
from edelweissfe.linsolve.base import FieldBlock  # noqa: E402
from edelweissfe.linsolve.composition import (  # noqa: E402
    Jacobi,
    SetUpContext,
    SystemMatrix,
)


@pytest.fixture(scope="module")
def problem():
    rng = np.random.default_rng(7)
    n, nCoarse = 300, 40
    A = (sp.random(n, n, density=0.04, random_state=rng) + sp.diags(rng.uniform(1.0, 5.0, n))).tocsr()
    P = sp.random(n, nCoarse, density=0.08, random_state=rng, format="csr")
    return A, P, PyAMGCLProlongation(P)


def test_the_restriction_and_the_prolongation_are_scipys(problem):
    A, P, products = problem
    rng = np.random.default_rng(8)
    r, z_c = rng.standard_normal(P.shape[0]), rng.standard_normal(P.shape[1])
    np.testing.assert_allclose(products.restrict(r), P.T @ r, rtol=1e-13, atol=1e-13)
    np.testing.assert_allclose(products.prolongate(z_c), P @ z_c, rtol=1e-13, atol=1e-13)


def test_the_galerkin_operator_is_scipys(problem):
    A, P, products = problem
    A_c = products.galerkinOperator(A)
    assert A_c.shape == (P.shape[1], P.shape[1])
    assert A_c.has_sorted_indices
    np.testing.assert_allclose(A_c.toarray(), (P.T @ A @ P).toarray(), rtol=1e-13, atol=1e-13)


def test_vectors_and_matrices_of_the_wrong_size_raise(problem):
    A, P, products = problem
    with pytest.raises(ValueError):
        products.restrict(np.ones(P.shape[0] + 1))
    with pytest.raises(ValueError):
        products.prolongate(np.ones(P.shape[1] - 1))
    with pytest.raises(RuntimeError):
        products.galerkinOperator(sp.identity(P.shape[0] - 1, format="csr"))


def test_jacobi_divides_by_the_diagonal_of_the_scaled_matrix(problem):
    A, P, products = problem
    D = np.random.default_rng(9).uniform(0.5, 2.0, A.shape[0])
    jacobi = Jacobi(omega=0.4)
    jacobi.setUp(SystemMatrix(A).scaled(D), SetUpContext(fields=(FieldBlock("u", 0, A.shape[0], 1),)))
    r = np.random.default_rng(10).standard_normal(A.shape[0])
    np.testing.assert_allclose(jacobi(r), 0.4 * r / (D * D * A.diagonal()), rtol=1e-15)
