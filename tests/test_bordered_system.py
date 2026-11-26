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
"""The bordered system of the arc length method, :meth:`~edelweissfe.linsolve.base.LinearSolver.solveBordered`: by
block elimination with a direct solver, and as one Krylov solve with the composed iterative solvers -- the same
solution, and the Krylov solve with an exact preconditioner of the matrix converges in two iterations.
"""

import numpy as np
import pytest
import scipy.sparse as sp
from _linsolvesystems import displacementDamageSystem

from edelweissfe.config.linsolve import getLinSolverByName
from edelweissfe.linsolve.base import Border
from edelweissfe.linsolve.composition import (
    LGMRES,
    ComposedLinearSolver,
    DirectSolver,
    FixedTolerance,
)


def borderedProblem(A: sp.csr_matrix, seed: int = 1):
    """A border of the matrix ``A`` as an arc length method has it -- a dense column (the reference load), a sparse row
    (a few controlled dofs) and a zero corner -- with a right-hand side, and the dense solution of the bordered system.
    """
    rng = np.random.default_rng(seed)
    n = A.shape[0]
    column = rng.standard_normal(n)
    row = np.zeros(n)
    row[rng.choice(n, size=5, replace=False)] = rng.standard_normal(5)
    border = Border(column, row, 0.0)
    b, beta = rng.standard_normal(n), 0.3

    bordered = np.zeros((n + 1, n + 1))
    bordered[:n, :n] = A.toarray()
    bordered[:n, n], bordered[n, :n] = column, row
    solution = np.linalg.solve(bordered, np.append(b, beta))
    return border, b, beta, solution


def relativeError(x: np.ndarray, xi: float, solution: np.ndarray) -> float:
    return np.linalg.norm(np.append(x, xi) - solution) / np.linalg.norm(solution)


def test_a_direct_solver_eliminates_the_border():
    A, _, blocks = displacementDamageSystem(6, coupling="one-way")
    border, b, beta, solution = borderedProblem(A)
    solver = getLinSolverByName("superlu", {})
    solver.setFieldStructure(blocks)

    x, xi = solver.solveBordered(A, border, b, beta)

    assert relativeError(x, xi, solution) < 1e-10


@pytest.mark.parametrize("krylovOnScaledSystem", [True, False])
def test_blockamg_solves_the_bordered_system_as_one_krylov_solve(krylovOnScaledSystem):
    A, _, blocks = displacementDamageSystem(10, coupling="one-way", rowScaling=True)
    border, b, beta, solution = borderedProblem(A)
    solver = getLinSolverByName(
        "blockamg", {"verbosity": "silent", "tolerance": 1e-10, "krylovOnScaledSystem": krylovOnScaledSystem}
    )
    solver.setFieldStructure(blocks)

    x, xi = solver.solveBordered(A, border, b, beta)

    assert relativeError(x, xi, solution) < 1e-7
    assert solver.lastSolveSummary.residualMet


def test_with_an_exact_preconditioner_two_iterations_suffice():
    """With M = A, the bordered matrix preconditioned by [M 0; c^T gamma] has a minimal polynomial of degree two."""
    A, _, blocks = displacementDamageSystem(6, coupling="one-way")
    border, b, beta, solution = borderedProblem(A)
    solver = ComposedLinearSolver(LGMRES(DirectSolver(), tolerance=FixedTolerance(1e-12)), verbosity="silent")
    solver.setFieldStructure(blocks)

    x, xi = solver.solveBordered(A, border, b, beta)

    assert relativeError(x, xi, solution) < 1e-10
    assert solver.lastSolveSummary.iters <= 2


def test_a_border_singular_for_the_preconditioner_is_declined():
    """A row of zeros and a zero corner: the bordered matrix is singular, the solve returns NaN (a cutback)."""
    A, _, blocks = displacementDamageSystem(6, coupling="one-way")
    n = A.shape[0]
    solver = ComposedLinearSolver(LGMRES(DirectSolver(), tolerance=FixedTolerance(1e-12)), verbosity="silent")
    solver.setFieldStructure(blocks)

    x, xi = solver.solveBordered(A, Border(np.ones(n), np.zeros(n), 0.0), np.ones(n), 0.0)

    assert np.isnan(xi) and np.isnan(x).all()
