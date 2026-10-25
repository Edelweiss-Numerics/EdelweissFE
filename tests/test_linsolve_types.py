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
"""Every type of linear solver is a composed solver behind the one adapter: the direct solvers as trees of one leaf,
with the adapter's dumps."""

import json

import numpy as np
import pytest
import scipy.sparse as sp

from edelweissfe.config.linsolve import getDefaultLinSolver, getLinSolverByName
from edelweissfe.linsolve.composition import (
    ComposedLinearSolver,
    DirectSolver,
    SolveDumps,
)


def _system(n: int = 50, seed: int = 0):
    rng = np.random.default_rng(seed)
    A = (sp.diags([-1.0, 2.5, -1.0], [-1, 0, 1], shape=(n, n)) + sp.random(n, n, 0.05, random_state=seed)).tocsr()
    A.sort_indices()
    return A, rng.standard_normal(n)


def test_a_direct_solver_is_one_leaf_behind_the_adapter():
    A, b = _system()
    solver = getLinSolverByName("superlu", {})
    assert isinstance(solver, ComposedLinearSolver) and isinstance(solver.root, DirectSolver)
    assert not solver.needsStableSparsityPattern and not solver.reportsSolveSummary
    x = solver(A, b)
    assert np.linalg.norm(A @ x - b) < 1e-12 * np.linalg.norm(b)
    assert solver.lastSolveSummary is None


def test_the_default_linear_solver_solves():
    A, b = _system()
    x = getDefaultLinSolver()(A, b)
    assert np.linalg.norm(A @ x - b) < 1e-12 * np.linalg.norm(b)


def test_pardiso_keeps_its_backend_and_symbolic_factorization_across_solves():
    pytest.importorskip("edelweissfe.linsolve.pardiso.pardiso")
    A, b = _system()
    solver = getLinSolverByName("pardiso", {"reuseSymbolicFactorization": "True"})
    backend = solver.root.solver
    for scale in (1.0, 2.0):
        x = solver((scale * A).tocsr(), b)
        assert np.linalg.norm(scale * (A @ x) - b) < 1e-12 * np.linalg.norm(b)
        assert solver.root.solver is backend
    assert backend._solver is not None


def test_a_singular_matrix_gives_nan_for_a_cutback():
    solver = getLinSolverByName("superlu", {})
    x = solver(sp.csr_matrix(np.array([[1.0, 1.0], [1.0, 1.0]])), np.ones(2))
    assert np.isnan(x).all()


def test_an_unknown_option_is_refused():
    with pytest.raises(ValueError, match="reuseSymbolic"):
        getLinSolverByName("superlu", {"reuseSymbolicFactorization": "True"})


@pytest.mark.parametrize("dumps, dumped", [("every", [1, 2, 3]), ("2 3", [2, 3])])
def test_the_dumps_of_a_direct_solver(tmp_path, monkeypatch, dumps, dumped):
    monkeypatch.setattr(SolveDumps, "dumpsWritten", 0)
    A, b = _system()
    solver = getLinSolverByName("superlu", {"dumps": dumps, "dumpDirectory": str(tmp_path), "verbosity": "silent"})
    for k in range(3):
        solver(A, (k + 1) * b)
    with open(tmp_path / "manifest.jsonl") as manifestFile:
        records = [json.loads(line) for line in manifestFile]
    assert [record["solveCount"] for record in records] == dumped
    for record in records:
        assert np.array_equal(np.load(tmp_path / record["rhsFile"]), record["solveCount"] * b)
        assert abs(sp.load_npz(tmp_path / record["matrixFile"]) - A).max() == 0.0


def test_a_wrong_dump_trigger_is_refused():
    with pytest.raises(ValueError, match="dumps="):
        getLinSolverByName("superlu", {"dumps": "sometimes"})


def test_several_right_hand_sides_are_solved_column_by_column():
    """As an arc-length method asks: the columns of a matrix, each with the same set-up."""
    A, b = _system()
    B = np.column_stack([b, 2.0 * b[::-1]])
    X = getLinSolverByName("superlu", {})(A, B)
    assert X.shape == B.shape
    assert np.linalg.norm(A @ X - B) < 1e-12 * np.linalg.norm(B)
