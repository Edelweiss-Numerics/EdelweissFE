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
"""blockamg, step by step: the field-split preconditioned Krylov solve written out with NumPy and SciPy on a small
displacement / damage system, and the same solve with :class:`~edelweissfe.linsolve.blockamg.blockamg.BlockAMGSolver`.
The theory page includes this test as its worked example."""

import re

import numpy as np
import pytest
import scipy.sparse as sp
from _blockamgsystems import displacementDamageSystem
from scipy.sparse.linalg import LinearOperator, gmres, splu, spsolve

pytest.importorskip("edelweissfe.linsolve.amgcl.amgcl")

from edelweissfe.linsolve.amgcl.amgcl import PyAMGCLSolver  # noqa: E402
from edelweissfe.linsolve.blockamg import createSolver  # noqa: E402

#: One AMGCL hierarchy per field: smoothed aggregation, Chebyshev smoothing.
_FIELD_AMG = {"coarsening": {"type": "smoothed_aggregation"}, "relax": {"type": "chebyshev"}}


def _levels(amg) -> int:
    """The number of levels of an AMGCL hierarchy, from its report."""
    return int(re.search(r"Number of levels:\s+(\d+)", amg.report()).group(1))


def test_the_algorithm_step_by_step():
    # A coupled system: a displacement field (3 DOFs per node) and a nonlocal damage field (1 DOF per node) on a
    # 15^3 node grid, coupled both ways, the displacement rows scaled over two orders of magnitude.
    A, b, fieldBlocks = displacementDamageSystem(15, coupling="symmetric", rowScaling=True)
    fields = [slice(block.start, block.stop) for block in fieldBlocks]

    # 1. Equilibrate: solve (D A D) z = D b with D = diag(scaling), scaling = 1 / sqrt|a_ii|, then x = D z.
    scaling = 1.0 / np.sqrt(np.abs(A.diagonal()))
    D = sp.diags(scaling)
    A_scaled = (D @ A @ D).tocsr()
    b_scaled = scaling * b

    # 2. Split by field, and build one AMG hierarchy per diagonal block A_ii.
    fieldAMG = []
    for field in fields:
        amg = PyAMGCLSolver({"precond": _FIELD_AMG})
        amg.build(A_scaled[field, field])
        assert _levels(amg) > 1  # a real hierarchy, not a direct solve of a small block
        fieldAMG.append(amg)

    # 3. The preconditioner: one forward block Gauss-Seidel sweep over the fields, each field correcting its
    #    estimate by one AMG cycle on its own residual, x_i = M_i^-1 (r_i - sum_{j < i} A_ij x_j).
    def blockGaussSeidel(r, fieldSolves=fieldAMG):
        x = np.zeros_like(r)
        for i, field in enumerate(fields):
            residual = r[field] - sum(A_scaled[field, earlier] @ x[earlier] for earlier in fields[:i])
            x[field] = fieldSolves[i].applyPreconditioner(residual)
        return x

    #    With exact field solves, the sweep is the block lower triangular solve M^-1 = (B + L)^-1, B the
    #    diagonal blocks and L the blocks below them: what the AMG cycles approximate.
    class ExactSolve:
        def __init__(self, block):
            self.factorization = splu(block.tocsc())

        def applyPreconditioner(self, residual):
            return self.factorization.solve(residual)

    exact = [ExactSolve(A_scaled[field, field]) for field in fields]
    B_plus_L = sp.bmat(
        [
            [A_scaled[row, column] if j <= i else None for j, column in enumerate(fields)]
            for i, row in enumerate(fields)
        ],
        format="csc",
    )
    r = np.random.default_rng(1).standard_normal(A.shape[0])
    assert np.allclose(blockGaussSeidel(r, exact), spsolve(B_plus_L, r), rtol=1e-8, atol=1e-12)

    # 4. The outer Krylov solve on the coupled system, preconditioned by the sweep.
    M = LinearOperator(A.shape, matvec=blockGaussSeidel)
    z, info = gmres(A_scaled, b_scaled, M=M, rtol=1e-10, atol=0.0, restart=200, maxiter=1)
    x = scaling * z
    assert info == 0 and np.linalg.norm(b - A @ x) <= 1e-8 * np.linalg.norm(b)

    # The same solve with BlockAMGSolver: the same steps in threaded C++, plus the reuse of the hierarchies across
    # Newton iterations, an adaptive (Eisenstat--Walker) tolerance when outerTol is unset, and -- given the model
    # (setModel), not only the field layout as here -- the rigid-body near null-space of the displacement field.
    solver = createSolver(
        {
            "outerTol": 1e-10,
            "sweeps": 1,
            "symmetric": False,
            "fieldPreconds": {"displacement": _FIELD_AMG, "nonlocal damage": _FIELD_AMG},
            "verbosity": "silent",
        }
    )
    solver.setFieldStructure(fieldBlocks)
    assert np.linalg.norm(solver(A, b) - x) <= 1e-6 * np.linalg.norm(x)
