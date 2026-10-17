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

import numpy as np
import pytest
import scipy.sparse as sp
from _blockamgsystems import displacementDamageSystem
from scipy.sparse.linalg import LinearOperator, gmres

pytest.importorskip("edelweissfe.linsolve.amgcl.amgcl")

from edelweissfe.linsolve.amgcl.amgcl import PyAMGCLSolver  # noqa: E402
from edelweissfe.linsolve.blockamg import createSolver  # noqa: E402

#: One AMGCL hierarchy per field: smoothed aggregation, Chebyshev smoothing.
_FIELD_AMG = {"coarsening": {"type": "smoothed_aggregation"}, "relax": {"type": "chebyshev"}}


def test_the_algorithm_step_by_step():
    # A coupled system: a displacement field (3 DOFs per node) and a nonlocal damage field (1 DOF per node),
    # its rows scaled over two orders of magnitude.
    A, b, fieldBlocks = displacementDamageSystem(10, coupling="one-way", rowScaling=True)
    fields = [slice(block.start, block.stop) for block in fieldBlocks]

    # 1. Equilibrate: solve (D A D) z = D b with D = diag(1 / sqrt|a_ii|), then x = D z.
    D = 1.0 / np.sqrt(np.abs(A.diagonal()))
    A_scaled = (sp.diags(D) @ A @ sp.diags(D)).tocsr()
    b_scaled = D * b

    # 2. Split by field, and build one AMG hierarchy per diagonal block A_ii.
    fieldAMG = []
    for field in fields:
        amg = PyAMGCLSolver({"precond": _FIELD_AMG})
        amg.build(A_scaled[field, field])
        fieldAMG.append(amg)

    # 3. The preconditioner: one block Gauss-Seidel sweep over the fields, each field correcting its estimate by
    #    one AMG cycle on its own residual, x_i = M_i^-1 (r_i - sum_{j < i} A_ij x_j).
    def blockGaussSeidel(r):
        x = np.zeros_like(r)
        for i, field in enumerate(fields):
            residual = r[field] - sum(A_scaled[field, earlier] @ x[earlier] for earlier in fields[:i])
            x[field] = fieldAMG[i].applyPreconditioner(residual)
        return x

    # 4. The outer Krylov solve on the coupled system, preconditioned by the sweep.
    M = LinearOperator(A.shape, matvec=blockGaussSeidel)
    z, info = gmres(A_scaled, b_scaled, M=M, rtol=1e-10, atol=0.0, restart=200, maxiter=1)
    x = D * z
    assert info == 0 and np.linalg.norm(b - A @ x) <= 1e-8 * np.linalg.norm(b)

    # The same solve with BlockAMGSolver, which adds a threaded C++ implementation of every step, the
    # rigid-body near null-space of the displacement field, an adaptive (Eisenstat--Walker) tolerance and the
    # reuse of the hierarchies across Newton iterations.
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
