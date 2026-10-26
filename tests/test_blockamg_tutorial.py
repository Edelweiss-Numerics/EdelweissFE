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
displacement / damage system, and the same solve as a composed solver (the tree of the ``blockamg`` preset, with the
field trees of this example).
The theory page includes this file as its worked example, from the marker comment below on.
"""

# The worked example of the theory page.
import numpy as np
import scipy.sparse as sp
from _linsolvesystems import ExactFieldSolve, displacementDamageSystem, hierarchyLevels
from scipy.sparse.linalg import LinearOperator, gmres

from edelweissfe.linsolve.composition import (
    AMG,
    LGMRES,
    ComposedLinearSolver,
    DiagonalScaling,
    FixedTolerance,
    Multiplicative,
    OnField,
)

#: One AMGCL hierarchy per field: smoothed aggregation, Chebyshev smoothing.
FIELD_AMG = {"coarsening": {"type": "smoothed_aggregation"}, "relax": {"type": "chebyshev"}}


def test_the_algorithm_step_by_step():
    # A coupled system with the field structure of a gradient-enhanced damage model: a "displacement" field (3 DOFs
    # per node) and a nonlocal damage field (1 DOF per node) on a 15^3 node grid, coupled both ways, the displacement
    # rows scaled over two orders of magnitude. Not its physics: the displacement block is a vector Laplacian (three
    # copies of a grid Laplacian) standing in for elasticity, so it shows the method, not why elasticity needs the
    # rigid-body modes.
    A, b, fieldBlocks = displacementDamageSystem(15, coupling="symmetric", rowScaling=True)
    fields = [slice(block.start, block.stop) for block in fieldBlocks]
    # The steps are numbered as in "The solve pipeline" of the theory page. Steps (1) and (7) carry state from one
    # Newton iteration to the next and have no part in a single solve.

    # (2) Equilibrate: solve (D A D) z = D b with D = diag(scaling), scaling = 1 / sqrt|a_ii|, then x = D z.
    scaling = 1.0 / np.sqrt(np.abs(A.diagonal()))
    D = sp.diags(scaling)
    A_scaled = (D @ A @ D).tocsr()
    b_scaled = scaling * b

    # (3) Split by field into the blocks A_ij, and (4) build one AMG hierarchy per diagonal block A_ii, with AMGCL
    #     (an optional extension of EdelweissFE, hence imported here).
    A_blocks = [[A_scaled[row, column] for column in fields] for row in fields]

    from edelweissfe.linsolve.amgcl.amgcl import PyAMGCLSolver

    fieldAMG = []
    for i in range(len(fields)):
        amg = PyAMGCLSolver({"precond": FIELD_AMG})
        amg.build(A_blocks[i][i])
        assert hierarchyLevels(amg) > 1  # a real hierarchy, not a direct solve of a small block
        fieldAMG.append(amg)

    # (5a) The preconditioner v = M^-1 r: one forward block Gauss-Seidel sweep over the fields, each field's part of
    #      v from one AMG cycle on its own residual, v_i = M_i^-1 (r_i - sum_{j < i} A_ij v_j). Without the coupling
    #      sum it is block Jacobi.
    def blockGaussSeidel(r, fieldSolves=fieldAMG, coupled=True):
        v = np.zeros_like(r)
        for i, field in enumerate(fields):
            residual = r[field].copy()
            if coupled:
                for j in range(i):
                    residual -= A_blocks[i][j] @ v[fields[j]]
            v[field] = fieldSolves[i].applyPreconditioner(residual)
        return v

    #      With exact field solves, the sweep is the block lower triangular solve M^-1 = (B + L)^-1, B the
    #      diagonal blocks and L the blocks below them: what the AMG cycles approximate. Check: (B + L) v = r.
    exact = [ExactFieldSolve(A_blocks[i][i]) for i in range(len(fields))]
    B_plus_L = sp.bmat([[A_blocks[i][j] if j <= i else None for j in range(len(fields))] for i in range(len(fields))])
    r = np.random.default_rng(1).standard_normal(A.shape[0])
    assert np.linalg.norm(B_plus_L @ blockGaussSeidel(r, exact) - r) <= 1e-10 * np.linalg.norm(r)

    # (5b) The outer Krylov solve on the coupled system, right-preconditioned by the sweep: GMRES solves
    #      (A_scaled M^-1) y = b_scaled for y, and z = M^-1 y. GMRES converges the residual of the operator it is
    #      given, b_scaled - A_scaled M^-1 y = b_scaled - A_scaled z, the scaled residual D (b - A x).
    A_scaled_Minv = LinearOperator(A.shape, matvec=lambda y: A_scaled @ blockGaussSeidel(y))
    y, info = gmres(A_scaled_Minv, b_scaled, rtol=1e-10, atol=0.0, restart=200, maxiter=1)
    z = blockGaussSeidel(y)
    x = scaling * z
    # (6) The true residual b - A x = D^-1 (b_scaled - A_scaled z) is then at most max(D) / min(D) times larger,
    #     relative to b: here 100 (sqrt of the 1e4 spread of the diagonal), hence 1e-8 after a 1e-10 solve.
    assert info == 0 and np.linalg.norm(b - A @ x) <= 1e-8 * np.linalg.norm(b)

    # (5b) on the unscaled system, with the same preconditioner between two scalings (krylovOnScaledSystem = false,
    #      the default): GMRES solves (A D M^-1 D) y = b for y, and x = D M^-1 D y. The residual it
    #      converges, b - A D M^-1 D y = b - A x, is then the true one, and step (6) has nothing left to check.
    A_DMinvD = LinearOperator(A.shape, matvec=lambda y: A @ (scaling * blockGaussSeidel(scaling * y)))
    y, info = gmres(A_DMinvD, b, rtol=1e-10, atol=0.0, restart=200, maxiter=1)
    x_unscaled = scaling * blockGaussSeidel(scaling * y)
    assert info == 0 and np.linalg.norm(b - A @ x_unscaled) <= 1.0001e-10 * np.linalg.norm(b)
    assert np.linalg.norm(x_unscaled - x) <= 1e-6 * np.linalg.norm(x)

    # What the preconditioner buys: GMRES iterations to the same 1e-10, without a preconditioner, with the sweep
    # (AMG cycles), and with exact field solves, block Jacobi against block Gauss-Seidel. Measured: far fewer
    # iterations with the sweep than without a preconditioner at all; with AMG-cycle field solves, block Jacobi
    # needs about as many iterations as block Gauss-Seidel (the coupling is weak, and the AMG cycles' own error
    # dominates); with exact field solves, the coupling term of the sweep becomes visible and Gauss-Seidel needs
    # fewer iterations than Jacobi.
    def iterations(preconditioner):
        history = []
        A_scaled_Minv = LinearOperator(A.shape, matvec=lambda y: A_scaled @ preconditioner(y))
        gmres(
            A_scaled_Minv,
            b_scaled,
            rtol=1e-10,
            atol=0.0,
            restart=200,
            maxiter=1,
            callback=history.append,
            callback_type="pr_norm",
        )
        return len(history)

    unpreconditioned = iterations(lambda y: y)
    assert unpreconditioned > 5 * iterations(blockGaussSeidel)
    assert iterations(lambda y: blockGaussSeidel(y, exact)) < iterations(lambda y: blockGaussSeidel(y, exact, False))

    # The same solve as a composed solver: the same method, with the field split in threaded C++ and AMGCL's
    # right-preconditioned LGMRES as the outer solver (5b), on the unscaled system, LGMRES(DiagonalScaling(...)) as
    # the blockamg preset builds it, or on the scaled one, DiagonalScaling(LGMRES(...)). The preset adds the reuse of
    # the hierarchies across Newton iterations, an adaptive (Eisenstat--Walker) tolerance, and a near null-space for
    # the displacement field: its translations here, given only the field layout, and all rigid-body modes given the
    # model (setModel).
    def blockGaussSeidelTree():
        return Multiplicative([OnField(field.name, AMG(tree=dict(FIELD_AMG))) for field in fieldBlocks])

    for tree in (
        LGMRES(DiagonalScaling(blockGaussSeidelTree()), tolerance=FixedTolerance(1e-10)),
        DiagonalScaling(LGMRES(blockGaussSeidelTree(), tolerance=FixedTolerance(1e-10))),
    ):
        solver = ComposedLinearSolver(tree, verbosity="silent")
        solver.setFieldStructure(fieldBlocks)
        assert np.linalg.norm(solver(A, b) - x) <= 1e-6 * np.linalg.norm(x)
