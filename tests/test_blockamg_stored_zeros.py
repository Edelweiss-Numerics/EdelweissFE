#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""BlockAMGSolver on a system constrained by the solvers' own Dirichlet elimination, which keeps the eliminated
entries as explicitly stored zeros: the solve must be the one of the same system without them -- bitwise on one
thread; with several, AMGCL's own power iterations sum in thread order, so even two identical solves differ in
the last bits -- while the hierarchy-reuse pattern gate still sees the input's own nonzero count.
"""

import numpy as np
import scipy.sparse as sp

from edelweissfe.linsolve.base import FieldBlock
from edelweissfe.linsolve.blockamg import createSolver
from edelweissfe.solvers.base.dirichlet import applyDirichletToStiffness


class _DirichletStub:
    """What ``applyDirichletToStiffness`` reads from a boundary condition."""

    def __init__(self, dofs):
        self.constrainedDofIndices = np.asarray(dofs, dtype=np.int64)


def _system(n=14):
    lap1 = sp.diags([-1.0, 2.0, -1.0], [-1, 0, 1], shape=(n, n))
    eye = sp.identity(n)
    lap3 = sp.kron(sp.kron(lap1, eye), eye) + sp.kron(sp.kron(eye, lap1), eye) + sp.kron(sp.kron(eye, eye), lap1)
    nodes = n**3
    Kuu = sp.kron(lap3 + 0.01 * sp.identity(nodes), sp.identity(3))
    Kdd = lap3 + sp.identity(nodes)
    Kud = sp.random(3 * nodes, nodes, density=1e-3, random_state=1) * 1e-2
    K = sp.bmat([[Kuu, Kud], [Kud.T, Kdd]], format="csr")
    K.sort_indices()
    blocks = [FieldBlock("displacement", 0, 3 * nodes, 3), FieldBlock("nonlocal damage", 3 * nodes, 4 * nodes, 1)]
    b = np.random.default_rng(0).standard_normal(K.shape[0])
    dofs = np.arange(0, 3 * n * n)  # the first face of the displacement field
    b[dofs] = 0.0
    K = applyDirichletToStiffness(K, [_DirichletStub(dofs)], b)
    return K, b, blocks


def _solve(A, b, blocks):
    solver = createSolver({"outerTol": 1e-8, "verbosity": "silent"})
    solver.setFieldStructure(blocks)
    return solver(A, b), solver


def test_stored_zeros_change_nothing_but_the_pattern_gate_input():
    withZeros, b, blocks = _system()
    assert np.any(withZeros.data == 0.0), "the Dirichlet elimination must leave stored zeros for this test"
    withoutZeros = withZeros.copy()
    withoutZeros.eliminate_zeros()

    x, solver = _solve(withZeros, b, blocks)
    y, reference = _solve(withoutZeros, b, blocks)
    assert np.linalg.norm(x - y) <= 1e-12 * np.linalg.norm(y)
    assert solver._lastOuterIters == reference._lastOuterIters
    assert solver._lastNnz == withZeros.nnz
