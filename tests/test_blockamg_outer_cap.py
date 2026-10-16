#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""BlockAMGSolver's outer-solve non-convergence report: a cap hit, not a residual comparison.

The AMGCL LGMRES outer solve reports its own *scaled*, right-preconditioned relative residual; comparing that
with the true-residual target eta flagged converged solves as failures and missed real cap hits. Its iteration
count reaching the cap is the outer solve's own non-convergence.
"""

import numpy as np
import scipy.sparse as sp

from edelweissfe.linsolve.base import FieldBlock
from edelweissfe.linsolve.blockamg import createSolver


def _system(n=12):
    lap1 = sp.diags([-1.0, 2.0, -1.0], [-1, 0, 1], shape=(n, n))
    eye = sp.identity(n)
    lap3 = sp.kron(sp.kron(lap1, eye), eye) + sp.kron(sp.kron(eye, lap1), eye) + sp.kron(sp.kron(eye, eye), lap1)
    nodes = n**3
    A = sp.bmat([[sp.kron(lap3 + 0.01 * sp.identity(nodes), sp.identity(3)), None], [None, lap3 + sp.identity(nodes)]])
    blocks = [FieldBlock("displacement", 0, 3 * nodes, 3), FieldBlock("nonlocal damage", 3 * nodes, 4 * nodes, 1)]
    b = np.random.default_rng(0).standard_normal(A.shape[0])
    return A.tocsr(), b, blocks


def _warnings(**options):
    A, b, blocks = _system()
    config = {"verbosity": "warning", "trueResidualMaxContinuations": 0}
    config.update(options)
    solver = createSolver(config)
    solver.setFieldStructure(blocks)
    messages = []
    solver._log = lambda level, message: messages.append(message)
    solver(A, b)
    return [m for m in messages if "iteration cap" in m]


def test_a_cap_hit_is_reported():
    assert _warnings(outerTol=1e-12, outerRestart=2, outerMaxiter=1)


def test_a_converged_solve_is_not_reported():
    assert not _warnings(outerTol=1e-6, outerRestart=200, outerMaxiter=1)
