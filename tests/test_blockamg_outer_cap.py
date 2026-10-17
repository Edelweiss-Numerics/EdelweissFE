#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""BlockAMGSolver's outer-solve non-convergence report: a cap hit, not a residual comparison.

The AMGCL LGMRES outer solve reports its own *scaled*, right-preconditioned relative residual; comparing that
with the true-residual target eta flagged converged solves as failures and missed real cap hits. The outer
solve's own non-convergence is its iteration count at the cap with its own convergence test failed -- a pass
may converge on exactly its last allowed iteration.
"""

from types import SimpleNamespace

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


def _capHit(iterations, lastError, tolerance=1e-4):
    solver = createSolver({"verbosity": "silent", "outerRestart": 50, "outerMaxiter": 1})
    solver._lgmresSolver = SimpleNamespace(lastError=lastError)
    return solver._lgmresCapHit(iterations, tolerance)


def test_a_pass_converged_on_its_last_allowed_iteration_is_no_cap_hit():
    assert _capHit(50, lastError=9e-5) == 0


def test_a_pass_at_the_cap_short_of_its_tolerance_is_a_cap_hit():
    assert _capHit(50, lastError=2e-4) == 1
    assert _capHit(50, lastError=float("nan")) == 1


def test_a_pass_below_the_cap_is_no_cap_hit_whatever_its_residual():
    # LGMRES only stops early on its own convergence test; lastError is never compared with eta
    assert _capHit(49, lastError=2e-4) == 0


def _stale(solver, counts):
    """Feed outer iteration counts to the staleness gate -- the first one as the rebuild solve -- and return
    the decisions."""
    decisions = []
    for k, count in enumerate(counts):
        decisions.append(solver._hierarchyIsStale(count, rebuilt=k == 0, capHit=False))
    return decisions


def test_a_slow_drift_is_caught_against_the_rebuild_solve():
    solver = createSolver({"verbosity": "silent", "hierarchyStalenessFactor": 1.5})
    # every count is below 1.5 times its predecessor; 37 > max(1.5 * 12, 12 + 15)
    assert _stale(solver, [12, 17, 25, 37]) == [False, False, False, True]


def test_small_counts_need_an_absolute_excess():
    solver = createSolver({"verbosity": "silent", "hierarchyStalenessFactor": 1.5})
    assert _stale(solver, [4, 8, 12, 19, 20]) == [False, False, False, False, True]


def test_a_cap_hit_rebuilds_the_hierarchies_for_the_next_solve():
    A, b, blocks = _system()
    solver = createSolver(
        {
            "verbosity": "silent",
            "outerTol": 1e-12,
            "outerRestart": 2,
            "outerMaxiter": 1,
            "trueResidualMaxContinuations": 0,
        }
    )
    solver.setFieldStructure(blocks)
    solver(A, b)
    standing = solver._preconditioners
    solver(A, b)
    assert solver._preconditioners is not standing
