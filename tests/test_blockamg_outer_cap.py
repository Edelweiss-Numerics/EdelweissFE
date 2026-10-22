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
from _blockamgsystems import displacementDamageSystem

from edelweissfe.linsolve.blockamg import createSolver
from edelweissfe.linsolve.blockamg.blockamg import BlockAMGSolver


def _system(n=12):
    return displacementDamageSystem(n)


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


def test_a_cap_hit_is_dumped_whatever_the_threshold(tmp_path, monkeypatch):
    # the dump budget is process-wide: start from none spent, whatever ran before
    monkeypatch.setattr(BlockAMGSolver, "_degradationDumpsWritten", 0)
    A, b, blocks = _system()
    solver = createSolver(
        {
            "verbosity": "silent",
            "outerTol": 1e-12,
            "outerRestart": 2,
            "outerMaxiter": 1,
            "trueResidualMaxContinuations": 0,
            "dumpOnDegradationDir": str(tmp_path),
            "dumpOnDegradationMaxDumps": 1,
        }
    )
    solver.setFieldStructure(blocks)
    solver(A, b)
    # two outer iterations, far below the dump threshold (the warning threshold, 100), but at the cap
    assert (tmp_path / "manifest.jsonl").exists()


def test_a_cap_hit_the_solve_recovered_from_rebuilds_but_is_not_reported(tmp_path, monkeypatch):
    monkeypatch.setattr(BlockAMGSolver, "_degradationDumpsWritten", 0)
    A, b, blocks = _system()
    solver = createSolver(
        {
            "verbosity": "warning",
            "outerTol": 1e-6,
            "outerRestart": 200,
            "outerMaxiter": 1,
            "dumpOnDegradationDir": str(tmp_path),
        }
    )
    solver.setFieldStructure(blocks)
    # every pass counts as a cap hit, and the solve still ends with eta met (with continuations if needed)
    monkeypatch.setattr(solver, "_lgmresCapHit", lambda iterations, tolerance: 1)
    messages = []
    solver._log = lambda level, message: messages.append(message)
    x = solver(A, b)
    assert np.linalg.norm(b - A @ x) <= 1e-6 * np.linalg.norm(b)
    assert not [m for m in messages if "⚠" in m]
    assert not (tmp_path / "manifest.jsonl").exists()

    standing = solver._preconditioners
    solver(A, b)
    assert solver._preconditioners is not standing
