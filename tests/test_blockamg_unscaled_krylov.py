#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""BlockAMGSolver with ``krylovOnScaledSystem=False``: the outer solve runs on the unscaled system A, preconditioned
by x = D M^-1 (D r), so the residual it converges is the true one the solve is accepted on -- no gap, and no
continuations: a miss is a cap hit.
"""

import numpy as np
import pytest
from _blockamgsystems import displacementDamageSystem

from edelweissfe.linsolve.blockamg import createSolver


def test_the_unscaled_outer_solve_converges_the_true_residual():
    A, b, blocks = displacementDamageSystem(12, coupling="one-way", rowScaling=True)
    solver = createSolver(
        {
            "verbosity": "silent",
            "outerTol": 1e-8,
            "outerRestart": 200,
            "krylovOnScaledSystem": False,
        }
    )
    solver.setFieldStructure(blocks)
    x = solver(A, b)
    trueResidual = np.linalg.norm(b - A @ x) / np.linalg.norm(b)
    assert trueResidual <= 1.0001e-8
    # right-preconditioned: its own stopping test is on the true residual, nothing to continue
    assert solver._lastContinuations == 0
    assert solver._lgmresSolver.lastError == pytest.approx(trueResidual, rel=1e-6)


def test_a_miss_on_the_unscaled_system_is_a_cap_hit_and_not_continued():
    A, b, blocks = displacementDamageSystem(12, coupling="one-way", rowScaling=True)
    solver = createSolver(
        {
            "verbosity": "silent",
            "outerTol": 1e-8,
            "outerRestart": 2,
            "outerMaxiter": 1,
            "krylovOnScaledSystem": False,
            "trueResidualMaxContinuations": 5,
        }
    )
    solver.setFieldStructure(blocks)
    x = solver(A, b)
    assert np.linalg.norm(b - A @ x) > 1e-8 * np.linalg.norm(b)
    # one pass of two iterations: a continuation would only have extended the cap
    assert solver._lastOuterIters == 2 and solver._lastContinuations == 0


def test_both_modes_solve_the_same_system():
    A, b, blocks = displacementDamageSystem(12, coupling="one-way", rowScaling=True)
    solutions = []
    for scaled in (True, False):
        solver = createSolver({"verbosity": "silent", "outerTol": 1e-10, "krylovOnScaledSystem": scaled})
        solver.setFieldStructure(blocks)
        solutions.append(solver(A, b))
    assert np.linalg.norm(solutions[0] - solutions[1]) < 1e-7 * np.linalg.norm(solutions[1])
