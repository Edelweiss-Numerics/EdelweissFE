#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""The blockamg linear solver with ``krylovOnScaledSystem=False``: the outer solve runs on the unscaled system A, preconditioned
by x = D M^-1 (D r), so the residual it converges is the true one the solve is accepted on -- no gap, and no
continuations: a miss is a cap hit.
"""

import numpy as np
from _linsolvesystems import displacementDamageSystem

from edelweissfe.config.linsolve import getLinSolverByName


def test_both_modes_solve_the_same_system():
    A, b, blocks = displacementDamageSystem(12, coupling="one-way", rowScaling=True)
    solutions = []
    for scaled in (True, False):
        solver = getLinSolverByName(
            "blockamg", {"verbosity": "silent", "tolerance": 1e-10, "krylovOnScaledSystem": scaled}
        )
        solver.setFieldStructure(blocks)
        solutions.append(solver(A, b))
    assert np.linalg.norm(solutions[0] - solutions[1]) < 1e-7 * np.linalg.norm(solutions[1])
