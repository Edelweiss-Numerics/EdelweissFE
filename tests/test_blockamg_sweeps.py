#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""BlockAMGSolver's block Gauss-Seidel sweep in residual-correction form.

A revisit of a field -- a second sweep, or the reverse sweep of ``symmetric`` -- must improve its part of
``v = M^-1 r``, ``v_i += M_i^-1 (r_i - sum_j A_ij v_j)``. The former overwrite form
``v_i = M_i^-1 (r_i - sum_{j != i} A_ij v_j)`` threw the previous estimate away, which is only the same for an exact M_i; with an AMG cycle a second sweep
then bought nothing.
"""

import numpy as np
import scipy.sparse as sp
from _blockamgsystems import ExactFieldSolve, displacementDamageSystem, gridLaplacian

from edelweissfe.linsolve.base import FieldBlock
from edelweissfe.linsolve.blockamg import createSolver


def _system(n=18, nFields=3):
    """Three scalar fields on an n^3 grid Laplacian each, in a chain coupled both ways -- or the first one or
    two of them. n^3 above AMGCL's coarse_enough (3000), so that no field is solved directly."""
    nodes = n**3
    K = (gridLaplacian(n) + 0.05 * sp.identity(nodes)).tocsr()
    coupling = 0.8 * sp.identity(nodes)
    A = sp.bmat([[K, coupling, None], [coupling, 2.0 * K, coupling], [None, coupling, K]], format="csr")
    blocks = [
        FieldBlock("first", 0, nodes, 1),
        FieldBlock("second", nodes, 2 * nodes, 1),
        FieldBlock("third", 2 * nodes, 3 * nodes, 1),
    ]
    if nFields == 1:
        A, blocks = K, blocks[:1]
    elif nFields == 2:
        A, blocks = A[: 2 * nodes, : 2 * nodes].tocsr(), blocks[:2]
    b = np.random.default_rng(0).standard_normal(A.shape[0])
    return A, b, blocks


#: A deliberately weak per-field cycle (plain aggregation, one damped-Jacobi step each way), so that one
#: application of M_i is far from the exact block solve.
_WEAK = {"coarsening": {"type": "aggregation"}, "relax": {"type": "damped_jacobi"}, "npre": 1, "npost": 1}


def _outerIterations(nFields=3, **options):
    A, b, blocks = _system(nFields=nFields)
    b = b[: A.shape[0]]
    config = {
        "outerTol": 1e-8,
        "verbosity": "silent",
        "trueResidualMaxContinuations": 0,
        "outerRestart": 200,
        "fieldPreconds": {name: dict(_WEAK) for name in ("first", "second", "third")},
    }
    config.update(options)
    solver = createSolver(config)
    solver.setFieldStructure(blocks)
    x = solver(A, b)
    assert np.linalg.norm(b - A @ x) < 1e-6 * np.linalg.norm(b)
    return solver._lastOuterIters


def test_a_revisit_improves_a_single_field_estimate():
    # with one field the overwrite form recomputed the very same x = M r on every sweep (16, 16, 16
    # iterations); in correction form each sweep is one more preconditioned Richardson step (16, 9, 8)
    once = _outerIterations(nFields=1, sweeps=1, symmetric=False)
    twice = _outerIterations(nFields=1, sweeps=2, symmetric=False)
    assert twice < 0.75 * once
    assert _outerIterations(nFields=1, sweeps=3, symmetric=False) <= twice


def test_further_and_reverse_sweeps_reduce_the_outer_iterations_of_a_coupled_system():
    # measured (sweeps 1, sweeps 2, symmetric): overwrite form 1269, 1482, 1263; correction form 1269, 898, 941
    once = _outerIterations(sweeps=1, symmetric=False)
    assert _outerIterations(sweeps=2, symmetric=False) < 0.8 * once
    assert _outerIterations(sweeps=1, symmetric=True) < 0.8 * once


def test_the_reverse_sweep_of_the_default_symmetric_sweep_improves_two_fields():
    # with two fields the reverse sweep revisits only the first one: the overwrite form recomputed it from the
    # second field's estimate alone, the correction form improves it -- the code default symmetric = true is
    # therefore the one setting whose numerics the correction form changed on a two-field system
    once = _outerIterations(nFields=2, sweeps=1, symmetric=False)
    assert _outerIterations(nFields=2, sweeps=1, symmetric=True) < 0.8 * once


def test_with_exact_field_solves_one_forward_sweep_inverts_a_block_lower_triangular_system():
    # K_du != 0, K_ud = 0: block lower triangular in DOF order (displacement first), so one forward block
    # Gauss-Seidel sweep with exact field solves is the exact inverse, and the outer solve needs one iteration
    A, b, blocks = displacementDamageSystem(8, coupling="one-way")
    A = A.T.tocsr()
    solver = createSolver({"verbosity": "silent", "outerTol": 1e-10, "sweeps": 1, "symmetric": False})
    solver.setFieldStructure(blocks)
    solver(A, b)
    # the second solve of the same system reuses the standing preconditioners: make them exact
    A_scaled = (sp.diags(solver._scaling) @ A @ sp.diags(solver._scaling)).tocsr()
    solver._preconditioners = [ExactFieldSolve(A_scaled[f.start : f.stop, f.start : f.stop]) for f in blocks]
    x = solver(A, b)
    assert solver._lastOuterIters == 1
    assert np.linalg.norm(b - A @ x) <= 1e-9 * np.linalg.norm(b)
