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
"""The composed ``blockAMG()`` preset is ``BlockAMGSolver``: the same outer iterations and the same solution on a
coupled displacement / nonlocal damage system, for one solve and along a sequence of solves that exercises the reuse
policy (new values, a new sparsity pattern, a new increment).

"The same solution" is bitwise with ``OMP_NUM_THREADS=1``. With more threads, AMGCL's threaded reductions make even two
``BlockAMGSolver`` solves of one system differ by up to about 1e-10 (relative), so the comparison allows 1e-9."""

import numpy as np
import pytest
import scipy.sparse as sp
from _elasticbeam import cantilever

pytest.importorskip("edelweissfe.linsolve.amgcl.amgcl")

from edelweissfe.linsolve.base import FieldBlock  # noqa: E402
from edelweissfe.linsolve.blockamg.blockamg import BlockAMGSolver  # noqa: E402
from edelweissfe.linsolve.composition import (  # noqa: E402
    ComposedLinearSolver,
    FixedTolerance,
    blockAMG,
)


def _beamWithDamage(seed: int = 0, couplingStrength: float = 1e-2):
    """A propped 3D cantilever (displacement) with a nonlocal damage field on its nodes (a graph Laplacian of the
    element edges plus a mass-like diagonal), coupled both ways by weak random blocks."""
    K, coordinates, constrained = cantilever((64, 6, 6), (16.0, 1.5, 1.5))
    nNodes = len(coordinates)
    distances = np.linalg.norm(coordinates[:, None, :] - coordinates[None, :, :], axis=2)
    neighbours = sp.csr_matrix((distances > 0) & (distances < 0.26), dtype=float)
    K_dd = (sp.diags(np.asarray(neighbours.sum(axis=1)).ravel()) - neighbours + sp.identity(nNodes)).tocsr()
    rng = np.random.default_rng(seed)
    K_ud = sp.random(3 * nNodes, nNodes, density=2e-3, random_state=seed + 1) * couplingStrength * 1e4
    K_du = sp.random(nNodes, 3 * nNodes, density=2e-3, random_state=seed + 2) * couplingStrength
    A = sp.bmat([[K, K_ud], [K_du, K_dd]]).tocsr()
    A.sort_indices()
    b = rng.standard_normal(A.shape[0])
    fields = [FieldBlock("displacement", 0, 3 * nNodes, 3), FieldBlock("nonlocal damage", 3 * nNodes, 4 * nNodes, 1)]
    return A, b, fields, coordinates


def _solverPair(fields, coordinates, rigidBodyCoarseLevel: bool, outerTol=None):
    reference = BlockAMGSolver(
        outerTol=outerTol,
        verbosity="silent",
        nodeCoordinates={"displacement": coordinates},
        rigidBodyCoarseLevel=None if rigidBodyCoarseLevel else False,
    )
    reference.setFieldStructure(fields)
    tolerance = FixedTolerance(outerTol) if outerTol is not None else None
    composed = ComposedLinearSolver(
        blockAMG(rigidBodyCoarseLevel=rigidBodyCoarseLevel, tolerance=tolerance),
        nodeCoordinates={"displacement": coordinates},
        verbosity="silent",
    )
    composed.setFieldStructure(fields)
    return reference, composed


def _assertSameSolve(reference, composed, A, b):
    x_reference = reference(A, b)
    x_composed = composed(A, b)
    assert composed.lastSolveSummary.iters == reference.lastSolveSummary.iters
    assert np.linalg.norm(x_composed - x_reference) <= 1e-9 * np.linalg.norm(x_reference)
    print("outer iterations", composed.lastSolveSummary.iters, "rebuilt:", composed.root.lastRebuildReason)
    return composed.lastSolveSummary.iters


@pytest.mark.parametrize("rigidBodyCoarseLevel", [True, False])
def test_one_solve_matches_blockamg(rigidBodyCoarseLevel):
    A, b, fields, coordinates = _beamWithDamage()
    reference, composed = _solverPair(fields, coordinates, rigidBodyCoarseLevel, outerTol=1e-8)
    iterations = _assertSameSolve(reference, composed, A, b)
    assert 1 < iterations < 100


@pytest.mark.parametrize("rigidBodyCoarseLevel", [True, False])
def test_a_sequence_of_solves_matches_blockamg_with_its_reuse_decisions(rigidBodyCoarseLevel):
    """Forcing tolerance, hierarchy reuse: new values only (reused), a shrinking right-hand side (Newton converging),
    a jump of the right-hand side (a new increment: rebuilt), and a new sparsity pattern (rebuilt)."""
    A, b, fields, coordinates = _beamWithDamage()
    reference, composed = _solverPair(fields, coordinates, rigidBodyCoarseLevel)
    rng = np.random.default_rng(7)
    perturbation = sp.diags(1.0 + 0.05 * rng.standard_normal(A.shape[0]))
    # a weak coupling of the first and the last damage dof, which are not neighbours: a new sparsity pattern
    first, last = fields[1].start, fields[1].stop - 1
    newEntries = sp.csr_matrix(([1e-6, 1e-6], ([first, last], [last, first])), shape=A.shape)
    sequence = [
        (A, b),
        ((perturbation @ A @ perturbation).tocsr(), 0.1 * b),
        ((perturbation @ A @ perturbation).tocsr(), 0.01 * rng.standard_normal(A.shape[0])),
        (A, 10.0 * b),
        ((A + newEntries).tocsr(), b),
    ]
    rebuilds = []
    for A_k, b_k in sequence:
        A_k.sort_indices()
        _assertSameSolve(reference, composed, A_k, b_k)
        rebuilds.append(composed.root.lastRebuildReason)
    assert rebuilds[0] == "first" and rebuilds[1] is None and rebuilds[3] == "increment" and rebuilds[4] == "pattern"


def test_a_solver_keeps_no_copy_of_the_last_matrix_between_solves():
    """Like blockamg, which drops its field split after every solve: between solves (while the nonlinear solver
    assembles the next matrix) only what can be reused is kept, i.e. the preconditioners, not the field blocks of the
    last matrix (on a million dofs, gigabytes)."""
    A, b, fields, coordinates = _beamWithDamage()
    _, composed = _solverPair(fields, coordinates, rigidBodyCoarseLevel=True)
    composed(A, b)
    lgmres = composed.root
    fieldSplit = lgmres.preconditioner.inner
    assert lgmres.A is None and fieldSplit.blocks is None
    # and the next solve with new values only still reuses the preconditioners
    composed((1.0001 * A).tocsr(), b)
    assert lgmres.lastRebuildReason is None
