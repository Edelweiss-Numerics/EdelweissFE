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
"""The components of a composed solver, each tested for the property that defines it, and the combinator twins: the
fast components compute what the generic combinators spell out."""

import numpy as np
import pytest
import scipy.sparse as sp
from _blockamgsystems import displacementDamageSystem
from _elasticbeam import cantilever
from _p1lattice import dirichletDofs, lattice, latticeSystem

pytest.importorskip("edelweissfe.linsolve.amgcl.amgcl")

from edelweissfe.linsolve.base import FieldBlock  # noqa: E402
from edelweissfe.linsolve.blockamg.blockamg import BlockAMGSolver  # noqa: E402
from edelweissfe.linsolve.blockamg.rigidbodycoarselevel import (  # noqa: E402
    RigidBodyCoarseLevelPreconditioner,
)
from edelweissfe.linsolve.blockamg.schurcomplement import (  # noqa: E402
    SchurComplementCondensation,
)
from edelweissfe.linsolve.composition import (  # noqa: E402
    AMG,
    LGMRES,
    Additive,
    CoarseCorrection,
    ComposedLinearSolver,
    DiagonalScaling,
    DirectSolver,
    FieldSplit,
    FixedTolerance,
    Jacobi,
    Multiplicative,
    OnField,
    P1Interpolation,
    SchurComplement,
    SetUpContext,
    SystemMatrix,
    TwoGrid,
)
from edelweissfe.linsolve.composition.amg import (  # noqa: E402
    amgclTreeOf,
    defaultFieldTree,
)


def _context(fields, coordinates=None):
    return SetUpContext(fields=tuple(fields), coordinatesOf=lambda name: (coordinates or {}).get(name))


def _relative(x, y) -> float:
    return np.linalg.norm(x - y) / np.linalg.norm(y)


def _solved(A, b, root, fields, **options):
    solver = ComposedLinearSolver(root, verbosity="silent", **options)
    solver.setFieldStructure(fields)
    return solver(A, b), solver


# ----- the components -----


def test_diagonal_scaling_around_an_exact_solve_is_the_exact_inverse():
    """D (D A D)^-1 D = A^-1, whatever D: the scaling changes the variables, not the solution."""
    A, b, fields = displacementDamageSystem(n=6, coupling="symmetric", rowScaling=True)
    scaling = DiagonalScaling(DirectSolver())
    scaling.setUp(SystemMatrix(A), _context(fields))
    assert np.allclose(np.abs(SystemMatrix(A).scaled(scaling.D).matrix().diagonal()), 1.0)
    assert _relative(A @ scaling(b), b) < 1e-10


def test_a_field_split_of_exact_solves_is_the_block_lower_triangular_solve():
    """With exact solves per field, one forward sweep solves (B + L) z = r exactly."""
    A, b, fields = displacementDamageSystem(n=6, coupling="symmetric")
    split = FieldSplit({field.name: DirectSolver() for field in fields})
    split.setUp(SystemMatrix(A), _context(fields))
    z = split(b)
    lowerBlockTriangle = A.tolil()
    u = slice(fields[0].start, fields[0].stop)
    d = slice(fields[1].start, fields[1].stop)
    lowerBlockTriangle[u, d] = 0.0
    assert _relative(lowerBlockTriangle.tocsr() @ z, b) < 1e-10


def test_the_field_split_is_the_multiplicative_combination_of_corrections_on_single_fields():
    """FieldSplit (Gauss-Seidel, fast: coupling blocks only) and Multiplicative([OnField, OnField]) (each correction a
    product with the whole matrix) are the same preconditioner."""
    A, b, fields = displacementDamageSystem(n=12, coupling="symmetric")
    fast = FieldSplit({"displacement": AMG(), "nonlocal damage": AMG()})
    generic = Multiplicative([OnField("displacement", AMG()), OnField("nonlocal damage", AMG())])
    for preconditioner in (fast, generic):
        preconditioner.setUp(SystemMatrix(A), _context(fields))
    assert _relative(fast(b), generic(b)) < 1e-10


def test_an_additive_combination_sums_its_corrections():
    A, b, fields = displacementDamageSystem(n=6)
    combination = Additive([Jacobi(omega=0.5), Jacobi(omega=0.25)])
    combination.setUp(SystemMatrix(A), _context(fields))
    assert _relative(combination(b), 0.75 * b / A.diagonal()) < 1e-14


def test_an_exact_coarse_correction_is_the_a_orthogonal_projection_onto_the_coarse_space():
    """P A_c^-1 P^T A P = P: the correction reproduces everything in the range of P."""
    K, coordinates, _ = cantilever((8, 2, 2), (4.0, 1.0, 1.0))
    P = sp.random(K.shape[0], 12, density=0.3, random_state=4, format="csr")
    correction = CoarseCorrection(P, DirectSolver())
    correction.setUp(SystemMatrix(K), _context([FieldBlock("displacement", 0, K.shape[0], 3)]))
    KP = (K @ P).toarray()
    reproduced = np.column_stack([correction(KP[:, j]) for j in range(P.shape[1])])
    assert np.allclose(reproduced, P.toarray(), atol=1e-10 * abs(P).max())


def test_a_two_grid_cycle_spelled_out_by_combinators_beats_its_smoother_alone():
    """Multiplicative([smoother, CoarseCorrection(P, exact), smoother]) on the rigid-body modes of a slender
    cantilever: the coarse space carries what Jacobi cannot reduce, so far fewer outer iterations."""
    K, coordinates, _ = cantilever((32, 2, 2), (16.0, 1.0, 1.0))
    fields = [FieldBlock("displacement", 0, K.shape[0], 3)]
    b = np.random.default_rng(2).standard_normal(K.shape[0])
    D = 1.0 / np.sqrt(K.diagonal())
    K_scaled = (sp.diags(D) @ K @ sp.diags(D)).tocsr()
    K_scaled.sort_indices()
    rigidBody = RigidBodyCoarseLevelPreconditioner(amgclTreeOf(defaultFieldTree(3)), 3, "double")
    rigidBody.build(K_scaled, D, coordinates)

    def prolongation(A, context):
        return rigidBody.prolongationMatrix()

    twoGrid = Multiplicative([Jacobi(), CoarseCorrection(prolongation, DirectSolver()), Jacobi()])
    iterations = {}
    for name, smoother in (("jacobi", Jacobi()), ("two-grid", twoGrid)):
        root = LGMRES(DiagonalScaling(smoother), tolerance=FixedTolerance(1e-8), maxIterations=2000)
        x, _ = _solved(K, b, root, fields)
        iterations[name] = root.lastIterations
    # Jacobi alone does not even converge within the cap; the two-grid cycle converges
    assert _relative(K @ x, b) < 1e-7
    assert iterations["two-grid"] < iterations["jacobi"] / 5


def test_the_p1_two_grid_matches_blockamg():
    coords, isCorner, edgeEndpoints = lattice(6)
    constrained = dirichletDofs(coords)
    As, dinv = latticeSystem(coords, constrained)
    A = (As.multiply(1.0 / dinv[:, None]).multiply(1.0 / dinv[None, :])).tocsr()
    b = np.random.default_rng(3).standard_normal(A.shape[0])
    b[constrained] = 0.0
    fields = [FieldBlock("displacement", 0, 3 * len(isCorner), 3)]
    p1Maps = {"displacement": (isCorner, edgeEndpoints)}
    reference = BlockAMGSolver(outerTol=1e-8, outerRestart=300, verbosity="silent", p1Maps=p1Maps)
    reference.setFieldStructure(fields)
    root = LGMRES(
        DiagonalScaling(FieldSplit({"displacement": TwoGrid(P1Interpolation())})),
        tolerance=FixedTolerance(1e-8),
        maxIterations=300,
    )
    x, composed = _solved(A, b, root, fields, p1Maps=p1Maps)
    x_reference = reference(A, b)
    assert composed.lastSolveSummary.iters == reference.lastSolveSummary.iters
    assert _relative(x, x_reference) < 1e-9


# ----- the Schur complement -----


def _barOnFoundation(n=10):
    """A grid Laplacian (host) and a bar tied to cell centres by springs, as in test_blockamg_schur_complement."""
    from _blockamgsystems import gridLaplacian

    host = (gridLaplacian(n) + 1e-3 * sp.identity(n**3)).tocsr()
    nBar = n * n
    bar = 1e4 * sp.diags([-1.0, 2.0, -1.0], [-1, 0, 1], shape=(nBar, nBar))
    cells = np.arange(nBar) % (n - 1)
    corners = [
        cells + dx + n * (cells + dy) + n * n * ((cells * 7) % (n - 1) + dz)
        for dx in (0, 1)
        for dy in (0, 1)
        for dz in (0, 1)
    ]
    N = sp.csr_matrix(
        (np.full(8 * nBar, 1.0 / 8.0), (np.tile(np.arange(nBar), 8), np.concatenate(corners))), shape=(nBar, n**3)
    )
    k = 1e3
    A = sp.bmat([[host + k * (N.T @ N), -k * N.T], [-k * N, bar + k * sp.identity(nBar)]]).tocsr()
    A.sort_indices()
    barDofs = np.arange(n**3, n**3 + nBar)
    b = np.random.default_rng(1).standard_normal(A.shape[0])
    return A, b, barDofs


class _FixedDofs(SchurComplementCondensation):
    """A condensation of given dofs, without a model."""

    def __init__(self, dofs, createDirectSolver, **options):
        super().__init__([], createDirectSolver, **options)
        self.fixedDofs = dofs

    def _selectCondensedDofs(self):
        return self.fixedDofs, self.fixedDofs


def test_block_lu_with_the_exact_schur_complement_and_exact_solves_is_the_inverse():
    A, b, barDofs = _barOnFoundation(n=6)
    solveA_bb = DirectSolver()
    condensation = _FixedDofs(barDofs, solveA_bb.newSolver, dropTolerance=0.0, windowNodes=10_000, blockFactors="LU")
    schur = SchurComplement(
        preconditionS=DirectSolver(), solveA_bb=solveA_bb, blockFactors="LU", condensation=condensation
    )
    schur.setUp(SystemMatrix(A), _context([FieldBlock("u", 0, A.shape[0], 1)]))
    assert _relative(A @ schur(b), b) < 1e-10


def test_the_schur_complement_tree_matches_blockamg_with_condensation():
    A, b, barDofs = _barOnFoundation(n=18)
    fields = [FieldBlock("u", 0, A.shape[0], 1)]
    reference = BlockAMGSolver(
        outerTol=1e-8, verbosity="silent", sweeps=2, condensation=_FixedDofs(barDofs, DirectSolver().newSolver)
    )
    reference.setFieldStructure(fields)
    x_reference = reference(A, b)
    solveA_bb = DirectSolver()
    root = LGMRES(
        SchurComplement(
            preconditionS=DiagonalScaling(FieldSplit({"u": AMG()}, sweeps=2)),
            solveA_bb=solveA_bb,
            condensation=_FixedDofs(barDofs, solveA_bb.newSolver),
        ),
        tolerance=FixedTolerance(1e-8),
    )
    x, composed = _solved(A, b, root, fields)
    assert composed.lastSolveSummary.iters == reference.lastSolveSummary.iters < 30
    assert _relative(x, x_reference) < 1e-9


# ----- statistics -----


def test_every_component_records_its_calls():
    A, b, fields = displacementDamageSystem(n=10, coupling="symmetric")
    fieldSplit = FieldSplit({"displacement": AMG(), "nonlocal damage": AMG()})
    root = LGMRES(DiagonalScaling(fieldSplit), tolerance=FixedTolerance(1e-8))
    _solved(A, b, root, fields)
    assert root.statistics.applications == 1 and root.statistics.updates == 1
    # once per outer iteration (and once more for the initial residual of AMGCL's LGMRES, if it asks)
    assert root.lastIterations <= fieldSplit.statistics.applications <= root.lastIterations + 1
    assert fieldSplit.fieldPreconditioners["displacement"].statistics.setUps == 1
    assert fieldSplit.fieldPreconditioners["displacement"].statistics.sizes["rows"] == fields[0].stop
