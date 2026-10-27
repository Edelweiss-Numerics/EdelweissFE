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
"""The components of a composed solver, each tested for the property that defines it; and wrong compositions, which
fail as early as possible."""

import numpy as np
import pytest
import scipy.sparse as sp
from _elasticbeam import cantilever
from _linsolvesystems import (
    barOnFoundation,
    condensingGivenDofs,
    displacementDamageSystem,
)
from _p1lattice import dirichletDofs, lattice, latticeSystem, orphanSystem

pytest.importorskip("edelweissfe.linsolve.amgcl.amgcl")

from edelweissfe.linsolve.base import FieldBlock  # noqa: E402
from edelweissfe.linsolve.composition import (  # noqa: E402
    AMG,
    FGMRES,
    LGMRES,
    Additive,
    CoarseCorrection,
    ComposedLinearSolver,
    DiagonalScaling,
    DirectSolver,
    FixedTolerance,
    Jacobi,
    Multiplicative,
    OnField,
    P1Interpolation,
    RigidBodyModes,
    SchurComplement,
    SetUpContext,
    SystemMatrix,
    TwoLevel,
    gaussSeidelOrder,
)
from edelweissfe.linsolve.composition.amg import defaultFieldTree  # noqa: E402
from edelweissfe.linsolve.composition.rigidbodycoarselevel import (  # noqa: E402
    RigidBodyCoarseLevelPreconditioner,
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


def _blockGaussSeidel(fields, makeInner):
    return Multiplicative([OnField(field.name, makeInner()) for field in fields])


def test_block_gauss_seidel_of_exact_solves_is_the_block_lower_triangular_solve():
    """With exact solves per field, one forward sweep solves (B + L) z = r exactly."""
    A, b, fields = displacementDamageSystem(n=6, coupling="symmetric")
    split = _blockGaussSeidel(fields, DirectSolver)
    split.setUp(SystemMatrix(A), _context(fields))
    z = split(b)
    lowerBlockTriangle = A.tolil()
    u = slice(fields[0].start, fields[0].stop)
    d = slice(fields[1].start, fields[1].stop)
    lowerBlockTriangle[u, d] = 0.0
    assert _relative(lowerBlockTriangle.tocsr() @ z, b) < 1e-10


def test_a_multiplicative_combination_of_field_corrections_is_the_textbook_formula():
    """Multiplicative([OnField(u), OnField(d), OnField(u)]) forms each field's residual from the coupling blocks of the
    fields corrected so far; it equals z <- z + R^T M^-1 R (r - A z) with the whole matrix."""
    A, b, fields = displacementDamageSystem(n=12, coupling="symmetric")
    u, d = OnField("displacement", AMG()), OnField("nonlocal damage", AMG())
    symmetric = Multiplicative([u, d, u])
    symmetric.setUp(SystemMatrix(A), _context(fields))
    assert u.statistics.setUps == 1
    z = np.zeros_like(b)
    for correction in (u, d, u):
        dofs = correction.dofs
        z[dofs] += correction.inner((b - A @ z)[dofs])
    assert _relative(symmetric(b), z) < 1e-10


def test_the_corrections_of_one_matrix_share_one_field_split():
    """One threaded pass forms all field blocks of a matrix, whichever component asks first; the Krylov operator comes
    from the same pass."""
    A, b, fields = displacementDamageSystem(n=10, coupling="symmetric")
    import edelweissfe.linsolve.amgcl.amgcl as amgcl

    builds = []

    class Counted(amgcl.PyAMGCLEquilibratedSystem):
        def build(self, *args, **kwargs):
            builds.append("split")
            return super().build(*args, **kwargs)

        def buildOperator(self, *args, **kwargs):
            builds.append("operator")
            return super().buildOperator(*args, **kwargs)

    original = amgcl.PyAMGCLEquilibratedSystem
    amgcl.PyAMGCLEquilibratedSystem = Counted
    try:
        root = LGMRES(DiagonalScaling(_blockGaussSeidel(fields, AMG)), tolerance=FixedTolerance(1e-8))
        _solved(A, b, root, fields)
    finally:
        amgcl.PyAMGCLEquilibratedSystem = original
    assert builds == ["split"]


def test_gauss_seidel_orders():
    assert gaussSeidelOrder(["u", "d"]) == ["u", "d"]
    assert gaussSeidelOrder(["u", "d"], symmetric=True) == ["u", "d", "u"]
    assert gaussSeidelOrder(["u", "d", "p"], sweeps=2, symmetric=True) == ["u", "d", "p", "d", "u"] * 2
    assert gaussSeidelOrder(["u"], sweeps=2) == ["u", "u"]


def test_an_additive_combination_of_field_corrections_is_block_jacobi():
    A, b, fields = displacementDamageSystem(n=6, coupling="symmetric")
    jacobi = Additive([OnField(field.name, DirectSolver()) for field in fields])
    jacobi.setUp(SystemMatrix(A), _context(fields))
    blockDiagonal = sp.block_diag([A[f.start : f.stop, f.start : f.stop] for f in fields]).tocsr()
    assert _relative(blockDiagonal @ jacobi(b), b) < 1e-10


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
    rigidBody = RigidBodyCoarseLevelPreconditioner(defaultFieldTree(3), 3, "double")
    rigidBody.build(K_scaled, D, coordinates)

    def prolongation(A, context):
        P = rigidBody.prolongationMatrix()
        return P, context.ofCoarseDofs(P.shape[1])

    twoLevel = Multiplicative([Jacobi(), CoarseCorrection(prolongation, DirectSolver()), Jacobi()])
    iterations = {}
    for name, smoother in (("jacobi", Jacobi()), ("two-level", twoLevel)):
        root = LGMRES(DiagonalScaling(smoother), tolerance=FixedTolerance(1e-8), maxIterations=2000)
        x, _ = _solved(K, b, root, fields)
        iterations[name] = root.lastIterations
    # Jacobi alone does not even converge within the cap; the two-level cycle converges
    assert _relative(K @ x, b) < 1e-7
    assert iterations["two-level"] < iterations["jacobi"] / 5


def _spelledOutP1Interpolation(A, context):
    """The P1 interpolation of the field's displacements in the variables of the matrix: D^-1 P_geo D_c, with the
    scaling of the context (P_geo without one), and the corner-node field as the coarse context."""
    field = context.onlyField("p1Interpolation")
    isCorner, edgeEndpoints = context.p1Map(field.name)
    cornerNumber = np.cumsum(isCorner) - 1
    rows, columns, weights = [], [], []
    for node in range(len(isCorner)):
        corners = [node] if isCorner[node] else edgeEndpoints[node]
        for corner in corners:
            rows, columns, weights = rows + [node], columns + [cornerNumber[corner]], weights + [1.0 / len(corners)]
    nodalP = sp.csr_matrix((weights, (rows, columns)), shape=(len(isCorner), np.count_nonzero(isCorner)))
    P_geo = sp.kron(nodalP, sp.identity(field.dimension), format="csr")
    coarseContext = context.ofCoarseNodes(field, isCorner)
    s, s_c = context.scalingOf(slice(None)), coarseContext.scalingOf(slice(None))
    return (sp.diags(1.0 / s) @ P_geo @ sp.diags(s_c)).tocsr(), coarseContext


def test_a_spelled_out_p1_two_grid_solves_its_coarse_field_with_the_rigid_body_coarse_level():
    """Multiplicative([smoother, CoarseCorrection(p1Interpolation, DiagonalScaling(TwoLevel(RigidBodyModes()))),
    smoother]): the prolongation callable says that the coarse dofs are the displacement field on the corner nodes, so
    the coarse solve of a displacement field works on them."""
    A, b, fields, coords, isCorner, edgeEndpoints, constrained = _p1System()
    coarse = DiagonalScaling(TwoLevel(coarseSpace=RigidBodyModes()))
    smoother = Jacobi(omega=0.4)
    cycle = Multiplicative([smoother, CoarseCorrection(_spelledOutP1Interpolation, coarse), smoother])
    iterations = {}
    for name, root in (
        ("smoother alone", LGMRES(DiagonalScaling(smoother), tolerance=FixedTolerance(1e-8), maxIterations=2000)),
        ("two-level", LGMRES(DiagonalScaling(cycle), tolerance=FixedTolerance(1e-8), maxIterations=2000)),
    ):
        x, composed = _solved(
            A,
            b,
            root,
            fields,
            nodeCoordinates={"displacement": coords},
            p1Maps={"displacement": (isCorner, edgeEndpoints)},
        )
        assert _relative(A @ x, b) < 1e-7
        iterations[name] = composed.lastSolveSummary.iters
    assert iterations["two-level"] < iterations["smoother alone"] / 2
    # the rigid-body coarse level was built once, on the dofs of the corner nodes
    assert coarse.inner.statistics.setUps == 1
    assert coarse.inner.statistics.sizes["rows"] == 3 * np.count_nonzero(isCorner)


def test_a_prolongation_callable_must_say_what_the_coarse_dofs_are():
    A, b, fields, coords, isCorner, edgeEndpoints, constrained = _p1System(4)
    correction = CoarseCorrection(lambda A, context: sp.identity(A.shape[0], format="csr"), DirectSolver())
    with pytest.raises((TypeError, ValueError)):
        correction.setUp(SystemMatrix(A), _context(fields))


def test_a_reused_coarse_correction_keeps_its_prolongation_and_its_coarse_hierarchy():
    """Between rebuilds, P stays and the coarse AMG takes A_c as an update, keeping its hierarchy, as in TwoLevel."""
    K, coordinates, _ = cantilever((16, 2, 2), (8.0, 1.0, 1.0))
    fields = [FieldBlock("displacement", 0, K.shape[0], 3)]
    b = np.random.default_rng(3).standard_normal(K.shape[0])
    # piecewise constant translations on aggregates of 8 nodes
    nodes = K.shape[0] // 3
    aggregates = sp.csr_matrix((np.ones(nodes), (np.arange(nodes), np.arange(nodes) // 8)))
    P = sp.kron(aggregates, sp.identity(3)).tocsr()
    prolongations = []

    def prolongation(A, context):
        prolongations.append(A.shape)
        return P, context.ofCoarseDofs(P.shape[1])

    coarse = AMG()
    root = LGMRES(
        DiagonalScaling(Multiplicative([Jacobi(), CoarseCorrection(prolongation, coarse), Jacobi()])),
        tolerance=FixedTolerance(1e-8),
        maxIterations=2000,
    )
    _, composed = _solved(K, b, root, fields)
    x = composed((1.0001 * K).tocsr(), b)
    assert root.lastRebuildReason is None and _relative(1.0001 * (K @ x), b) < 1e-7
    assert len(prolongations) == 1
    assert (coarse.statistics.setUps, coarse.statistics.updates) == (1, 1)


def _p1System(nCornersPerAxis=6):
    """The lattice block as an unscaled system with a right-hand side zero on the Dirichlet dofs, its field and its
    node coordinates and P1 map."""
    coords, isCorner, edgeEndpoints = lattice(nCornersPerAxis)
    constrained = dirichletDofs(coords)
    As, scaling = latticeSystem(coords, constrained)
    A = (As.multiply(1.0 / scaling[:, None]).multiply(1.0 / scaling[None, :])).tocsr()
    b = np.random.default_rng(3).standard_normal(A.shape[0])
    b[constrained] = 0.0
    fields = [FieldBlock("displacement", 0, 3 * len(isCorner), 3)]
    return A, b, fields, coords, isCorner, edgeEndpoints, constrained


class _RecordingExactSolve(DirectSolver):
    """An exact solve that keeps the matrix and the context of its set-up."""

    def _setUp(self, A, context):
        self.recorded = (A.matrix().copy(), context)
        super()._setUp(A, context)


def test_the_p1_coarse_problem_is_the_corner_node_field():
    """The coarse component gets the corner-node field: the field's name and dimension, the corners' coordinates in
    their node order, the scaling of the corner dofs, and an identity row for each Dirichlet corner dof."""
    A, b, fields, coords, isCorner, edgeEndpoints, constrained = _p1System()
    coarse = _RecordingExactSolve()
    twoLevel = TwoLevel(P1Interpolation(coarse=coarse))
    scaling = DiagonalScaling(twoLevel)
    scaling.setUp(
        SystemMatrix(A),
        SetUpContext(
            fields=tuple(fields),
            coordinatesOf={"displacement": coords}.get,
            p1MapOf={"displacement": (isCorner, edgeEndpoints)}.get,
        ),
    )
    A_c, context = coarse.recorded
    cornerNodes = np.nonzero(isCorner)[0]
    cornerDofs = (3 * cornerNodes[:, None] + np.arange(3)).ravel()
    assert context.fields == (FieldBlock("displacement", 0, len(cornerDofs), 3),)
    assert np.array_equal(context.coordinates("displacement"), coords[cornerNodes])
    assert np.array_equal(context.scaling, scaling.D[cornerDofs])
    assert context.p1Map("displacement") is None and context.model is None
    coarseDirichlet = np.isin(cornerDofs, constrained)
    assert coarseDirichlet.any()
    identityRows = np.diff(A_c.indptr) == 1
    assert np.array_equal(identityRows, coarseDirichlet)
    assert np.all(A_c.diagonal()[coarseDirichlet] == 1.0)
    # the coarse dofs are the corners' dofs, and the coarse component is a child of the cycle
    assert twoLevel.cycle.nCoarse == len(cornerDofs)
    assert twoLevel.children() == (("coarse", coarse),)


def test_the_p1_coarse_field_includes_a_midside_node_made_a_coarse_node():
    """Issue #223: a free midside dof between two constrained corner dofs makes its node a coarse node, and the coarse
    field the default coarse solve gets includes it: its coordinates, its scaling, its Dirichlet rows."""
    As, scaling, coords, isCorner, edgeEndpoints, orphan = orphanSystem()
    A = (As.multiply(1.0 / scaling[:, None]).multiply(1.0 / scaling[None, :])).tocsr()
    b = np.random.default_rng(3).standard_normal(A.shape[0])
    b[np.diff(As.indptr) == 1] = 0.0
    fields = [FieldBlock("displacement", 0, A.shape[0], 3)]
    coarse = _RecordingExactSolve()
    p1TwoGrid = TwoLevel(P1Interpolation(coarse=coarse))
    root = LGMRES(DiagonalScaling(p1TwoGrid), tolerance=FixedTolerance(1e-8), maxIterations=300)
    x, _ = _solved(
        A, b, root, fields, nodeCoordinates={"displacement": coords}, p1Maps={"displacement": (isCorner, edgeEndpoints)}
    )
    assert _relative(A @ x, b) < 1e-8
    A_c, context = coarse.recorded
    coarseNodes = np.nonzero(isCorner | (np.arange(len(isCorner)) == orphan))[0]
    coarseDofs = (3 * coarseNodes[:, None] + np.arange(3)).ravel()
    assert np.array_equal(context.coordinates("displacement"), coords[coarseNodes])
    assert np.array_equal(context.scaling, root.preconditioner.D[coarseDofs])
    assert np.array_equal(np.diff(A_c.indptr) == 1, (np.diff(As.indptr) == 1)[coarseDofs])
    # the default coarse solve runs on it too
    default = LGMRES(DiagonalScaling(TwoLevel(P1Interpolation())), tolerance=FixedTolerance(1e-8), maxIterations=300)
    x, _ = _solved(
        A,
        b,
        default,
        fields,
        nodeCoordinates={"displacement": coords},
        p1Maps={"displacement": (isCorner, edgeEndpoints)},
    )
    assert _relative(A @ x, b) < 1e-8


def test_the_p1_two_grid_with_its_default_coarse_solve_converges():
    """The default coarse solve, DiagonalScaling(TwoLevel(coarseSpace=RigidBodyModes())), runs on the corner-node field:
    three levels, quadratic -> linear -> rigid-body modes of aggregates."""
    A, b, fields, coords, isCorner, edgeEndpoints, constrained = _p1System()
    p1TwoGrid = TwoLevel(P1Interpolation())
    root = LGMRES(DiagonalScaling(p1TwoGrid), tolerance=FixedTolerance(1e-8), maxIterations=300)
    x, composed = _solved(
        A,
        b,
        root,
        fields,
        nodeCoordinates={"displacement": coords},
        p1Maps={"displacement": (isCorner, edgeEndpoints)},
    )
    assert _relative(A @ x, b) < 1e-8
    coarse = p1TwoGrid.coarseSpace.coarse
    assert isinstance(coarse, DiagonalScaling) and isinstance(coarse.inner, TwoLevel)
    assert coarse.statistics.setUps == 1
    assert coarse.statistics.applications == p1TwoGrid.statistics.applications
    assert composed.lastSolveSummary.iters <= 12


def test_the_p1_coarse_solve_with_an_exact_solve_beats_the_default_in_iterations_only():
    """An exact coarse solve bounds what the coarse level can do: the two-grid with it needs no more iterations than
    with the default (inexact) coarse solve."""
    A, b, fields, coords, isCorner, edgeEndpoints, constrained = _p1System()
    iterations = {}
    for name, coarse in (("exact", DirectSolver()), ("default", None)):
        root = LGMRES(DiagonalScaling(TwoLevel(P1Interpolation(coarse))), tolerance=FixedTolerance(1e-8))
        x, composed = _solved(
            A,
            b,
            root,
            fields,
            nodeCoordinates={"displacement": coords},
            p1Maps={"displacement": (isCorner, edgeEndpoints)},
        )
        assert _relative(A @ x, b) < 1e-8
        iterations[name] = composed.lastSolveSummary.iters
    assert iterations["exact"] <= iterations["default"]


def test_the_p1_two_grid_reuses_its_coarse_solve_while_only_the_values_change():
    A, b, fields, coords, isCorner, edgeEndpoints, constrained = _p1System()
    p1TwoGrid = TwoLevel(P1Interpolation())
    root = LGMRES(DiagonalScaling(p1TwoGrid), tolerance=FixedTolerance(1e-8), maxIterations=300)
    _, composed = _solved(
        A, b, root, fields, nodeCoordinates={"displacement": coords}, p1Maps={"displacement": (isCorner, edgeEndpoints)}
    )
    x = composed((1.0001 * A).tocsr(), b)
    assert _relative(1.0001 * (A @ x), b) < 1e-7
    coarse = p1TwoGrid.coarseSpace.coarse
    assert (p1TwoGrid.statistics.setUps, coarse.statistics.setUps, coarse.inner.statistics.setUps) == (1, 1, 1)
    assert coarse.statistics.applications == p1TwoGrid.statistics.applications


def test_the_p1_coarse_solve_takes_any_component_and_shows_in_the_views():
    A, b, fields, coords, isCorner, edgeEndpoints, constrained = _p1System()
    p1TwoGrid = TwoLevel(P1Interpolation(coarse=AMG(backendBlockSize=1)))
    root = LGMRES(DiagonalScaling(p1TwoGrid), tolerance=FixedTolerance(1e-8), maxIterations=300)
    x, _ = _solved(
        A, b, root, fields, nodeCoordinates={"displacement": coords}, p1Maps={"displacement": (isCorner, edgeEndpoints)}
    )
    assert _relative(A @ x, b) < 1e-8
    from edelweissfe.linsolve.composition.views import formula

    assert "AMG" in formula(root) and "P1 interpolation" in formula(root)
    assert "AMG" in str(root)


def test_the_p1_coarse_solve_must_be_a_component():
    with pytest.raises(TypeError, match="P1Interpolation"):
        P1Interpolation(coarse=AMG)


# ----- the Schur complement -----


def test_block_lu_with_the_exact_schur_complement_and_exact_solves_is_the_inverse():
    A, b, barDofs = barOnFoundation(n=6)
    schur = condensingGivenDofs(
        SchurComplement(["bars"], preconditionS=DirectSolver(), blockFactors="LU"),
        barDofs,
        dropTolerance=0.0,
        windowNodes=10_000,
    )
    schur.setUp(SystemMatrix(A), _context([FieldBlock("u", 0, A.shape[0], 1)]))
    assert _relative(A @ schur(b), b) < 1e-10


def test_the_schur_complement_tree_has_blockamgs_numbers_with_condensation():
    """The outer iterations and ||x|| of BlockAMGSolver with its Schur complement condensation (deleted; recorded at
    ffe4b681), recorded again with the Arnoldi estimate of the spectral radius, see
    test_linsolve_composition_equivalence."""
    A, b, barDofs = barOnFoundation(n=18)
    fields = [FieldBlock("u", 0, A.shape[0], 1)]
    root = LGMRES(
        condensingGivenDofs(
            SchurComplement(
                ["bars"],
                preconditionS=DiagonalScaling(Multiplicative(gaussSeidelOrder([OnField("u", AMG())], sweeps=2))),
            ),
            barDofs,
        ),
        tolerance=FixedTolerance(1e-8),
    )
    x, composed = _solved(A, b, root, fields)
    assert composed.lastSolveSummary.iters == 15
    assert np.linalg.norm(x) == pytest.approx(24.215507056460062, rel=1e-8)
    assert _relative(A @ x, b) < 1e-8


def test_the_schur_complement_keeps_nothing_formed_from_its_matrices_between_solves():
    """S~ is kept for the next solve, but not the native split of it the solve formed (two copies of S~ on a large
    model), and the next solve with new values still reuses the preconditioner of S~."""
    A, b, barDofs = barOnFoundation(n=8)
    fields = [FieldBlock("u", 0, A.shape[0], 1)]
    schur = condensingGivenDofs(
        SchurComplement(["bars"], preconditionS=DiagonalScaling(Multiplicative([OnField("u", AMG())]))), barDofs
    )
    root = LGMRES(schur, tolerance=FixedTolerance(1e-8))
    _, composed = _solved(A, b, root, fields)
    assert schur.S_tilde._fieldSplit is None and schur.S_tilde._krylovOperator is None
    x = composed((1.0001 * A).tocsr(), b)
    assert root.lastRebuildReason is None and _relative(1.0001 * (A @ x), b) < 1e-7


def test_the_schur_complement_notes_what_it_condensed_for_the_debug_column_of_a_rebuild_only():
    A, b, barDofs = barOnFoundation(n=8)
    fields = [FieldBlock("u", 0, A.shape[0], 1)]
    schur = condensingGivenDofs(
        SchurComplement(["bars"], preconditionS=DiagonalScaling(Multiplicative([OnField("u", AMG())]))), barDofs
    )
    solver = ComposedLinearSolver(LGMRES(schur, tolerance=FixedTolerance(1e-8)), verbosity="debug")
    solver.setFieldStructure(fields)
    solver(A, b)
    first = solver.lastSolveSummary.detailLines
    assert first[0] == "rebuilt(first)"
    assert first[2] == "{:} dofs condensed".format(len(barDofs)) and first[3].startswith("S̃ ")
    assert all(len(note) <= 24 for note in first)
    solver((1.0001 * A).tocsr(), b)
    assert solver.lastSolveSummary.detailLines[0] == "reused" and len(solver.lastSolveSummary.detailLines) == 2


@pytest.mark.parametrize(
    "count, compact", [(7, "7"), (999, "999"), (240084, "240k"), (1830294, "1.83M"), (233398138, "233M")]
)
def test_a_count_in_a_note_has_three_significant_digits(count, compact):
    from edelweissfe.linsolve.composition.schurcomplement import _compactCount

    assert _compactCount(count) == compact


# ----- statistics -----


def test_every_component_records_its_calls():
    A, b, fields = displacementDamageSystem(n=10, coupling="symmetric")
    displacement = AMG()
    blockGaussSeidel = Multiplicative([OnField("displacement", displacement), OnField("nonlocal damage", AMG())])
    root = LGMRES(DiagonalScaling(blockGaussSeidel), tolerance=FixedTolerance(1e-8))
    _solved(A, b, root, fields)
    assert root.statistics.applications == 1 and root.statistics.updates == 1
    # once per outer iteration (and once more for the initial residual of AMGCL's LGMRES, if it asks)
    assert root.lastIterations <= blockGaussSeidel.statistics.applications <= root.lastIterations + 1
    assert displacement.statistics.setUps == 1
    assert displacement.statistics.applications == blockGaussSeidel.statistics.applications
    assert displacement.statistics.sizes["rows"] == fields[0].stop


def test_one_call_is_counted_once_also_through_a_hook_of_the_same_component():
    """A Krylov solver's set-up is its update for a first matrix: one call of the component. A combination's first
    update builds it, through its public set-up: one update with a set-up in it."""
    A, b, fields = displacementDamageSystem(n=10, coupling="symmetric")
    blockGaussSeidel = Multiplicative([OnField(field.name, AMG()) for field in fields])
    root = LGMRES(DiagonalScaling(blockGaussSeidel), tolerance=FixedTolerance(1e-8))
    root.setUp(SystemMatrix(A), _context(fields))
    assert (root.statistics.setUps, root.statistics.updates) == (1, 0)
    assert (blockGaussSeidel.statistics.setUps, blockGaussSeidel.statistics.updates) == (1, 1)
    onField = blockGaussSeidel.corrections[0]
    assert (onField.statistics.setUps, onField.statistics.updates) == (1, 0)

    # driven by the adapter, which only updates the root: two solves are two updates of every combination
    blockGaussSeidel = Multiplicative(lambda context: [OnField(field.name, AMG()) for field in context.fields])
    root = LGMRES(DiagonalScaling(blockGaussSeidel), tolerance=FixedTolerance(1e-8))
    _, composed = _solved(A, b, root, fields)
    composed(A, b)
    assert (root.statistics.setUps, root.statistics.updates, root.statistics.applications) == (0, 2, 2)
    assert (blockGaussSeidel.statistics.setUps, blockGaussSeidel.statistics.updates) == (1, 2)
    for _, onField in blockGaussSeidel.children():
        assert (onField.statistics.setUps, onField.statistics.updates) == (1, 1)
        assert onField.inner.statistics.setUps == 1


# ----- a variable preconditioner: FGMRES -----


def test_lgmres_refuses_a_variable_preconditioner_and_names_fgmres():
    inner = LGMRES(AMG(), tolerance=FixedTolerance(1e-2))
    assert not inner.isFixedLinearOperator and DiagonalScaling(AMG()).isFixedLinearOperator
    with pytest.raises(ValueError, match="FGMRES"):
        LGMRES(inner)
    with pytest.raises(ValueError, match="FGMRES"):
        LGMRES(Multiplicative([inner]))


def test_lgmres_refuses_a_variable_preconditioner_made_at_the_set_up():
    """A combination that makes its corrections per field shows what they are only at the first set-up."""
    A, b, fields = displacementDamageSystem(n=6, coupling="symmetric")
    root = LGMRES(Multiplicative(lambda context: [LGMRES(AMG(), tolerance=FixedTolerance(1e-2))]))
    with pytest.raises(ValueError, match="FGMRES"):
        _solved(A, b, root, [FieldBlock("all", 0, A.shape[0], 1)])


def test_fgmres_with_an_inner_krylov_solve_converges():
    A, b, fields = displacementDamageSystem(n=10, coupling="symmetric")
    inner = LGMRES(DiagonalScaling(_blockGaussSeidel(fields, AMG)), tolerance=FixedTolerance(1e-1), maxIterations=5)
    root = FGMRES(inner, tolerance=FixedTolerance(1e-8), maxIterations=100)
    x, solver = _solved(A, b, root, fields)
    assert _relative(A @ x, b) < 1e-7
    assert 0 < root.lastIterations < 20 and solver.lastSolveSummary.residualMet


def test_fgmres_with_a_fixed_preconditioner_needs_the_iterations_of_gmres():
    A, b, fields = displacementDamageSystem(n=10, coupling="symmetric")
    iterations = {}
    for Krylov in (LGMRES, FGMRES):
        root = Krylov(DiagonalScaling(_blockGaussSeidel(fields, AMG)), tolerance=FixedTolerance(1e-8))
        x, _ = _solved(A, b, root, fields)
        assert _relative(A @ x, b) < 1e-7
        iterations[Krylov.__name__] = root.lastIterations
    assert abs(iterations["FGMRES"] - iterations["LGMRES"]) <= 2


# ----- wrong compositions fail early -----


def test_a_child_must_be_a_component():
    with pytest.raises(TypeError, match="must be a component"):
        DiagonalScaling(AMG)
    with pytest.raises(TypeError, match="must be a component"):
        Multiplicative([AMG(), "AMG"])
    with pytest.raises(ValueError, match="at least one correction"):
        Additive([])
    with pytest.raises(ValueError, match="nothing to condense"):
        SchurComplement(preconditionS=AMG())
    with pytest.raises(ValueError, match="blockFactors"):
        SchurComplement(["a marker"], preconditionS=AMG(), blockFactors="L")
    with pytest.raises(TypeError, match="solveA_bb must be a DirectSolver"):
        SchurComplement(["a marker"], preconditionS=AMG(), solveA_bb=AMG())
    with pytest.raises(ValueError, match="approximation"):
        SchurComplement(["a marker"], preconditionS=AMG(), approximation="exact")


def test_a_missing_field_fails_at_the_set_up_and_names_the_fields():
    A, b, fields = displacementDamageSystem(n=6, coupling="symmetric")
    with pytest.raises(ValueError, match="no field 'temperature'; its fields are displacement, nonlocal damage"):
        OnField("temperature", AMG()).setUp(SystemMatrix(A), _context(fields))


@pytest.mark.parametrize("Combination", [Multiplicative, Additive])
def test_a_combination_that_leaves_a_field_uncorrected_fails_at_the_set_up(Combination):
    A, b, fields = displacementDamageSystem(n=6, coupling="symmetric")
    with pytest.raises(ValueError, match="no correction acts on the field\\(s\\) nonlocal damage"):
        Combination([OnField("displacement", AMG())]).setUp(SystemMatrix(A), _context(fields))


def test_a_one_field_component_on_several_fields_fails_at_the_set_up():
    A, b, fields = displacementDamageSystem(n=6, coupling="symmetric")
    with pytest.raises(ValueError, match="AMG is built on the matrix of one field.*OnField"):
        AMG().setUp(SystemMatrix(A), _context(fields))


def test_rigid_body_modes_need_a_vector_field_with_coordinates():
    A, b, fields = displacementDamageSystem(n=6, coupling="symmetric")
    from edelweissfe.linsolve.composition import RigidBodyModes

    damage = fields[1]
    block = SystemMatrix(A[damage.start : damage.stop, damage.start : damage.stop])
    context = _context([FieldBlock(damage.name, 0, damage.stop - damage.start, 1)])
    with pytest.raises(ValueError, match="2D or 3D displacement field"):
        TwoLevel(RigidBodyModes()).setUp(block, context)
    with pytest.raises(ValueError, match="prolongation"):
        RigidBodyModes(prolongation="smooth")


def test_the_default_trees_on_the_reference_page_are_the_code_defaults():
    import json
    import pathlib

    from edelweissfe.linsolve.composition.amg import defaultFieldTree

    page = (pathlib.Path(__file__).parents[1] / "doc/source/documentation/composedlinsolversreference.rst").read_text()
    block = page.split("and change only what is intended. The defaults are", 1)[1].split(".. code-block:: json", 1)[1]
    trees = json.JSONDecoder().raw_decode(block.strip())[0]
    assert trees == {
        "3D vector field (e.g. displacement)": defaultFieldTree(3),
        "2D vector field": defaultFieldTree(2),
        "vector field of a dimension other than 2 or 3": defaultFieldTree(4),
        "scalar field (e.g. nonlocal damage)": defaultFieldTree(1),
    }
