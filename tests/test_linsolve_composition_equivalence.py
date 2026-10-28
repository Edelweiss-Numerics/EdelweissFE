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
"""The composed ``blockAMG()`` preset against the numbers of ``BlockAMGSolver``, the monolithic solver it replaced.

``BlockAMGSolver`` was deleted once the composition matched it (outer iterations, rebuild decisions, solutions and
degradation dumps, on these systems and on the workbench dumps). Its numbers on the systems below were recorded before
the deletion (xeon, branch ``feat/linsolve-composition`` at ``ffe4b681``) and frozen here. When the spectral radius
estimate changed from AMGCL's power iteration to Arnoldi steps (``amgcl-spectral-radius.hpp``), the numbers were
recorded again, with the composed solver (xeon, ``OMP_NUM_THREADS`` 16 and 1: the same iterations, which no longer
depend on the thread count); the rebuild decisions stayed BlockAMGSolver's. Per solve: the outer iterations, whether
the preconditioner was rebuilt, the true-residual continuations, whether the tolerance was met, and ``||x||``.

``||x||`` is compared to 1e-8 for a solve to ``1e-8``; a solve to the forcing tolerance (eta about 1e-3) leaves an
error that depends on the order of threaded reductions, so its ``||x||`` is compared to 1e-4, and its true residual is
checked against the tolerance instead."""

import json

import numpy as np
import pytest
import scipy.sparse as sp
from _elasticbeam import cantilever

pytest.importorskip("edelweissfe.linsolve.amgcl.amgcl")

from edelweissfe.linsolve.base import FieldBlock  # noqa: E402
from edelweissfe.linsolve.composition import (  # noqa: E402
    AMG,
    LGMRES,
    ComposedLinearSolver,
    DiagonalScaling,
    FixedTolerance,
    GapCompensation,
    Multiplicative,
    OnField,
    RebuildPolicy,
    SchurComplement,
    SolveDumps,
    blockAMG,
)
from edelweissfe.linsolve.composition.amg import droppedSmallEntries  # noqa: E402

#: The numbers of the composed solver, per solve: (outer iterations, continuations, tolerance met, ||x||).
_TIGHT = {
    "rigidBodyModesPerAggregate": [(13, 0, True, 418.5446467174296)],
    "translations": [(40, 0, True, 418.54464670055614)],
    "sweeps2": [(26, 0, True, 418.54464656671)],
    "symmetric": [(30, 0, True, 418.5446466851319)],
    "sweeps2symmetric": [(20, 0, True, 418.54464669228247)],
    "drop rigidBodyModesPerAggregate": [(25, 0, True, 418.54464673746827)],
    "drop rigidBodyModesPerAggregate lumped": [(25, 0, True, 418.54464678508566)],
    "drop translations": [(42, 0, True, 418.5446466908262)],
    "drop translations lumped": [(42, 0, True, 418.5446467190722)],
}
_SEQUENCE = {
    "rigidBodyModesPerAggregate": [
        (8, 0, True, 418.54618977034244),
        (50, 0, True, 41.32241300015936),
        (16, 0, True, 5.624814130343251),
        (8, 0, True, 4185.461897520253),
        (8, 0, True, 418.54618990569605),
    ],
    "translations": [
        (27, 0, True, 418.53900395601215),
        (59, 0, True, 41.32246698054644),
        (63, 0, True, 5.625213216680407),
        (27, 0, True, 4185.3900395600285),
        (27, 0, True, 418.5390041105333),
    ],
    "scaled": [
        (37, 1, True, 418.5379846987327),
        (84, 1, True, 41.32191911886246),
        (84, 1, True, 5.624716436920195),
        (37, 1, True, 4185.379846987914),
        (37, 1, True, 418.53798485533997),
    ],
    "scaled gapCompensated": [
        (32, 1, True, 418.53492930097934),
        (65, 0, True, 41.322211355387054),
        (68, 0, True, 5.625254935072965),
        (30, 0, True, 4185.440747574256),
        (30, 0, True, 418.5440749131303),
    ],
}
#: Whether BlockAMGSolver rebuilt its preconditioner, per solve of the sequence (its dumps' ``mustRefresh``).
_REBUILT = {
    "rigidBodyModesPerAggregate": [True, False, True, True, True],
    "translations": [True, False, False, True, True],
}
#: The learnt gap between the scaled and the true residual after the sequence.
_GAP = {"scaled": 19.24, "scaled gapCompensated": 19.26}


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


def _composed(coordinates, fields, **presetOptions) -> ComposedLinearSolver:
    composed = ComposedLinearSolver(
        blockAMG(rebuild=RebuildPolicy(reuse=True), **presetOptions),
        nodeCoordinates={"displacement": coordinates},
        verbosity="silent",
    )
    composed.setFieldStructure(fields)
    return composed


def _assertSolve(composed, A, b, frozen, tight: bool):
    """One solve of ``composed`` against BlockAMGSolver's frozen ``(iterations, continuations, met, ||x||)``."""
    iterations, continuations, met, norm = frozen
    x = composed(A, b)
    summary = composed.lastSolveSummary
    assert summary.iters in (iterations if isinstance(iterations, tuple) else (iterations,))
    assert summary.retries == continuations
    assert summary.residualMet == met
    assert np.linalg.norm(x) == pytest.approx(norm, rel=1e-8 if tight else 1e-4)
    if met:
        tolerance = 1e-8 if tight else composed.root.solveRecord().eta
        assert np.linalg.norm(b - A @ x) <= 1.01 * tolerance * np.linalg.norm(b)
    return x


@pytest.mark.parametrize("rigidBodyModesPerAggregate", [True, False])
def test_one_solve_has_blockamgs_numbers(rigidBodyModesPerAggregate):
    A, b, fields, coordinates = _beamWithDamage()
    composed = _composed(
        coordinates, fields, rigidBodyModesPerAggregate=rigidBodyModesPerAggregate, tolerance=FixedTolerance(1e-8)
    )
    (frozen,) = _TIGHT["rigidBodyModesPerAggregate" if rigidBodyModesPerAggregate else "translations"]
    _assertSolve(composed, A, b, frozen, tight=True)
    assert 1 < composed.lastSolveSummary.iters < 100


@pytest.mark.parametrize(
    "sweeps, symmetric, name", [(2, False, "sweeps2"), (1, True, "symmetric"), (2, True, "sweeps2symmetric")]
)
def test_several_and_symmetric_sweeps_have_blockamgs_numbers(sweeps, symmetric, name):
    A, b, fields, coordinates = _beamWithDamage()
    composed = _composed(
        coordinates,
        fields,
        rigidBodyModesPerAggregate=False,
        tolerance=FixedTolerance(1e-8),
        sweeps=sweeps,
        symmetric=symmetric,
    )
    (frozen,) = _TIGHT[name]
    _assertSolve(composed, A, b, frozen, tight=True)


@pytest.mark.parametrize("rigidBodyModesPerAggregate", [True, False])
def test_a_sequence_of_solves_has_blockamgs_numbers_and_reuse_decisions(rigidBodyModesPerAggregate):
    """Forcing tolerance, hierarchy reuse: new values only (reused), a shrinking right-hand side (Newton converging),
    a jump of the right-hand side (a new increment: rebuilt), and a new sparsity pattern (rebuilt)."""
    A, b, fields, coordinates = _beamWithDamage()
    name = "rigidBodyModesPerAggregate" if rigidBodyModesPerAggregate else "translations"
    composed = _composed(coordinates, fields, rigidBodyModesPerAggregate=rigidBodyModesPerAggregate)
    rebuilds = []
    for (A_k, b_k), frozen in zip(_newtonLikeSequence(A, b, fields), _SEQUENCE[name]):
        _assertSolve(composed, A_k, b_k, frozen, tight=False)
        rebuilds.append(composed.root.lastRebuildReason)
    assert [reason is not None for reason in rebuilds] == _REBUILT[name]
    assert rebuilds[0] == "first" and rebuilds[1] is None and rebuilds[3] == "increment" and rebuilds[4] == "pattern"


def test_a_solver_keeps_no_copy_of_the_last_matrix_between_solves():
    """Like blockamg, which dropped its field split after every solve: between solves (while the nonlinear solver
    assembles the next matrix) only what can be reused is kept, i.e. the preconditioners, not the field blocks of the
    last matrix (on a million dofs, gigabytes)."""
    A, b, fields, coordinates = _beamWithDamage()
    composed = _composed(coordinates, fields)
    composed(A, b)
    lgmres = composed.root
    blockGaussSeidel = lgmres.preconditioner.inner
    assert lgmres.A is None and blockGaussSeidel.blocks is None
    # and the next solve with new values only still reuses the preconditioners
    composed((1.0001 * A).tocsr(), b)
    assert lgmres.lastRebuildReason is None


def _newtonLikeSequence(A, b, fields, seed: int = 7):
    """New values (reused), a shrinking right-hand side, a jump (a new increment) and a new sparsity pattern."""
    rng = np.random.default_rng(seed)
    perturbation = sp.diags(1.0 + 0.05 * rng.standard_normal(A.shape[0]))
    first, last = fields[1].start, fields[1].stop - 1
    newEntries = sp.csr_matrix(([1e-6, 1e-6], ([first, last], [last, first])), shape=A.shape)
    sequence = [
        (A, b),
        ((perturbation @ A @ perturbation).tocsr(), 0.1 * b),
        ((perturbation @ A @ perturbation).tocsr(), 0.01 * rng.standard_normal(A.shape[0])),
        (A, 10.0 * b),
        ((A + newEntries).tocsr(), b),
    ]
    for A_k, _ in sequence:
        A_k.sort_indices()
    return sequence


# ----- AMG/TwoLevel(dropTolerance, lumpDroppedEntries) -----


def test_the_filtered_matrix_keeps_the_diagonal_symmetry_and_with_lumping_the_row_sums():
    A, _, _, _ = _beamWithDamage()
    K = A[: A.shape[0] * 3 // 4, : A.shape[0] * 3 // 4].tocsr()
    K = (K + K.T).tocsr()
    filtered = droppedSmallEntries(K, 0.1)
    assert 0 < filtered.nnz < K.nnz
    assert np.array_equal(filtered.diagonal(), K.diagonal())
    assert abs(filtered - filtered.T).max() == 0.0
    lumped = droppedSmallEntries(K, 0.1, lumpDroppedEntries=True)
    ones = np.ones(K.shape[0])
    assert np.allclose(lumped @ ones, K @ ones, rtol=0.0, atol=1e-12 * abs(K).max())


@pytest.mark.parametrize("lump", [False, True])
@pytest.mark.parametrize("rigidBodyModesPerAggregate", [True, False])
def test_a_drop_tolerance_has_blockamgs_numbers(rigidBodyModesPerAggregate, lump):
    A, b, fields, coordinates = _beamWithDamage()
    composed = _composed(
        coordinates,
        fields,
        tolerance=FixedTolerance(1e-8),
        rigidBodyModesPerAggregate=rigidBodyModesPerAggregate,
        dropTolerance=0.01,
        lumpDroppedEntries=lump,
    )
    name = (
        "drop "
        + ("rigidBodyModesPerAggregate" if rigidBodyModesPerAggregate else "translations")
        + (" lumped" if lump else "")
    )
    (frozen,) = _TIGHT[name]
    _assertSolve(composed, A, b, frozen, tight=True)
    displacement = composed.root.preconditioner.inner.corrections[0].inner
    assert displacement.statistics.sizes["nnz kept"] < displacement.statistics.sizes["nnz"]


# ----- DiagonalScaling(LGMRES(...)): continuations and the gap compensation -----


@pytest.mark.parametrize("gapCompensated", [False, True])
def test_a_krylov_solve_on_the_scaled_system_has_blockamgs_numbers_along_a_sequence(gapCompensated):
    """The tree DiagonalScaling(LGMRES(...)): the outer iterations, continuations and solutions of blockamg's
    krylovOnScaledSystem, with and without the gap compensation (learnt across the solves)."""
    A, b, fields, coordinates = _beamWithDamage()
    name = "scaled gapCompensated" if gapCompensated else "scaled"
    composed = _composed(
        coordinates,
        fields,
        krylovOnScaledSystem=True,
        gapCompensation=GapCompensation() if gapCompensated else None,
        rigidBodyModesPerAggregate=False,
    )
    for (A_k, b_k), frozen in zip(_newtonLikeSequence(A, b, fields), _SEQUENCE[name]):
        _assertSolve(composed, A_k, b_k, frozen, tight=False)
    if gapCompensated:
        assert composed.root.inner.gapCompensation.gap == pytest.approx(_GAP[name], rel=1e-2)


def test_no_continuations_on_the_scaled_system_has_blockamgs_numbers():
    A, b, fields, coordinates = _beamWithDamage()
    composed = _composed(
        coordinates, fields, krylovOnScaledSystem=True, continuations=0, rigidBodyModesPerAggregate=False
    )
    _assertSolve(composed, A, b, (19, 0, False, 418.3153141552291), tight=False)


# ----- the degradation dumps -----


def _manifest(directory) -> list:
    with open(directory / "manifest.jsonl") as manifestFile:
        return [json.loads(line) for line in manifestFile]


#: BlockAMGSolver's manifest lines for the sequence (threshold 0: every solve dumped), without the file names and the
#: true residual: (solveCount, iterations, rebuilt, newIncrement, patternChanged, nnz, previousIterations).
_MANIFEST = [
    (1, 8, True, False, False, 670422, None),
    (2, 50, False, False, False, 670422, 8),
    (3, 16, True, False, False, 670422, 50),
    (4, 8, True, True, False, 670422, 16),
    (5, 8, True, False, True, 670424, 8),
]
_MANIFEST_KEYS = {
    "blocks",
    "capHit",
    "contextOffset",
    "continuations",
    "eta",
    "geometryFile",
    "iterations",
    "matrixFile",
    "newIncrement",
    "nnz",
    "patternChanged",
    "previousIterations",
    "rebuildReason",
    "residual",
    "rhsFile",
    "role",
    "rows",
    "scaledSystem",
    "solveCount",
    "triggerSolveCount",
}


def test_the_degradation_dumps_are_blockamgs_on_disk(tmp_path):
    """The files and manifest lines blockamg wrote, for every solve of a sequence (threshold 0: all dumped)."""
    A, b, fields, coordinates = _beamWithDamage()
    composed = ComposedLinearSolver(
        blockAMG(),
        nodeCoordinates={"displacement": coordinates},
        verbosity="silent",
        dumps=SolveDumps(str(tmp_path), threshold=0, maxDumps=20),
    )
    composed.setFieldStructure(fields)
    sequence = _newtonLikeSequence(A, b, fields)
    for A_k, b_k in sequence:
        composed(A_k, b_k)
    records = _manifest(tmp_path)
    assert len(records) == len(sequence)
    for (A_k, b_k), frozen, record in zip(sequence, _MANIFEST, records):
        assert set(record) == _MANIFEST_KEYS
        solveCount, iterations, rebuilt, newIncrement, patternChanged, nnz, previousIterations = frozen
        assert (record["solveCount"], record["iterations"], record["rebuildReason"] is not None) == (
            solveCount,
            iterations,
            rebuilt,
        )
        assert (record["newIncrement"], record["patternChanged"], record["nnz"]) == (newIncrement, patternChanged, nnz)
        assert record["previousIterations"] == previousIterations
        assert (record["role"], record["triggerSolveCount"], record["contextOffset"]) == ("trigger", solveCount, 0)
        assert (record["rows"], record["continuations"], record["capHit"], record["eta"]) == (12740, 0, False, 0.001)
        assert record["blocks"] == [
            {"name": "displacement", "start": 0, "stop": 9555, "dimension": 3},
            {"name": "nonlocal damage", "start": 9555, "stop": 12740, "dimension": 1},
        ]
        assert record["residual"] <= record["eta"]
        # the geometry a replay of the rigid-body coarse level needs
        geometry = np.load(tmp_path / record["geometryFile"])
        assert list(geometry) == ["displacement_coordinates"]
        assert np.array_equal(geometry["displacement_coordinates"], coordinates)
        assert record["matrixFile"] == "A_{:02d}_{:05d}.npz".format(composed._instanceOrdinal, record["solveCount"])
        assert abs(sp.load_npz(tmp_path / record["matrixFile"]) - A_k).max() == 0.0
        assert np.array_equal(np.load(tmp_path / record["rhsFile"]), b_k)


def test_a_dump_of_a_condensed_solve_holds_its_condensed_dofs(tmp_path):
    from _linsolvesystems import barOnFoundation, condensingGivenDofs

    A, b, barDofs = barOnFoundation(n=8)
    fields = [FieldBlock("u", 0, A.shape[0], 1)]
    root = LGMRES(
        condensingGivenDofs(
            SchurComplement(["bars"], preconditionS=DiagonalScaling(Multiplicative([OnField("u", AMG())]))),
            barDofs,
        ),
        tolerance=FixedTolerance(1e-8),
    )
    composed = ComposedLinearSolver(root, verbosity="silent", dumps=SolveDumps(str(tmp_path), threshold=0))
    composed.setFieldStructure(fields)
    composed(A, b)
    (record,) = _manifest(tmp_path)
    assert np.array_equal(np.load(tmp_path / record["condensedDofsFile"]), barDofs)


def test_a_dump_holds_the_context_solves_before_its_trigger(tmp_path):
    A, b, fields, _ = _beamWithDamage()
    dumps = SolveDumps(str(tmp_path), maxDumps=10, contextSolves=2)
    for solveCount, degraded in ((1, False), (2, False), (3, False), (4, True)):
        dumps.afterSolve(0, solveCount, A, solveCount * b, fields, {}, None, {}, degraded, lambda level, message: None)
    records = _manifest(tmp_path)
    assert [(record["solveCount"], record["role"]) for record in records] == [
        (4, "trigger"),
        (2, "context"),
        (3, "context"),
    ]
    assert np.array_equal(np.load(tmp_path / records[1]["rhsFile"]), 2 * b)
