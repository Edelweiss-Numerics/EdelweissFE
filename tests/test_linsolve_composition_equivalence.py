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
    AMG,
    LGMRES,
    ComposedLinearSolver,
    DegradationDumps,
    DiagonalScaling,
    DirectSolver,
    FixedTolerance,
    GapCompensation,
    Multiplicative,
    OnField,
    SchurComplement,
    blockAMG,
)
from edelweissfe.linsolve.composition.amg import droppedSmallEntries  # noqa: E402


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


def _pair(fields, coordinates, blockamgOptions: dict, presetOptions: dict, **driverOptions):
    """BlockAMGSolver with ``blockamgOptions`` and the composed ``blockAMG(**presetOptions)``."""
    reference = BlockAMGSolver(verbosity="silent", nodeCoordinates={"displacement": coordinates}, **blockamgOptions)
    reference.setFieldStructure(fields)
    composed = ComposedLinearSolver(
        blockAMG(**presetOptions), nodeCoordinates={"displacement": coordinates}, verbosity="silent", **driverOptions
    )
    composed.setFieldStructure(fields)
    return reference, composed


def _solverPair(fields, coordinates, rigidBodyCoarseLevel: bool, outerTol=None, sweeps=1, symmetric=False):
    return _pair(
        fields,
        coordinates,
        dict(
            outerTol=outerTol,
            rigidBodyCoarseLevel=None if rigidBodyCoarseLevel else False,
            sweeps=sweeps,
            symmetric=symmetric,
        ),
        dict(
            rigidBodyCoarseLevel=rigidBodyCoarseLevel,
            tolerance=FixedTolerance(outerTol) if outerTol is not None else None,
            sweeps=sweeps,
            symmetric=symmetric,
        ),
    )


def _assertSameSolve(reference, composed, A, b):
    x_reference = reference(A, b)
    x_composed = composed(A, b)
    assert composed.lastSolveSummary.iters == reference.lastSolveSummary.iters
    assert np.linalg.norm(x_composed - x_reference) <= 1e-9 * np.linalg.norm(x_reference)
    print("outer iterations", composed.lastSolveSummary.iters, "rebuilt:", composed.root.solveRecord().rebuildReason)
    return composed.lastSolveSummary.iters


@pytest.mark.parametrize("rigidBodyCoarseLevel", [True, False])
def test_one_solve_matches_blockamg(rigidBodyCoarseLevel):
    A, b, fields, coordinates = _beamWithDamage()
    reference, composed = _solverPair(fields, coordinates, rigidBodyCoarseLevel, outerTol=1e-8)
    iterations = _assertSameSolve(reference, composed, A, b)
    assert 1 < iterations < 100


@pytest.mark.parametrize("sweeps, symmetric", [(2, False), (1, True), (2, True)])
def test_several_and_symmetric_sweeps_match_blockamg(sweeps, symmetric):
    A, b, fields, coordinates = _beamWithDamage()
    reference, composed = _solverPair(fields, coordinates, False, outerTol=1e-8, sweeps=sweeps, symmetric=symmetric)
    _assertSameSolve(reference, composed, A, b)


@pytest.mark.parametrize("rigidBodyCoarseLevel", [True, False])
def test_a_sequence_of_solves_matches_blockamg_with_its_reuse_decisions(rigidBodyCoarseLevel):
    """Forcing tolerance, hierarchy reuse: new values only (reused), a shrinking right-hand side (Newton converging),
    a jump of the right-hand side (a new increment: rebuilt), and a new sparsity pattern (rebuilt)."""
    A, b, fields, coordinates = _beamWithDamage()
    reference, composed = _solverPair(fields, coordinates, rigidBodyCoarseLevel)
    rebuilds = []
    for A_k, b_k in _newtonLikeSequence(A, b, fields):
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


# ----- hierarchyDropTol, hierarchyDropLumping -> AMG/TwoGrid(dropTolerance, lumpDroppedEntries) -----


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
@pytest.mark.parametrize("rigidBodyCoarseLevel", [True, False])
def test_a_drop_tolerance_matches_blockamgs_hierarchy_drop_tolerance(rigidBodyCoarseLevel, lump):
    A, b, fields, coordinates = _beamWithDamage()
    reference, composed = _pair(
        fields,
        coordinates,
        dict(
            outerTol=1e-8,
            rigidBodyCoarseLevel=None if rigidBodyCoarseLevel else False,
            hierarchyDropTol=0.01,
            hierarchyDropLumping=lump,
        ),
        dict(
            tolerance=FixedTolerance(1e-8),
            rigidBodyCoarseLevel=rigidBodyCoarseLevel,
            dropTolerance=0.01,
            lumpDroppedEntries=lump,
        ),
    )
    _assertSameSolve(reference, composed, A, b)
    displacement = composed.root.preconditioner.inner.corrections[0].inner
    assert displacement.statistics.sizes["nnz kept"] < displacement.statistics.sizes["nnz"]


# ----- krylovOnScaledSystem, trueResidualMaxContinuations, gapCompensatedTolerance -> DiagonalScaling(LGMRES) -----


@pytest.mark.parametrize("gapCompensated", [False, True])
def test_a_krylov_solve_on_the_scaled_system_matches_blockamg_along_a_sequence(gapCompensated):
    """The tree DiagonalScaling(LGMRES(...)): the same outer iterations, continuations, solutions and rebuilds as
    blockamg's krylovOnScaledSystem, with and without the gap compensation (learnt across the solves)."""
    A, b, fields, coordinates = _beamWithDamage()
    reference, composed = _pair(
        fields,
        coordinates,
        dict(krylovOnScaledSystem=True, gapCompensatedTolerance=gapCompensated, rigidBodyCoarseLevel=False),
        dict(
            krylovOnScaledSystem=True,
            gapCompensation=GapCompensation() if gapCompensated else None,
            rigidBodyCoarseLevel=False,
        ),
    )
    continuations = []
    for A_k, b_k in _newtonLikeSequence(A, b, fields):
        _assertSameSolve(reference, composed, A_k, b_k)
        assert composed.lastSolveSummary.retries == reference.lastSolveSummary.retries
        assert composed.lastSolveSummary.residualMet == reference.lastSolveSummary.residualMet
        continuations.append(composed.lastSolveSummary.retries)
    if gapCompensated:
        assert composed.root.inner.gapCompensation.gap == pytest.approx(reference._trueResidualGap, rel=1e-6)
    print("continuations", continuations)


def test_no_continuations_on_the_scaled_system_matches_blockamg():
    A, b, fields, coordinates = _beamWithDamage()
    reference, composed = _pair(
        fields,
        coordinates,
        dict(krylovOnScaledSystem=True, trueResidualMaxContinuations=0, rigidBodyCoarseLevel=False),
        dict(krylovOnScaledSystem=True, continuations=0, rigidBodyCoarseLevel=False),
    )
    _assertSameSolve(reference, composed, A, b)
    assert composed.lastSolveSummary.retries == 0
    assert composed.lastSolveSummary.residualMet == reference.lastSolveSummary.residualMet


# ----- dumpOnDegradation* -> ComposedLinearSolver(degradationDumps=DegradationDumps(...)) -----


def _manifest(directory) -> list:
    import json

    with open(directory / "manifest.jsonl") as manifestFile:
        return [json.loads(line) for line in manifestFile]


def test_the_degradation_dumps_are_blockamgs_on_disk(tmp_path, monkeypatch):
    """The same files and manifest lines as blockamg's, for every solve of a sequence (threshold 0: all dumped)."""
    A, b, fields, coordinates = _beamWithDamage()
    monkeypatch.setattr(DegradationDumps, "dumpsWritten", 0)
    reference, composed = _pair(
        fields,
        coordinates,
        dict(
            dumpOnDegradationDir=str(tmp_path / "blockamg"), dumpOnDegradationThreshold=0, dumpOnDegradationMaxDumps=20
        ),
        {},
        degradationDumps=DegradationDumps(str(tmp_path / "composed"), threshold=0, maxDumps=20),
    )
    sequence = _newtonLikeSequence(A, b, fields)
    for A_k, b_k in sequence:
        _assertSameSolve(reference, composed, A_k, b_k)
    references, records = _manifest(tmp_path / "blockamg"), _manifest(tmp_path / "composed")
    assert len(records) == len(references) == len(sequence)
    for (A_k, b_k), reference, record in zip(sequence, references, records):
        assert set(record) == set(reference)
        for key in reference:
            if key == "trueResidual":
                assert record[key] == pytest.approx(reference[key], rel=1e-3)
            elif key not in ("matrixFile", "rhsFile"):
                assert record[key] == reference[key], key
        assert record["matrixFile"] == "A_{:02d}_{:05d}.npz".format(composed._instanceOrdinal, record["solveCount"])
        assert abs(sp.load_npz(tmp_path / "composed" / record["matrixFile"]) - A_k).max() == 0.0
        assert np.array_equal(np.load(tmp_path / "composed" / record["rhsFile"]), b_k)


def test_a_dump_of_a_condensed_solve_holds_its_condensed_dofs(tmp_path, monkeypatch):
    from test_linsolve_composition_components import _barOnFoundation, _FixedDofs

    monkeypatch.setattr(DegradationDumps, "dumpsWritten", 0)
    A, b, barDofs = _barOnFoundation(n=8)
    fields = [FieldBlock("u", 0, A.shape[0], 1)]
    solveA_bb = DirectSolver()
    root = LGMRES(
        SchurComplement(
            preconditionS=DiagonalScaling(Multiplicative([OnField("u", AMG())])),
            solveA_bb=solveA_bb,
            condensation=_FixedDofs(barDofs, solveA_bb.newSolver),
        ),
        tolerance=FixedTolerance(1e-8),
    )
    composed = ComposedLinearSolver(
        root, verbosity="silent", degradationDumps=DegradationDumps(str(tmp_path), threshold=0)
    )
    composed.setFieldStructure(fields)
    composed(A, b)
    (record,) = _manifest(tmp_path)
    assert np.array_equal(np.load(tmp_path / record["condensedDofsFile"]), barDofs)


def test_a_dump_holds_the_context_solves_before_its_trigger(tmp_path, monkeypatch):
    monkeypatch.setattr(DegradationDumps, "dumpsWritten", 0)
    A, b, fields, _ = _beamWithDamage()
    dumps = DegradationDumps(str(tmp_path), maxDumps=10, contextSolves=2)
    for solveCount, degraded in ((1, False), (2, False), (3, False), (4, True)):
        dumps.afterSolve(0, solveCount, A, solveCount * b, fields, {}, None, degraded, lambda level, message: None)
    records = _manifest(tmp_path)
    assert [(record["solveCount"], record["role"]) for record in records] == [
        (4, "trigger"),
        (2, "context"),
        (3, "context"),
    ]
    assert np.array_equal(np.load(tmp_path / records[1]["rhsFile"]), 2 * b)
