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
"""``blockTranslationNullspace``, on in the default 2D/3D trees: smoothed aggregation on a block backend whose
tentative prolongation is built from the translations D^-1 t of the equilibrated block D A D
(``amgcl-block-translations.hpp``), instead of AMGCL's constant blocks, which are D-weighted translations."""

import copy

import numpy as np
import pytest
import scipy.sparse as sp
from _blockamgsystems import (
    NODAL_COUPLING,
    displacementDamageSystem,
    gridLaplacian,
    hierarchyLevels,
    variableCoefficientLaplacian,
)

pytest.importorskip("edelweissfe.linsolve.amgcl.amgcl")

import edelweissfe.linsolve.amgcl.amgcl as amgclModule  # noqa: E402
from edelweissfe.linsolve.amgcl.amgcl import PyAMGCLSolver  # noqa: E402
from edelweissfe.linsolve.base import FieldBlock  # noqa: E402
from edelweissfe.linsolve.blockamg import createSolver  # noqa: E402
from edelweissfe.linsolve.blockamg.blockamg import _defaultFieldPrecond  # noqa: E402
from edelweissfe.linsolve.nullspace import translationNullspace  # noqa: E402

#: Every block backend blockamg builds, as (nodal dimension = block size, backendPrecision, nodes per grid
#: direction): the grids give a hierarchy of exactly two levels (AMGCL solves directly below 3000 scalar rows).
_BLOCK_BACKENDS = [(3, "double", 12), (3, "float", 12), (2, "double", 40)]
_BLOCK_BACKEND_IDS = ["3x3 double", "3x3 float", "2x2 double"]


def _scaledVectorLaplacian(dimension: int, n: int):
    """A vector Laplacian kron(L + 0.01 I, C), C the nodal coupling of ``dimension`` components, on a grid of n nodes
    per direction, scaled on both sides by a random diagonal spanning two orders of magnitude: its diagonal varies
    from DOF to DOF, within a node too, so the constant blocks of its equilibrated block are far from the
    translations of that block. Returns the matrix (CSR, sorted columns) and its field block."""
    nodes = n**dimension
    stiffness = sp.kron(gridLaplacian(n, dimension) + 0.01 * sp.identity(nodes), NODAL_COUPLING[:dimension, :dimension])
    scaling = sp.diags(10.0 ** np.random.default_rng(3).uniform(0.0, 2.0, dimension * nodes))
    A = (scaling @ stiffness @ scaling).tocsr()
    A.sort_indices()
    return A, FieldBlock("displacement", 0, dimension * nodes, dimension)


def _heterogeneousVectorField(dimension: int, n: int):
    """A vector field kron(K + 0.01 I, C) on a grid of n nodes per direction, K the variable-coefficient Laplacian
    of edge stiffnesses spanning two orders of magnitude, C the nodal coupling of ``dimension`` components. Unlike
    :func:`_scaledVectorLaplacian`, its near-null-space is the translations t, as for a finite element stiffness of a
    heterogeneous material, while its diagonal varies from node to node: the constant blocks of its equilibrated
    block D A D are D-weighted translations, and D^-1 t is the right coarse space. Returns the matrix (CSR, sorted
    columns) and its field block."""
    nodes = n**dimension
    stiffness = variableCoefficientLaplacian(n, dimension) + 0.01 * sp.identity(nodes)
    A = sp.kron(stiffness, NODAL_COUPLING[:dimension, :dimension]).tocsr()
    A.sort_indices()
    return A, FieldBlock("displacement", 0, dimension * nodes, dimension)


def _equilibrated(A):
    """D A D with D = diag(|a_ii|^-1/2), as blockamg equilibrates, and the diagonal of D."""
    scaling = 1.0 / np.sqrt(np.abs(A.diagonal()))
    D = sp.diags(scaling)
    scaled = (D @ A @ D).tocsr()
    scaled.sort_indices()
    return scaled, scaling


def _hierarchy(A, dimension: int, precision: str, precond: dict, nullspace=None):
    """An AMGCL hierarchy for A on the block backend, with the near null-space ``nullspace`` if given."""
    hierarchy = PyAMGCLSolver(
        {
            "precond": copy.deepcopy(precond),
            "solver": {"type": "preonly"},
            "backendBlockSize": dimension,
            "backendPrecision": precision,
        }
    )
    if nullspace is not None:
        hierarchy.set_nullspace(nullspace)
    hierarchy.build(A)
    return hierarchy


def _relativeDifference(x, y) -> float:
    return np.linalg.norm(x - y) / np.linalg.norm(y)


@pytest.mark.parametrize("dimension, precision, n", _BLOCK_BACKENDS, ids=_BLOCK_BACKEND_IDS)
def test_the_translations_of_d_a_d_give_the_hierarchy_of_a_on_its_constant_blocks(dimension, precision, n):
    # The defining property of the tentative prolongation: its coarse space holds the translations D^-1 t of
    # D A D exactly, component by component. With the same aggregates (eps_strong 0: every connection is strong,
    # whatever the scaling), the same prolongation damping (no spectral-radius estimate) and a block-Jacobi
    # smoother, which is invariant under the scaling, the hierarchy of D A D on D^-1 t is then the hierarchy of A
    # on its constant blocks t in other variables, M^-1 = D^-1 AMG(A)^-1 D^-1, to rounding. The per-aggregate
    # normalization only scales the coarse variables, which the direct solve on the coarse level does not see.
    A, block = _scaledVectorLaplacian(dimension, n)
    scaled, scaling = _equilibrated(A)
    precond = {
        "coarsening": {"type": "smoothed_aggregation", "aggr": {"eps_strong": 0.0}},
        "relax": {"type": "damped_jacobi"},
    }
    # the reference in double: A is far worse conditioned than D A D, which rounding to float would show
    onA = _hierarchy(A, dimension, "double", precond)
    withTranslations = _hierarchy(scaled, dimension, precision, precond, translationNullspace(block, scaling))
    withConstantBlocks = _hierarchy(scaled, dimension, precision, precond)
    assert hierarchyLevels(onA) == hierarchyLevels(withTranslations) == 2

    r = np.random.default_rng(5).standard_normal(A.shape[0])
    # one cycle of the hierarchy of A as a preconditioner of D A D: (D A D)^-1 = D^-1 A^-1 D^-1
    expected = onA.applyPreconditioner(r / scaling) / scaling
    # float: the level matrices of D A D are rounded (set up in double either way)
    tolerance = 1e-4 if precision == "float" else 1e-7
    assert _relativeDifference(withTranslations.applyPreconditioner(r), expected) < tolerance
    # and the test can tell: AMGCL's constant blocks of D A D are another coarse space
    assert _relativeDifference(withConstantBlocks.applyPreconditioner(r), expected) > 1e-3


@pytest.mark.parametrize("dimension, precision, n", _BLOCK_BACKENDS, ids=_BLOCK_BACKEND_IDS)
def test_unit_translation_weights_give_amgcls_constant_blocks(dimension, precision, n):
    # Weights 1 make the tentative prolongation AMGCL's constant blocks, normalized per aggregate: on a hierarchy of
    # two levels (the coarse one a direct solve) that is the same preconditioner to rounding, here with the
    # vector-field default tree. An (n, 0) near null-space clears the translations again.
    scaled, _ = _equilibrated(_scaledVectorLaplacian(dimension, n)[0])
    block = FieldBlock("displacement", 0, scaled.shape[0], dimension)
    precond = {
        key: value
        for key, value in _defaultFieldPrecond(dimension).items()
        if key not in ("backendPrecision", "backendBlockSize")
    }
    constantBlocks = _hierarchy(scaled, dimension, precision, precond)
    unitWeights = _hierarchy(
        scaled, dimension, precision, precond, translationNullspace(block, np.ones(scaled.shape[0]))
    )
    assert hierarchyLevels(constantBlocks) == hierarchyLevels(unitWeights) == 2

    cleared = PyAMGCLSolver(
        {
            "precond": precond,
            "solver": {"type": "preonly"},
            "backendBlockSize": dimension,
            "backendPrecision": precision,
        }
    )
    cleared.set_nullspace(translationNullspace(block, np.ones(scaled.shape[0])))
    cleared.set_nullspace(np.zeros((scaled.shape[0], 0)))
    cleared.build(scaled)

    r = np.random.default_rng(6).standard_normal(scaled.shape[0])
    expected = constantBlocks.applyPreconditioner(r)
    # float: the prolongations are rounded, and they differ by the normalization
    tolerance = 1e-4 if precision == "float" else 1e-9
    assert _relativeDifference(unitWeights.applyPreconditioner(r), expected) < tolerance
    assert _relativeDifference(cleared.applyPreconditioner(r), expected) < 1e-12


def _recordNullspaces(monkeypatch) -> list:
    """Record every near null-space blockamg hands to an AMGCL hierarchy, as (block size, array)."""
    recorded = []

    class RecordingSolver(PyAMGCLSolver):
        def set_nullspace(self, B):
            recorded.append((self.blockSize, np.array(B)))
            super().set_nullspace(B)

    monkeypatch.setattr(amgclModule, "PyAMGCLSolver", RecordingSolver)
    return recorded


def _displacementSystem(dimension: int):
    """A system with a displacement field of ``dimension`` components, as (A, b, field blocks): in 3D the
    displacement / damage system of the suite, the displacement rows scaled over two orders of magnitude; in 2D a
    heterogeneous vector field alone (:func:`_heterogeneousVectorField`), whose near-null-space the translations
    are, so they must not cost iterations there."""
    if dimension == 3:
        return displacementDamageSystem(12, coupling="symmetric", rowScaling=True)
    A, displacement = _heterogeneousVectorField(2, 40)
    return A, np.random.default_rng(8).standard_normal(A.shape[0]), [displacement]


def _solveWithAndWithoutTranslations(monkeypatch, A, b, blocks, fieldTree):
    """Solve with blockamg twice, with ``fieldTree`` for the displacement field and ``blockTranslationNullspace``
    switched off and on. Returns the second solver, the near null-spaces it handed over, and the outer iterations of
    both solves (constant blocks, translations)."""
    iterations = []
    for translations in (False, True):
        recorded = _recordNullspaces(monkeypatch)
        tree = {**copy.deepcopy(fieldTree), "blockTranslationNullspace": translations}
        solver = createSolver({"fieldPreconds": {"displacement": tree}, "verbosity": "silent", "outerTol": 1e-8})
        solver.setFieldStructure(blocks)
        x = solver(A, b)
        assert np.linalg.norm(b - A @ x) <= 1.0001e-8 * np.linalg.norm(b)
        iterations.append(solver._lastOuterIters)
        # switched off, a block backend hands AMGCL no near null-space: it takes its constant blocks
        assert translations or recorded == []
    print("outer iterations, constant blocks / translations:", iterations)
    return solver, recorded, iterations


def test_a_3d_field_on_float_blocks_solves_with_the_translations_of_its_scaled_block(monkeypatch):
    A, b, blocks = _displacementSystem(3)
    solver, recorded, _ = _solveWithAndWithoutTranslations(monkeypatch, A, b, blocks, _defaultFieldPrecond(3))
    # the displacement field only (the damage field is scalar), on its 3x3 float blocks
    displacement = blocks[0]
    assert len(recorded) == 1 and recorded[0][0] == 3
    expected = translationNullspace(displacement, solver._scaling[displacement.start : displacement.stop])
    assert np.array_equal(recorded[0][1], expected)
    assert hierarchyLevels(solver._preconditioners[0]) > 1


def test_a_2d_field_on_double_blocks_solves_with_the_translations_of_its_scaled_block(monkeypatch):
    A, b, blocks = _displacementSystem(2)
    solver, recorded, iterations = _solveWithAndWithoutTranslations(monkeypatch, A, b, blocks, _defaultFieldPrecond(2))
    assert iterations[1] <= iterations[0]
    assert len(recorded) == 1 and recorded[0][0] == 2
    assert np.array_equal(recorded[0][1], translationNullspace(blocks[0], solver._scaling))
    assert hierarchyLevels(solver._preconditioners[0]) > 1


@pytest.mark.parametrize("dimension", [3, 2], ids=["3D, 3x3 float", "2D, 2x2 double"])
def test_the_default_trees_build_the_hierarchy_from_the_translations(monkeypatch, dimension):
    # no fieldPreconds at all: the default tree of the displacement field's dimension puts it on a block backend
    # with the translations D^-1 t of its equilibrated block as near null-space
    assert _defaultFieldPrecond(dimension)["blockTranslationNullspace"] is True
    A, b, blocks = _displacementSystem(dimension)
    recorded = _recordNullspaces(monkeypatch)
    solver = createSolver({"verbosity": "silent", "outerTol": 1e-8})
    solver.setFieldStructure(blocks)
    x = solver(A, b)
    assert np.linalg.norm(b - A @ x) <= 1.0001e-8 * np.linalg.norm(b)
    # the displacement field only (a damage field is scalar and takes AMGCL's default constant)
    displacement = blocks[0]
    assert len(recorded) == 1 and recorded[0][0] == dimension
    expected = translationNullspace(displacement, solver._scaling[displacement.start : displacement.stop])
    assert np.array_equal(recorded[0][1], expected)
    assert hierarchyLevels(solver._preconditioners[0]) > 1


def test_the_key_changes_nothing_on_the_scalar_backend(monkeypatch):
    # The 3D default tree with the scalar backend set on top is valid as it is, and gives the same hierarchy
    # whatever blockTranslationNullspace says: the scalar backend takes the rigid-body modes (the translations
    # here, without node coordinates). In double: the float scalar backend sets its hierarchy up in single
    # precision, whose spectral-radius estimates are not reproducible to the last bit from one build to the next.
    A, b, blocks = _displacementSystem(3)
    solutions, iterations, recordings = [], [], []
    for translations in (True, False):
        recorded = _recordNullspaces(monkeypatch)
        tree = {
            **_defaultFieldPrecond(3),
            "backendBlockSize": 1,
            "backendPrecision": "double",
            "blockTranslationNullspace": translations,
        }
        solver = createSolver({"fieldPreconds": {"displacement": tree}, "verbosity": "silent", "outerTol": 1e-8})
        solver.setFieldStructure(blocks)
        solutions.append(solver(A, b))
        iterations.append(solver._lastOuterIters)
        recordings.append(recorded)
    assert iterations[0] == iterations[1]
    assert np.linalg.norm(solutions[0] - solutions[1]) <= 1e-12 * np.linalg.norm(solutions[1])
    displacement = blocks[0]
    expected = translationNullspace(displacement, solver._scaling[displacement.start : displacement.stop])
    for recorded in recordings:
        assert len(recorded) == 1 and recorded[0][0] == 1 and np.array_equal(recorded[0][1], expected)


@pytest.mark.parametrize(
    "change, reason",
    [
        ({"blockTranslationNullspace": 1}, "blockTranslationNullspace must be true or false"),
        # the default tree has the translations on, so another coarsening has to switch them off
        ({"coarsening": {"type": "ruge_stuben"}}, "needs smoothed aggregation"),
    ],
    ids=["not a bool", "not smoothed aggregation"],
)
def test_blockamg_refuses_a_translation_near_null_space_it_cannot_build(change, reason):
    with pytest.raises(ValueError, match=reason):
        createSolver({"fieldPreconds": {"displacement": {**_defaultFieldPrecond(3), **change}}})


def test_another_coarsening_takes_the_translations_switched_off():
    tree = {**_defaultFieldPrecond(3), "coarsening": {"type": "ruge_stuben"}, "blockTranslationNullspace": False}
    assert createSolver({"fieldPreconds": {"displacement": tree}})._fieldPreconds == {"displacement": tree}


def test_a_block_backend_takes_no_near_null_space_but_the_translations():
    A, block = _scaledVectorLaplacian(3, 4)
    scaled, scaling = _equilibrated(A)
    translations = translationNullspace(block, scaling)
    params = {
        "precond": {"coarsening": {"type": "smoothed_aggregation"}},
        "solver": {"type": "preonly"},
        "backendBlockSize": 3,
    }
    # more modes than the translations: the rotations need the scalar backend
    with pytest.raises(RuntimeError, match="rotations need the scalar backend"):
        PyAMGCLSolver(params).set_nullspace(np.ones((scaled.shape[0], 6)))
    # a coupling of two components inside a node's block, or a weight that is not positive
    offDiagonal, negative = translations.copy(), translations.copy()
    offDiagonal[0, 1] = 1.0
    negative[4, 1] = -negative[4, 1]
    for corrupted in (offDiagonal, negative):
        with pytest.raises(RuntimeError, match="only the translations are supported"):
            PyAMGCLSolver(params).set_nullspace(corrupted)
    # a hierarchy that is not smoothed aggregation
    with pytest.raises(RuntimeError, match="near null-space for smoothed aggregation"):
        PyAMGCLSolver({**params, "precond": {"coarsening": {"type": "ruge_stuben"}}}).set_nullspace(translations)
    # the translations of another matrix
    hierarchy = PyAMGCLSolver(params)
    hierarchy.set_nullspace(translations[:-3])
    with pytest.raises(RuntimeError, match="gave the translations of"):
        hierarchy.build(scaled)
