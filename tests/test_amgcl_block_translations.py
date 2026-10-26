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
import re

import numpy as np
import pytest
import scipy.sparse as sp
from _linsolvesystems import (
    NODAL_COUPLING,
    equilibrated,
    gridLaplacian,
    hierarchyLevels,
)

pytest.importorskip("edelweissfe.linsolve.amgcl.amgcl")

from edelweissfe.linsolve.amgcl.amgcl import PyAMGCLSolver  # noqa: E402
from edelweissfe.linsolve.base import FieldBlock  # noqa: E402
from edelweissfe.linsolve.composition.amg import defaultFieldTree  # noqa: E402
from edelweissfe.linsolve.nullspace import translationNullspace  # noqa: E402

#: Every block backend blockamg builds, as (nodal dimension = block size, backendPrecision, nodes per grid
#: direction): the grids give a hierarchy of exactly two levels (AMGCL solves directly below 3000 scalar rows).
#: With :data:`_DEEP_COARSE_ENOUGH` they give three or more.
_BLOCK_BACKENDS = [(3, "double", 12), (3, "float", 12), (2, "double", 40)]
_BLOCK_BACKEND_IDS = ["3x3 double", "3x3 float", "2x2 double"]

#: AMGCL's ``coarse_enough`` for the tests of a hierarchy of three or more levels: it solves directly below this many
#: block rows. The grids above have 1728 (3x3) or 1600 (2x2) nodes, and an aggregate of the plain aggregation holds
#: a node and its strong neighbours, a few nodes, so the first coarsening leaves hundreds of nodes, more than this,
#: and a second coarsening follows.
_DEEP_COARSE_ENOUGH = 20


def _scaledVectorLaplacian(dimension: int, n: int, coupled: bool = True):
    """A vector Laplacian kron(L + 0.01 I, C), C the nodal coupling of ``dimension`` components, on a grid of n nodes
    per direction, scaled on both sides by a random diagonal spanning two orders of magnitude: its diagonal varies
    from DOF to DOF, within a node too, so the constant blocks of its equilibrated block are far from the
    translations of that block. ``coupled=False`` keeps only the diagonal of C: the components of a node are not
    coupled, and every block of every level of a hierarchy is diagonal. Returns the matrix (CSR, sorted columns) and
    its field block."""
    nodes = n**dimension
    coupling = NODAL_COUPLING[:dimension, :dimension]
    if not coupled:
        coupling = np.diag(np.diag(coupling))
    stiffness = sp.kron(gridLaplacian(n, dimension) + 0.01 * sp.identity(nodes), coupling)
    scaling = sp.diags(10.0 ** np.random.default_rng(3).uniform(0.0, 2.0, dimension * nodes))
    A = (scaling @ stiffness @ scaling).tocsr()
    A.sort_indices()
    return A, FieldBlock("displacement", 0, dimension * nodes, dimension)


def _hierarchy(A, dimension: int, precision: str, precond: dict, nullspace=None):
    """An AMGCL hierarchy for A on the block backend, with the near null-space ``nullspace`` if given."""
    hierarchy = PyAMGCLSolver(
        {
            "precond": copy.deepcopy(precond),
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


@pytest.mark.parametrize(
    "coarseEnough, coupled",
    [(None, True), (_DEEP_COARSE_ENOUGH, False)],
    ids=["two levels", "three or more levels, uncoupled components"],
)
@pytest.mark.parametrize("dimension, precision, n", _BLOCK_BACKENDS, ids=_BLOCK_BACKEND_IDS)
def test_the_translations_of_d_a_d_give_the_hierarchy_of_a_on_its_constant_blocks(
    dimension, precision, n, coarseEnough, coupled
):
    # The defining property of the tentative prolongation: its coarse space holds the translations D^-1 t of
    # D A D exactly, component by component. With the same aggregates (eps_strong 0: every connection is strong,
    # whatever the scaling), the same prolongation damping (no spectral-radius estimate) and a block-Jacobi
    # smoother, which is invariant under the scaling, the hierarchy of D A D on D^-1 t is then the hierarchy of A
    # on its constant blocks t in other variables, M^-1 = D^-1 AMG(A)^-1 D^-1, to rounding. The per-aggregate
    # normalization only scales the coarse variables, which the direct solve on the coarse level does not see.
    #
    # On three or more levels the next coarsening sees them, and the identity does not extend as such. The coarse
    # matrix of D A D is C A_1 C, A_1 the coarse matrix of A and C = R^-1 the per-aggregate, per-component
    # normalization, which scales the components of a coarse node differently. Two steps of AMGCL's smoothed
    # aggregation are not invariant under that. Its block strength test, tr(eps^2 a_ii a_jj) < tr(a_ij a_ij),
    # takes the trace of a product, not a norm: tr(a_ij a_ij) of a nonsymmetric coarse block can be negative,
    # which makes the connection weak even at eps_strong 0, and its sign changes with the scaling. And the
    # prolongation smoothing lumps the weak blocks of a row into its diagonal block unweighted, which a scaling
    # that differs from node to node changes. With the nodal coupling, the first coarse level of the 3x3 case has
    # such weak blocks, in both hierarchies but not the same ones, and the preconditioners differ by 4e-5. Uncoupled
    # components keep every block of every level diagonal: tr(a_ij a_ij) is then the squared norm of a_ij, no
    # connection is weak at eps_strong 0, and the identity holds level by level. That checks the coarse
    # translations R each coarsening hands to the next.
    A, block = _scaledVectorLaplacian(dimension, n, coupled)
    scaled, scaling = equilibrated(A)
    precond = {
        "coarsening": {"type": "smoothed_aggregation", "aggr": {"eps_strong": 0.0}},
        "relax": {"type": "damped_jacobi"},
    }
    if coarseEnough is not None:
        precond["coarse_enough"] = coarseEnough
    # the reference in double: A is far worse conditioned than D A D, which rounding to float would show
    onA = _hierarchy(A, dimension, "double", precond)
    withTranslations = _hierarchy(scaled, dimension, precision, precond, translationNullspace(block, scaling))
    withConstantBlocks = _hierarchy(scaled, dimension, precision, precond)
    if coarseEnough is None:
        assert hierarchyLevels(onA) == hierarchyLevels(withTranslations) == 2
    else:
        assert hierarchyLevels(onA) == hierarchyLevels(withTranslations) >= 3

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
    scaled, _ = equilibrated(_scaledVectorLaplacian(dimension, n)[0])
    block = FieldBlock("displacement", 0, scaled.shape[0], dimension)
    precond = defaultFieldTree(dimension)
    constantBlocks = _hierarchy(scaled, dimension, precision, precond)
    unitWeights = _hierarchy(
        scaled, dimension, precision, precond, translationNullspace(block, np.ones(scaled.shape[0]))
    )
    assert hierarchyLevels(constantBlocks) == hierarchyLevels(unitWeights) == 2

    cleared = PyAMGCLSolver(
        {
            "precond": precond,
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


def _levelSizes(amg) -> list:
    """The unknowns and nonzeros of every level of an AMGCL hierarchy, read from its report."""
    table = re.findall(r"^\s*(\d+)\s+(\d+)\s+(\d+)\s", amg.report(), re.M)
    return [(int(rows), int(nonzeros)) for _, rows, nonzeros in table]


@pytest.mark.parametrize("dimension, precision, n", _BLOCK_BACKENDS, ids=_BLOCK_BACKEND_IDS)
def test_unit_translation_weights_aggregate_the_first_coarse_level_as_amgcl_does(dimension, precision, n):
    # Three or more levels, with AMGCL's default eps_strong: the second coarsening, the first on a coarse level,
    # has to find AMGCL's aggregates and strong connections, which it does only if the translations'
    # strength-of-connection threshold is halved from level to level as AMGCL's own is. Weights 1 give the coarse
    # matrix N^-1 A_1 N^-1 for AMGCL's coarse matrix A_1, N the square root of each aggregate's size, and the block
    # strength test, tr(eps^2 a_ii a_jj) < tr(a_ij a_ij), does not change under a scaling by a multiple of the
    # identity per node. So the first three levels have the same unknowns and nonzeros.
    #
    # The two-level identity of the preconditioners does not extend: the prolongation smoothing of the coarse level
    # lumps the weak blocks of a row into its diagonal block unweighted, which preserves AMGCL's coarse constants
    # but not the coarse translations N wherever a weak connection joins aggregates of different sizes (relative
    # differences of 2e-4 to 2e-3 here). From the third level on the matrices differ in more than a scaling, so
    # their aggregates need not agree.
    scaled, _ = equilibrated(_scaledVectorLaplacian(dimension, n)[0])
    block = FieldBlock("displacement", 0, scaled.shape[0], dimension)
    precond = {"coarsening": {"type": "smoothed_aggregation"}, "coarse_enough": _DEEP_COARSE_ENOUGH}
    constantBlocks = _hierarchy(scaled, dimension, precision, precond)
    unitWeights = _hierarchy(
        scaled, dimension, precision, precond, translationNullspace(block, np.ones(scaled.shape[0]))
    )
    assert hierarchyLevels(constantBlocks) >= 3 and hierarchyLevels(unitWeights) >= 3
    assert _levelSizes(unitWeights)[:3] == _levelSizes(constantBlocks)[:3]


def test_a_block_backend_takes_no_near_null_space_but_the_translations():
    A, block = _scaledVectorLaplacian(3, 4)
    scaled, scaling = equilibrated(A)
    translations = translationNullspace(block, scaling)
    params = {
        "precond": {"coarsening": {"type": "smoothed_aggregation"}},
        "backendBlockSize": 3,
    }
    # more modes than the translations: the rotations are supported on the scalar backend
    with pytest.raises(RuntimeError, match="rotations are supported on the scalar backend"):
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
