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
"""``DirectCSRAssembler``'s own docstring names ``assembleFromVIJ`` the equivalence path meant to
check its addressing against ``CSRGenerator.updateCSR`` "on real models rather than argued about" --
but no such test existed anywhere in this codebase. This pins that comparison, plus the fused
begin/scatter/reduce path a threaded entity loop actually uses, against a small synthetic COO system
with duplicate (row, col) pairs (multiple "entities" contributing to the same DOF pair), which is
exactly the case that would expose a wrong offset into a shared CSR entry.
"""

import numpy as np
import pytest

from edelweissfe.numerics.csrgeneratorv2 import CSRGenerator, DirectCSRAssembler


class _SystemMatrix:
    """Minimal stand-in for the DofManager-provided VIJ system matrix: just I/J/nDof."""

    def __init__(self, I, J, nDof):  # noqa: E741
        self.I = np.asarray(I, dtype=np.intc)  # noqa: E741
        self.J = np.asarray(J, dtype=np.intc)
        self.nDof = nDof


def _syntheticSystem():
    """Two overlapping 2x2 "element" blocks (DOFs {0, 1} and {1, 2}) plus one isolated 1x1 block
    (DOF 3), so DOF 1 receives contributions from both elements at the same (1, 1) CSR entry -- the
    case that actually exercises whether the two assembly paths agree, not just whether they run.

    Concatenated **column-major per entity** (row varies fastest within each entity's block) --
    matching ``CSRDirectAssembler::registerEntities``'s own documented convention, so this same
    array can seed both the plain COO-scatter test and the fused per-entity one.
    """

    I = [0, 1, 0, 1, 1, 2, 1, 2, 3]  # noqa: E741
    J = [0, 0, 1, 1, 1, 1, 2, 2, 3]
    V = np.array([4.0, 1.0, 1.0, 5.0, 2.0, 0.5, 0.5, 3.0, 7.0])
    nDof = 4
    return _SystemMatrix(I, J, nDof), V


def test_assembleFromVIJ_matches_the_staged_generator_bit_for_bit():
    systemMatrix, V = _syntheticSystem()

    generator = CSRGenerator(systemMatrix)
    staged = generator.updateCSR(V)

    direct = DirectCSRAssembler(generator, systemMatrix, numThreads=1)
    directResult = direct.assembleFromVIJ(V)

    assert directResult is generator.csrMatrix
    np.testing.assert_array_equal(directResult.toarray(), staged.toarray())


def test_fused_scatter_reduce_path_matches_the_staged_generator():
    """The path a threaded entity loop actually uses: registerEntities() once, then
    beginAssembly()/scatterBlock() per entity/reduce() -- rather than assembleFromVIJ's one-shot COO
    scatter. Same synthetic system, grouped into its three entities (two 2x2 blocks sharing DOF 1,
    one 1x1 block), each entity's dense block scattered in the column-major order
    ``_syntheticSystem`` already lays out."""

    systemMatrix, V = _syntheticSystem()

    generator = CSRGenerator(systemMatrix)
    staged = generator.updateCSR(V)

    direct = DirectCSRAssembler(generator, systemMatrix, numThreads=1)

    # mapStarts[e]: entity e's offset into the VIJ ordering above; nDofs[e]: its local DOF count.
    mapStarts = np.array([0, 4, 8], dtype=np.int64)
    nDofs = np.array([2, 2, 1], dtype=np.intc)
    direct.registerEntities(mapStarts, nDofs)

    direct.beginAssembly()
    direct.scatterBlock(0, 0, V[0:4].copy())
    direct.scatterBlock(0, 1, V[4:8].copy())
    direct.scatterBlock(0, 2, V[8:9].copy())
    fusedResult = direct.reduce()

    np.testing.assert_array_equal(fusedResult.toarray(), staged.toarray())


def test_releaseGatherMap_leaves_the_pattern_usable_but_forbids_further_gathering():
    systemMatrix, V = _syntheticSystem()

    generator = CSRGenerator(systemMatrix)
    generator.updateCSR(V)
    patternBefore = generator.csrMatrix.indices.copy()

    generator.releaseGatherMap()
    assert generator.gatherMapReleased

    np.testing.assert_array_equal(generator.csrMatrix.indices, patternBefore)
    with pytest.raises(RuntimeError):
        generator.updateInPlace(V)


def test_patternOnly_generator_builds_the_same_pattern_as_a_full_one():
    systemMatrix, V = _syntheticSystem()

    full = CSRGenerator(systemMatrix)
    patternOnly = CSRGenerator(systemMatrix, patternOnly=True)

    np.testing.assert_array_equal(patternOnly.csrMatrix.indptr, full.csrMatrix.indptr)
    np.testing.assert_array_equal(patternOnly.csrMatrix.indices, full.csrMatrix.indices)
    assert patternOnly.gatherMapReleased

    with pytest.raises(RuntimeError):
        patternOnly.updateInPlace(V)
