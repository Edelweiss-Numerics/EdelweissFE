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
"""``hierarchyOnUnscaledBlock``: a field's hierarchy built on its unscaled block A_ii and applied as
D^-1 AMG(A_ii)^-1 D^-1, so that a block backend's constant blocks are the translations of A_ii."""

import copy
import json
from pathlib import Path

import numpy as np
import pytest
from _blockamgsystems import displacementDamageSystem, hierarchyLevels

from edelweissfe.linsolve.amgcl.amgcl import PyAMGCLEquilibratedSystem
from edelweissfe.linsolve.blockamg import createSolver
from edelweissfe.linsolve.blockamg.blockamg import UnscaledBlockPreconditioner

_RECOMMENDED = Path(__file__).resolve().parents[1] / "doc/source/documentation/blockamg_recommended.json"


def _onUnscaledBlock(config: dict) -> dict:
    config = copy.deepcopy(config)
    tree = config["fieldPreconds"]["displacement"]
    tree["hierarchyOnUnscaledBlock"] = True
    tree["relax"]["scale"] = True
    return config


@pytest.mark.parametrize("storedUnscaled", [True, False], ids=["A stored", "D A D stored"])
def test_the_unscaled_diagonal_block_is_the_one_of_a(storedUnscaled):
    A, b, blocks = displacementDamageSystem(6, coupling="symmetric", rowScaling=True)
    scaling = 1.0 / np.sqrt(np.abs(A.diagonal()))
    system = PyAMGCLEquilibratedSystem()
    system.build(A, scaling, [(block.start, block.stop) for block in blocks], unscaled=storedUnscaled)
    for i, block in enumerate(blocks):
        expected = A[block.start : block.stop, block.start : block.stop].tocsr()
        actual = system.diagonalBlock(i, unscaled=True)
        assert np.array_equal(actual.indptr, expected.indptr) and np.array_equal(actual.indices, expected.indices)
        if storedUnscaled:
            assert np.array_equal(actual.data, expected.data)
        else:
            # recovered from (a_ij * s_j) * s_i by two divisions
            assert np.allclose(actual.data, expected.data, rtol=1e-14, atol=0.0)


def test_the_recommended_configuration_on_the_unscaled_block_solves_and_iterates_less():
    A, b, blocks = displacementDamageSystem(12, coupling="symmetric", rowScaling=True)
    recommended = {**json.load(open(_RECOMMENDED)), "verbosity": "silent", "outerTol": 1e-8}
    iterations = []
    for config in (recommended, _onUnscaledBlock(recommended)):
        solver = createSolver(config)
        solver.setFieldStructure(blocks)
        x = solver(A, b)
        assert np.linalg.norm(b - A @ x) <= 1.0001e-8 * np.linalg.norm(b)
        iterations.append(solver._lastOuterIters)
    displacement = solver._preconditioners[0]
    assert type(displacement) is UnscaledBlockPreconditioner and hierarchyLevels(displacement) > 1
    print("outer iterations, on A_scaled_uu / on A_uu:", iterations)
    assert iterations[1] <= iterations[0]


def test_an_unscaled_block_hierarchy_needs_a_jacobi_scaled_chebyshev_smoother():
    config = _onUnscaledBlock(json.load(open(_RECOMMENDED)))
    del config["fieldPreconds"]["displacement"]["relax"]["scale"]
    with pytest.raises(ValueError, match="relax.scale"):
        createSolver(config)
