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
"""The p-two-grid preconditioner driven through :class:`~edelweissfe.linsolve.blockamg.blockamg.BlockAMGSolver`
itself, not built directly: a solve through it, and the fine spectral radius cached across hierarchy rebuilds
(``p1FineSpectralRadiusReuse``).
"""

import numpy as np
import pytest
from _p1lattice import dirichletDofs, lattice, latticeSystem

pytest.importorskip("edelweissfe.linsolve.amgcl.amgcl")

from edelweissfe.linsolve.base import FieldBlock  # noqa: E402
from edelweissfe.linsolve.blockamg.blockamg import BlockAMGSolver  # noqa: E402

#: Tight enough that a wrong preconditioner shows in the residual, loose enough for the small lattice.
_TOLERANCE = 1e-8


@pytest.fixture(scope="module")
def problem():
    """The lattice block with its Dirichlet rows, unequilibrated again (the solver equilibrates itself), a
    right-hand side that is zero on the Dirichlet rows, and the P1 map."""
    coords, isCorner, edgeEndpoints = lattice(6)
    constrained = dirichletDofs(coords)
    As, dinv = latticeSystem(coords, constrained)
    A = (As.multiply(1.0 / dinv[:, None]).multiply(1.0 / dinv[None, :])).tocsr()
    b = np.random.default_rng(3).standard_normal(A.shape[0])
    b[constrained] = 0.0
    return A, b, coords, isCorner, edgeEndpoints


def _solver(isCorner, edgeEndpoints, **options):
    settings = {
        "outerTol": _TOLERANCE,
        "outerRestart": 300,
        "outerMaxiter": 1,
        "verbosity": "silent",
        "p1Maps": {"displacement": (isCorner, edgeEndpoints)},
    }
    settings.update(options)
    solver = BlockAMGSolver(**settings)
    solver.setFieldStructure([FieldBlock("displacement", 0, 3 * len(isCorner), 3)])
    return solver


def _relativeResidual(A, x, b):
    return np.linalg.norm(b - A @ x) / np.linalg.norm(b)


def test_the_p1_path_solves_through_the_solver(problem):
    A, b, coords, isCorner, edgeEndpoints = problem
    solver = _solver(isCorner, edgeEndpoints)
    x = solver(A, b)
    assert _relativeResidual(A, x, b) <= _TOLERANCE
    assert type(solver._preconditioners[0]).__name__ == "PTwoGridPreconditioner"


def test_a_rebuild_reuses_the_cached_fine_spectral_radius(problem):
    A, b, coords, isCorner, edgeEndpoints = problem
    solver = _solver(isCorner, edgeEndpoints, p1FineSpectralRadiusReuse=True)
    solver(A, b)
    first = solver._preconditioners[0]
    assert first.fineSpectralRadiusSource == "estimated"
    cachedRadius, cachedVector = solver._p1FineSpectralRadiusCache["displacement"]
    assert cachedRadius == first.fineSpectralRadius and len(cachedVector) == A.shape[0]

    # the same operator again, rebuilt: the re-check from the cached dominant vector keeps the radius
    solver._refreshNext = True
    x = solver(A, b)
    second = solver._preconditioners[0]
    assert second is not first
    assert second.fineSpectralRadiusSource == "supplied"
    assert second.fineSpectralRadius == cachedRadius
    assert _relativeResidual(A, x, b) <= _TOLERANCE


def test_without_reuse_every_rebuild_estimates_in_full(problem):
    A, b, coords, isCorner, edgeEndpoints = problem
    solver = _solver(isCorner, edgeEndpoints)
    solver(A, b)
    solver._refreshNext = True
    solver(A, b)
    assert solver._preconditioners[0].fineSpectralRadiusSource == "estimated"
    assert not solver._p1FineSpectralRadiusCache
