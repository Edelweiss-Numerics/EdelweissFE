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
#  This library is free software; you can redistribute it and/or
#  modify it under the terms of the GNU Lesser General Public
#  License as published by the Free Software Foundation; either
#  version 2.1 of the License, or (at your option) any later version.
#
#  The full text of the license can be found in the file LICENSE.md at
#  the top level directory of EdelweissFE.
#  ---------------------------------------------------------------------
"""Regression guard for the scalar double AMGCL backend (``PyAMGCLSolver`` with the default
``backendBlockSize = 1``, ``backendPrecision = "double"``): ``relax.type = "gauss_seidel"`` and
``"ilu0"`` must build and apply, which was briefly broken for every ``PyAMGCLSolver`` user while level matrices were
stored with
AMGCL's ``builtin_stored_matrices`` instead of plain ``builtin<double>``: that derived backend makes AMGCL
reject ``gauss_seidel`` outright (``relaxation_is_supported`` keys off ``provides_row_iterator``) and
silently falls back ILU0 to its generic, approximate triangular solve instead of the exact one plain
``builtin`` gets. The scalar backend went back to plain ``builtin<double>`` (see amgcl-wrapper.hpp's
``LinearSolver`` typedef); this test exercises exactly the smoothers that regression broke.

Needs a system above AMGCL's own ``coarse_enough`` (3000 scalar rows) so a real multi-level hierarchy is
built -- below it, AMG solves directly on the first level and never constructs a relaxation object at
all, so a bug confined to the smoother construction/apply would pass unnoticed.
"""

import numpy as np
import pytest
import scipy.sparse as sp
from _linsolvesystems import gridLaplacian

pytest.importorskip("edelweissfe.linsolve.amgcl.amgcl")

from edelweissfe.linsolve.amgcl.amgcl import PyAMGCLSolver  # noqa: E402

#: 16**3 = 4096 scalar rows, well above AMGCL's coarse_enough (3000) for a scalar direct solve.
_N = 16


@pytest.fixture(scope="module")
def system():
    A = (gridLaplacian(_N) + 0.01 * sp.identity(_N**3)).tocsr()
    b = np.random.default_rng(11).standard_normal(A.shape[0])
    return A, b


@pytest.mark.parametrize("relaxationType", ["gauss_seidel", "ilu0"])
def test_scalar_backend_builds_and_applies(system, relaxationType):
    A, b = system
    solver = PyAMGCLSolver(
        {
            "precond": {"coarsening": {"type": "smoothed_aggregation"}, "relax": {"type": relaxationType}},
        }
    )
    solver.build(A)
    x = solver.applyPreconditioner(b)
    assert np.all(np.isfinite(x))
    assert np.linalg.norm(x) > 0.0


@pytest.mark.parametrize("relaxationType", ["gauss_seidel", "ilu0"])
def test_scalar_backend_preconditions_a_krylov_solve(system, relaxationType):
    from edelweissfe.linsolve.base import FieldBlock
    from edelweissfe.linsolve.composition import (
        AMG,
        LGMRES,
        ComposedLinearSolver,
        FixedTolerance,
    )

    A, b = system
    tree = {"coarsening": {"type": "smoothed_aggregation"}, "relax": {"type": relaxationType}}
    solver = ComposedLinearSolver(
        LGMRES(AMG(tree=tree, backendBlockSize=1), tolerance=FixedTolerance(1e-8), maxIterations=300),
        verbosity="silent",
    )
    solver.setFieldStructure([FieldBlock("u", 0, A.shape[0], 1)])
    x = solver(A, b)
    assert solver.lastSolveSummary is None or solver.lastSolveSummary.residualMet
    assert np.linalg.norm(b - A @ x) < 1e-6 * np.linalg.norm(b)


def test_ilu0_is_an_exact_triangular_solve():
    # On a tridiagonal matrix ILU0 has no fill and is the exact LU factorization, so one application of the
    # ILU0 relaxation as the preconditioner solves the system to rounding. AMGCL's exact triangular solves (serial
    # or level-scheduled, plain builtin) do; its generic fallback for other backends -- two damped Jacobi sweeps
    # per triangle -- leaves a residual of order 1e-1, so this catches the ILU half of the regression.
    n = 1000
    A = sp.diags([-np.ones(n - 1), 4.0 * np.ones(n), -np.ones(n - 1)], [-1, 0, 1], format="csr")
    b = np.random.default_rng(12).standard_normal(n)
    solver = PyAMGCLSolver({"precond": {"class": "relaxation", "type": "ilu0"}})
    solver.build(A)
    x = solver.applyPreconditioner(b)
    assert np.linalg.norm(b - A @ x) < 1e-12 * np.linalg.norm(b)
