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
``"ilu0"`` must build and apply, which was briefly broken for every ``PyAMGCLSolver`` user (the registered
``linsolver = amgcl``, the p-two-grid field's double block backend) while level matrices were stored with
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
from _blockamgsystems import gridLaplacian

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
def test_scalar_backend_solves(system, relaxationType):
    A, b = system
    solver = PyAMGCLSolver(
        {
            "solver": {"type": "bicgstab", "tol": 1e-8},
            "precond": {"coarsening": {"type": "smoothed_aggregation"}, "relax": {"type": relaxationType}},
        }
    )
    x = solver.solve(A, b)
    assert solver.lastError < 1.0001e-8
    assert np.linalg.norm(b - A @ x) < 1e-6 * np.linalg.norm(b)
