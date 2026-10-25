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
"""Tests for the 3x3-block AMGCL backend stored in float (``LinearSolverBlock3Float``,
``backendBlockSize = 3`` with ``backendPrecision = "float"``): set up in double, stored and applied in
float with double vectors -- a single-cycle preconditioner close to the double one, and as good inside
GMRES.
"""

import numpy as np
import pytest
import scipy.sparse as sp
from _linsolvesystems import NODAL_COUPLING, gridLaplacian
from scipy.sparse.linalg import LinearOperator, gmres

pytest.importorskip("edelweissfe.linsolve.amgcl.amgcl")

from edelweissfe.linsolve.amgcl.amgcl import PyAMGCLSolver  # noqa: E402

_PRECOND = {
    "coarsening": {"type": "smoothed_aggregation", "aggr": {"eps_strong": 0.01}, "estimate_spectral_radius": True},
    "relax": {"type": "chebyshev", "degree": 8, "power_iters": 50, "lower": 0.01, "higher": 1.1},
    "npre": 0,
    "npost": 1,
}


@pytest.fixture(scope="module")
def system():
    """An equilibrated vector-Laplacian-like operator on a 12^3 node grid, 3 DOFs per node."""
    m = 12
    A = sp.kron(gridLaplacian(m) + 0.01 * sp.identity(m**3), NODAL_COUPLING).tocsr()
    dinv = 1.0 / np.sqrt(A.diagonal())
    A = (sp.diags(dinv) @ A @ sp.diags(dinv)).tocsr()
    A.sort_indices()
    b = np.random.default_rng(7).standard_normal(A.shape[0])
    return A, b


def _built(A, precision):
    solver = PyAMGCLSolver({"precond": dict(_PRECOND), "backendBlockSize": 3, "backendPrecision": precision})
    solver.build(A)
    return solver


def test_the_float_block_cycle_stays_close_to_the_double_one(system):
    A, b = system
    double, single = _built(A, "double"), _built(A, "float")
    assert single.isFloat and single.blockSize == 3
    x, y = single.applyPreconditioner(b), double.applyPreconditioner(b)
    # set up identically in double; only the stored level matrices are rounded (relative 6e-8)
    assert np.linalg.norm(x - y) / np.linalg.norm(y) < 1e-5


def test_the_float_block_cycle_preconditions_gmres_as_well(system):
    A, b = system
    counts = {}
    for precision in ("double", "float"):
        solver = _built(A, precision)
        history = []
        x, _ = gmres(
            A,
            b,
            M=LinearOperator(A.shape, matvec=solver.applyPreconditioner),
            rtol=1e-8,
            restart=200,
            maxiter=1,
            callback=history.append,
            callback_type="pr_norm",
        )
        assert np.linalg.norm(b - A @ x) < 1e-6 * np.linalg.norm(b)
        counts[precision] = len(history)
    assert counts["float"] <= counts["double"] + 1


def test_float_with_block_size_2_is_still_refused():
    with pytest.raises(NotImplementedError):
        PyAMGCLSolver({"precond": dict(_PRECOND), "backendBlockSize": 2, "backendPrecision": "float"})


def test_the_scalar_float_backend_rounds_the_values_as_numpy_does(system):
    # the double values cross the wrapper and are rounded to float in C++: the same hierarchy as from values
    # rounded beforehand. spai0 and plain smoothed aggregation are deterministic, so the two applications
    # compare bit for bit.
    A, b = system
    rounded = A.copy()
    rounded.data = rounded.data.astype(np.float32).astype(np.float64)
    precond = {"coarsening": {"type": "smoothed_aggregation"}, "relax": {"type": "spai0"}}
    applications = []
    for matrix in (A, rounded):
        solver = PyAMGCLSolver({"precond": precond, "backendPrecision": "float"})
        solver.build(matrix)
        applications.append(solver.applyPreconditioner(b))
    assert np.array_equal(applications[0], applications[1])
