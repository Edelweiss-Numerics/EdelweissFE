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
"""An exact solve as a component: :class:`DirectSolver`, :math:`z = A^{-1} r` by a sparse LU factorization.

A direct solver alone is the composed solver of ``type=pardiso`` (``superlu``, ``klu``, ...): one leaf, set up (i.e.
factorized) on every solve and applied once. Inside a larger tree it solves a block exactly, e.g. the condensed dofs
of a Schur complement, factorized once per set-up and applied in every outer iteration: every backend factorizes in
:meth:`~edelweissfe.linsolve.base.DirectSolverBackend.factorize` and only back-substitutes afterwards.
"""

import numpy as np

from edelweissfe.config import registry
from edelweissfe.linsolve.composition.base import (
    Description,
    Preconditioner,
    SetUpContext,
)


class DirectSolver(Preconditioner):
    """:math:`z = A^{-1} r`, exactly: one of the registered direct solver backends (``superlu``, ``umfpack``,
    ``pardiso``, ``klu``, ...; see :class:`~edelweissfe.linsolve.base.DirectSolverBackend`), factorized once per
    set-up -- a singular matrix raises :class:`numpy.linalg.LinAlgError` there -- and applied by back substitution.

    The backend is created here, at construction, and lives as long as this component: a backend that keeps its
    symbolic factorization while only the values of the matrix change (PARDISO's ``reuseSymbolicFactorization``)
    keeps it from one set-up to the next, while the numeric factorization of the last matrix is released before the
    next one is computed (:meth:`_forget`). A backend whose optional extension is not installed raises
    :class:`ImportError` here.

    Parameters
    ----------
    backend
        The registered name of the backend (registry category ``directsolver``).
    **options
        The options of the backend, e.g. ``reuseSymbolicFactorization=True`` for ``pardiso``.
    """

    def __init__(self, backend: str = "superlu", **options):
        self.backend = backend
        self.options = options
        backendClass, _ = registry.lookup("directsolver", backend)
        #: The factorizing backend of this component.
        self.solver = backendClass(**options)
        # after the backend: what this component builds is the backend's factorization (see _forget)
        super().__init__()

    def _forget(self) -> None:
        self.solver.releaseFactorization()

    def _setUp(self, A, context: SetUpContext) -> None:
        self.solver.factorize(A.matrix())
        self.statistics.sizes["rows"] = A.shape[0]

    def _apply(self, r: np.ndarray) -> np.ndarray:
        return self.solver.solveFactorized(r)

    def describe(self) -> Description:
        return Description(
            name="DirectSolver", formula="A⁻¹", latex="A^{-1}", parameters={"backend": self.backend, **self.options}
        )
