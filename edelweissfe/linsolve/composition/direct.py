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
"""An exact solve as a component: :class:`DirectSolver`, :math:`z = A^{-1} r` by a sparse LU factorization."""

import numpy as np

from edelweissfe.linsolve.composition.base import (
    Description,
    Preconditioner,
    SetUpContext,
)


class DirectSolver(Preconditioner):
    """:math:`z = A^{-1} r`, exactly: one of the registered direct solvers (``superlu``, ``pardiso``, ``klu``, ...),
    factorized once per set-up and applied by back substitution.

    Parameters
    ----------
    linsolver
        The registered name of the direct solver; it must support
        :meth:`~edelweissfe.linsolve.base.LinearSolver.factorize`.
    options
        Its options, as in a ``linsolverConfigFile``.
    """

    def __init__(self, linsolver: str = "superlu", options: dict = None):
        self.linsolver = linsolver
        self.options = dict(options or {})
        #: The factorizing solver: created at the first set-up, or by the parent that factorizes it (see
        #: :meth:`newSolver`).
        self.solver = None

    def newSolver(self):
        """A new instance of the direct solver. The first one becomes this component's own: a parent that factorizes
        the matrix itself (the Schur complement factorizes :math:`A_{bb}`) creates it through here, and this
        component then only applies it."""
        from edelweissfe.config.linsolve import getLinSolverByName

        solver = getLinSolverByName(self.linsolver, dict(self.options))
        if self.solver is None:
            self.solver = solver
        return solver

    def setUp(self, A, context: SetUpContext) -> None:
        if self.solver is None:
            self.newSolver()
        self.solver.factorize(A.matrix())
        self.statistics.sizes["rows"] = A.shape[0]

    def __call__(self, r: np.ndarray) -> np.ndarray:
        return self.solver.solveFactorized(r)

    def describe(self) -> Description:
        return Description(name="DirectSolver", formula="A⁻¹", latex="A^{-1}", parameters={"linsolver": self.linsolver})
