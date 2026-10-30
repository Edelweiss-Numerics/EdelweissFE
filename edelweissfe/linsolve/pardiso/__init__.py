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

"""Intel MKL PARDISO as a direct solver backend.

The binding is the optional Cython extension :mod:`edelweissfe.linsolve.pardiso.pardiso` (it needs MKL at build time),
imported when a backend is created: an install without it raises :class:`ImportError` there, which
:func:`~edelweissfe.config.linsolve.getDefaultLinSolver` catches to fall back to SuperLU.
"""

import numpy as np

from edelweissfe.linsolve.base import DirectSolverBackend


class PardisoBackend(DirectSolverBackend):
    """PARDISO through :class:`~edelweissfe.linsolve.pardiso.pardiso.PardisoSolver`: reordering and symbolic
    factorization (phase 11), numeric factorization (phase 22) in :meth:`factorize`, back substitution (phase 33) in
    :meth:`solveFactorized`.

    Parameters
    ----------
    reuseSymbolicFactorization
        Keep the reordering and symbolic factorization while the sparsity pattern stays the same, so that a new matrix
        costs only its numeric factorization. Off by default: the reuse has been observed to silently produce wrong
        (but not NaN, so undetected) results for some coupled-dof problems; opt in once it has been verified for the
        problem at hand.
    """

    def __init__(self, reuseSymbolicFactorization: bool = False):
        from edelweissfe.linsolve.pardiso.pardiso import PardisoSolver

        self._solver = PardisoSolver(reuseSymbolicFactorization=reuseSymbolicFactorization)

    def factorize(self, A) -> None:
        try:
            # a copy: PARDISO reads the values again in the back substitution (iterative refinement), which must see
            # the matrix it factorized, also after the assembly overwrote it in place
            self._solver.factorize(A.copy())
        except RuntimeError as error:
            raise np.linalg.LinAlgError(str(error)) from error

    def solveFactorized(self, b):
        return self._solver.solveFactorized(b)

    def releaseFactorization(self) -> None:
        self._solver.releaseFactorization()
