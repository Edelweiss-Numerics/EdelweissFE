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

"""The UMFPACK sparse LU, through SciPy, as a direct solver backend.

SciPy uses UMFPACK where ``scikit-umfpack`` is installed, and silently falls back to SuperLU where it is not: this
backend never fails to solve -- it just may not be UMFPACK.
"""

import warnings

import numpy as np

from edelweissfe.linsolve.base import DirectSolverBackend


class UMFPACKBackend(DirectSolverBackend):
    """UMFPACK through :func:`scipy.sparse.linalg.factorized`, which factorizes once and returns the solve with the
    factors. Takes no options."""

    def __init__(self):
        # the solve with the factors of the last factorize(), for solveFactorized()
        self._solve = None

    def factorize(self, A) -> None:
        from scipy.sparse.linalg import factorized

        try:
            # UMFPACK reports a singular matrix by a warning, SuperLU by an error: both raise here
            with warnings.catch_warnings():
                warnings.simplefilter("error", UserWarning)
                self._solve = factorized(A.tocsc())
        except (RuntimeError, UserWarning) as error:
            raise np.linalg.LinAlgError(str(error)) from error

    def releaseFactorization(self) -> None:
        self._solve = None

    def solveFactorized(self, b):
        b = np.asarray(b, dtype=float)
        if b.ndim == 1:
            return self._solve(b)
        return np.column_stack([self._solve(np.ascontiguousarray(column)) for column in b.T])
