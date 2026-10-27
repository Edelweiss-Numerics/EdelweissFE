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

"""SciPy's own sparse LU (SuperLU) as a direct solver backend.

There is no extension to build and no optional dependency behind it (SciPy is a hard requirement), so it is always
available, which is also what makes ``type=superlu`` the fallback of
:func:`~edelweissfe.config.linsolve.getDefaultLinSolver`.
"""

import numpy as np

from edelweissfe.linsolve.base import DirectSolverBackend


class SuperLUBackend(DirectSolverBackend):
    """SuperLU through :func:`scipy.sparse.linalg.splu`, with SciPy's defaults. Takes no options."""

    def __init__(self):
        # the factorization of the last factorize(), for solveFactorized()
        self._factorization = None

    def factorize(self, A) -> None:
        from scipy.sparse.linalg import splu

        # A is CSR, i.e. the CSC arrays of A^T: factorize A^T and solve transposed, as scipy's spsolve does for CSR,
        # which also gives bitwise its results (factorizing A.tocsc() instead differs in the round-off)
        try:
            self._factorization = splu(A.T)
        except RuntimeError as error:  # SuperLU's "Factor is exactly singular"
            raise np.linalg.LinAlgError(str(error)) from error

    def solveFactorized(self, b):
        return self._factorization.solve(b, trans="T")

    def releaseFactorization(self) -> None:
        self._factorization = None
