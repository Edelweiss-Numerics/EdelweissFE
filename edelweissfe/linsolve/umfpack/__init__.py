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

``use_umfpack=True`` is a *request*, not a guarantee: SciPy silently falls back to SuperLU when ``scikit-umfpack`` is
not installed, so this backend never fails to solve -- it just may not be UMFPACK.
"""

from edelweissfe.linsolve.base import OneCallDirectSolverBackend


def _umfpackSolve(A, b):
    from scipy.sparse.linalg import spsolve

    return spsolve(A.copy(), b, use_umfpack=True)


class UMFPACKBackend(OneCallDirectSolverBackend):
    """UMFPACK through :func:`scipy.sparse.linalg.spsolve`, which factorizes and solves in one call. Takes no
    options."""

    def __init__(self):
        super().__init__(_umfpackSolve)
