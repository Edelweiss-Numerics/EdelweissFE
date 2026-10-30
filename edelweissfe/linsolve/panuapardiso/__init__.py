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

"""The Panua PARDISO solver as a direct solver backend.

The binding is the optional Cython extension :mod:`edelweissfe.linsolve.panuapardiso.panuapardiso` (it needs the Panua
PARDISO library at build time), imported when a backend is created: an install without it raises :class:`ImportError`
there.
"""

from edelweissfe.linsolve.base import DirectSolverBackend


class PanuaPardisoBackend(DirectSolverBackend):
    """Panua PARDISO through :class:`~edelweissfe.linsolve.panuapardiso.panuapardiso.PanuaPardisoFactorization`:
    reordering and factorization (phases 11 and 22) in :meth:`factorize`, back substitution (phase 33) in
    :meth:`solveFactorized`. Takes no options."""

    def __init__(self):
        from edelweissfe.linsolve.panuapardiso.panuapardiso import (
            PanuaPardisoFactorization,
        )

        self._factorization = PanuaPardisoFactorization()

    def factorize(self, A) -> None:
        self._factorization.factorize(A)

    def solveFactorized(self, b):
        return self._factorization.solve(b)

    def releaseFactorization(self) -> None:
        self._factorization.release()
