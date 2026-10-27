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

"""PETSc's LU as a direct solver backend.

The binding, :mod:`edelweissfe.linsolve.petsclu.petsclu`, imports the optional ``petsc4py`` when a backend is created:
an install without it raises :class:`ImportError` there.
"""

from edelweissfe.linsolve.base import DirectSolverBackend


class PetscLUBackend(DirectSolverBackend):
    """PETSc's LU through :class:`~edelweissfe.linsolve.petsclu.petsclu.PetscLUFactorization`: the factorization in
    :meth:`factorize`, the solves in :meth:`solveFactorized`. Takes no options."""

    def __init__(self):
        from edelweissfe.linsolve.petsclu.petsclu import PetscLUFactorization

        self._factorization = PetscLUFactorization()

    def factorize(self, A) -> None:
        self._factorization.factorize(A)

    def solveFactorized(self, b):
        return self._factorization.solve(b)

    def releaseFactorization(self) -> None:
        self._factorization.release()
