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
#  Alexander Dummer alexander.dummer@uibk.ac.at
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

"""
PETSc's sparse LU, through petsc4py: :class:`PetscLUFactorization` factorizes a matrix once and solves for as many
right-hand sides as needed.
"""
import numpy as np
from petsc4py import PETSc


class PetscLUFactorization:
    """The PETSc LU factorization of the last :meth:`factorize` (a ``preonly`` KSP with an ``lu`` PC, set up once), for
    :meth:`solve`."""

    def __init__(self):
        self._ksp = None
        self._matrix = None

    def release(self) -> None:
        """Destroy the KSP and the matrix, and with them the factorization (nothing to destroy before the first)."""
        if self._ksp is not None:
            self._ksp.destroy()
            self._matrix.destroy()
            self._ksp = self._matrix = None

    def __del__(self):
        self.release()

    def factorize(self, A) -> None:
        """Factorize the square sparse matrix ``A``.

        Raises
        ------
        numpy.linalg.LinAlgError
            If the LU factorization fails (e.g. a zero pivot of a singular matrix): PETSc records that as the failure
            of the PC, which is checked here instead of surfacing as a diverged solve later.
        """
        self.release()
        A = A.tocsr()
        matrix = PETSc.Mat().createAIJ(size=A.shape, csr=(A.indptr, A.indices, A.data))
        ksp = PETSc.KSP().create()
        ksp.setOperators(matrix)
        ksp.setType("preonly")
        pc = ksp.getPC()
        pc.setType("lu")
        try:
            ksp.setUp()
        except PETSc.Error as error:
            ksp.destroy()
            matrix.destroy()
            raise np.linalg.LinAlgError("PETSc LU: {:}".format(error)) from error
        if pc.getFailedReason() != PETSc.PC.FailedReason.NOERROR:
            reason = pc.getFailedReason()
            ksp.destroy()
            matrix.destroy()
            raise np.linalg.LinAlgError("PETSc LU: the factorization failed (PC failed reason {:})".format(reason))
        self._ksp, self._matrix = ksp, matrix

    def solve(self, b):
        """``A^-1 b`` with the factorization of the last :meth:`factorize`, for one right-hand side (a vector) or
        several (the columns of a matrix)."""
        rhs = np.reshape(np.asarray(b, dtype=float), (self._matrix.getSize()[0], -1))
        x = self._matrix.createVecLeft()
        current = self._matrix.createVecRight()
        solution = np.zeros_like(rhs)
        for i in range(rhs.shape[1]):
            current.setArray(rhs[:, i])
            self._ksp.solve(current, x)
            solution[:, i] = x.getArray()
        x.destroy()
        current.destroy()
        return np.reshape(solution, np.shape(b))
