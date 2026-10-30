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
MUMPS, a parallel sparse direct solver: :class:`MumpsFactorization` runs its analysis and factorization once (jobs 1
and 2) and its solve (job 3) for as many right-hand sides as needed.
"""
import numpy as np
from mumps import DMumpsContext


class MumpsFactorization:
    """The MUMPS factorization of the last :meth:`factorize`, for :meth:`solve`. Owns the MUMPS instance, which it
    destroys when it factorizes again or goes."""

    def __init__(self):
        self._context = None
        # the matrix the factorization belongs to: MUMPS keeps pointers to its arrays
        self._A = None

    def release(self) -> None:
        """Destroy the MUMPS instance, and with it the factorization (nothing to destroy before the first)."""
        if self._context is not None:
            self._context.destroy()
            self._context = None
        self._A = None

    def __del__(self):
        self.release()

    def factorize(self, A) -> None:
        """Analyse and factorize the square sparse matrix ``A``.

        Raises
        ------
        numpy.linalg.LinAlgError
            If MUMPS reports an error (e.g. a singular matrix): the bindings raise it as a ``RuntimeError`` with
            MUMPS's error code.
        """
        self.release()
        self._A = A.tocsr(copy=True)
        context = DMumpsContext()
        context.set_silent()
        try:
            if context.myid == 0:
                context.set_centralized_sparse(self._A)
            context.run(job=1)
            if context.myid == 0:
                context.set_centralized_assembled_values(self._A.data)
            context.run(job=2)
        except RuntimeError as error:
            context.destroy()
            raise np.linalg.LinAlgError("MUMPS: {:}".format(error)) from error
        self._context = context

    def solve(self, b):
        """``A^-1 b`` with the factorization of the last :meth:`factorize`, for one right-hand side (a vector) or
        several (the columns of a matrix); ``b`` is not changed."""
        x = np.reshape(np.array(b, dtype=float, copy=True), (self._A.shape[0], -1), order="F")
        for i in range(x.shape[1]):
            # MUMPS overwrites the right-hand side with the solution
            rhs = x[:, i].copy()
            if self._context.myid == 0:
                self._context.set_rhs(rhs)
            self._context.run(job=3)
            x[:, i] = rhs
        return np.reshape(x, np.shape(b), order="F")
