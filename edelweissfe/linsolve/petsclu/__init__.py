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

"""Registry-facing factory for PETSc's LU solver.

The implementation lives in :mod:`edelweissfe.linsolve.petsclu.petsclu`, which is pure Python but
imports ``petsc4py`` at module scope -- an optional dependency. The factory below therefore imports
it lazily; see :func:`createSolver`.
"""

from collections.abc import Callable

from edelweissfe.linsolve.base import LinearSolver


class PetscLUSolver(LinearSolver):
    """PETSc's LU direct solver. Callable as ``(A, b) -> x``.

    Takes no options; ``setJournal``/``setFieldStructure`` are inherited no-ops (this solver has no
    field-split structure and does not log).
    """

    def __init__(self, solveFunction):
        self._solveFunction = solveFunction

    def __call__(self, A, b):
        return self._solveFunction(A, b)


def createSolver(opts) -> Callable:
    """Create a PETSc-LU-backed linear solver.

    The factory the ``linsolver`` registry category resolves for the name ``petsclu``: every ``linsolve`` subpackage
    exposes this one signature, so that a third party can contribute a linear solver through an entry point.

    Parameters
    ----------
    opts
        The linear-solver options parsed from the solver's ``linsolverConfigFile``. Not consulted:
        this backend takes no options. It is still accepted so that every ``createSolver`` presents
        the identical signature.

    Returns
    -------
    Callable
        A :class:`PetscLUSolver`, callable as ``(A, b) -> x``, delegating to
        :func:`~edelweissfe.linsolve.petsclu.petsclu.petscluSolve`.

    Raises
    ------
    ImportError
        If the optional ``petsc4py`` dependency is not installed -- raised here, at construction
        time (matching the pre-refactor behaviour), not deferred to the first solve.
    """
    # Imported inside the function body, not at module scope: `petsc4py` is an optional dependency,
    # so at module scope its absence would break anyone who merely resolves a `linsolver` registry
    # name rather than this one.
    from edelweissfe.linsolve.petsclu.petsclu import petscluSolve

    return PetscLUSolver(petscluSolve)
