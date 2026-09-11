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

"""Registry-facing factory for SciPy's own sparse LU solver (SuperLU).

There is no extension to build and no optional dependency behind it (SciPy is a hard requirement),
so unlike its siblings this one is always importable, which is also what makes it the fallback of
:func:`~edelweissfe.config.linsolve.getDefaultLinSolver`.
"""

from collections.abc import Callable

from edelweissfe.linsolve.base import LinearSolver


class SuperLUSolver(LinearSolver):
    """SciPy's SuperLU-backed direct solver. Callable as ``(A, b) -> x``.

    Takes no options; ``setJournal``/``setFieldStructure`` are inherited no-ops (this solver has no
    field-split structure and does not log).
    """

    def __call__(self, A, b):
        # Imported inside the call, not at module scope: several linsolve backends are optional and
        # genuinely absent in some installs, and `getDefaultLinSolver` relies on catching the
        # resulting ImportError at construction time (see createSolver below). SciPy is not optional,
        # but the convention is kept uniform across all factories.
        from scipy.sparse.linalg import spsolve

        return spsolve(A, b, use_umfpack=False)


def createSolver(opts) -> Callable:
    """Create a SuperLU-backed linear solver.

    The factory the ``linsolver`` registry category resolves for the name ``superlu``: every ``linsolve`` subpackage
    exposes this one signature, so that a third party can contribute a linear solver through an entry point.

    Parameters
    ----------
    opts
        The linear-solver options parsed from the solver's ``linsolverConfigFile``. Not consulted:
        SuperLU is used with SciPy's defaults and takes no options here. It is still accepted so
        that every ``createSolver`` presents the identical signature.

    Returns
    -------
    Callable
        A :class:`SuperLUSolver`, callable as ``(A, b) -> x``, solving ``A x = b`` via
        :func:`scipy.sparse.linalg.spsolve` with ``use_umfpack=False``.
    """
    return SuperLUSolver()
