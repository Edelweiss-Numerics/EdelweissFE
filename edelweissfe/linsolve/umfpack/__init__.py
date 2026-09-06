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

"""Registry-facing factory for the UMFPACK sparse LU solver, through SciPy.

``use_umfpack=True`` is a *request*, not a guarantee: SciPy silently falls back to SuperLU when
``scikit-umfpack`` is not installed, so this name never fails to produce a working solver -- it
just may not produce an UMFPACK-backed one.
"""

from collections.abc import Callable

from edelweissfe.linsolve.base import LinearSolver


class UMFPACKSolver(LinearSolver):
    """SciPy's UMFPACK-backed direct solver. Callable as ``(A, b) -> x``.

    Takes no options; ``setJournal``/``setFieldStructure`` are inherited no-ops (this solver has no
    field-split structure and does not log).
    """

    def __call__(self, A, b):
        # Imported inside the call, not at module scope -- see the note in the `superlu` sibling for
        # why every one of the factories does it this way.
        from scipy.sparse.linalg import spsolve

        return spsolve(A, b, use_umfpack=True)


def createSolver(opts) -> Callable:
    """Create an UMFPACK-backed linear solver.

    The factory the ``linsolver`` registry category resolves for the name ``umfpack``: every ``linsolve``
    subpackage exposes this one signature, so that a third party can contribute a linear solver through an
    entry point.

    Parameters
    ----------
    opts
        The linear-solver options parsed from the solver's ``linsolverConfigFile``. Not consulted:
        UMFPACK is used with SciPy's defaults and takes no options here. It is still accepted so
        that every ``createSolver`` presents the identical signature.

    Returns
    -------
    Callable
        A :class:`UMFPACKSolver`, callable as ``(A, b) -> x``, solving ``A x = b`` via
        :func:`scipy.sparse.linalg.spsolve` with ``use_umfpack=True``.
    """
    return UMFPACKSolver()
