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

"""Interface to the UMFPACK sparse LU solver, through SciPy.

Like its ``superlu`` sibling this subpackage exists only to give the solver a *module* of its own:
it used to be an inline ``lambda A, b: spsolve(A, b, use_umfpack=True)`` inside
``config/linsolve.py``'s ``if/elif`` chain, which the L3 registry's ``"module.path:Attr"`` dotted
strings cannot address (``PLAN_INPUT_SYSTEM.md`` §9).

Note that ``use_umfpack=True`` is a *request*, not a guarantee: SciPy silently falls back to
SuperLU when ``scikit-umfpack`` is not installed, so this name never fails to produce a working
solver -- it just may not produce an UMFPACK-backed one. That is pre-existing SciPy behaviour and
is deliberately left as it was.
"""

from collections.abc import Callable


def createSolver(opts) -> Callable:
    """Create an UMFPACK-backed linear solver.

    The factory the ``linsolver`` registry category resolves for the name ``umfpack`` (see
    ``PLAN_INPUT_SYSTEM.md`` §9): every ``linsolve`` subpackage exposes this one signature, so that
    a third party can contribute a linear solver through an entry point.

    Parameters
    ----------
    opts
        The linear-solver options parsed from the solver's ``linsolverConfigFile``. Not consulted:
        UMFPACK is used with SciPy's defaults and takes no options here. It is still accepted so
        that every ``createSolver`` presents the identical signature.

    Returns
    -------
    Callable
        A callable ``(A, b) -> x`` solving ``A x = b`` via
        :func:`scipy.sparse.linalg.spsolve` with ``use_umfpack=True``.
    """

    # Imported inside the function body, not at module scope -- see the note in the `superlu`
    # sibling for why every one of the nine factories does it this way.
    from scipy.sparse.linalg import spsolve

    return lambda A, b: spsolve(A, b, use_umfpack=True)
