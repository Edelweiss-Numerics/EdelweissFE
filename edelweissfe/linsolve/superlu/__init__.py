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

"""Interface to SciPy's own sparse LU solver (SuperLU).

This subpackage exists only to give that solver a *module* of its own. It used to be an inline
``lambda A, b: spsolve(A, b, use_umfpack=False)`` inside ``config/linsolve.py``'s ``if/elif``
chain, i.e. a closure with no module-level name for the L3 registry's ``"module.path:Attr"``
dotted string to point at -- which is why ``linsolver`` was the one category P1 had to leave out.
There is no extension to build and no optional dependency behind it (SciPy is a hard requirement),
so unlike its siblings this one is always importable, which is also what makes it the fallback of
:func:`~edelweissfe.config.linsolve.getDefaultLinSolver`.
"""

from collections.abc import Callable


def createSolver(opts) -> Callable:
    """Create a SuperLU-backed linear solver.

    The factory the ``linsolver`` registry category resolves for the name ``superlu`` (see
    ``PLAN_INPUT_SYSTEM.md`` §9): every ``linsolve`` subpackage exposes this one signature, so that
    a third party can contribute a linear solver through an entry point.

    Parameters
    ----------
    opts
        The linear-solver options parsed from the solver's ``linsolverConfigFile``. Not consulted:
        SuperLU is used with SciPy's defaults and takes no options here. It is still accepted so
        that every ``createSolver`` presents the identical signature.

    Returns
    -------
    Callable
        A callable ``(A, b) -> x`` solving ``A x = b`` via
        :func:`scipy.sparse.linalg.spsolve` with ``use_umfpack=False``.
    """

    # Imported inside the function body, not at module scope: several linsolve backends are
    # optional and genuinely absent in some installs, and `getDefaultLinSolver` relies on catching
    # the resulting ImportError. A module-scope import would turn "backend not built" from a
    # graceful fallback into an import error for anyone merely resolving a registry name. SciPy is
    # not optional, but the convention is kept uniform across all nine factories.
    from scipy.sparse.linalg import spsolve

    return lambda A, b: spsolve(A, b, use_umfpack=False)
