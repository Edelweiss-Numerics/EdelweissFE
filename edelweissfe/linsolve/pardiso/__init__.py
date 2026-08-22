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

"""Registry-facing factory for the Intel MKL PARDISO solver.

The implementation lives in the Cython extension :mod:`edelweissfe.linsolve.pardiso.pardiso`, which
is optional -- it needs MKL at build time. The factory below therefore imports it lazily; see
:func:`createSolver`.
"""

from collections.abc import Callable, Mapping


def createSolver(opts) -> Callable:
    """Create a PARDISO-backed linear solver.

    The factory the ``linsolver`` registry category resolves for the name ``pardiso``: every ``linsolve`` subpackage
    exposes this one signature, so that a third party can contribute a linear solver through an entry point.

    Parameters
    ----------
    opts
        The linear-solver options parsed from the solver's ``linsolverConfigFile``. Only
        ``opts["reuseSymbolicFactorization"]`` is consulted, defaulting to ``False``; ``opts`` is
        tolerated as any non-mapping (the implicit-static solver passes ``""`` when no configuration
        file is given), in which case that default applies.

    Returns
    -------
    Callable
        A :class:`~edelweissfe.linsolve.pardiso.pardiso.PardisoSolver` instance, which is callable
        as ``(A, b) -> x``.

    Raises
    ------
    ImportError
        If the optional ``pardiso`` extension was not built.
        :func:`~edelweissfe.config.linsolve.getDefaultLinSolver` catches this to fall back to SciPy.
    """

    # Imported inside the function body, not at module scope: this extension is optional and
    # genuinely absent in installs built without MKL, and `getDefaultLinSolver` relies on catching
    # the ImportError to fall back to SciPy. At module scope that failure would instead hit anyone
    # who merely resolves this registry name.
    from edelweissfe.linsolve.pardiso.pardiso import PardisoSolver

    # Symbolic-factorization reuse across solves is only correct if the caller
    # can guarantee the sparsity pattern stays genuinely stable for the solver
    # instance's entire lifetime; it has been observed to silently produce wrong
    # (but not NaN, so undetected by the usual failure check) results for some
    # coupled-DOF problems. Off by default; opt in explicitly via
    # opts["reuseSymbolicFactorization"] = True once that has been verified safe
    # for the problem at hand.
    reuseSymbolicFactorization = (
        bool(opts.get("reuseSymbolicFactorization", False)) if isinstance(opts, Mapping) else False
    )

    return PardisoSolver(reuseSymbolicFactorization=reuseSymbolicFactorization)
