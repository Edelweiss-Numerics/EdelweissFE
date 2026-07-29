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
"""
Created on Sat Feb 10 10:27:25 2018

@author: Matthias Neuner
"""

from collections.abc import Callable

from edelweissfe.config import registry


def getDefaultLinSolver() -> Callable:
    """Get the linear solver to use when a step declares no ``linsolver`` option.

    Returns
    -------
    Callable
        A callable ``(A, b) -> x``: PARDISO if its optional extension was built, else SciPy's own
        sparse LU (``superlu``).
    """

    try:
        # symbolic-factorization reuse is opt-in only, see the pardiso factory; passing an empty
        # option mapping here intentionally matches that safe default.
        #
        # Routed through the same seam as a named request rather than constructing PardisoSolver
        # directly, so there is exactly one place that knows how a pardiso solver is built. The
        # ImportError semantics are unchanged: the factory imports the extension inside its own
        # body, so an install without it still raises ImportError from this call, and still falls
        # back to scipy -- which is now the `superlu` factory rather than a duplicate lambda.
        return getLinSolverByName("pardiso", {})
    except ImportError:
        return getLinSolverByName("superlu", {})


def getLinSolverByName(linsolverName: str, opts) -> Callable:
    """Get the linear solver registered under ``linsolverName``, configured with ``opts``.

    Resolved through the L3 registry (``linsolver`` category), which replaces a nine-arm ``if/elif``
    chain of local imports. That chain could only ever name solvers living inside this package, so an
    external package -- EdelweissMeshfree, a plugin -- had no way to contribute one; and its nine
    arms had four different shapes (inline SciPy lambdas, an option-constructed class, plain
    module-level functions, and bound methods of option-constructed objects), which is why this was
    the last category folded in. The uniform shape they collapse to is a module-level
    ``createSolver(opts) -> Callable[[A, b], x]`` factory per ``linsolve`` subpackage, which is what
    the registry's dotted strings point at (``PLAN_INPUT_SYSTEM.md`` §9). Each factory keeps the
    option handling that used to live in this function's corresponding arm, including the tolerance
    for a non-mapping ``opts``.

    An unknown name now raises :class:`~edelweissfe.config.registry.RegistryLookupError` -- a
    ``LookupError``, naming the available solvers and suggesting a similar one -- instead of
    ``AttributeError("invalid linear solver ... requested")``.

    Parameters
    ----------
    linsolverName
        The name of the linear solver, case-insensitively (e.g. ``"pardiso"``, ``"amgcl"``).
    opts
        The linear-solver options parsed from the solver's ``linsolverConfigFile``, passed to the
        factory unchanged. Not necessarily a mapping: the implicit-static solver passes ``""`` when
        no configuration file is given.

    Returns
    -------
    Callable
        A callable ``(A, b) -> x`` solving ``A x = b``.

    Raises
    ------
    edelweissfe.config.registry.RegistryLookupError
        If no linear solver is registered under ``linsolverName``.
    ImportError
        If the requested solver's optional backend is not available in this installation. Raised by
        the factory, not by the lookup, and deliberately not caught here -- see
        :func:`getDefaultLinSolver`, which relies on it.
    """

    factory, _ = registry.lookup("linsolver", linsolverName)

    return factory(opts)
