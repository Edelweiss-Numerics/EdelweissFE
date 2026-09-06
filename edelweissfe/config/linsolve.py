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
        # An empty option mapping matches PARDISO's safe default: symbolic-factorization reuse is
        # opt-in only, see the pardiso factory. If the PARDISO extension is not installed, the
        # factory raises ImportError, which is caught below to fall back to SciPy's superlu.
        return getLinSolverByName("pardiso", {})
    except ImportError:
        return getLinSolverByName("superlu", {})


def getLinSolverByName(linsolverName: str, opts) -> Callable:
    """Return the linear solver registered under ``linsolverName``, configured with ``opts``.

    Resolved through the registry (``linsolver`` category). Each ``linsolve`` subpackage provides
    a module-level ``createSolver(opts) -> Callable[[A, b], x]`` factory, which the registry's
    dotted strings point at and which handles its own option parsing, including the case of a
    non-mapping ``opts``.

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
        If the requested solver's optional backend is not available in this installation. Raised
        by the factory and not caught here; see :func:`getDefaultLinSolver`, which relies on it.
    """

    factory, _ = registry.lookup("linsolver", linsolverName)

    return factory(opts)
