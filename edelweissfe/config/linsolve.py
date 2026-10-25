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

from edelweissfe.config import registry
from edelweissfe.linsolve.base import LinearSolver


def getDefaultLinSolver() -> LinearSolver:
    """Get the linear solver to use when a step declares no ``linsolver`` option.

    Returns
    -------
    LinearSolver
        ``pardiso`` if its optional extension was built, else SciPy's own sparse LU (``superlu``), with their defaults.
    """

    try:
        # creating a PARDISO solver raises ImportError if the extension is not installed
        return getLinSolverByName("pardiso", {})
    except ImportError:
        return getLinSolverByName("superlu", {})


def getLinSolverByName(linsolverName: str, options: dict) -> LinearSolver:
    """Return a linear solver of the type ``linsolverName``, configured with ``options``.

    Parameters
    ----------
    linsolverName
        The type of the linear solver, case-insensitively (e.g. ``"pardiso"``, ``"blockamg"``; registry category
        ``linsolver``, see :mod:`edelweissfe.linsolve.types`), or a solver script, ``script:<file>.py`` or
        ``script:<file>.py:<name>`` (a composed solver, see :mod:`edelweissfe.linsolve.composition`).
    options
        The data lines of that type by name, e.g. ``{"reuseSymbolicFactorization": True}``; ``{}`` for its defaults.

    Returns
    -------
    LinearSolver
        A callable ``(A, b) -> x`` solving ``A x = b``.

    Raises
    ------
    edelweissfe.config.registry.RegistryLookupError
        If no linear solver is registered under ``linsolverName``.
    ValueError
        If ``options`` names an option the type does not have, or a value it cannot take.
    ImportError
        If the requested solver's optional backend is not available in this installation (see
        :func:`getDefaultLinSolver`, which relies on it).
    """

    options = dict(options or {})
    if linsolverName.startswith("script:"):
        return _solverFromScript(linsolverName.removeprefix("script:"), options)

    solverType, _ = registry.lookup("linsolver", linsolverName)
    return solverType.fromOptions(options)


def _solverFromScript(reference: str, opts) -> LinearSolver:
    """The composed solver of a solver script, named ``<file>.py`` (its global ``linearSolver``) or
    ``<file>.py:<name>``; see :mod:`edelweissfe.linsolve.composition.solver`."""
    from edelweissfe.linsolve.composition.solver import loadSolverScript

    path, name = (reference, "linearSolver") if reference.endswith(".py") else reference.rsplit(":", 1)
    return loadSolverScript(path, name, opts)
