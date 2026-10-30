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

from dataclasses import dataclass

from edelweissfe.config import registry
from edelweissfe.linsolve.base import LinearSolver
from edelweissfe.utils.caseinsensitivedict import CaseInsensitiveDict
from edelweissfe.utils.misc import parseDatalinesToArgsAndKwargs
from edelweissfe.utils.schema import buildSchemaFromOptions


@dataclass(frozen=True)
class LinearSolverDefinition:
    """A ``*linearSolver`` block: its type and the configuration its data lines give (checked when the input file is
    read). A nonlinear solver creates a linear solver from it (:meth:`create`) and keeps it for all the steps that
    name it.

    Attributes
    ----------
    solverType
        The type (a :class:`~edelweissfe.linsolve.types.LinearSolverType`).
    configuration
        The type's schema, built from the data lines.
    """

    solverType: type
    configuration: object

    def create(self) -> LinearSolver:
        """A new linear solver of this definition."""
        return self.solverType.create(self.configuration)


def linearSolverDefinitionsFromInputFile(inputfile: dict) -> CaseInsensitiveDict:
    """The ``*linearSolver`` blocks of an input file, by name.

    Raises
    ------
    ValueError
        For a block named like a type (``linsolver=<type>`` means that type with its defaults), a name used twice, a
        data line that is not ``key=value``, or an option the type does not have.
    """
    definitions = CaseInsensitiveDict()
    for block in inputfile["linearSolver"]:
        name, typeName = block["name"], block["type"]
        if name.casefold() in (known.casefold() for known in registry.availableNames("linsolver")):
            raise ValueError(
                "*linearSolver, name={:}: a linear solver must not be named like a type, since linsolver={:} means "
                "that type with its defaults".format(name, name)
            )
        if name in definitions:
            raise ValueError("*linearSolver, name={:} is defined twice".format(name))
        args, options = parseDatalinesToArgsAndKwargs(block["datalines"])
        if args:
            raise ValueError("*linearSolver, name={:}: data lines are key=value options, got {:}".format(name, args))
        solverType, schema = registry.lookup("linsolver", typeName)
        try:
            configuration = buildSchemaFromOptions(schema, options)
        except ValueError as error:
            raise ValueError("*linearSolver, name={:}, type={:}: {:}".format(name, typeName, error)) from error
        definitions[name] = LinearSolverDefinition(solverType, configuration)
    return definitions


def createLinearSolver(linsolver: str, definitions: dict) -> LinearSolver:
    """The linear solver a nonlinear solver's ``linsolver=`` names: a ``*linearSolver`` block, else a type with its
    defaults, else (empty) the default linear solver.

    Parameters
    ----------
    linsolver
        The name of a block of ``definitions``, or of a type; ``""`` for the default.
    definitions
        The ``*linearSolver`` blocks by name (:func:`linearSolverDefinitionsFromInputFile`).
    """
    if not linsolver:
        return getDefaultLinSolver()
    if linsolver in definitions:
        return definitions[linsolver].create()
    if linsolver.casefold() not in (known.casefold() for known in registry.availableNames("linsolver")):
        raise ValueError(
            "linsolver={:} names neither a *linearSolver block ({:}) nor a type of linear solver ({:})".format(
                linsolver,
                ", ".join(definitions) or "none defined",
                ", ".join(sorted(registry.availableNames("linsolver"))),
            )
        )
    return getLinSolverByName(linsolver, {})


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
        The type of the linear solver, case-insensitively (e.g. ``"pardiso"``, ``"blockamg"``, ``"custom"``; registry
        category ``linsolver``, see :mod:`edelweissfe.linsolve.types`).
    options
        The data lines of that type by name, e.g. ``{"reuseSymbolicFactorization": True}`` or
        ``{"script": "solver.py"}``; ``{}`` for its defaults.

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

    solverType, _ = registry.lookup("linsolver", linsolverName)
    return solverType.fromOptions(dict(options))
