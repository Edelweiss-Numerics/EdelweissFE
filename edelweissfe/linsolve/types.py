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

"""The types of linear solver: what ``linsolver=<type>`` (and, with its data lines, ``*linearSolver, type=<type>``)
names.

Every type is a composed solver behind the one adapter,
:class:`~edelweissfe.linsolve.composition.solver.ComposedLinearSolver`: a type is the schema of its data lines and
the tree they define (:meth:`LinearSolverType.tree`). A direct solver is a tree of one leaf, ``type=pardiso`` is
``DirectSolver("pardiso")``. So what the adapter does -- the verbosity, the dumps, the "linear solve" column -- works
the same for every type, and the data lines of :class:`LinearSolverSchema` are those of every type.
"""

from dataclasses import dataclass

from edelweissfe.linsolve.composition.base import Preconditioner
from edelweissfe.linsolve.composition.direct import DirectSolver
from edelweissfe.linsolve.composition.solvedumps import SolveDumps
from edelweissfe.linsolve.composition.solver import ComposedLinearSolver
from edelweissfe.utils.schema import (
    OptionSchemaProvider,
    buildSchemaFromOptions,
    schemaField,
)


@dataclass(frozen=True)
class LinearSolverSchema:
    """The data lines of every type: how its solves are reported and dumped."""

    verbosity: str = schemaField(
        description=(
            "silent, warning (a warning for a slow or unconverged solve), info (one line per solve, the structure of "
            "the solver, and the 'linear solve' column of the nonlinear solver) or debug (also why a preconditioner "
            "was rebuilt or reused)."
        ),
        dtype=str,
        default="warning",
    )
    warnOuterIterations: int = schemaField(
        description="A solve with more outer Krylov iterations than this is warned about, and is a slow solve.",
        dtype=int,
        default=100,
    )
    residualGrowthFactor: float = schemaField(
        description="A right-hand side this many times larger than the previous one marks a new increment.",
        dtype=float,
        default=4.0,
    )
    dumps: str = schemaField(
        description=(
            "Which solves are written to disk for an offline replay: none, slow (above warnOuterIterations, or "
            "stopped at the iteration cap), every, or the numbers of the solves separated by spaces, e.g. '12 13 14'."
        ),
        dtype=str,
        default="none",
    )
    dumpDirectory: str = schemaField(description="Where the dumps go.", dtype=str, default="linsolveDumps")
    maxDumps: int = schemaField(description="The most systems dumped in the run (a disk guard).", dtype=int, default=10)
    dumpContextSolves: int = schemaField(
        description="How many of the solves preceding a dumped one are dumped with it.", dtype=int, default=0
    )


def solveDumpsOf(configuration: LinearSolverSchema) -> "SolveDumps | None":
    """The dumps the data lines ask for, or ``None``."""
    if configuration.dumps == "none":
        return None
    trigger = configuration.dumps if configuration.dumps in ("slow", "every") else configuration.dumps.split()
    try:
        return SolveDumps(
            configuration.dumpDirectory,
            trigger=trigger,
            maxDumps=configuration.maxDumps,
            contextSolves=configuration.dumpContextSolves,
        )
    except ValueError as error:
        raise ValueError(
            "linear solver: dumps={!r}: none, slow, every, or solve numbers separated by spaces ({:})".format(
                configuration.dumps, error
            )
        ) from error


class LinearSolverType(OptionSchemaProvider):
    """A type of linear solver: the schema of its data lines, and the composed solver they define."""

    schema = LinearSolverSchema

    @classmethod
    def tree(cls, configuration) -> Preconditioner:
        """The composed solver of these data lines."""
        raise NotImplementedError

    @classmethod
    def fromOptions(cls, options: dict) -> ComposedLinearSolver:
        """The linear solver of these options (the data lines, by name; unknown names raise)."""
        return cls.create(buildSchemaFromOptions(cls.schema, options))

    @classmethod
    def create(cls, configuration) -> ComposedLinearSolver:
        """The linear solver of these data lines: the tree behind the adapter."""
        return ComposedLinearSolver(
            cls.tree(configuration),
            residualGrowthFactor=configuration.residualGrowthFactor,
            verbosity=configuration.verbosity,
            warnOuterIterations=configuration.warnOuterIterations,
            dumps=solveDumpsOf(configuration),
        )


class DirectSolverType(LinearSolverType):
    """A direct solver: the tree of one :class:`~edelweissfe.linsolve.composition.direct.DirectSolver` leaf on the
    backend :attr:`backend`, factorized anew on every solve."""

    #: The registered name of the backend (registry category ``directsolver``).
    backend: str = None

    @classmethod
    def tree(cls, configuration) -> Preconditioner:
        return DirectSolver(cls.backend)


class SuperLU(DirectSolverType):
    """``superlu``: SciPy's own sparse LU, always available."""

    backend = "superlu"


class UMFPACK(DirectSolverType):
    """``umfpack``: UMFPACK through SciPy (SuperLU if ``scikit-umfpack`` is missing)."""

    backend = "umfpack"


@dataclass(frozen=True)
class PardisoSchema(LinearSolverSchema):
    """The data lines of ``pardiso``."""

    reuseSymbolicFactorization: bool = schemaField(
        description=(
            "Keep PARDISO's reordering and symbolic factorization while the sparsity pattern stays the same. Off by "
            "default: it has been observed to give silently wrong results for some coupled problems."
        ),
        dtype=bool,
        default=False,
    )


class Pardiso(DirectSolverType):
    """``pardiso``: Intel MKL PARDISO (the default linear solver, if built)."""

    schema = PardisoSchema
    backend = "pardiso"

    @classmethod
    def tree(cls, configuration: PardisoSchema) -> Preconditioner:
        return DirectSolver(cls.backend, reuseSymbolicFactorization=configuration.reuseSymbolicFactorization)


class PanuaPardiso(DirectSolverType):
    """``panuapardiso``: Panua PARDISO, if built."""

    backend = "panuapardiso"


class KLU(DirectSolverType):
    """``klu``: SuiteSparse's KLU, if built."""

    backend = "klu"


class PetscLU(DirectSolverType):
    """``petsclu``: PETSc's LU, if ``petsc4py`` is installed."""

    backend = "petsclu"


class Mumps(DirectSolverType):
    """``mumps``: MUMPS, if its Python bindings are installed."""

    backend = "mumps"
