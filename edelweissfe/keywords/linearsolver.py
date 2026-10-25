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

"""``*linearSolver``: defines a linear solver, as ``*material`` defines a material.

The keyword line holds only its identity, ``name`` and ``type``; the data lines are the options of that type, one
``key=value`` per line, declared by the type's own schema (an unknown option is an error). A ``*solver`` names the
block in its ``linsolver=`` option::

    *linearSolver, name=fast, type=blockamg
    rigidBodyCoarseLevel=false

    *solver, solver=NIST, name=theSolver
    linsolver=fast

A ``linsolver=`` naming a type instead of a block (``linsolver=pardiso``) means that type with its defaults, so a block
must not be named like a type. The types are those of the ``linsolver`` registry category, see
:mod:`edelweissfe.linsolve.types`.
"""

from __future__ import annotations

from dataclasses import dataclass

from edelweissfe.keywords.base.keywordbase import KeywordBase
from edelweissfe.utils.schema import datalineField, schemaField


@dataclass(frozen=True)
class LinearSolverKeywordSchema:
    """Options and dataline payload of the ``*linearSolver`` keyword."""

    name: str | None = schemaField(
        description="name of the linear solver, named by a *solver's linsolver=", dtype=str, default=None, required=True
    )
    type: str | None = schemaField(
        description="type of the linear solver: custom, blockamg, pardiso, superlu, umfpack, klu, petsclu, mumps, ...",
        dtype=str,
        default=None,
        required=True,
    )
    datalines: list | None = datalineField(description="the options of the type, one key=value per line")


class LinearSolverKeyword(KeywordBase):
    """``*linearSolver``: define a linear solver."""

    #: Schema class describing this keyword's options and dataline payload.
    schema = LinearSolverKeywordSchema

    keywordName = "linearSolver"
    keywordDescription = "define a linear solver"
