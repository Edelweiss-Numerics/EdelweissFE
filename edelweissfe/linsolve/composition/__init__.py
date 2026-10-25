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
"""Composed linear solvers: a solver as a tree of small objects, built with plain constructors in a Python script.

A solver script next to the input file::

    from edelweissfe.linsolve.composition import AMG, LGMRES, DiagonalScaling, FieldSplit, RigidBodyModes, TwoGrid

    displacement = TwoGrid(coarseSpace=RigidBodyModes())
    fieldSplit = FieldSplit({"displacement": displacement, "nonlocal damage": AMG()})
    linearSolver = LGMRES(DiagonalScaling(fieldSplit))

and in the input file ``linsolver=script:solver.py:linearSolver``. ``print(linearSolver)`` shows the tree; see
:mod:`~edelweissfe.linsolve.composition.views` for the other views, and
:mod:`~edelweissfe.linsolve.composition.base` for how a component works.
"""

from edelweissfe.linsolve.composition.amg import AMG
from edelweissfe.linsolve.composition.base import (
    Description,
    MatrixChange,
    Preconditioner,
    SetUpContext,
    Statistics,
    Step,
)
from edelweissfe.linsolve.composition.combinators import (
    Additive,
    CoarseCorrection,
    Jacobi,
    Multiplicative,
    OnField,
)
from edelweissfe.linsolve.composition.diagonalscaling import DiagonalScaling
from edelweissfe.linsolve.composition.direct import DirectSolver
from edelweissfe.linsolve.composition.fieldsplit import FieldSplit
from edelweissfe.linsolve.composition.krylov import (
    LGMRES,
    FixedTolerance,
    ForcingTolerance,
    RebuildPolicy,
)
from edelweissfe.linsolve.composition.presets import blockAMG
from edelweissfe.linsolve.composition.schurcomplement import SchurComplement
from edelweissfe.linsolve.composition.solver import (
    ComposedLinearSolver,
    loadSolverScript,
)
from edelweissfe.linsolve.composition.systemmatrix import SystemMatrix
from edelweissfe.linsolve.composition.twogrid import (
    P1Interpolation,
    RigidBodyModes,
    TwoGrid,
)

__all__ = [
    "AMG",
    "Additive",
    "CoarseCorrection",
    "ComposedLinearSolver",
    "Description",
    "DiagonalScaling",
    "DirectSolver",
    "FieldSplit",
    "FixedTolerance",
    "ForcingTolerance",
    "Jacobi",
    "LGMRES",
    "MatrixChange",
    "Multiplicative",
    "OnField",
    "P1Interpolation",
    "Preconditioner",
    "RebuildPolicy",
    "RigidBodyModes",
    "SchurComplement",
    "SetUpContext",
    "Statistics",
    "Step",
    "SystemMatrix",
    "TwoGrid",
    "blockAMG",
    "loadSolverScript",
]
