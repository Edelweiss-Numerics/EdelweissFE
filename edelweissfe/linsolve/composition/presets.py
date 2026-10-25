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
"""Presets: plain functions that build a composed solver.

A preset is nothing but a function returning a tree, so there is no preset machinery: read it to see what it
builds, copy it into a solver script to change it.
"""

from edelweissfe.linsolve.composition.amg import AMG
from edelweissfe.linsolve.composition.base import Preconditioner
from edelweissfe.linsolve.composition.combinators import (
    Multiplicative,
    OnField,
    gaussSeidelOrder,
)
from edelweissfe.linsolve.composition.diagonalscaling import DiagonalScaling
from edelweissfe.linsolve.composition.krylov import (
    LGMRES,
    GapCompensation,
    RebuildPolicy,
)
from edelweissfe.linsolve.composition.twolevel import RigidBodyModes, TwoLevel
from edelweissfe.linsolve.nullspace import carriesRigidBodyModes


def blockAMGFieldPreconditioner(
    field, context, rigidBodyCoarseLevel: bool = True, dropTolerance: float = 0.0, lumpDroppedEntries: bool = False
):
    """The preconditioner of a field in the ``blockamg`` preset: a two-level cycle on the rigid-body modes
    for a 3D displacement field (with node coordinates), one AMG hierarchy otherwise."""
    isDisplacement3D = field.dimension == 3 and carriesRigidBodyModes(field)
    if rigidBodyCoarseLevel and isDisplacement3D and context.coordinates(field.name) is None:
        context.message(
            "info",
            "blockAMG: the field '{:}' has no node coordinates (no model, and no nodeCoordinates given): it gets AMG() on "
            "the translations instead of the two-level cycle on its rigid-body modes".format(field.name),
        )
    elif rigidBodyCoarseLevel and isDisplacement3D:
        return TwoLevel(
            coarseSpace=RigidBodyModes(), dropTolerance=dropTolerance, lumpDroppedEntries=lumpDroppedEntries
        )
    return AMG(dropTolerance=dropTolerance, lumpDroppedEntries=lumpDroppedEntries)


def blockAMG(
    rigidBodyCoarseLevel: bool = True,
    tolerance=None,
    maxIterations: int = 100,
    sweeps: int = 1,
    symmetric: bool = False,
    rebuild: RebuildPolicy = None,
    krylovOnScaledSystem: bool = False,
    continuations: int = 2,
    gapCompensation: GapCompensation = None,
    dropTolerance: float = 0.0,
    lumpDroppedEntries: bool = False,
) -> Preconditioner:
    """The ``blockamg`` linear solver (formerly the monolithic ``BlockAMGSolver``, whose numbers it reproduces):
    block Gauss--Seidel over the fields of the diagonally scaled matrix, inside LGMRES::

        LGMRES(
            DiagonalScaling(
                Multiplicative([OnField(f, M_f) for every field f])   # M_f: TwoLevel(RigidBodyModes()) for a
            )                                                         # 3D displacement field, else AMG()
        )

    The corrections are made at the first set-up, for the fields of the model (:class:`~edelweissfe.linsolve.
    composition.combinators.Multiplicative` takes a function of the context). A tree, not a solver: a solver script
    binds it, ``linearSolver = blockAMG()``.

    Parameters
    ----------
    rigidBodyCoarseLevel
        Whether a 3D displacement field gets the two-level cycle on its rigid-body modes (else one AMG hierarchy, on
        the translations).
    tolerance
        The tolerance per solve (default: the Eisenstat--Walker forcing tolerance).
    maxIterations
        The outer iteration cap.
    sweeps, symmetric
        The block Gauss--Seidel sweeps, see :func:`~edelweissfe.linsolve.composition.combinators.gaussSeidelOrder`.
    rebuild
        When the preconditioner is rebuilt.
    krylovOnScaledSystem
        Run LGMRES on the scaled matrix, ``DiagonalScaling(LGMRES(Multiplicative(...)))``, with its true-residual
        ``continuations`` and ``gapCompensation`` (see :class:`~edelweissfe.linsolve.composition.krylov.KrylovSolver`).
    continuations, gapCompensation
        See :class:`~edelweissfe.linsolve.composition.krylov.KrylovSolver`; only with ``krylovOnScaledSystem``.
    dropTolerance, lumpDroppedEntries
        Build every field's preconditioner on its filtered block, see
        :func:`~edelweissfe.linsolve.composition.amg.droppedSmallEntries`.
    """

    def fieldCorrections(context):
        corrections = [
            OnField(
                field.name,
                blockAMGFieldPreconditioner(field, context, rigidBodyCoarseLevel, dropTolerance, lumpDroppedEntries),
            )
            for field in context.fields
        ]
        return gaussSeidelOrder(corrections, sweeps, symmetric)

    blockGaussSeidel = Multiplicative(fieldCorrections)

    def lgmres(preconditioner):
        return LGMRES(
            preconditioner,
            tolerance=tolerance,
            maxIterations=maxIterations,
            rebuild=rebuild,
            continuations=continuations,
            gapCompensation=gapCompensation,
        )

    if krylovOnScaledSystem:
        return DiagonalScaling(lgmres(blockGaussSeidel))
    return lgmres(DiagonalScaling(blockGaussSeidel))
