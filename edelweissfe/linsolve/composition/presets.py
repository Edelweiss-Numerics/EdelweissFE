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

from edelweissfe.config.phenomena import phenomena
from edelweissfe.linsolve.composition.amg import AMG
from edelweissfe.linsolve.composition.combinators import (
    Multiplicative,
    OnField,
    gaussSeidelOrder,
)
from edelweissfe.linsolve.composition.diagonalscaling import DiagonalScaling
from edelweissfe.linsolve.composition.krylov import LGMRES, RebuildPolicy
from edelweissfe.linsolve.composition.twogrid import RigidBodyModes, TwoGrid


def blockAMGFieldPreconditioner(field, context, rigidBodyCoarseLevel: bool = True):
    """The preconditioner of a field as ``blockamg`` chooses it by default: a two-grid cycle on the rigid-body modes
    for a 3D displacement field (with node coordinates), one AMG hierarchy otherwise."""
    isDisplacement3D = field.dimension == 3 and phenomena.get(field.name) == "vector"
    if rigidBodyCoarseLevel and isDisplacement3D and context.coordinates(field.name) is not None:
        return TwoGrid(coarseSpace=RigidBodyModes())
    return AMG()


def blockAMG(
    rigidBodyCoarseLevel: bool = True,
    tolerance=None,
    maxIterations: int = 100,
    sweeps: int = 1,
    symmetric: bool = False,
    rebuild: RebuildPolicy = None,
) -> LGMRES:
    """The composed equivalent of :class:`~edelweissfe.linsolve.blockamg.blockamg.BlockAMGSolver` with its defaults:
    block Gauss--Seidel over the fields of the diagonally scaled matrix, inside LGMRES::

        LGMRES(
            DiagonalScaling(
                Multiplicative([OnField(f, M_f) for every field f])   # M_f: TwoGrid(RigidBodyModes()) for a
            )                                                         # 3D displacement field, else AMG()
        )

    The corrections are made at the first set-up, for the fields of the model (:class:`~edelweissfe.linsolve.
    composition.combinators.Multiplicative` takes a function of the context).

    Parameters
    ----------
    rigidBodyCoarseLevel
        Whether a 3D displacement field gets the two-grid cycle on its rigid-body modes (else one AMG hierarchy, on
        the translations).
    tolerance
        The tolerance per solve (default: the Eisenstat--Walker forcing tolerance).
    maxIterations
        The outer iteration cap.
    sweeps, symmetric
        The block Gauss--Seidel sweeps, see :func:`~edelweissfe.linsolve.composition.combinators.gaussSeidelOrder`.
    rebuild
        When the preconditioner is rebuilt.
    """

    def fieldCorrections(context):
        corrections = [
            OnField(field.name, blockAMGFieldPreconditioner(field, context, rigidBodyCoarseLevel))
            for field in context.fields
        ]
        return gaussSeidelOrder(corrections, sweeps, symmetric)

    return LGMRES(
        DiagonalScaling(Multiplicative(fieldCorrections)),
        tolerance=tolerance,
        maxIterations=maxIterations,
        rebuild=rebuild,
    )
