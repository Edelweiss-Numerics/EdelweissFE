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

from edelweissfe.linsolve.composition.amg import AMG, defaultCornerNodeTwoLevelTree
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
from edelweissfe.linsolve.composition.twolevel import (
    CornerNodeInterpolation,
    RigidBodyModesPerAggregate,
    TwoLevel,
)
from edelweissfe.linsolve.nullspace import carriesRigidBodyModes


def blockAMGFieldPreconditioner(
    field,
    context,
    rigidBodyModesPerAggregate: bool = True,
    cornerNodeInterpolation: bool = True,
    dropTolerance: float = 0.0,
    lumpDroppedEntries: bool = False,
    tree: dict = None,
):
    """The preconditioner of a field in the ``blockamg`` preset:

    - a quadratic 3D displacement field (with node coordinates): the two-level cycle on its corner nodes,
      ``TwoLevel(coarseSpace=CornerNodeInterpolation(coarseSolve))``, whose coarse field (linear) gets the next case,
      under a diagonal scaling of its own and with the smoother of the corner-node two-level cycle
      (:func:`~edelweissfe.linsolve.composition.amg.defaultCornerNodeTwoLevelTree`);
    - a linear 3D displacement field (with node coordinates): the two-level cycle on its rigid-body modes,
      ``TwoLevel(coarseSpace=RigidBodyModesPerAggregate())`` (:func:`linearDisplacementFieldPreconditioner`);
    - any other field: one AMG hierarchy, ``AMG()``.

    ``tree`` is the field's AMGCL tree (``None``: the component's default); ``dropTolerance`` and ``lumpDroppedEntries``
    reach every level. A field is quadratic if it has midside nodes
    (:meth:`~edelweissfe.linsolve.composition.base.SetUpContext.midsideCorners`); a field on elements whose topology the
    corner-node interpolation does not know (e.g. Tet10) has none, and gets the rigid-body cycle."""
    isDisplacement3D = field.dimension == 3 and carriesRigidBodyModes(field)
    filtering = dict(dropTolerance=dropTolerance, lumpDroppedEntries=lumpDroppedEntries)
    if not isDisplacement3D or not (rigidBodyModesPerAggregate or cornerNodeInterpolation):
        return AMG(tree=tree, **filtering)
    if context.coordinates(field.name) is None:
        context.message(
            "info",
            "blockAMG: the field '{:}' has no node coordinates (no model, and no nodeCoordinates given): it gets AMG() on "
            "the translations instead of a two-level cycle".format(field.name),
        )
        return AMG(tree=tree, **filtering)
    midsideCorners = context.midsideCorners(field.name)
    isQuadratic = midsideCorners is not None and not midsideCorners[0].all()
    if cornerNodeInterpolation and isQuadratic:
        coarseSolve = DiagonalScaling(
            linearDisplacementFieldPreconditioner(
                rigidBodyModesPerAggregate, twoLevelTree=defaultCornerNodeTwoLevelTree(), **filtering
            )
        )
        return TwoLevel(coarseSpace=CornerNodeInterpolation(coarseSolve=coarseSolve), tree=tree, **filtering)
    return linearDisplacementFieldPreconditioner(
        rigidBodyModesPerAggregate, twoLevelTree=tree, amgTree=tree, **filtering
    )


def linearDisplacementFieldPreconditioner(
    rigidBodyModesPerAggregate: bool,
    twoLevelTree: dict = None,
    amgTree: dict = None,
    dropTolerance: float = 0.0,
    lumpDroppedEntries: bool = False,
):
    """The preconditioner of a linear 3D displacement field in the ``blockamg`` preset, a field of its own or the coarse
    field of the corner-node interpolation: the two-level cycle on its rigid-body modes,
    ``TwoLevel(coarseSpace=RigidBodyModesPerAggregate(), tree=twoLevelTree)``, or with ``rigidBodyModesPerAggregate``
    false one AMG hierarchy on the translations, ``AMG(tree=amgTree)``. Each tree ``None``: the component's default."""
    filtering = dict(dropTolerance=dropTolerance, lumpDroppedEntries=lumpDroppedEntries)
    if rigidBodyModesPerAggregate:
        return TwoLevel(coarseSpace=RigidBodyModesPerAggregate(), tree=twoLevelTree, **filtering)
    return AMG(tree=amgTree, **filtering)


def blockAMG(
    rigidBodyModesPerAggregate: bool = True,
    cornerNodeInterpolation: bool = True,
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
                Multiplicative([OnField(f, M_f) for every field f])   # M_f: blockAMGFieldPreconditioner(f)
            )
        )

    with each field's preconditioner chosen for it by measurement (:func:`blockAMGFieldPreconditioner`): the
    two-level cycle on the corner nodes for a quadratic 3D displacement field, on the rigid-body modes for a linear
    one, one AMG hierarchy for any other field.

    The corrections are made at the first set-up, for the fields of the model (:class:`~edelweissfe.linsolve.
    composition.combinators.Multiplicative` takes a function of the context). A tree, not a solver: a solver script
    binds it, ``linearSolver = blockAMG()``.

    Parameters
    ----------
    rigidBodyModesPerAggregate
        Whether a linear 3D displacement field, and the coarse field of the corner-node interpolation, get the
        two-level cycle on their rigid-body modes (else one AMG hierarchy, on the translations).
    cornerNodeInterpolation
        Whether a quadratic 3D displacement field gets the two-level cycle on its corner nodes (else the case of a
        linear one).
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
                blockAMGFieldPreconditioner(
                    field,
                    context,
                    rigidBodyModesPerAggregate,
                    cornerNodeInterpolation,
                    dropTolerance,
                    lumpDroppedEntries,
                ),
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
