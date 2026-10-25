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
"""A two-grid cycle with a coarse space of its own: :class:`TwoGrid`.

One cycle, for :math:`z \\approx A^{-1} r` with a prolongation :math:`P` from the coarse space:

.. math::

    z &\\leftarrow S^{\\nu_1}(0, r)                       && \\text{pre-smoothing (Chebyshev)} \\\\
    z &\\leftarrow z + P \\, A_c^{-1} P^T (r - A z)       && \\text{coarse correction}, \\quad A_c = P^T A P \\\\
    z &\\leftarrow S^{\\nu_2}(z, r)                       && \\text{post-smoothing}

with :math:`A_c^{-1}` approximated by an AMG hierarchy. The coarse space decides what the cycle can correct:

- :class:`RigidBodyModes`: all rigid-body modes of each aggregate of nodes (3 translations and 3 rotations in
  3D), where an AMG hierarchy on node-wise 3x3 blocks takes the translations alone;
- :class:`P1Interpolation`: the linear (P1) interpolant of a quadratic field from its corner nodes.

The cycle runs in C++ (:mod:`~edelweissfe.linsolve.blockamg.rigidbodycoarselevel`,
:mod:`~edelweissfe.linsolve.blockamg.ptwogrid`). Its steps, written out in Python, are
``Multiplicative([smoother, CoarseCorrection(P, coarse), smoother])``
(:mod:`~edelweissfe.linsolve.composition.combinators`).
"""

import edelweissfe.utils.performancetiming as performancetiming
from edelweissfe.linsolve.composition.amg import amgclTreeOf, defaultFieldTree
from edelweissfe.linsolve.composition.base import (
    Description,
    Preconditioner,
    SetUpContext,
    Step,
)


class RigidBodyModes:
    """The coarse space of all rigid-body modes per aggregate of nodes: ``TwoGrid(coarseSpace=RigidBodyModes())``.

    Parameters
    ----------
    prolongation
        ``"smoothed"`` (:math:`P = (I - \\omega D_F^{-1} A_F) P_{\\mathrm{tent}}`) or ``"tentative"``
        (:math:`P_{\\mathrm{tent}}`, the modes orthonormalized per aggregate).
    coarseHierarchyPrecision
        ``"double"`` or ``"float"``: the precision of the AMG hierarchy below the coarse level.
    fineSpectralRadiusReuse
        Whether a rebuild re-checks the previous build's fine spectral radius instead of estimating it in full.
    """

    name = "rigid-body modes"

    def __init__(
        self, prolongation: str = "smoothed", coarseHierarchyPrecision: str = "double", fineSpectralRadiusReuse=False
    ):
        self.prolongation = prolongation
        self.coarseHierarchyPrecision = coarseHierarchyPrecision
        self.fineSpectralRadiusReuse = fineSpectralRadiusReuse

    def build(self, field, tree: dict, A, context: SetUpContext, fineSpectralRadius, fineDominantVector):
        """The two-grid cycle on this coarse space for a field's matrix ``A``."""
        from edelweissfe.linsolve.blockamg.rigidbodycoarselevel import (
            RigidBodyCoarseLevelPreconditioner,
        )

        coords = context.coordinates(field.name)
        if coords is None:
            raise ValueError(
                "TwoGrid: the rigid-body modes of the field '{:}' need its node coordinates".format(field.name)
            )
        cycle = RigidBodyCoarseLevelPreconditioner(
            amgclTreeOf(tree),
            tree.get("backendBlockSize", 1),
            tree.get("backendPrecision", "double"),
            prolongation=self.prolongation,
            coarseHierarchyPrecision=self.coarseHierarchyPrecision,
            fineSpectralRadius=fineSpectralRadius,
            fineDominantVector=fineDominantVector,
        )
        with performancetiming.timeit("{:} rigid-body coarse level build".format(field.name)):
            cycle.build(A.A, context.scalingOf(slice(None)), coords)
        return cycle

    def sizesOf(self, cycle) -> dict:
        """What a built cycle reports about its coarse level."""
        return {"aggregates": cycle.nAggregates, "coarse dofs": cycle.nCoarse, "dropped modes": cycle.droppedModes}

    def describe(self) -> dict:
        return {"coarse space": self.name, "prolongation": self.prolongation}


class P1Interpolation:
    """The coarse space of the linear (P1) interpolant of a quadratic field from its corner nodes:
    ``TwoGrid(coarseSpace=P1Interpolation())``. Needs the field's corner/midside topology.

    Parameters
    ----------
    rigidBodyNullspace
        Whether the AMG hierarchy of the coarse (P1) operator takes the rotations too (needs node coordinates).
    fineSpectralRadiusReuse
        Whether a rebuild re-checks the previous build's fine spectral radius instead of estimating it in full.
    options
        :class:`~edelweissfe.linsolve.blockamg.ptwogrid.PTwoGridPreconditioner` keywords (``nu``, ``fineDegree``,
        ...).
    """

    name = "P1 interpolation"

    def __init__(self, rigidBodyNullspace: bool = True, fineSpectralRadiusReuse: bool = False, **options):
        self.rigidBodyNullspace = rigidBodyNullspace
        self.fineSpectralRadiusReuse = fineSpectralRadiusReuse
        self.options = options

    def build(self, field, tree: dict, A, context: SetUpContext, fineSpectralRadius, fineDominantVector):
        """The two-grid cycle on this coarse space for a field's matrix ``A``."""
        from edelweissfe.linsolve.blockamg.ptwogrid import PTwoGridPreconditioner

        p1Map = context.p1Map(field.name)
        if p1Map is None:
            raise ValueError("TwoGrid: the P1 interpolation of the field '{:}' needs its P1 map".format(field.name))
        cycle = PTwoGridPreconditioner(
            *p1Map, fineSpectralRadius=fineSpectralRadius, fineDominantVector=fineDominantVector, **self.options
        )
        coords = context.coordinates(field.name) if self.rigidBodyNullspace else None
        with performancetiming.timeit("{:} p-two-grid build".format(field.name)):
            cycle.build(A.A, context.scalingOf(slice(None)), coords=coords)
        return cycle

    def sizesOf(self, cycle) -> dict:
        """What a built cycle reports about its coarse level."""
        return {}

    def describe(self) -> dict:
        return {"coarse space": self.name, **self.options}


class TwoGrid(Preconditioner):
    """A two-grid cycle on a field's matrix with the coarse space ``coarseSpace``; see the module documentation.

    Reused while only the values of the matrix change, like :class:`~edelweissfe.linsolve.composition.amg.AMG`.

    Parameters
    ----------
    coarseSpace
        :class:`RigidBodyModes` or :class:`P1Interpolation`.
    tree
        The field tree: the fine smoother and the AMG hierarchy below the coarse level read it as
        :class:`~edelweissfe.linsolve.composition.amg.AMG` does. ``None``: the default tree of the field's dimension.
    """

    def __init__(self, coarseSpace, tree: dict = None):
        self.coarseSpace = coarseSpace
        self.tree = tree
        self.cycle = None
        self._field = None
        self._fineSpectralRadius = (None, None)

    def setModel(self, model, dofManager) -> None:
        # the fine spectral radius of another mesh is no estimate for this one
        self._fineSpectralRadius = (None, None)
        super().setModel(model, dofManager)

    def setUp(self, A, context: SetUpContext) -> None:
        (field,) = context.fields
        self._field = field
        tree = self.tree if self.tree is not None else defaultFieldTree(field.dimension)
        fineSpectralRadius, fineDominantVector = (
            self._fineSpectralRadius if self.coarseSpace.fineSpectralRadiusReuse else (None, None)
        )
        self.cycle = self.coarseSpace.build(field, tree, A, context, fineSpectralRadius, fineDominantVector)
        if self.coarseSpace.fineSpectralRadiusReuse:
            self._fineSpectralRadius = (self.cycle.fineSpectralRadius, self.cycle.fineDominantVector)
        self.statistics.sizes.update(rows=A.shape[0], **self.coarseSpace.sizesOf(self.cycle))

    def update(self, A, change) -> None:
        if change.requiresRebuild or self.cycle is None:
            self.setUp(A, change.context)

    def __call__(self, r):
        return self.cycle.applyPreconditioner(r)

    def describe(self) -> Description:
        name = self._field.name if self._field is not None else "A"
        return Description(
            name="TwoGrid",
            formula="TwoGrid(" + name + "; " + self.coarseSpace.name + ")⁻¹",
            latex="\\mathrm{TG}(A_{\\text{" + name + "}})^{-1}",
            parameters=self.coarseSpace.describe(),
            steps=(
                Step("z = S(0, r)   (pre-smoothing)"),
                Step("z += P A_c⁻¹ Pᵀ (r − A z),  A_c = Pᵀ A P"),
                Step("z = S(z, r)   (post-smoothing)"),
            ),
        )
