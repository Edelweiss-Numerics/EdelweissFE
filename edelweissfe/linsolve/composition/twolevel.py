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
"""A two-level cycle with a coarse space of its own: :class:`TwoLevel`.

One cycle, for :math:`z \\approx A^{-1} r` with a prolongation :math:`P` from the coarse space:

.. math::

    z &\\leftarrow S^{\\nu_1}(0, r)                       && \\text{pre-smoothing (Chebyshev)} \\\\
    z &\\leftarrow z + P \\, A_c^{-1} P^T (r - A z)       && \\text{coarse correction}, \\quad A_c = P^T A P \\\\
    z &\\leftarrow S^{\\nu_2}(z, r)                       && \\text{post-smoothing}

with :math:`A_c^{-1}` approximated by an AMG hierarchy. The coarse space decides what the cycle can correct:

- :class:`RigidBodyModes`: all rigid-body modes of each aggregate of nodes (3 translations and 3 rotations in
  3D), where an AMG hierarchy on node-wise 3x3 blocks takes the translations alone;
- :class:`P1Interpolation`: the linear (P1) interpolant of a quadratic field from its corner nodes.

Why "two-level": the component is the fine level and one coarse correction, and its coarse problem is solved inexactly,
by one V-cycle of an AMG hierarchy. The coarse space need not be a coarser mesh. With :class:`RigidBodyModes` it is
spanned by aggregates of nodes, and the whole cycle is smoothed aggregation with all rigid-body modes on every level,
the method Trilinos ML and MueLu use for elasticity (Vaněk, Mandel and Brezina, Computing 56, 1996). The first coarse
level is built here, on the node-wise blocks of the fine level, with the modes scaled by the diagonal scaling and zero on
the Dirichlet rows. The levels below are built by AMGCL, which takes the coefficients of the modes in the coarse space
as its near null-space. ``prolongation="tentative"`` makes the first level plain aggregation. With
:class:`P1Interpolation` the coarse level is a coarser grid, the corner nodes (p-multigrid), and its AMG hierarchy takes
the rigid-body modes of the corner nodes as its near null-space.

The cycle runs in C++ (:mod:`~edelweissfe.linsolve.composition.rigidbodycoarselevel`,
:mod:`~edelweissfe.linsolve.composition.ptwogrid`). Its steps, written out in Python, are
``Multiplicative([smoother, CoarseCorrection(P, coarse), smoother])``
(:mod:`~edelweissfe.linsolve.composition.combinators`). Both ways are kept on purpose: this one is fast (about half the
cost per application), the spelled-out one shows the steps and takes any smoother or coarse space. Why, and what would
end the exception: the "two-level cycle: two ways" section of ``doc/source/documentation/composedlinsolvers.rst``.
"""

import edelweissfe.utils.performancetiming as performancetiming
from edelweissfe.linsolve.composition.amg import (
    amgclTreeOf,
    checkedFieldTree,
    preconditionerMatrix,
)
from edelweissfe.linsolve.composition.base import (
    Description,
    Preconditioner,
    SetUpContext,
    Step,
)


class RigidBodyModes:
    """The coarse space of all rigid-body modes per aggregate of nodes: ``TwoLevel(coarseSpace=RigidBodyModes())``.

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
        if prolongation not in ("smoothed", "tentative"):
            raise ValueError(
                "RigidBodyModes: prolongation must be 'smoothed' or 'tentative', got {!r}".format(prolongation)
            )
        if coarseHierarchyPrecision not in ("double", "float"):
            raise ValueError(
                "RigidBodyModes: coarseHierarchyPrecision must be 'double' or 'float', got {!r}".format(
                    coarseHierarchyPrecision
                )
            )
        self.prolongation = prolongation
        self.coarseHierarchyPrecision = coarseHierarchyPrecision
        self.fineSpectralRadiusReuse = fineSpectralRadiusReuse

    def check(self, field, tree: dict, context: SetUpContext) -> None:
        """Raise if this coarse space cannot be built for the field with its tree: rigid-body modes are those of a 2D or
        3D displacement-like field with node coordinates, on a block backend (the scalar backend takes them as the near
        null-space of its hierarchy itself, ``AMG()``), with a tree the coarse level can read."""
        if field.dimension not in (2, 3):
            raise ValueError(
                "TwoLevel: the rigid-body modes need a 2D or 3D vector field, but the field '{:}' has {:} component(s) "
                "per node".format(field.name, field.dimension)
            )
        if context.coordinates(field.name) is None:
            raise ValueError(
                "TwoLevel: the rigid-body modes of the field '{:}' need its node coordinates".format(field.name)
            )
        from edelweissfe.linsolve.composition.rigidbodycoarselevel import (
            parameterRejection,
        )

        blockSize = tree.get("backendBlockSize", 1)
        if blockSize not in (2, 3):
            raise ValueError(
                "TwoLevel: the field '{:}' is on the scalar backend (backendBlockSize {:}), which takes the rigid-body "
                "modes as the near null-space of AMG() itself; the rigid-body coarse level is for the block backends, "
                "backendBlockSize 2 or 3".format(field.name, blockSize)
            )
        reason = parameterRejection(
            amgclTreeOf(tree),
            blockSize,
            tree.get("backendPrecision", "double"),
            self.prolongation,
            self.coarseHierarchyPrecision,
        )
        if reason is not None:
            raise ValueError("TwoLevel on the field '{:}': {:}".format(field.name, reason))

    def build(self, field, tree: dict, matrix, context: SetUpContext, fineSpectralRadius, fineDominantVector):
        """The two-level cycle on this coarse space for a field's matrix (SciPy CSR)."""
        from edelweissfe.linsolve.composition.rigidbodycoarselevel import (
            RigidBodyCoarseLevelPreconditioner,
        )

        coords = context.coordinates(field.name)
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
            cycle.build(matrix, context.scalingOf(slice(None)), coords)
        return cycle

    def sizesOf(self, cycle) -> dict:
        """What a built cycle reports about its coarse level."""
        return {"aggregates": cycle.nAggregates, "coarse dofs": cycle.nCoarse, "dropped modes": cycle.droppedModes}

    def describe(self) -> dict:
        return {"coarse space": self.name, "prolongation": self.prolongation, "coarse solve": "AMG V-cycle"}


class P1Interpolation:
    """The coarse space of the linear (P1) interpolant of a quadratic field from its corner nodes:
    ``TwoLevel(coarseSpace=P1Interpolation())``. Needs the field's corner/midside topology.

    Parameters
    ----------
    rigidBodyNullspace
        Whether the AMG hierarchy of the coarse (P1) operator takes the rotations too (needs node coordinates).
    fineSpectralRadiusReuse
        Whether a rebuild re-checks the previous build's fine spectral radius instead of estimating it in full.
    options
        :class:`~edelweissfe.linsolve.composition.ptwogrid.PTwoGridPreconditioner` keywords (``nu``, ``fineDegree``,
        ...).
    """

    name = "P1 interpolation"

    def __init__(self, rigidBodyNullspace: bool = True, fineSpectralRadiusReuse: bool = False, **options):
        from edelweissfe.linsolve.composition.ptwogrid import parameterRejection

        reason = parameterRejection(**options)
        if reason is not None:
            raise ValueError("P1Interpolation: " + reason)
        self.rigidBodyNullspace = rigidBodyNullspace
        self.fineSpectralRadiusReuse = fineSpectralRadiusReuse
        self.options = options

    def check(self, field, tree: dict, context: SetUpContext) -> None:
        """Raise if this coarse space cannot be built for the field: it needs the field's P1 map, and node-wise 3x3
        fine blocks need a 3D field."""
        if self.options.get("fineBlockSize", 1) == 3 and field.dimension != 3:
            raise ValueError(
                "TwoLevel: fineBlockSize 3 needs a 3D field, but the field '{:}' has {:} component(s) per node".format(
                    field.name, field.dimension
                )
            )
        if context.p1Map(field.name) is None:
            raise ValueError("TwoLevel: the P1 interpolation of the field '{:}' needs its P1 map".format(field.name))

    def build(self, field, tree: dict, matrix, context: SetUpContext, fineSpectralRadius, fineDominantVector):
        """The two-level cycle on this coarse space for a field's matrix (SciPy CSR)."""
        from edelweissfe.linsolve.composition.ptwogrid import PTwoGridPreconditioner

        p1Map = context.p1Map(field.name)
        cycle = PTwoGridPreconditioner(
            *p1Map, fineSpectralRadius=fineSpectralRadius, fineDominantVector=fineDominantVector, **self.options
        )
        coords = context.coordinates(field.name) if self.rigidBodyNullspace else None
        with performancetiming.timeit("{:} p-two-grid build".format(field.name)):
            cycle.build(matrix, context.scalingOf(slice(None)), coords=coords)
        return cycle

    def sizesOf(self, cycle) -> dict:
        """What a built cycle reports about its coarse level."""
        return {}

    def describe(self) -> dict:
        return {"coarse space": self.name, "coarse solve": "AMG V-cycle", **self.options}


class TwoLevel(Preconditioner):
    """A two-level cycle on a field's matrix with the coarse space ``coarseSpace``; see the module documentation.

    Reused while only the values of the matrix change, like :class:`~edelweissfe.linsolve.composition.amg.AMG`.

    Parameters
    ----------
    coarseSpace
        :class:`RigidBodyModes` or :class:`P1Interpolation`.
    tree
        The field tree: the fine smoother and the AMG hierarchy below the coarse level read it as
        :class:`~edelweissfe.linsolve.composition.amg.AMG` does. ``None``: the default tree of the field's dimension.
    dropTolerance, lumpDroppedEntries
        Build the cycle (its smoother, coarse level and hierarchy) on the filtered matrix, see
        :func:`~edelweissfe.linsolve.composition.amg.droppedSmallEntries`; ``0`` (the default) builds it on the
        matrix itself.
    """

    def __init__(self, coarseSpace, tree: dict = None, dropTolerance: float = 0.0, lumpDroppedEntries: bool = False):
        if not dropTolerance >= 0.0:
            raise ValueError("TwoLevel: dropTolerance must not be negative, got {!r}".format(dropTolerance))
        self.dropTolerance = dropTolerance
        self.lumpDroppedEntries = lumpDroppedEntries
        if not isinstance(coarseSpace, (RigidBodyModes, P1Interpolation)):
            raise TypeError(
                "TwoLevel: coarseSpace must be RigidBodyModes() or P1Interpolation(), got {!r}".format(coarseSpace)
            )
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
        field = context.onlyField("TwoLevel")
        tree = checkedFieldTree("TwoLevel", self.tree, field)
        self.coarseSpace.check(field, tree, context)
        self._field = field
        fineSpectralRadius, fineDominantVector = (
            self._fineSpectralRadius if self.coarseSpace.fineSpectralRadiusReuse else (None, None)
        )
        if fineDominantVector is not None and len(fineDominantVector) != A.shape[0]:
            # built for other dofs (e.g. other condensed dofs of a Schur complement): no estimate for these
            fineSpectralRadius, fineDominantVector = None, None
        matrix = preconditionerMatrix(A, self.dropTolerance, self.lumpDroppedEntries, self.statistics)
        self.cycle = self.coarseSpace.build(field, tree, matrix, context, fineSpectralRadius, fineDominantVector)
        if self.coarseSpace.fineSpectralRadiusReuse:
            self._fineSpectralRadius = (self.cycle.fineSpectralRadius, self.cycle.fineDominantVector)
        self.statistics.sizes.update(rows=A.shape[0], **self.coarseSpace.sizesOf(self.cycle))

    def update(self, A, change) -> None:
        if change.requiresRebuild or self.cycle is None:
            self.setUp(A, change.context)

    def __call__(self, r):
        return self.cycle.applyPreconditioner(r)

    def _parameters(self) -> dict:
        parameters = self.coarseSpace.describe()
        if self.dropTolerance > 0.0:
            parameters.update(dropTolerance=self.dropTolerance, lumpDroppedEntries=self.lumpDroppedEntries)
        return parameters

    def describe(self) -> Description:
        name = self._field.name if self._field is not None else "A"
        return Description(
            name="TwoLevel",
            formula="TwoLevel(" + name + "; " + self.coarseSpace.name + ")⁻¹",
            latex="\\mathrm{TL}(A_{\\text{" + name + "}})^{-1}",
            parameters=self._parameters(),
            steps=(
                Step("z = S(0, r)   (pre-smoothing)"),
                Step("z += P A_c⁻¹ Pᵀ (r − A z),  A_c = Pᵀ A P,  A_c⁻¹ ≈ one AMG V-cycle"),
                Step("z = S(z, r)   (post-smoothing)"),
            ),
        )
