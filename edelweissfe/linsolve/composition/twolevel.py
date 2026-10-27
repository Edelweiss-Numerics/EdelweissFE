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
    backendOf,
    checkBackend,
    fieldTreeOf,
    preconditionerMatrix,
)
from edelweissfe.linsolve.composition.base import (
    Description,
    Preconditioner,
    SetUpContext,
    Step,
)
from edelweissfe.linsolve.nullspace import carriesRigidBodyModes


class CoarseSpace:
    """What spans the coarse level of a :class:`TwoLevel` cycle: it checks that it can be built for a field, and builds
    the cycle on it. The cycle's own parameters (the fine smoother of the field's tree, the backend, the precision of
    the coarse hierarchy) come from :class:`TwoLevel`, as ``cycle``, the keywords of
    :meth:`~edelweissfe.linsolve.composition.twolevelcycle.TwoLevelPreconditioner.cycleParameters`.
    """

    #: The name of the coarse space in the views.
    name = None

    def check(self, field, context: SetUpContext, cycle: dict) -> None:
        """Raise, naming the field, if the cycle cannot be built on this coarse space for the field: before anything is
        built."""
        raise NotImplementedError

    def build(self, field, matrix, context: SetUpContext, cycle: dict, fineSpectralRadius, fineDominantVector):
        """The two-level cycle on this coarse space for a field's matrix (SciPy CSR), built: a
        :class:`~edelweissfe.linsolve.composition.twolevelcycle.TwoLevelPreconditioner`.

        Parameters
        ----------
        field
            The field.
        matrix
            Its (scaled) matrix.
        context
            Its context: the node coordinates, the P1 map, the scaling.
        cycle
            The cycle's parameters, see
            :meth:`~edelweissfe.linsolve.composition.twolevelcycle.TwoLevelPreconditioner.cycleParameters`.
        fineSpectralRadius, fineDominantVector
            A previous build's fine spectral radius to start from, or ``None``.
        """
        raise NotImplementedError

    def sizesOf(self, cycle) -> dict:
        """What a built cycle reports about its coarse level (for the run statistics)."""
        return {}

    def describe(self) -> dict:
        """The settings of the coarse space, for the views."""
        return {"coarse space": self.name, "coarse solve": "AMG V-cycle"}


def _rejected(field, reason: "str | None") -> None:
    """Raise if the C++ cycle rejected the parameters: it names itself in the message; here the field does."""
    if reason is not None:
        raise ValueError("TwoLevel on the field '{:}': {:}".format(field.name, reason))


class RigidBodyModes(CoarseSpace):
    """The coarse space of all rigid-body modes per aggregate of nodes: ``TwoLevel(coarseSpace=RigidBodyModes())``.

    Parameters
    ----------
    prolongation
        ``"smoothed"`` (:math:`P = (I - \\omega D_F^{-1} A_F) P_{\\mathrm{tent}}`) or ``"tentative"``
        (:math:`P_{\\mathrm{tent}}`, the modes orthonormalized per aggregate).
    """

    name = "rigid-body modes"

    def __init__(self, prolongation: str = "smoothed"):
        if prolongation not in ("smoothed", "tentative"):
            raise ValueError(
                "RigidBodyModes: prolongation must be 'smoothed' or 'tentative', got {!r}".format(prolongation)
            )
        self.prolongation = prolongation

    def check(self, field, context: SetUpContext, cycle: dict) -> None:
        """Rigid-body modes are those of a 2D or 3D displacement field with node coordinates
        (:func:`~edelweissfe.linsolve.nullspace.carriesRigidBodyModes`), on a block backend (the scalar backend takes
        them as the near null-space of its hierarchy itself, ``AMG()``)."""
        if not carriesRigidBodyModes(field):
            raise ValueError(
                "TwoLevel: the rigid-body modes are those of a 2D or 3D displacement field, but the field '{:}' is "
                "another field (or has {:} component(s) per node): its near null-space is its translations, which "
                "AMG() takes".format(field.name, field.dimension)
            )
        if context.coordinates(field.name) is None:
            raise ValueError(
                "TwoLevel: the rigid-body modes of the field '{:}' need its node coordinates".format(field.name)
            )
        if cycle["backendBlockSize"] == 1:
            raise ValueError(
                "TwoLevel: the field '{:}' is on the scalar backend (backendBlockSize 1), which takes the rigid-body "
                "modes as the near null-space of AMG() itself; the rigid-body coarse level is for the block backends, "
                "backendBlockSize 2 or 3".format(field.name)
            )
        from edelweissfe.linsolve.composition.rigidbodycoarselevel import (
            RigidBodyCoarseLevelPreconditioner,
        )

        _rejected(field, RigidBodyCoarseLevelPreconditioner.parameterRejection(prolongation=self.prolongation, **cycle))

    def build(self, field, matrix, context: SetUpContext, cycle: dict, fineSpectralRadius, fineDominantVector):
        from edelweissfe.linsolve.composition.rigidbodycoarselevel import (
            RigidBodyCoarseLevelPreconditioner,
        )

        built = RigidBodyCoarseLevelPreconditioner(
            prolongation=self.prolongation,
            fineSpectralRadius=fineSpectralRadius,
            fineDominantVector=fineDominantVector,
            **cycle,
        )
        with performancetiming.timeit("{:} rigid-body coarse level build".format(field.name)):
            built.build(matrix, context.scalingOf(slice(None)), context.coordinates(field.name))
        return built

    def sizesOf(self, cycle) -> dict:
        return {"aggregates": cycle.nAggregates, "coarse dofs": cycle.nCoarse, "dropped modes": cycle.droppedModes}

    def describe(self) -> dict:
        return {"coarse space": self.name, "prolongation": self.prolongation, "coarse solve": "AMG V-cycle"}


class P1Interpolation(CoarseSpace):
    """The coarse space of the linear (P1) interpolant of a quadratic field from its corner nodes:
    ``TwoLevel(coarseSpace=P1Interpolation())``. Needs the field's corner/midside topology.

    Parameters
    ----------
    coarsePrecond
        The AMGCL tree of the coarse (P1) operator's hierarchy; ``None``: the default of
        :class:`~edelweissfe.linsolve.composition.ptwogrid.PTwoGridPreconditioner`.
    useCoarseNullspace
        Whether that hierarchy takes a near null-space of the corner nodes: their rigid-body modes for a displacement
        field with node coordinates (:func:`~edelweissfe.linsolve.nullspace.carriesRigidBodyModes`), else their
        translations; else AMGCL's default.
    """

    name = "P1 interpolation"

    def __init__(self, coarsePrecond: dict = None, useCoarseNullspace: bool = True):
        self.coarsePrecond = coarsePrecond
        self.useCoarseNullspace = useCoarseNullspace

    def check(self, field, context: SetUpContext, cycle: dict) -> None:
        """It needs the field's P1 map, and parameters the C++ cycle can use."""
        if context.p1Map(field.name) is None:
            raise ValueError("TwoLevel: the P1 interpolation of the field '{:}' needs its P1 map".format(field.name))
        from edelweissfe.linsolve.composition.ptwogrid import PTwoGridPreconditioner

        _rejected(
            field,
            PTwoGridPreconditioner.parameterRejection(
                coarsePrecond=self.coarsePrecond, useCoarseNullspace=self.useCoarseNullspace, **cycle
            ),
        )

    def build(self, field, matrix, context: SetUpContext, cycle: dict, fineSpectralRadius, fineDominantVector):
        from edelweissfe.linsolve.composition.ptwogrid import PTwoGridPreconditioner

        built = PTwoGridPreconditioner(
            *context.p1Map(field.name),
            coarsePrecond=self.coarsePrecond,
            useCoarseNullspace=self.useCoarseNullspace,
            fineSpectralRadius=fineSpectralRadius,
            fineDominantVector=fineDominantVector,
            **cycle,
        )
        coordinates = context.coordinates(field.name) if carriesRigidBodyModes(field) else None
        with performancetiming.timeit("{:} p-two-grid build".format(field.name)):
            built.build(matrix, context.scalingOf(slice(None)), coordinates)
        return built

    def describe(self) -> dict:
        return {
            "coarse space": self.name,
            "coarse solve": "AMG V-cycle",
            "coarsePrecond": "default" if self.coarsePrecond is None else "given",
        }


class TwoLevel(Preconditioner):
    """A two-level cycle on a field's matrix with the coarse space ``coarseSpace``; see the module documentation.

    Reused while only the values of the matrix change, like :class:`~edelweissfe.linsolve.composition.amg.AMG`.

    Parameters
    ----------
    coarseSpace
        A :class:`CoarseSpace`: :class:`RigidBodyModes` or :class:`P1Interpolation`.
    tree
        The field's AMGCL tree: its ``relax`` (a Chebyshev polynomial), ``npre`` and ``npost`` are the fine smoother;
        with :class:`RigidBodyModes` its coarsening also builds the coarse level, and the whole tree the AMG hierarchy
        below. ``None``: the default tree of the field's dimension
        (:func:`~edelweissfe.linsolve.composition.amg.defaultFieldTree`).
    backendBlockSize, backendPrecision
        The storage of the fine operator and of :math:`P`, :math:`P^T`, as for
        :class:`~edelweissfe.linsolve.composition.amg.AMG`; ``None``: the field's default
        (:func:`~edelweissfe.linsolve.composition.amg.backendOf`).
    coarseHierarchyPrecision
        The value type of the AMG hierarchy of the coarse level: ``"double"`` or ``"float"``.
    fineSpectralRadiusReuse
        Whether a rebuild starts from the previous build's fine spectral radius, re-checked by a few power iterations,
        instead of estimating it in full (see
        :class:`~edelweissfe.linsolve.composition.twolevelcycle.TwoLevelPreconditioner`).
    dropTolerance, lumpDroppedEntries
        Build the cycle (its smoother, coarse level and hierarchy) on the filtered matrix, see
        :func:`~edelweissfe.linsolve.composition.amg.droppedSmallEntries`; ``0`` (the default) builds it on the
        matrix itself.
    """

    def __init__(
        self,
        coarseSpace: CoarseSpace,
        tree: dict = None,
        backendBlockSize: int = None,
        backendPrecision: str = None,
        coarseHierarchyPrecision: str = "double",
        fineSpectralRadiusReuse: bool = False,
        dropTolerance: float = 0.0,
        lumpDroppedEntries: bool = False,
    ):
        if not isinstance(coarseSpace, CoarseSpace):
            raise TypeError(
                "TwoLevel: coarseSpace must be a CoarseSpace such as RigidBodyModes() or P1Interpolation(), got "
                "{!r}".format(coarseSpace)
            )
        if not dropTolerance >= 0.0:
            raise ValueError("TwoLevel: dropTolerance must not be negative, got {!r}".format(dropTolerance))
        checkBackend("TwoLevel", backendBlockSize, backendPrecision)
        if coarseHierarchyPrecision not in ("double", "float"):
            raise ValueError(
                "TwoLevel: coarseHierarchyPrecision must be 'double' or 'float', got {!r}".format(
                    coarseHierarchyPrecision
                )
            )
        super().__init__()
        self.coarseSpace = coarseSpace
        self.tree = tree
        self.backendBlockSize = backendBlockSize
        self.backendPrecision = backendPrecision
        self.coarseHierarchyPrecision = coarseHierarchyPrecision
        self.fineSpectralRadiusReuse = fineSpectralRadiusReuse
        self.dropTolerance = dropTolerance
        self.lumpDroppedEntries = lumpDroppedEntries
        # the fine spectral radius and dominant vector of the last build, which the next one may start from
        # (fineSpectralRadiusReuse): kept on purpose when the cycle is forgotten
        self._fineSpectralRadius = (None, None)

    def _forget(self) -> None:
        self.cycle = None
        self._field = None

    def setModel(self, model, dofManager) -> None:
        # the fine spectral radius of another mesh is no estimate for this one
        self._fineSpectralRadius = (None, None)
        super().setModel(model, dofManager)

    def _cycleParameters(self, field) -> dict:
        """The cycle's parameters for the field: the tree and the backend, the field's defaults for what is not given
        (see :meth:`~edelweissfe.linsolve.composition.twolevelcycle.TwoLevelPreconditioner.cycleParameters`)."""
        tree = fieldTreeOf("TwoLevel", self.tree, field)
        # the block translations belong to AMG's hierarchy on the block backend; a coarse space is its own
        backendBlockSize, backendPrecision, _ = backendOf(
            "TwoLevel", field, tree, self.backendBlockSize, self.backendPrecision, blockTranslationNullspace=False
        )
        return {
            "amgclTree": tree,
            "backendBlockSize": backendBlockSize,
            "backendPrecision": backendPrecision,
            "coarseHierarchyPrecision": self.coarseHierarchyPrecision,
        }

    def _setUp(self, A, context: SetUpContext) -> None:
        field = context.onlyField("TwoLevel")
        cycle = self._cycleParameters(field)
        self.coarseSpace.check(field, context, cycle)
        self._field = field
        fineSpectralRadius, fineDominantVector = (
            self._fineSpectralRadius if self.fineSpectralRadiusReuse else (None, None)
        )
        if fineDominantVector is not None and len(fineDominantVector) != A.shape[0]:
            # built for other dofs (e.g. other condensed dofs of a Schur complement): no estimate for these
            fineSpectralRadius, fineDominantVector = None, None
        matrix = preconditionerMatrix(A, self.dropTolerance, self.lumpDroppedEntries, self.statistics)
        self.cycle = self.coarseSpace.build(field, matrix, context, cycle, fineSpectralRadius, fineDominantVector)
        if self.fineSpectralRadiusReuse:
            self._fineSpectralRadius = (self.cycle.fineSpectralRadius, self.cycle.fineDominantVector)
        self.statistics.sizes.update(rows=A.shape[0], **self.coarseSpace.sizesOf(self.cycle))

    def _update(self, A, change) -> None:
        if change.requiresRebuild or self.cycle is None:
            self.setUp(A, change.context)

    def _apply(self, r):
        return self.cycle.applyPreconditioner(r)

    def _parameters(self) -> dict:
        parameters = self.coarseSpace.describe()
        backend = {"backendBlockSize": self.backendBlockSize, "backendPrecision": self.backendPrecision}
        parameters.update({key: value for key, value in backend.items() if value is not None})
        if self.coarseHierarchyPrecision != "double":
            parameters["coarseHierarchyPrecision"] = self.coarseHierarchyPrecision
        if self.fineSpectralRadiusReuse:
            parameters["fineSpectralRadiusReuse"] = True
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
