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
"""The rigid-body coarse level of a vector field: the cycle of
``TwoLevel(coarseSpace=RigidBodyModes())`` (:class:`~edelweissfe.linsolve.composition.twolevel.TwoLevel`).

Why
---

The near null-space of linear elasticity -- the displacements that cost (almost) no strain energy -- are the rigid-body
modes: three translations and three rotations in 3D, two translations and one rotation in 2D. Smoothed aggregation
needs them in its coarse space. An error the Chebyshev smoother leaves alone -- its eigenvalues lie below the
smoother's window, which starts at a fraction of the spectral radius by construction -- is removed by the coarse-grid
correction only as far as the coarse space can represent it on every aggregate.

The block backends of :class:`~edelweissfe.linsolve.composition.amg.AMG`, the default for 2D and 3D vector fields, take the translations alone (AMGCL's
hierarchy on node-wise blocks cannot take more): their coarse space holds displacements that are constant on every
aggregate. In a bending-dominated structure -- a slender beam, a tunnel lining, a plate -- the smooth error is bending,
which rotates every aggregate almost as a rigid body. The translations represent it poorly, the smoother does not touch
it, and the outer Krylov solve pays with iterations.

What
----

The rigid-body coarse level keeps the fine level of the block backend -- the field's equilibrated block
:math:`\\hat A = D A D` in node-wise blocks, smoothed by the Chebyshev polynomial of the field's tree -- and builds the
first coarse level itself, from all rigid-body modes:

#. **aggregates**: AMGCL's plain aggregation of the nodes, as AMGCL's own first level on the block backend;
#. **tentative prolongation**: the field's rigid-body modes about its centroid, divided by the equilibration (the near
   null-space of :math:`\\hat A` is :math:`D^{-1}` times that of :math:`A`) -- built in Python by
   :func:`~edelweissfe.linsolve.nullspace.rigidBodyNullspace`, the one definition of the modes, and handed to C++ --
   restricted to every aggregate, zero on the Dirichlet rows (which the cycle solves exactly), and made orthonormal by
   a QR factorization, as AMGCL's tentative prolongation does with a near null-space. A mode that depends linearly on
   the ones before it -- a rotation of a single-node aggregate, the rotation about the line through a collinear
   aggregate, a mode whose components are all constrained -- is dropped;
#. **smoothed prolongation**: :math:`P = (I - \\omega\\, \\mathrm{diag}_B(\\hat A_F)^{-1} \\hat A_F) P_{tent}`,
   AMGCL's, with :math:`\\mathrm{diag}_B` the node-wise :math:`B \\times B` diagonal blocks of the filtered operator;
#. **Galerkin coarse operator**: :math:`A_c = P^T \\hat A P`, 6 coarse dofs per aggregate in 3D (3 in 2D), on which an
   AMGCL hierarchy continues the field's tree, with the R factors of the QR factorizations -- the coefficients of the
   rigid-body modes in the coarse basis -- as its near null-space.

One application is the two-level cycle the p-two-grid (:mod:`~edelweissfe.linsolve.composition.ptwogrid`) uses too:
restriction of the residual, one V-cycle of the AMGCL hierarchy, prolongation, and the fine Chebyshev sweeps. Everything
runs in C++: the steps in ``amgcl-rigid-body-coarse-level.hpp``, the cycle in ``amgcl-wrapper.hpp``
(``RigidBodyCoarseLevelPreconditionerT``).

When
----

It is the default of the ``blockAMG()`` preset for the 3D displacement field (of any element type), and pays most where the solves would need many outer iterations although the operator is (nearly) elastic: typically a
bending-dominated structure. The rigid-body modes are the near null-space of elasticity on the displacement field; a
rotation field, or any other vector field, has another, so a solver script gives it the coarse level only by name
(``OnField("displacement", TwoLevel(coarseSpace=RigidBodyModes()))``, e.g. for a 2D displacement field). It needs the
field's node coordinates (read from the model), and the field's tree configures its smoother, its aggregation and the
AMGCL hierarchy below it. Its setup costs more than AMGCL's hierarchy on the translations (the
Galerkin product with 6 coarse dofs per aggregate, the larger hierarchy below), so on a model where that converges in a
few iterations it gains little. The measurements are in :ref:`measured-rigid-body-coarse-level`.
"""

import numpy as np

from edelweissfe.linsolve.base import FieldBlock
from edelweissfe.linsolve.composition.twolevelcycle import TwoLevelPreconditioner
from edelweissfe.linsolve.nullspace import rigidBodyNullspace


class RigidBodyCoarseLevelPreconditioner(TwoLevelPreconditioner):
    """The preconditioner of one vector field with a rigid-body coarse level (see the module docstring): a two-level
    preconditioner (:class:`~edelweissfe.linsolve.composition.twolevelcycle.TwoLevelPreconditioner`) whose coarse level is spanned
    by the rigid-body modes of the aggregates. The C++ class takes any near null-space; this class hands it the
    rigid-body modes (:meth:`nearNullspace`).

    Parameters
    ----------
    amgclTree, backendBlockSize, backendPrecision, coarseHierarchyPrecision
        The parameters of the cycle, see
        :meth:`~edelweissfe.linsolve.composition.twolevelcycle.TwoLevelPreconditioner.cycleParameters`;
        ``backendBlockSize`` is the field's components per node, 3 (6 rigid-body modes per aggregate) or 2 (3). The
        AMGCL tree means what it means to AMGCL's own hierarchy, with AMGCL's defaults for what it leaves out:
        ``relax``, ``npre`` and ``npost`` the fine smoother; ``coarsening`` (smoothed aggregation) the aggregation and
        the prolongation smoothing of the rigid-body coarse level; and the whole tree the AMGCL hierarchy below it, with
        the strong-connection threshold halved, as AMGCL halves it from one level to the next.
    prolongation
        ``"smoothed"`` (default) or ``"tentative"``: the tentative prolongation alone, a cheaper setup and a sparser
        coarse operator at the price of more outer iterations.
    fineSpectralRadius, fineDominantVector
        A previous build's fine spectral radius and dominant vector to start from (``fineSpectralRadiusReuse``), or
        ``None`` to estimate the radius in full, see :class:`~edelweissfe.linsolve.composition.twolevelcycle.TwoLevelPreconditioner`.

    Raises
    ------
    ValueError
        For settings that cannot be used, before anything is built.
    """

    def __init__(
        self,
        amgclTree: dict,
        backendBlockSize: int,
        backendPrecision: str = "double",
        coarseHierarchyPrecision: str = "double",
        prolongation: str = "smoothed",
        fineSpectralRadius: float = None,
        fineDominantVector=None,
    ):
        from edelweissfe.linsolve.amgcl.amgcl import PyAMGCLRigidBodyCoarseLevel

        self._blockSize = backendBlockSize
        super().__init__(
            PyAMGCLRigidBodyCoarseLevel,
            {
                **self.cycleParameters(amgclTree, backendBlockSize, backendPrecision, coarseHierarchyPrecision),
                "prolongation": prolongation,
            },
            fineSpectralRadius,
            fineDominantVector,
        )

    @staticmethod
    def nearNullspace(block: FieldBlock, coordinates: np.ndarray, scaling: np.ndarray) -> np.ndarray:
        """The near null-space whose vectors span the coarse space of every aggregate: the field's rigid-body modes,
        divided by the equilibration -- :func:`~edelweissfe.linsolve.nullspace.rigidBodyNullspace`, the one definition
        of the modes (6 in 3D, 3 in 2D). The C++ class takes any near null-space of ``B`` or ``B (B + 1) / 2`` columns;
        a subclass may hand it another (the tests' control takes the translations, AMGCL's own coarse space)."""
        return rigidBodyNullspace(block, coordinates, scaling)

    def build(self, A, scaling, coordinates) -> None:
        """Build the rigid-body coarse level, the AMGCL hierarchy below it and the fine smoother, all in C++, on the
        near null-space of :meth:`nearNullspace`.

        Parameters
        ----------
        A
            The field's equilibrated diagonal block :math:`\\hat A = D A D` (scipy CSR), node-major, with its Dirichlet
            identity rows.
        scaling
            The field's slice of the equilibration :math:`D`: the modes are divided by it.
        coordinates
            The field's node coordinates in its dof order, ``(nNodes, >= blockSize)`` (columns beyond the field's
            dimension, e.g. ``z`` of a 2D field, are ignored): the modes are built from them.
        """
        if coordinates is None:
            raise ValueError("rigidBodyCoarseLevel: needs the field's node coordinates to build its rigid-body modes")
        block = FieldBlock("rigidBodyCoarseLevel", 0, A.shape[0], self._blockSize)
        self._preconditioner.build(
            A,
            self.nearNullspace(block, coordinates, scaling),
            self._suppliedFineSpectralRadius(),
            self._fineDominantVector,
        )

    @property
    def nAggregates(self) -> int:
        """The number of aggregates."""
        return self._preconditioner.nAggregates

    @property
    def nCoarse(self) -> int:
        """The coarse dofs: one per mode of the near null-space on every aggregate."""
        return self._preconditioner.nCoarse

    @property
    def droppedModes(self) -> int:
        """The number of modes dropped as linearly dependent."""
        return self._preconditioner.droppedModes

    @property
    def prolongationSmoothingWeight(self) -> float:
        """The damping weight omega of the prolongation smoothing (0 for the tentative prolongation)."""
        return self._preconditioner.prolongationSmoothingWeight

    @property
    def aggregates(self) -> np.ndarray:
        """The aggregate of every node, ``< 0`` for a node in none (e.g. a fully constrained one)."""
        return self._preconditioner.aggregates

    def prolongationMatrix(self):
        """The prolongation ``P`` as the cycle uses it, a scipy CSR matrix (for inspection)."""
        return self._preconditioner.prolongationMatrix()

    def coarseOperator(self):
        """The Galerkin coarse operator ``P^T \\hat A P`` as the hierarchy below holds it, a scipy CSR matrix (for
        inspection); a dropped mode has a zero row and column but for its diagonal entry."""
        return self._preconditioner.coarseOperator()

    def coarseNullspace(self) -> np.ndarray:
        """The near null-space of the hierarchy below: the coefficients of the near null-space -- the field's
        rigid-body modes, about its centroid -- in the coarse basis, per aggregate the R factor of its QR factorization
        (for inspection)."""
        return self._preconditioner.coarseNullspace()
