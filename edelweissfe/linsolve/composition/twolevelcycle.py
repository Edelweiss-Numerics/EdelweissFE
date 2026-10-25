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
#  This library is free software; you can redistribute it and/or
#  modify it under the terms of the GNU Lesser General Public
#  License as published by the Free Software Foundation; either
#  version 2.1 of the License, or (at your option) any later version.
#
#  The full text of the license can be found in the file LICENSE.md at
#  the top level directory of EdelweissFE.
#  ---------------------------------------------------------------------
"""What the two-level cycles of a vector field (:class:`~edelweissfe.linsolve.composition.twolevel.TwoLevel`) share: the p-two-grid (:mod:`~edelweissfe.linsolve.composition.ptwogrid`) and the rigid-body coarse level
(:mod:`~edelweissfe.linsolve.composition.rigidbodycoarselevel`).

Both are one two-level cycle -- fine Chebyshev sweeps on the field's equilibrated block, and the coarse correction
through an AMGCL hierarchy on the Galerkin coarse operator -- that differ in their coarse level alone, and both are built
and applied in C++: the shared cycle is ``TwoLevelPreconditioner`` in ``amgcl-wrapper.hpp``, wrapped as
``_PyAMGCLTwoLevel`` in ``amgcl.pyx``. :class:`TwoLevelPreconditioner` mirrors those bases here: it creates the C++ object,
which validates every parameter, keeps the fine spectral radius a build may start from, and passes the cycle, the fine
spectral radius a build ends with, the timings and the report through. A derived class chooses the parameters and builds
its coarse level, in :meth:`TwoLevelPreconditioner.build`, whose signature is the same for both.
"""

import numpy as np


class TwoLevelPreconditioner:
    """A two-level preconditioner of one vector field's equilibrated diagonal block, built once per hierarchy refresh and
    applied many times within the outer Krylov solve, with the ``build`` / ``applyPreconditioner`` life cycle of
    :class:`~edelweissfe.linsolve.amgcl.amgcl.PyAMGCLSolver`.

    Parameters
    ----------
    cppClass
        The Cython class of the C++ preconditioner, created here from ``parameters``.
    parameters
        Its parameters, which its constructor checks -- the one place they are checked.
    fineSpectralRadius, fineDominantVector
        The fine operator's spectral radius for the Chebyshev bounds, and the dominant vector to re-check it from, or
        ``None`` to estimate it in full by power iteration from a deterministic start vector -- the most expensive stage
        of a build. A caller rebuilding on a slowly changing operator can hand back a previous build's
        :attr:`fineSpectralRadius` and :attr:`fineDominantVector`. The radius is then not reused blindly but re-checked
        by a few power iterations continued from that vector: a re-check estimate more than 1% above the supplied radius
        replaces it, and one beyond the margin of the smoother's upper bound -- the operator stiffened more than its
        bounds cover -- triggers the full estimate. A radius without a vector is used as supplied.

    Raises
    ------
    ValueError
        For parameters the C++ class cannot use, before anything is built.
    """

    def __init__(self, cppClass, parameters: dict, fineSpectralRadius: float = None, fineDominantVector=None):
        self._fineSpectralRadius = fineSpectralRadius
        self._fineDominantVector = fineDominantVector
        try:
            self._preconditioner = cppClass(parameters)
        except RuntimeError as error:
            raise ValueError(str(error)) from error

    @staticmethod
    def cycleParameters(amgclTree: dict, backendBlockSize: int, backendPrecision: str, coarseHierarchyPrecision: str):
        """The parameters of the cycle, in the one spelling both C++ classes read (``TwoLevelCycleParameters`` in
        ``amgcl-wrapper.hpp``).

        Parameters
        ----------
        amgclTree
            The field's AMGCL tree. Its ``relax`` (a Chebyshev polynomial, without Jacobi scaling), ``npre`` and
            ``npost`` are the fine smoother, with AMGCL's defaults for what it leaves out; the default trees, the one
            place of EdelweissFE's defaults, are :func:`~edelweissfe.linsolve.composition.amg.defaultFieldTree`.
        backendBlockSize
            The storage of the fine operator: ``1`` (scalar entries), or ``2``/``3`` (node-wise blocks of the field's
            components: less memory traffic per product, the same cycle up to the order of summation).
        backendPrecision
            The value type of the stored fine operator and of :math:`P`, :math:`P^T`: ``"double"``, or ``"float"``
            (not for 2x2 blocks) for half the bytes per entry; vectors and sums stay double, and the spectral radius
            is estimated on the double operator.
        coarseHierarchyPrecision
            The value type of the AMGCL hierarchy of the coarse level (its whole setup): ``"double"`` or ``"float"``.
        """
        return {
            "amgcl": amgclTree,
            "backendBlockSize": backendBlockSize,
            "backendPrecision": backendPrecision,
            "coarseHierarchyPrecision": coarseHierarchyPrecision,
        }

    @classmethod
    def parameterRejection(cls, *args, **kwargs) -> "str | None":
        """Why these constructor arguments cannot be used, or ``None``. The C++ class checks its parameters as it is
        created, so this creates one, without a build."""
        try:
            cls(*args, **kwargs)
        except (TypeError, ValueError) as error:
            return str(error)
        return None

    def build(self, A, scaling: np.ndarray, coordinates: "np.ndarray | None") -> None:
        """Build the coarse level, its AMGCL hierarchy and the fine Chebyshev smoother for the field's equilibrated
        diagonal block, all in C++: everything :meth:`applyPreconditioner` needs.

        Parameters
        ----------
        A
            The field's equilibrated diagonal block :math:`\\hat A = D A D` (scipy CSR), node-major, with its Dirichlet
            identity rows.
        scaling
            The field's slice of the equilibration :math:`D`: a near null-space of :math:`\\hat A` is :math:`D^{-1}`
            times one of :math:`A`.
        coordinates
            The field's node coordinates in its dof order, ``(nNodes, >= dimension)`` (columns beyond the field's
            dimension, e.g. ``z`` of a 2D field, are ignored), or ``None`` if they are unavailable -- whether a coarse
            level can do without them is its own business.
        """
        raise NotImplementedError()

    def _suppliedFineSpectralRadius(self) -> float:
        """The fine spectral radius to hand to the C++ build: the supplied one, or 0 for none, which the C++ build takes
        as "estimate it in full"."""
        return self._fineSpectralRadius if self._fineSpectralRadius is not None else 0.0

    @property
    def fineSpectralRadius(self) -> float:
        """The fine spectral radius the smoother was built with (estimated, or the one supplied), to hand back to the
        next build of the field instead of paying for the estimate again."""
        return self._preconditioner.fineSpectralRadius

    @property
    def fineDominantVector(self) -> np.ndarray:
        """The estimate of the fine operator's dominant eigenvector, to hand back together with
        :attr:`fineSpectralRadius` for the re-check of a reused radius."""
        return self._preconditioner.fineDominantVector

    @property
    def fineSpectralRadiusSource(self) -> str:
        """How the last build arrived at its fine spectral radius: ``"estimated"`` in full, ``"supplied"`` and kept,
        ``"raised"`` by the re-check, or ``"re-estimated"`` in full -- see the ``fineSpectralRadius`` parameter."""
        return self._preconditioner.fineSpectralRadiusSource

    def applyPreconditioner(self, r: np.ndarray) -> np.ndarray:
        """One two-level cycle: fine Chebyshev sweeps and the coarse correction on the free dofs; the Dirichlet rows
        are solved exactly, ``x[i] = r[i] / A[i, i]``."""
        return self._preconditioner.applyPreconditioner(r)

    def timings(self) -> dict:
        """Wall time [s] of the last build's stages and the cumulative apply (fine vs. coarse) -- kept measured, not
        assumed, since the split decides where the next optimization goes."""
        return self._preconditioner.timings()

    def report(self) -> str:
        """The coarse level in figures, and the report of its AMGCL hierarchy (diagnostics, not a convergence gate)."""
        return self._preconditioner.report()
