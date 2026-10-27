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

Both are one two-level cycle -- fine Chebyshev sweeps on the field's scaled block, and the coarse correction on the
Galerkin coarse operator -- that differ in their coarse level and coarse solve: the rigid-body coarse level solves with
an AMGCL hierarchy of its own, the p-two-grid with a component of the composed solver. Both are built and applied in
C++: the shared cycle is ``TwoLevelPreconditioner`` in ``amgcl-wrapper.hpp``, wrapped as ``_PyAMGCLTwoLevel`` in
``amgcl.pyx``. :class:`TwoLevelPreconditioner` mirrors those bases here: it creates the C++ object, which validates
every parameter, keeps the fine spectral radius a build may start from, and passes the cycle, the fine spectral radius a
build ends with, the timings and the report through. A derived class chooses the parameters and builds its coarse level,
in its ``build``, from what that coarse level needs.
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

    Raises
    ------
    ValueError
        For parameters the C++ class cannot use, before anything is built.
    """

    def __init__(self, cppClass, parameters: dict):
        try:
            self._preconditioner = cppClass(parameters)
        except RuntimeError as error:
            raise ValueError(str(error)) from error

    @staticmethod
    def cycleParameters(amgclTree: dict, backendBlockSize: int, backendPrecision: str):
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
        """
        return {
            "amgcl": amgclTree,
            "backendBlockSize": backendBlockSize,
            "backendPrecision": backendPrecision,
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

    @property
    def fineSpectralRadius(self) -> float:
        """The estimate of the fine spectral radius the smoother was built with: ``relax.spectralRadiusSteps`` Arnoldi
        steps from a start vector seeded per row, zero on the Dirichlet rows (``amgcl-spectral-radius.hpp``)."""
        return self._preconditioner.fineSpectralRadius

    def applyPreconditioner(self, r: np.ndarray) -> np.ndarray:
        """One two-level cycle: fine Chebyshev sweeps and the coarse correction on the free dofs; the Dirichlet rows
        are solved exactly, ``x[i] = r[i] / A[i, i]``."""
        return self._preconditioner.applyPreconditioner(r)

    def timings(self) -> dict:
        """Wall time [s] of the last build's stages and the cumulative apply (fine vs. coarse) -- kept measured, not
        assumed, since the split decides where the next optimization goes."""
        return self._preconditioner.timings()

    def report(self) -> str:
        """The coarse level in figures, and the report of its coarse solve (diagnostics, not a convergence gate)."""
        return self._preconditioner.report()
