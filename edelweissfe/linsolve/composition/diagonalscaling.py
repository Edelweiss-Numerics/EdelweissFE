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
"""Symmetric diagonal scaling: :class:`DiagonalScaling`.

.. math::

    M^{-1} = D \\, \\hat M^{-1} D, \\qquad \\hat M \\text{ built on } \\hat A = D A D, \\qquad
    D = \\operatorname{diag}(\\lvert a_{ii} \\rvert^{-1/2})

Every diagonal entry of :math:`\\hat A` is :math:`\\pm 1`. The fields of a coupled system have very different
units -- the rows of a displacement field carry stiffnesses, those of a nonlocal damage field entries of order one
-- and AMG and a block Gauss--Seidel sweep compare matrix entries with each other (the strength-of-connection test
:math:`\\lvert a_{ij} \\rvert \\geq \\theta \\sqrt{\\lvert a_{ii} a_{jj} \\rvert}`, the couplings between fields,
the Chebyshev eigenvalue bounds). Scaled to a unit diagonal, those comparisons measure the matrix, not the choice
of units.

Where the scaling sits in the tree says what it acts on: ``SchurComplement(..., DiagonalScaling(FieldSplit(...)))``
scales only what the field split sees, while the outer Krylov solve still converges the true residual of
:math:`A`.
"""

import numpy as np

import edelweissfe.utils.performancetiming as performancetiming
from edelweissfe.linsolve.composition.base import (
    Description,
    Preconditioner,
    SetUpContext,
    Step,
)


def diagonalScalingOf(absoluteDiagonal: np.ndarray, owner: str = "DiagonalScaling") -> tuple:
    """:math:`D = \\operatorname{diag}(\\lvert a_{ii} \\rvert^{-1/2})` of a matrix, from its
    ``absoluteDiagonal`` :math:`\\lvert a_{ii} \\rvert`. Shared by :class:`DiagonalScaling` and
    :class:`~edelweissfe.linsolve.blockamg.blockamg.BlockAMGSolver`.

    A zero diagonal entry cannot be scaled: :math:`1/\\sqrt{0}` is infinite, which would poison :math:`\\hat A` with
    non-finite values before the preconditioner build, and the failure would surface much later as a NaN correction
    (i.e. as a diverged solution, which reads like a physics problem rather than an unsupported system). Flooring
    the divisor is not a fix either -- a 1e-300 floor merely turns inf into ~1e150 and blows the scaled operator up
    to ~1e149, which is finite and equally useless. A system with a structurally zero diagonal is a saddle-point
    system (e.g. the multiplier block of a Lagrangian constraint, which never writes its own diagonal), and
    smoothed-aggregation AMG on a zero block is not something these preconditioners do: it raises, and names the
    offending dofs.

    A diagonal entry that is NaN or infinite is no better: it makes :math:`D`, and with it a near null-space
    :math:`D^{-1} t`, non-finite, which AMGCL's hierarchy build rejects with a ``RuntimeError`` -- which nothing in
    the nonlinear solvers catches, whereas a NaN correction is a diverged solution and a cutback. Such a system is
    declined like a non-finite right-hand side (checked after the zero diagonal, which is the user's error to see
    first).

    Parameters
    ----------
    absoluteDiagonal
        :math:`\\lvert a_{ii} \\rvert`.
    owner
        The name the error message starts with.

    Returns
    -------
    tuple
        ``(D, None)``, or ``(None, reason)`` for a non-finite diagonal.
    """
    diagonal = absoluteDiagonal
    zeroDiagonal = np.nonzero(diagonal == 0.0)[0]
    if zeroDiagonal.size:
        raise ValueError(
            "{:}: {:} dof(s) have a zero diagonal entry (first at index {:}) and "
            "cannot be diagonally equilibrated. This is characteristic of a saddle-point "
            "system -- a Lagrange-multiplier constraint, for instance, couples the "
            "multiplier without ever writing its own diagonal -- which this field-split "
            "AMG scheme does not support. Use a direct solver for such a model.".format(
                owner, zeroDiagonal.size, int(zeroDiagonal[0])
            )
        )
    nonFiniteDiagonal = np.nonzero(~np.isfinite(diagonal))[0]
    if nonFiniteDiagonal.size:
        return None, "the diagonal is not finite at {:} dof(s) (first at index {:})".format(
            nonFiniteDiagonal.size, int(nonFiniteDiagonal[0])
        )
    return 1.0 / np.sqrt(diagonal), None


class DiagonalScaling(Preconditioner):
    """:math:`M^{-1} = D \\hat M^{-1} D`, with :math:`\\hat M` (the child ``inner``) built on :math:`\\hat A = D A D`,
    :math:`D = \\operatorname{diag}(\\lvert a_{ii} \\rvert^{-1/2})`. See the module documentation.

    When only the values of :math:`A` change (no rebuild asked for), :math:`D` of the last build is kept: any
    :math:`D` gives the same system in other variables, and the child's reused preconditioner was built for this
    one.

    Parameters
    ----------
    inner
        The preconditioner :math:`\\hat M^{-1}` of the scaled matrix.
    """

    def __init__(self, inner: Preconditioner):
        self.inner = inner
        self.D = None

    def setUp(self, A, context: SetUpContext) -> None:
        with performancetiming.timeit("equilibration"):
            D, reason = diagonalScalingOf(A.absoluteDiagonal())
        if D is None:
            raise FloatingPointError("DiagonalScaling: " + reason)
        self.D = D
        self.inner.setUp(A.scaled(D), context.scaledBy(D))

    def update(self, A, change) -> None:
        if change.requiresRebuild or self.D is None:
            self.setUp(A, change.context)
        else:
            self.inner.update(A.scaled(self.D), change.withContext(change.context.scaledBy(self.D)))

    def __call__(self, r: np.ndarray) -> np.ndarray:
        return self.D * self.inner(self.D * r)

    def describe(self) -> Description:
        return Description(
            name="DiagonalScaling",
            formula="D · {inner} · D",
            latex="D \\, {inner} \\, D",
            parameters={"D": "diag(1/√|aᵢᵢ|)"},
            children=(("inner", self.inner),),
            steps=(Step("r̂ = D r"), Step("ẑ = M̂⁻¹ r̂", "inner"), Step("z = D ẑ")),
        )
