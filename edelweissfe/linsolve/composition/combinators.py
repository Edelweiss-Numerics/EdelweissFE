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
"""Generic combinators: :class:`Multiplicative`, :class:`Additive`, :class:`OnField`, :class:`CoarseCorrection`, and the
simplest smoother, :class:`Jacobi`.

They build new methods without new classes, and they show that methods which look unrelated share one structure:

- a block Gauss--Seidel field split is a multiplicative combination of corrections on single fields,
  ``Multiplicative([OnField("displacement", M_u), OnField("nonlocal damage", M_d)])``;
- a two-grid cycle is a multiplicative combination of a smoother, a coarse correction and the smoother again,
  ``Multiplicative([Jacobi(), CoarseCorrection(P, coarse), Jacobi()])``.

They are written for clarity, not speed: each correction of :class:`Multiplicative` costs a product with the whole
matrix. The fast versions of the same methods are :class:`~edelweissfe.linsolve.composition.fieldsplit.FieldSplit`
and :class:`~edelweissfe.linsolve.composition.twogrid.TwoGrid`.
"""

import numpy as np
import scipy.sparse as sp

from edelweissfe.linsolve.base import FieldBlock
from edelweissfe.linsolve.composition.base import (
    Description,
    Preconditioner,
    SetUpContext,
    Step,
)
from edelweissfe.linsolve.composition.systemmatrix import SystemMatrix


class Multiplicative(Preconditioner):
    """Successive corrections, each one of the residual the previous ones left:

    .. math::

        z \\leftarrow M_1^{-1} r, \\qquad z \\leftarrow z + M_k^{-1} (r - A z), \\quad k = 2, \\ldots

    Parameters
    ----------
    corrections
        The preconditioners :math:`M_k^{-1}`, all set up for the same matrix.
    """

    def __init__(self, corrections: list):
        self.corrections = list(corrections)
        self.A = None

    def setUp(self, A, context: SetUpContext) -> None:
        self.A = A
        for correction in self.corrections:
            correction.setUp(A, context)

    def update(self, A, change) -> None:
        self.A = A
        for correction in self.corrections:
            correction.update(A, change)

    def __call__(self, r: np.ndarray) -> np.ndarray:
        first, *others = self.corrections
        z = first(r)
        for correction in others:
            z = z + correction(r - self.A @ z)
        return z

    def describe(self) -> Description:
        roles = ["M" + str(k + 1) for k in range(len(self.corrections))]
        return Description(
            name="Multiplicative",
            formula="(" + " ∘ ".join("{" + role + "}" for role in reversed(roles)) + ")",
            latex="\\left(" + " \\circ ".join("{" + role + "}" for role in reversed(roles)) + "\\right)",
            children=tuple(zip(roles, self.corrections)),
            steps=(Step("z = M1⁻¹ r", roles[0]),)
            + tuple(Step("z += {0}⁻¹ (r − A z)".format(role), role) for role in roles[1:]),
        )


class Additive(Preconditioner):
    """Corrections of the same residual, summed: :math:`z = \\sum_k M_k^{-1} r`.

    Parameters
    ----------
    corrections
        The preconditioners :math:`M_k^{-1}`, all set up for the same matrix.
    """

    def __init__(self, corrections: list):
        self.corrections = list(corrections)

    def setUp(self, A, context: SetUpContext) -> None:
        for correction in self.corrections:
            correction.setUp(A, context)

    def update(self, A, change) -> None:
        for correction in self.corrections:
            correction.update(A, change)

    def __call__(self, r: np.ndarray) -> np.ndarray:
        return sum(correction(r) for correction in self.corrections)

    def describe(self) -> Description:
        roles = ["M" + str(k + 1) for k in range(len(self.corrections))]
        return Description(
            name="Additive",
            formula="(" + " + ".join("{" + role + "}" for role in roles) + ")",
            latex="\\left(" + " + ".join("{" + role + "}" for role in roles) + "\\right)",
            children=tuple(zip(roles, self.corrections)),
            steps=tuple(Step("z += {0}⁻¹ r".format(role), role) for role in roles),
        )


class OnField(Preconditioner):
    """A correction on the dofs of one field: :math:`z = R^T M^{-1} R r`, with :math:`R` the restriction to the field
    and :math:`M` built on its diagonal block :math:`R A R^T`.

    Parameters
    ----------
    fieldName
        The field.
    inner
        :math:`M^{-1}`, the preconditioner of the field's block.
    """

    def __init__(self, fieldName: str, inner: Preconditioner):
        self.fieldName = fieldName
        self.inner = inner
        self.dofs = None

    def _fieldIn(self, context: SetUpContext) -> FieldBlock:
        for field in context.fields:
            if field.name == self.fieldName:
                return field
        raise ValueError("OnField: the system has no field '{:}'".format(self.fieldName))

    def setUp(self, A, context: SetUpContext) -> None:
        field = self._fieldIn(context)
        self.dofs = slice(field.start, field.stop)
        self.inner.setUp(SystemMatrix(A.matrix()[self.dofs, self.dofs]), context.restrictedToField(field))

    def update(self, A, change) -> None:
        field = self._fieldIn(change.context)
        self.dofs = slice(field.start, field.stop)
        fieldChange = change.withContext(change.context.restrictedToField(field))
        self.inner.update(SystemMatrix(A.matrix()[self.dofs, self.dofs]), fieldChange)

    def __call__(self, r: np.ndarray) -> np.ndarray:
        z = np.zeros_like(r)
        z[self.dofs] = self.inner(r[self.dofs])
        return z

    def describe(self) -> Description:
        return Description(
            name="OnField",
            formula="Rᵀ {inner} R",
            latex="R^T {inner} R",
            parameters={"field": self.fieldName},
            children=(("inner", self.inner),),
            steps=(Step("z_f = M⁻¹ r_f   (f = " + self.fieldName + ")", "inner"),),
        )


class CoarseCorrection(Preconditioner):
    """The coarse-grid correction :math:`z = P A_c^{-1} P^T r` with the Galerkin coarse operator
    :math:`A_c = P^T A P`. With an exact coarse solve, :math:`P A_c^{-1} P^T A` is the :math:`A`-orthogonal projection
    onto the range of :math:`P`.

    Parameters
    ----------
    prolongation
        :math:`P`, a sparse matrix, or a callable ``(A, context) -> P`` that builds it for the matrix.
    coarse
        :math:`A_c^{-1}`, e.g. a :class:`~edelweissfe.linsolve.composition.direct.DirectSolver` or an
        :class:`~edelweissfe.linsolve.composition.amg.AMG`.
    """

    def __init__(self, prolongation, coarse: Preconditioner):
        self.prolongation = prolongation
        self.coarse = coarse
        self.P = None

    def setUp(self, A, context: SetUpContext) -> None:
        P = self.prolongation(A, context) if callable(self.prolongation) else self.prolongation
        self.P = sp.csr_matrix(P)
        A_c = (self.P.T @ A.matrix() @ self.P).tocsr()
        coarseContext = SetUpContext(fields=(FieldBlock("coarse", 0, A_c.shape[0], 1),))
        self.coarse.setUp(SystemMatrix(A_c), coarseContext)
        self.statistics.sizes["coarse dofs"] = A_c.shape[0]

    def __call__(self, r: np.ndarray) -> np.ndarray:
        return self.P @ self.coarse(self.P.T @ r)

    def describe(self) -> Description:
        return Description(
            name="CoarseCorrection",
            formula="P {coarse} Pᵀ",
            latex="P \\, {coarse} \\, P^T",
            children=(("coarse", self.coarse),),
            steps=(Step("r_c = Pᵀ r"), Step("z_c = A_c⁻¹ r_c,  A_c = Pᵀ A P", "coarse"), Step("z = P z_c")),
        )


class Jacobi(Preconditioner):
    """Damped Jacobi, the simplest smoother: :math:`z = \\omega \\operatorname{diag}(A)^{-1} r`.

    Parameters
    ----------
    omega
        The damping.
    """

    def __init__(self, omega: float = 2.0 / 3.0):
        self.omega = omega
        self.inverseDiagonal = None

    def setUp(self, A, context: SetUpContext) -> None:
        self.inverseDiagonal = 1.0 / A.matrix().diagonal()

    def __call__(self, r: np.ndarray) -> np.ndarray:
        return self.omega * self.inverseDiagonal * r

    def describe(self) -> Description:
        return Description(
            name="Jacobi",
            formula="ω diag(A)⁻¹",
            latex="\\omega \\, \\mathrm{diag}(A)^{-1}",
            parameters={"omega": round(self.omega, 4)},
        )
