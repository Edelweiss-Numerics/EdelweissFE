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
"""Combining preconditioners: :class:`Multiplicative`, :class:`Additive`, :class:`OnField`, :class:`CoarseCorrection`,
and the simplest smoother, :class:`Jacobi`.

They are the vocabulary of Schwarz methods, and methods which look unrelated share their structure:

- a correction on one field is :math:`R_i^T M_i^{-1} R_i`, with :math:`R_i` the restriction to the field's dofs and
  :math:`M_i^{-1}` a preconditioner of its diagonal block :math:`A_{ii} = R_i A R_i^T`: ``OnField(name, M_i)``;
- block Gauss--Seidel over the fields is their multiplicative combination,
  ``Multiplicative([OnField("displacement", M_u), OnField("nonlocal damage", M_d)])``, and block Jacobi their additive
  one, ``Additive([...])``; a symmetric sweep visits the fields forward and back, see :func:`gaussSeidelOrder`;
- a two-level cycle is a multiplicative combination of a smoother, a coarse correction and the smoother again,
  ``Multiplicative([Jacobi(), CoarseCorrection(P, coarseSolve), Jacobi()])``.

This is also the fast way. A combination asks each correction which field it acts on (:attr:`~edelweissfe.linsolve.
composition.base.Preconditioner.fieldIndex`), and computes the residual of a correction on field :math:`i` only on
that field's rows, from the fields corrected so far:

.. math::

    r_i - \\sum_{j:\\, z_j \\neq 0} A_{ij} z_j

with the blocks :math:`A_{ij}` formed once per matrix in one threaded pass
(:meth:`~edelweissfe.linsolve.composition.systemmatrix.SystemMatrix.fieldSplit`) and read only as products. A
correction on all dofs gets the residual of all fields.
"""

from collections.abc import Callable

import numpy as np
import scipy.sparse as sp

from edelweissfe.linsolve.composition.base import (
    Description,
    Preconditioner,
    SetUpContext,
    Step,
    requirePreconditioner,
)
from edelweissfe.linsolve.composition.systemmatrix import (
    FieldBlockMatrix,
    FieldSplitBlocks,
    SystemMatrix,
)


def gaussSeidelOrder(fieldCorrections: list, sweeps: int = 1, symmetric: bool = False) -> list:
    """The corrections of block Gauss--Seidel sweeps over fields, in the order a :class:`Multiplicative` applies
    them: ``sweeps`` forward sweeps, each followed by a backward one if ``symmetric`` -- which starts one field before
    the last, since the last field was just corrected with everything else current. For two fields ``u, d``:
    ``[u, d]``, symmetric ``[u, d, u]``, two symmetric sweeps ``[u, d, u, u, d, u]`` (a field corrected twice in a row
    is corrected twice: the second correction is one of the residual the first left)."""
    if sweeps < 1:
        raise ValueError("gaussSeidelOrder: sweeps must be at least 1, got {:}".format(sweeps))
    fieldCorrections = list(fieldCorrections)
    return (fieldCorrections + (fieldCorrections[-2::-1] if symmetric else [])) * sweeps


def multiplicativeCorrections(blocks: FieldSplitBlocks, corrections: list, r: np.ndarray) -> np.ndarray:
    """Successive corrections, each of the residual the previous ones left: :math:`z \\leftarrow z + R_k^T M_k^{-1} R_k
    (r - A z)`, with the residual of correction :math:`k` formed only on its field's rows (see the module
    documentation). The application of :class:`Multiplicative`.

    Parameters
    ----------
    blocks
        The field blocks :math:`A_{ij}` (their products with vectors).
    corrections
        ``(i, apply)`` per correction, in order: the index ``i`` of its field (``None``: all dofs) and its
        :math:`M_k^{-1}`, a callable on the residual of its dofs.
    r
        The vector to precondition.

    Returns
    -------
    np.ndarray
        :math:`z`.
    """
    fields = blocks.slices
    # z per field; None while still zero
    z = [None] * len(fields)

    def residualOf(i: int) -> np.ndarray:
        r_i = r[fields[i]].copy()
        for j, z_j in enumerate(z):
            if j != i and z_j is not None:
                r_i -= blocks.couplingTimes(i, j, z_j)
        if z[i] is not None:
            r_i -= blocks.diagonalBlockTimes(i, z[i])
        return r_i

    for i, apply in corrections:
        if i is None:
            residual = r if all(z_j is None for z_j in z) else np.concatenate([residualOf(k) for k in range(len(z))])
            dz = apply(residual)
            for k, field in enumerate(fields):
                z[k] = dz[field] if z[k] is None else z[k] + dz[field]
        else:
            dz_i = apply(residualOf(i))
            z[i] = dz_i if z[i] is None else z[i] + dz_i
    return _assembled(z, fields)


def _assembled(z: list, fields: list) -> np.ndarray:
    """The vector of the per-field parts ``z`` (``None``: zero)."""
    return np.concatenate([np.zeros(f.stop - f.start) if z_i is None else z_i for z_i, f in zip(z, fields)])


class _Combination(Preconditioner):
    """What :class:`Multiplicative` and :class:`Additive` share: their corrections, given as a list, or as a function
    of the context that makes them once the fields are known (e.g. one correction per field, whatever the model)."""

    def __init__(self, corrections: "list | Callable"):
        super().__init__()
        name = type(self).__name__
        if callable(corrections) and not isinstance(corrections, Preconditioner):
            self._makeCorrections = corrections
            self.corrections = []
        else:
            corrections = list(corrections)
            if not corrections:
                raise ValueError("{:}: needs at least one correction".format(name))
            for k, correction in enumerate(corrections):
                requirePreconditioner(name, "correction {:}".format(k + 1), correction)
            self._makeCorrections = None
            self.corrections = corrections
        # the fields the corrections were made for: they are kept, with what they learnt, while the fields stay
        self._madeFor = None
        # the dof ranges of the fields, taken from every matrix (None: never set up)
        self.bounds = None

    def _distinct(self) -> list:
        """Each correction once (a correction may be applied several times, e.g. in a symmetric sweep)."""
        distinct = []
        for correction in self.corrections:
            if not any(correction is other for other in distinct):
                distinct.append(correction)
        return distinct

    def _makeFor(self, context: SetUpContext) -> bool:
        """Make the corrections for the fields of ``context``, if they are made from a function and not yet for these
        fields; whether new ones were made."""
        names = tuple(field.name for field in context.fields)
        if self._makeCorrections is None or names == self._madeFor:
            return False
        corrections = list(self._makeCorrections(context))
        name = type(self).__name__
        if not corrections:
            raise ValueError("{:}: the function of the corrections made none for the fields {:}".format(name, names))
        for k, correction in enumerate(corrections):
            requirePreconditioner(name, "correction {:}".format(k + 1), correction)
        self.corrections, self._madeFor = corrections, names
        return True

    def _setUp(self, A, context: SetUpContext) -> None:
        self._makeFor(context)
        self.bounds = [(field.start, field.stop) for field in context.fields]
        for correction in self._distinct():
            correction.setUp(A, context)
        self._requireEveryFieldCorrected(context)

    def _update(self, A, change) -> None:
        # set up only with nothing to update (new corrections, or the first matrix); a rebuild is the corrections' own
        if self._makeFor(change.context) or self.bounds is None:
            self.setUp(A, change.context)
            return
        self.bounds = [(field.start, field.stop) for field in change.context.fields]
        for correction in self._distinct():
            correction.update(A, change)
        self._requireEveryFieldCorrected(change.context)

    def _requireEveryFieldCorrected(self, context: SetUpContext) -> None:
        """Raise for a field (with dofs) that no correction acts on: z would stay zero there, a singular
        preconditioner."""
        corrected = {correction.fieldIndex for correction in self._distinct()}
        if None in corrected:
            return
        missing = [f.name for i, f in enumerate(context.fields) if i not in corrected and f.stop > f.start]
        if missing:
            raise ValueError(
                "{:}: no correction acts on the field(s) {:}; add one (e.g. OnField(name, AMG())) for each".format(
                    type(self).__name__, ", ".join(missing)
                )
            )

    def children(self) -> tuple:
        """``("M1", correction)``, ``("M2", ...)``: each distinct correction once, in the order of first use (none
        while the corrections are still to be made per field)."""
        return tuple(("M" + str(k + 1), correction) for k, correction in enumerate(self._distinct()))

    def _rolesInOrder(self) -> list:
        """The role (see :meth:`children`) of each correction, in the order they are applied."""
        children = self.children()
        return [next(role for role, child in children if child is correction) for correction in self.corrections]


class Multiplicative(_Combination):
    """Successive corrections, each one of the residual the previous ones left:

    .. math::

        z \\leftarrow M_1^{-1} r, \\qquad z \\leftarrow z + M_k^{-1} (r - A z), \\quad k = 2, \\ldots

    A correction on one field (:class:`OnField`) gets the residual of that field alone, see the module documentation.

    Parameters
    ----------
    corrections
        The preconditioners :math:`M_k^{-1}`, all set up for the same matrix; one may appear several times (it is set
        up once). Or a function ``context -> list`` that makes them at the first set-up, when the fields are known.
    """

    def _forget(self) -> None:
        self.blocks = None

    def _setUp(self, A, context: SetUpContext) -> None:
        super()._setUp(A, context)
        self.blocks = A.fieldSplit(self.bounds)

    def _update(self, A, change) -> None:
        super()._update(A, change)
        # the residuals always come from the current matrix
        self.blocks = A.fieldSplit(self.bounds)

    def _apply(self, r: np.ndarray) -> np.ndarray:
        return multiplicativeCorrections(self.blocks, [(c.fieldIndex, c.applyRestricted) for c in self.corrections], r)

    def finishSolve(self) -> None:
        # the blocks belong to this solve's matrix
        self.blocks = None
        super().finishSolve()

    def describe(self) -> Description:
        if not self.corrections:
            return Description(name="Multiplicative", formula="Multiplicative(made per field)", latex="")
        order = self._rolesInOrder()
        return Description(
            name="Multiplicative",
            formula="(" + " ∘ ".join("{" + role + "}" for role in reversed(order)) + ")",
            latex="\\left(" + " \\circ ".join("{" + role + "}" for role in reversed(order)) + "\\right)",
            steps=(Step("z = {0}⁻¹ r".format(order[0]), order[0]),)
            + tuple(Step("z += {0}⁻¹ (r − A z)".format(role), role) for role in order[1:]),
        )


class Additive(_Combination):
    """Corrections of the same residual, summed: :math:`z = \\sum_k M_k^{-1} r`. Over corrections on single fields,
    this is block Jacobi.

    Parameters
    ----------
    corrections
        The preconditioners :math:`M_k^{-1}`, all set up for the same matrix; or a function ``context -> list`` that
        makes them at the first set-up.
    """

    def _apply(self, r: np.ndarray) -> np.ndarray:
        fields = [slice(start, stop) for start, stop in self.bounds]
        z = [None] * len(fields)
        for correction in self.corrections:
            i = correction.fieldIndex
            if i is None:
                dz = correction(r)
                parts = [(k, dz[field]) for k, field in enumerate(fields)]
            else:
                parts = [(i, correction.applyRestricted(r[fields[i]]))]
            for k, dz_k in parts:
                z[k] = dz_k if z[k] is None else z[k] + dz_k
        return _assembled(z, fields)

    def describe(self) -> Description:
        if not self.corrections:
            return Description(name="Additive", formula="Additive(made per field)", latex="")
        order = self._rolesInOrder()
        return Description(
            name="Additive",
            formula="(" + " + ".join("{" + role + "}" for role in order) + ")",
            latex="\\left(" + " + ".join("{" + role + "}" for role in order) + "\\right)",
            steps=tuple(Step("z += {0}⁻¹ r".format(role), role) for role in order),
        )


class OnField(Preconditioner):
    """A correction on the dofs of one field: :math:`z = R^T M^{-1} R r`, with :math:`R` the restriction to the field
    and :math:`M` built on its diagonal block :math:`A_{ff} = R A R^T`.

    The block comes from the field split of the matrix (:meth:`~edelweissfe.linsolve.composition.systemmatrix.
    SystemMatrix.fieldSplit`, one threaded pass for all fields), and is cut out only if ``inner`` builds on it: a
    preconditioner reused for the new values never needs it.

    Parameters
    ----------
    fieldName
        The field.
    inner
        :math:`M^{-1}`, the preconditioner of the field's block.
    """

    def __init__(self, fieldName: str, inner: Preconditioner):
        if not isinstance(fieldName, str):
            raise TypeError("OnField: the field is given by its name, got {!r}".format(fieldName))
        requirePreconditioner("OnField('{:}')".format(fieldName), "inner", inner)
        super().__init__()
        self.fieldName = fieldName
        self.inner = inner

    def _forget(self) -> None:
        self.dofs = None
        self._fieldIndex = None

    @property
    def fieldIndex(self) -> "int | None":
        return self._fieldIndex

    def _restricted(self, A, context: SetUpContext) -> tuple:
        """The field's index, its block as a matrix, and its context."""
        names = [field.name for field in context.fields]
        if self.fieldName not in names:
            raise ValueError(
                "OnField: the system has no field '{:}'; its fields are {:}".format(self.fieldName, ", ".join(names))
            )
        i = names.index(self.fieldName)
        field = context.fields[i]
        self._fieldIndex = i
        self.dofs = slice(field.start, field.stop)
        bounds = [(f.start, f.stop) for f in context.fields]
        return FieldBlockMatrix(lambda: A.fieldSplit(bounds), i), context.restrictedToField(field)

    def _setUp(self, A, context: SetUpContext) -> None:
        block, fieldContext = self._restricted(A, context)
        self.inner.setUp(block, fieldContext)

    def _update(self, A, change) -> None:
        block, fieldContext = self._restricted(A, change.context)
        self.inner.update(block, change.withContext(fieldContext))

    def _applyRestricted(self, r_f: np.ndarray) -> np.ndarray:
        return self.inner(r_f)

    def _apply(self, r: np.ndarray) -> np.ndarray:
        z = np.zeros_like(r)
        z[self.dofs] = self.inner(r[self.dofs])
        return z

    def children(self) -> tuple:
        return (("inner", self.inner),)

    def describe(self) -> Description:
        return Description(
            name="OnField",
            formula="Rᵀ {inner} R",
            latex="R^T {inner} R",
            parameters={"field": self.fieldName},
            steps=(Step("z_f = M⁻¹ r_f   (f = " + self.fieldName + ")", "inner"),),
        )


class CoarseCorrection(Preconditioner):
    """The coarse-grid correction :math:`z = P A_c^{-1} P^T r` with the Galerkin coarse operator
    :math:`A_c = P^T A P`. With an exact coarse solve, :math:`P A_c^{-1} P^T A` is the :math:`A`-orthogonal projection
    onto the range of :math:`P`.

    :math:`P` is built at a set-up and kept until the next rebuild (see
    :attr:`~edelweissfe.linsolve.composition.base.MatrixChange.requiresRebuild`); in between, the coarse solve takes
    the new :math:`A_c` as an update, and keeps what it can (an :class:`~edelweissfe.linsolve.composition.amg.AMG` its
    hierarchy, as :class:`~edelweissfe.linsolve.composition.twolevel.TwoLevel` does).

    The coarse solve is a component like any other, set up on :math:`A_c` with a context that says what the coarse
    dofs are. A callable that builds :math:`P` knows that, and returns it with :math:`P`: for the P1 interpolant,
    ``context.ofCoarseNodes(field, isCorner)`` -- the field on its corner nodes, their coordinates -- on which
    ``DiagonalScaling(TwoLevel(coarseSpace=RigidBodyModesPerAggregate()))`` works as on any displacement field; for
    coarse dofs that are no nodes, ``context.ofCoarseDofs(nCoarse)``. A fixed :math:`P` says nothing about its columns:
    its coarse dofs are one field of scalar dofs, ``context.ofCoarseDofs``.

    :math:`P` acts on the matrix the correction gets. Under a
    :class:`~edelweissfe.linsolve.composition.diagonalscaling.DiagonalScaling` that is :math:`\\hat A = D A D`, whose
    variables are :math:`D^{-1} u`: an interpolation of displacements is then :math:`D^{-1} P_{geo} D_c` (the scaling
    is ``context.scaling``), as :class:`~edelweissfe.linsolve.composition.twolevel.CornerNodeInterpolation` builds it.

    With smoothers around it, ``Multiplicative([smoother, CoarseCorrection(P, coarseSolve), smoother])`` is a two-level
    cycle spelled out: any :math:`P`, any coarse solve, but in SciPy and about twice as costly per application as the
    C++ cycle :class:`~edelweissfe.linsolve.composition.twolevel.TwoLevel`, which is kept beside it on purpose (see the
    "two-level cycle: two ways" section of ``doc/source/documentation/composedlinsolvers.rst``).

    Parameters
    ----------
    prolongation
        :math:`P`, a sparse matrix; or a callable ``(A, context) -> (P, coarseContext)`` that builds it for the matrix
        and says what its coarse dofs are.
    coarseSolve
        :math:`A_c^{-1}`, e.g. a :class:`~edelweissfe.linsolve.composition.direct.DirectSolver`, an
        :class:`~edelweissfe.linsolve.composition.amg.AMG`, or a two-level cycle of its own.
    """

    def __init__(self, prolongation, coarseSolve: Preconditioner):
        requirePreconditioner("CoarseCorrection", "coarseSolve", coarseSolve)
        super().__init__()
        self.prolongation = prolongation
        self.coarseSolve = coarseSolve

    def _forget(self) -> None:
        self.P = None
        self.coarseContext = None

    def _setUp(self, A, context: SetUpContext) -> None:
        if callable(self.prolongation):
            built = self.prolongation(A, context)
            if not (isinstance(built, tuple) and len(built) == 2 and isinstance(built[1], SetUpContext)):
                raise TypeError(
                    "CoarseCorrection: the prolongation callable must return a tuple (P, coarseContext), the context "
                    "of the coarse dofs (e.g. context.ofCoarseNodes(field, isCorner) or context.ofCoarseDofs(nCoarse)), "
                    "got {!r}".format(type(built).__name__)
                )
            P, self.coarseContext = built
        else:
            P = self.prolongation
            self.coarseContext = context.ofCoarseDofs(P.shape[1])
        self.P = sp.csr_matrix(P)
        if self.P.shape[0] != A.shape[0]:
            raise ValueError("CoarseCorrection: P has {:} rows, but the matrix {:}".format(self.P.shape[0], A.shape[0]))
        if self.coarseContext.size != self.P.shape[1]:
            raise ValueError(
                "CoarseCorrection: P has {:} columns, but its coarse context {:} dofs".format(
                    self.P.shape[1], self.coarseContext.size
                )
            )
        A_c = self._coarseMatrix(A)
        self.coarseSolve.setUp(A_c, self.coarseContext)
        self.statistics.sizes["coarse dofs"] = A_c.shape[0]

    def _update(self, A, change) -> None:
        if change.requiresRebuild or self.P is None:
            self.setUp(A, change.context)
            return
        # P stays; A_c follows the matrix, and the coarse solve decides what it keeps (an AMG its hierarchy)
        self.coarseSolve.update(self._coarseMatrix(A), change.withContext(self.coarseContext))

    def _coarseMatrix(self, A) -> SystemMatrix:
        """:math:`A_c = P^T A P`."""
        return SystemMatrix((self.P.T @ A.matrix() @ self.P).tocsr())

    def _apply(self, r: np.ndarray) -> np.ndarray:
        return self.P @ self.coarseSolve(self.P.T @ r)

    def children(self) -> tuple:
        return (("coarseSolve", self.coarseSolve),)

    def describe(self) -> Description:
        return Description(
            name="CoarseCorrection",
            formula="P {coarseSolve} Pᵀ",
            latex="P \\, {coarseSolve} \\, P^T",
            steps=(Step("r_c = Pᵀ r"), Step("z_c = A_c⁻¹ r_c,  A_c = Pᵀ A P", "coarseSolve"), Step("z = P z_c")),
        )


class Jacobi(Preconditioner):
    """Damped Jacobi, the simplest smoother: :math:`z = \\omega \\operatorname{diag}(A)^{-1} r`.

    Parameters
    ----------
    omega
        The damping. Damped Jacobi damps the fast error modes only for
        :math:`0 < \\omega < 2 / \\lambda_{max}`, :math:`\\lambda_{max}` the largest eigenvalue of
        :math:`\\operatorname{diag}(A)^{-1} A`. The default 2/3 is the textbook value for the Laplacian; quadratic
        elements have a larger :math:`\\lambda_{max}` and need a smaller damping (testfile ``TwoLevelSpelledOut``).
    """

    def __init__(self, omega: float = 2.0 / 3.0):
        super().__init__()
        self.omega = omega

    def _forget(self) -> None:
        self.inverseDiagonal = None

    def _setUp(self, A, context: SetUpContext) -> None:
        diagonal = A.matrix().diagonal()
        if np.any(diagonal == 0.0):
            raise ValueError(
                "Jacobi: {:} dof(s) have a zero diagonal entry".format(int(np.count_nonzero(diagonal == 0.0)))
            )
        self.inverseDiagonal = 1.0 / diagonal

    def _apply(self, r: np.ndarray) -> np.ndarray:
        return self.omega * self.inverseDiagonal * r

    def describe(self) -> Description:
        return Description(
            name="Jacobi",
            formula="ω diag(A)⁻¹",
            latex="\\omega \\, \\mathrm{diag}(A)^{-1}",
            parameters={"omega": round(self.omega, 4)},
        )
