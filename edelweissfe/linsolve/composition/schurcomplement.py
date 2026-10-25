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
"""Condensing hard dofs exactly: :class:`SchurComplement`, a block LU factorization around a child.

Split the dofs into the condensed ones :math:`b` (e.g. those of embedded bars with bond-slip, which make AMG stall)
and the remaining ones :math:`k`:

.. math::

    A = \\begin{bmatrix} A_{bb} & A_{bk} \\\\ A_{kb} & A_{kk} \\end{bmatrix}, \\qquad
    S = A_{kk} - A_{kb} A_{bb}^{-1} A_{bk}

One application solves with :math:`A_{bb}` exactly (a sparse LU) and preconditions the Schur complement :math:`S`
by the child, which is built on a local, sparse approximation :math:`\\tilde S`
(:mod:`~edelweissfe.linsolve.blockamg.schurcomplement` explains how it is formed):

.. math::

    y_b = A_{bb}^{-1} r_b, \\qquad z_k = M_{\\tilde S}^{-1} (r_k - A_{kb} y_b), \\qquad
    z_b = A_{bb}^{-1} (r_b - A_{bk} z_k)

With ``blockFactors="U"`` (the default) the first solve is skipped, :math:`z_k = M_{\\tilde S}^{-1} r_k`: only the
block upper triangular factor is inverted, one solve with :math:`A_{bb}` per application.
"""

import numpy as np

import edelweissfe.utils.performancetiming as performancetiming
from edelweissfe.linsolve.blockamg.schurcomplement import SchurComplementCondensation
from edelweissfe.linsolve.composition.base import (
    Description,
    Preconditioner,
    SetUpContext,
    Step,
    requirePreconditioner,
)
from edelweissfe.linsolve.composition.direct import DirectSolver
from edelweissfe.linsolve.composition.systemmatrix import SystemMatrix


def blockLU(r: np.ndarray, condensation, preconditionS, solveA_bb=None) -> np.ndarray:
    """One application of the block LU preconditioner, see the module documentation:

    .. math::

        y_b = A_{bb}^{-1} r_b, \\qquad z_k = M_{\\tilde S}^{-1} (r_k - A_{kb} y_b), \\qquad
        z_b = A_{bb}^{-1} (r_b - A_{bk} z_k)

    or, with ``blockFactors="U"``, :math:`z_k = M_{\\tilde S}^{-1} r_k` (the remaining dofs first, then the condensed
    ones follow them: the bars follow the concrete). Shared by :class:`SchurComplement` and
    :class:`~edelweissfe.linsolve.blockamg.blockamg.BlockAMGSolver`.

    Parameters
    ----------
    r
        The vector to precondition (all dofs).
    condensation
        The :class:`~edelweissfe.linsolve.blockamg.schurcomplement.SchurComplementCondensation`: the dofs
        :math:`b, k`, the blocks :math:`A_{bk}, A_{kb}` and the block factors.
    preconditionS
        :math:`M_{\\tilde S}^{-1}`, applied to a vector of the remaining dofs :math:`k`.
    solveA_bb
        :math:`A_{bb}^{-1}`, default the condensation's own factorization.
    """
    solveA_bb = condensation.solveA_bb if solveA_bb is None else solveA_bb
    A_bk, A_kb = condensation.A_bk, condensation.A_kb
    b, k = condensation.condensedDofs, condensation.remainingDofs
    r_b, r_k = r[b], r[k]
    if condensation.blockFactors == "LU":
        y_b = solveA_bb(r_b)
        z_k = preconditionS(r_k - A_kb @ y_b)
    else:
        z_k = preconditionS(r_k)
    z_b = solveA_bb(r_b - A_bk @ z_k)
    z = np.empty_like(r)
    z[b], z[k] = z_b, z_k
    return z


class SchurComplement(Preconditioner):
    """The block LU preconditioner around the Schur complement of the condensed dofs; see the module documentation.

    Every solve selects the condensed dofs again and factorizes :math:`A_{bb}`; :math:`\\tilde S` and the child are
    rebuilt when a rebuild is asked for, and when the condensed dofs change. If :math:`A_{bb}` cannot be factorized
    (a bond that lost all stiffness), that solve does without the condensation: the child is built on the whole
    :math:`A`.

    Parameters
    ----------
    condense
        The markers whose marked elements' interior nodes are condensed (e.g. an element set marker).
    preconditionS
        :math:`M_{\\tilde S}^{-1}`, built on :math:`\\tilde S`.
    solveA_bb
        The exact solve with :math:`A_{bb}` (default: SuperLU).
    approximation, dropTolerance, windowNodes, haloNodes
        How :math:`\\tilde S` is formed, see
        :class:`~edelweissfe.linsolve.blockamg.schurcomplement.SchurComplementCondensation`.
    blockFactors
        ``"U"`` or ``"LU"``.
    condensation
        A condensation to use instead of one created from ``condense`` (e.g. of given dofs, without a model). It must
        create its direct solvers through ``solveA_bb.newSolver``.
    """

    def __init__(
        self,
        condense: list = (),
        preconditionS: Preconditioner = None,
        solveA_bb: DirectSolver = None,
        approximation: str = "local",
        dropTolerance: float = 1e-4,
        windowNodes: int = 50,
        haloNodes: int = None,
        blockFactors: str = "U",
        condensation: SchurComplementCondensation = None,
    ):
        if preconditionS is None:
            raise ValueError("SchurComplement: preconditionS, the preconditioner of the Schur complement, is needed")
        requirePreconditioner("SchurComplement", "preconditionS", preconditionS)
        if solveA_bb is not None:
            requirePreconditioner("SchurComplement", "solveA_bb", solveA_bb)
        if blockFactors not in ("U", "LU"):
            raise ValueError("SchurComplement: blockFactors must be 'U' or 'LU', got {!r}".format(blockFactors))
        if condensation is None and not list(condense):
            raise ValueError(
                "SchurComplement: nothing to condense -- give the markers of the condensed elements (condense=[...]) "
                "or a condensation"
            )
        self.preconditionS = preconditionS
        self.solveA_bb = solveA_bb if solveA_bb is not None else DirectSolver("superlu")
        self.blockFactors = blockFactors
        # The condensation factorizes A_bb with the first solver it creates, which becomes solveA_bb's own (see
        # DirectSolver.newSolver); the second one is for the zones it corrects exactly.
        self.condensation = (
            condensation
            if condensation is not None
            else SchurComplementCondensation(
                list(condense),
                self.solveA_bb.newSolver,
                approximation=approximation,
                dropTolerance=dropTolerance,
                windowNodes=windowNodes,
                haloNodes=windowNodes if haloNodes is None else haloNodes,
                blockFactors=blockFactors,
            )
        )
        # a given condensation brings its own block factors
        self.blockFactors = self.condensation.blockFactors
        self.condensing = False
        self.S_tilde = None
        self._builtForDofs = None
        # the messages shown once per model
        self._messagesShown = set()

    def setModel(self, model, dofManager) -> None:
        self.condensation.setModel(model, dofManager)
        self._builtForDofs = None
        self._messagesShown = set()
        super().setModel(model, dofManager)

    def setUp(self, A, context: SetUpContext) -> None:
        self._condense(A, context, rebuild=True, change=None)

    def update(self, A, change) -> None:
        self._condense(A, change.context, rebuild=change.requiresRebuild, change=change)

    def _condense(self, A, context: SetUpContext, rebuild: bool, change) -> None:
        """Select the condensed dofs and factorize A_bb, then set up or update the child on S~ (or, without condensed
        dofs, on A)."""
        condensation = self.condensation
        try:
            condensation.update(A.matrix())
            condensing = condensation.hasCondensedDofs()
        except np.linalg.LinAlgError as error:
            # e.g. a softening zone or a bond that lost all stiffness. The condensed dofs come from the last converged
            # increment, so a cutback would meet the same block again: this solve does without.
            self._warn(context, "the condensed block A_bb cannot be factorized ({:}); solved without it".format(error))
            condensing = False
        if condensation.selectedAllDofs:
            self._warnOnce(context, "the markers select every dof; nothing is condensed")
        elif condensing and condensation.condensedDofs.size > A.shape[0] // 2:
            self._warnOnce(
                context,
                "the markers select more than half of the dofs: the solve is mostly a direct one, with the cost of the "
                "preconditioner of S̃ on top",
            )
        rebuild = rebuild or change is None
        dofs, dofsChanged = self._condensedDofsOf(condensing)
        if condensing and (rebuild or dofsChanged):
            try:
                with performancetiming.timeit("schur complement"):
                    self.S_tilde = SystemMatrix(condensation.approximateSchurComplement(A.matrix()))
            except np.linalg.LinAlgError as error:
                # e.g. a singular window block of A_bb, which factorizes as a whole: like a failed factorization, this
                # solve does without the condensation
                self._warn(
                    context, "the approximate Schur complement cannot be formed ({:}); solved without it".format(error)
                )
                condensing = False
                dofs, dofsChanged = self._condensedDofsOf(condensing)
            else:
                context.message(
                    "info",
                    "SchurComplement: {:} dofs condensed; the approximate Schur complement of the {:} remaining dofs "
                    "has {:} nonzeros".format(dofs.size, A.shape[0] - dofs.size, self.S_tilde.nnz),
                )
                if condensation.nonPositiveDiagonalEntries:
                    self._warn(
                        context,
                        "the approximate Schur complement has {:} non-positive diagonal entries: the condensed dofs do "
                        "not hold all of the softening, and the preconditioner of the remaining dofs may "
                        "struggle".format(condensation.nonPositiveDiagonalEntries),
                    )
        self.condensing = condensing

        if not condensing:
            childMatrix, childContext = A, context
        else:
            remainingFields = condensation.remainingFieldBlocks(list(context.fields))
            childMatrix = self.S_tilde
            childContext = context.restrictedTo(
                remainingFields, condensation.remainingDofs, nodesOf=condensation.remainingNodes
            )
        self._builtForDofs = dofs
        if change is None:
            self.preconditionS.setUp(childMatrix, childContext)
        else:
            childChange = change.withContext(childContext)
            if dofsChanged:
                childChange = childChange.withRebuild("condensed dofs")
            self.preconditionS.update(childMatrix, childChange)

    def _condensedDofsOf(self, condensing: bool) -> tuple:
        """The condensed dofs of this solve (``None`` without condensation), and whether they differ from those the child
        was built for."""
        dofs = self.condensation.condensedDofs if condensing else None
        changed = (dofs is None) != (self._builtForDofs is None) or (
            dofs is not None and not np.array_equal(dofs, self._builtForDofs)
        )
        return dofs, changed

    def builtMatrices(self) -> tuple:
        if not self.condensing:
            return ()
        return (("A_bb", self.condensation.A_bb), ("S̃", self.S_tilde.A))

    def _warn(self, context: SetUpContext, message: str) -> None:
        context.message("warning", "⚠ SchurComplement: " + message)

    def _warnOnce(self, context: SetUpContext, message: str) -> None:
        """A warning shown only the first time for the current model."""
        if message not in self._messagesShown:
            self._messagesShown.add(message)
            self._warn(context, message)

    def condensedDofs(self) -> "np.ndarray | None":
        return self.condensation.condensedDofs if self.condensing else None

    def __call__(self, r: np.ndarray) -> np.ndarray:
        if not self.condensing:
            return self.preconditionS(r)
        return blockLU(r, self.condensation, self.preconditionS, self.solveA_bb)

    def describe(self) -> Description:
        if self.blockFactors == "LU":
            steps = (
                Step("y_b = A_bb⁻¹ r_b", "solveA_bb"),
                Step("z_k = M_S̃⁻¹ (r_k − A_kb y_b)", "preconditionS"),
                Step("z_b = A_bb⁻¹ (r_b − A_bk z_k)", "solveA_bb"),
            )
        else:
            steps = (
                Step("z_k = M_S̃⁻¹ r_k", "preconditionS"),
                Step("z_b = A_bb⁻¹ (r_b − A_bk z_k)", "solveA_bb"),
            )
        return Description(
            name="SchurComplement",
            formula="BlockLU[{solveA_bb}, {preconditionS}]",
            latex="\\mathrm{BlockLU}\\left[{solveA_bb},\\, {preconditionS}\\right]",
            parameters={"blockFactors": self.blockFactors, "S̃": "A_kk − A_kb A_bb⁻¹ A_bk, local"},
            children=(("solveA_bb", self.solveA_bb), ("preconditionS", self.preconditionS)),
            steps=steps,
        )
