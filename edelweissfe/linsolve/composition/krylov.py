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
"""The outer Krylov solve: :class:`LGMRES` or :class:`FGMRES`, its tolerance per solve (:class:`ForcingTolerance`,
:class:`FixedTolerance`) and when it rebuilds its preconditioner (:class:`RebuildPolicy`).

A Krylov method accelerates a basic iteration :math:`x \\leftarrow x + M^{-1}(b - A x)`: from the preconditioned
residuals it builds a subspace and takes the best solution in it. LGMRES is restarted GMRES, augmented at each
restart with a few error approximations of the previous cycles. AMGCL's implementation is used, right-preconditioned,
so the residual it converges is the true one, :math:`\\lVert b - A x \\rVert / \\lVert b \\rVert`.

Inside Newton's method, a linear solve need not be accurate while Newton is still far from converged: the forcing
tolerance of Eisenstat and Walker asks for as much accuracy as the last Newton step earned. And the preconditioner
need not be rebuilt for every Newton iteration: while the matrix changes little, the old one does nearly as well, and
a rebuild costs more than the few extra iterations.
"""

from dataclasses import dataclass

import numpy as np

import edelweissfe.utils.performancetiming as performancetiming
from edelweissfe.linsolve.base import LinearSolveSummary
from edelweissfe.linsolve.composition.base import (
    Description,
    MatrixChange,
    Preconditioner,
    SetUpContext,
    Step,
    requirePreconditioner,
)

#: The least excess over the reference outer iteration count that makes a reused preconditioner stale, on top of the
#: staleness factor: at small counts a factor alone fires on noise.
STALENESS_MIN_EXCESS = 15


def eisenstatWalkerTolerance(
    residualNorm: float,
    previousResidualNorm: "float | None",
    previousEta: float,
    newIncrement: bool,
    minimum: float,
    maximum: float,
    gamma: float,
    alpha: float,
) -> float:
    """The Eisenstat--Walker "choice 2" forcing tolerance, clamped and safeguarded.

    :math:`\\eta_k = \\gamma (\\lVert b_k \\rVert / \\lVert b_{k-1} \\rVert)^\\alpha`, with the classic safeguard
    against over-solving (a large tightening step is only trusted if the previous tolerance was already small),
    clamped to ``[minimum, maximum]``. The safeguard only acts while :math:`\\gamma \\eta_{k-1}^\\alpha` exceeds 0.1,
    i.e. for a previous tolerance above about 0.26 at the classic parameters: never with a maximum below that. Falls
    back to ``maximum`` with no history to compare against (the first solve, or the one right after a new-increment
    jump -- the ratio across that jump does not reflect Newton convergence and is not meaningful). A previous norm of
    exactly zero (typically an already-converged, e.g. linear-elastic, prior iterate) makes the ratio undefined rather
    than just uninformative, and falls back the same way.
    """
    if previousResidualNorm is None or newIncrement or previousResidualNorm == 0.0:
        return maximum
    eta = gamma * (residualNorm / previousResidualNorm) ** alpha
    safeguard = gamma * previousEta**alpha
    if safeguard > 0.1:
        eta = max(eta, safeguard)
    return min(maximum, max(minimum, eta))


def outerIterationsDrifted(iterations: int, referenceIterations: int, stalenessFactor: float) -> bool:
    """Whether a solve with a reused preconditioner needed so many more outer iterations than the solve it was built
    for (``referenceIterations``) that it should be rebuilt: more than both ``stalenessFactor`` times the reference
    and the reference plus :data:`STALENESS_MIN_EXCESS`. The reference is the solve the preconditioner was built for,
    not the previous one: compared with its predecessor only, a slow drift across several solves never trips."""
    return iterations > max(stalenessFactor * referenceIterations, referenceIterations + STALENESS_MIN_EXCESS)


class FixedTolerance:
    """The same relative tolerance :math:`\\eta` for every solve.

    Parameters
    ----------
    eta
        :math:`\\lVert b - A x \\rVert \\leq \\eta \\lVert b \\rVert`.
    """

    def __init__(self, eta: float):
        self.eta = eta

    def forSolve(self, change: MatrixChange) -> float:
        """The tolerance of a solve."""
        return self.eta

    def __repr__(self) -> str:
        return "FixedTolerance({:g})".format(self.eta)


class ForcingTolerance:
    """The Eisenstat--Walker forcing tolerance per solve, see :func:`eisenstatWalkerTolerance`.

    Parameters
    ----------
    minimum, maximum
        The bounds of :math:`\\eta`.
    gamma, alpha
        The parameters of "choice 2".
    """

    def __init__(
        self, minimum: float = 1.0e-4, maximum: float = 1.0e-3, gamma: float = 0.9, alpha: float = 1.618033988749895
    ):
        self.minimum = minimum
        self.maximum = maximum
        self.gamma = gamma
        self.alpha = alpha
        self.previousEta = maximum

    def forSolve(self, change: MatrixChange) -> float:
        """The tolerance of a solve, from its right-hand side norm and the previous one."""
        eta = eisenstatWalkerTolerance(
            change.residualNorm,
            change.previousResidualNorm,
            self.previousEta,
            change.newIncrement,
            self.minimum,
            self.maximum,
            self.gamma,
            self.alpha,
        )
        self.previousEta = eta
        return eta

    def __repr__(self) -> str:
        return "ForcingTolerance(minimum={:g}, maximum={:g})".format(self.minimum, self.maximum)


@dataclass
class RebuildPolicy:
    """When a Krylov solve rebuilds its preconditioner, beyond what the matrix itself demands (the first solve, a new
    model, a resized system, a new sparsity pattern).

    Attributes
    ----------
    stalenessFactor
        A solve that needs more outer iterations than this factor times those of the solve the preconditioner was
        built for (see :func:`outerIterationsDrifted`) has it rebuilt for the next solve; so has a solve that hit the
        iteration cap.
    rebuildOnNewIncrement
        Whether a new increment of the nonlinear solver rebuilds it.
    reuse
        Whether it is reused at all: ``False`` rebuilds it for every solve.
    """

    stalenessFactor: float = 3.0
    rebuildOnNewIncrement: bool = True
    reuse: bool = True

    def decide(self, change: MatrixChange, stale: bool) -> MatrixChange:
        """``change``, with a rebuild added if this policy asks for one (in this order of reasons)."""
        if change.newIncrement and self.rebuildOnNewIncrement:
            change = change.withRebuild("increment")
        if stale:
            change = change.withRebuild("stale")
        if not self.reuse:
            change = change.withRebuild("always")
        return change


class KrylovSolver(Preconditioner):
    """An outer Krylov solve :math:`A x = b`, right-preconditioned by ``preconditioner``, with AMGCL's implementation
    of the method: what :class:`LGMRES` and :class:`FGMRES` share. As a component it is an approximate inverse of
    :math:`A` like any other, solved to the tolerance of the solve -- but not a fixed linear operator
    (:attr:`isFixedLinearOperator`).

    Parameters
    ----------
    preconditioner
        :math:`M^{-1}`.
    tolerance
        The tolerance per solve: :class:`ForcingTolerance` (default) or :class:`FixedTolerance`. May be replaced
        between solves (an offline replay pins it to the live one).
    maxIterations
        The iteration cap of a solve (across restarts). A solve that reaches it returns its iterate.
    rebuild
        When the preconditioner is rebuilt, see :class:`RebuildPolicy`.
    """

    #: AMGCL's name of the method.
    method = None

    def __init__(self, preconditioner: Preconditioner, tolerance=None, maxIterations: int = 100, rebuild=None):
        name = type(self).__name__
        requirePreconditioner(name, "preconditioner", preconditioner)
        if maxIterations < 1:
            raise ValueError("{:}: maxIterations must be at least 1, got {:}".format(name, maxIterations))
        self.preconditioner = preconditioner
        self._checkPreconditioner()
        self.tolerance = tolerance if tolerance is not None else ForcingTolerance()
        self.maxIterations = maxIterations
        self.rebuild = rebuild if rebuild is not None else RebuildPolicy()
        self.A = None
        self.eta = None
        #: Of the last solve: its outer iterations, its true relative residual, whether it stopped at the cap, and
        #: whether the preconditioner was rebuilt for it (and why).
        self.lastIterations = None
        self.lastResidual = None
        self.lastCapHit = False
        self.lastRebuildReason = None
        self._krylov = None
        # the outer iterations of the solve the preconditioner was built for, and whether the last solve made it stale
        self._referenceIterations = None
        self._stale = False

    @property
    def isFixedLinearOperator(self) -> bool:
        return False

    def _checkPreconditioner(self) -> None:
        """Raise if the method cannot take this preconditioner: at construction, and again after the set-up, when a
        combination has made its corrections."""

    def _parameters(self) -> dict:
        """AMGCL's parameters of the method (its restart length, ...)."""
        raise NotImplementedError

    def setUp(self, A, context: SetUpContext) -> None:
        self.update(A, MatrixChange(context, rebuildReason="first"))

    def update(self, A, change: MatrixChange) -> None:
        change = self.rebuild.decide(change, self._stale)
        self.A = A
        self.eta = self.tolerance.forSolve(change)
        self.lastRebuildReason = change.rebuildReason
        if change.requiresRebuild:
            with performancetiming.timeit("hierarchy build"):
                self.preconditioner.update(A, change)
            self._stale = False
            self._checkPreconditioner()
        else:
            self.preconditioner.update(A, change)

    def _solverFor(self, n: int):
        """AMGCL's solver for size ``n``: its scratch vectors are allocated for one size."""
        if self._krylov is None or self._krylov.n != n:
            from edelweissfe.linsolve.amgcl.amgcl import PyAMGCLKrylovSolver

            self._krylov = PyAMGCLKrylovSolver(n, self.method, self._parameters())
        return self._krylov

    def __call__(self, b: np.ndarray) -> np.ndarray:
        krylov = self._solverFor(self.A.shape[0])
        with performancetiming.timeit("outer Krylov solve"):
            x = krylov.solve(
                self.A.krylovOperator(), b, self.preconditioner, self.eta, self.maxIterations, onUnscaledOperator=True
            )
        self.lastIterations = krylov.lastIterations
        self.lastResidual = krylov.lastError
        self.lastCapHit = self.lastIterations >= self.maxIterations and not krylov.lastError < self.eta
        self._judgeStaleness()
        return x

    def finishSolve(self) -> None:
        # the matrix (and the operator formed from it) belongs to this solve
        self.A = None
        super().finishSolve()

    def _judgeStaleness(self) -> None:
        """Whether the preconditioner should be rebuilt before the next solve (see :class:`RebuildPolicy`)."""
        if self.lastRebuildReason is not None or self._referenceIterations is None:
            self._referenceIterations = self.lastIterations
            drifted = False
        else:
            drifted = outerIterationsDrifted(
                self.lastIterations, self._referenceIterations, self.rebuild.stalenessFactor
            )
        self._stale = drifted or self.lastCapHit

    def solveSummary(self) -> LinearSolveSummary:
        return LinearSolveSummary(
            iters=self.lastIterations,
            residual=self.lastResidual,
            residualMet=self.lastResidual <= self.eta,
            retries=0,
            detailLines=() if self.lastRebuildReason is None else ("rebuilt({:})".format(self.lastRebuildReason),),
        )

    def describe(self) -> Description:
        name = type(self).__name__
        return Description(
            name=name,
            formula=name + "(A, {preconditioner})⁻¹",
            latex="\\mathrm{" + name + "}\\left(A,\\, {preconditioner}\\right)^{-1}",
            parameters={"tolerance": repr(self.tolerance), "maxIterations": self.maxIterations, **self._parameters()},
            children=(("preconditioner", self.preconditioner),),
            steps=self._steps(),
        )

    def _steps(self) -> tuple:
        raise NotImplementedError


class LGMRES(KrylovSolver):
    """LGMRES: restarted GMRES, augmented at each restart with ``K`` error approximations of the previous cycles,
    AMGCL's implementation. Right-preconditioned: :math:`x = M^{-1} y` with :math:`y` from the Krylov space of
    :math:`A M^{-1}`, which is a Krylov space only for a fixed linear :math:`M^{-1}` -- so LGMRES refuses a variable
    preconditioner (a Krylov solve inside), for which :class:`FGMRES` is made.

    Parameters
    ----------
    preconditioner, tolerance, maxIterations, rebuild
        See :class:`KrylovSolver`.
    M, K
        The restart length and the number of augmentation vectors.
    """

    method = "lgmres"

    def __init__(
        self,
        preconditioner: Preconditioner,
        tolerance=None,
        maxIterations: int = 100,
        M: int = 50,
        K: int = 3,
        rebuild: RebuildPolicy = None,
    ):
        self.M = M
        self.K = K
        super().__init__(preconditioner, tolerance, maxIterations, rebuild)

    def _checkPreconditioner(self) -> None:
        if not self.preconditioner.isFixedLinearOperator:
            raise ValueError(
                "LGMRES: its preconditioner {:} is not a fixed linear operator (it contains a Krylov solve), which "
                "LGMRES needs: use FGMRES (flexible GMRES) for a variable preconditioner".format(
                    self.preconditioner.describe().name
                )
            )

    def _parameters(self) -> dict:
        return {"M": self.M, "K": self.K}

    def _steps(self) -> tuple:
        return (
            Step("vₖ₊₁ ⟂ span(A M⁻¹ v₁ … A M⁻¹ vₖ)  (Arnoldi, augmented at restarts)"),
            Step("M⁻¹ vₖ, once per iteration", "preconditioner"),
            Step("x = M⁻¹ Vₘ yₘ,  yₘ minimizes ‖b − A x‖"),
        )


class FGMRES(KrylovSolver):
    """Flexible GMRES (Saad 1993), AMGCL's implementation: GMRES that keeps each preconditioned vector
    :math:`z_k = M_k^{-1} v_k` and takes the solution from their span, :math:`x = Z_m y_m`, so the preconditioner may
    change from one iteration to the next -- e.g. an inner Krylov solve to a loose tolerance. It stores twice the
    vectors of GMRES (:math:`V` and :math:`Z`).

    Parameters
    ----------
    preconditioner, tolerance, maxIterations, rebuild
        See :class:`KrylovSolver`.
    M
        The restart length.
    """

    method = "fgmres"

    def __init__(
        self,
        preconditioner: Preconditioner,
        tolerance=None,
        maxIterations: int = 100,
        M: int = 30,
        rebuild: RebuildPolicy = None,
    ):
        self.M = M
        super().__init__(preconditioner, tolerance, maxIterations, rebuild)

    def _parameters(self) -> dict:
        return {"M": self.M}

    def _steps(self) -> tuple:
        return (
            Step("zₖ = Mₖ⁻¹ vₖ, once per iteration (Mₖ may change)", "preconditioner"),
            Step("vₖ₊₁ ⟂ span(A z₁ … A zₖ)  (Arnoldi)"),
            Step("x = Zₘ yₘ,  yₘ minimizes ‖b − A x‖"),
        )
