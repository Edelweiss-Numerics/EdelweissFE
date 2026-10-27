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
        # written as "not (valid)" so that NaN is refused too
        if not 0.0 < eta < 1.0:
            raise ValueError("FixedTolerance: needs 0 < eta < 1, got {!r}".format(eta))
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
        if not 0.0 < minimum <= maximum < 1.0:
            raise ValueError(
                "ForcingTolerance: needs 0 < minimum <= maximum < 1, got {!r}, {!r}".format(minimum, maximum)
            )
        # the ranges of Eisenstat and Walker's choice 2
        if not 0.0 < gamma <= 1.0 or not 1.0 < alpha <= 2.0:
            raise ValueError(
                "ForcingTolerance: needs 0 < gamma <= 1 and 1 < alpha <= 2, got {!r}, {!r}".format(gamma, alpha)
            )
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

    def __post_init__(self):
        if not self.stalenessFactor >= 1.0:
            raise ValueError("RebuildPolicy: stalenessFactor must be at least 1, got {!r}".format(self.stalenessFactor))

    def decide(self, change: MatrixChange, stale: bool) -> MatrixChange:
        """``change``, with a rebuild added if this policy asks for one (in this order of reasons)."""
        if change.newIncrement and self.rebuildOnNewIncrement:
            change = change.withRebuild("increment")
        if stale:
            change = change.withRebuild("stale")
        if not self.reuse:
            change = change.withRebuild("always")
        return change


class GapCompensation:
    """For a Krylov solve on a diagonally scaled matrix :math:`\\hat A = D A D` (``DiagonalScaling(LGMRES(...))``):
    ask the first pass for :math:`s \\, \\eta / g` instead of :math:`\\eta`, with :math:`g` the gap between the true
    and the scaled relative residual of the same iterate, learned from the previous solves, so that the pass lands on
    the true-residual target without a continuation; a continuation then tightens by the measured gap instead of
    the fixed factor 0.01.

    The Krylov method converges the scaled residual :math:`\\lVert D (b - A x) \\rVert`, while the solve is accepted
    on the true one, :math:`\\lVert b - A x \\rVert / \\lVert b \\rVert`; their ratio was measured from close to one
    (gradient-enhanced damage) to far larger (contact with multi-point constraints). Measured to save continuations
    and iterations per increment, but the uncompensated scaled solve over-solves, and Newton relies on that: per
    unit of simulated time the saving was largely given back. Hence it is opt-in.

    Parameters
    ----------
    safetyFactor
        :math:`s`: smaller is more accurate (closer to the uncompensated Newton path).
    maxFactor
        The upper bound of :math:`g`: one non-converged solve with an outsized gap would otherwise compound, through
        the smoothing, into an ever tighter, unmeetable tolerance.
    """

    def __init__(self, safetyFactor: float = 0.3, maxFactor: float = 1e3):
        if not safetyFactor > 0.0:
            raise ValueError("GapCompensation: safetyFactor must be positive, got {!r}".format(safetyFactor))
        if not maxFactor >= 1.0:
            raise ValueError("GapCompensation: maxFactor must be at least 1 (the gap is floored at 1)")
        self.safetyFactor = safetyFactor
        self.maxFactor = maxFactor
        #: :math:`g`, smoothed over the solves (an exponential moving average with weight 1/2).
        self.gap = 1.0

    def firstPassTolerance(self, eta: float) -> float:
        """The tolerance of the first pass: :math:`s \\, \\eta / g`."""
        return max(self.safetyFactor * eta / max(self.gap, 1.0), 1e-14)

    def learn(self, trueResidual: float, scaledResidual: float) -> None:
        """Take the gap of a solve's first pass into :math:`g`."""
        measuredGap = trueResidual / max(scaledResidual, 1e-300)
        if np.isfinite(measuredGap) and measuredGap > 0.0:
            self.gap = min(max(0.5 * self.gap + 0.5 * measuredGap, 1.0), self.maxFactor)

    def continuationTolerance(self, eta: float, previousTolerance: float, trueResidual: float, scaledResidual):
        """The tolerance of a continuation: aimed at :math:`s \\, \\eta / g` of the current gap, and at least half the
        previous one, so that the continuations end."""
        gap = trueResidual / max(scaledResidual, 1e-300)
        return max(min(self.safetyFactor * eta / max(gap, 1.0), 0.5 * previousTolerance), 1e-14)

    def __repr__(self) -> str:
        return "GapCompensation(safetyFactor={:g}, maxFactor={:g})".format(self.safetyFactor, self.maxFactor)


@dataclass(frozen=True)
class SolveRecord:
    """What a Krylov solve did, for the journal, the convergence table and a degradation dump.

    Attributes
    ----------
    iterations
        Its outer iterations, continuations included.
    residual
        Its true relative residual :math:`\\lVert b - A x \\rVert / \\lVert b \\rVert`, in the physical variables of
        its (sub-)problem.
    eta
        Its tolerance.
    capHit
        Whether a pass stopped at the iteration cap short of its tolerance.
    continuations
        How many warm-restarted passes followed the first (only on a diagonally scaled matrix).
    rebuildReason
        Why the preconditioner was rebuilt for it, or ``None`` (reused).
    scaledSystem
        Whether it ran on a diagonally scaled matrix (its own residual then is not the true one).
    """

    iterations: int
    residual: float
    eta: float
    capHit: bool
    continuations: int
    rebuildReason: "str | None"
    scaledSystem: bool

    @property
    def converged(self) -> bool:
        """Whether the true residual met the tolerance (written so that NaN is a miss)."""
        return self.residual <= self.eta


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
    continuations
        On a diagonally scaled matrix (``DiagonalScaling(LGMRES(...))``) the method converges the scaled residual,
        while the solve is accepted on the true one, which can be larger (stiff rows are weighted down by
        :math:`D`): a pass that misses :math:`\\eta` on the true residual is continued from its iterate, at most this
        many times, with a tighter tolerance (0.01 times the previous, or see :class:`GapCompensation`). On an
        unscaled matrix the residual the method converges is the true one, and a pass that misses :math:`\\eta` has
        hit the iteration cap, which a warm restart would only extend: no continuations there.
    gapCompensation
        A :class:`GapCompensation`, or ``None`` (the default): only on a diagonally scaled matrix.
    """

    #: AMGCL's name of the method.
    method = None

    def __init__(
        self,
        preconditioner: Preconditioner,
        tolerance=None,
        maxIterations: int = 100,
        rebuild=None,
        continuations: int = 2,
        gapCompensation: GapCompensation = None,
    ):
        name = type(self).__name__
        requirePreconditioner(name, "preconditioner", preconditioner)
        super().__init__()
        if maxIterations < 1:
            raise ValueError("{:}: maxIterations must be at least 1, got {:}".format(name, maxIterations))
        if continuations < 0:
            raise ValueError("{:}: continuations must not be negative, got {:}".format(name, continuations))
        self.preconditioner = preconditioner
        self._checkPreconditioner()
        self.tolerance = tolerance if tolerance is not None else ForcingTolerance()
        self.maxIterations = maxIterations
        self.rebuild = rebuild if rebuild is not None else RebuildPolicy()
        self.continuations = continuations
        self.gapCompensation = gapCompensation
        self.A = None
        self.eta = None
        self._context = None
        #: Of the last solve: its outer iterations, its true relative residual, whether it stopped at the cap, and
        #: whether the preconditioner was rebuilt for it (and why).
        self.lastIterations = None
        self.lastResidual = None
        self.lastCapHit = False
        self.lastContinuations = 0
        self.lastScaledSystem = False
        self.lastRebuildReason = None
        self._krylov = None
        # the outer iterations of the solve the preconditioner was built for, and whether the last solve made it stale
        self._referenceIterations = None
        self._stale = False

    @property
    def containsKrylovSolver(self) -> bool:
        return True

    @property
    def isFixedLinearOperator(self) -> bool:
        return False

    def _checkPreconditioner(self) -> None:
        """Raise if the method cannot take this preconditioner: at construction, and again after the set-up, when a
        combination has made its corrections."""

    def _parameters(self) -> dict:
        """AMGCL's parameters of the method (its restart length, ...), in AMGCL's names."""
        raise NotImplementedError

    def _setUp(self, A, context: SetUpContext) -> None:
        self._update(A, MatrixChange(context, rebuildReason="first"))

    def withRebuildReasons(self, change: MatrixChange) -> MatrixChange:
        return super().withRebuildReasons(self.rebuild.decide(change, self._stale))

    def _update(self, A, change: MatrixChange) -> None:
        change = self.rebuild.decide(change, self._stale)
        self.A = A
        self._context = change.context
        # before the preconditioner forms the field split of A: that one pass then keeps A as this solve's operator
        A.useAsKrylovOperator()
        self.eta = self.tolerance.forSolve(change)
        self.lastRebuildReason = change.rebuildReason
        if change.requiresRebuild:
            with performancetiming.timeit("hierarchy build"):
                # all of the preconditioner is built anew: the old one goes first, never two at once
                self.preconditioner.forget()
                self.preconditioner.update(A, change)
            self._stale = False
            self._checkPreconditioner()
        else:
            self.preconditioner.update(A, change)

    def _solverFor(self, n: int):
        """AMGCL's solver for size ``n``: its scratch vectors are allocated for one size."""
        if self._krylov is None or self._krylov.n != n:
            from edelweissfe.linsolve.amgcl.amgcl import PyAMGCLKrylovSolver

            # the scratch vectors for the old size go first
            self._krylov = None
            self._krylov = PyAMGCLKrylovSolver(n, self.method, self._parameters())
        return self._krylov

    def _apply(self, b: np.ndarray) -> np.ndarray:
        """The solve: a first pass, and on a diagonally scaled matrix the continuations while the true residual misses
        :math:`\\eta` (see ``continuations``)."""
        # on D A D (DiagonalScaling(LGMRES(...))): the method converges the scaled residual, not the true one
        onScaledSystem = self.A.scaling is not None
        eta = self.eta
        compensating = self.gapCompensation is not None and onScaledSystem
        passTolerance = self.gapCompensation.firstPassTolerance(eta) if compensating else eta
        with performancetiming.timeit("outer Krylov solve"):
            x, iterations, capHit = self._pass(b, passTolerance)
            trueResidual, scaledResidual = self._residuals(b, x, onScaledSystem)
        if compensating:
            self.gapCompensation.learn(trueResidual, scaledResidual)
        continuations = 0
        with performancetiming.timeit("true-residual continuations"):
            if compensating:
                passTolerance = min(eta, passTolerance)
            while onScaledSystem and trueResidual > eta and continuations < self.continuations:
                continuations += 1
                if compensating:
                    passTolerance = self.gapCompensation.continuationTolerance(
                        eta, passTolerance, trueResidual, scaledResidual
                    )
                else:
                    passTolerance *= 0.01
                x, passIterations, passCapHit = self._pass(b, passTolerance, x0=x)
                trueResidual, scaledResidual = self._residuals(b, x, onScaledSystem)
                iterations += passIterations
                capHit = capHit or passCapHit
                self._context.message(
                    "debug",
                    "{:}:   continuation {:}/{:} eta={:.1e} +{:}it res={:.1e}".format(
                        type(self).__name__,
                        continuations,
                        self.continuations,
                        passTolerance,
                        passIterations,
                        trueResidual,
                    ),
                )
        self.lastIterations = iterations
        self.lastResidual = trueResidual
        self.lastCapHit = capHit
        self.lastContinuations = continuations
        self.lastScaledSystem = onScaledSystem
        self._judgeStaleness()
        return x

    def _pass(self, b: np.ndarray, tolerance: float, x0: np.ndarray = None) -> tuple:
        """One pass of the method from ``x0`` (zero if ``None``): its iterate, its iterations, and whether it stopped at
        the iteration cap short of ``tolerance`` (a pass may converge on exactly its last allowed iteration)."""
        krylov = self._solverFor(self.A.shape[0])
        x = krylov.solve(
            self.A.krylovOperator(),
            b,
            self.preconditioner,
            tolerance,
            self.maxIterations,
            x0=x0,
            onUnscaledOperator=self.A.scaling is None,
        )
        capHit = krylov.lastIterations >= self.maxIterations and not krylov.lastError < tolerance
        return x, krylov.lastIterations, capHit

    def _residuals(self, b: np.ndarray, x: np.ndarray, onScaledSystem: bool) -> tuple:
        """The true relative residual of the iterate ``x`` -- in the physical variables of this (sub-)problem, the one
        the solve is accepted on -- and on a scaled matrix its scaled one (NaN otherwise). On an unscaled matrix the
        method recomputed :math:`b - A x` at its exit, and its residual is the true one."""
        if not onScaledSystem:
            return self._krylov.lastError, float("nan")
        # b - A x = D^-1 (b̂ - Â x̂) exactly: the true residual from the scaled one, without a product with A
        D = self.A.scaling
        r_scaled = self.A.krylovOperator().residual(b, x, unscaled=False)
        return (
            np.linalg.norm(r_scaled / D) / max(np.linalg.norm(b / D), 1e-300),
            np.linalg.norm(r_scaled) / max(np.linalg.norm(b), 1e-300),
        )

    def finishSolve(self) -> None:
        # the matrix (and the operator formed from it) belongs to this solve
        self.A = None
        self._context = None
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

    def children(self) -> tuple:
        return (("preconditioner", self.preconditioner),)

    def solveRecord(self) -> SolveRecord:
        return SolveRecord(
            iterations=self.lastIterations,
            residual=self.lastResidual,
            eta=self.eta,
            capHit=self.lastCapHit,
            continuations=self.lastContinuations,
            rebuildReason=self.lastRebuildReason,
            scaledSystem=self.lastScaledSystem,
        )

    def describe(self) -> Description:
        name = type(self).__name__
        return Description(
            name=name,
            formula=name + "(A, {preconditioner})⁻¹",
            latex="\\mathrm{" + name + "}\\left(A,\\, {preconditioner}\\right)^{-1}",
            parameters=self._describedParameters(),
            steps=self._steps(),
        )

    def _describedParameters(self) -> dict:
        """The parameters the views show, in the names of this class (a method adds its own)."""
        parameters = {"tolerance": repr(self.tolerance), "maxIterations": self.maxIterations}
        if self.gapCompensation is not None:
            parameters["gapCompensation"] = repr(self.gapCompensation)
        return parameters

    def _steps(self) -> tuple:
        raise NotImplementedError


class LGMRES(KrylovSolver):
    """LGMRES (Baker, Jessup and Manteuffel 2005): GMRES restarted every ``restart`` iterations, each cycle augmented
    with ``augmentationVectors`` error approximations of the previous cycles, AMGCL's implementation.
    Right-preconditioned: :math:`x = M^{-1} y` with :math:`y` from the Krylov space of :math:`A M^{-1}`, which is a Krylov space only for a fixed linear :math:`M^{-1}` -- so LGMRES refuses a variable
    preconditioner (a Krylov solve inside), for which :class:`FGMRES` is made.

    Parameters
    ----------
    preconditioner, tolerance, maxIterations, rebuild, continuations, gapCompensation
        See :class:`KrylovSolver`.
    restart
        The iterations of a cycle, :math:`m` (the Krylov vectors kept before a restart).
    augmentationVectors
        The error approximations of the previous cycles each cycle is augmented with, :math:`k`.
    """

    method = "lgmres"

    def __init__(
        self,
        preconditioner: Preconditioner,
        tolerance=None,
        maxIterations: int = 100,
        restart: int = 50,
        augmentationVectors: int = 3,
        rebuild: RebuildPolicy = None,
        continuations: int = 2,
        gapCompensation: GapCompensation = None,
    ):
        if restart < 1 or augmentationVectors < 0:
            raise ValueError(
                "LGMRES: needs restart >= 1 and augmentationVectors >= 0, got {:} and {:}".format(
                    restart, augmentationVectors
                )
            )
        self.restart = restart
        self.augmentationVectors = augmentationVectors
        super().__init__(preconditioner, tolerance, maxIterations, rebuild, continuations, gapCompensation)

    def _checkPreconditioner(self) -> None:
        if not self.preconditioner.isFixedLinearOperator:
            raise ValueError(
                "LGMRES: its preconditioner {:} is not a fixed linear operator (it contains a Krylov solve), which "
                "LGMRES needs: use FGMRES (flexible GMRES) for a variable preconditioner".format(
                    self.preconditioner.describe().name
                )
            )

    def _parameters(self) -> dict:
        # AMGCL's names: M the restart length, K the augmentation vectors
        return {"M": self.restart, "K": self.augmentationVectors}

    def _describedParameters(self) -> dict:
        return {
            **super()._describedParameters(),
            "restart": self.restart,
            "augmentationVectors": self.augmentationVectors,
        }

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
    preconditioner, tolerance, maxIterations, rebuild, continuations, gapCompensation
        See :class:`KrylovSolver`.
    restart
        The iterations of a cycle (the vectors kept before a restart).
    """

    method = "fgmres"

    def __init__(
        self,
        preconditioner: Preconditioner,
        tolerance=None,
        maxIterations: int = 100,
        restart: int = 30,
        rebuild: RebuildPolicy = None,
        continuations: int = 2,
        gapCompensation: GapCompensation = None,
    ):
        if restart < 1:
            raise ValueError("FGMRES: needs restart >= 1, got {:}".format(restart))
        self.restart = restart
        super().__init__(preconditioner, tolerance, maxIterations, rebuild, continuations, gapCompensation)

    def _parameters(self) -> dict:
        # AMGCL's name of the restart length
        return {"M": self.restart}

    def _describedParameters(self) -> dict:
        return {**super()._describedParameters(), "restart": self.restart}

    def _steps(self) -> tuple:
        return (
            Step("zₖ = Mₖ⁻¹ vₖ, once per iteration (Mₖ may change)", "preconditioner"),
            Step("vₖ₊₁ ⟂ span(A z₁ … A zₖ)  (Arnoldi)"),
            Step("x = Zₘ yₘ,  yₘ minimizes ‖b − A x‖"),
        )
