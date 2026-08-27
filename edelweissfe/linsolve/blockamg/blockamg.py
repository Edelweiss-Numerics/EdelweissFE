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

"""A field-split block-AMG linear solver for large coupled multi-field systems.

Why this exists
---------------

On the large coupled fracture models (displacement + gradient-enhanced damage, penalty contact,
adaptive refinement), a direct factorization dominates the run and -- more importantly -- hits a
memory wall past ~1M dof, because its fill-in grows superlinearly. Algebraic multigrid has O(n)
memory and is the route to those sizes, but applied *monolithically* to the coupled system it is
ineffective: a single AMG hierarchy cannot represent the disparate physics and scales of the fields
at once -- measured directly on a real coupled system, a monolithic hierarchy's residual reduction
plateaus around 0.2 rather than converging, regardless of how the near null-space is chosen; the
missing ingredient is the field-block structure itself, not a better single hierarchy.

The remedy, following Alkmim et al. (IJNME 2026), is a *block* preconditioner: an AMG hierarchy per
field, combined by a block Gauss-Seidel sweep, used to precondition an outer GMRES over the full
coupled system. Each field's operator (elasticity for a displacement field, a Helmholtz-like operator
for a damage field) is individually AMG-friendly even though their monolithic coupling is not.

The field structure -- which DOFs belong to which field, and each field's nodal dimension -- is not
carried by the matrix; it is pushed in from the DofManager by the nonlinear solver (via
:class:`~edelweissfe.linsolve.base.LinearSolver`), so nothing about the block layout
has to be specified by hand.

What it does per solve
----------------------

#. **Equilibrate.** Symmetric diagonal (Jacobi) scaling :math:`\\hat A = D^{-1/2} A D^{-1/2}` removes
   the large dynamic range (Dirichlet penalties + stiffness) that otherwise wrecks AMG's
   strength-of-connection. The solve is done on :math:`\\hat A` and unscaled at the end.
#. **Split** :math:`\\hat A` into the field diagonal blocks and their couplings, from the field ranges.
#. **Build one AMG hierarchy per field** (AMGCL, built once per solve via ``build`` and applied many
   times via ``applyPreconditioner`` -- the pattern churns between Newton iterations, so the hierarchy
   cannot be reused *across* solves, but it is reused across the outer GMRES iterations *within* a
   solve). A vector field (nodal dimension > 1, e.g. displacement) is given its full rigid-body
   near-null-space -- 3 translations plus 3 infinitesimal rotations, the standard basis for 3D
   elasticity AMG, since both are directions in which the discretized operator has essentially zero
   stiffness -- when node coordinates are available, translations alone otherwise; a scalar field
   takes the default constant.
#. **Precondition GMRES** with a block Gauss-Seidel sweep over the fields, each field's correction
   coming from one AMG V-cycle on its block, the couplings folded in between fields.

This is a *feasibility-grade* solver: on the reference model AMGCL's smoothed aggregation converges
but not tightly on the (non-symmetric, contact + tie condensed) displacement block, so the outer
GMRES needs O(100) iterations. That is acceptable where the point is to fit in memory at sizes a
direct solver cannot reach; the iteration count would likely come down further with a
nonsymmetric-aware AMG library (e.g. Trilinos/MueLu) built specifically to precondition
non-symmetric operators, at the cost of an additional heavy external dependency -- AMGCL's own
symmetric-aggregation-based hierarchy already reaches a working feasibility point without one.
"""

import collections
import json
import os
import time

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import LinearOperator, gmres

import edelweissfe.utils.performancetiming as performancetiming
from edelweissfe.linsolve.base import FieldBlock, LinearSolver
from edelweissfe.linsolve.nullspace import rigidBodyNullspace, translationNullspace

# Ordered low-to-high; index comparison decides whether a message at a given level should print.
# "warning": only abnormal conditions (excessive outer iterations, unmet true-residual tolerance,
# non-convergence) -- the default, so a normal run stays quiet. "info": one compact line per solve.
# "debug": full detail, including every true-residual continuation attempt. Independent of, and does
# not affect, the Journal instance's own message-level suppression (see _log()) -- this gate decides
# whether blockamg attempts to log at all; Journal's own level decides whether the attempt is shown.
_VERBOSITY_LEVELS = ("silent", "warning", "info", "debug")
_JOURNAL_LEVEL = {"warning": 0, "info": 1, "debug": 2}
_IDENTIFICATION = "BlockAMGSolver"

# "backendPrecision" and "backendBlockSize" are not AMGCL parameters -- they select the AMGCL
# wrapper's own backend value type ("double" (default) or "float" for backendPrecision; 1 (default,
# scalar), 2, or 3 for a block-valued backend operating on B x B nodal blocks for backendBlockSize).
# __call__ pops both out of whichever precond dict applies (default or a fieldPreconds override)
# before forwarding the rest verbatim as the AMGCL JSON parameter tree.
#
# backendPrecision defaults to "double": measured on all 9 dumped ords of a real reference model,
# "float" inflated outer GMRES iterations by up to 30% and only won on wall-clock about half the
# time, netting only ~3% aggregate rather than the roughly halved memory/bandwidth traffic a
# single-precision backend should in principle buy -- plausibly because the Chebyshev smoother's own
# power-iteration spectral-radius estimate is itself sensitive to reduced precision, losing accuracy
# in float32 in a way that degrades smoother quality enough to eat most of the bandwidth saving.
#
# backendBlockSize defaults to 1 (scalar): kept opt-in via fieldPreconds, not the default, since a
# block-valued backend has not been validated to win inside this block-Gauss-Seidel preconditioner
# (a faster standalone solver is not automatically a better single-cycle preconditioner component). A
# vector field of nodal dimension d maps to backendBlockSize: d (3 for a 3D displacement field, 2 for
# a 2D one). Setting it > 1 skips set_nullspace (see __call__) -- AMGCL's near-null-space path is
# unimplemented for block value types.
#
# Both are available as an opt-in via fieldPreconds, e.g.
# {"displacement": {**_DEFAULT_VECTOR_PRECOND, "backendPrecision": "float"}} or
# {"displacement": {**_DEFAULT_VECTOR_PRECOND, "backendBlockSize": 3}}.
# power_iters=300 (was 50): AMGCL's Chebyshev smoother estimates the operator's spectral radius via
# power iteration whose start vector is seeded per-OpenMP-thread by thread id
# (amgcl/backend/builtin.hpp) -- deterministic for a given thread count, but a different vector at a
# different thread count (a parallel power iteration typically partitions the random starting vector
# across worker threads, so the number of independent per-thread random streams -- and hence the
# effective starting vector -- changes with thread count even though the operator itself does not).
# The estimate itself (and hence the smoother's damping quality) was therefore thread-count dependent.
# At 50 iterations the estimate can be badly under-converged on a hard operator at some thread counts
# (measured: 1460 vs 43 outer iterations, same matrix, only OMP_NUM_THREADS differing, on a solve
# where the true spectral radius -- verified independently via ARPACK -- barely moves between Newton
# iterations). 300 iterations converge the estimate close enough that it stops depending on thread
# count; measured 2.68x faster in aggregate on 10 captured degraded systems at production's real 16
# threads (811s -> 302s), with no observed downside on systems that were already fine.
_DEFAULT_VECTOR_PRECOND = {
    "backendPrecision": "double",
    "backendBlockSize": 1,
    "coarsening": {"type": "smoothed_aggregation", "aggr": {"eps_strong": 0.01}},
    "relax": {"type": "chebyshev", "degree": 5, "power_iters": 300, "lower": 0.01},
    "npre": 1,
    "npost": 1,
}
_DEFAULT_SCALAR_PRECOND = {
    "backendPrecision": "double",
    "backendBlockSize": 1,
    "coarsening": {"type": "smoothed_aggregation"},
    "relax": {"type": "chebyshev"},
}


class BlockAMGSolver(LinearSolver):
    """Field-split block-AMG preconditioned GMRES. Callable as ``(A, b) -> x``.

    The block structure is not configured here -- it is derived by the base class from the model and
    DOF manager the nonlinear solver hands over via
    :meth:`~edelweissfe.linsolve.base.LinearSolver.setModel`. A field's near null-space is decided
    from its nodal dimension: a vector field (dimension > 1) gets its full rigid-body basis
    (translations and rotations) when node coordinates are available, translations alone otherwise;
    a scalar field takes the default constant.

    Stateful across calls in two independent ways (both driven by ``||b||`` alone -- this solver, like
    every other registered linear solver, sees only ``(A, b)`` per call and
    reconstructs Newton-loop state from residual jumps rather than being told about them):

    #. **Adaptive outer tolerance** (Eisenstat--Walker forcing, "choice 2"), the default. Most Newton
       iterates do not need the linear solve tight, and loosening the requested tolerance for them cuts
       outer iterations substantially without changing the Newton path -- provided GMRES's own
       *preconditioned*-residual stopping check is not allowed to silently let the *true* residual run
       looser than the requested ``eta`` (a real risk with an imperfect preconditioner, see the
       true-residual enforcement below); once that gap is closed, adaptive forcing reproduces the exact
       same Newton trajectory as a tight fixed tolerance, just faster. Pass ``outerTol=<a float>`` to
       pin a fixed tolerance instead if a specific model needs that guarantee.
    #. **Per-field AMG hierarchy reuse across Newton iterations**, on by default. Building a hierarchy
       is a large, avoidable fraction of a solve when the Jacobian has moved only a little since the
       last one. The outer GMRES always operates on the current, fresh matrix; only the block
       preconditioner ``M`` may be stale, so the *converged solution* is unaffected regardless of ``M``
       -- a stale ``M`` only costs a few extra outer iterations at the *same* requested tolerance, so
       (unlike (a)) it cannot by itself change the Newton path. Refreshed on a residual jump (new
       increment / cutback), on a field-structure change (e.g. AMR), on any sparsity-pattern change
       (a model with contact and/or tie constraints typically churns its sparsity pattern on every
       single Newton iteration as the active contact/tie set evolves, so in practice this keeps the
       hierarchy fresh every call on such a model -- a safe no-op there, not a win; a model without
       that churn may see a real one), or when the previous solve's outer count grew past
       ``hierarchyStalenessFactor`` times the one before it.

    Every solve also enforces the requested tolerance on the *true* (unpreconditioned) residual, not
    just GMRES's own preconditioned stopping check -- see ``trueResidualMaxContinuations``. GMRES's
    convergence test here is on the preconditioned residual (``callback_type="pr_norm"``), which can be
    considerably smaller than the true one under an imperfect preconditioner (this one, by design), so
    "converged" could otherwise mean a true residual well above ``eta`` (measured on a real reference
    model: a 1.6e-2 true residual when 1e-4 was requested).

    Parameters
    ----------
    outerTol
        A fixed outer GMRES relative tolerance, overriding Eisenstat--Walker forcing. ``None`` (the
        default) uses adaptive forcing instead -- see ``etaMin``/``etaMax`` below.
    outerRestart, outerMaxiter
        The outer GMRES restart length and maximum restart cycles.
    outerSolver
        ``"amgcl_lgmres"`` (default) uses AMGCL's own native ``amgcl::solver::lgmres`` at both
        outer-solve call sites (the main solve and the true-residual continuation retry below) -- see
        ``edelweissfe/linsolve/amgcl/amgcl-wrapper.hpp``'s ``LGMRESOuterSolverT`` for why: SciPy's own
        GMRES orchestration (Arnoldi/Gram-Schmidt/restart bookkeeping, all run from Python) was found
        to be the single largest unaddressed cost bucket on the outer loop of a real reference model
        (roughly 38% of solve wall-clock). Live-gated at multiple thread counts: trajectory-identical
        to the SciPy default, no NaN (once ``lgmresAlwaysReset`` -- see below -- was fixed), and
        increasingly faster than SciPy as thread count grows -- SciPy's own orchestration anti-scales
        past one NUMA node on multi-socket hardware, AMGCL's native implementation does not. ``"scipy"``
        uses ``scipy.sparse.linalg.gmres`` instead, kept as a fallback/opt-out, not removed.
    lgmresM, lgmresK, lgmresAlwaysReset
        Only used when ``outerSolver == "amgcl_lgmres"``: forwarded to AMGCL's own ``lgmres::params``
        fields ``M`` (inner iterations per outer restart, default ``30``), ``K`` (recycled/augmented
        vectors carried between restarts, default ``3``), and ``always_reset`` (default ``True`` here,
        matching AMGCL's own upstream default). ``always_reset=False`` keeps the recycled Krylov
        vectors alive *across* separate outer solves, on the theory that a hard solve's converged
        subspace is a useful warm start for the next one; measured on real dumped systems this
        contributed *nothing* (bit-identical iteration counts whether reset or not), and on a real live
        run it was actively *harmful* -- a solve that already struggled (125 outer iterations, flagged
        as "possible preconditioner degradation") left a poorly-conditioned recycled subspace that,
        left unreset, compounded into a subsequent solve needing 428 iterations (vs. 159 with
        ``always_reset=True``) and, in the original live run, an outright NaN. There is no evidence
        recycling ever helps here, and clear evidence it can hurt badly -- so this now defaults to
        AMGCL's own choice of resetting every solve.
    lgmresResetOnNewIncrement
        Only used when ``outerSolver == "amgcl_lgmres"``. When ``True``, the recycled Krylov vectors are
        discarded (one-shot, via the ``resetOnce`` parameter to ``PyAMGCLLGMRESSolver.solve``) exactly on
        a solve where ``newIncrement`` is detected -- i.e. at an increment/cutback boundary, never on a
        true-residual continuation retry of the same solve -- and kept alive across every other call. A
        candidate middle ground between always resetting and never resetting, for a model where
        within-increment recycling might help even though cross-increment recycling (see
        ``lgmresAlwaysReset`` above) measurably does not. Default ``False``: unvalidated, so this
        leaves ``lgmres``'s behaviour identical to always-resetting unless explicitly turned on.
    sweeps
        Block Gauss-Seidel sweeps per preconditioner application.
    symmetric
        If True, each sweep is followed by a reverse-order sweep (symmetric block Gauss-Seidel).
    fieldPreconds
        Optional mapping of field name to an AMGCL preconditioner parameter tree, overriding the
        dimension-based default for that field.
    useRigidBodyNullspace
        ``True`` (the default) builds a vector field's near null-space as the full rigid-body basis --
        translations *and* rotations -- once nodal coordinates for that field are available via
        ``self._model`` (set by :meth:`~edelweissfe.linsolve.base.LinearSolver.setModel`), instead of
        translations alone. The rotations matter because, for 3D elasticity, they are (like the
        translations) directions in which the discretized operator has essentially zero stiffness, so
        smoothed aggregation needs them in its near-null-space basis to represent that error class
        exactly on the coarse levels -- without them, that class of error has nowhere efficient to go,
        since a Chebyshev smoother is (by its own spectral-window construction) not targeting
        near-zero-eigenvalue error either. Measured ~28-31% fewer isolated per-field outer iterations on
        two real captured systems, robust to both thread count and the Chebyshev ``power_iters``
        setting, unlike translations-only, which is sensitive to both. Falls back to translations-only
        automatically for any field whose coordinates are unavailable (e.g. an offline probe that only
        ever calls the lower-level ``setFieldStructure``, with no model to read coordinates from) --
        never a hard requirement. Set ``False`` to force translations-only unconditionally.
    p1Maps
        Optional mapping of field name to a ``(isCorner, edgeEndpoints)`` P1 topology map
        (:func:`edelweissfe.numerics.p1topology.buildP1Map`), injected directly. A vector field named
        here gets a :class:`~edelweissfe.linsolve.blockamg.ptwogrid.PTwoGridPreconditioner` (an
        opt-in, experimental two-grid variant -- see the module docstring of
        :mod:`edelweissfe.linsolve.blockamg.ptwogrid`) instead of the single-level AMGCL default -- the
        map's presence *is* the opt-in, matching ``fieldPreconds``'s own by-presence convention. This
        is the offline-probe construction path (the map is known before the solver exists); for a live
        run built from a ``linsolverConfigFile``, use ``p1FieldNames`` instead -- the actual topology is
        computed lazily, once, the first time one of those fields' hierarchies is built (from
        ``self._model``, set by ``setModel``, rather than pushed in by the nonlinear solver ahead of
        time).
    p1FieldNames
        Optional list of vector field names to use p-multigrid for. A field named here gets its P1
        topology map computed on first need (via :func:`edelweissfe.numerics.p1topology.buildP1Map` on
        ``self._model``) and cached for this instance's lifetime. Ignored for a field also present in
        ``p1Maps`` (that field's map is already known; nothing to compute).
    etaMin, etaMax
        Clamp on the Eisenstat--Walker forcing tolerance (ignored if ``outerTol`` is given). ``etaMax``
        is also the tolerance used whenever there is no residual history to base a ratio on (the first
        solve, or the one right after a detected new increment/cutback).
    ewGamma, ewAlpha
        The Eisenstat--Walker "choice 2" parameters: ``eta_k = gamma * (||b_k|| / ||b_{k-1}||) **
        alpha``. Defaults ``gamma = 0.9``, ``alpha = (1 + sqrt 5) / 2`` are the classic values.
    residualGrowthFactor
        A solve whose ``||b||`` exceeds this multiple of the previous solve's ``||b||`` is treated as
        the first solve of a new increment (or a cutback): the forcing tolerance resets to ``etaMax``
        and the AMG hierarchies are refreshed rather than reused.
    gapCompensatedTolerance, gapSafetyFactor
        Opt-in (default ``False``, i.e. unchanged behaviour). The outer Krylov converges the
        *scaled* (equilibrated) residual while acceptance is on the *true* relative residual; the
        ratio between them (the "gap", measured 1.0-3.5 here) is why ~99% of production solves need
        a warm-restart continuation. When enabled, the first pass asks for
        ``gapSafetyFactor * eta / gap`` (gap EMA-smoothed across solves) so it lands on target in one
        pass, and any remaining continuation tightens by the *measured* gap instead of the fixed
        0.01x factor.

        **Measured, and why it is not the default.** Per increment it is a clear win: on a live
        12-increment AnchorPryOut window, continuations 126 -> 1 and outer iterations 8943 -> 7124
        (-20%), with no cutbacks or return-mapping failures. But the default path over-solves by
        ~30x, and Newton quietly relies on that: landing merely in-spec costs about one extra Newton
        iteration per increment, which parks the run just above the adaptive time stepper's
        ``criticalIter`` growth threshold. In the same window the default arm grew its increment
        0.000375 -> 0.000413 and covered 16% more simulated time, while this arm never grew. Judged
        per unit of *simulated time* rather than per increment, the per-increment saving is largely
        or wholly given back. ``gapSafetyFactor`` is the knob between the two regimes (smaller =
        more accurate = closer to the default's Newton behaviour); the sweet spot is unexplored.
    hierarchyStalenessFactor
        Refresh the AMG hierarchies before the *next* solve if this solve's outer GMRES count exceeded
        this factor times the previous solve's -- a growing count is the signal that the reused
        hierarchies are drifting away from the current Jacobian.
    trueResidualMaxContinuations
        How many times to warm-restart GMRES (``x0`` from the previous attempt, same ``As``/``bs``/``M``
        /``eta``) if the true relative residual still exceeds ``eta`` after GMRES itself reports
        convergence. ``0`` disables this and restores the original preconditioned-residual-only
        behaviour.
    verbosity
        One of ``"silent"``, ``"warning"`` (default), ``"info"``, ``"debug"``. Messages go through the
        injected Journal (:meth:`~edelweissfe.linsolve.base.LinearSolver.setJournal`), falling back to
        ``print`` if none was set. ``"warning"`` emits nothing on a normal solve and only speaks up on
        an abnormal one: outer iterations past ``warnOuterIterationsThreshold``, the true-residual
        tolerance still unmet after every continuation, or GMRES itself reporting non-convergence.
        ``"info"`` adds one compact line per solve; ``"debug"`` adds one line per true-residual
        continuation attempt too. Per-stage wall-clock (equilibration, off-diagonal split, hierarchy
        build, outer GMRES, continuations) is recorded via
        :mod:`edelweissfe.utils.performancetiming` regardless of verbosity, nested under "linear solve"
        in the job's own performance table.
    warnOuterIterationsThreshold
        A solve needing more outer GMRES iterations than this triggers a ``"warning"``-level message
        (a preconditioner-quality red flag), even when ``verbosity="silent"`` is not set to suppress it.
    dumpOnDegradationDir
        If set, write the raw ``(A, b)`` of every solve whose outer-iteration count exceeds
        ``dumpOnDegradationThreshold`` to this directory (created if missing), for offline diagnosis
        of *why* the preconditioner degraded -- unlike
        :class:`~edelweissfe.linsolve.matrixdump.matrixdump.MatrixDumpSolver`, which dumps by a
        fixed ordinal decided *before* the solve runs, this decides by the solve's own outcome, so it
        is the mechanism for capturing pathological systems (e.g. a late-increment, heavily damaged
        Jacobian) without knowing in advance which solve ordinal that will be. ``None`` (the default)
        disables this entirely -- it costs nothing when off. Dumped alongside each ``(A, b)`` pair is
        the field-block layout (name, DOF range, nodal dimension) active for that solve, needed to
        replay it through this same block preconditioner offline without a live model.
    dumpOnDegradationThreshold
        The outer-iteration count above which a solve's system is dumped. ``None`` (the default) reuses
        ``warnOuterIterationsThreshold``, so a dump and its corresponding warning message fire on
        exactly the same solves.
    dumpOnDegradationMaxDumps
        A process-wide ceiling on the number of degradation dumps written (across every
        ``BlockAMGSolver`` instance in this process, e.g. one per analysis step) -- a disk-space guard
        in the same spirit as :class:`~edelweissfe.linsolve.matrixdump.matrixdump.MatrixDumpSolver`'s
        ``maxDumps``, since a run that degrades badly could otherwise trigger the condition on every
        remaining solve. Counts every individual ``(A, b)`` pair written, including the ones
        ``dumpOnDegradationContextSolves`` adds -- a trigger with a full context window can spend
        several units of this budget at once.
    dumpOnDegradationContextSolves
        How many of the solves immediately *preceding* a degraded one to also dump (``0``, the
        default, dumps only the triggering solve itself, matching the original behaviour). A single
        ``(A, b)`` snapshot cannot distinguish "this operator is intrinsically hard" from "this
        solve's own state -- a reused, now-stale AMG hierarchy, or (with ``lgmresAlwaysReset=False``)
        a carried-over Krylov subspace -- degraded independently of the operator", because both look
        identical from the matrix alone; a naive offline diagnosis can be fooled by exactly this --
        a fresh solver instance replaying the dumped matrix alone can converge in a fraction of the
        live iteration count if the live degradation was actually state, not the operator. Capturing
        the preceding sequence lets an offline replay feed a *persistent*
        solver instance the same solves in the same order, reproducing whatever cross-solve state the
        live run had at the triggering solve, instead of only ever seeing what a cold start would do.
        Every dumped solve (trigger or context) also records ``mustRefresh``/``patternChanged``/
        ``newIncrement``/``previousOuterIters`` in the manifest, so the hierarchy-staleness question
        can often be answered directly from the manifest without needing a replay at all.
    """

    #: Degradation dumps written across every instance in this process, so
    #: ``dumpOnDegradationMaxDumps`` is a genuine process-wide ceiling -- the same reasoning as
    #: :class:`~edelweissfe.linsolve.matrixdump.matrixdump.MatrixDumpSolver`'s ``_totalDumpsWritten``.
    _degradationDumpsWritten = 0

    #: How many instances have been created in this process (one nonlinear solver may build a fresh
    #: ``BlockAMGSolver`` per analysis step), so dumps from different instances get distinct filenames
    #: -- same reasoning as :class:`~edelweissfe.linsolve.matrixdump.matrixdump.MatrixDumpSolver`'s
    #: ``_instancesCreated``.
    _instancesCreated = 0

    def __init__(
        self,
        *,
        outerTol: float = None,
        outerRestart: int = 100,
        outerMaxiter: int = 8,
        outerSolver: str = "amgcl_lgmres",
        lgmresM: int = 30,
        lgmresK: int = 3,
        lgmresAlwaysReset: bool = True,
        lgmresResetOnNewIncrement: bool = False,
        sweeps: int = 1,
        symmetric: bool = True,
        fieldPreconds: dict = None,
        useRigidBodyNullspace: bool = True,
        p1Maps: dict = None,
        p1FieldNames: list = None,
        etaMin: float = 1.0e-6,
        etaMax: float = 3.0e-4,
        ewGamma: float = 0.9,
        ewAlpha: float = 1.618033988749895,
        residualGrowthFactor: float = 4.0,
        hierarchyStalenessFactor: float = 1.5,
        gapCompensatedTolerance: bool = False,
        gapSafetyFactor: float = 0.3,
        trueResidualMaxContinuations: int = 2,
        verbosity: str = "warning",
        warnOuterIterationsThreshold: int = 100,
        dumpOnDegradationDir: str = None,
        dumpOnDegradationThreshold: int = None,
        dumpOnDegradationMaxDumps: int = 10,
        dumpOnDegradationContextSolves: int = 0,
    ):
        self._outerTol = outerTol
        self._outerRestart = outerRestart
        self._outerMaxiter = outerMaxiter
        if outerSolver not in ("scipy", "amgcl_lgmres"):
            raise ValueError("outerSolver must be 'scipy' or 'amgcl_lgmres', got {!r}".format(outerSolver))
        self._outerSolver = outerSolver
        self._lgmresM = lgmresM
        self._lgmresK = lgmresK
        self._lgmresAlwaysReset = lgmresAlwaysReset
        self._lgmresResetOnNewIncrement = lgmresResetOnNewIncrement
        self._sweeps = sweeps
        self._symmetric = symmetric
        self._fieldPreconds = fieldPreconds or {}
        self._useRigidBodyNullspace = useRigidBodyNullspace
        # Explicitly supplied offline maps: never invalidated, since they are not derived from any
        # model this solver was handed. Lazily computed ones go here too but are tracked in
        # _lazyP1MapNames so setModel() can drop exactly those on an AMR/connectivity rebuild.
        self._p1Maps = p1Maps or {}
        self._lazyP1MapNames = set()
        # Per-field node coordinates, derived from self._model; dropped in setModel().
        self._nodeCoordinateCache = {}
        # Per-field p-two-grid restriction operators, derived from the P1 topology; ditto.
        self._pNodeCache = {}
        self._p1FieldNamesRequested = set(p1FieldNames or [])
        self._etaMin = etaMin
        self._etaMax = etaMax
        self._ewGamma = ewGamma
        self._ewAlpha = ewAlpha
        self._residualGrowthFactor = residualGrowthFactor
        self._hierarchyStalenessFactor = hierarchyStalenessFactor
        self._gapCompensatedTolerance = gapCompensatedTolerance
        self._gapSafetyFactor = gapSafetyFactor
        self._trueResidualMaxContinuations = trueResidualMaxContinuations
        if verbosity not in _VERBOSITY_LEVELS:
            raise ValueError("verbosity must be one of {:}, got {!r}".format(_VERBOSITY_LEVELS, verbosity))
        self._verbosityIndex = _VERBOSITY_LEVELS.index(verbosity)
        self._warnOuterIterationsThreshold = warnOuterIterationsThreshold

        self._dumpOnDegradationDir = dumpOnDegradationDir
        self._dumpOnDegradationThreshold = (
            warnOuterIterationsThreshold if dumpOnDegradationThreshold is None else dumpOnDegradationThreshold
        )
        self._dumpOnDegradationMaxDumps = dumpOnDegradationMaxDumps
        self._dumpOnDegradationContextSolves = dumpOnDegradationContextSolves
        if self._dumpOnDegradationDir is not None:
            os.makedirs(self._dumpOnDegradationDir, exist_ok=True)
        # Rolling window of the last dumpOnDegradationContextSolves solves (each entry: solveCount, A,
        # b, blocks, mustRefresh, patternChanged, newIncrement, previousOuterIters), oldest first --
        # only populated when dumpOnDegradationDir is set, so this costs nothing otherwise. Bounded by
        # maxlen, so memory is O(contextSolves) regardless of run length.
        self._recentSolveHistory = collections.deque(maxlen=max(self._dumpOnDegradationContextSolves, 0))
        # solveCounts already written to disk this run, so an overlapping context window on a second
        # nearby trigger never dumps (or double-counts against dumpOnDegradationMaxDumps) the same
        # solve twice.
        self._dumpedSolveCounts = set()

        self._instanceOrdinal = BlockAMGSolver._instancesCreated
        BlockAMGSolver._instancesCreated += 1

        self._solveCount = 0
        self._fieldsAnnounced = None

        # Eisenstat-Walker forcing state.
        self._lastResidualNorm = None
        self._lastEta = etaMax

        # Reused hierarchy state: the built per-field AMG hierarchies, the equilibration they were
        # built for, and the field-block layout they assume -- all None until the first solve.
        self._preconditioners = None
        self._dinv = None
        self._blocks = None
        self._n = None
        self._lastNnz = None
        self._lastOuterIters = None
        self._lastContinuations = None
        # INV6: cached scaled->true relative-residual gap, used to pre-compensate the first pass.
        self._trueResidualGap = 1.0
        self._refreshNext = False

        # The persistent AMGCL lgmres outer-solver instance (only used when outerSolver ==
        # "amgcl_lgmres"), and the problem size it was built for. Unlike the per-field AMG
        # hierarchies above, this is deliberately *not* torn down and rebuilt whenever mustRefresh
        # fires -- see edelweissfe/linsolve/amgcl/amgcl-wrapper.hpp's LGMRESOuterSolverT for why: its
        # whole point is to keep its own recycled Krylov vectors alive across every solve over this
        # BlockAMGSolver's lifetime, regardless of whether the AMG hierarchies themselves were
        # refreshed that solve. It is only rebuilt on an actual size change (see __call__), the one
        # condition under which AMGCL's own preallocated scratch vectors are no longer valid.
        self._lgmresSolver = None
        self._lgmresN = None

    def _log(self, level: str, message: str) -> None:
        """Emit ``message`` through the injected Journal (see ``setJournal``, inherited from
        :class:`~edelweissfe.linsolve.base.LinearSolver`) if this instance's verbosity is at least
        ``level`` -- falls back to a plain ``print`` if no Journal has been set (e.g. an offline probe
        script driving this solver directly, outside the full nonlinear-solver/driver stack).
        """
        if self._verbosityIndex < _VERBOSITY_LEVELS.index(level):
            return
        if self._journal is not None:
            self._journal.message(message, _IDENTIFICATION, level=_JOURNAL_LEVEL[level])
        else:
            print(message, flush=True)

    def _forcingTolerance(self, residualNorm: float, newIncrement: bool) -> float:
        """The Eisenstat--Walker "choice 2" forcing tolerance for this solve, clamped and safeguarded.

        ``eta_k = gamma (||b_k|| / ||b_{k-1}||) ** alpha``, with the classic safeguard against
        over-solving (a large tightening step is only trusted if the previous tolerance was already
        small), clamped to ``[etaMin, etaMax]``. Falls back to ``etaMax`` with no history to compare
        against (the first solve, or the one right after a new increment / cutback jump -- the ratio
        across that jump does not reflect Newton convergence and is not meaningful).
        """
        if self._lastResidualNorm is None or newIncrement or self._lastResidualNorm == 0.0:
            # A previous residual of exactly zero (typically an already-converged, e.g.
            # linear-elastic, prior iterate) makes the ratio undefined rather than just
            # uninformative -- fall back the same way as the no-history case.
            return self._etaMax

        ratio = residualNorm / self._lastResidualNorm
        eta = self._ewGamma * ratio**self._ewAlpha

        safeguard = self._ewGamma * self._lastEta**self._ewAlpha
        if safeguard > 0.1:
            eta = max(eta, safeguard)

        return min(self._etaMax, max(self._etaMin, eta))

    def setModel(self, model, dofManager) -> None:
        """Take the live model/DOF layout, and drop any P1 topology map derived from the previous one.

        The base class stores the references and the field-block structure. On top of that, every
        lazily computed P1 map has to go: this is called again after each AMR/connectivity rebuild,
        and a map built for the old mesh carries the old node ordering and length, which would either
        fail :meth:`~edelweissfe.linsolve.blockamg.ptwogrid.PTwoGridPreconditioner.build`'s size
        check or, worse, silently build the wrong prolongation if the sizes happened to agree.
        Explicitly supplied ``p1Maps`` are kept -- they were never derived from this model.
        """
        super().setModel(model, dofManager)
        for fieldName in self._lazyP1MapNames:
            self._p1Maps.pop(fieldName, None)
        self._lazyP1MapNames.clear()
        self._nodeCoordinateCache.clear()
        self._pNodeCache.clear()

    def _getNodeCoordinates(self, fieldName: str) -> "np.ndarray | None":
        """This field's node coordinates, node-major, or ``None`` if unavailable.

        Reads directly from ``self._model`` (set by :meth:`~edelweissfe.linsolve.base.LinearSolver.
        setModel`) rather than a value pushed in ahead of time -- there is no live-run scenario where
        this is unavailable when it matters (``setModel`` is called on every rebuild, same as
        ``setFieldStructure`` always was), so the only real callers of the ``None`` path are offline
        probes driving this solver directly via the lower-level :meth:`setFieldStructure`, or
        ``useRigidBodyNullspace=False``.
        """
        if self._model is None:
            return None
        cached = self._nodeCoordinateCache.get(fieldName)
        if cached is not None:
            return cached
        field = self._model.nodeFields.get(fieldName)
        if field is None:
            return None
        # Cached per field: node coordinates do not change, but this is reached from the hierarchy
        # refresh, which fires essentially every Newton iteration on contact/tie models whose
        # sparsity pattern churns -- so the O(nNodes) Python loop below would otherwise rerun on
        # every linear solve. Invalidated by setModel(), the one point where the mesh can differ.
        coordinates = np.array([node.coordinates for node in field.nodes], dtype=float)
        self._nodeCoordinateCache[fieldName] = coordinates
        return coordinates

    def _getP1Map(self, fieldName: str):
        """This field's P1 corner/midside topology map, computed lazily on first need and
        cached for the lifetime of this instance (never refreshed even across AMR -- matching this
        opt-in p-multigrid path's original behaviour, unvalidated for a mesh that changes after the
        map was built and not a focus of this refactor).

        Returns the map directly supplied via the constructor's ``p1Maps`` (the offline-probe path)
        unchanged; otherwise computes it via
        :func:`~edelweissfe.numerics.p1topology.buildP1Map` from ``self._model`` (set by
        :meth:`~edelweissfe.linsolve.base.LinearSolver.setModel`) the first time this field is asked
        for and ``self._model`` is available.
        """
        if fieldName not in self._p1Maps and self._model is not None:
            from edelweissfe.numerics.p1topology import buildP1Map

            isCorner, edgeEndpoints, p1Warnings = buildP1Map(self._model, fieldName)
            self._p1Maps[fieldName] = (isCorner, edgeEndpoints)
            self._lazyP1MapNames.add(fieldName)
            for warning in p1Warnings:
                self._log("warning", warning)
        return self._p1Maps.get(fieldName)

    def _resolveBlocks(self, n: int) -> list:
        """The field blocks tiling ``[0, n)``, in DOF order, with any trailing DOFs not covered by a
        node field (e.g. scalar variables) folded into a final scalar block."""

        if self._fieldStructure is None:
            raise RuntimeError(
                "blockamg: field structure not set. It is derived from setModel() (or, at the lower "
                "level, setFieldStructure()); this solver must be driven by a caller that calls one of "
                "those."
            )
        blocks = sorted(self._fieldStructure, key=lambda field: field.start)
        cursor = 0
        for block in blocks:
            if block.start != cursor:
                raise ValueError(
                    "blockamg: field '{:}' starts at {:}, expected {:} -- fields must tile the DOF "
                    "vector contiguously".format(block.name, block.start, cursor)
                )
            cursor = block.stop
        if cursor < n:
            # DOFs past the last node field: scalar variables. One scalar block.
            blocks = blocks + [FieldBlock("scalar variables", cursor, n, 1)]
        elif cursor != n:
            raise ValueError("blockamg: field blocks cover {:} dofs, but the matrix is {:}x{:}".format(cursor, n, n))
        return blocks

    # translationNullspace/rigidBodyNullspace live in edelweissfe.linsolve.nullspace --
    # pure functions of a field's block layout, equilibration scaling, and (for the richer basis) node
    # coordinates, with no BlockAMGSolver-instance state, so there is nothing to keep here as a method.

    def _dumpOneSystem(
        self,
        solveCount: int,
        A: sp.csr_matrix,
        b: np.ndarray,
        blocks: list,
        role: str,
        triggerSolveCount: int,
        stateFields: dict,
    ) -> None:
        """Write one raw ``(A, b)`` pair, plus its field-block layout and solver-state bookkeeping, to
        ``dumpOnDegradationDir`` -- see the class docstring's ``dumpOnDegradationDir``/
        ``dumpOnDegradationContextSolves`` entries. Used both for the solve that actually crossed the
        threshold (``role="trigger"``) and for the preceding solves ``dumpOnDegradationContextSolves``
        adds (``role="context"``) -- both are plain ``(A, b)`` snapshots on disk, distinguished only by
        the manifest record, so either can be replayed the same way offline.

        ``A``/``b`` are the raw system as handed to :meth:`__call__` (before equilibration), matching
        :class:`~edelweissfe.linsolve.matrixdump.matrixdump.MatrixDumpSolver`'s own dump format so the
        same offline tooling can read either.
        """
        stem = "{:02d}_{:05d}".format(self._instanceOrdinal, solveCount)
        matrixPath = os.path.join(self._dumpOnDegradationDir, "A_{:}.npz".format(stem))
        rhsPath = os.path.join(self._dumpOnDegradationDir, "b_{:}.npy".format(stem))

        # Uncompressed, same reasoning as MatrixDumpSolver: these systems are large, and compression
        # would spend more time than the degraded solve being captured.
        sp.save_npz(matrixPath, A, compressed=False)
        np.save(rhsPath, np.asarray(b))

        record = {
            "solveCount": solveCount,
            "role": role,
            "triggerSolveCount": triggerSolveCount,
            "contextOffset": triggerSolveCount - solveCount,
            "rows": int(A.shape[0]),
            "nnz": int(A.nnz),
            "matrixFile": os.path.basename(matrixPath),
            "rhsFile": os.path.basename(rhsPath),
            "blocks": [
                {"name": block.name, "start": block.start, "stop": block.stop, "dimension": block.dimension}
                for block in blocks
            ],
            **stateFields,
        }

        manifestPath = os.path.join(self._dumpOnDegradationDir, "manifest.jsonl")
        with open(manifestPath, "a") as manifestFile:
            manifestFile.write(json.dumps(record) + "\n")

        self._dumpedSolveCounts.add(solveCount)
        BlockAMGSolver._degradationDumpsWritten += 1

        self._log(
            "warning",
            "blockamg: dumped {:} solve #{:} (of trigger #{:}) to {:}; {:} of {:} degradation dumps "
            "used".format(
                role,
                solveCount,
                triggerSolveCount,
                os.path.basename(matrixPath),
                BlockAMGSolver._degradationDumpsWritten,
                self._dumpOnDegradationMaxDumps,
            ),
        )

    def _dumpDegradedSystem(
        self,
        A: sp.csr_matrix,
        b: np.ndarray,
        blocks: list,
        outerIters: int,
        trueResidual: float,
        eta: float,
        continuations: int,
        info: int,
        mustRefresh: bool,
        patternChanged: bool,
        newIncrement: bool,
        previousOuterIters,
    ) -> None:
        """Dump the triggering solve itself, then as much of its context window
        (``dumpOnDegradationContextSolves`` preceding solves, from :attr:`_recentSolveHistory`) as the
        remaining ``dumpOnDegradationMaxDumps`` budget allows, oldest first.
        """
        triggerSolveCount = self._solveCount
        if triggerSolveCount not in self._dumpedSolveCounts and (
            BlockAMGSolver._degradationDumpsWritten < self._dumpOnDegradationMaxDumps
        ):
            self._dumpOneSystem(
                triggerSolveCount,
                A,
                b,
                blocks,
                role="trigger",
                triggerSolveCount=triggerSolveCount,
                stateFields={
                    "outerIters": outerIters,
                    "trueResidual": trueResidual,
                    "eta": eta,
                    "continuations": continuations,
                    "info": info,
                    "mustRefresh": mustRefresh,
                    "patternChanged": patternChanged,
                    "newIncrement": newIncrement,
                    "previousOuterIters": previousOuterIters,
                },
            )

        for entry in self._recentSolveHistory:
            if BlockAMGSolver._degradationDumpsWritten >= self._dumpOnDegradationMaxDumps:
                break
            if entry["solveCount"] in self._dumpedSolveCounts:
                continue
            self._dumpOneSystem(
                entry["solveCount"],
                entry["A"],
                entry["b"],
                entry["blocks"],
                role="context",
                triggerSolveCount=triggerSolveCount,
                stateFields={
                    "outerIters": entry["outerIters"],
                    "trueResidual": entry["trueResidual"],
                    "eta": entry["eta"],
                    "continuations": entry["continuations"],
                    "info": entry["info"],
                    "mustRefresh": entry["mustRefresh"],
                    "patternChanged": entry["patternChanged"],
                    "newIncrement": entry["newIncrement"],
                    "previousOuterIters": entry["previousOuterIters"],
                },
            )

    def __call__(self, A, b):
        solveStartTime = time.time()
        from edelweissfe.linsolve.amgcl.amgcl import PyAMGCLMatrix, PyAMGCLSolver

        self._solveCount += 1
        A = A.tocsr()
        n = A.shape[0]
        blocks = self._resolveBlocks(n)
        slices = [slice(block.start, block.stop) for block in blocks]
        b = np.asarray(b).reshape(n)

        if self._outerSolver == "amgcl_lgmres" and (self._lgmresSolver is None or n != self._lgmresN):
            # A size change invalidates AMGCL's own preallocated Arnoldi/augmentation-vector scratch
            # (fixed at construction, see LGMRESOuterSolverT/PyAMGCLLGMRESSolver) -- the same condition
            # mustRefresh tracks below for the AMG hierarchies (n != self._n), reused here. Note this
            # constructs independently of mustRefresh: a hierarchy refresh alone (e.g. a residual jump)
            # must *not* rebuild this object, since keeping its recycled Krylov vectors alive across
            # exactly those Newton-iteration boundaries is the entire reason to use it.
            from edelweissfe.linsolve.amgcl.amgcl import PyAMGCLLGMRESSolver

            self._lgmresSolver = PyAMGCLLGMRESSolver(
                n,
                {"M": self._lgmresM, "K": self._lgmresK, "always_reset": self._lgmresAlwaysReset},
            )
            self._lgmresN = n

        fieldNames = [block.name for block in blocks]
        if fieldNames != self._fieldsAnnounced:
            self._log("info", "blockamg: fields = {:}".format(fieldNames))
            self._fieldsAnnounced = fieldNames

        residualNorm = float(np.linalg.norm(b))
        newIncrement = (
            self._lastResidualNorm is not None and residualNorm > self._residualGrowthFactor * self._lastResidualNorm
        )
        # A's sparsity pattern churns between Newton iterations on this class of problem (condensed
        # contact/tie systems, whose active constraint set changes every Newton iteration) -- a
        # hierarchy built for a different pattern is not just "a bit stale", it can be a drastically
        # worse preconditioner (measured: 494 vs. 94 outer iterations on one such transition, an
        # outright wall-clock regression, not a graceful few-extra-iterations degradation). ``nnz`` is a
        # cheap, free (O(1)) proxy for "the pattern changed" -- not exact (two different patterns could
        # coincidentally share a total nnz), but it caught the one measured failure case and errs in the
        # safe direction (an unnecessary refresh costs time, never correctness).
        patternChanged = self._lastNnz is not None and A.nnz != self._lastNnz

        # Refresh the per-field AMG hierarchies (rather than reuse the standing ones) when there is
        # nothing to reuse yet, the field-block layout changed (e.g. an AMR event resized the DOF
        # vector), the sparsity pattern changed, a residual jump marks a new increment / cutback, or the
        # previous solve's own outer count asked for it (drifted too far from the one before it).
        mustRefresh = (
            self._preconditioners is None
            or blocks != self._blocks
            or n != self._n
            or patternChanged
            or newIncrement
            or self._refreshNext
        )
        self._refreshNext = False

        with performancetiming.timeit("equilibration"):
            if mustRefresh:
                # Symmetric diagonal equilibration. Solve A x = b as (D A D)(D^-1 x) = D b, i.e.
                # As z = bs with x = D z; D = diag(dinv), dinv = 1/sqrt(|diag A|).
                diagonal = np.abs(A.diagonal())
                # A zero diagonal entry cannot be equilibrated: 1/sqrt(0) is inf, which would poison
                # As and bs with non-finite values before the hierarchy build, and the failure would
                # surface much later as a NaN correction (i.e. as DivergingSolution, which reads like
                # a physics problem rather than an unsupported system). Flooring the divisor is not a
                # fix either -- a 1e-300 floor merely turns inf into ~1e150 and blows the scaled
                # operator up to ~1e149, which is finite and equally useless. The honest answer is
                # that a system with a structurally zero diagonal is a saddle-point system (e.g. the
                # multiplier block of a Lagrangian constraint, which never writes its own diagonal),
                # and smoothed-aggregation AMG on a zero block is not something this solver does.
                # Say so, and name the offending dofs.
                zeroDiagonal = np.nonzero(diagonal == 0.0)[0]
                if zeroDiagonal.size:
                    raise ValueError(
                        "blockamg: {:} dof(s) have a zero diagonal entry (first at index {:}) and "
                        "cannot be diagonally equilibrated. This is characteristic of a saddle-point "
                        "system -- a Lagrange-multiplier constraint, for instance, couples the "
                        "multiplier without ever writing its own diagonal -- which this field-split "
                        "AMG scheme does not support. Use a direct solver for such a model.".format(
                            zeroDiagonal.size, int(zeroDiagonal[0])
                        )
                    )
                dinv = 1.0 / np.sqrt(diagonal)
            else:
                # Reuse the equilibration the standing hierarchies were built for. This stays a valid
                # diagonal similarity scaling of the *current* A x = b regardless of how it was chosen,
                # so correctness (the outer GMRES converges on the true, fresh As/bs) is unaffected;
                # only the preconditioner's quality can drift, which costs outer iterations, never a
                # wrong answer.
                dinv = self._dinv

            # INV1: fast in-place diagonal equilibration As = D A D (D = diag(dinv)), vectorized
            # over the CSR data array -- avoids two full scipy SpGEMMs. Numerically identical.
            As = A.copy()
            As.data *= dinv[As.indices]
            As.data *= np.repeat(dinv, np.diff(As.indptr))
            bs = dinv * b

        # The outer GMRES operator SpMV and the true-residual check both otherwise run through a
        # plain scipy.sparse CSR matvec on the full coupled system -- not OpenMP-threaded regardless
        # of OMP_NUM_THREADS, since scipy's own sparse matvec is single-threaded C code (the same
        # mechanism affects a raw SciPy CSR matvec anywhere in this codebase, not just here; measured
        # at roughly 15% of this solver's own wall-clock on a real reference model). This conversion is
        # not amortized across solves -- As's sparsity pattern churns every solve on a model with
        # contact/tie constraints, so there is no stable pattern to cache a threaded build against --
        # so it is paid fresh here every call, same as equilibration above.
        with performancetiming.timeit("threaded operator build"):
            threadedAs = PyAMGCLMatrix()
            threadedAs.build(As)
        outerOperator = LinearOperator((n, n), matvec=threadedAs.matvec, dtype=As.dtype)

        # Off-diagonal couplings (for the sweep) are needed every solve regardless of refresh/reuse.
        with performancetiming.timeit("off-diagonal split"):
            offBlocks = {}
            for i in range(len(slices)):
                rowBlock = As[slices[i], :]
                for j in range(len(slices)):
                    if i != j:
                        # INV4: hand the coupling block to AMGCL's OpenMP-threaded matvec instead of
                        # keeping it as a scipy CSR whose `@` is single-threaded C (measured at 12-16%
                        # of the whole preconditioner apply). Same operator, same arithmetic.
                        threadedOffBlock = PyAMGCLMatrix()
                        threadedOffBlock.buildRect(rowBlock[:, slices[j]].tocsr())
                        offBlocks[(i, j)] = threadedOffBlock

        if mustRefresh:
            with performancetiming.timeit("hierarchy build"):
                # One AMG hierarchy per field, built fresh. A vector field gets its translations as the
                # near null-space; a scalar field the default constant.
                diagBlocks = [As[sl, :][:, sl].tocsr() for sl in slices]
                preconditioners = []
                for i, block in enumerate(blocks):
                    isVectorField = block.dimension > 1
                    p1Map = None
                    if isVectorField and (block.name in self._p1Maps or block.name in self._p1FieldNamesRequested):
                        # p-two-grid: opted in via the constructor's p1Maps (offline-probe path) or
                        # p1FieldNames (live path: computed lazily here from self._model -- set by
                        # setModel -- on first need, instead of waiting for a push from the driver).
                        with performancetiming.timeit("p1 topology"):
                            p1Map = self._getP1Map(block.name)
                    if p1Map is not None:
                        from edelweissfe.linsolve.blockamg.ptwogrid import (
                            PTwoGridPreconditioner,
                        )

                        isCorner, edgeEndpoints = p1Map
                        pNode = self._pNodeCache.get(block.name)
                        solver = PTwoGridPreconditioner(isCorner, edgeEndpoints, pNode=pNode)
                        if pNode is None:
                            self._pNodeCache[block.name] = solver.pNode
                        # Give p-two-grid's coarse solve the same rigid-body-vs-translations choice
                        # the full field gets below -- None here (useRigidBodyNullspace disabled, or
                        # no coordinates available) makes build() fall back to translations-only
                        # internally, exactly like the non-p1Map path below.
                        coords = self._getNodeCoordinates(block.name) if self._useRigidBodyNullspace else None
                        solver.build(diagBlocks[i], dinv[slices[i]], coords=coords)
                        preconditioners.append(solver)
                        continue

                    precondParams = dict(
                        self._fieldPreconds.get(
                            block.name, _DEFAULT_VECTOR_PRECOND if isVectorField else _DEFAULT_SCALAR_PRECOND
                        )
                    )
                    backendPrecision = precondParams.pop("backendPrecision", "double")
                    backendBlockSize = precondParams.pop("backendBlockSize", 1)
                    solver = PyAMGCLSolver(
                        {
                            "precond": precondParams,
                            "backendPrecision": backendPrecision,
                            "backendBlockSize": backendBlockSize,
                        }
                    )
                    # set_nullspace() is unsupported (and always raises) on a block backend -- AMGCL's
                    # own near-null-space path is unimplemented for block value types; untouched by the
                    # rigid-body-basis choice below since backendBlockSize > 1 stays opt-in.
                    if isVectorField and backendBlockSize == 1:
                        with performancetiming.timeit("nullspace construction"):
                            # Coordinates come from self._model (set by setModel), read fresh every
                            # rebuild rather than pushed in ahead of time. Gated on
                            # useRigidBodyNullspace so that opting out really does fall through to
                            # translations-only below, exactly as the p1Map path above does.
                            coords = self._getNodeCoordinates(block.name) if self._useRigidBodyNullspace else None
                            if coords is not None:
                                # Full rigid-body basis (translations + rotations): measured ~28-31%
                                # fewer isolated outer iterations than translations alone on two real
                                # captured systems, and robust to both thread count and power_iters
                                # where translations-only is sensitive to both.
                                nullspace = rigidBodyNullspace(block, coords, dinv[slices[i]])
                            else:
                                # Coordinates never arrived (useRigidBodyNullspace=False, or an offline
                                # probe driving this solver directly via the lower-level
                                # setFieldStructure, with no model to read coordinates from) -- fall
                                # back to translations-only. Under a Chebyshev smoother, translations
                                # alone are still a real improvement over no null-space at all, just
                                # not the further ~30% the full rigid-body basis measures (a Chebyshev
                                # smoother's own spectral window structurally excludes the near-zero
                                # eigenvalues rigid-body modes correspond to, so how well the coarse
                                # levels represent that error class -- via the near-null-space basis --
                                # directly determines how much of it a V-cycle actually removes).
                                nullspace = translationNullspace(block, dinv[slices[i]])
                        solver.set_nullspace(nullspace)
                    solver.build(diagBlocks[i])
                    preconditioners.append(solver)
                self._preconditioners = preconditioners
                self._dinv = dinv
                self._blocks = blocks
                self._n = n
                self._lastNnz = A.nnz
        preconditioners = self._preconditioners

        nFields = len(slices)
        sizes = [block.stop - block.start for block in blocks]

        def sweepOnce(order, residual, x):
            for i in order:
                localResidual = residual[slices[i]].copy()
                for j in range(nFields):
                    if j != i:
                        localResidual -= offBlocks[(i, j)].matvecRect(x[j])  # INV4: threaded
                x[i] = preconditioners[i].applyPreconditioner(localResidual)

        def blockGaussSeidel(residual):
            x = [np.zeros(sizes[i]) for i in range(nFields)]
            for _ in range(self._sweeps):
                sweepOnce(range(nFields), residual, x)
                if self._symmetric:
                    sweepOnce(range(nFields - 1, -1, -1), residual, x)
            return np.concatenate(x)

        preconditioner = LinearOperator((n, n), matvec=blockGaussSeidel, dtype=As.dtype)

        if self._outerTol is not None:
            eta = self._outerTol
        else:
            eta = self._forcingTolerance(residualNorm, newIncrement)

        # INV6: the outer Krylov converges the *scaled* residual, but acceptance is on the *true*
        # residual; ask the first pass for eta/gap (gap cached from previous solves, safety 0.3) so it
        # lands on target in one pass instead of paying a warm-restart continuation.
        if self._gapCompensatedTolerance:
            firstPassEta = max(self._gapSafetyFactor * eta / max(self._trueResidualGap, 1.0), 1e-14)
        else:
            firstPassEta = eta

        # outerSolver == "amgcl_lgmres" (the default): both outer-solve call sites below (this one and
        # the true-residual continuation retry) dispatch to AMGCL's own native amgcl::solver::lgmres
        # (self._lgmresSolver, built/reused above) instead of scipy.sparse.linalg.gmres. lgmres's own
        # `maxiter` bounds *all* Arnoldi steps across every internal restart, unlike scipy's separate
        # restart/maxiter (restart cycles) knobs -- passing outerRestart * outerMaxiter through as a
        # single total-iteration budget is a generous translation between the two, not an attempt at
        # matching scipy's semantics bit-for-bit. blockGaussSeidel itself
        # is reused unchanged for both paths -- it already has exactly the "1D array in, 1D array out"
        # shape AMGCL's callback needs, with no LinearOperator indirection required.
        with performancetiming.timeit("outer GMRES"):
            history = []
            if self._outerSolver == "scipy":
                z, info = gmres(
                    outerOperator,
                    bs,
                    M=preconditioner,
                    rtol=firstPassEta,
                    atol=0.0,
                    restart=self._outerRestart,
                    maxiter=self._outerMaxiter,
                    callback=lambda residualNorm: history.append(residualNorm),
                    callback_type="pr_norm",
                )
                outerIters = len(history)
            else:
                # resetOnce (opt-in via lgmresResetOnNewIncrement -- unvalidated, defaults False,
                # leaving lgmres's behaviour unchanged unless explicitly turned on): discard the
                # recycled Krylov subspace
                # exactly at an increment/cutback boundary -- where the previous solve's Jacobian
                # belongs to a different (possibly harder-or-easier) problem -- and keep recycling
                # within an increment's own Newton sequence, AMGCL's own intended use case for it.
                # Never set on the continuation retry below: that call is a warm restart of *this same*
                # outer solve at a tighter tolerance, not a new increment.
                z = self._lgmresSolver.solve(
                    As,
                    bs,
                    blockGaussSeidel,
                    firstPassEta,
                    self._outerRestart * self._outerMaxiter,
                    resetOnce=(newIncrement and self._lgmresResetOnNewIncrement),
                )
                outerIters = self._lgmresSolver.lastIterations
                info = 0 if self._lgmresSolver.lastError <= eta else 1
            x = dinv * z
            # A x - b = D^-1 (As z - bs) exactly (D = diag(dinv)) -- verified numerically against the
            # direct A @ x - b computation on a real dumped system before use. Rides on the same
            # threaded operator built above; no second, unscaled full-system matvec.
            trueResidual = np.linalg.norm(threadedAs.residual(bs, z) / dinv) / max(np.linalg.norm(b), 1e-300)

        # GMRES's own stopping check (callback_type="pr_norm") is on the *preconditioned* residual,
        # not the true one -- with an imperfect preconditioner (this one, by design -- a block
        # Gauss-Seidel sweep of per-field AMG V-cycles is only an approximate inverse of the coupled
        # operator) the two can diverge substantially, so "converged" per GMRES can still leave a true
        # residual well above the requested eta (measured on a real reference model: one solve reached
        # a 1.6e-2 true residual when 1e-4 was requested).
        #
        # Two failed attempts before this one, both found empirically (every continuation logged "0
        # more outer GMRES iters" until fixed):
        #   1. A warm restart with the *same* rtol is a no-op -- x0=z already satisfies that exact
        #      preconditioned-residual criterion, so GMRES re-declares convergence immediately.
        #   2. Tightening the *requested* eta proportionally (eta * (eta/trueResidual)), or relative to
        #      the callback's last-reported "pr_norm" -- both still occasionally produced a target
        #      *larger* than eta itself (impossible if the callback's pr_norm were on the same relative
        #      scale as rtol; it evidently is not -- likely an absolute residual, not one normalized by
        #      ||bs||). Rather than reverse-engineer scipy's internal residual bookkeeping, avoid it
        #      entirely: tighten purely within rtol's own units, which scipy already interprets
        #      correctly by construction.
        # Fix: geometrically tighten the *requested* rtol itself by a fixed factor each continuation
        # (0.01x), guaranteeing a strictly smaller, dimensionally-consistent target with no dependency
        # on the callback's residual scale.
        # INV6: learn the gap from this pass (true residual achieved vs the scaled tolerance it was
        # given), smoothed so one atypical solve cannot swing the next one's first-pass tolerance.
        _measuredGap = trueResidual / max(firstPassEta, 1e-300)
        if np.isfinite(_measuredGap) and _measuredGap > 0.0:
            self._trueResidualGap = max(0.5 * self._trueResidualGap + 0.5 * _measuredGap, 1.0)

        with performancetiming.timeit("true-residual continuations"):
            continuationEta = min(eta, firstPassEta) if self._gapCompensatedTolerance else eta
            continuations = 0
            while trueResidual > eta and continuations < self._trueResidualMaxContinuations:
                continuations += 1
                # INV5 (experiment): gap-informed tightening instead of a blunt fixed 0.01x. The pass
                # just finished converged the *scaled* residual to continuationEta and landed at
                # trueResidual; both are relative residuals in the same units, so their ratio is the
                # scaled->true gap g. Landing the next pass on target needs ~eta/g, so tighten by that
                # (with a safety margin), never looser, and always at least 2x per round so the loop
                # still terminates.
                if self._gapCompensatedTolerance:
                    _gap = trueResidual / max(continuationEta, 1e-300)
                    _proposed = self._gapSafetyFactor * eta / max(_gap, 1.0)
                    continuationEta = max(min(_proposed, continuationEta * 0.5), 1e-14)
                else:
                    continuationEta *= 0.01
                if self._outerSolver == "scipy":
                    continuationHistory = []
                    z, info = gmres(
                        outerOperator,
                        bs,
                        x0=z,
                        M=preconditioner,
                        rtol=continuationEta,
                        atol=0.0,
                        restart=self._outerRestart,
                        maxiter=self._outerMaxiter,
                        callback=lambda residualNorm: continuationHistory.append(residualNorm),
                        callback_type="pr_norm",
                    )
                    continuationIters = len(continuationHistory)
                else:
                    z = self._lgmresSolver.solve(
                        As,
                        bs,
                        blockGaussSeidel,
                        continuationEta,
                        self._outerRestart * self._outerMaxiter,
                        x0=z,
                    )
                    continuationIters = self._lgmresSolver.lastIterations
                    info = 0 if self._lgmresSolver.lastError <= continuationEta else 1
                x = dinv * z
                trueResidual = np.linalg.norm(threadedAs.residual(bs, z) / dinv) / max(np.linalg.norm(b), 1e-300)
                outerIters += continuationIters
                self._log(
                    "debug",
                    "blockamg:   continuation {:}/{:} eta={:.1e} +{:}it res={:.1e}".format(
                        continuations,
                        self._trueResidualMaxContinuations,
                        continuationEta,
                        continuationIters,
                        trueResidual,
                    ),
                )

        previousOuterIters = self._lastOuterIters
        if self._lastOuterIters is not None and outerIters > self._hierarchyStalenessFactor * self._lastOuterIters:
            self._refreshNext = True
        self._lastOuterIters = outerIters
        self._lastContinuations = continuations
        self._lastResidualNorm = residualNorm
        self._lastEta = eta

        solveElapsedTime = time.time() - solveStartTime
        self._log(
            "info",
            "blockamg: solve #{:<4d} {:7s} {:3d}it eta={:.1e} res={:.1e} cont={:} time={:7.3f}s".format(
                self._solveCount,
                "REFRESH" if mustRefresh else "reuse",
                outerIters,
                eta,
                trueResidual,
                continuations,
                solveElapsedTime,
            ),
        )
        if outerIters > self._warnOuterIterationsThreshold:
            self._log(
                "warning",
                "blockamg: WARNING solve #{:} needed {:} outer GMRES iterations (> threshold {:}) -- "
                "possible preconditioner degradation".format(
                    self._solveCount, outerIters, self._warnOuterIterationsThreshold
                ),
            )
        if (
            self._dumpOnDegradationDir is not None
            and outerIters > self._dumpOnDegradationThreshold
            and BlockAMGSolver._degradationDumpsWritten < self._dumpOnDegradationMaxDumps
        ):
            self._dumpDegradedSystem(
                A,
                b,
                blocks,
                outerIters,
                trueResidual,
                eta,
                continuations,
                info,
                mustRefresh,
                patternChanged,
                newIncrement,
                previousOuterIters,
            )
        # Recorded *after* the dump above, so a trigger's own context window (from
        # _recentSolveHistory) never includes itself -- only solves strictly preceding it. Guarded on
        # dumpOnDegradationContextSolves so a run with the feature off never holds onto extra matrices.
        if self._dumpOnDegradationDir is not None and self._dumpOnDegradationContextSolves > 0:
            self._recentSolveHistory.append(
                {
                    "solveCount": self._solveCount,
                    # Copied, not referenced: NIST hands the same CSR matrix and residual buffer back
                    # every Newton iteration (CSRGenerator.updateInPlace, and `R[:] = ...`), so storing
                    # references would make every history entry alias the trigger-time values -- the
                    # exact opposite of the authentic preceding-solve sequence this window exists for.
                    "A": A.copy(),
                    "b": np.array(b, copy=True),
                    "blocks": blocks,
                    "outerIters": outerIters,
                    "trueResidual": trueResidual,
                    "eta": eta,
                    "continuations": continuations,
                    "info": info,
                    "mustRefresh": mustRefresh,
                    "patternChanged": patternChanged,
                    "newIncrement": newIncrement,
                    "previousOuterIters": previousOuterIters,
                }
            )
        if trueResidual > eta:
            self._log(
                "warning",
                "blockamg: WARNING solve #{:} true residual {:.2e} still exceeds requested eta={:.2e} "
                "after {:} continuation(s) -- did not fully converge".format(
                    self._solveCount, trueResidual, eta, continuations
                ),
            )
        if info != 0:
            self._log(
                "warning",
                "blockamg: WARNING solve #{:} GMRES reported info={:} (did not converge within "
                "maxiter on its own preconditioned-residual criterion)".format(self._solveCount, info),
            )

        return x
