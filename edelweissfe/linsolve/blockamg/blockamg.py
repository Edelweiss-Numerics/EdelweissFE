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
field, combined by a block Gauss-Seidel sweep, used to precondition an outer Krylov solve (GMRES-type) over the full
coupled system. Each field's operator (elasticity for a displacement field, a Helmholtz-like operator
for a damage field) is individually AMG-friendly even though their monolithic coupling is not.

The field structure -- which DOFs belong to which field, and each field's nodal dimension -- is not
carried by the matrix; it is pushed in from the DofManager by the nonlinear solver (via
:class:`~edelweissfe.linsolve.base.LinearSolver`), so nothing about the block layout
has to be specified by hand.

What it does per solve
----------------------

#. **Equilibrate.** Symmetric diagonal (Jacobi) scaling :math:`\\hat A = D A D`, with
   :math:`D = \\operatorname{diag}(\\lvert A_{ii} \\rvert^{-1/2})` (its diagonal is ``scaling`` in the code), removes
   the large dynamic range (Dirichlet penalties + stiffness) that otherwise wrecks AMG's
   strength-of-connection. The solve is done on :math:`\\hat A` and unscaled at the end.
#. **Split** :math:`\\hat A` into the field diagonal blocks and their couplings, from the field ranges.
#. **Build one AMG hierarchy per field** (AMGCL: built via ``build`` and applied via ``applyPreconditioner``
   on every visit of the sweep to its field, i.e. once per outer Krylov iteration with ``symmetric`` off;
   reused across Newton iterations while the Jacobian and its sparsity pattern change little -- see the
   reuse rules below). A vector field (nodal dimension > 1, e.g. displacement) is given its full rigid-body
   near-null-space -- 3 translations plus 3 infinitesimal rotations, the standard basis for 3D
   elasticity AMG, since both are directions in which the discretized operator has essentially zero
   stiffness -- when node coordinates are available, translations alone otherwise; a scalar field
   takes the default constant. A block backend (``backendBlockSize > 1``, part of the recommended
   configuration) cannot take a near-null-space and uses AMGCL's constant blocks instead.
#. **Precondition GMRES** with a block Gauss-Seidel sweep over the fields, each field's correction
   coming from one AMG V-cycle on its block, the couplings folded in between fields.

Its reach is memory: AMG needs O(n) of it where a direct factorization does not fit. Its limit is the
displacement block of a damaging, non-associated material, which is non-symmetric and whose symmetric
part becomes indefinite at the damage onset -- smoothed aggregation and Chebyshev smoothing assume
neither. With the recommended configuration of the theory page most solves take 5 to 50 outer
iterations; the hardest damage-onset solves stop at the iteration cap of 50, return their iterate, and the
hierarchies are rebuilt for the next solve. The cap is a deliberate budget, not a tolerance: a solve that
needs more is taken as a sign that the hierarchies no longer fit the Jacobian, not given more iterations. A
non-symmetric-aware AMG library (e.g. Trilinos/MueLu) might do better, at the cost of a heavy external
dependency.
"""

import collections
import inspect
import json
import os
import time

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import LinearOperator, gmres

import edelweissfe.utils.performancetiming as performancetiming
from edelweissfe.linsolve.base import FieldBlock, LinearSolver, LinearSolveSummary
from edelweissfe.linsolve.nullspace import rigidBodyNullspace, translationNullspace

# Ordered low-to-high; index comparison decides whether a message at a given level should print.
# "warning": only abnormal conditions (excessive outer iterations, unmet true-residual tolerance,
# non-convergence) -- the default, so a normal run stays quiet. "info": one compact line per solve.
# "debug": full detail, including every true-residual continuation attempt. Independent of, and does
# not affect, the Journal instance's own message-level suppression (see _log()) -- this gate decides
# whether blockamg attempts to log at all; Journal's own level decides whether the attempt is shown.
_VERBOSITY_LEVELS = ("silent", "warning", "info", "debug")
# Journal indentation level, not verbosity: "info"/"debug" both nest at level 2, alongside the Newton
# residual line each solve corresponds to -- they are the same nesting depth (a per-Newton-iteration
# detail), just gated by different verbosity settings. "warning" stays at level 0 (the least indented,
# most visible spot in the log) regardless of nesting, since a warning is rare and meant to stand out.
_JOURNAL_LEVEL = {"warning": 0, "info": 2, "debug": 2}
_IDENTIFICATION = "BlockAMGSolver"

# "backendPrecision" and "backendBlockSize" are not AMGCL parameters -- they select the AMGCL
# wrapper's own backend value type ("double" (default) or "float" for backendPrecision; 1 (default,
# scalar), 2, or 3 for a block-valued backend operating on B x B nodal blocks for backendBlockSize).
# __call__ pops both out of whichever precond dict applies (default or a fieldPreconds override)
# before forwarding the rest verbatim as the AMGCL JSON parameter tree.
#
# backendPrecision defaults to "double". On the scalar backend "float" is AMGCL's builtin<float>,
# which builds the whole hierarchy in single precision; measured on all 9 dumped ords of a real
# reference model, it inflated outer GMRES iterations by up to 30% and only won on wall-clock about
# half the time, netting only ~3% aggregate -- plausibly because the Chebyshev smoother's own
# power-iteration spectral-radius estimate then runs in float32 too. With backendBlockSize 3, "float"
# instead sets the hierarchy up in double and only stores the level matrices the cycle reads in float
# (see builtin_float_matrices in amgcl-wrapper.hpp): measured with outer iteration counts identical to
# the double hierarchy's on a captured damage-onset sequence, and about 15% faster.
#
# backendBlockSize defaults to 1 (scalar). A vector field of nodal dimension d maps to
# backendBlockSize: d (3 for a 3D displacement field, 2 for a 2D one). Setting it > 1 skips
# set_nullspace (see __call__) -- AMGCL's near-null-space path is unimplemented for block value
# types, so the hierarchy gets AMGCL's identity blocks: constants per component in the equilibrated
# variables, i.e. D-weighted translations rather than the translations D^-1 t of the scaled block
# (nullspace.py), and no rotational modes. Inside this block Gauss-Seidel preconditioner, 3x3 blocks
# stored in float with a smoothed-aggregation spectral-radius estimate are nevertheless the measured
# winner on the reference models (less index and value traffic per cycle), the recommended
# configuration of doc/source/documentation/blockamgtheory.rst -- the shipped default is unchanged.
#
# Both are set per field via fieldPreconds, e.g.
# {"displacement": {**_DEFAULT_VECTOR_PRECOND, "backendPrecision": "float", "backendBlockSize": 3}}.
# power_iters 300: AMGCL's default (0) bounds the spectral radius by Gershgorin's theorem, pessimistically; a
# power iteration is tighter but starts from a vector seeded per OpenMP thread, so it depends on the thread count.
# The former 50 iterations left it badly under-converged on hard operators at some thread counts (1460 instead of
# 43 outer iterations on the same matrix), 300 do not (see the Chebyshev section of the theory page).
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


#: The least excess over the reference outer iteration count that makes reused hierarchies stale, on top of
#: ``hierarchyStalenessFactor`` -- see :meth:`BlockAMGSolver._hierarchyIsStale`.
_STALENESS_MIN_EXCESS = 15


#: The deprecated LGMRES recycling keys and the one value each that is still accepted -- the behaviour that
#: is left (every solve starts from an empty augmentation subspace).
_DEPRECATED_LGMRES_KEYS = {"lgmresAlwaysReset": True, "lgmresResetOnNewIncrement": False}


def _checkLgmresRecyclingKeys(lgmresAlwaysReset: bool, lgmresResetOnNewIncrement: bool) -> None:
    """Raise for a value of the deprecated LGMRES recycling keys that asks for recycling."""
    for name, value in (
        ("lgmresAlwaysReset", lgmresAlwaysReset),
        ("lgmresResetOnNewIncrement", lgmresResetOnNewIncrement),
    ):
        if value != _DEPRECATED_LGMRES_KEYS[name]:
            raise ValueError(
                "blockamg: {:}={!r} asks for LGMRES Krylov recycling across solves, which was removed "
                "(measured to give nothing and to produce NaN on a live run). Drop the key, or set it to "
                "{!r}.".format(name, value, _DEPRECATED_LGMRES_KEYS[name])
            )


def _boolFromConfig(value) -> bool:
    """A JSON boolean -- or its spelling as a string, ``"true"``/``"false"`` -- and nothing else: ``bool("false")``
    is ``True``, so a plain ``bool`` cast would turn a quoted ``"false"`` into the opposite setting."""
    if isinstance(value, bool):
        return value
    if isinstance(value, str) and value.strip().lower() in ("true", "false"):
        return value.strip().lower() == "true"
    raise ValueError("expected true or false, got {!r}".format(value))


def _fieldNamesFromConfig(value) -> list | None:
    """A list of field names; a single name given as a plain string is that one name, not its characters, and
    ``null`` is ``None`` (unset)."""
    if value is None:
        return None
    if isinstance(value, str):
        return [value]
    if isinstance(value, (list, tuple)) and all(isinstance(name, str) for name in value):
        return list(value)
    raise ValueError("expected a field name or a list of field names, got {!r}".format(value))


def _mappingFromConfig(value) -> dict | None:
    """A JSON object, or ``null`` for unset -- not anything ``dict()`` happens to accept (``[]`` is no tree)."""
    if value is None or isinstance(value, dict):
        return value
    raise ValueError("expected a JSON object or null, got {!r}".format(value))


def _optional(cast):
    """``cast``, except that a JSON ``null`` stays ``None`` (unset) -- for the keys whose default is ``None``:
    ``str(None)`` would be the string ``"None"`` and ``int(None)`` raises."""

    def optionalCast(value):
        return None if value is None else cast(value)

    return optionalCast


def _outerTolFromConfig(value) -> float | None:
    """``outerTol`` unset, ``None``, or the literal string ``"adaptive"`` all select the adaptive forcing."""
    return None if value in (None, "adaptive") else float(value)


#: The conversion of every configuration file key from its JSON value to the constructor argument of
#: :class:`BlockAMGSolver`. Shared by the factory :func:`edelweissfe.linsolve.blockamg.createSolver` and
#: the hot reload, so that a value means the same thing at construction and when reloaded during a run.
CONFIG_CASTS = {
    "outerTol": _outerTolFromConfig,
    "outerRestart": int,
    "outerMaxiter": int,
    "outerSolver": str,
    "lgmresM": int,
    "lgmresK": int,
    "lgmresAlwaysReset": _boolFromConfig,
    "lgmresResetOnNewIncrement": _boolFromConfig,
    "sweeps": int,
    "symmetric": _boolFromConfig,
    "useRigidBodyNullspace": _boolFromConfig,
    "etaMin": float,
    "etaMax": float,
    "ewGamma": float,
    "ewAlpha": float,
    "residualGrowthFactor": float,
    "hierarchyStalenessFactor": float,
    "refreshOnNewIncrement": _boolFromConfig,
    "trueResidualMaxContinuations": int,
    "gapCompensatedTolerance": _boolFromConfig,
    "gapSafetyFactor": float,
    "verbosity": str,
    "warnOuterIterationsThreshold": int,
    "dumpOnDegradationDir": _optional(str),
    "dumpOnDegradationThreshold": _optional(int),
    "dumpOnDegradationMaxDumps": int,
    "dumpOnDegradationContextSolves": int,
    "hotReloadConfigFile": _optional(str),
    "hierarchyDropTol": float,
    "hierarchyDropLumping": _boolFromConfig,
    "gapMaxFactor": float,
    "fieldPreconds": _mappingFromConfig,
    "p1FieldNames": _fieldNamesFromConfig,
    "p1Nu": _optional(int),
    "p1NuPre": _optional(int),
    "p1NuPost": _optional(int),
    "p1FineDegree": _optional(int),
    "p1FineBlockSize": int,
    "p1FineValueType": str,
    "p1CoarseValueType": str,
    "p1FineSpectralRadiusReuse": _boolFromConfig,
    "p1CoarsePrecond": _mappingFromConfig,
}


#: The p-two-grid settings and the :class:`~edelweissfe.linsolve.blockamg.ptwogrid.PTwoGridPreconditioner` keyword
#: each one is.
_P1_OPTIONS = {
    "p1Nu": "nu",
    "p1NuPre": "nuPre",
    "p1NuPost": "nuPost",
    "p1FineDegree": "fineDegree",
    "p1FineBlockSize": "fineBlockSize",
    "p1FineValueType": "fineValueType",
    "p1CoarseValueType": "coarseValueType",
    "p1CoarsePrecond": "coarsePrecond",
}


class BlockAMGSolver(LinearSolver):
    """Field-split block-AMG preconditioned Krylov solve (LGMRES by default). Callable as ``(A, b) -> x``.

    The block structure is not configured here -- it is derived by the base class from the model and
    DOF manager the nonlinear solver hands over via
    :meth:`~edelweissfe.linsolve.base.LinearSolver.setModel`. A field's near null-space is decided
    from its nodal dimension: a vector field (dimension > 1) gets its full rigid-body basis
    (translations and rotations) when node coordinates are available, translations alone otherwise;
    a scalar field takes the default constant.

    Stateful across calls in two independent ways. This solver, like every other registered linear solver, sees
    only ``(A, b)`` per call, so it reconstructs the Newton-loop state rather than being told about it: from
    ``||b||`` (the forcing ratio, and residual jumps marking a new increment) and from its own bookkeeping (the
    field layout, the stored nonzeros, the outer iteration counts):

    #. **Adaptive outer tolerance** (Eisenstat--Walker forcing, "choice 2"), the default. Most Newton
       iterates do not need the linear solve tight, and loosening the requested tolerance for them cuts
       outer iterations substantially -- provided the outer solve's own stopping test (on the scaled
       residual) is not allowed to silently let the *true* residual run looser than the requested ``eta``
       (a real risk with an imperfect preconditioner, see the true-residual enforcement below). It does not
       reproduce the Newton path of a tight fixed tolerance exactly: through a damage onset, Newton needs
       linear solves about ten times tighter than the default ``etaMax`` allows (see the recommended
       configuration of the theory page). Pass ``outerTol=<a float>`` to pin a fixed tolerance instead.
    #. **Per-field AMG hierarchy reuse across Newton iterations**, on by default. Building a hierarchy
       is a large, avoidable fraction of a solve when the Jacobian has moved only a little since the
       last one. The outer Krylov solve always operates on the current matrix; only the block
       preconditioner ``M`` may be stale. A stale ``M`` costs outer iterations and, if it drives a solve to
       the iteration cap, the accuracy of that one Newton step; the next solve is then rebuilt. Refreshed on a
       residual jump (a new increment), on a field-structure change (e.g. AMR), on any
       sparsity-pattern change (when the active contact or tie set changes, and on almost every Newton
       iteration while the multi-point condensation prunes exact zeros, ``pruneCondensedMatrixZeros``, which
       then defeats reuse), or when the previous solve's outer count grew past
       ``hierarchyStalenessFactor`` times (and 15 more than) that of the solve the hierarchies were built
       for, or hit the iteration cap.

    Every solve also enforces the requested tolerance on the *true* residual, not just the outer solve's
    own stopping test -- see ``trueResidualMaxContinuations``. Both outer solvers converge the residual of the
    system they iterate on, the *scaled* (equilibrated) one: AMGCL's right-preconditioned LGMRES tests it
    directly, and SciPy's left-preconditioned GMRES checks it at the end of every restart cycle. The scaled
    residual can be considerably smaller than the true one (stiff rows are down-weighted by the
    equilibration), so "converged" could otherwise mean a true residual well above ``eta`` (measured on a
    real reference model: a 1.6e-2 true residual when 1e-4 was requested).

    Parameters
    ----------
    outerTol
        A fixed outer relative tolerance, overriding Eisenstat--Walker forcing. ``None`` (the
        default) uses adaptive forcing instead -- see ``etaMin``/``etaMax`` below.
    outerRestart, outerMaxiter
        With ``outerSolver = "scipy"``, the GMRES restart length and the number of restart cycles. With AMGCL's
        LGMRES their product is the iteration cap of one pass (its restart cycle is ``lgmresM + lgmresK``); the
        recommended configuration sets the cap of 50 as 50 x 1.
    outerSolver
        ``"amgcl_lgmres"`` (default) uses AMGCL's own native ``amgcl::solver::lgmres`` at both
        outer-solve call sites (the main solve and the true-residual continuation retry below) -- see
        ``edelweissfe/linsolve/amgcl/amgcl-wrapper.hpp``'s ``LGMRESOuterSolverT`` for why: SciPy's own
        GMRES orchestration (Arnoldi/Gram-Schmidt/restart bookkeeping, all run from Python) was found
        to be the single largest unaddressed cost bucket on the outer loop of a real reference model
        (roughly 38% of solve wall-clock). Live-gated at multiple thread counts: trajectory-identical
        to the SciPy path, no NaN (once cross-solve recycling -- see below -- was removed), and
        increasingly faster than SciPy as thread count grows -- SciPy's own orchestration anti-scales
        past one NUMA node on multi-socket hardware, AMGCL's native implementation does not. ``"scipy"``
        uses ``scipy.sparse.linalg.gmres`` instead, kept as a fallback/opt-out, not removed.
    lgmresM, lgmresK
        Only used when ``outerSolver == "amgcl_lgmres"``: forwarded to AMGCL's own ``lgmres::params``
        fields ``M`` (Krylov steps per restart cycle, default ``30``) and ``K`` (augmentation vectors
        carried from one restart cycle to the next within a solve, default ``3``); a cycle is ``M + K``
        iterations, each one preconditioner application, and the first cycle of a solve is plain
        GMRES(``M + K``). Every solve starts
        from an empty augmentation subspace: carrying it across solves ("recycling") was measured to give
        nothing on captured systems (bit-identical iteration counts) and, on a live run, to compound a
        struggling solve's poorly conditioned subspace into a following solve needing 428 instead of 159
        iterations and once an outright NaN -- so it was removed.
    lgmresAlwaysReset, lgmresResetOnNewIncrement
        Deprecated, kept accepted for existing configuration files: ``True`` / ``False`` respectively --
        the only behaviour left -- are accepted (a configuration file setting them gets a deprecation
        note); a value asking for recycling raises ``ValueError``.
    sweeps
        Block Gauss-Seidel sweeps per preconditioner application ``v = M^-1 r``, each field correcting its
        part of ``v`` by one cycle of its own hierarchy on its current residual (``v_i += M_i^-1 (r_i - sum_j
        A_ij v_j)``).
    symmetric
        If True, each sweep is followed by a reverse-order sweep, starting one field before the one
        just updated (with exact field solves it has nothing left to correct; with AMG cycles the skip saves
        one cycle of that field). Not required by the GMRES-type outer solve (right preconditioning only needs a
        fixed linear operator).
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
        opt-in two-grid variant -- see the module docstring of
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
        ``self._model``) and cached until the model changes (:meth:`setModel`). Ignored for a field also present in
        ``p1Maps`` (that field's map is already known; nothing to compute).
    p1Nu, p1FineDegree, p1CoarsePrecond
        The p-two-grid fine Chebyshev sweeps before and after the coarse correction, the fine
        Chebyshev degree, and the coarse level's AMGCL preconditioner tree. ``None`` (the default)
        keeps :mod:`~edelweissfe.linsolve.blockamg.ptwogrid`'s own defaults (1, 5, and smoothed
        aggregation with a degree-4 Chebyshev smoother).
    p1NuPre, p1NuPost
        Separate p-two-grid pre- and post-smoothing sweep counts, each defaulting to ``p1Nu`` --
        e.g. ``0`` / ``1`` for a cycle that starts with the coarse-grid correction and ends on one
        (higher-degree) smoothing sweep; see
        :class:`~edelweissfe.linsolve.blockamg.ptwogrid.PTwoGridPreconditioner`.
    p1FineBlockSize
        Storage of the p-two-grid fine operator: ``1`` (scalar, default) or ``3`` (node-wise 3x3
        blocks, 3D fields only) -- the same cycle up to summation order, with roughly half the memory
        traffic in the fine smoother and residual, the dominant cost of the apply.
    p1FineValueType
        Value type of the p-two-grid fine operator for the smoother and residual: ``"double"``
        (default) or ``"float"`` -- half the bytes per stored entry in the dominant cost of an outer
        iteration. Vectors and accumulation stay double, and the spectral radius is estimated on the
        double operator; see :class:`~edelweissfe.linsolve.blockamg.ptwogrid.PTwoGridPreconditioner`.
    p1CoarseValueType
        Value type of the p-two-grid coarse AMGCL hierarchy: ``"double"`` (default) or ``"float"``
        (AMGCL's single-precision backend, setup included).
    p1FineSpectralRadiusReuse
        Estimate the p-two-grid fine Chebyshev smoother's spectral radius in full (300 power
        iterations, the most expensive stage of its build) only on a field's first build, and on
        every later rebuild only re-check it: 15 power iterations continued from the previous
        build's dominant vector, raising the radius if the operator stiffened by more than 1%, and re-estimating it
        in full if it stiffened beyond the smoother's 10% safety margin. The cache is dropped when
        the model changes (``setModel``, e.g. after AMR) or a hot reload applies. ``False``
        (default) estimates it in full on every build.
    hierarchyDropTol
        Build the AMG hierarchies from a *sparsified* copy of each diagonal block: drop off-diagonal
        ``a_ij`` where ``|a_ij| < hierarchyDropTol * sqrt(|a_ii| |a_jj|)``. Zero (the default) keeps the
        block as-is.

        Why this exists. The RKPM operators this solver is used on store ~1643 entries per row at 43,350
        DOF (3.79% dense) while the *median* off-diagonal is **1.29e-06 of the diagonal** -- numerically
        far sparser than structurally. Smoothed aggregation then has to weigh 1643 candidates per row
        where a handful matter, and the Galerkin product carries the rest down to every coarse level.
        Measured on such a matrix: ``1e-6`` keeps 50.8% of the entries, ``1e-4`` keeps 29.6%, ``1e-3``
        keeps 17.4%, and no row loses all of its off-diagonals even at ``1e-2``.

        Only the *preconditioner* is sparsified; the operator the Krylov method applies, and therefore
        the residual it drives to zero, is the unmodified matrix. So this cannot change the converged
        solution beyond the outer tolerance -- it can only change how many iterations reaching it takes.

        The criterion is scale-invariant and, for a symmetric matrix, symmetric in ``(i, j)``, so the
        sparsified block stays symmetric -- which smoothed aggregation and the Chebyshev smoother both
        assume. Verified: zero asymmetric mask entries at 1e-6, 1e-4 and 1e-2.
    hierarchyDropLumping
        When dropping entries (see ``hierarchyDropTol``), add each discarded off-diagonal onto its row's
        diagonal instead of discarding it outright. This preserves row sums exactly, so
        ``A_filtered @ 1 == A @ 1`` and the constant near-null-space vector survives filtering -- which
        is what the AMG literature's "filtered matrix" construction does, and the reason it is preferred
        over plain truncation.

        Note the caveat for elasticity: off-diagonals are typically negative, so lumping *reduces* the
        diagonal and therefore weakens diagonal dominance. Whether that helps or hurts is a property of
        the operator, which is why this is a separate switch rather than folded into
        ``hierarchyDropTol``. Defaults to ``False``, i.e. plain truncation, which is what the published
        measurements for this solver used.
    etaMin, etaMax
        Clamp on the Eisenstat--Walker forcing tolerance (ignored if ``outerTol`` is given). ``etaMax``
        is also the tolerance used whenever there is no residual history to base a ratio on (the first
        solve, or the one right after a detected new increment).
    ewGamma, ewAlpha
        The Eisenstat--Walker "choice 2" parameters: ``eta_k = gamma * (||b_k|| / ||b_{k-1}||) **
        alpha``. Defaults ``gamma = 0.9``, ``alpha = (1 + sqrt 5) / 2`` are the classic values.
    residualGrowthFactor
        A solve whose ``||b||`` exceeds this multiple of the previous solve's ``||b||`` is treated as
        the first solve of a new increment: the forcing tolerance resets to ``etaMax`` and the AMG
        hierarchies are refreshed rather than reused. A guess with a blind spot: the retry after a cutback
        of a *diverged* attempt usually starts below that attempt's last residual and is not recognized
        (see the theory page).
    gapCompensatedTolerance, gapSafetyFactor
        Opt-in (default ``False``, i.e. unchanged behaviour). The outer Krylov converges the
        *scaled* (equilibrated) residual while acceptance is on the *true* relative residual; the
        ratio between them (the "gap", measured 1.0-3.5 on a gradient-enhanced damage model and up to about 40
        on a contact model with multi-point constraints) is why ~99% of production solves need
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

    gapMaxFactor
        Upper clamp on the gap EMA (default 1e3, well above the measured gaps of up to about 40). Without
        it, one non-converged solve can report an outsized gap that compounds through the EMA into
        an ever-tighter, unmeetable tolerance for every solve after it.
    hierarchyStalenessFactor
        Refresh the AMG hierarchies before the *next* solve if this solve's outer iteration count exceeded
        both this factor times, and 15 more than, the count of the solve the hierarchies were built for
        -- a growing count is the signal that the reused hierarchies are drifting away from the current
        Jacobian; measured against that reference rather than the previous solve, a slow drift is not
        missed. A solve that hit the iteration cap always refreshes them (a missed tolerance does not:
        it reflects the scaled-to-true residual gap, which a rebuild cannot close). With an iteration cap
        ``c`` and no continuations, a solve can exceed ``f`` times the reference only while the reference is
        below ``c / f``: with the recommended cap of 50 and factor 3, a reference of 17 or more leaves the cap
        hits as the only rebuild trigger of this kind.
    refreshOnNewIncrement
        Refresh the AMG hierarchies whenever a residual jump marks a new increment
        (default ``True``). With ``False`` a new increment alone keeps the standing hierarchies -- the
        first Newton iteration's Jacobian is the converged one of the previous increment -- and only
        the layout/pattern checks and ``hierarchyStalenessFactor`` decide.
    trueResidualMaxContinuations
        How many times to warm-restart the outer solve (``x0`` from the previous pass, the same ``A_scaled``,
        ``b_scaled`` and ``M``, a tighter tolerance) if the true relative residual still exceeds ``eta`` after
        a pass, whether that pass converged its own (scaled) residual or stopped at the iteration cap. ``0``
        disables this: the outer solve's own stopping test on the scaled residual decides.
    verbosity
        One of ``"silent"``, ``"warning"`` (default), ``"info"``, ``"debug"``. Messages go through the
        injected Journal (:meth:`~edelweissfe.linsolve.base.LinearSolver.setJournal`), falling back to
        ``print`` if none was set. ``"warning"`` emits nothing on a normal solve and only speaks up on
        an abnormal one: outer iterations past ``warnOuterIterationsThreshold``, or the true-residual
        tolerance still unmet after every continuation (saying so if the outer solve stopped at its cap).
        ``"info"`` adds one compact line per solve; ``"debug"`` adds one line per true-residual
        continuation attempt too. Per-stage wall-clock (equilibration, equilibrated system, hierarchy
        build, outer Krylov solve, continuations) is recorded via
        :mod:`edelweissfe.utils.performancetiming` regardless of verbosity, nested under "linear solve"
        in the job's own performance table.
    warnOuterIterationsThreshold
        A solve needing more outer Krylov iterations than this triggers a ``"warning"``-level message (a
        preconditioner-quality red flag), shown at every verbosity but ``"silent"``. A solve that misses
        ``eta`` has a warning of its own, whatever this threshold.
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
        the same solves. A solve that stops at the iteration cap and misses ``eta`` is dumped too, whatever the
        threshold.
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
        solve's own state -- a reused, now-stale AMG hierarchy, or (with the former cross-solve recycling)
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
    hotReloadConfigFile
        Path to this solver's own JSON config file. When set, every solve checks whether the file
        changed on disk and, if it did, re-reads it and applies the recognized settings, forcing a
        hierarchy rebuild. Unset (the default) means the settings are fixed at construction.

        This exists to make solver settings testable against a *live* run. A degraded AMG hierarchy
        typically only appears tens of hours into a nonlinear analysis, and rebuilding that state to
        try one parameter is the expensive part of every tuning round -- so the parameter can now be
        changed in place instead, with the log recording which settings were in force from which
        solve onwards.

        Editing a file a running process reads is inherently racy, so the reload is built to fail
        safely: the file is parsed *before* its modification stamp is accepted, so the half-written
        state an editor briefly leaves behind is retried on the next solve rather than recorded as
        seen, and a missing, unreadable or malformed config is reported while the settings in force
        are kept. Nothing about a bad edit can end the run -- which matters precisely because the
        runs worth using this on are the ones that were expensive to reach.

        Values the run could not use -- an unknown ``outerSolver``, a non-positive iteration count, an
        invalid p-two-grid option, a recycling request, a dump directory that cannot be created -- are
        rejected and reported the same way. An applied ``outerSolver``/``lgmresM``/``lgmresK`` rebuilds
        the outer LGMRES solver, ``dumpOnDegradationDir`` is created, and ``dumpOnDegradationContextSolves``
        resizes the context window.

        Unrecognized keys are reported rather than ignored, and so are recognized ones that cannot
        be applied after construction.
    """

    #: See :attr:`~edelweissfe.linsolve.base.LinearSolver.identification`.
    identification = _IDENTIFICATION

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
        hierarchyDropTol: float = 0.0,
        hierarchyDropLumping: bool = False,
        fieldPreconds: dict = None,
        useRigidBodyNullspace: bool = True,
        p1Maps: dict = None,
        p1FieldNames: list = None,
        p1Nu: int = None,
        p1NuPre: int = None,
        p1NuPost: int = None,
        p1FineDegree: int = None,
        p1FineBlockSize: int = 1,
        p1FineValueType: str = "double",
        p1CoarseValueType: str = "double",
        p1FineSpectralRadiusReuse: bool = False,
        p1CoarsePrecond: dict = None,
        etaMin: float = 1.0e-6,
        etaMax: float = 3.0e-4,
        ewGamma: float = 0.9,
        ewAlpha: float = 1.618033988749895,
        residualGrowthFactor: float = 4.0,
        hierarchyStalenessFactor: float = 1.5,
        refreshOnNewIncrement: bool = True,
        gapCompensatedTolerance: bool = False,
        gapSafetyFactor: float = 0.3,
        gapMaxFactor: float = 1e3,
        trueResidualMaxContinuations: int = 2,
        verbosity: str = "warning",
        warnOuterIterationsThreshold: int = 100,
        dumpOnDegradationDir: str = None,
        dumpOnDegradationThreshold: int = None,
        dumpOnDegradationMaxDumps: int = 10,
        dumpOnDegradationContextSolves: int = 0,
        hotReloadConfigFile: str = None,
    ):
        self._outerTol = outerTol
        self._outerRestart = outerRestart
        self._outerMaxiter = outerMaxiter
        if outerSolver not in ("scipy", "amgcl_lgmres"):
            raise ValueError("outerSolver must be 'scipy' or 'amgcl_lgmres', got {!r}".format(outerSolver))
        self._outerSolver = outerSolver
        self._lgmresM = lgmresM
        self._lgmresK = lgmresK
        _checkLgmresRecyclingKeys(lgmresAlwaysReset, lgmresResetOnNewIncrement)
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
        # Per-field fine spectral radius and dominant vector of the p-two-grid smoother, kept for the next build
        # only with p1FineSpectralRadiusReuse; dropped in setModel() like the caches above.
        self._p1FineSpectralRadiusCache = {}
        self._p1FieldNamesRequested = set(_fieldNamesFromConfig(p1FieldNames) or [])
        self._p1Nu = p1Nu
        self._p1NuPre = p1NuPre
        self._p1NuPost = p1NuPost
        self._p1FineDegree = p1FineDegree
        self._p1FineBlockSize = p1FineBlockSize
        self._p1FineValueType = p1FineValueType
        self._p1CoarseValueType = p1CoarseValueType
        self._p1FineSpectralRadiusReuse = p1FineSpectralRadiusReuse
        self._p1CoarsePrecond = p1CoarsePrecond
        self._hierarchyDropTol = hierarchyDropTol
        self._hierarchyDropLumping = hierarchyDropLumping
        self._etaMin = etaMin
        self._etaMax = etaMax
        self._ewGamma = ewGamma
        self._ewAlpha = ewAlpha
        self._residualGrowthFactor = residualGrowthFactor
        self._hierarchyStalenessFactor = hierarchyStalenessFactor
        self._refreshOnNewIncrement = refreshOnNewIncrement
        self._gapCompensatedTolerance = gapCompensatedTolerance
        self._gapSafetyFactor = gapSafetyFactor
        self._gapMaxFactor = gapMaxFactor
        self._trueResidualMaxContinuations = trueResidualMaxContinuations
        if verbosity not in _VERBOSITY_LEVELS:
            raise ValueError("verbosity must be one of {:}, got {!r}".format(_VERBOSITY_LEVELS, verbosity))
        self._verbosityIndex = _VERBOSITY_LEVELS.index(verbosity)
        self._warnOuterIterationsThreshold = warnOuterIterationsThreshold

        self._dumpOnDegradationDir = dumpOnDegradationDir
        # None: the warning threshold, resolved where it is used, so that a reloaded warning threshold applies
        self._dumpOnDegradationThreshold = dumpOnDegradationThreshold
        self._dumpOnDegradationMaxDumps = dumpOnDegradationMaxDumps
        self._dumpOnDegradationContextSolves = dumpOnDegradationContextSolves

        # Hot reload: the path to watch, and the file contents it was last applied from. Seeded as
        # None rather than from the file, so the first solve reads and reports the settings actually
        # in force -- which is the whole point when the file is being edited during a long run.
        self._hotReloadConfigFile = hotReloadConfigFile
        self._hotReloadStamp = None
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

        # Eisenstat-Walker forcing state.
        self._lastResidualNorm = None
        self._lastEta = etaMax

        # Reused hierarchy state: the built per-field AMG hierarchies, the equilibration they were
        # built for, and the field-block layout they assume -- all None until the first solve.
        self._preconditioners = None
        self._scaling = None
        self._blocks = None
        self._n = None
        self._lastNnz = None
        self._lastOuterIters = None
        # outer iterations of the solve the standing hierarchies were built for (see _hierarchyIsStale)
        self._stalenessReferenceIters = None
        self._lastContinuations = None
        # cached scaled->true relative-residual gap, used to pre-compensate the first pass.
        self._trueResidualGap = 1.0
        self._refreshNext = False

        # The persistent AMGCL lgmres outer-solver instance (only used when outerSolver ==
        # "amgcl_lgmres"), and the problem size it was built for. Unlike the per-field AMG
        # hierarchies above, it is not rebuilt when mustRefresh fires: it holds no state of a solve,
        # only AMGCL's scratch vectors preallocated for a fixed size. It is rebuilt on a size change
        # (see __call__) and when its own settings change (hot reload).
        self._lgmresSolver = None
        self._lgmresN = None

        self._lastSolveSummary = None

        # Settings applied by a hot reload, with the values they replaced, until a hierarchy build has accepted
        # them -- see _maybeHotReloadConfig.
        self._hotReloadRollback = []

        # The checks a hot reload applies, so that a value the run cannot use fails here rather than in the
        # first solve or hierarchy build. Those against the fields' dimensions follow at the first build.
        rejections = self._settingsRejections(self._settingsInForce(), {})
        if rejections:
            raise ValueError(
                "blockamg: " + "; ".join("{:} {:}".format(name, reason) for name, reason in rejections.items())
            )

    @property
    def reportsSolveSummary(self) -> bool:
        """Whether the nonlinear solver should render a "linear solve" column for this instance.

        Tied to this instance's own verbosity rather than a bare ``True`` -- at ``"silent"`` or
        ``"warning"`` (the default) there is nothing routine to show, matching those settings' own
        contract (see the ``verbosity`` parameter docstring above); only at ``"info"`` or ``"debug"``
        does :attr:`lastSolveSummary` carry anything worth a column.
        """
        return self._verbosityIndex >= _VERBOSITY_LEVELS.index("info")

    @property
    def lastSolveSummary(self) -> "LinearSolveSummary | None":
        """The most recent call's diagnostics -- see
        :attr:`~edelweissfe.linsolve.base.LinearSolver.lastSolveSummary`. ``None`` before the first
        call."""
        return self._lastSolveSummary

    def _hierarchyIsStale(self, outerIters: int, rebuilt: bool, capHit: bool) -> bool:
        """Whether the standing hierarchies should be rebuilt before the next solve.

        The reference is the outer iteration count of the solve the hierarchies were built for (the first
        one after the last rebuild), not that of the previous solve: compared with its predecessor only, a
        slow drift (12, 17, 25, 37, 50) never trips a 1.5 factor. A rebuild needs the count to exceed both
        ``hierarchyStalenessFactor`` times that reference and the reference plus
        :data:`_STALENESS_MIN_EXCESS` -- at small counts a factor alone fires on noise (8 -> 13). A solve
        that hit the iteration cap always asks for a rebuild: its count says nothing about how far the
        hierarchies drifted, only that they no longer do the job -- also when that solve had just rebuilt them
        (not rebuilding again was measured neutral, see the theory page). A missed tolerance does not: with
        continuations off, it comes from the gap between the scaled residual the outer solve converges
        and the true one (stiff rows down-weighted by the equilibration), which a rebuild cannot close --
        measured +18% on a contact model with multi-point constraints when it did trigger a rebuild.
        """
        if rebuilt or self._stalenessReferenceIters is None:
            self._stalenessReferenceIters = outerIters
            drifted = False
        else:
            reference = self._stalenessReferenceIters
            drifted = outerIters > max(self._hierarchyStalenessFactor * reference, reference + _STALENESS_MIN_EXCESS)
        return drifted or capHit

    def _lgmresCapHit(self, iterations: int, tolerance: float) -> int:
        """``1`` if the last AMGCL LGMRES pass stopped at the iteration cap short of its own tolerance, else ``0``
        (SciPy's ``info`` convention).

        LGMRES stops on its own convergence test, its relative residual (``lastError``, recomputed at its exit,
        of the operator it iterated on -- the scaled one, not the true residual eta is accepted on) below the
        tolerance of that pass, or at the cap. Reaching the cap is not failure by itself: a pass may converge on
        exactly its last allowed iteration. Neither is comparing ``lastError`` with eta, the true-residual target.
        """
        capped = iterations >= self._outerRestart * self._outerMaxiter
        return 1 if capped and not self._lgmresSolver.lastError < tolerance else 0

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

    def _dropSmallEntries(self, block):
        """Return `block` with off-diagonals below the relative drop tolerance removed.

        Runs only on a hierarchy refresh, not on every solve, so a few array passes over nnz are cheap
        against the hierarchy build they feed. The diagonal is always kept, so no row can be emptied.
        """
        tau = self._hierarchyDropTol
        diagonal = np.abs(block.diagonal())
        coo = block.tocoo()
        scale = np.sqrt(diagonal[coo.row] * diagonal[coo.col])
        keep = (coo.row == coo.col) | (np.abs(coo.data) >= tau * scale)
        dropped = sp.coo_matrix((coo.data[keep], (coo.row[keep], coo.col[keep])), shape=block.shape).tocsr()

        lumped = ""
        if self._hierarchyDropLumping:
            # Row-sum-preserving filtering: the discarded content of each row goes onto its diagonal, so
            # A_filtered @ 1 == A @ 1 exactly and the constant vector is unaffected by filtering. The
            # diagonal is always kept, so this does not change the sparsity pattern.
            discardedRowSums = np.bincount(coo.row[~keep], weights=coo.data[~keep], minlength=block.shape[0])
            dropped = (dropped + sp.diags(discardedRowSums, format="csr")).tocsr()
            lumped = ", lumped onto the diagonal"

        self._log(
            "info",
            "blockamg: hierarchy drop tol {:.0e}: nnz {:,} -> {:,} ({:.1f}%){:}".format(
                tau, block.nnz, dropped.nnz, 100.0 * dropped.nnz / max(block.nnz, 1), lumped
            ),
        )
        return dropped

    def _forcingTolerance(self, residualNorm: float, newIncrement: bool) -> float:
        """The Eisenstat--Walker "choice 2" forcing tolerance for this solve, clamped and safeguarded.

        ``eta_k = gamma (||b_k|| / ||b_{k-1}||) ** alpha``, with the classic safeguard against
        over-solving (a large tightening step is only trusted if the previous tolerance was already
        small), clamped to ``[etaMin, etaMax]``. The safeguard only acts while ``gamma eta_{k-1} ** alpha`` exceeds
        0.1, i.e. for a previous tolerance above about 0.26 at the classic parameters: never with an ``etaMax``
        below that. Falls back to ``etaMax`` with no history to compare
        against (the first solve, or the one right after a new-increment jump -- the ratio
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

        The standing hierarchies are rebuilt on the next solve: they were built from the dropped caches
        (node coordinates for the near null-space, P1 maps), and a new model with an unchanged field
        layout and sparsity pattern would otherwise keep them. The residual history is dropped too: the
        residual norm of the previous model would otherwise enter the forcing ratio and the new-increment
        test of the first solve on this one.
        """
        super().setModel(model, dofManager)
        self._dropDerivedCaches()
        self._refreshNext = True
        self._lastResidualNorm = None

    def _dropDerivedCaches(self) -> None:
        """Drop what the hierarchy builds derived from the model and the settings: the node coordinates, the P1
        topology maps computed from the model (not the ones supplied as ``p1Maps``), and the cached p-two-grid
        spectral radii. Called when either changes -- :meth:`setModel` and an applied hot reload."""
        for fieldName in self._lazyP1MapNames:
            self._p1Maps.pop(fieldName, None)
        self._lazyP1MapNames.clear()
        self._nodeCoordinateCache.clear()
        self._p1FineSpectralRadiusCache.clear()

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
        """This field's P1 corner/midside topology map, computed lazily on first need and cached until
        the model changes (:meth:`setModel`, e.g. after adaptive refinement, drops every computed map) or
        a hot reload applies.

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

    def _dumpDegradedSystem(self, A: sp.csr_matrix, b: np.ndarray, blocks: list, solveState: dict) -> None:
        """Dump the triggering solve itself, then as much of its context window
        (``dumpOnDegradationContextSolves`` preceding solves, from :attr:`_recentSolveHistory`) as the
        remaining ``dumpOnDegradationMaxDumps`` budget allows, oldest first. ``solveState`` is the solve's
        bookkeeping (outer iterations, residuals, rebuild flags) recorded in the manifest.
        """
        triggerSolveCount = self._solveCount
        if triggerSolveCount not in self._dumpedSolveCounts and (
            BlockAMGSolver._degradationDumpsWritten < self._dumpOnDegradationMaxDumps
        ):
            self._dumpOneSystem(
                triggerSolveCount, A, b, blocks, "trigger", triggerSolveCount=triggerSolveCount, stateFields=solveState
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
                "context",
                triggerSolveCount=triggerSolveCount,
                stateFields=entry["state"],
            )

    # Settings whose stored attribute does not follow the self._<name> convention. Listed
    # explicitly rather than inferred, because a wrong guess here would apply a setting to nothing
    # and report success while doing so.
    _HOT_RELOAD_ALIASES = {
        "verbosity": "_verbosityIndex",
        "p1FieldNames": "_p1FieldNamesRequested",
    }
    # Changing which file is watched, from inside that file, is not a thing worth supporting.
    # The deprecated LGMRES recycling keys have no setting left to change. p1Maps are arrays handed to the
    # constructor, not a configuration file setting.
    _HOT_RELOAD_IGNORED = ("hotReloadConfigFile", "lgmresAlwaysReset", "lgmresResetOnNewIncrement", "p1Maps")

    def _fieldDimensions(self) -> dict:
        """The nodal dimension of every field of the last hierarchy build, by name; empty before the first."""
        return {block.name: block.dimension for block in self._blocks or []}

    def _settingsInForce(self) -> dict:
        """The settings :meth:`_settingsRejections` checks, as they are in force, by configuration key."""
        return {
            "outerTol": self._outerTol,
            "outerSolver": self._outerSolver,
            "outerRestart": self._outerRestart,
            "outerMaxiter": self._outerMaxiter,
            "lgmresM": self._lgmresM,
            "lgmresK": self._lgmresK,
            "sweeps": self._sweeps,
            "etaMin": self._etaMin,
            "etaMax": self._etaMax,
            "ewGamma": self._ewGamma,
            "ewAlpha": self._ewAlpha,
            "residualGrowthFactor": self._residualGrowthFactor,
            "hierarchyStalenessFactor": self._hierarchyStalenessFactor,
            "trueResidualMaxContinuations": self._trueResidualMaxContinuations,
            "gapSafetyFactor": self._gapSafetyFactor,
            "gapMaxFactor": self._gapMaxFactor,
            "warnOuterIterationsThreshold": self._warnOuterIterationsThreshold,
            "dumpOnDegradationDir": self._dumpOnDegradationDir,
            "dumpOnDegradationThreshold": self._dumpOnDegradationThreshold,
            "dumpOnDegradationMaxDumps": self._dumpOnDegradationMaxDumps,
            "dumpOnDegradationContextSolves": self._dumpOnDegradationContextSolves,
            "fieldPreconds": self._fieldPreconds,
            "p1FieldNames": self._p1FieldNamesRequested,
            "p1Nu": self._p1Nu,
            "p1NuPre": self._p1NuPre,
            "p1NuPost": self._p1NuPost,
            "p1FineDegree": self._p1FineDegree,
            "p1FineBlockSize": self._p1FineBlockSize,
            "p1FineValueType": self._p1FineValueType,
            "p1CoarseValueType": self._p1CoarseValueType,
            "p1CoarsePrecond": self._p1CoarsePrecond,
        }

    def _settingsRejections(self, settings: dict, dimensions: dict) -> dict:
        """Why a configuration cannot be used, as ``{key: reason}`` (empty if it can). ``settings`` is a whole
        configuration (:meth:`_settingsInForce`, with a hot reload's changes merged in), so that keys that only
        fail together -- ``p1FineBlockSize`` 3 with a 2D field in ``p1FieldNames`` -- are caught together;
        such a reason is given for each of its keys. ``dimensions`` are the fields' nodal dimensions by name,
        empty before the fields are known. Checked at construction, on every hot reload, and when a build first
        sees the fields: what would otherwise only fail in the middle of a solve or a hierarchy build.

        The generic apply of a hot reload still goes through ``hasattr``/``setattr`` on the ``_<name>``
        attribute convention; an explicit per-key table is left for the configuration refactor.
        """
        from edelweissfe.linsolve.blockamg.ptwogrid import parameterRejection

        rejections = {}

        def reject(reason: str, *keys: str) -> None:
            for key in keys:
                rejections.setdefault(key, reason)

        if settings["outerSolver"] not in ("scipy", "amgcl_lgmres"):
            reject("must be 'scipy' or 'amgcl_lgmres'", "outerSolver")
        for key in ("lgmresM", "outerRestart", "outerMaxiter", "sweeps"):
            if settings[key] < 1:
                reject("must be at least 1", key)
        for key in (
            "lgmresK",
            "trueResidualMaxContinuations",
            "warnOuterIterationsThreshold",
            "dumpOnDegradationMaxDumps",
            "dumpOnDegradationContextSolves",
        ):
            if settings[key] < 0:
                reject("must not be negative", key)
        if settings["dumpOnDegradationThreshold"] is not None and settings["dumpOnDegradationThreshold"] < 0:
            reject("must not be negative", "dumpOnDegradationThreshold")
        # written as "not (valid)" so that NaN is rejected too
        if settings["outerTol"] is not None and not 0.0 < settings["outerTol"] < 1.0:
            reject("needs 0 < outerTol < 1", "outerTol")
        if not 0.0 < settings["etaMin"] <= settings["etaMax"] < 1.0:
            reject("needs 0 < etaMin <= etaMax < 1", "etaMin", "etaMax")
        # the ranges of Eisenstat and Walker's choice 2
        if not 0.0 < settings["ewGamma"] <= 1.0:
            reject("needs 0 < ewGamma <= 1", "ewGamma")
        if not 1.0 < settings["ewAlpha"] <= 2.0:
            reject("needs 1 < ewAlpha <= 2", "ewAlpha")
        if not settings["residualGrowthFactor"] > 1.0:
            reject("must be greater than 1", "residualGrowthFactor")
        if not settings["hierarchyStalenessFactor"] >= 1.0:
            reject("must be at least 1", "hierarchyStalenessFactor")
        if not settings["gapSafetyFactor"] > 0.0:
            reject("must be positive", "gapSafetyFactor")
        if not settings["gapMaxFactor"] >= 1.0:
            reject("must be at least 1 (the gap is floored at 1)", "gapMaxFactor")

        if settings["p1CoarsePrecond"] is not None and not isinstance(settings["p1CoarsePrecond"], dict):
            reject("must be an AMGCL preconditioner tree (a JSON object) or null", "p1CoarsePrecond")
        else:
            p1Options = {option: settings[key] for key, option in _P1_OPTIONS.items() if settings[key] is not None}
            reason = parameterRejection(**p1Options)
            if reason is not None:
                reject(reason, *_P1_OPTIONS)
        if settings["p1FineBlockSize"] == 3:
            notThreeDimensional = sorted(name for name in settings["p1FieldNames"] if dimensions.get(name, 3) != 3)
            if notThreeDimensional:
                reject(
                    "p1FineBlockSize 3 needs 3D fields, but the p-two-grid fields {:} are not".format(
                        notThreeDimensional
                    ),
                    "p1FineBlockSize",
                    "p1FieldNames",
                )

        reason = self._fieldPrecondsRejection(settings["fieldPreconds"], dimensions)
        if reason is not None:
            reject(reason, "fieldPreconds")

        directory = settings["dumpOnDegradationDir"]
        if directory is not None and not os.path.isdir(directory):
            # checked, not created: a rejected configuration must leave no trace
            existing = os.path.dirname(os.path.abspath(directory))
            while not os.path.isdir(existing):
                existing = os.path.dirname(existing)
            if os.path.exists(directory) or not os.access(existing, os.W_OK | os.X_OK):
                reject("cannot be created as a directory", "dumpOnDegradationDir")
        return rejections

    @staticmethod
    def _fieldPrecondsRejection(fieldPreconds, dimensions: dict) -> "str | None":
        """Why a ``fieldPreconds`` mapping cannot be used, or ``None`` -- see :meth:`_settingsRejections`. Each
        entry is checked for what the hierarchy build would fail on: the backend keys, and, once the fields are
        known, a block size that matches the field's nodal dimension. An entry for a field the system does not
        have is unused, not an error: one configuration file may serve several models."""
        if not isinstance(fieldPreconds, dict):
            return "must map field names to AMGCL preconditioner trees"
        for fieldName, precond in fieldPreconds.items():
            if not isinstance(precond, dict):
                return "entry '{:}' must be an AMGCL preconditioner tree (a JSON object)".format(fieldName)
            precision = precond.get("backendPrecision", "double")
            blockSize = precond.get("backendBlockSize", 1)
            if precision not in ("double", "float"):
                return "entry '{:}': backendPrecision must be 'double' or 'float'".format(fieldName)
            if blockSize not in (1, 2, 3) or (blockSize == 2 and precision == "float"):
                return "entry '{:}': backendBlockSize must be 1, 2 or 3 (2 only in double)".format(fieldName)
            if blockSize > 1 and fieldName in dimensions and dimensions[fieldName] != blockSize:
                return "entry '{:}': backendBlockSize {:} does not match the field's dimension {:}".format(
                    fieldName, blockSize, dimensions[fieldName]
                )
        return None

    def _degradationThreshold(self) -> int:
        """The outer-iteration count above which a solve is dumped: ``dumpOnDegradationThreshold``, or the
        warning threshold if that is unset."""
        if self._dumpOnDegradationThreshold is None:
            return self._warnOuterIterationsThreshold
        return self._dumpOnDegradationThreshold

    def _outerLgmres(self, n: int):
        """AMGCL's LGMRES for a system of size ``n``, created on demand: its scratch vectors are allocated for one
        size, so a size change (adaptive refinement) needs a new one, and so does a change of ``lgmresM``,
        ``lgmresK`` or ``outerSolver`` -- a hot reload, or the rollback of one, drops it for that."""
        if self._lgmresSolver is None or self._lgmresN != n:
            from edelweissfe.linsolve.amgcl.amgcl import PyAMGCLLGMRESSolver

            self._lgmresSolver = PyAMGCLLGMRESSolver(n, {"M": self._lgmresM, "K": self._lgmresK})
            self._lgmresN = n
        return self._lgmresSolver

    def _afterHotReload(self, name: str) -> None:
        """Whatever an applied setting needs beyond its attribute: objects built from it are rebuilt."""
        if name in ("outerSolver", "lgmresM", "lgmresK"):
            # rebuilt with the new settings on the next solve that needs it
            self._lgmresSolver = None
        elif name == "dumpOnDegradationDir" and self._dumpOnDegradationDir is not None:
            os.makedirs(self._dumpOnDegradationDir, exist_ok=True)
        elif name == "dumpOnDegradationContextSolves":
            self._recentSolveHistory = collections.deque(
                self._recentSolveHistory, maxlen=max(self._dumpOnDegradationContextSolves, 0)
            )

    def _rollBackHotReload(self, error: Exception) -> None:
        """Restore the settings the last hot reload replaced, after the hierarchy build failed with them: a
        value the validation cannot judge -- an AMGCL tree naming an unknown smoother, say -- must not end the
        run either. The file stays recorded as seen, so it is not applied again until it is edited."""
        lines = [
            "hot reload of {:} at solve #{:} ROLLED BACK, the hierarchy build failed with it: {:}".format(
                self._hotReloadConfigFile, self._solveCount, error
            )
        ]
        for name, attribute, oldValue in self._hotReloadRollback:
            setattr(self, attribute, oldValue)
            self._afterHotReload(name)
            lines.append("  restored {:}: {!r}".format(name, oldValue))
        self._hotReloadRollback = []
        self._dropDerivedCaches()
        self._log("warning", "\n".join(lines))

    def _nullspaceCoordinates(self, fieldName: str) -> "np.ndarray | None":
        """The node coordinates for a vector field's rigid-body near null-space (translations and rotations), or
        ``None`` for translations alone: with ``useRigidBodyNullspace`` off, or without coordinates (an offline
        probe driving this solver through :meth:`setFieldStructure`, with no model)."""
        return self._getNodeCoordinates(fieldName) if self._useRigidBodyNullspace else None

    def _buildFieldPreconditioners(self, system, blocks: list, scaling: np.ndarray) -> list:
        """One preconditioner per field, built for its equilibrated diagonal block. The first build for a field
        layout also checks the settings against the fields' dimensions, which construction could not."""
        if blocks != self._blocks:
            rejections = self._settingsRejections(
                self._settingsInForce(), {block.name: block.dimension for block in blocks}
            )
            if rejections:
                raise ValueError(
                    "blockamg: settings that do not fit the fields: "
                    + "; ".join("{:} {:}".format(name, reason) for name, reason in rejections.items())
                )
        with performancetiming.timeit("diagonal blocks"):
            diagonalBlocks = [system.diagonalBlock(i) for i in range(len(blocks))]
            if self._hierarchyDropTol > 0.0:
                diagonalBlocks = [self._dropSmallEntries(block) for block in diagonalBlocks]
        return [
            self._buildFieldPreconditioner(block, diagonalBlock, scaling[block.start : block.stop])
            for block, diagonalBlock in zip(blocks, diagonalBlocks)
        ]

    def _buildFieldPreconditioner(self, block, diagonalBlock, scaling: np.ndarray):
        """The preconditioner of one field: the p-two-grid cycle for a vector field that opted into it
        (``p1Maps`` / ``p1FieldNames``), one AMGCL hierarchy otherwise. A vector field's near null-space is its
        rigid-body modes (see :meth:`_nullspaceCoordinates`); a scalar field takes AMGCL's default constant."""
        isVectorField = block.dimension > 1
        coords = self._nullspaceCoordinates(block.name) if isVectorField else None

        p1Map = None
        if isVectorField and (block.name in self._p1Maps or block.name in self._p1FieldNamesRequested):
            with performancetiming.timeit("p1 topology"):
                p1Map = self._getP1Map(block.name)
        if p1Map is not None:
            from edelweissfe.linsolve.blockamg.ptwogrid import PTwoGridPreconditioner

            p1Options = {option: value for option, value in self._p1Options().items() if value is not None}
            fineSpectralRadius, fineDominantVector = (
                self._p1FineSpectralRadiusCache.get(block.name, (None, None))
                if self._p1FineSpectralRadiusReuse
                else (None, None)
            )
            preconditioner = PTwoGridPreconditioner(
                *p1Map, fineSpectralRadius=fineSpectralRadius, fineDominantVector=fineDominantVector, **p1Options
            )
            with performancetiming.timeit("{:} p-two-grid build".format(block.name)):
                preconditioner.build(diagonalBlock, scaling, coords=coords)
            if self._p1FineSpectralRadiusReuse:
                self._p1FineSpectralRadiusCache[block.name] = (
                    preconditioner.fineSpectralRadius,
                    preconditioner.fineDominantVector,
                )
            return preconditioner

        from edelweissfe.linsolve.amgcl.amgcl import PyAMGCLSolver

        precondParams = dict(
            self._fieldPreconds.get(block.name, _DEFAULT_VECTOR_PRECOND if isVectorField else _DEFAULT_SCALAR_PRECOND)
        )
        backendPrecision = precondParams.pop("backendPrecision", "double")
        backendBlockSize = precondParams.pop("backendBlockSize", 1)
        preconditioner = PyAMGCLSolver(
            {"precond": precondParams, "backendPrecision": backendPrecision, "backendBlockSize": backendBlockSize}
        )
        # A block backend cannot take a near null-space (AMGCL's own path is unimplemented for block value
        # types); a scalar backend gets the rigid-body modes -- the error components a Chebyshev smoother does
        # not reduce, which the coarse levels must represent.
        if isVectorField and backendBlockSize == 1:
            with performancetiming.timeit("nullspace construction"):
                if coords is not None:
                    nullspace = rigidBodyNullspace(block, coords, scaling)
                else:
                    nullspace = translationNullspace(block, scaling)
            preconditioner.set_nullspace(nullspace)
        with performancetiming.timeit("{:} AMG build".format(block.name)):
            preconditioner.build(diagonalBlock)
        return preconditioner

    def _p1Options(self) -> dict:
        """The p-two-grid settings as :class:`~edelweissfe.linsolve.blockamg.ptwogrid.PTwoGridPreconditioner`
        keywords (``None`` for those left at its defaults)."""
        settings = self._settingsInForce()
        return {option: settings[key] for key, option in _P1_OPTIONS.items()}

    def _maybeHotReloadConfig(self) -> bool:
        """Re-read the JSON config if it changed on disk; return whether anything was applied.

        Point ``hotReloadConfigFile`` at the config's own path to enable this. It exists so that
        solver settings can be tried against a *live* run: a degraded AMG hierarchy typically only
        appears tens of hours into a nonlinear analysis, and rebuilding that state to test one
        parameter is the expensive part of every tuning round.

        Anything applied forces a hierarchy refresh through ``_refreshNext``, and every derived
        cache (node coordinates, P1 topology, lazily built P1 maps) is dropped first, because those
        were built under the *old* settings.

        The file is parsed before its stamp is accepted, so a half-written file -- the normal
        transient state of an editor saving in place -- is retried on the next solve instead of
        being recorded as "seen". No failure here can end the run: a missing, unreadable or
        malformed config is reported and the settings in force are kept. That asymmetry is
        deliberate. The feature's whole purpose is to be used on a run that is expensive to have
        reached, so a typo in a config edit must never be able to destroy it.
        """
        if self._hotReloadConfigFile is None:
            return False

        # The contents themselves are the stamp, not (mtime, size): on kernels with coarse file
        # timestamps, two same-size saves within one clock tick share an mtime, and the second edit
        # would be skipped for good. The file is a few hundred bytes, read once per solve.
        try:
            with open(self._hotReloadConfigFile, "r") as configFile:
                stamp = configFile.read()
        except OSError as error:
            self._log("warning", "hot reload: cannot read {:}: {:}".format(self._hotReloadConfigFile, error))
            return False

        if stamp == self._hotReloadStamp:
            return False

        try:
            newOptions = json.loads(stamp)
            if not isinstance(newOptions, dict):
                raise ValueError("expected a JSON object, got {:}".format(type(newOptions).__name__))
        except ValueError as error:
            self._log(
                "warning",
                "hot reload: {:} is not readable as a JSON object ({:}); keeping the settings in "
                "force and retrying on the next solve".format(self._hotReloadConfigFile, error),
            )
            return False

        self._hotReloadStamp = stamp

        validNames = set(inspect.signature(type(self).__init__).parameters) - {"self"}

        # Convert and validate first, mutate second: the file's settings are checked as one configuration
        # together with the ones in force, and the caches are dropped only if something is applied.
        candidates, pending, rejected, unknown = {}, [], [], []
        for name, value in newOptions.items():
            if name in _DEPRECATED_LGMRES_KEYS:
                # nothing left to change, but a value asking for the removed recycling is an error
                try:
                    _checkLgmresRecyclingKeys(**{**_DEPRECATED_LGMRES_KEYS, name: CONFIG_CASTS[name](value)})
                except (TypeError, ValueError) as error:
                    rejected.append("{:} ({:})".format(name, error))
                continue
            if name in self._HOT_RELOAD_IGNORED:
                continue
            if name not in validNames:
                unknown.append(name)
                continue

            # The same conversion as at construction: the raw JSON value can differ in type or even
            # meaning from the argument it stands for (``"outerTol": "adaptive"`` is ``None``).
            if name in CONFIG_CASTS:
                try:
                    value = CONFIG_CASTS[name](value)
                except (TypeError, ValueError) as error:
                    rejected.append("{:} ({:})".format(name, error))
                    continue

            if name == "verbosity":
                if value not in _VERBOSITY_LEVELS:
                    rejected.append("{:} (must be one of {:})".format(name, _VERBOSITY_LEVELS))
                    continue
                newValue = _VERBOSITY_LEVELS.index(value)
            elif name == "p1FieldNames":
                newValue = set(value or [])
            elif name == "fieldPreconds":
                newValue = value or {}
            else:
                newValue = value
            candidates[name] = newValue

        configuration = self._settingsInForce()
        configuration.update({name: value for name, value in candidates.items() if name in configuration})
        rejections = self._settingsRejections(configuration, self._fieldDimensions())
        for name, newValue in candidates.items():
            if name in rejections:
                rejected.append("{:} ({:})".format(name, rejections[name]))
                continue

            attribute = self._HOT_RELOAD_ALIASES.get(name, "_" + name)
            if not hasattr(self, attribute):
                rejected.append("{:} (recognized at construction but not hot-reloadable)".format(name))
                continue

            oldValue = getattr(self, attribute)
            if newValue != oldValue:
                pending.append((name, attribute, oldValue, newValue))

        if pending:
            self._dropDerivedCaches()
            for name, attribute, _, newValue in pending:
                setattr(self, attribute, newValue)
                self._afterHotReload(name)
            # kept until a hierarchy build has accepted the new settings, see _rollBackHotReload
            self._hotReloadRollback = [(name, attribute, oldValue) for name, attribute, oldValue, _ in pending]
            self._refreshNext = True

        # Reported at "warning" so it lands in the log of a run whose verbosity is not raised: what
        # settings were in force from which solve onwards is the one thing a later reader of a
        # hot-reloaded run cannot reconstruct otherwise.
        if pending or rejected or unknown:
            summary = ["hot reload of {:} at solve #{:}".format(self._hotReloadConfigFile, self._solveCount)]
            for name, _, oldValue, newValue in pending:
                summary.append("  applied  {:}: {!r} -> {!r}".format(name, oldValue, newValue))
            for entry in rejected:
                summary.append("  REJECTED {:}".format(entry))
            for name in unknown:
                summary.append("  UNKNOWN  {:} (not a BlockAMGSolver setting)".format(name))
            if not pending:
                summary.append("  no effective change; hierarchies kept")
            else:
                summary.append("  hierarchies will be rebuilt on this solve")
            self._log("warning", "\n".join(summary))

        return bool(pending)

    def __call__(self, A, b):
        solveStartTime = time.time()
        from edelweissfe.linsolve.amgcl.amgcl import PyAMGCLEquilibratedSystem

        self._solveCount += 1
        self._maybeHotReloadConfig()
        A = A.tocsr()
        if not A.has_canonical_format:
            # AMGCL's block backends read the columns as sorted, and no backend sums duplicate entries: a
            # canonical copy, leaving the caller's matrix as it is
            A = A.copy()
            A.sum_duplicates()
        n = A.shape[0]
        blocks = self._resolveBlocks(n)
        slices = [slice(block.start, block.stop) for block in blocks]
        b = np.asarray(b).reshape(n)

        # (1) Newton-state detection: a residual jump marks a new increment (not a cutback after a diverged
        #     attempt, whose retry starts below its last residual -- see residualGrowthFactor)
        residualNorm = float(np.linalg.norm(b))
        if not np.isfinite(residualNorm):
            # Nothing to solve: the outer solve would spend its whole iteration cap on it, and the norm would
            # pin the next forcing tolerance. The NaN correction reaches the nonlinear solver as a diverged
            # solution, and this solver's state is left as it was.
            self._log(
                "warning",
                "⚠ solve #{:}: the right-hand side is not finite; returning NaN without solving.".format(
                    self._solveCount
                ),
            )
            self._lastSolveSummary = LinearSolveSummary(
                iters=0, residual=float("nan"), residualMet=False, retries=0, detailLines=()
            )
            return np.full(n, np.nan)
        newIncrement = (
            self._lastResidualNorm is not None and residualNorm > self._residualGrowthFactor * self._lastResidualNorm
        )
        # A hierarchy built for another sparsity pattern can be a far worse preconditioner (measured: 494 instead
        # of 94 outer iterations). nnz is an O(1) proxy for a pattern change; it errs on the safe side.
        patternChanged = self._lastNnz is not None and A.nnz != self._lastNnz

        # Refresh the per-field AMG hierarchies (rather than reuse the standing ones) when there is
        # nothing to reuse yet, the field-block layout changed (e.g. an AMR event resized the DOF
        # vector), the sparsity pattern changed, a residual jump marks a new increment, or the
        # previous solve asked for it: its outer count drifted too far from that of the solve the hierarchies
        # were built for, or it hit the cap (_hierarchyIsStale), or the model or the settings changed.
        wasStale = self._refreshNext
        mustRefresh = (
            self._preconditioners is None
            or blocks != self._blocks
            or n != self._n
            or patternChanged
            or (newIncrement and self._refreshOnNewIncrement)
            or wasStale
        )

        # Captured now, in priority order matching mustRefresh above -- by the time the debug detail
        # is built at the end of this call, self._refreshNext has already been overwritten for the
        # *next* call, so "why" has to be recorded at the point mustRefresh itself is decided.
        if not mustRefresh:
            rebuildReason = None
        elif self._preconditioners is None:
            rebuildReason = "first"
        elif blocks != self._blocks or n != self._n:
            rebuildReason = "amr"
        elif patternChanged:
            rebuildReason = "pattern"
        elif newIncrement and self._refreshOnNewIncrement:
            rebuildReason = "increment"
        else:
            rebuildReason = "stale"

        # (2) Equilibration: solve A x = b as (D A D) z = D b, x = D z, D = diag(scaling) with
        #     scaling = 1 / sqrt|a_ii|. A_scaled = D A D, b_scaled = D b in the code.
        with performancetiming.timeit("equilibration"):
            if mustRefresh:
                diagonal = PyAMGCLEquilibratedSystem.absoluteDiagonal(A)
                # A zero diagonal entry cannot be equilibrated: 1/sqrt(0) is inf, which would poison
                # A_scaled and b_scaled with non-finite values before the hierarchy build, and the failure would
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
                scaling = 1.0 / np.sqrt(diagonal)
            else:
                # Reuse the equilibration the standing hierarchies were built for. Any D gives the same
                # system in other variables (A_scaled z = b_scaled with x = D z), so the outer solve still solves the
                # *current* A x = b; only the preconditioner's quality can drift -- which costs outer
                # iterations and, if it drives the solve to the cap, the accuracy of that one solve.
                scaling = self._scaling

            b_scaled = scaling * b

        # (3) Field split: A_scaled and its blocks A_scaled_ij in one threaded pass over A -- the outer Krylov
        #     operator, the couplings the sweep needs on every solve, and on a rebuild the diagonal blocks the
        #     hierarchies are built from.
        with performancetiming.timeit("equilibrated system"):
            system = PyAMGCLEquilibratedSystem()
            system.build(A, scaling, [(block.start, block.stop) for block in blocks])
        outerOperator = LinearOperator((n, n), matvec=system.matvec, dtype=np.float64)

        # (4) Hierarchies: one per field, rebuilt or reused (see mustRefresh above)
        if mustRefresh:
            with performancetiming.timeit("hierarchy build"):
                try:
                    preconditioners = self._buildFieldPreconditioners(system, blocks, scaling)
                except Exception as error:
                    if not self._hotReloadRollback:
                        raise
                    self._rollBackHotReload(error)
                    preconditioners = self._buildFieldPreconditioners(system, blocks, scaling)
                self._hotReloadRollback = []
                # cleared only now: a failed build leaves the request standing for the next solve
                self._refreshNext = False
                self._preconditioners = preconditioners
                self._scaling = scaling
                self._blocks = blocks
                self._n = n
                self._lastNnz = A.nnz
        preconditioners = self._preconditioners

        nFields = len(slices)
        sizes = [block.stop - block.start for block in blocks]

        # (5a) The preconditioner v = M^-1 r: block Gauss-Seidel sweeps over the fields, M^-1 = (B + L)^-1 for one
        #      forward sweep, B = blockdiag(M_i), M_i one AMG cycle on A_scaled_ii.
        def sweepOnce(order, residual, v, visited):
            # Residual-correction form, v_i += M_i^-1 (r_i - sum_j A_scaled_ij v_j): a revisit of a field (sweeps > 1,
            # or the symmetric reverse sweep) improves its v_i. On a first visit v_i = 0, and only the fields
            # already visited contribute to the sum.
            for i in order:
                localResidual = residual[slices[i]].copy()
                for j in range(nFields):
                    if j != i and visited[j]:
                        localResidual -= system.couplingMatvec(i, j, v[j])
                if visited[i]:
                    localResidual -= system.diagonalBlockMatvec(i, v[i])
                    v[i] = v[i] + preconditioners[i].applyPreconditioner(localResidual)
                else:
                    v[i] = preconditioners[i].applyPreconditioner(localResidual)
                    visited[i] = True

        def blockGaussSeidel(residual):
            v = [np.zeros(sizes[i]) for i in range(nFields)]
            visited = [False] * nFields
            for _ in range(self._sweeps):
                sweepOnce(range(nFields), residual, v, visited)
                if self._symmetric:
                    # the last field was just updated with everything else current: start one before it (with an
                    # AMG cycle it still has a residual left, but a cycle on it is saved)
                    sweepOnce(range(nFields - 2, -1, -1), residual, v, visited)
            return np.concatenate(v)

        preconditioner = LinearOperator((n, n), matvec=blockGaussSeidel, dtype=np.float64)

        if self._outerTol is not None:
            eta = self._outerTol
        else:
            eta = self._forcingTolerance(residualNorm, newIncrement)

        # the outer Krylov converges the *scaled* residual, but acceptance is on the *true*
        # residual; ask the first pass for gapSafetyFactor * eta / gap (gap learned from previous solves) so
        # that it lands on target in one pass instead of paying a warm-restart continuation.
        if self._gapCompensatedTolerance:
            firstPassEta = max(self._gapSafetyFactor * eta / max(self._trueResidualGap, 1.0), 1e-14)
        else:
            firstPassEta = eta

        # (5b) The outer Krylov solve on A_scaled z = b_scaled, preconditioned by the sweep: AMGCL's LGMRES (the
        #      default, right-preconditioned) or SciPy's GMRES (left-preconditioned); both converge the residual of
        #      A_scaled z = b_scaled. For LGMRES, outerRestart x outerMaxiter is the iteration cap of a pass.
        def outerPass(tolerance: float, x0: np.ndarray = None) -> tuple:
            """One outer Krylov pass from x0 (zero if None): the Krylov iterate, the iteration count, and
            SciPy's ``info`` (non-zero: stopped at the iteration cap short of ``tolerance``)."""
            if self._outerSolver == "scipy":
                residualHistory = []
                z, info = gmres(
                    outerOperator,
                    b_scaled,
                    x0=x0,
                    M=preconditioner,
                    rtol=tolerance,
                    atol=0.0,
                    restart=self._outerRestart,
                    maxiter=self._outerMaxiter,
                    callback=lambda residualNorm: residualHistory.append(residualNorm),
                    callback_type="pr_norm",
                )
                return z, len(residualHistory), info
            lgmres = self._outerLgmres(n)
            z = lgmres.solve(
                system,
                b_scaled,
                blockGaussSeidel,
                tolerance,
                self._outerRestart * self._outerMaxiter,
                x0=x0,
            )
            iterations = lgmres.lastIterations
            return z, iterations, self._lgmresCapHit(iterations, tolerance)

        def solutionAndResiduals(z: np.ndarray) -> tuple:
            """The solution x of the Krylov iterate z, its true relative residual (the one the solve is
            accepted on) and its scaled relative residual (the equilibrated system's)."""
            # A x - b = D^-1 (A_scaled z - b_scaled) exactly: the true residual from the scaled one, without a
            # product with A
            scaledResidualVector = system.residual(b_scaled, z)
            return (
                scaling * z,
                np.linalg.norm(scaledResidualVector / scaling) / max(np.linalg.norm(b), 1e-300),
                np.linalg.norm(scaledResidualVector) / max(np.linalg.norm(b_scaled), 1e-300),
            )

        with performancetiming.timeit("outer Krylov solve"):
            z, outerIters, info = outerPass(firstPassEta)
            x, trueResidual, scaledResidual = solutionAndResiduals(z)

        # (6) True-residual check. The outer solve converges the scaled residual, while the solve is accepted on
        #     the true relative residual ||b - A x|| / ||b||, which can be larger (stiff rows are down-weighted
        #     by D). A pass that misses eta is continued from its iterate with a tighter tolerance; the tolerance
        #     is tightened in its own units (0.01x), since a warm restart at the same tolerance stops at once. The
        #     ratio of the true to the scaled residual of the same iterate -- the gap -- is learned for
        #     gapCompensatedTolerance, smoothed and bounded by gapMaxFactor.
        measuredGap = trueResidual / max(scaledResidual, 1e-300)
        if np.isfinite(measuredGap) and measuredGap > 0.0:
            # Ceiled at gapMaxFactor too: an unbounded gap from one non-converged solve compounds
            # through the EMA into an ever-tighter, unmeetable tolerance.
            self._trueResidualGap = min(max(0.5 * self._trueResidualGap + 0.5 * measuredGap, 1.0), self._gapMaxFactor)

        with performancetiming.timeit("true-residual continuations"):
            capHit = info != 0
            continuationEta = min(eta, firstPassEta) if self._gapCompensatedTolerance else eta
            continuations = 0
            while trueResidual > eta and continuations < self._trueResidualMaxContinuations:
                continuations += 1
                # with gapCompensatedTolerance: aim at eta / gap (with a safety factor), at least 2x tighter
                # per round so that the loop terminates
                if self._gapCompensatedTolerance:
                    gap = trueResidual / max(scaledResidual, 1e-300)
                    proposedEta = self._gapSafetyFactor * eta / max(gap, 1.0)
                    continuationEta = max(min(proposedEta, continuationEta * 0.5), 1e-14)
                else:
                    continuationEta *= 0.01
                z, continuationIters, info = outerPass(continuationEta, x0=z)
                x, trueResidual, scaledResidual = solutionAndResiduals(z)
                outerIters += continuationIters
                capHit = capHit or info != 0
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

        # (7) Bookkeeping for the next solve: residual norm, tolerance, nnz, outer iterations, staleness
        previousOuterIters = self._lastOuterIters
        if self._hierarchyIsStale(outerIters, mustRefresh, capHit):
            self._refreshNext = True
        self._lastOuterIters = outerIters
        self._lastContinuations = continuations
        self._lastResidualNorm = residualNorm
        self._lastEta = eta

        solveElapsedTime = time.time() - solveStartTime

        # Populated unconditionally (cheap: one dataclass construction) -- reportsSolveSummary, not
        # this, is what decides whether the nonlinear solver ever looks at it, so there is nothing to
        # gate here beyond detailLines, which is genuinely wasted work at anything below "debug".
        detailLines = ()
        if self._verbosityIndex >= _VERBOSITY_LEVELS.index("debug"):
            precondWord = "rebuilt({:})".format(rebuildReason) if mustRefresh else "reused"
            detailLines = (
                precondWord,
                "η={:.1e} · {:.1f}s".format(eta, solveElapsedTime),
            )
        self._lastSolveSummary = LinearSolveSummary(
            iters=outerIters,
            residual=trueResidual,
            residualMet=trueResidual <= eta,
            retries=continuations,
            detailLines=detailLines,
        )
        if outerIters > self._warnOuterIterationsThreshold:
            # Rare and actionable, so this stays at "warning" (Journal level 0, the least indented and
            # most visible spot in the log) and spells out the likely cause, unlike the routine
            # per-solve line above it.
            self._log(
                "warning",
                "⚠ solve #{:} needed {:} outer Krylov iterations, above the warning threshold of {:}. "
                "Likely a preconditioner-quality problem, not a physics one -- worth checking whether "
                "the Jacobian changed sharply since the last hierarchy rebuild.".format(
                    self._solveCount, outerIters, self._warnOuterIterationsThreshold
                ),
            )
        # the solve's bookkeeping, recorded with a dumped system so that a replay can tell a stale hierarchy from a
        # hard operator without rerunning
        solveState = {
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
        # Written "not (eta met)" so that a NaN residual counts as a miss. A cap hit that the solve recovered from
        # (a continuation met eta) still rebuilds the hierarchies (_hierarchyIsStale above), but is not reported.
        converged = trueResidual <= eta
        # a cap hit with eta missed counts as degraded whatever the threshold: with a cap below it (outerRestart x
        # outerMaxiter 50 and no continuations, say), the count alone would never trigger a dump
        if (
            self._dumpOnDegradationDir is not None
            and (outerIters > self._degradationThreshold() or (capHit and not converged))
            and BlockAMGSolver._degradationDumpsWritten < self._dumpOnDegradationMaxDumps
        ):
            self._dumpDegradedSystem(A, b, blocks, solveState)
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
                    "state": solveState,
                }
            )
        if not converged and capHit:
            self._log(
                "warning",
                "⚠ solve #{:} did not converge: ‖r‖={:.1e} still exceeds η={:.1e} after {:} continuations, "
                "and an outer pass stopped at its iteration cap (outerRestart x outerMaxiter = {:}). The cap "
                "is a budget, not a tolerance: a solve that needs more is a preconditioning problem -- the "
                "hierarchies are rebuilt for the next solve; see also fieldPreconds and the recommended "
                "configuration of the theory page.".format(
                    self._solveCount, trueResidual, eta, continuations, self._outerRestart * self._outerMaxiter
                ),
            )
        elif not converged:
            self._log(
                "warning",
                "⚠ solve #{:} did not fully converge: ‖r‖={:.1e} still exceeds η={:.1e} after {:} "
                "continuations, though the outer solve converged its own (scaled) residual: the gap between the "
                "scaled and the true residual, which more continuations (trueResidualMaxContinuations) or "
                "gapCompensatedTolerance close; a rebuilt preconditioner does not.".format(
                    self._solveCount, trueResidual, eta, continuations
                ),
            )

        return x
