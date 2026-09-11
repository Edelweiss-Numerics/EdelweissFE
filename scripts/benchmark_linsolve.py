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
#  ---------------------------------------------------------------------

"""Offline benchmark of linear-solver strategies on a captured sequence of equation systems.

Reads what the ``matrixdump`` linear solver wrote (see
:mod:`edelweissfe.linsolve.matrixdump.matrixdump`) and replays it, so solver variants are compared
on byte-identical input instead of by rerunning the simulation. Each check is a separate subcommand
because they cost wildly different amounts of time: ``pattern`` is seconds, ``reuse`` and ``threads``
are minutes, and ``lagged`` involves a SuperLU factorization of the whole system and can run much
longer. Nothing here mutates the dumps.

Usage::

    python benchmark_linsolve.py pattern  <dumpDir>
    python benchmark_linsolve.py reuse    <dumpDir>
    python benchmark_linsolve.py threads  <dumpDir> [--threads 1,4,8,16,36]
    python benchmark_linsolve.py lagged   <dumpDir> [--maxiter 200] [--rtol 1e-8]

``threads`` must be run one thread count per process -- MKL latches its thread count at first use, so
sweeping inside a single process would silently measure the first setting repeatedly. The subcommand
therefore re-executes itself once per value.
"""

import argparse
import json
import os
import subprocess
import sys
from time import perf_counter

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla


def loadManifest(dumpDir: str) -> list[dict]:
    """Read the dump manifest, in capture order."""

    manifestPath = os.path.join(dumpDir, "manifest.jsonl")
    with open(manifestPath) as manifestFile:
        records = [json.loads(line) for line in manifestFile if line.strip()]

    records.sort(key=lambda record: (record["instance"], record["ordinal"]))
    return records


def loadSystem(dumpDir: str, record: dict):
    """Load one dumped system, checking it against its manifest fingerprints."""

    A = sp.load_npz(os.path.join(dumpDir, record["matrixFile"])).tocsr()
    b = np.load(os.path.join(dumpDir, record["rhsFile"]))

    if A.nnz != record["nnz"]:
        raise ValueError(
            "{:}: nnz {:} does not match the manifest's {:}".format(record["matrixFile"], A.nnz, record["nnz"])
        )

    return A, b


def commandPattern(args):
    """Report how the sparsity pattern evolves along the captured sequence.

    This decides whether PARDISO's symbolic factorization can be reused at all: reuse is only
    possible if the pattern is genuinely identical from one solve to the next. The nnz column is the
    cheap tell, and the explicit index comparison is the real one -- two matrices can share an nnz
    count with different patterns.
    """

    records = loadManifest(args.dumpDir)
    print("{:} dumped systems in {:}\n".format(len(records), args.dumpDir))

    print(
        "{:>4} {:>4} {:>12} {:>14} {:>14} {:>10} {:>12}".format(
            "inst", "ord", "nnz", "||b||", "||diag(A)||", "nnz delta", "pattern"
        )
    )

    previousIndices = None
    previousIndptr = None

    for record in records:
        A, b = loadSystem(args.dumpDir, record)

        if previousIndices is None:
            patternNote = "-"
            nnzDelta = "-"
        else:
            samePattern = np.array_equal(A.indices, previousIndices) and np.array_equal(A.indptr, previousIndptr)
            patternNote = "identical" if samePattern else "CHANGED"
            nnzDelta = "{:+}".format(A.nnz - len(previousIndices))

        print(
            "{:>4} {:>4} {:>12} {:>14.6e} {:>14.6e} {:>10} {:>12}".format(
                record["instance"],
                record["ordinal"],
                A.nnz,
                record["rhsNorm"],
                record["diagonalNorm"],
                nnzDelta,
                patternNote,
            )
        )

        previousIndices = A.indices.copy()
        previousIndptr = A.indptr.copy()

    # Structural symmetry decides whether the cheaper structurally-symmetric PARDISO matrix types are
    # even applicable, and how much fill-in the reordering has to guess at. Measured on the last
    # system loaded.
    pattern = sp.csr_matrix((np.ones(A.nnz), A.indices, A.indptr), shape=A.shape)
    asymmetric = (pattern - pattern.T).nnz
    print(
        "\nlast system: {:} x {:}, {:} nnz ({:.1f} per row), {:} structurally asymmetric entries".format(
            A.shape[0], A.shape[1], A.nnz, A.nnz / A.shape[0], asymmetric
        )
    )
    print("explicitly stored zeros: {:}".format(int(np.count_nonzero(A.data == 0.0))))


def unifyPattern(systems: list) -> list:
    """Re-express every system on the structural union of all their patterns.

    A captured sequence may have a pattern that wobbles from one Newton iteration to the next, in
    which case PARDISO's reuse check refuses to engage and measuring "reuse on" measures nothing.
    Projecting the whole sequence onto one common pattern -- keeping the extra entries as explicitly
    stored zeros, which leaves every matrix mathematically unchanged -- makes reuse engage, and so
    prices the fix: it is what the solver would see if the assembly stopped pruning zeros per
    iteration.

    The union over the captured iterations is a proxy for that fixed pattern, and a slightly generous
    one: it converges to the true unpruned pattern from below as more iterations are included, so any
    speedup measured here is, if anything, a mild underestimate.

    SciPy's own sparse addition cannot do this -- it prunes explicit zeros, which is exactly the
    behaviour being worked around -- so the values are scattered into the union's index arrays
    directly, via a linearized (row, column) key.
    """

    unionPattern = systems[0][0].copy()
    unionPattern.data[:] = 1.0
    for A, _ in systems[1:]:
        contribution = A.copy()
        contribution.data[:] = 1.0
        unionPattern = unionPattern + contribution

    unionPattern.sort_indices()
    rows, columns = unionPattern.shape

    def linearKeys(indptr, indices):
        rowOf = np.repeat(np.arange(rows, dtype=np.int64), np.diff(indptr))
        return rowOf * np.int64(columns) + indices.astype(np.int64)

    unionKeys = linearKeys(unionPattern.indptr, unionPattern.indices)

    unified = []
    for A, b in systems:
        A.sort_indices()
        positions = np.searchsorted(unionKeys, linearKeys(A.indptr, A.indices))

        data = np.zeros(unionPattern.nnz, dtype=float)
        data[positions] = A.data

        projected = sp.csr_matrix((data, unionPattern.indices.copy(), unionPattern.indptr.copy()), shape=A.shape)

        # The projection must be value-preserving, not merely plausible: a mismatch would mean the
        # union was not a superset and the scatter silently dropped entries.
        if abs(projected - A).max() != 0.0:
            raise ValueError("pattern unification changed the matrix values")

        unified.append((projected, b))

    print(
        "unified pattern: {:} nnz (per-system nnz ranged {:} .. {:})\n".format(
            unionPattern.nnz,
            min(A.nnz for A, _ in systems),
            max(A.nnz for A, _ in systems),
        )
    )

    return unified


def commandReuse(args):
    """Time and cross-check PARDISO with symbolic-factorization reuse on versus off.

    Reuse is opt-in precisely because it has been observed to return wrong-but-finite results on
    coupled problems, so this does not just time the two: it compares their solutions against the
    no-reuse result on every system of the sequence. A speedup is only worth anything if the answers
    agree, and this is the cheapest place to find out -- the sequence is real, and no simulation has
    to be rerun to test it.
    """

    from edelweissfe.linsolve.pardiso.pardiso import PardisoSolver

    records = loadManifest(args.dumpDir)
    systems = [loadSystem(args.dumpDir, record) for record in records]

    print("MKL threads: {:}\n".format(os.environ.get("OMP_NUM_THREADS", "(unset)")))

    if args.unifyPattern:
        systems = unifyPattern(systems)

    referenceSolutions = []
    print("--- reuse OFF (current default) ---")
    solverOff = PardisoSolver(reuseSymbolicFactorization=False)
    for record, (A, b) in zip(records, systems):
        start = perf_counter()
        x = solverOff(A, b)
        elapsed = perf_counter() - start
        residual = np.linalg.norm(A @ x - b) / max(np.linalg.norm(b), 1.0e-300)
        referenceSolutions.append(x)
        print("  ord {:>3}: {:8.3f} s   rel. residual {:9.3e}".format(record["ordinal"], elapsed, residual))

    print("\n--- reuse ON (one persistent solver across the sequence) ---")
    solverOn = PardisoSolver(reuseSymbolicFactorization=True)
    for record, (A, b), reference in zip(records, systems, referenceSolutions):
        start = perf_counter()
        x = solverOn(A, b)
        elapsed = perf_counter() - start
        residual = np.linalg.norm(A @ x - b) / max(np.linalg.norm(b), 1.0e-300)
        deviation = np.linalg.norm(x - reference) / max(np.linalg.norm(reference), 1.0e-300)
        flag = "" if deviation < 1.0e-10 else "   <-- DIVERGES FROM NO-REUSE RESULT"
        print(
            "  ord {:>3}: {:8.3f} s   rel. residual {:9.3e}   rel. dev. {:9.3e}{:}".format(
                record["ordinal"], elapsed, residual, deviation, flag
            )
        )


def commandThreads(args):
    """Measure how a single PARDISO solve scales with MKL thread count.

    Re-executes itself once per thread count: MKL fixes its thread count on first use, so a sweep
    inside one process would report the first setting for every entry. Answers what the production
    ``OMP_NUM_THREADS`` should be, and -- read against the single-threaded SciPy SpGEMMs of the MPC
    condensation -- how the serial fraction of an increment grows as cores are added.
    """

    if args.child is None:
        threadCounts = [int(value) for value in args.threads.split(",")]
        for threadCount in threadCounts:
            environment = dict(os.environ)
            environment["OMP_NUM_THREADS"] = str(threadCount)
            environment["MKL_NUM_THREADS"] = str(threadCount)
            subprocess.run(
                [sys.executable, __file__, "threads", args.dumpDir, "--child", str(threadCount)],
                env=environment,
                check=True,
            )
        return

    from edelweissfe.linsolve.pardiso.pardiso import PardisoSolver

    records = loadManifest(args.dumpDir)
    A, b = loadSystem(args.dumpDir, records[0])

    solver = PardisoSolver(reuseSymbolicFactorization=False)
    solver(A, b)  # warm up: first call also pays page faults on the freshly loaded arrays

    timings = []
    for _ in range(args.repeats):
        start = perf_counter()
        solver(A, b)
        timings.append(perf_counter() - start)

    print(
        "threads {:>3}: best {:8.3f} s   median {:8.3f} s   (of {:} repeats)".format(
            args.child, min(timings), sorted(timings)[len(timings) // 2], args.repeats
        )
    )


def commandLagged(args):
    """Count GMRES iterations when an earlier iterate's LU is reused as a preconditioner.

    This is the make-or-break number for a modified Newton-Krylov scheme: if the factorization of
    system k preconditions system k+j well enough that GMRES converges in a handful of matrix-vector
    products, one expensive factorization can serve several Newton iterations, and the cost per
    iteration collapses. If the count instead climbs steeply with staleness j, the whole route is
    dead and the effort belongs on the direct solve.

    Either LU backend gives the same iteration counts, because the count depends on how well
    ``A_k^-1 A_(k+j)`` is conditioned -- a property of the matrices, not of whichever implementation
    inverted the first one. They differ enormously in how long the setup takes, though: on a 280k-dof,
    40M-nnz condensed system PARDISO factorizes in seconds where SuperLU runs for tens of minutes, so
    ``--backend pardiso`` is the practical choice and is also the code path a production
    implementation would use.
    """

    records = loadManifest(args.dumpDir)
    systems = [loadSystem(args.dumpDir, record) for record in records]

    print(
        "MKL threads: {:} (OMP_NUM_THREADS={:}, MKL_NUM_THREADS={:})\n".format(
            os.environ.get("MKL_NUM_THREADS") or os.environ.get("OMP_NUM_THREADS", "(unset)"),
            os.environ.get("OMP_NUM_THREADS", "(unset)"),
            os.environ.get("MKL_NUM_THREADS", "(unset)"),
        )
    )

    baseRecord, (baseMatrix, _) = records[0], systems[0]
    print(
        "factorizing the base system (instance {:}, ordinal {:}) with {:} ...".format(
            baseRecord["instance"], baseRecord["ordinal"], args.backend
        )
    )
    start = perf_counter()

    if args.backend == "pardiso":
        from edelweissfe.linsolve.pardiso.pardiso import PardisoSolver

        baseSolver = PardisoSolver(reuseSymbolicFactorization=True)
        baseSolver.factorize(baseMatrix)
        applyInverse = baseSolver.solveFactorized
    else:
        applyInverse = spla.splu(baseMatrix.tocsc()).solve

    print("  done in {:.1f} s\n".format(perf_counter() - start))

    preconditioner = spla.LinearOperator(baseMatrix.shape, matvec=applyInverse, dtype=baseMatrix.dtype)

    # Which parts of a GMRES iteration are actually threaded is the first thing anyone will ask of a
    # verdict against the iterative route, so measure the pieces rather than inferring them. The
    # preconditioner apply is MKL-threaded; SciPy's CSR matvec and the orthogonalization are not.
    probe = np.ones(baseMatrix.shape[0])

    start = perf_counter()
    for _ in range(args.probeRepeats):
        applyInverse(probe)
    applyTime = (perf_counter() - start) / args.probeRepeats

    start = perf_counter()
    for _ in range(args.probeRepeats):
        baseMatrix @ probe
    matvecTime = (perf_counter() - start) / args.probeRepeats

    print(
        "per-iteration cost floor: preconditioner apply {:.3f} s (MKL-threaded)"
        " + matvec {:.3f} s (SciPy, serial) = {:.3f} s\n".format(applyTime, matvecTime, applyTime + matvecTime)
    )

    # The thing the iterative route has to beat, measured in this same process rather than quoted
    # from a different run. Reuse is on so this is the *post-fix* direct solve -- the fair target,
    # since a fixed pattern is assumed to be in place before any of this would be considered.
    from edelweissfe.linsolve.pardiso.pardiso import PardisoSolver

    referenceSolver = PardisoSolver(reuseSymbolicFactorization=True)
    referenceSolver(baseMatrix, systems[0][1])  # absorb the one-off analyze
    start = perf_counter()
    referenceSolver(baseMatrix, systems[0][1])
    directTime = perf_counter() - start
    print(
        "reference: one direct solve with a reused symbolic factorization takes {:.3f} s"
        " -- so GMRES pays off only below ~{:.0f} iterations\n".format(
            directTime, directTime / max(applyTime + matvecTime, 1.0e-12)
        )
    )

    # How far the preconditioned operator actually sits from the identity. This is the quantity that
    # governs the iteration count -- NOT the norm-wise difference between the matrices, which on a
    # system with this dynamic range (||diag|| ~ 1e8 against residuals of order 1) is dominated by the
    # large entries and says nothing about the stiff directions. Estimated from below by applying
    # E = A_base^-1 (A_k - A_base) to random unit vectors, which needs only a matvec.
    rng = np.random.default_rng(0)
    print("distance of the preconditioned operator from the identity, ||A_base^-1 (A_k - A_base)||:")
    for record, (A, _) in list(zip(records, systems))[1:4]:
        delta = A - baseMatrix
        amplification = 0.0
        for _ in range(args.probeRepeats):
            v = rng.standard_normal(baseMatrix.shape[0])
            v /= np.linalg.norm(v)
            amplification = max(amplification, np.linalg.norm(applyInverse(delta @ v)))
        print(
            "  ord {:>3}: >= {:10.3e}   (while ||A_k - A_base||_F / ||A_base||_F = {:9.3e})".format(
                record["ordinal"],
                amplification,
                np.linalg.norm(delta.data) / np.linalg.norm(baseMatrix.data),
            )
        )
    print()

    # Iterations to reach a range of tolerances, from one run each. A Newton step does not need the
    # linear system solved to 1e-8: an inexact-Newton (Eisenstat-Walker) scheme would ask for 1e-2 to
    # 1e-4 early on and tighten only near convergence. Reporting a single tight-tolerance count would
    # benchmark the iterative route against a target it never has to hit.
    tolerances = [1.0e-2, 1.0e-4, 1.0e-6, 1.0e-8]

    print("GMRES preconditioned by the base factorization, restart={:}, maxiter={:}".format(args.restart, args.maxiter))
    print(
        "{:>4} {:>10} {:>32} {:>12} {:>12}".format("ord", "staleness", "iterations to reach rtol", "total", "wall time")
    )
    print("{:>4} {:>10} {:>8} {:>7} {:>7} {:>8} {:>12} {:>12}".format("", "", "1e-2", "1e-4", "1e-6", "1e-8", "", ""))

    for offset, (record, (A, b)) in enumerate(zip(records, systems)):
        history = []

        def recordResidual(residualNorm):
            history.append(residualNorm)

        start = perf_counter()
        x, info = spla.gmres(
            A,
            b,
            M=preconditioner,
            rtol=args.rtol,
            atol=0.0,
            restart=args.restart,
            maxiter=args.maxiter,
            callback=recordResidual,
            callback_type="pr_norm",
        )
        elapsed = perf_counter() - start

        def itersToReach(tolerance):
            for iteration, residualNorm in enumerate(history, start=1):
                if residualNorm <= tolerance:
                    return str(iteration)
            return "-"

        print(
            "{:>4} {:>10} {:>8} {:>7} {:>7} {:>8} {:>12} {:>10.2f} s".format(
                record["ordinal"],
                offset,
                *[itersToReach(tolerance) for tolerance in tolerances],
                "{:} ({:})".format(len(history), "ok" if info == 0 else "info={:}".format(info)),
                elapsed,
            )
        )


#: AMGCL configurations to try, in rough order of how plausible they are for this system. All run on
#: AMGCL's OpenMP `builtin` backend, so unlike the SciPy path the matvec, the smoother and the
#: orthogonalization are all threaded.
#:
#: The system is a poor fit for classical AMG and it is worth being explicit about why, so that a
#: failure here is not mistaken for a configuration mistake: it is non-symmetric with 52% of its
#: entries structurally asymmetric, it couples two fields with no constant nodal block size (nonlocal
#: damage lives on corner nodes only, so point-block smoothing does not apply), penalty contact and
#: the MPC/Dirichlet row replacements give it a ~1e8 dynamic range, and rows replaced by constraint
#: equations are not an elliptic operator at all. Single-level ILU0 is included precisely because it
#: makes none of AMG's smoothness assumptions.
_AMGCL_CONFIGURATIONS = [
    (
        "bicgstab + AMG(SA, ilu0)  [wrapper default]",
        {
            "solver": {"type": "bicgstab", "tol": 1e-8, "maxiter": 500},
            "precond": {"coarsening": {"type": "smoothed_aggregation"}, "relax": {"type": "ilu0"}},
        },
    ),
    (
        "gmres(100) + AMG(SA, ilu0)",
        {
            "solver": {"type": "gmres", "M": 100, "tol": 1e-8, "maxiter": 500},
            "precond": {"coarsening": {"type": "smoothed_aggregation"}, "relax": {"type": "ilu0"}},
        },
    ),
    (
        "gmres(100) + ILU0 only (no AMG)",
        {
            "solver": {"type": "gmres", "M": 100, "tol": 1e-8, "maxiter": 500},
            "precond": {"class": "relaxation", "type": "ilu0"},
        },
    ),
    (
        "fgmres(100) + AMG(SA, spai0)",
        {
            "solver": {"type": "fgmres", "M": 100, "tol": 1e-8, "maxiter": 500},
            "precond": {"coarsening": {"type": "smoothed_aggregation"}, "relax": {"type": "spai0"}},
        },
    ),
    (
        "idrs(4) + ILU0 only",
        {
            "solver": {"type": "idrs", "s": 4, "tol": 1e-8, "maxiter": 500},
            "precond": {"class": "relaxation", "type": "ilu0"},
        },
    ),
]


def commandAmgcl(args):
    """Try AMGCL's own OpenMP-parallel Krylov solvers on the captured systems.

    AMGCL is the one option here that is threaded end to end -- its matvec, smoother and
    orthogonalization all run on the builtin OpenMP backend -- so it is the fair test of "would a
    properly parallel iterative solver win?", which the SciPy-based `lagged` benchmark cannot answer
    (SciPy's matvec is serial, though it turns out to be only ~7% of that scheme's cost).

    Reported per configuration: AMGCL's own iteration count and error estimate, the *true* relative
    residual recomputed independently, and wall time -- against the direct solve it has to beat.
    Recomputing the residual matters: AMGCL's error estimate is what its stopping rule looked at, and
    a solver that stops early on a misleading estimate would otherwise look like a winner.

    Note the wrapper rebuilds the AMG hierarchy on every call, so these times include setup. That is
    the production-relevant number rather than a handicap: the sparsity pattern changes every Newton
    iteration on this model (see the `pattern` subcommand), so a cached hierarchy could never be
    reused anyway.
    """

    from edelweissfe.linsolve.amgcl.amgcl import PyAMGCLSolver
    from edelweissfe.linsolve.pardiso.pardiso import PardisoSolver

    records = loadManifest(args.dumpDir)
    A, b = loadSystem(args.dumpDir, records[0])

    print(
        "MKL/OMP threads: {:}   system: {:} dof, {:} nnz\n".format(
            os.environ.get("OMP_NUM_THREADS", "(unset)"), A.shape[0], A.nnz
        )
    )

    referenceSolver = PardisoSolver(reuseSymbolicFactorization=True)
    reference = referenceSolver(A, b)
    start = perf_counter()
    referenceSolver(A, b)
    directTime = perf_counter() - start
    print(
        "PARDISO direct solve with reused symbolic factorization: {:.2f} s  (the target to beat)\n".format(directTime)
    )

    bNorm = max(np.linalg.norm(b), 1.0e-300)
    referenceNorm = max(np.linalg.norm(reference), 1.0e-300)

    # Swept, not fixed at 1e-8: a Newton step does not need a tight linear solve, and judging an
    # iterative solver only at 1e-8 is what produced a wrong verdict in the `lagged` benchmark.
    tolerances = [float(value) for value in args.tolerances.split(",")]

    def runConfiguration(description, configured, nullspace):
        """Build a solver for one configuration (optionally with a near null-space), solve, print."""
        try:
            solver = PyAMGCLSolver(configured)
            if nullspace is not None:
                solver.set_nullspace(nullspace)
            start = perf_counter()
            x = solver.solve(A, b)
            elapsed = perf_counter() - start
        except Exception as failure:  # noqa: BLE001 - a rejected config should not abort the sweep
            print("{:<44} {:>8}   {:}".format(description, "ERROR", failure))
            return None
        trueResidual = np.linalg.norm(A @ x - b) / bNorm
        deviation = np.linalg.norm(x - reference) / referenceNorm
        hitCap = solver.lastIterations >= args.maxiter
        print(
            "{:<44} {:>8} {:>12.2e} {:>13.2e} {:>8.2f} s  {:}{:}".format(
                description,
                solver.lastIterations,
                solver.lastError,
                trueResidual,
                elapsed,
                "dev {:.1e}".format(deviation),
                (
                    "  <-- HIT MAXITER, did not converge"
                    if hitCap
                    else ("  <-- beats direct" if elapsed < directTime else "")
                ),
            )
        )
        return not hitCap

    if args.nullspace == "translations":
        # The cheap, no-geometry partial test of PERF_LINSOLVE_INVESTIGATION.md section 4, step 1:
        # does supplying smoothed aggregation the three rigid-body *translations* of the displacement
        # block (its default is a single constant vector) improve monolithic AMG at all? Translations
        # dominate the elasticity near null-space, and they are constructible from the DOF layout
        # alone -- the displacement block is node-major with 3 components per node -- so no nodal
        # coordinates are needed. If three translations do not help over the default one constant,
        # the six rigid body modes will not rescue it and block-unaware AMG is a dead end here.
        n = A.shape[0]
        dispDofs = min(args.dispDofs, n)
        rowIndex = np.arange(dispDofs)

        # Pure translations: 1 on each displacement component, zero on the coupled (damage) block.
        translations = np.zeros((n, 3), dtype=np.float64)
        translations[rowIndex, rowIndex % 3] = 1.0

        # Block-structured variant: the same three translations on the displacement block, plus a
        # fourth vector that is constant on the coupled block and zero on displacement -- so the
        # coupled rows are not left with a degenerate (all-zero) near null-space. This isolates
        # whether the pure-translations result above is an artefact of the zero coupled block rather
        # than a genuine verdict on monolithic AMG.
        blockTranslations = np.zeros((n, 4), dtype=np.float64)
        blockTranslations[rowIndex, rowIndex % 3] = 1.0
        blockTranslations[dispDofs:, 3] = 1.0

        print(
            "near null-space test: {:} displacement dofs (3 components, node-major) + {:} coupled "
            "dofs, of {:} total\n".format(dispDofs, n - dispDofs, n)
        )

        tolerance = tolerances[0]
        print(
            "{:<44} {:>8} {:>12} {:>13} {:>10}   rtol={:.0e}\n".format(
                "configuration", "iters", "amgcl err", "true rel.res", "wall", tolerance
            )
        )
        candidates = [
            ("1 constant", None),
            ("3 translations", translations),
            ("3 transl + coupled const", blockTranslations),
        ]
        smootherByName = {"ilu0": {"type": "ilu0"}, "spai0": {"type": "spai0"}}
        for smootherName, smoother in smootherByName.items():
            configured = {
                "solver": {"type": "gmres", "M": 100, "tol": tolerance, "maxiter": args.maxiter},
                "precond": {"coarsening": {"type": "smoothed_aggregation"}, "relax": smoother},
            }
            for label, nullspace in candidates:
                runConfiguration("gmres(100) + AMG(SA, {:}), {:}".format(smootherName, label), configured, nullspace)
        return

    print(
        "{:<40} {:>8} {:>8} {:>12} {:>13} {:>10}".format(
            "configuration", "rtol", "iters", "amgcl err", "true rel.res", "wall"
        )
    )

    for description, parameters in _AMGCL_CONFIGURATIONS:
        for tolerance in tolerances:
            configured = {
                "solver": dict(parameters["solver"], tol=tolerance, maxiter=args.maxiter),
                "precond": parameters["precond"],
            }

            try:
                solver = PyAMGCLSolver(configured)
                start = perf_counter()
                x = solver.solve(A, b)
                elapsed = perf_counter() - start
            except Exception as failure:  # noqa: BLE001 - a rejected config should not abort the sweep
                print("{:<40} {:>8.0e} {:>8}   {:}".format(description, tolerance, "ERROR", failure))
                continue

            trueResidual = np.linalg.norm(A @ x - b) / bNorm
            deviation = np.linalg.norm(x - reference) / referenceNorm
            hitCap = solver.lastIterations >= args.maxiter

            print(
                "{:<40} {:>8.0e} {:>8} {:>12.2e} {:>13.2e} {:>8.2f} s  {:}{:}".format(
                    description,
                    tolerance,
                    solver.lastIterations,
                    solver.lastError,
                    trueResidual,
                    elapsed,
                    "dev {:.1e}".format(deviation),
                    (
                        "  <-- HIT MAXITER, did not converge"
                        if hitCap
                        else ("  <-- beats direct" if elapsed < directTime else "")
                    ),
                )
            )

            # No point tightening a tolerance this configuration could not even reach.
            if hitCap:
                break


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    subparsers = parser.add_subparsers(dest="command", required=True)

    amgclParser = subparsers.add_parser("amgcl", help="AMGCL's OpenMP-parallel Krylov solvers")
    amgclParser.add_argument("dumpDir")
    amgclParser.add_argument(
        "--tolerances",
        default="1e-2,1e-4,1e-8",
        help="loosest first: the sweep stops tightening once a configuration fails to converge",
    )
    amgclParser.add_argument("--maxiter", type=int, default=500)
    amgclParser.add_argument(
        "--nullspace",
        choices=["none", "translations"],
        default="none",
        help="feed smoothed aggregation a near null-space: 'translations' supplies the 3 rigid-body "
        "translations of the displacement block (constructible from the DOF layout alone), and runs "
        "each SA configuration both with and without it for a direct comparison",
    )
    amgclParser.add_argument(
        "--dispDofs",
        type=int,
        default=214659,
        help="size of the leading displacement block (3 components per node, node-major); the "
        "translational near null-space is nonzero only on it. Default matches the reference model.",
    )
    amgclParser.set_defaults(function=commandAmgcl)

    patternParser = subparsers.add_parser("pattern", help="report sparsity-pattern evolution")
    patternParser.add_argument("dumpDir")
    patternParser.set_defaults(function=commandPattern)

    reuseParser = subparsers.add_parser("reuse", help="time and cross-check symbolic-factorization reuse")
    reuseParser.add_argument("dumpDir")
    reuseParser.add_argument(
        "--unifyPattern",
        action="store_true",
        help="first re-express all systems on their common pattern, so reuse can engage at all",
    )
    reuseParser.set_defaults(function=commandReuse)

    threadsParser = subparsers.add_parser("threads", help="MKL thread scaling of one solve")
    threadsParser.add_argument("dumpDir")
    threadsParser.add_argument("--threads", default="1,4,8,16,36")
    threadsParser.add_argument("--repeats", type=int, default=3)
    threadsParser.add_argument("--child", type=int, default=None, help=argparse.SUPPRESS)
    threadsParser.set_defaults(function=commandThreads)

    laggedParser = subparsers.add_parser("lagged", help="GMRES iterations with a lagged LU preconditioner")
    laggedParser.add_argument("dumpDir")
    laggedParser.add_argument(
        "--backend",
        choices=["pardiso", "superlu"],
        default="pardiso",
        help="which LU factorizes the base system; same iteration counts, wildly different setup cost",
    )
    laggedParser.add_argument("--probeRepeats", type=int, default=5)
    laggedParser.add_argument("--maxiter", type=int, default=200)
    laggedParser.add_argument("--restart", type=int, default=50)
    laggedParser.add_argument("--rtol", type=float, default=1.0e-8)
    laggedParser.set_defaults(function=commandLagged)

    args = parser.parse_args()
    args.function(args)


if __name__ == "__main__":
    main()
