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

Reads the systems a linear solver dumped (data line ``dumps=every`` or ``dumps=<solve numbers>``, see
:mod:`edelweissfe.linsolve.composition.solvedumps`) and replays them, so solver variants are compared
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

    for record in records:
        # the file names are A_<instance>_<solve>.npz
        record["instance"] = int(record["matrixFile"].split("_")[1])
        record["ordinal"] = record["solveCount"]
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
                np.linalg.norm(b),
                np.linalg.norm(A.diagonal()),
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


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    subparsers = parser.add_subparsers(dest="command", required=True)

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
