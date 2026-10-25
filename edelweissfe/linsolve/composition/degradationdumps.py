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
"""Writing the systems of slow solves to disk: :class:`DegradationDumps`.

A preconditioner that degrades -- a reused hierarchy that no longer fits the Jacobian, a damaged operator AMG cannot
handle -- typically shows only tens of hours into a nonlinear analysis, at a solve nobody can name in advance. A
degradation dump decides by the solve's own outcome: a solve that needs more outer iterations than a threshold, or
stops at the iteration cap short of its tolerance, has its raw system :math:`(A, b)` written, with its field layout
and its bookkeeping, so that it can be replayed offline through any solver (the workbench's ``bench.py``).

One snapshot cannot tell an intrinsically hard operator from a solve made slow by its own state (a stale
preconditioner): a fresh solver replaying the matrix alone may converge in a fraction of the live iterations. So a
dump can also hold the solves that *preceded* the slow one (``contextSolves``), for a replay through one persistent
solver in the same order; and every dumped solve records why it was (not) rebuilt.

On disk, in ``directory``: ``A_<instance>_<solve>.npz`` (SciPy, uncompressed), ``b_<instance>_<solve>.npy``,
``schurDofs_<instance>_<solve>.npy`` (the condensed dofs of a Schur complement, if any) and one line per dumped solve
in ``manifest.jsonl``.
"""

import collections
import json
import os

import numpy as np
import scipy.sparse as sp


class DegradationDumps:
    """The degradation dumps of a composed solver (see the module documentation), and of
    :class:`~edelweissfe.linsolve.blockamg.blockamg.BlockAMGSolver` (``dumpOnDegradation*``), in one on-disk format.

    Parameters
    ----------
    directory
        Where the dumps go; created if missing.
    threshold
        A solve with more outer iterations than this is dumped; ``None``: the warning threshold of the solver
        (``warnOuterIterations``), so that a dump and its warning fire on the same solves. A solve that stops at the
        iteration cap short of its tolerance is dumped whatever the threshold.
    maxDumps
        The most systems written in this process, across all solvers (e.g. one per analysis step), context solves
        included: a disk-space guard for a run that degrades on every remaining solve.
    contextSolves
        How many of the solves preceding a dumped one are dumped with it (``0``: only the slow solve itself). They are
        kept in memory, as copies, only while this is positive.
    """

    #: The systems written in this process, by every solver: what ``maxDumps`` bounds.
    dumpsWritten = 0

    def __init__(self, directory: str, threshold: int = None, maxDumps: int = 10, contextSolves: int = 0):
        if threshold is not None and threshold < 0:
            raise ValueError("DegradationDumps: threshold must not be negative, got {!r}".format(threshold))
        if maxDumps < 0 or contextSolves < 0:
            raise ValueError("DegradationDumps: maxDumps and contextSolves must not be negative")
        self.threshold = threshold
        self.maxDumps = maxDumps
        #: The last ``contextSolves`` solves, oldest first: ``{"solveCount", "A", "b", "blocks", "state",
        #: "condensedDofs"}``.
        self.history = collections.deque(maxlen=contextSolves)
        #: The solves already written, so that overlapping context windows never write (or count) one twice.
        self.dumpedSolves = set()
        self.directory = None
        self.moveTo(directory)

    @property
    def contextSolves(self) -> int:
        return self.history.maxlen

    def keepContextSolves(self, contextSolves: int) -> None:
        """Keep this many preceding solves from now on (the newest of those kept so far stay)."""
        self.history = collections.deque(self.history, maxlen=max(contextSolves, 0))

    def moveTo(self, directory: "str | None") -> None:
        """Write the dumps to ``directory`` from now on (created if missing; ``None``: write none)."""
        self.directory = directory
        if directory is not None:
            os.makedirs(directory, exist_ok=True)

    def thresholdOr(self, warningThreshold: int) -> int:
        """The outer iterations above which a solve is dumped: :attr:`threshold`, or ``warningThreshold``."""
        return warningThreshold if self.threshold is None else self.threshold

    def afterSolve(
        self,
        instance: int,
        solveCount: int,
        A: sp.csr_matrix,
        b: np.ndarray,
        blocks: list,
        state: dict,
        condensedDofs: "np.ndarray | None",
        degraded: bool,
        log,
    ) -> None:
        """Take a finished solve: dump it, with its context window, if it is ``degraded`` and the budget allows, then
        keep it (copied) as context for the next ones.

        Parameters
        ----------
        instance, solveCount
            The ordinal of the solver in this process and of the solve: the file names.
        A, b
            The raw system as handed to the solver.
        blocks
            Its :class:`~edelweissfe.linsolve.base.FieldBlock` layout.
        state
            The solve's bookkeeping, recorded in the manifest (``outerIters``, ``trueResidual``, ``eta``,
            ``continuations``, ``info``, ``mustRefresh``, ``patternChanged``, ``newIncrement``, ``previousOuterIters``).
        condensedDofs
            The dofs a Schur complement condensed in this solve, or ``None``.
        degraded
            Whether the solve is dumped (see :meth:`thresholdOr`).
        log
            ``(level, message) -> None``.
        """
        if self.directory is None:
            return
        if degraded and DegradationDumps.dumpsWritten < self.maxDumps:
            if solveCount not in self.dumpedSolves:
                self._write(instance, solveCount, A, b, blocks, "trigger", solveCount, state, condensedDofs, log)
            for entry in self.history:
                if DegradationDumps.dumpsWritten >= self.maxDumps:
                    break
                if entry["solveCount"] in self.dumpedSolves:
                    continue
                self._write(
                    instance,
                    entry["solveCount"],
                    entry["A"],
                    entry["b"],
                    entry["blocks"],
                    "context",
                    solveCount,
                    entry["state"],
                    entry["condensedDofs"],
                    log,
                )
        # kept after the dump above, so that a trigger's context window holds only the solves strictly before it
        if self.contextSolves > 0:
            self.history.append(
                {
                    "solveCount": solveCount,
                    # copied, not referenced: the nonlinear solver hands the same matrix and residual buffer back every
                    # Newton iteration, so references would all alias the trigger-time values
                    "A": A.copy(),
                    "b": np.array(b, copy=True),
                    "blocks": blocks,
                    "state": state,
                    "condensedDofs": condensedDofs,
                }
            )

    def _write(self, instance, solveCount, A, b, blocks, role, triggerSolveCount, state, condensedDofs, log) -> None:
        """Write one system and its manifest line."""
        stem = "{:02d}_{:05d}".format(instance, solveCount)
        matrixFile, rhsFile = "A_{:}.npz".format(stem), "b_{:}.npy".format(stem)
        # uncompressed: these systems are large, and compressing would take longer than the slow solve itself
        sp.save_npz(os.path.join(self.directory, matrixFile), A, compressed=False)
        np.save(os.path.join(self.directory, rhsFile), np.asarray(b))
        record = {
            "solveCount": solveCount,
            "role": role,
            "triggerSolveCount": triggerSolveCount,
            "contextOffset": triggerSolveCount - solveCount,
            "rows": int(A.shape[0]),
            "nnz": int(A.nnz),
            "matrixFile": matrixFile,
            "rhsFile": rhsFile,
            "blocks": [
                {"name": block.name, "start": block.start, "stop": block.stop, "dimension": block.dimension}
                for block in blocks
            ],
            **state,
        }
        if condensedDofs is not None:
            record["condensedDofsFile"] = "schurDofs_{:}.npy".format(stem)
            np.save(os.path.join(self.directory, record["condensedDofsFile"]), condensedDofs)
        with open(os.path.join(self.directory, "manifest.jsonl"), "a") as manifestFile:
            manifestFile.write(json.dumps(record) + "\n")
        self.dumpedSolves.add(solveCount)
        DegradationDumps.dumpsWritten += 1
        log(
            "warning",
            "dumped {:} solve #{:} (of trigger #{:}) to {:}; {:} of {:} degradation dumps used".format(
                role, solveCount, triggerSolveCount, matrixFile, DegradationDumps.dumpsWritten, self.maxDumps
            ),
        )
