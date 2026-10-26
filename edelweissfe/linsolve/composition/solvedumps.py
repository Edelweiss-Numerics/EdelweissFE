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
"""Writing the systems of chosen solves to disk, for an offline replay: :class:`SolveDumps`.

Comparing solver variants by rerunning the simulation is slow and confounded: every run repeats the assembly, and a
variant that changes the Newton iterates changes the sequence of matrices being compared. Dumping authentic systems
:math:`(A, b)` once and replaying them offline -- through a
:class:`~edelweissfe.linsolve.composition.solver.ComposedLinearSolver` with ``setFieldStructure`` and the dumped
``nodeCoordinates`` and ``p1Maps`` -- makes the comparison fast and controlled.

Which solves are dumped is the trigger:

- ``"every"``: every solve (a sequence: whether a symbolic factorization or a preconditioner can be reused across
  Newton iterations is a question about a sequence, not about one matrix);
- solve numbers, e.g. ``{12, 13, 14}``: these solves of each solver (counted from 1);
- ``"slow"``: a solve that needs more outer iterations than a threshold, or stops at the iteration cap short of its
  tolerance. A preconditioner that degrades -- a reused hierarchy that no longer fits the Jacobian, a damaged operator
  AMG cannot handle -- typically shows only tens of hours into an analysis, at a solve nobody can name in advance; this
  trigger decides by the solve's own outcome.

One snapshot cannot tell an intrinsically hard operator from a solve made slow by its own state (a stale
preconditioner): a fresh solver replaying the matrix alone may converge in a fraction of the live iterations. So a
dump can also hold the solves that *preceded* a dumped one (``contextSolves``), for a replay through one persistent
solver in the same order; and every dumped solve records why it was (not) rebuilt.

``maxDumps`` caps the systems written to one directory in a process: a coupled 3D system of a million dofs takes
gigabytes on disk. The solvers of all analysis steps that dump to the same directory share that cap.

On disk, in ``directory``: ``A_<instance>_<solve>.npz`` (SciPy, uncompressed), ``b_<instance>_<solve>.npy``,
``schurDofs_<instance>_<solve>.npy`` (the condensed dofs of a Schur complement, if any),
``geometry_<instance>_<solve>.npz`` (the node coordinates of the vector fields, ``<field>_coordinates``, and the P1 maps
the tree used, ``<field>_isCorner`` and ``<field>_edgeEndpoints``, if any) and one line per dumped solve in
``manifest.jsonl``: the field blocks and file names, and the bookkeeping of a Krylov solve in the names of its
:class:`~edelweissfe.linsolve.composition.krylov.SolveRecord` (``iterations``, ``residual``, ``eta``, ``capHit``,
``continuations``, ``rebuildReason``, ``scaledSystem``), with ``patternChanged``, ``newIncrement`` and
``previousIterations``. ``<instance>`` is the ordinal of the solver in the process (one per analysis step).
"""

import collections
import json
import os

import numpy as np
import scipy.sparse as sp


class SolveDumps:
    """The dumps of a composed solver, see the module documentation; the data lines ``dumps``, ``dumpDirectory``,
    ``maxDumps`` and ``dumpContextSolves`` of every type of linear solver.

    Parameters
    ----------
    directory
        Where the dumps go; created if missing.
    trigger
        Which solves are dumped: ``"slow"`` (default), ``"every"``, or a collection of solve numbers.
    threshold
        For ``"slow"``: a solve with more outer iterations than this is dumped; ``None``: the warning threshold of the solver
        (``warnOuterIterations``), so that a dump and its warning fire on the same solves. A solve that stops at the
        iteration cap short of its tolerance is dumped whatever the threshold.
    maxDumps
        The most systems written to ``directory`` in this process, across all solvers that dump there (e.g. one per
        analysis step), context solves included: a disk-space guard for a run that degrades on every remaining solve.
    contextSolves
        How many of the solves preceding a dumped one are dumped with it (``0``: only the slow solve itself). They are
        kept in memory, as copies, only while this is positive.
    """

    #: The systems written to each directory (by absolute path) in this process, by every solver: what ``maxDumps``
    #: bounds.
    dumpsWrittenTo: dict = {}

    def __init__(
        self, directory: str, trigger="slow", threshold: int = None, maxDumps: int = 10, contextSolves: int = 0
    ):
        if isinstance(trigger, str):
            if trigger not in ("slow", "every"):
                raise ValueError(
                    "SolveDumps: trigger must be 'slow', 'every' or solve numbers, got {!r}".format(trigger)
                )
            self.trigger = trigger
        else:
            self.trigger = frozenset(int(solveCount) for solveCount in trigger)
        if threshold is not None and threshold < 0:
            raise ValueError("SolveDumps: threshold must not be negative, got {!r}".format(threshold))
        if maxDumps < 0 or contextSolves < 0:
            raise ValueError("SolveDumps: maxDumps and contextSolves must not be negative")
        self.threshold = threshold
        self.maxDumps = maxDumps
        #: The last ``contextSolves`` solves, oldest first: ``{"solveCount", "A", "b", "blocks", "state",
        #: "condensedDofs", "geometry"}``.
        self.history = collections.deque(maxlen=contextSolves)
        #: The solves already written, so that overlapping context windows never write (or count) one twice.
        self.dumpedSolves = set()
        self.directory = None
        self.moveTo(directory)

    @property
    def dumpsWritten(self) -> int:
        """The systems written to :attr:`directory` in this process, by every solver that dumps there."""
        return SolveDumps.dumpsWrittenTo.get(os.path.abspath(self.directory), 0)

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
        """The outer iterations above which a solve is slow: :attr:`threshold`, or ``warningThreshold``."""
        return warningThreshold if self.threshold is None else self.threshold

    def selects(self, solveCount: int, slow: bool) -> bool:
        """Whether the trigger selects this solve, given whether it was slow."""
        if self.trigger == "slow":
            return slow
        if self.trigger == "every":
            return True
        return solveCount in self.trigger

    def afterSolve(
        self,
        instance: int,
        solveCount: int,
        A: sp.csr_matrix,
        b: np.ndarray,
        blocks: list,
        state: dict,
        condensedDofs: "np.ndarray | None",
        geometry: dict,
        slow: bool,
        log,
    ) -> None:
        """Take a finished solve: dump it, with its context window, if the trigger selects it and the budget allows,
        then keep it (copied) as context for the next ones.

        Parameters
        ----------
        instance, solveCount
            The ordinal of the solver in this process and of the solve: the file names.
        A, b
            The raw system as handed to the solver.
        blocks
            Its :class:`~edelweissfe.linsolve.base.FieldBlock` layout.
        state
            The solve's bookkeeping, recorded in the manifest (see the module documentation).
        condensedDofs
            The dofs a Schur complement condensed in this solve, or ``None``.
        geometry
            ``{name: array}``: the node coordinates and P1 maps of the fields, written to ``geometry_*.npz`` (nothing
            if empty).
        slow
            Whether the solve was slow (see :meth:`thresholdOr`), for the trigger ``"slow"``.
        log
            ``(level, message) -> None``.
        """
        if self.directory is None:
            return
        if self.selects(solveCount, slow) and self.dumpsWritten < self.maxDumps:
            if solveCount not in self.dumpedSolves:
                self._write(
                    instance, solveCount, A, b, blocks, "trigger", solveCount, state, condensedDofs, geometry, log
                )
            for entry in self.history:
                if self.dumpsWritten >= self.maxDumps:
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
                    entry["geometry"],
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
                    "geometry": geometry,
                }
            )

    def _write(
        self, instance, solveCount, A, b, blocks, role, triggerSolveCount, state, condensedDofs, geometry, log
    ) -> None:
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
        if geometry:
            record["geometryFile"] = "geometry_{:}.npz".format(stem)
            np.savez(os.path.join(self.directory, record["geometryFile"]), **geometry)
        with open(os.path.join(self.directory, "manifest.jsonl"), "a") as manifestFile:
            manifestFile.write(json.dumps(record) + "\n")
        self.dumpedSolves.add(solveCount)
        SolveDumps.dumpsWrittenTo[os.path.abspath(self.directory)] = self.dumpsWritten + 1
        log(
            "warning",
            "dumped {:} solve #{:} (of trigger #{:}) to {:}; {:} of {:} dumps used".format(
                role, solveCount, triggerSolveCount, matrixFile, self.dumpsWritten, self.maxDumps
            ),
        )
