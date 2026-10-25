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
"""A composed solver as the linear solver of a nonlinear solver: :class:`ComposedLinearSolver`, and the solver
script of an input file (:func:`loadSolverScript`).

The nonlinear solver calls its linear solver as ``x = solver(A, b)`` once per Newton iteration.
:class:`ComposedLinearSolver` turns that call into the life cycle of a composed solver: it finds what changed since
the last call (a new pattern, a resized system, a jump of the right-hand side that marks a new increment),
hands the changed matrix down the tree (:meth:`~edelweissfe.linsolve.composition.base.Preconditioner.update`), and
applies the root to :math:`b`. It is also what reports on the solves: the journal, the warnings, the degradation
dumps (:mod:`~edelweissfe.linsolve.composition.degradationdumps`), and the hot reload of its script.

In an input file, a solver script is named as the linear solver::

    *solver, solver=NIST, name=theSolver
    linsolver=script:solver.py:linearSolver

``solver.py`` builds the tree with plain constructors and binds the driver around it to ``linearSolver``::

    linearSolver = ComposedLinearSolver(LGMRES(DiagonalScaling(...)))
"""

import hashlib
import runpy
import time
from dataclasses import dataclass, field

import numpy as np

import edelweissfe.utils.performancetiming as performancetiming
from edelweissfe.linsolve.base import FieldBlock, LinearSolver, LinearSolveSummary
from edelweissfe.linsolve.composition.base import (
    MatrixChange,
    Preconditioner,
    SetUpContext,
)
from edelweissfe.linsolve.composition.degradationdumps import DegradationDumps
from edelweissfe.linsolve.composition.systemmatrix import SystemMatrix

_IDENTIFICATION = "ComposedLinearSolver"
_VERBOSITY_LEVELS = ("silent", "warning", "info", "debug")
_WARNING_MARKER = "⚠ "


def tileFieldBlocks(fieldStructure: "list | None", n: int, owner: str) -> list:
    """The field blocks tiling ``[0, n)``, in dof order, with any trailing dofs not covered by a node field (e.g.
    scalar variables) folded into a final scalar block. Shared with
    :class:`~edelweissfe.linsolve.blockamg.blockamg.BlockAMGSolver`.

    Parameters
    ----------
    fieldStructure
        The field blocks of the model, as :meth:`~edelweissfe.linsolve.base.LinearSolver.setModel` derived them.
    n
        The number of dofs.
    owner
        The name error messages start with.
    """
    if fieldStructure is None:
        raise RuntimeError(
            "{:}: field structure not set. It is derived from setModel() (or, at the lower "
            "level, setFieldStructure()); this solver must be driven by a caller that calls one of "
            "those.".format(owner)
        )
    blocks = sorted(fieldStructure, key=lambda field: field.start)
    cursor = 0
    for block in blocks:
        if block.start != cursor:
            raise ValueError(
                "{:}: field '{:}' starts at {:}, expected {:} -- fields must tile the DOF "
                "vector contiguously".format(owner, block.name, block.start, cursor)
            )
        cursor = block.stop
    if cursor < n:
        # DOFs past the last node field: scalar variables. One scalar block.
        blocks = blocks + [FieldBlock("scalar variables", cursor, n, 1)]
    elif cursor != n:
        raise ValueError("{:}: field blocks cover {:} dofs, but the matrix is {:}x{:}".format(owner, cursor, n, n))
    return blocks


@dataclass(frozen=True)
class _Settings:
    """What a solver script decides about a :class:`ComposedLinearSolver`: its tree and the settings of the driver
    (see its parameters). A hot reload replaces all of them at once."""

    root: Preconditioner
    residualGrowthFactor: float
    verbosity: str
    warnOuterIterations: int
    degradationDumps: "DegradationDumps | None"
    hotReload: bool
    nodeCoordinates: dict = field(default_factory=dict)
    p1Maps: dict = field(default_factory=dict)


class ComposedLinearSolver(LinearSolver):
    """A composed solver (the ``root`` of a tree of preconditioners, usually a Krylov solver) as a
    :class:`~edelweissfe.linsolve.base.LinearSolver`, callable as ``x = solver(A, b)``: the driver of the tree.

    Per call it finds what changed and hands it down the tree as a
    :class:`~edelweissfe.linsolve.composition.base.MatrixChange`: ``"first"`` (nothing built yet), ``"model"``
    (after :meth:`setModel`), ``"reload"`` (a new tree from a hot reload), ``"amr"`` (another field layout or size),
    ``"pattern"`` (another number of stored entries, an O(1) proxy for the sparsity pattern), or only new values.
    Whether a right-hand side jumped by more than ``residualGrowthFactor`` (a new increment) is passed along too; the
    Krylov solver decides what it means.

    Parameters
    ----------
    root
        The composed solver.
    residualGrowthFactor
        A right-hand side this many times larger than the previous one marks a new increment (a cutback after a
        diverged attempt starts below its last residual, and is not one).
    verbosity
        ``"silent"``, ``"warning"`` (default: nothing on a normal solve; a warning for a slow or unconverged one),
        ``"info"`` (one line per solve, the structure of the solver at the first solve, and the "linear solve" column
        of the nonlinear solver's table) or ``"debug"`` (also each continuation of a Krylov solve, and why the
        preconditioner was rebuilt or reused, in that column).
    warnOuterIterations
        A solve with more outer iterations than this gets a warning (a sign that the preconditioner no longer fits).
        A solve that misses its tolerance gets a warning of its own, whatever this says.
    degradationDumps
        A :class:`~edelweissfe.linsolve.composition.degradationdumps.DegradationDumps`: write the systems of slow
        solves to disk, for an offline replay. ``None`` (default): none.
    hotReload
        Re-read the solver script whenever it changed on disk, before a solve, and continue with the solver it builds:
        tree and settings (see :meth:`_hotReload`). Only for a solver built by a script (:func:`loadSolverScript`).
    nodeCoordinates
        ``{fieldName: (nNodes, dimension) array}``, coordinates to use instead of the model's (an offline replay).
    p1Maps
        ``{fieldName: (isCorner, edgeEndpoints)}``, P1 maps to use instead of those derived from the model.
    """

    identification = _IDENTIFICATION
    needsStableSparsityPattern = True

    #: Instances created in this process: the ordinal of an instance names its degradation dumps, so that the dumps of
    #: several solvers (one per analysis step, say) never overwrite each other.
    _instancesCreated = 0

    def __init__(
        self,
        root: Preconditioner,
        residualGrowthFactor: float = 4.0,
        verbosity: str = "warning",
        warnOuterIterations: int = 100,
        degradationDumps: DegradationDumps = None,
        hotReload: bool = False,
        nodeCoordinates: dict = None,
        p1Maps: dict = None,
    ):
        if not isinstance(root, Preconditioner):
            raise TypeError(
                "ComposedLinearSolver: its root must be a composed solver (a Preconditioner such as LGMRES(...)), "
                "got {!r}".format(root)
            )
        if verbosity not in _VERBOSITY_LEVELS:
            raise ValueError(
                "ComposedLinearSolver: verbosity must be one of {:}, got {!r}".format(_VERBOSITY_LEVELS, verbosity)
            )
        if not residualGrowthFactor > 1.0:
            raise ValueError("ComposedLinearSolver: residualGrowthFactor must be greater than 1")
        if warnOuterIterations < 0:
            raise ValueError("ComposedLinearSolver: warnOuterIterations must not be negative")
        self._apply(
            _Settings(
                root,
                residualGrowthFactor,
                verbosity,
                warnOuterIterations,
                degradationDumps,
                hotReload,
                {name: np.asarray(coordinates, dtype=float) for name, coordinates in (nodeCoordinates or {}).items()},
                dict(p1Maps or {}),
            )
        )
        #: The solver script this solver was built by, its options, and the contents it was last built from (see
        #: :func:`loadSolverScript`); ``None`` for a solver built in Python.
        self.script = None
        self._scriptName = None
        self._scriptOptions = {}
        self._scriptStamp = None
        # the settings a hot reload replaced, until the new tree has been set up (see _hotReload)
        self._settingsBeforeReload = None
        self._reloaded = False
        self._coordinateCache = {}
        self._p1MapCache = {}
        #: The stored entries of the last matrix: a new count is a new sparsity pattern.
        self.lastNnz = None
        #: The right-hand side norm of the last solve, ``None`` before the first one and after a new model.
        self.lastResidualNorm = None
        self._fields = None
        self._modelChanged = False
        self._solveCount = 0
        self._previousIterations = None
        self._lastSolveSummary = None
        self._instanceOrdinal = ComposedLinearSolver._instancesCreated
        ComposedLinearSolver._instancesCreated += 1

    # ----- the settings -----

    def _settings(self) -> _Settings:
        return _Settings(
            self.root,
            self.residualGrowthFactor,
            self.verbosity,
            self.warnOuterIterations,
            self.degradationDumps,
            self.hotReload,
            self._givenCoordinates,
            self._givenP1Maps,
        )

    def _apply(self, settings: _Settings) -> None:
        self.root = settings.root
        self.residualGrowthFactor = settings.residualGrowthFactor
        self.verbosity = settings.verbosity
        self.warnOuterIterations = settings.warnOuterIterations
        self.degradationDumps = settings.degradationDumps
        self.hotReload = settings.hotReload
        self._givenCoordinates = settings.nodeCoordinates
        self._givenP1Maps = settings.p1Maps

    # ----- what the nonlinear solver hands over -----

    def setModel(self, model, dofManager) -> None:
        """Take the model: the field layout (base class), and for the tree what is derived from it."""
        super().setModel(model, dofManager)
        self._coordinateCache.clear()
        self._p1MapCache.clear()
        self._modelChanged = True
        self.lastResidualNorm = None
        self.root.setModel(model, dofManager)

    def coordinatesOf(self, fieldName: str) -> "np.ndarray | None":
        """The node coordinates of a field in its dof order: given ones first, else the model's, else ``None``."""
        if fieldName in self._givenCoordinates:
            return self._givenCoordinates[fieldName]
        if self._model is None or fieldName not in self._model.nodeFields:
            return None
        if fieldName not in self._coordinateCache:
            nodeField = self._model.nodeFields[fieldName]
            self._coordinateCache[fieldName] = np.array([node.coordinates for node in nodeField.nodes], dtype=float)
        return self._coordinateCache[fieldName]

    def p1MapOf(self, fieldName: str):
        """The P1 corner/midside map of a field: a given one first, else derived from the model, else ``None``."""
        if fieldName in self._givenP1Maps:
            return self._givenP1Maps[fieldName]
        if self._model is None:
            return None
        if fieldName not in self._p1MapCache:
            from edelweissfe.numerics.p1topology import buildP1Map

            with performancetiming.timeit("p1 topology"):
                isCorner, edgeEndpoints, warnings = buildP1Map(self._model, fieldName)
            for warning in warnings:
                self._log("warning", warning)
            self._p1MapCache[fieldName] = (isCorner, edgeEndpoints)
        return self._p1MapCache[fieldName]

    # ----- the solve -----

    @property
    def reportsSolveSummary(self) -> bool:
        return self.verbosity in ("info", "debug")

    @property
    def lastSolveSummary(self) -> "LinearSolveSummary | None":
        return self._lastSolveSummary

    def _log(self, level: str, message: str) -> None:
        """Show ``message`` if the verbosity reaches ``level``: in the journal, or printed without one. A warning is
        marked "⚠"."""
        if _VERBOSITY_LEVELS.index(self.verbosity) < _VERBOSITY_LEVELS.index(level):
            return
        if level == "warning" and not message.startswith(_WARNING_MARKER):
            message = _WARNING_MARKER + message
        if self._journal is not None:
            self._journal.message(message, _IDENTIFICATION, level=2)
        else:
            print(message, flush=True)

    def _changeFor(self, A: SystemMatrix, fields: list, residualNorm: float) -> MatrixChange:
        """What changed since the last solve."""
        context = SetUpContext(
            fields=tuple(fields),
            coordinatesOf=self.coordinatesOf,
            p1MapOf=self.p1MapOf,
            model=self._model,
            journal=self._journal,
            log=self._log,
        )
        if self._fields is None:
            reason = "first"
        elif self._modelChanged:
            reason = "model"
        elif self._reloaded:
            reason = "reload"
        elif fields != self._fields:
            reason = "amr"
        elif A.nnz != self.lastNnz:
            reason = "pattern"
        else:
            reason = None
        newIncrement = (
            self.lastResidualNorm is not None and residualNorm > self.residualGrowthFactor * self.lastResidualNorm
        )
        return MatrixChange(context, reason, newIncrement, residualNorm, self.lastResidualNorm)

    def _notSolved(self, n: int, reason: str) -> np.ndarray:
        """A declined solve: NaN, which reaches the nonlinear solver as a diverged solution (a cutback)."""
        self._log("warning", "solve #{:}: {:}; returning NaN without solving.".format(self._solveCount, reason))
        self._lastSolveSummary = LinearSolveSummary(iters=0, residual=float("nan"), residualMet=False, retries=0)
        return np.full(n, np.nan)

    def __call__(self, A, b):
        start = time.time()
        self._solveCount += 1
        if self.hotReload:
            self._hotReload()
        A = SystemMatrix(A)
        n = A.shape[0]
        fields = tileFieldBlocks(self._fieldStructure, n, _IDENTIFICATION)
        b = np.asarray(b, dtype=float).reshape(n)
        residualNorm = float(np.linalg.norm(b))
        if not np.isfinite(residualNorm):
            return self._notSolved(n, "the right-hand side is not finite")

        change = self._changeFor(A, fields, residualNorm)
        try:
            self.root.update(A, change)
        except FloatingPointError as error:
            return self._notSolved(n, str(error))
        except Exception as error:
            if self._settingsBeforeReload is None:
                raise
            self._rollBackHotReload(error)
            change = self._changeFor(A, fields, residualNorm)
            self.root.update(A, change)
        self._settingsBeforeReload = None
        if self._fields is None or self._reloaded:
            # after the first set-up: a combination made per field shows its corrections only then
            self._log("info", "composed linear solver:\n" + str(self.root) + self._scriptLine())
        x = self.root(b)
        record = self.root.solveRecord()
        condensedDofs = self.root.condensedDofs()
        self.root.finishSolve()

        patternChanged = self.lastNnz is not None and A.nnz != self.lastNnz
        self._fields = fields
        self._modelChanged = False
        self._reloaded = False
        self.lastNnz = A.nnz
        self.lastResidualNorm = residualNorm
        if record is not None:
            self._report(record, change, time.time() - start)
            if self.degradationDumps is not None:
                self._dump(A.A, b, fields, record, change, patternChanged, condensedDofs)
            self._previousIterations = record.iterations
        else:
            self._lastSolveSummary = None
        return x

    def _report(self, record, change: MatrixChange, seconds: float) -> None:
        """The summary of a solve for the nonlinear solver's table, its line in the journal, and its warnings."""
        detailLines = ()
        if self.verbosity == "debug":
            rebuilt = "rebuilt({:})".format(record.rebuildReason) if record.rebuildReason is not None else "reused"
            detailLines = (rebuilt, "η={:.1e} · {:.1f}s".format(record.eta, seconds))
        self._lastSolveSummary = LinearSolveSummary(
            iters=record.iterations,
            residual=record.residual,
            residualMet=record.converged,
            retries=record.continuations,
            detailLines=detailLines,
        )
        self._log(
            "info",
            "solve #{:}: {:} outer iterations, ‖r‖/‖b‖ = {:.1e}{:} · {:.1f}s".format(
                self._solveCount,
                record.iterations,
                record.residual,
                "" if record.rebuildReason is None else ", rebuilt ({:})".format(record.rebuildReason),
                seconds,
            ),
        )
        if record.iterations > self.warnOuterIterations:
            self._log(
                "warning",
                "solve #{:} needed {:} outer Krylov iterations, above the warning threshold of {:}. Likely a "
                "preconditioner-quality problem, not a physics one -- worth checking whether the Jacobian changed "
                "sharply since the last rebuild of the preconditioner.".format(
                    self._solveCount, record.iterations, self.warnOuterIterations
                ),
            )
        if record.converged:
            return
        if record.capHit:
            self._log(
                "warning",
                "solve #{:} did not converge: ‖r‖={:.1e} still exceeds η={:.1e} after {:} continuations, and a pass "
                "of the Krylov solver stopped at its iteration cap. The cap is a budget, not a tolerance: a solve that "
                "needs more is a preconditioning problem -- the preconditioner is rebuilt for the next "
                "solve.".format(self._solveCount, record.residual, record.eta, record.continuations),
            )
        elif record.scaledSystem:
            self._log(
                "warning",
                "solve #{:} did not fully converge: ‖r‖={:.1e} still exceeds η={:.1e} after {:} continuations, though "
                "the Krylov solve converged its own (scaled) residual: the gap between the scaled and the true "
                "residual, which more continuations or a GapCompensation close; a rebuilt preconditioner does "
                "not.".format(self._solveCount, record.residual, record.eta, record.continuations),
            )
        else:
            self._log(
                "warning",
                "solve #{:} did not fully converge: ‖r‖={:.1e} still exceeds η={:.1e}, though the Krylov solve "
                "reported convergence on the same residual (rounding).".format(
                    self._solveCount, record.residual, record.eta
                ),
            )

    def _dump(
        self, A, b: np.ndarray, fields: list, record, change: MatrixChange, patternChanged: bool, condensedDofs
    ) -> None:
        """Hand a finished solve to the degradation dumps, with the bookkeeping blockamg records."""
        state = {
            "outerIters": record.iterations,
            "trueResidual": record.residual,
            "eta": record.eta,
            "continuations": record.continuations,
            "info": int(record.capHit),
            "mustRefresh": record.rebuildReason is not None,
            "patternChanged": patternChanged,
            "newIncrement": change.newIncrement,
            "previousOuterIters": self._previousIterations,
        }
        self.degradationDumps.afterSolve(
            self._instanceOrdinal,
            self._solveCount,
            A,
            b,
            fields,
            state,
            condensedDofs,
            degraded=record.iterations > self.degradationDumps.thresholdOr(self.warnOuterIterations)
            or (record.capHit and not record.converged),
            log=self._log,
        )

    # ----- the solver script -----

    def _scriptLine(self) -> str:
        if self.script is None:
            return ""
        return "\nfrom {:} (sha256 {:})".format(self.script, _digest(self._scriptStamp))

    def _hotReload(self) -> None:
        """Before a solve: if the solver script changed on disk, run it again and continue with the solver it builds
        -- its tree and its settings, all of them; what the run has learnt stays (the solve count, the residual
        history, the numbering of the degradation dumps, so that a reload never overwrites earlier dumps). The new tree
        is set up on this solve (rebuild reason ``"reload"``).

        Editing a file a running process reads is racy, so this fails safely, and nothing about a bad edit can end
        the run: a script that cannot be read or does not compile (e.g. half written by an editor) is retried on the
        next solve; one that runs into an error, or builds no :class:`ComposedLinearSolver`, is reported and the solver
        in force is kept, until the script changes again; a new tree whose set-up fails is rolled back to the solver
        in force (:meth:`_rollBackHotReload`). The contents themselves are the stamp, not the modification time,
        which coarse file timestamps can leave unchanged across two quick edits.
        """
        if self.script is None:
            raise ValueError(
                "ComposedLinearSolver: hotReload re-reads the solver script, but this solver was not built by one "
                "(see loadSolverScript)"
            )
        try:
            with open(self.script, "rb") as scriptFile:
                stamp = scriptFile.read()
        except OSError as error:
            self._log("warning", "hot reload: cannot read {:}: {:}".format(self.script, error))
            return
        if stamp == self._scriptStamp:
            return
        try:
            compile(stamp, self.script, "exec")
        except (SyntaxError, ValueError) as error:
            self._log(
                "warning",
                "hot reload: {:} does not compile ({:}); keeping the solver in force and retrying on the next "
                "solve".format(self.script, error),
            )
            return
        self._scriptStamp = stamp
        header = "hot reload of {:} (sha256 {:}) at solve #{:}".format(self.script, _digest(stamp), self._solveCount)
        try:
            fresh = _runSolverScript(self.script, self._scriptName, self._scriptOptions)
        except Exception as error:
            self._log(
                "warning",
                "{:}: REJECTED, {:}: {:}; the solver in force is kept".format(header, type(error).__name__, error),
            )
            return
        self._settingsBeforeReload = self._settings()
        self._apply(fresh._settings())
        if self._model is not None:
            self.root.setModel(self._model, self._dofManager)
        self._reloaded = True
        self._log(
            "warning",
            "{:}: the whole solver is replaced, its settings included (verbosity {:}, residualGrowthFactor {:g}, "
            "warnOuterIterations {:}, degradationDumps {:}, hotReload {:}); it is set up on this solve:\n{:}".format(
                header,
                self.verbosity,
                self.residualGrowthFactor,
                self.warnOuterIterations,
                "none" if self.degradationDumps is None else self.degradationDumps.directory,
                self.hotReload,
                self.root,
            ),
        )

    def _rollBackHotReload(self, error: Exception) -> None:
        """Back to the solver in force before the last hot reload, whose tree failed to set up (e.g. an AMGCL tree
        naming an unknown smoother, which only the build can judge). The script stays recorded as seen, so it is not
        run again until it changes."""
        self._apply(self._settingsBeforeReload)
        self._settingsBeforeReload = None
        self._reloaded = False
        self._log(
            "warning",
            "hot reload of {:} at solve #{:} ROLLED BACK, the new solver failed to set up: {:}: {:}; the solver in force "
            "before it is kept".format(self.script, self._solveCount, type(error).__name__, error),
        )


def _digest(contents: bytes) -> str:
    """The short hash of a solver script, by which the journal names the version in force."""
    return hashlib.sha256(contents).hexdigest()[:12]


def _runSolverScript(path: str, name: str, options: dict) -> ComposedLinearSolver:
    """Run a solver script and return the solver it binds to ``name``."""
    namespace = runpy.run_path(path, init_globals={"options": options})
    if name not in namespace:
        raise ValueError("solver script {:} defines no '{:}'".format(path, name))
    solver = namespace[name]
    if isinstance(solver, Preconditioner):
        raise TypeError(
            "solver script {:}: '{:}' is a tree of components ({:}), not the solver: bind the driver around it, "
            "{:} = ComposedLinearSolver(<your tree>)".format(path, name, solver.describe().name, name)
        )
    if not isinstance(solver, ComposedLinearSolver):
        raise TypeError(
            "solver script {:}: '{:}' is a {:}, not a solver: bind {:} = ComposedLinearSolver(<your tree>)".format(
                path, name, type(solver).__name__, name
            )
        )
    return solver


def loadSolverScript(path: str, name: str = "linearSolver", options=None) -> ComposedLinearSolver:
    """The composed solver a solver script builds: the :class:`ComposedLinearSolver` it binds to ``name``.

    The script runs with the global ``options`` set to the parsed ``linsolverConfigFile`` of the solver (empty if it
    has none), so one script can serve several configurations.

    Parameters
    ----------
    path
        The script.
    name
        The global of the script that holds the solver.
    options
        The parsed ``linsolverConfigFile``.
    """
    options = dict(options or {})
    with open(path, "rb") as scriptFile:
        stamp = scriptFile.read()
    solver = _runSolverScript(path, name, options)
    solver.script = path
    solver._scriptName = name
    solver._scriptOptions = options
    solver._scriptStamp = stamp
    return solver
