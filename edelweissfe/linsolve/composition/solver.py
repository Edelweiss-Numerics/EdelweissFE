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
"""The one adapter between a nonlinear solver and a composed solver: :class:`ComposedLinearSolver`, and the solver
script of ``*linearSolver, type=custom`` (:func:`loadSolverScript`).

The nonlinear solver calls its linear solver as ``x = solver(A, b)`` once per Newton iteration.
:class:`ComposedLinearSolver` turns that call into the life cycle of a composed solver: it finds what changed since
the last call (a new pattern, a resized system, a jump of the right-hand side that marks a new increment),
hands the changed matrix down the tree (:meth:`~edelweissfe.linsolve.composition.base.Preconditioner.update`), and
applies the root to :math:`b`. It is also what reports on the solves: the journal, the warnings, the dumps
(:mod:`~edelweissfe.linsolve.composition.solvedumps`), and the hot reload of a solver script. Every type of linear
solver is a tree behind this adapter (:mod:`edelweissfe.linsolve.types`); a direct solver is a tree of one leaf.

In an input file, a solver script is the ``custom`` type::

    *linearSolver, name=mySolver, type=custom
    script=solver.py

    *solver, solver=NIST, name=theSolver
    linsolver=mySolver

``solver.py`` builds the tree with plain constructors and binds it to the global ``linearSolver``; it holds only the
math, the adapter's settings (verbosity, dumps, hot reload) are data lines of the block::

    linearSolver = LGMRES(DiagonalScaling(...))
"""

import dataclasses
import hashlib
import runpy
import time

import numpy as np

import edelweissfe.utils.performancetiming as performancetiming
from edelweissfe.linsolve.base import FieldBlock, LinearSolver, LinearSolveSummary
from edelweissfe.linsolve.composition.base import (
    MatrixChange,
    Preconditioner,
    SetUpContext,
)
from edelweissfe.linsolve.composition.solvedumps import SolveDumps
from edelweissfe.linsolve.composition.systemmatrix import SystemMatrix

_IDENTIFICATION = "ComposedSolver"
_VERBOSITY_LEVELS = ("silent", "warning", "info", "debug")
_WARNING_MARKER = "⚠ "


def tileFieldBlocks(fieldStructure: "list | None", n: int, owner: str) -> list:
    """The field blocks tiling ``[0, n)``, in dof order, with any trailing dofs not covered by a node field (e.g.
    scalar variables) folded into a final scalar block.

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
        ``"info"`` (the structure of the solver at the first solve, in a frame, and the "linear solve" column of the
        nonlinear solver's table) or ``"debug"`` (also each continuation of a Krylov solve, and why the
        preconditioner was rebuilt or reused, in that column).
    warnOuterIterations
        A solve with more outer iterations than this gets a warning (a sign that the preconditioner no longer fits).
        A solve that misses its tolerance gets a warning of its own, whatever this says.
    dumps
        A :class:`~edelweissfe.linsolve.composition.solvedumps.SolveDumps`: write the systems of chosen solves to disk,
        for an offline replay. ``None`` (default): none.
    nodeCoordinates
        ``{fieldName: (nNodes, dimension) array}``, coordinates to use instead of the model's (an offline replay).
    p1Maps
        ``{fieldName: (isCorner, edgeEndpoints)}``, P1 maps to use instead of those derived from the model.
    """

    identification = _IDENTIFICATION

    #: Instances created in this process: the ordinal of an instance names its dumps, so that the dumps of
    #: several solvers (one per analysis step, say) never overwrite each other.
    _instancesCreated = 0

    def __init__(
        self,
        root: Preconditioner,
        residualGrowthFactor: float = 4.0,
        verbosity: str = "warning",
        warnOuterIterations: int = 100,
        dumps: SolveDumps = None,
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
        self.root = root
        self.residualGrowthFactor = residualGrowthFactor
        self.verbosity = verbosity
        self.warnOuterIterations = warnOuterIterations
        self.dumps = dumps
        self._givenCoordinates = {
            name: np.asarray(coordinates, dtype=float) for name, coordinates in (nodeCoordinates or {}).items()
        }
        self._givenP1Maps = dict(p1Maps or {})
        #: The solver script the tree was built by (see :meth:`fromScript`), ``None`` for a tree built in Python.
        self.script = None
        #: Whether the tree is built again whenever the solver script changes on disk (see :meth:`_hotReload`).
        self.hotReload = False
        # the contents of the script the tree was last built from
        self._scriptStamp = None
        # the tree a hot reload replaced, until the new tree has been set up (see _hotReload)
        self._rootBeforeReload = None
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

    @classmethod
    def fromScript(cls, script: str, hotReload: bool = False, **settings) -> "ComposedLinearSolver":
        """The composed solver of a solver script (see :func:`loadSolverScript`) behind the adapter.

        Parameters
        ----------
        script
            The solver script.
        hotReload
            Build the tree again whenever the script changed on disk, before a solve, and continue with it (see
            :meth:`_hotReload`).
        **settings
            The settings of the adapter, see :class:`ComposedLinearSolver`.
        """
        with open(script, "rb") as scriptFile:
            stamp = scriptFile.read()
        solver = cls(loadSolverScript(script), **settings)
        solver.script = script
        solver.hotReload = hotReload
        solver._scriptStamp = stamp
        return solver

    # ----- what the nonlinear solver hands over -----

    def setModel(self, model, dofManager) -> None:
        """Take the model: the field layout (base class), and for the tree what is derived from it. What the tree built
        for the previous model is forgotten right away: the tree is built anew at the next solve, and the nonlinear
        solver builds the new equation system before that."""
        self.root.forget()
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
    def needsStableSparsityPattern(self) -> bool:
        """A Krylov solver reuses its preconditioner across solves, which a new sparsity pattern defeats; a direct
        solve factorizes anew every time."""
        return self.root.containsKrylovSolver

    @property
    def reportsSolveSummary(self) -> bool:
        """The "linear solve" column of the nonlinear solver: the outer iterations of a Krylov solve, at verbosity
        ``"info"`` or ``"debug"``."""
        return self.verbosity in ("info", "debug") and self.root.containsKrylovSolver

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

    def _logFramed(self, level: str, title: str, message: str) -> None:
        """Show ``message`` in a frame with ``title`` if the verbosity reaches ``level``: in the journal, or printed
        without one."""
        if _VERBOSITY_LEVELS.index(self.verbosity) < _VERBOSITY_LEVELS.index(level):
            return
        if self._journal is not None:
            self._journal.framedMessage(message, _IDENTIFICATION, title, level=2)
        else:
            print(title + ":\n" + message, flush=True)

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

    def _notSolved(self, shape: tuple, reason: str) -> np.ndarray:
        """A declined solve: NaN, which reaches the nonlinear solver as a diverged solution (a cutback)."""
        self._log("warning", "solve #{:}: {:}; returning NaN without solving.".format(self._solveCount, reason))
        self._lastSolveSummary = LinearSolveSummary(iters=0, residual=float("nan"), residualMet=False, retries=0)
        return np.full(shape, np.nan)

    def __call__(self, A, b):
        """Solve :math:`A x = b`, for one right-hand side ``b`` (a vector) or several (the columns of a matrix, e.g.
        the two of an arc-length method), each with the same set-up of the tree."""
        start = time.time()
        self._solveCount += 1
        if self.hotReload:
            self._hotReload()
        A = SystemMatrix(A)
        n = A.shape[0]
        # a direct solve needs no fields: a caller may not hand over any (EdelweissMeshfree's solvers)
        fields = [] if self._fieldStructure is None else tileFieldBlocks(self._fieldStructure, n, _IDENTIFICATION)
        b = np.asarray(b, dtype=float)
        rightHandSides = b.reshape(n, -1)
        residualNorm = float(np.linalg.norm(b))
        if not np.isfinite(residualNorm):
            return self._notSolved(b.shape, "the right-hand side is not finite")

        change = self._changeFor(A, fields, residualNorm)
        try:
            self.root.update(A, change)
        except (FloatingPointError, np.linalg.LinAlgError) as error:
            # a non-finite scaling, a singular matrix: the nonlinear solver cuts the increment back, and what the
            # tree formed from A goes before the next matrix is assembled
            self.root.finishSolve()
            return self._notSolved(b.shape, str(error))
        except Exception as error:
            if self._rootBeforeReload is None:
                raise
            self._rollBackHotReload(error)
            change = self._changeFor(A, fields, residualNorm)
            self.root.update(A, change)
        self._rootBeforeReload = None
        if self._fields is None or self._reloaded:
            # after the first set-up: a combination made per field shows its corrections only then
            self._logFramed("info", "composed linear solver", str(self.root) + self._scriptLine())
        x = np.column_stack([self.root(np.ascontiguousarray(r)) for r in rightHandSides.T]).reshape(b.shape)
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
        else:
            self._lastSolveSummary = None
        if self.dumps is not None:
            self._dump(A.A, b, fields, record, change, patternChanged, condensedDofs)
        if record is not None:
            self._previousIterations = record.iterations
        return x

    def _report(self, record, change: MatrixChange, seconds: float) -> None:
        """The summary of a solve for the nonlinear solver's table, its line in the journal, and its warnings."""
        detailLines = ()
        if self.verbosity == "debug":
            rebuilt = "rebuilt({:})".format(record.rebuildReason) if record.rebuildReason is not None else "reused"
            detailLines = (rebuilt, "η={:.1e} · {:.1f}s".format(record.eta, seconds))
            if record.rebuildReason is not None:
                detailLines += self.root.setUpNotes()
        self._lastSolveSummary = LinearSolveSummary(
            iters=record.iterations,
            residual=record.residual,
            residualMet=record.converged,
            retries=record.continuations,
            detailLines=detailLines,
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
        """Hand a finished solve to the dumps, with its bookkeeping -- that of its Krylov solve in the names of its
        :class:`~edelweissfe.linsolve.composition.krylov.SolveRecord`, if it has one (a direct solve has none, and is
        never slow) -- and the geometry of its fields that a replay of the tree may need (:meth:`_dumpedGeometry`)."""
        state = {"patternChanged": patternChanged, "newIncrement": change.newIncrement}
        slow = False
        if record is not None:
            state = {**dataclasses.asdict(record), **state, "previousIterations": self._previousIterations}
            slow = record.iterations > self.dumps.thresholdOr(self.warnOuterIterations) or (
                record.capHit and not record.converged
            )
        self.dumps.afterSolve(
            self._instanceOrdinal,
            self._solveCount,
            A,
            b,
            fields,
            state,
            condensedDofs,
            self._dumpedGeometry(fields),
            slow=slow,
            log=self._log,
        )

    def _dumpedGeometry(self, fields: list) -> dict:
        """What a replay without the model needs of the fields (``nodeCoordinates`` and ``p1Maps`` of the adapter):
        the node coordinates of every vector field, and the P1 maps the tree asked for, as ``{name: array}``."""
        geometry = {}
        for field in fields:
            coordinates = self.coordinatesOf(field.name) if field.dimension > 1 else None
            if coordinates is not None:
                geometry[field.name + "_coordinates"] = coordinates
        for name, (isCorner, edgeEndpoints) in {**self._p1MapCache, **self._givenP1Maps}.items():
            geometry[name + "_isCorner"], geometry[name + "_edgeEndpoints"] = isCorner, edgeEndpoints
        return geometry

    # ----- the solver script -----

    def _scriptLine(self) -> str:
        if self.script is None:
            return ""
        return "\nfrom {:} (sha256 {:})".format(self.script, _digest(self._scriptStamp))

    def _hotReload(self) -> None:
        """Before a solve: if the solver script changed on disk, run it again and continue with the tree it builds;
        what the run has learnt stays (the solve count, the residual history, the numbering of the dumps, so that a
        reload never overwrites earlier dumps), and so do the settings of the adapter (data lines of the input file).
        The new tree is set up on this solve (rebuild reason ``"reload"``), after the tree in force has forgotten what
        it built (it is kept only to roll back to).

        Editing a file a running process reads is racy, so this fails safely, and nothing about a bad edit can end
        the run: a script that cannot be read or does not compile (e.g. half written by an editor) is retried on the
        next solve; one that runs into an error, or defines no composed solver, is reported and the tree in force is
        kept, until the script changes again; a new tree whose set-up fails is rolled back to the tree in force
        (:meth:`_rollBackHotReload`). The contents themselves are the stamp, not the modification time, which coarse
        file timestamps can leave unchanged across two quick edits.
        """
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
            tree = loadSolverScript(self.script)
        except Exception as error:
            self._log(
                "warning",
                "{:}: REJECTED, {:}: {:}; the solver in force is kept".format(header, type(error).__name__, error),
            )
            return
        # the tree in force is kept to roll back to, but not what it built: never two built trees at once
        self.root.forget()
        self._rootBeforeReload = self.root
        self.root = tree
        if self._model is not None:
            self.root.setModel(self._model, self._dofManager)
        self._reloaded = True
        self._log("warning", "{:}: the tree is replaced; it is set up on this solve:\n{:}".format(header, self.root))

    def _rollBackHotReload(self, error: Exception) -> None:
        """Back to the tree in force before the last hot reload, whose tree failed to set up (e.g. an AMGCL tree naming
        an unknown smoother, which only the build can judge). The script stays recorded as seen, so it is not run again
        until it changes. That tree forgot what it built at the reload, and is set up again on this solve (rebuild
        reason ``"reload"``)."""
        self.root.forget()
        self.root = self._rootBeforeReload
        self._rootBeforeReload = None
        self._log(
            "warning",
            "hot reload of {:} at solve #{:} ROLLED BACK, the new tree failed to set up: {:}: {:}; the tree in force "
            "before it is kept".format(self.script, self._solveCount, type(error).__name__, error),
        )


def _digest(contents: bytes) -> str:
    """The short hash of a solver script, by which the journal names the version in force."""
    return hashlib.sha256(contents).hexdigest()[:12]


def loadSolverScript(path: str) -> Preconditioner:
    """The composed solver a solver script defines: the tree of components it binds to its global ``linearSolver``.

    The script holds only the math, e.g. ``linearSolver = LGMRES(DiagonalScaling(...))``; the settings of the adapter
    (verbosity, dumps, hot reload) are data lines of its ``*linearSolver, type=custom`` block.

    Parameters
    ----------
    path
        The script.
    """
    namespace = runpy.run_path(path)
    if "linearSolver" not in namespace:
        raise ValueError("solver script {:} defines no 'linearSolver'".format(path))
    tree = namespace["linearSolver"]
    if isinstance(tree, LinearSolver):
        raise TypeError(
            "solver script {:}: 'linearSolver' is a {:}, but a script holds only the math: bind the tree itself, "
            "linearSolver = LGMRES(...); verbosity, dumps and hotReload are data lines of *linearSolver".format(
                path, type(tree).__name__
            )
        )
    if not isinstance(tree, Preconditioner):
        raise TypeError(
            "solver script {:}: 'linearSolver' is a {:}, not a composed solver: bind a tree of components, e.g. "
            "linearSolver = LGMRES(AMG())".format(path, type(tree).__name__)
        )
    return tree
