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
applies the root to :math:`b`.

In an input file, a solver script is named as the linear solver::

    *solver, solver=NIST, name=theSolver
    linsolver=script:solver.py:linearSolver

``solver.py`` builds the tree with plain constructors and binds it to ``linearSolver``.
"""

import hashlib
import runpy
import time

import numpy as np

from edelweissfe.linsolve.base import FieldBlock, LinearSolver, LinearSolveSummary
from edelweissfe.linsolve.composition.base import (
    MatrixChange,
    Preconditioner,
    SetUpContext,
)
from edelweissfe.linsolve.composition.systemmatrix import SystemMatrix

_IDENTIFICATION = "ComposedLinearSolver"


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


class ComposedLinearSolver(LinearSolver):
    """A composed solver (the ``root`` of a tree of preconditioners, usually a Krylov solver) as a
    :class:`~edelweissfe.linsolve.base.LinearSolver`, callable as ``x = solver(A, b)``.

    Per call it finds what changed and hands it down the tree as a
    :class:`~edelweissfe.linsolve.composition.base.MatrixChange`: ``"first"`` (nothing built yet), ``"model"``
    (after :meth:`setModel`), ``"amr"`` (another field layout or size), ``"pattern"`` (another number of stored
    entries, an O(1) proxy for the sparsity pattern), or only new values. Whether a right-hand side jumped by more
    than ``residualGrowthFactor`` (a new increment) is passed along too; the Krylov solver decides what it means.

    Parameters
    ----------
    root
        The composed solver.
    residualGrowthFactor
        A right-hand side this many times larger than the previous one marks a new increment (a cutback after a
        diverged attempt starts below its last residual, and is not one).
    nodeCoordinates
        ``{fieldName: (nNodes, dimension) array}``, coordinates to use instead of the model's (an offline replay).
    p1Maps
        ``{fieldName: (isCorner, edgeEndpoints)}``, P1 maps to use instead of those derived from the model.
    verbosity
        ``"silent"``, ``"warning"`` (default), ``"info"`` (one line per solve, and the structure of the solver at the
        first solve) or ``"debug"``.
    script
        The path of the solver script it came from, if any, logged with its hash.
    """

    identification = _IDENTIFICATION
    needsStableSparsityPattern = True

    def __init__(
        self,
        root: Preconditioner,
        residualGrowthFactor: float = 4.0,
        nodeCoordinates: dict = None,
        p1Maps: dict = None,
        verbosity: str = "warning",
        script: str = None,
    ):
        self.root = root
        self.residualGrowthFactor = residualGrowthFactor
        self.verbosity = verbosity
        self.script = script
        self._givenCoordinates = dict(nodeCoordinates or {})
        self._givenP1Maps = dict(p1Maps or {})
        self._coordinateCache = {}
        self._p1MapCache = {}
        #: The stored entries of the last matrix: a new count is a new sparsity pattern.
        self.lastNnz = None
        #: The right-hand side norm of the last solve, ``None`` before the first one and after a new model.
        self.lastResidualNorm = None
        self._fields = None
        self._modelChanged = False
        self._solveCount = 0
        self._lastSolveSummary = None

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
            field = self._model.nodeFields[fieldName]
            self._coordinateCache[fieldName] = np.array([node.coordinates for node in field.nodes], dtype=float)
        return self._coordinateCache[fieldName]

    def p1MapOf(self, fieldName: str):
        """The P1 corner/midside map of a field: a given one first, else derived from the model, else ``None``."""
        if fieldName in self._givenP1Maps:
            return self._givenP1Maps[fieldName]
        if self._model is None:
            return None
        if fieldName not in self._p1MapCache:
            from edelweissfe.numerics.p1topology import buildP1Map

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
        levels = ("silent", "warning", "info", "debug")
        if levels.index(self.verbosity) < levels.index(level):
            return
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
        )
        if self._fields is None:
            reason = "first"
        elif self._modelChanged:
            reason = "model"
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
        self._log("warning", "⚠ solve #{:}: {:}; returning NaN without solving.".format(self._solveCount, reason))
        self._lastSolveSummary = LinearSolveSummary(iters=0, residual=float("nan"), residualMet=False, retries=0)
        return np.full(n, np.nan)

    def __call__(self, A, b):
        start = time.time()
        self._solveCount += 1
        A = SystemMatrix(A)
        n = A.shape[0]
        fields = tileFieldBlocks(self._fieldStructure, n, _IDENTIFICATION)
        b = np.asarray(b, dtype=float).reshape(n)
        residualNorm = float(np.linalg.norm(b))
        if not np.isfinite(residualNorm):
            return self._notSolved(n, "the right-hand side is not finite")

        change = self._changeFor(A, fields, residualNorm)
        if self._fields is None:
            self._log("info", "composed linear solver:\n" + str(self.root) + self._scriptLine())
        try:
            self.root.update(A, change)
        except FloatingPointError as error:
            return self._notSolved(n, str(error))
        x = self.root(b)

        self._fields = fields
        self._modelChanged = False
        self.lastNnz = A.nnz
        self.lastResidualNorm = residualNorm
        self._lastSolveSummary = self.root.solveSummary()
        if self._lastSolveSummary is not None:
            self._log(
                "info",
                "solve #{:}: {:} outer iterations, ‖r‖/‖b‖ = {:.1e}{:} · {:.1f}s".format(
                    self._solveCount,
                    self._lastSolveSummary.iters,
                    self._lastSolveSummary.residual,
                    "" if change.rebuildReason is None else ", rebuilt ({:})".format(change.rebuildReason),
                    time.time() - start,
                ),
            )
        return x

    def _scriptLine(self) -> str:
        if self.script is None:
            return ""
        with open(self.script, "rb") as file:
            digest = hashlib.sha256(file.read()).hexdigest()[:12]
        return "\nfrom {:} (sha256 {:})".format(self.script, digest)


def loadSolverScript(path: str, name: str = "linearSolver", options=None) -> ComposedLinearSolver:
    """The composed solver a solver script builds.

    The script runs with the global ``options`` set to the parsed ``linsolverConfigFile`` of the solver (empty if it
    has none), so one script can serve several configurations.

    Parameters
    ----------
    path
        The script.
    name
        The global of the script that holds the composed solver, a
        :class:`~edelweissfe.linsolve.composition.base.Preconditioner` (usually a Krylov solver).
    options
        The parsed ``linsolverConfigFile``.
    """
    namespace = runpy.run_path(path, init_globals={"options": options or {}})
    if name not in namespace:
        raise ValueError("solver script {:} defines no '{:}'".format(path, name))
    root = namespace[name]
    if not isinstance(root, Preconditioner):
        raise TypeError(
            "solver script {:}: '{:}' is a {:}, not a composed solver (a Preconditioner such as LGMRES(...))".format(
                path, name, type(root).__name__
            )
        )
    return ComposedLinearSolver(root, script=path)
