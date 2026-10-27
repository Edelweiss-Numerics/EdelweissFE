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
"""The building block of a composed linear solver: :class:`Preconditioner`, an approximate inverse :math:`M^{-1}`
of a matrix :math:`A`, and what is handed to it -- the :class:`SetUpContext` of its (sub-)problem and the
:class:`MatrixChange` since its last set-up.

A composed solver is a tree of preconditioners. Each one solves its part of the problem and hands a smaller
problem to its children: a Krylov solver calls its preconditioner once per iteration, a Schur complement solves
the condensed dofs exactly and hands the remaining ones to its child, a diagonal scaling multiplies by
:math:`D` before and after its child. A Krylov solver is itself a preconditioner -- an approximate inverse of
:math:`A` -- so solvers nest like any other component.

A component implements what makes it what it is, in protected hooks:

- :meth:`Preconditioner._setUp` builds it for a matrix (and its children for their sub-problems);
- :meth:`Preconditioner._forget` drops what it built, and so declares what that is (the constructor calls it);
- :meth:`Preconditioner._update` takes a changed matrix and decides what to keep (by default: set up again);
- :meth:`Preconditioner._apply` applies it, :math:`z = M^{-1} r` -- the formula of the textbook;
- :meth:`Preconditioner.children` names its children, if it has any (the tree is walked through it);
- :meth:`Preconditioner.describe` says what it is, for the views (:mod:`~edelweissfe.linsolve.composition.views`).

It is used through the public methods of the base class, :meth:`~Preconditioner.setUp`,
:meth:`~Preconditioner.update` and :meth:`~Preconditioner.__call__` (a template method each): they record the call
and its time in the component's :class:`Statistics` and call the hook. So a component's ``_apply`` holds nothing but
its formula, and a hook that calls another hook of its own component (a Krylov solver's ``_setUp`` is its ``_update``
for a first matrix) counts as one call.

A component builds only through :meth:`~Preconditioner.setUp`, also when an update decides to rebuild (it calls
``self.setUp``). :meth:`~Preconditioner.setUp` first forgets what was built before (:meth:`~Preconditioner.forget`:
the :meth:`~Preconditioner._forget` of the component and of its children), so that the old and the new hierarchies,
factorizations and matrices, gigabytes for a large model, are never held at once. An update that does not rebuild
keeps what was built, and what is reused on purpose is never forgotten: a fine spectral radius carried to the next
build, the selection of a Schur complement's condensed dofs, a symbolic factorization, the statistics. A component
with nothing built builds at its next update, whatever the update says.
"""

import time
from dataclasses import dataclass, field, replace

import numpy as np

from edelweissfe.linsolve.base import FieldBlock
from edelweissfe.numerics.p1topology import restrictP1Map


@dataclass
class Statistics:
    """What a component recorded while it ran: filled at run time, shown by the run-statistics view.

    The times are inclusive: a component's apply time contains its children's.

    Attributes
    ----------
    setUps
        How often it was built from scratch (:meth:`Preconditioner.setUp`).
    updates
        How often it took a changed matrix (:meth:`Preconditioner.update`), whether it rebuilt or not.
    applications
        How often it was applied (:meth:`Preconditioner.__call__`).
    setUpSeconds, updateSeconds, applySeconds
        The wall time spent in each.
    sizes
        What the component wants to report about what it built, e.g. ``{"rows": 841260, "levels": 4}``.
    """

    setUps: int = 0
    updates: int = 0
    applications: int = 0
    setUpSeconds: float = 0.0
    updateSeconds: float = 0.0
    applySeconds: float = 0.0
    sizes: dict = field(default_factory=dict)

    def record(self, kind: str, seconds: float) -> None:
        """Count one call of ``kind`` (``"setUp"``, ``"update"`` or ``"apply"``) that took ``seconds``."""
        if kind == "setUp":
            self.setUps += 1
            self.setUpSeconds += seconds
        elif kind == "update":
            self.updates += 1
            self.updateSeconds += seconds
        else:
            self.applications += 1
            self.applySeconds += seconds


@dataclass(frozen=True)
class Step:
    """One step of a component's application (:meth:`Preconditioner._apply`), for the data-flow view.

    Attributes
    ----------
    operation
        What the step computes, in the symbols of the component's formula, e.g. ``"y_b = A_bb⁻¹ r_b"``.
    child
        The role of the child the step delegates to (a role of :meth:`Preconditioner.children`), or ``None`` for a
        step the component does itself.
    """

    operation: str
    child: "str | None" = None


@dataclass(frozen=True)
class Description:
    """What a component is, independent of how it is shown: the views
    (:mod:`~edelweissfe.linsolve.composition.views`) draw it.

    Attributes
    ----------
    name
        The component's name, e.g. ``"SchurComplement"``.
    formula
        Its approximate inverse in unicode, with ``{role}`` for the formula of each of its
        :meth:`~Preconditioner.children`, e.g. ``"D · {inner} · D"``.
    latex
        The same in LaTeX, with ``{role}`` placeholders.
    parameters
        The settings that make it what it is, e.g. ``{"restart": 50}``.
    steps
        The steps of one application, in order.
    """

    name: str
    formula: str
    latex: str = ""
    parameters: dict = field(default_factory=dict)
    steps: tuple = ()


@dataclass(frozen=True)
class SetUpContext:
    """What a component may need besides its matrix: the fields of its (sub-)problem and where their nodes are.

    Attributes
    ----------
    fields
        The field blocks of the dofs of this (sub-)problem, contiguous and in dof order.
    scaling
        The diagonal scaling between the physical dofs and the variables of this (sub-)problem: the matrix a
        component gets is :math:`D A D` of the physical one, ``None`` for :math:`D = I`. A near null-space,
        physical by nature (the rigid-body modes of a displacement field), is :math:`D^{-1}` times its physical
        modes in these variables.
    coordinatesOf
        ``fieldName -> (nNodes, dimension)`` node coordinates of the field, in its dof order, or ``None`` if
        unknown (an offline replay without a model).
    p1MapOf
        ``fieldName -> (isCorner, edgeEndpoints)``, the corner/midside topology of a quadratic field, or ``None``.
    model
        The model, if there is one (markers read it), else ``None``.
    journal
        The journal, if there is one, else ``None``.
    log
        ``(level, message) -> None``: the composed solver's log, which shows a message if its verbosity reaches
        ``level`` (``"warning"``, ``"info"``, ``"debug"``), or ``None`` (then :meth:`message` prints warnings only).
    """

    fields: tuple
    scaling: "np.ndarray | None" = None
    coordinatesOf: object = None
    p1MapOf: object = None
    model: object = None
    journal: object = None
    log: object = None

    def message(self, level: str, text: str) -> None:
        """Log ``text`` at ``level`` through the composed solver (see :attr:`log`)."""
        if self.log is not None:
            self.log(level, text)
        elif level == "warning":
            print(text, flush=True)

    @property
    def size(self) -> int:
        """The number of dofs of this (sub-)problem."""
        return self.fields[-1].stop if self.fields else 0

    def onlyField(self, owner: str) -> FieldBlock:
        """The one field of this (sub-)problem, for a component that is built on one field's matrix; raises, naming
        ``owner``, if there are several."""
        if len(self.fields) != 1:
            raise ValueError(
                "{:} is built on the matrix of one field, but got the fields {:}: put it under OnField(name, ...) "
                "for each field, combined by Multiplicative([...]) (block Gauss-Seidel) or Additive([...]) (block "
                "Jacobi)".format(owner, ", ".join(field.name for field in self.fields))
            )
        return self.fields[0]

    def coordinates(self, fieldName: str) -> "np.ndarray | None":
        """The node coordinates of a field, or ``None``."""
        return None if self.coordinatesOf is None else self.coordinatesOf(fieldName)

    def p1Map(self, fieldName: str):
        """The P1 corner/midside map of a field, or ``None``."""
        return None if self.p1MapOf is None else self.p1MapOf(fieldName)

    def scalingOf(self, dofs: "slice | np.ndarray") -> np.ndarray:
        """The scaling of some dofs (ones without a scaling)."""
        if self.scaling is None:
            return np.ones(self.size)[dofs]
        return self.scaling[dofs]

    def scaledBy(self, D: np.ndarray) -> "SetUpContext":
        """The context of :math:`D \\hat A D`, for a child that gets this problem scaled once more by ``D``."""
        return replace(self, scaling=D if self.scaling is None else self.scaling * D)

    def restrictedToField(self, block: FieldBlock) -> "SetUpContext":
        """The context of one field block of this problem: its dofs renumbered from zero."""
        return replace(
            self,
            fields=(FieldBlock(block.name, 0, block.stop - block.start, block.dimension),),
            scaling=None if self.scaling is None else self.scaling[block.start : block.stop],
        )

    def restrictedTo(self, fields: tuple, dofs: np.ndarray, nodesOf=None) -> "SetUpContext":
        """The context of a subset ``dofs`` of this problem's dofs, whose fields are ``fields`` (as the Schur
        complement leaves them). ``nodesOf``: ``fieldName -> boolean mask`` of the nodes of the field that remain,
        to restrict the coordinates and P1 maps."""

        def coordinatesOf(fieldName):
            coordinates = self.coordinates(fieldName)
            mask = None if nodesOf is None else nodesOf(fieldName)
            return coordinates if coordinates is None or mask is None else coordinates[mask]

        def p1MapOf(fieldName):
            p1Map = self.p1Map(fieldName)
            mask = None if nodesOf is None else nodesOf(fieldName)
            return p1Map if p1Map is None or mask is None else restrictP1Map(*p1Map, mask)

        return replace(
            self,
            fields=tuple(fields),
            scaling=None if self.scaling is None else self.scaling[dofs],
            coordinatesOf=coordinatesOf,
            p1MapOf=p1MapOf,
        )


@dataclass(frozen=True)
class MatrixChange:
    """What changed since a component's last set-up, and whether what it built must be built again.

    The facts about the matrix (a new pattern, a resized system) are found by the composed solver
    (:class:`~edelweissfe.linsolve.composition.solver.ComposedLinearSolver`); a Krylov solver adds its own
    reasons (an increment, a stale preconditioner, see
    :class:`~edelweissfe.linsolve.composition.krylov.RebuildPolicy`); a Schur complement adds its own when its
    condensed dofs change. Each component then decides in :meth:`Preconditioner.update` what to keep.

    Attributes
    ----------
    context
        The context of the changed matrix.
    rebuildReason
        Why what was built must be built again (``"first"``, ``"model"``, ``"amr"``, ``"pattern"``,
        ``"increment"``, ``"stale"``, ``"always"``, ``"condensed dofs"``), or ``None``: only the values changed,
        and what was built for the previous matrix is reused.
    newIncrement
        Whether the right-hand side jumped, which marks a new increment of the nonlinear solver.
    residualNorm
        :math:`\\lVert b \\rVert` of this solve.
    previousResidualNorm
        :math:`\\lVert b \\rVert` of the previous solve, ``None`` before the first one (or after a change of the
        model).
    """

    context: SetUpContext
    rebuildReason: "str | None" = None
    newIncrement: bool = False
    residualNorm: float = 0.0
    previousResidualNorm: "float | None" = None

    @property
    def requiresRebuild(self) -> bool:
        """Whether what was built must be built again."""
        return self.rebuildReason is not None

    def withRebuild(self, reason: str) -> "MatrixChange":
        """This change, asking for a rebuild for ``reason`` -- unless it asks for one already, whose reason stays."""
        return self if self.requiresRebuild else replace(self, rebuildReason=reason)

    def withContext(self, context: SetUpContext) -> "MatrixChange":
        """This change, for a child with its own context."""
        return replace(self, context=context)


class Preconditioner:
    """An approximate inverse :math:`M^{-1}` of a matrix :math:`A`: built by :meth:`setUp`, applied as
    :math:`z = M^{-1} r`.

    See the module documentation. A subclass implements the hooks :meth:`_setUp`, :meth:`_apply`, :meth:`_forget` if
    it builds anything, and, to reuse what it built, :meth:`_update`, and calls ``super().__init__()`` in its
    constructor (which creates its :attr:`statistics` and declares what it builds, by :meth:`_forget`).
    """

    def __init__(self):
        #: What this component recorded while it ran, see :class:`Statistics`.
        self.statistics = Statistics()
        # nothing is built yet
        self._forget()

    # ----- the public methods: record the call, then call the hook -----

    def setUp(self, A, context: SetUpContext) -> None:
        """Build this component for the matrix ``A`` (a
        :class:`~edelweissfe.linsolve.composition.systemmatrix.SystemMatrix`), and its children for their
        sub-problems: :meth:`forget` what was built before, then :meth:`_setUp`, recorded."""
        self.forget()
        self._recorded("setUp", self._setUp, A, context)

    def forget(self) -> None:
        """Drop what this component and its children built (:meth:`_forget`), so that nothing is held twice while it
        is built again. What they reuse on purpose stays (see the module documentation)."""
        self._forget()
        for _, child in self.children():
            child.forget()

    def update(self, A, change: MatrixChange) -> None:
        """Take the changed matrix ``A``: :meth:`_update`, recorded."""
        self._recorded("update", self._update, A, change)

    def __call__(self, r: np.ndarray) -> np.ndarray:
        """Apply this component, :math:`z = M^{-1} r`: :meth:`_apply`, recorded."""
        return self._recorded("apply", self._apply, r)

    def applyRestricted(self, r_f: np.ndarray) -> np.ndarray:
        """:math:`M^{-1} r_f` on the dofs of :attr:`fieldIndex` alone: :meth:`_applyRestricted`, recorded as an
        application."""
        return self._recorded("apply", self._applyRestricted, r_f)

    def _recorded(self, kind: str, hook, *arguments):
        """Call ``hook(*arguments)`` and count the call and its time in :attr:`statistics` as ``kind``."""
        start = time.perf_counter()
        try:
            return hook(*arguments)
        finally:
            self.statistics.record(kind, time.perf_counter() - start)

    # ----- the hooks a component implements -----

    def _setUp(self, A, context: SetUpContext) -> None:
        """Build this component for the matrix ``A``, and its children (through their public :meth:`setUp`)."""
        raise NotImplementedError

    def _forget(self) -> None:
        """Drop what this component built: set each attribute that :meth:`_setUp` builds to its value before the first
        set-up (``self.hierarchy = None``). The constructor calls it, so this is where those attributes are declared;
        a component that builds nothing of its own (its children are forgotten by :meth:`forget`) keeps this
        default."""

    def _update(self, A, change: MatrixChange) -> None:
        """Take the changed matrix ``A``. By default, build anew; a component that can reuse what it built
        overrides this and rebuilds (through :meth:`setUp`) only if ``change.requiresRebuild`` or nothing is
        built."""
        self.setUp(A, change.context)

    def _apply(self, r: np.ndarray) -> np.ndarray:
        """:math:`z = M^{-1} r`."""
        raise NotImplementedError

    def _applyRestricted(self, r_f: np.ndarray) -> np.ndarray:
        """:math:`M^{-1} r_f` on the dofs of :attr:`fieldIndex` alone (all dofs: the same as :meth:`_apply`)."""
        return self._apply(r_f)

    def children(self) -> tuple:
        """The ``(role, child)`` pairs of this component, in the order the children are used (empty for a leaf). A
        component with children overrides this: the tree is walked through it (:meth:`setModel`,
        :meth:`finishSolve`, :attr:`isFixedLinearOperator`, ..., and the views), and the role is the placeholder of
        the child in the component's formula (:attr:`Description.formula`)."""
        return ()

    # ----- what the tree is asked as a whole -----

    def withRebuildReasons(self, change: MatrixChange) -> MatrixChange:
        """``change``, with the reasons for a rebuild that this component and its children add on their own (a Krylov
        solver's :class:`~edelweissfe.linsolve.composition.krylov.RebuildPolicy`: a new increment, a stale
        preconditioner). A component that keeps something built for the matrix above such a child (the :math:`D` of a
        :class:`~edelweissfe.linsolve.composition.diagonalscaling.DiagonalScaling` around a Krylov solve) asks for
        them first, so that it rebuilds together with the child."""
        for _, child in self.children():
            change = child.withRebuildReasons(change)
        return change

    @property
    def fieldIndex(self) -> "int | None":
        """The field this component acts on, :math:`R` in :math:`z = R^T M^{-1} R r`, as the index of the field in its
        context (known after :meth:`setUp`), or ``None`` for all dofs. A combination
        (:mod:`~edelweissfe.linsolve.composition.combinators`) then forms the residual of that field alone and calls
        :meth:`applyRestricted`."""
        return None

    @property
    def isFixedLinearOperator(self) -> bool:
        """Whether :math:`z = M^{-1} r` is one fixed linear operator between two set-ups: true for a component whose
        children all are; a Krylov solve is not (its result depends on :math:`r` nonlinearly, through the tolerance
        and the subspace it builds). A Krylov method that relies on a fixed preconditioner (LGMRES, CG) refuses one
        that is not."""
        return all(child.isFixedLinearOperator for _, child in self.children())

    @property
    def containsKrylovSolver(self) -> bool:
        """Whether this component is, or has among its children, a Krylov solver. A Krylov solver reuses its
        preconditioner across solves (so the nonlinear solver keeps the sparsity pattern of the matrix stable) and
        records each solve (the "linear solve" column); a direct solve does neither."""
        return any(child.containsKrylovSolver for _, child in self.children())

    def setModel(self, model, dofManager) -> None:
        """Take a new model (the first, or one after a change of the mesh), and hand it to the children. A component
        that derives something from the model (e.g. which dofs a marker selects) overrides this."""
        for _, child in self.children():
            child.setModel(model, dofManager)

    def finishSolve(self) -> None:
        """The solve is done: release what only this solve needed (e.g. the field blocks of its matrix, which the next
        solve forms anew from its own matrix), and tell the children. What the next solve may reuse (a
        preconditioner) is kept. Without this, a solver would hold a copy of the last matrix between solves -- while
        the nonlinear solver assembles the next one."""
        for _, child in self.children():
            child.finishSolve()

    def builtMatrices(self) -> tuple:
        """``(name, scipy matrix)`` pairs of the matrices this component formed and keeps (for the matrix view);
        empty for most."""
        return ()

    def solveRecord(self):
        """What the last Krylov solve in this component did, a
        :class:`~edelweissfe.linsolve.composition.krylov.SolveRecord`: a Krylov solver's own, else that of the first
        child that has one (e.g. the Krylov solve under a diagonal scaling), else ``None``."""
        for _, child in self.children():
            record = child.solveRecord()
            if record is not None:
                return record
        return None

    def setUpNotes(self) -> tuple:
        """Short facts about what this component and its children built in the last set-up or update, one string
        each, for the "linear solve" column of the nonlinear solver's table at verbosity ``"debug"``: at most 24
        characters, the width of that column. Most components have none."""
        return tuple(note for _, child in self.children() for note in child.setUpNotes())

    def condensedDofs(self) -> "np.ndarray | None":
        """The dofs a Schur complement in this component condensed in the last solve, or ``None`` (for a degradation
        dump, which records them for an offline replay without the model)."""
        for _, child in self.children():
            dofs = child.condensedDofs()
            if dofs is not None:
                return dofs
        return None

    def describe(self) -> Description:
        """What this component is, for the views."""
        return Description(name=type(self).__name__, formula=type(self).__name__ + "⁻¹")

    def __str__(self) -> str:
        from edelweissfe.linsolve.composition.views import structure

        return structure(self)

    def _repr_html_(self) -> str:
        from edelweissfe.linsolve.composition.views import html

        return html(self, standalone=False)


def requirePreconditioner(owner: str, role: str, child) -> None:
    """Raise at construction if ``child``, the ``role`` of ``owner``, is not a component (e.g. a class instead of an
    instance, or a bare SciPy solver)."""
    if not isinstance(child, Preconditioner):
        raise TypeError(
            "{:}: its {:} must be a component (a Preconditioner such as AMG() or DirectSolver()), got {!r}".format(
                owner, role, child
            )
        )
