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

"""The common base class every registered ``linsolver`` inherits, so the nonlinear solver can treat
all of them uniformly instead of special-casing per capability.

Every ``linsolve/<name>/__init__.py``'s ``createSolver(opts)`` factory returns an instance of a
:class:`LinearSolver` subclass, callable as ``(A, b) -> x``. :meth:`LinearSolver.setJournal` and
:meth:`LinearSolver.setModel` are part of that base contract with safe defaults, so a caller can call
either on *any* registered solver unconditionally, whether or not that particular solver actually uses
the information -- a solver that needs more than the default overrides it; everything else inherits a
default that does nothing harmful.

:meth:`setModel` hands a solver the live :class:`~edelweissfe.models.femodel.FEModel` and
:class:`~edelweissfe.numerics.dofmanager.DofManager` it is solving for. Everything a solver may want is
derivable from those two objects (the field layout from the ``DofManager``, node coordinates and element
topology from the ``FEModel``), so a solver that wants more than the default field-structure bookkeeping
(e.g. the composed solver reading node coordinates for a rigid-body near null-space, or building a P1
topology map) keeps the references and computes what it needs on its own schedule -- the driver need not
know in advance what a solver might want. :meth:`setFieldStructure` remains as a lower-level way in for
callers that know the field-block layout directly but have no full model to hand over (e.g. an offline
replay of a dumped system).
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class FieldBlock:
    """One physical field's contiguous block in the DOF vector.

    Attributes
    ----------
    name
        The field name, e.g. ``"displacement"`` or ``"nonlocal damage"``.
    start, stop
        The half-open DOF range ``[start, stop)`` of the field (fields are contiguous and field-major).
    dimension
        The nodal dimension of the field: the number of components per node (e.g. ``3`` for a 3D
        displacement, ``1`` for a scalar damage field). Determines the field's near null-space -- a
        vector field's rigid-body translations, a scalar field's constant.
    """

    name: str
    start: int
    stop: int
    dimension: int


@dataclass(frozen=True)
class LinearSolveSummary:
    """One solver call's diagnostics -- exactly what the nonlinear solver renders as the extra
    "linear solve" column in its own convergence table (see
    :meth:`~edelweissfe.solvers.base.nonlinearsolverbase.NonlinearSolverBase.checkConvergence`).

    Attributes
    ----------
    iters
        This call's outer iteration count (e.g. GMRES outer iterations).
    residual
        The achieved residual, in whatever norm the solver checks against its own tolerance.
    residualMet
        Whether ``residual`` met the solver's own requested tolerance for this call -- rendered with
        the same "met tolerance" checkmark the Newton residual columns already use.
    retries
        How many extra internal attempts this call needed to meet its tolerance (``0`` for a clean
        first-pass solve). Rendered as a superscript on the iteration count rather than a plain digit,
        so it can never be misread as part of that number (e.g. iters=65, retries=2 must not look like
        the single number 652).
    detailLines
        Extra lines confined to the "linear solve" column's own width, for detail a routine glance
        does not need but solver tuning does (e.g. why a preconditioner was rebuilt). Empty unless the
        solver's own verbosity setting asks for it -- solvers that never populate this simply cost the
        caller nothing beyond checking for an empty tuple.
    """

    iters: int
    residual: float
    residualMet: bool
    retries: int
    detailLines: tuple = ()


class LinearSolver:
    """Common base for every ``linsolver`` registry entry. Callable as ``(A, b) -> x``.

    Subclasses implement :meth:`__call__`. :meth:`setJournal`, :meth:`setModel` and
    :meth:`setFieldStructure` have safe defaults here so the nonlinear solver can call any of them on
    any solver without asking first which ones care.
    """

    _journal = None
    _fieldStructure: "list[FieldBlock] | None" = None
    _model = None
    _dofManager = None

    #: Display name for this solver's own log lines (e.g. a "linear solve" column's debug detail).
    #: Overridden by a solver that wants its messages attributed to something other than the generic
    #: base name -- e.g. ``ComposedLinearSolver`` sets this to its own identification string.
    identification = "LinearSolver"

    #: Whether this solver reports per-solve diagnostics (an iteration count, a residual, etc.) that
    #: the nonlinear solver can render as an extra "linear solve" column in its own convergence table.
    #: ``False`` for solvers with nothing iterative to report -- a direct factorization has no
    #: per-solve iteration count or residual, so there is nothing there worth a column. A solver
    #: overriding this to ``True`` must also populate :attr:`lastSolveSummary` after every
    #: :meth:`__call__`.
    reportsSolveSummary = False

    #: The most recent call's diagnostics as a :class:`LinearSolveSummary`, or ``None`` before any
    #: call, or for a solver that never overrides :attr:`reportsSolveSummary`. Read by the nonlinear
    #: solver immediately after calling this solver, to build its per-iteration row -- never pushed
    #: proactively, since the caller alone knows when "this iteration's row" is being assembled.
    lastSolveSummary: "LinearSolveSummary | None" = None

    #: Whether this solver benefits from the system matrix keeping the same sparsity pattern from one
    #: Newton iteration to the next -- i.e. whether it does something *across* solves that a changing
    #: pattern defeats (a Krylov solver reuses a preconditioner built for the old pattern; a direct
    #: factorization like PARDISO's starts from scratch on every solve regardless and has nothing to
    #: lose). ``False`` here, inherited by every solver that does not override it. Read by
    #: :attr:`~edelweissfe.solvers.nonlinearimplicitstatic.NISTSchema.pruneCondensedMatrixZeros`'s
    #: "auto" default (``None``, unset in the input file) to decide whether pruning the
    #: multi-point-condensed system's explicit zeros -- which changes its pattern on almost every
    #: iteration -- is worth it *for this solver*: pruned (``True``) unless the solver says it needs
    #: the pattern stable, in which case it is left alone (``False``) so hierarchy reuse can engage.
    #: An explicit ``true``/``false`` in the input file always overrides this.
    needsStableSparsityPattern = False

    def setJournal(self, journal) -> None:
        """Receive the shared :class:`~edelweissfe.journal.journal.Journal` instance.

        Default just stores it on ``self._journal`` for solvers that want to log through it; solvers
        with no logging needs simply never read the attribute.
        """
        self._journal = journal

    def setModel(self, model, dofManager) -> None:
        """Receive the live model and DOF manager this solver is being asked to solve for.

        Called by the nonlinear solver whenever the equation system is (re)built -- i.e. on the first
        solve and again after any AMR/connectivity change. The default implementation derives and
        stores the per-field DOF-block structure (name, DOF range, nodal dimension) any field-split
        solver needs -- equivalent to calling :meth:`setFieldStructure` with those blocks -- and keeps
        ``model``/``dofManager`` themselves on ``self._model``/``self._dofManager`` for a solver that
        wants to go further (e.g. node coordinates for a rigid-body near null-space, or a P1 topology
        map) without the driver having to know about it.

        A solver overriding this should normally call ``super().setModel(model, dofManager)`` first to
        get the field-structure bookkeeping for free, then do whatever else it needs.

        Parameters
        ----------
        model
            The live :class:`~edelweissfe.models.femodel.FEModel`.
        dofManager
            The live :class:`~edelweissfe.numerics.dofmanager.DofManager` describing the current
            equation system's layout.
        """
        self._model = model
        self._dofManager = dofManager
        self._fieldStructure = [
            FieldBlock(fieldName, fieldIndices.start, fieldIndices.stop, model.nodeFields[fieldName].dimension)
            for fieldName, fieldIndices in dofManager.idcsOfFieldsInDofVector.items()
        ]

    def setFieldStructure(self, fields: "list[FieldBlock]") -> None:
        """Receive the ordered field blocks of the DOF vector directly (in DOF order), bypassing
        :meth:`setModel`.

        An escape hatch for a caller that knows the field-block layout but has no full
        ``FEModel``/``DofManager`` to hand over -- e.g. an offline probe script driving a solver
        directly on a captured ``(A, b)`` system. A live run goes through :meth:`setModel` instead,
        whose default implementation computes exactly this and stores it the same way; a solver that
        only ever needs the field-block structure (not the model/dofManager references themselves)
        does not need to care which path supplied it.

        Only field-split solvers (e.g. a composed solver with ``OnField`` corrections) read it; every
        other solver simply ignores it.
        """
        self._fieldStructure = list(fields)

    def __call__(self, A, b):
        raise NotImplementedError


class DirectSolverBackend:
    """A sparse direct solver library (SuperLU, PARDISO, KLU, ...): factorizes a matrix once, then solves for as many
    right-hand sides as needed by back substitution.

    Not a linear solver of its own: the composed leaf :class:`~edelweissfe.linsolve.composition.direct.DirectSolver`
    applies it, :math:`z = A^{-1} r`, and the ``type=pardiso`` (etc.) linear solvers are that one leaf. Backends are
    registered in the ``directsolver`` registry category, and take their options as keyword arguments.

    The contract every backend keeps: :meth:`factorize` factorizes (once per set-up of its leaf), and reports a matrix
    it cannot factorize there; :meth:`solveFactorized` only applies the factors, as often as asked -- inside a Krylov
    solve, once per iteration -- and solves with the factorization of the last :meth:`factorize`, also when the
    assembly has since overwritten that matrix in place. A backend that keeps its symbolic factorization (the
    reordering) while only the values of the matrix change does so inside :meth:`factorize`: one backend instance
    lives as long as its :class:`DirectSolver` leaf, which releases the numeric factorization
    (:meth:`releaseFactorization`) before it factorizes anew.
    """

    def factorize(self, A) -> None:
        """Factorize ``A`` (CSR) for the following :meth:`solveFactorized` calls. A matrix that cannot be factorized
        (singular) raises :class:`numpy.linalg.LinAlgError`."""
        raise NotImplementedError

    def solveFactorized(self, b):
        """Solve ``A x = b`` with the factorization of the last :meth:`factorize`, for one right-hand side (a vector)
        or several (the columns of a matrix), returned in the shape of ``b``."""
        raise NotImplementedError

    def releaseFactorization(self) -> None:
        """Free the numeric factorization of the last :meth:`factorize` (nothing to free before the first), so that
        it is gone before the next :meth:`factorize` computes another: never two at once. A symbolic factorization
        the backend reuses stays."""
        raise NotImplementedError
