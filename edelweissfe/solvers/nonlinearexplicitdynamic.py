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
#  Alexander Dummer alexander.dummer@uibk.ac.at
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
"""The nonlinear explicit dynamic solver.

An increment solves no equation system. Second-order fields are advanced by central differences
(a leapfrog, with the velocity staggered half an increment behind the displacement), first-order
fields -- a nonlocal damage field, for instance -- by forward Euler, and both divide by a **lumped**
mass, so the cost of an increment is one element pass plus vector arithmetic. There is no linear
solver, no Newton loop, and no cutback: the time increment is set by stability, not by convergence,
so a material that fails to integrate at the stable step is reported as an error rather than
answered with a smaller step.

**Inertia and damping.** An element reports two diagonals: the coefficient of each field's second
time derivative (its inertia) and of its first (its damping). The integrator divides by the one and
relaxes with the other, and *which* scheme a field gets is read off them -- an inertia means central
differences, a damping alone means forward Euler. The deck declares no field ordering, because an
inertia is exactly what a central-difference update divides by. At zero damping the update is
bit-identical to the undamped one.

The diagnostics need one thing more, and it is about units: which entries of the inertia diagonal
may be summed. That belongs to the FIELD rather than to a deck, and is recorded in
:mod:`edelweissfe.config.phenomena` as a mass, a rotational inertia, or neither. Linear momentum
sums the first kind, the energy balance the first two; conservation across a topology change is
checked per field regardless.

The case that forces it: a first-order field's forward-Euler limit falls off with the SQUARE of the
element size, so on a refined mesh it, not the mechanical field, bounds the increment. Giving it an
inertia makes it second order -- a damped wave equation, limit linear in h -- with its old
coefficient keeping its value and becoming the damping. Marmot supplies one through the ``nonlocal
micro inertia`` element property; that inertia is a time squared, so ``0.5 m_k rate^2`` is a volume
rather than an energy and the field is registered as carrying a non-mechanical inertia, which keeps
it out of both balances.

``first-order-fields``, ``second-order-fields``, ``first-order-scheme`` and ``second-order-scheme``
are removed rather than deprecated: a deck carrying any of them now fails with ``Invalid option``,
and deleting those lines is the whole of the migration.

The stable increment is computed once per mesh from the element wave speeds and scaled by
``courant-number``. It is recomputed whenever the mesh changes and deliberately *not* recomputed
when it has not: as a material softens the true limit only grows, so reusing it is the conservative
direction, and re-deriving it costs a full element pass.

**Three cadences, all in increments.** ``output-frequency`` is how often progress is logged and
field outputs are finalized (a divisor, so it must be at least 1). ``topology-check-frequency`` is
how often model modifiers may change the mesh mid-run, and must be a multiple of
``output-frequency`` because a marker reads the last finalized output; ``0`` runs the topology
update only once, before the increment loop. ``contact-update-frequency`` is how often constraints
whose connectivity is the outcome of a search re-run it; ``0`` disables that. The reasoning behind
the latter two, and what they cost in accuracy, is documented under
:doc:`/documentation/adaptivitytheory` and :doc:`/documentation/contacttheory` respectively.

**Constraints.** Multi-point constraints are enforced by folding a slave's mass and force onto its
masters (row-sum, mass-conserving) and slaving it kinematically, which leaves the critical time step
untouched. Other constraints -- contact above all -- are evaluated through
:meth:`~edelweissfe.constraints.base.constraintbase.ConstraintBase.applyConstraintExplicit`, which
asks for forces and no tangent, since a tangent assembled here would enter nothing. A constraint
that can act *only* through its tangent contributes nothing to an explicit increment and is
refused at validation rather than silently ignored.

**Energy balance.** Every reporting increment prints the external work accumulated at prescribed
degrees of freedom, the kinetic energy, the internal (strain) energy, and what remains unaccounted.
Its purpose is one check that nothing else performs: kinetic energy cannot exceed external work,
because the terms omitted from the balance are all non-negative, so a violation means energy is
appearing from nowhere and the time step is above the true stability limit -- which the critical
time step does not see in full, as it never sees contact penalty stiffness, and sees the nonlocal
field only where that field carries a non-mechanical inertia. Such a field's own
``0.5 m_k rate^2`` is reported on a separate line and kept out of the balance: with a coefficient in
seconds squared it has the units of a volume, not of an energy.

Two honest limits, both of which the solver states in the log rather than leaving to be
discovered. The external work accumulates only at *prescribed* degrees of freedom, so a model
driven by body forces, node forces, a distributed load or an initial velocity has none, and the
check cannot fire; and the internal energy is whatever the materials publish as a strain energy,
which not every material does. When either is identically zero -- or the external work is
negative, a structure returning work at its boundary -- the balance is not usable as a
quasi-static criterion, and the honest substitute is to integrate a reaction force against its
prescribed displacement, both of which are available as ``saveHistory`` field outputs.

**Diagnostics across a topology change.** Because the lumped mass is the operator an increment
divides by, a refinement's effect on it is reported: total mass, per-component linear momentum and
kinetic energy before and after, plus the relative mass drift accumulated over every such change,
so that many individually-tolerable changes cannot silently add up to a meaningful one. The
smallest coefficient the increment divides by is reported alongside -- an inertia at a
second-order degree of freedom, a damping at a first-order one -- since that is what bounds the
step.

**Restart.** A resumed run must not repeat the half-step that starts a leapfrog: the velocity in
the checkpoint already carries the half-increment offset, so applying the startup again would apply
one half-impulse too few. The solver therefore takes the previous time increment from the restart
state rather than synthesising it, and the velocity field is checkpointed in its own right -- an
implicit solver reconstructs everything it needs from the displacement, a central-difference scheme
does not.
"""

from copy import deepcopy
from dataclasses import dataclass
from time import perf_counter

import numpy as np

import edelweissfe.utils.performancetiming as performancetiming
from edelweissfe.config.phenomena import carriesKineticEnergy, carriesLinearMomentum
from edelweissfe.constraints.base.constraintbase import ConstraintBase
from edelweissfe.models.femodel import FEModel
from edelweissfe.numerics.dofmanager import DofManager, DofVector
from edelweissfe.numerics.mpctransformation import MultiPointConstraintTransformation
from edelweissfe.outputmanagers.base.outputmanagerbase import OutputManagerBase
from edelweissfe.solvers.base.conservationchecks import (
    CONSERVATION_TOLERANCE,
    ConservationCheck,
    formatMomentumAndKineticEnergy,
    linearMomentum,
)
from edelweissfe.solvers.base.nonlinearsolverbase import NonlinearSolverBase
from edelweissfe.timesteppers.timestep import TimeStep
from edelweissfe.utils.exceptions import CutbackRequest, StepFailed
from edelweissfe.utils.fieldoutput import FieldOutputController
from edelweissfe.utils.schema import schemaField

#: Fractional margin by which the kinetic energy may exceed the external work before it is
#: reported as energy creation. KE <= W_ext holds in the continuum; mass lumping and the re-lumping
#: at a topology change (see reportTopologyChangeConservation) violate it slightly.
_ENERGY_CREATION_TOLERANCE = 1e-2


@dataclass(frozen=True)
class NEDSchema:
    """The options of the ``*solver`` datalines and of an ``>>options`` block routed to this
    solver, owned by this module and never mutated from outside it.

    Mirrors :attr:`NED.SolverSpecificOptions` one-for-one; the plain ``self.options`` dict remains the actual
    source of truth consulted at runtime (see :class:`~edelweissfe.solvers.nonlinearimplicitstatic.NISTSchema`
    for why). The hyphenated option names are not valid Python identifiers, hence the
    ``optionName`` indirection.

    Every option here is a property of the ANALYSIS -- how far below the stability limit to run,
    how often to report, how often to let the mesh or a contact search change. Nothing here
    *describes* the model: which fields exist, which time derivative each carries, and what its
    coefficient means are all read from the model and from
    :mod:`edelweissfe.config.phenomena`, never asked of the deck. See
    :meth:`NED._classifyFieldsByScheme`.

    The two ``expect-*-fields`` options are the one apparent exception, and are not one: they
    describe nothing and control nothing. They state what the deck's author believes the derivation
    will produce, and :meth:`NED._checkDerivedSchemeAgainstTheDeck` stops the run if it does not.
    Omitted -- as in every deck that predates them -- they assert nothing at all.
    """

    courantNumber: float | None = schemaField(
        description="The fraction of the critical time step actually used.",
        dtype=float,
        default=0.8,
        optionName="courant-number",
    )
    outputFrequency: int | None = schemaField(
        description=(
            "The increment interval at which progress is logged and field outputs are finalized. "
            "Must be at least 1: it is a divisor, so 0 does not disable reporting."
        ),
        dtype=int,
        default=1000,
        optionName="output-frequency",
    )
    topologyCheckFrequency: int | None = schemaField(
        description=(
            "The increment interval at which model modifiers are offered a chance to change the mesh. "
            "0 runs the topology update only once, before the increment loop. Must be a multiple of "
            "output-frequency, because a marker reads the last finalized field output."
        ),
        dtype=int,
        default=0,
        optionName="topology-check-frequency",
    )
    contactUpdateFrequency: int | None = schemaField(
        description=(
            "The increment interval at which constraints whose connectivity is the outcome of a "
            "search (contact) re-run that search. 0 disables periodic updates mid-run."
        ),
        dtype=int,
        default=100,
        optionName="contact-update-frequency",
    )
    expectSecondOrderFields: list | None = schemaField(
        description=(
            "Fields the deck expects to be integrated SECOND order in time (central difference), "
            "as a comma-separated list. An ASSERTION, not a control: the scheme is always derived "
            "from the assembled inertia and damping, and this is compared against that derivation "
            "afterwards, never consulted to decide anything. A named field that turns out first "
            "order aborts the step. Optional; fields not named are not checked."
        ),
        dtype=str,
        default=None,
        optionName="expect-second-order-fields",
    )
    expectFirstOrderFields: list | None = schemaField(
        description=(
            "Fields the deck expects to be integrated FIRST order in time (forward Euler), as a "
            "comma-separated list. The counterpart of expect-second-order-fields and checked the "
            "same way. Optional; fields not named are not checked."
        ),
        dtype=str,
        default=None,
        optionName="expect-first-order-fields",
    )
    reportPerformance: bool = schemaField(
        description=(
            "Print the cumulative performance table (totals since the start of the step) on every "
            "progress report. Off by default. An explicit run is millions of increments long, so "
            "the table printed at the end of the step is otherwise the only one anyone sees."
        ),
        dtype=bool,
        default=False,
        optionName="report-performance",
    )
    lumpedQuantityConservationTolerance: float | None = schemaField(
        description=(
            "Relative-change tolerance for the per-topology-change conservation check on a "
            "row-sum-lumped quantity (mass, a first-order field's viscosity, a second-order "
            "field's non-mechanical inertia). Children of a refined element tile it and carry the "
            "same value, so this is conserved geometrically; the default is already generous "
            "relative to floating-point precision and exists to catch a genuinely wrong refinement "
            "or lumping, not to absorb ordinary quadrature noise. Raise this only when a specific, "
            "understood run trips it by a small margin (see "
            "edelweissfe.solvers.base.conservationchecks)."
        ),
        dtype=float,
        default=CONSERVATION_TOLERANCE,
        optionName="lumped-quantity-conservation-tolerance",
    )


@dataclass
class ExplicitSystem:
    """Everything an explicit increment operates on that is sized by the current equation system.

    None of these are independent state: each is indexed by the :class:`DofManager` in force when it
    was created, so the moment that system changes -- an h-adaptivity event creating nodes and
    elements, a contact constraint re-assigning its slave nodes to different master facets -- every
    one of them has to be rebuilt together, and any that is not becomes either a length mismatch or,
    worse, a silently mis-indexed vector. Bundling them makes "rebuild the system" one assignment at
    the call site instead of nine, which is what keeps the build before the increment loop and a
    rebuild inside it the same code path rather than two that drift apart.

    Parameters
    ----------
    Minv
        The inverse lumped mass. Zero on multi-point-constraint slave DOFs, which integrate no
        equation of motion of their own. The lumped mass it was built from is not carried here:
        the diagnostics need the *unfolded* mass, which is held as ``_rawLumpedMass``, and a
        second mass on this dataclass that no caller reads would only invite reading the wrong
        one.
    U
        The solution vector.
    dU
        The solution increment of the increment being computed.
    V
        The velocity vector, staggered half an increment behind ``U``.
    P
        The net nodal force -- external minus internal -- which drives the velocity update.
    criticalTimeStep
        The stable time increment for the current mesh, already scaled by the courant number.
    """

    Minv: DofVector
    U: DofVector
    dU: DofVector
    V: DofVector
    P: DofVector
    criticalTimeStep: float


@dataclass
class _ReusableExplicitOperators:
    """The lumped operators of an explicit system, and the model they were assembled from.

    A rebuild asked for by a constraint's connectivity re-assigns which nodes that constraint
    couples, and touches nothing the lumped inertia, its inverse, the mass-proportional damping
    rate or the multi-point-constraint transformation are a function of -- so those are exactly
    what they were, and :meth:`NED.buildEquationSystem` keeps them here instead of assembling
    them again.

    The recorded model description is what :meth:`NED._operatorsReusable` checks before it does,
    and anything that does not match makes it assemble from scratch instead.
    """

    elementKeys: frozenset
    multiPointConstraintKeys: frozenset
    nNodes: int
    nScalarVariables: int
    lumpedMass: np.ndarray
    rawLumpedMass: np.ndarray
    inverseLumpedMass: np.ndarray
    dampingRate: np.ndarray
    mpcTransformation: MultiPointConstraintTransformation | None


class NED(NonlinearSolverBase):
    """This is the Nonlinear Explicit Dynamic -- solver.

    Parameters
    ----------
    jobInfo
        A dictionary containing the job information.
    journal
        The journal instance for logging.
    """

    identification = "NEDSolver"

    supportsMPC = True

    #: Model modifiers act before the increment loop and every ``topology-check-frequency`` increments.
    supportsModelModifiers = True

    #: Option schema for this solver, per OptionSchemaProvider.
    schema = NEDSchema

    SolverSpecificOptions = {
        "courant-number": 0.8,
        "output-frequency": 1000,
        "contact-update-frequency": 100,
        "topology-check-frequency": 0,
        "report-performance": False,
        "lumped-quantity-conservation-tolerance": CONSERVATION_TOLERANCE,
        # Lists, so _updateOptions appends the comma-separated items. Empty means "assert nothing".
        "expect-second-order-fields": [],
        "expect-first-order-fields": [],
    }

    #: The last completed increment (read by the central-difference velocity update) and the external
    #: work, an accumulator that cannot be reconstructed from the solution.
    checkpointedState = {
        "prevTimeStep": TimeStep,
        "_externalWork": float,
        # The energy-balance warnings are given once per step, not again after a resume.
        "_warnedAboutMissingInternalEnergy": bool,
        "_warnedAboutMissingExternalWork": bool,
        "_conservationCheck": ConservationCheck,
    }

    def __init__(self, jobInfo, journal, **kwargs):
        self.journal = journal

        # Ensure mutable defaults (field lists) are isolated per solver instance.
        self.options = deepcopy(self.SolverSpecificOptions)
        self._updateOptions(kwargs, journal, strict=True)
        #: Fields integrated first order (forward Euler) and second order (central difference) in
        #: time, and their DOFs; derived in :meth:`_classifyFieldsByScheme`.
        self.firstOrderFields = []
        self.secondOrderFields = []
        self.ids_1st = None
        self.ids_2nd = None
        #: Second-order DOFs whose 0.5*m*v**2 is a mechanical energy (mass-carrying and rotational);
        #: see phenomena.carriesKineticEnergy.
        self.ids_mechanicalEnergy = None
        #: Second-order fields whose inertia is a mass; the momentum diagnostic walks these.
        self.linearMomentumFields = []
        #: Second-order fields whose inertia is neither a mass nor a rotational inertia; reported
        #: separately and kept out of both balances; see phenomena.inertiaKind.
        self.nonMechanicalSecondOrderFields = []
        #: Mass-proportional damping rate C/M per DOF; zero where elements report no damping, which
        #: reduces the update to the undamped central difference.
        self._dampingRate = None
        #: Half the damping rate on second-order DOFs, zero elsewhere; see solveIncrement.
        self._halfDampingRate = None
        #: 1.0 on second-order DOFs, 0.0 elsewhere; see solveIncrement.
        self._secondOrderMask = None
        self._warnedAboutMissingInternalEnergy = False
        self._warnedAboutMissingExternalWork = False
        #: Conservation check of every field's lumped total across topology changes; reset per step.
        self._conservationCheck = ConservationCheck(journal, self.identification)
        #: Work done by the prescribed DOFs, accumulated every increment; compared against the
        #: kinetic energy to detect energy creation.
        self._externalWork = 0.0
        #: The last completed increment; the velocity update reads 0.5 * (dT + dT_prev). None makes
        #: the first increment of a cold step the leapfrog's half step; a resumed step uses the
        #: checkpointed one, since the checkpointed velocity already carries the half-step offset.
        self.prevTimeStep = None
        #: Per-constraint force buffer and scatter plan, by constraint name; cleared on every
        #: DofManager rebuild.
        self._constraintForcePlans = {}
        #: The elements the increment loop evaluates (those with kernels, i.e. without contact
        #: facets); filled by :meth:`buildEquationSystem`. None, not {}, so an unbuilt system fails.
        self._kernelElements = None
        #: Lumped operators kept across a connectivity-only rebuild; see
        #: :class:`_ReusableExplicitOperators`.
        self._reusableOperators = None

    def beginStep(
        self,
        step,
        model: FEModel,
        fieldOutputController: FieldOutputController,
        outputmanagers: dict[str, OutputManagerBase],
    ):
        """Start the step; see
        :meth:`~edelweissfe.solvers.base.nonlinearsolverbase.NonlinearSolverBase.beginStep`. The equation
        system is built by the first :meth:`prepareIncrement`.

        Parameters
        ----------
        step
            The step to be solved.
        model
            The model tree.
        fieldOutputController
            The field output controller.
        outputmanagers
            The output managers.
        """

        self.validateModelCapabilities(model)

        # The timing table's untimed residue is measured against this wall clock, so timers and
        # clock must start together, before any timed work of this step.
        performancetiming.reset()
        self._stepWallClockTic = perf_counter()

        self._externalWork = 0.0
        self.prevTimeStep = None
        self._conservationCheck.reset()
        self._warnedAboutMissingInternalEnergy = False
        self._warnedAboutMissingExternalWork = False

        # Constraints whose DOF footprint is the outcome of a search, i.e. contact.
        self._dynamicConnectivityConstraints = [
            constraint
            for constraint in model.constraints.values()
            if type(constraint).updateConnectivity is not ConstraintBase.updateConnectivity
        ]

        # Modifiers that can still act once the analysis is running.
        self._liveTopologyModifiers = [
            modifier
            for modifier in model.modelModifiers.values()
            if modifier.initiatesTopologyChanges and not modifier.actsOnlyAtSimulationStart
        ]

        # Step actions before the equation system, matching NIST: nothing they do depends on it.
        self.applyStepActionsAtStepStart(model, step)

        self._system = None

    def prepareIncrement(self, step, model: FEModel, isRetry: bool):
        """The topology update, when due, and the equation system; see
        :meth:`~edelweissfe.solvers.base.nonlinearsolverbase.NonlinearSolverBase.prepareIncrement`.

        The topology update is due at the start of the step, and after every
        ``topology-check-frequency``-th increment. At the start of the step it runs before anything
        sized by the equation system exists, so the mesh is final before the lumped mass, the
        multi-point-constraint condensation and the critical time step are derived from it. It runs
        here, before the next increment is proposed, because a refinement may lower the stable time
        increment that increment has to use.

        Parameters
        ----------
        step
            The step being solved.
        model
            The model tree.
        isRetry
            Never True: this solver does not retry increments.
        """

        if step.timeStepper.isAtStepStart():
            # The step-start topology update: the modifiers acting at the start of the analysis (e.g.
            # hAdaptivity's initialOnly markers) and the first contact search.
            self.updateTopologyAndConnectivity(model, step)

        if self._system is None:
            self._buildSystem(self.buildEquationSystem(model, step))
            # The time stepper carries (and checkpoints) the stable increment; a resumed step keeps its own.
            if step.timeStepper.enforcedTimeIncrement is None:
                step.timeStepper.enforceTimeIncrement(self._system.criticalTimeStep)

        # --- h-adaptivity, mid-run ----------------------------------------------------------------
        # After every topology-check-frequency-th increment, i.e. after its output: the marker reads the
        # last finalized field output, and U and the half-step-staggered V pair up only between increments.
        topologyCheckFrequency = self.options["topology-check-frequency"]
        if not (
            self._liveTopologyModifiers
            and topologyCheckFrequency
            and self.prevTimeStep is not None
            and self.prevTimeStep.number % topologyCheckFrequency == 0
        ):
            return

        theSystem, V = self._system, self._V
        lumpedTotalsBefore = self._perFieldLumpedTotals()
        momentumBefore = self.secondOrderMomentum(self._rawLumpedMass, V, model)
        kineticBefore = 0.5 * float(
            np.sum(self._rawLumpedMass[self.ids_mechanicalEnergy] * V[self.ids_mechanicalEnergy] ** 2)
        )

        topologyUpdate = self.updateTopologyAndConnectivity(model, step)
        meshChanged = topologyUpdate.topologyChanged or topologyUpdate.meshDependentsRefreshed
        connectivityChanged = topologyUpdate.constraintConnectivityChanged

        if meshChanged or connectivityChanged:
            # A connectivity-only change carries solution, velocity and force over to the new system.
            self._buildSystem(self.buildEquationSystem(model, step, previous=None if meshChanged else theSystem))
            theSystem = self._system

        U, V, P = self._U, self._V, self._P

        if meshChanged:
            # The net force is not re-evaluated on the new mesh: that would run the constitutive law
            # off-cycle and write into the material state. Zeroing it is an O(dT) error for one increment.
            P[:] = 0.0
            self.publishNodeFields(model, U, V, P)

            self.reportTopologyChangeConservation(lumpedTotalsBefore, momentumBefore, kineticBefore, V, model)

            # Lower only: refinement tightens the limit; a raised limit is not taken up mid-step.
            if theSystem.criticalTimeStep < step.timeStepper.enforcedTimeIncrement:
                self.journal.message(
                    "Refinement lowered the stable time increment from {:e} to "
                    "{:e}".format(step.timeStepper.enforcedTimeIncrement, theSystem.criticalTimeStep),
                    self.identification,
                    1,
                )
                step.timeStepper.enforceTimeIncrement(theSystem.criticalTimeStep)

    def _buildSystem(self, theSystem):
        """Adopt a (re)built equation system, and the vectors the increments work on.

        Parameters
        ----------
        theSystem
            The equation system.
        """

        self._system = theSystem
        self._Minv = theSystem.Minv
        self._U, self._dU, self._V, self._P = theSystem.U, theSystem.dU, theSystem.V, theSystem.P

    def isOutputIncrement(self, timeStep: TimeStep) -> bool:
        """Only every ``output-frequency``-th increment: an explicit run has millions.

        Parameters
        ----------
        timeStep
            The accepted increment.

        Returns
        -------
        bool
            True to write output.
        """

        return timeStep.number % self.options["output-frequency"] == 0

    def attemptIncrement(self, step, model: FEModel, timeStep: TimeStep):
        """The periodic contact search, when due, and the central-difference update; see
        :meth:`~edelweissfe.solvers.base.nonlinearsolverbase.NonlinearSolverBase.attemptIncrement`.

        Parameters
        ----------
        step
            The step being solved.
        model
            The model tree.
        timeStep
            The increment.

        Raises
        ------
        StepFailed
            If a material requests a cutback: an explicit increment is not retried smaller.
        """

        contactUpdateFrequency = self.options["contact-update-frequency"]
        topologyCheckFrequency = self.options["topology-check-frequency"]
        theSystem, Minv = self._system, self._Minv
        U, dU, V, P = self._U, self._dU, self._V, self._P

        if timeStep.number % self.options["output-frequency"] == 0:
            self.journal.printSeperationLine()
            self.journal.message(
                "increment {:}: {:8e}, {:8e}; time {:10e} to {:10e}".format(
                    timeStep.number,
                    timeStep.stepProgressIncrement,
                    timeStep.stepProgress,
                    timeStep.totalTime - timeStep.timeIncrement,
                    timeStep.totalTime,
                ),
                self.identification,
                level=1,
            )

            if self.options["report-performance"]:
                # The cumulative table, as printed at the end of the step.
                self.journal.printPrettyTable(
                    performancetiming.makePrettyTable(wallTime=perf_counter() - self._stepWallClockTic),
                    self.identification,
                )

        # A topology check due at the end of this increment re-runs every connectivity search anyway
        # (and a refinement would invalidate this one), so the contact search is skipped then.
        topologyCheckDueThisIncrement = bool(
            self._liveTopologyModifiers
            and topologyCheckFrequency
            and timeStep.number > 0
            and timeStep.number % topologyCheckFrequency == 0
        )

        if (
            self._dynamicConnectivityConstraints
            and contactUpdateFrequency
            and timeStep.number > 0
            and timeStep.number % contactUpdateFrequency == 0
            and not topologyCheckDueThisIncrement
        ):
            connectivityChanged = self.updateConstraintConnectivity(model)

            if connectivityChanged:
                self.journal.message("Constraint connectivity changed", self.identification, 2)
                self._buildSystem(self.buildEquationSystem(model, step, previous=theSystem))
                theSystem, Minv = self._system, self._Minv
                U, dU, V, P = self._U, self._dU, self._V, self._P

        dU[:] = 0.0
        try:
            U, V, P = self.solveIncrement(
                U,
                dU,
                V,
                P,
                Minv,
                step.actions,
                model,
                timeStep,
                self.prevTimeStep,
            )

        except CutbackRequest as e:
            # The explicit time step is dictated by stability, not convergence, so a cutback is refused:
            # a reduced increment would also stay enforced for the rest of the step, as nothing raises it.
            raise StepFailed(
                "A material requested a cutback in increment {:}: {:}. The explicit time "
                "step is set by stability, not by convergence, so it cannot be reduced in "
                "response -- either the material cannot integrate at the stable step, or "
                "the state reaching it is already wrong. Both need the material or the "
                "model looked at, not a smaller step.".format(timeStep.number, e)
            ) from e

        self._U, self._V, self._P = U, V, P

    def acceptIncrement(self, step, model: FEModel, timeStep: TimeStep):
        """Commit the increment to the model; see
        :meth:`~edelweissfe.solvers.base.nonlinearsolverbase.NonlinearSolverBase.acceptIncrement`.

        Parameters
        ----------
        step
            The step being solved.
        model
            The model tree.
        timeStep
            The increment.
        """

        U, V, P = self._U, self._V, self._P

        # The zero increment is not recorded, so the first real increment is the leapfrog's half step.
        if timeStep.timeIncrement > 0.0:
            self.prevTimeStep = timeStep

        self.publishNodeFields(model, U, V, P)

        self.updateRigidBodies(model, timeStep)

        # Timed: a serial Python loop over all elements and constraints, run every increment here.
        with performancetiming.timeit("accept state"):
            model.advanceToTime(timeStep.totalTime)

    def endStep(self, step, model: FEModel):
        """Report the step's performance timing.

        Parameters
        ----------
        step
            The step that was solved.
        model
            The model tree.
        """

        prettyTable = performancetiming.makePrettyTable(wallTime=perf_counter() - self._stepWallClockTic)
        self.journal.printPrettyTable(prettyTable, self.identification)
        performancetiming.reset()

    @performancetiming.timeit("increment")
    def solveIncrement(
        self,
        U_n: DofVector,
        dU: DofVector,
        V: DofVector,
        P: DofVector,
        Minv: DofVector,
        stepActions: list,
        model: FEModel,
        timeStep: TimeStep,
        prevTimeStep: TimeStep,
    ) -> tuple[DofVector, DofVector, DofVector]:
        """Standard explicit update scheme to solve for an increment.

        Parameters
        ----------
        Un
            The old solution vector.
        V
            The old velocity vector.
        P
            The old reaction vector.
        M
            The lumped mass matrix to be used.
        elements
            The dictionary containing all elements.
        stepActions
            The list of active step actions.
        model
            The model tree.
        timeStep
            The time step.
        prevTimeStep
            The previous time step.

        Returns
        -------
        tuple[DofVector,DofVector,DofVector,DofVector]
            A tuple containing
                - the new solution vector
                - the solution increment
                - the new velocity vector
                - the new reaction vector
        """

        # only the entities that have kernels to evaluate; see buildEquationSystem()
        elements = self._kernelElements
        dirichlets = stepActions["dirichlet"].values()
        nodeforces = stepActions["nodeforces"].values()
        distributedLoads = stepActions["distributedload"].values()
        bodyForces = stepActions["bodyforce"].values()

        # Find which global DOFs the Dirichlet BCs constrain, once up front.
        self.locateConstrainedDofs(dirichlets)

        if timeStep.timeIncrement == 0.0:
            return U_n, V, P

        if prevTimeStep is None:

            prevTimeStep = TimeStep(
                timeStep.number,
                timeStep.stepProgressIncrement,
                timeStep.stepProgress,
                0.0,
                timeStep.stepTime,
                timeStep.totalTime - timeStep.timeIncrement,
            )

        with performancetiming.timeit("kinematic update"):
            # Enforce the Dirichlet boundary conditions on the constrained DOFs:
            # there is no free equilibrium there, so their force P is set to zero,
            # and their velocity is prescribed as (prescribed increment) / (time step).
            prescribedVelocities = []
            for dirichlet in dirichlets:
                prescribedIncrement = dirichlet.getPrescribedIncrement(timeStep).flatten()

                # External work = reaction (-P, the carried-over net force) times the prescribed
                # increment. Must be accumulated before P is zeroed below. Only for fields where this
                # product is an energy (not, e.g., a nonlocal damage field).
                if carriesKineticEnergy(dirichlet.field):
                    self._externalWork -= float(np.dot(P[dirichlet.constrainedDofIndices], prescribedIncrement))

                prescribedVelocity = prescribedIncrement / timeStep.timeIncrement

                P[dirichlet.constrainedDofIndices] = 0.0
                V[dirichlet.constrainedDofIndices] = prescribedVelocity
                prescribedVelocities.append((dirichlet.constrainedDofIndices, prescribedVelocity))

            # Second-order DOFs: central difference with mass-proportional damping,
            # V = ((1 - h) V + Minv P dt) / (1 + h), h = alpha dt / 2, alpha = C/M; at alpha = 0 this is
            # exactly the undamped update. Written on whole vectors: h and the mask are zero off the
            # second-order DOFs. First-order DOFs: V = Minv P (forward Euler).
            dtAverage = 0.5 * (timeStep.timeIncrement + prevTimeStep.timeIncrement)
            halfRateStep = self._halfDampingRate * dtAverage
            forceOverMass = Minv.asPlainArray() * P.asPlainArray()

            velocity = V.asPlainArray()
            velocity *= 1.0 - halfRateStep
            velocity += forceOverMass * (self._secondOrderMask * dtAverage)
            velocity /= 1.0 + halfRateStep
            if self.ids_1st is not None:
                velocity[self.ids_1st] = forceOverMass[self.ids_1st]

            # Restore the prescribed velocities, which both updates above overwrote.
            for constrainedDofIndices, prescribedVelocity in prescribedVelocities:
                V[constrainedDofIndices] = prescribedVelocity

            # MPC slave DOFs follow their masters (Minv is zero there); displacements follow via dU = V * dt
            if self.mpcTransformation is not None:
                self.mpcTransformation.applySlaveKinematics(V)

            # update displacement increment vector
            np.multiply(V, timeStep.timeIncrement, out=dU)
            np.add(U_n, dU, out=U_n)

        with performancetiming.timeit("step actions"):
            self.applyStepActionsAtIncrementStart(model, timeStep, stepActions)

            for geostatic in stepActions["geostatic"].values():
                geostatic.applyAtIterationStart()

        P[:] = 0.0
        P, psi = self.computeElements(elements, U_n, dU, P, timeStep)
        P[:] = -P[:]
        P, _ = self.assembleLoads(nodeforces, distributedLoads, bodyForces, U_n, P, None, timeStep)
        P = self.assembleConstraintForces(model.constraints, U_n, dU, P, timeStep)

        # Fold slave forces onto their masters here, so the next increment's Dirichlet handling sees
        # the folded vector.
        if self.mpcTransformation is not None:
            with performancetiming.timeit("mpc force fold"):
                P[:] = self.mpcTransformation.foldExplicitForce(P)

        if timeStep.number % self.options["output-frequency"] == 0:
            Wint = psi

            # Only over the DOFs where 0.5 * m * v**2 is an energy (not, e.g., first-order fields,
            # whose "mass" is a viscosity).
            Wkin = 0.5 * float(
                np.sum(self._rawLumpedMass[self.ids_mechanicalEnergy] * V[self.ids_mechanicalEnergy] ** 2)
            )

            Wext = self._externalWork

            # Unpublished strain energy, dissipation and damage; negative means energy is being created.
            unaccounted = Wext - Wkin - Wint

            def _share(value):
                return "{:7.2f} %".format(value / Wext * 100) if Wext > 0.0 else "      -- "

            energyRows = [
                ["energy", "value", "share of W_ext"],
                ["external work W_ext", "{:+.6e}".format(Wext), _share(Wext)],
                ["kinetic", "{:+.6e}".format(Wkin), _share(Wkin)],
                ["internal (strain)", "{:+.6e}".format(Wint), _share(Wint)],
                ["unaccounted", "{:+.6e}".format(unaccounted), _share(unaccounted)],
            ]

            # One row per non-mechanical second-order field, kept out of the total: for a hyperbolic
            # non-local field the inertia is a time squared, so 0.5 * m * rate**2 is a volume.
            for fieldName in self.nonMechanicalSecondOrderFields:
                indices = self.theDofManager.idcsOfFieldsInDofVector[fieldName]
                energyRows.append(
                    [
                        fieldName,
                        "{:+.6e}".format(0.5 * float(np.sum(self._rawLumpedMass[indices] * V[indices] ** 2))),
                        "not an energy",
                    ]
                )

            self.journal.printTable(
                energyRows,
                self.identification,
                2,
            )

            # KE <= W_ext holds in the continuum; a violation means the time step is above the true
            # stability limit. NaN is checked first, since every comparison against NaN is False.
            if not (np.isfinite(Wext) and np.isfinite(Wkin)):
                raise StepFailed(
                    "THE SOLUTION HAS DIVERGED in increment {:}: the energy balance is no longer a "
                    "finite number (external work {:e}, kinetic {:e}), which means the state itself "
                    "is not. The time step is above the true stability limit -- and dt_crit does not "
                    "see all of it: it never sees contact penalty stiffness, and it sees the nonlocal "
                    "field only where that field carries a non-mechanical inertia, a first-order one "
                    "having a forward-Euler limit that nothing checks. Resume from a checkpoint with "
                    "a smaller courant-number.".format(timeStep.number, Wext, Wkin)
                )

            if Wext > 0.0 and Wkin > Wext * (1.0 + _ENERGY_CREATION_TOLERANCE):
                self.journal.message(
                    "ENERGY IS BEING CREATED: the kinetic energy {:e} exceeds the external work "
                    "{:e} by {:.1f} %. That is impossible -- the unreported terms (strain energy, "
                    "dissipation, damage) can only add to the balance. The time step is above the "
                    "true stability limit, which dt_crit does not see in full: it ignores the "
                    "nonlocal field entirely and never sees contact penalty stiffness. Reduce "
                    "courant-number, or resume from a checkpoint with a smaller one.".format(
                        Wkin, Wext, (Wkin / Wext - 1.0) * 100
                    ),
                    self.identification,
                    0,
                )

            # Warn once when the energy-creation check cannot fire: external work accumulates only at
            # prescribed DOFs, so it is zero for models not driven by displacements, and negative for
            # one returning work at its boundary.
            if Wext <= 0.0 and not self._warnedAboutMissingExternalWork:
                self._warnedAboutMissingExternalWork = True
                self.journal.message(
                    "The external work is {:}: only work done at prescribed degrees of freedom is "
                    "accumulated, so this model is either not displacement-driven or is currently "
                    "returning work at its prescribed degrees of freedom. The energy balance above "
                    "is NOT usable, and in particular the energy-creation check -- the one that "
                    "catches the stability contributions dt_crit does not see -- cannot fire. "
                    "Watch the kinetic energy history directly instead.".format(
                        "identically zero" if Wext == 0.0 else "negative ({:e})".format(Wext)
                    ),
                    self.identification,
                    1,
                )

            if Wint == 0.0 and not self._warnedAboutMissingInternalEnergy:
                self._warnedAboutMissingInternalEnergy = True
                self.journal.message(
                    "The internal energy is identically zero: no material in this model populates a "
                    "strain energy, so the internal/kinetic split above is NOT usable as a "
                    "quasi-static criterion. Integrate a reaction force against its prescribed "
                    "displacement instead -- both are available as saveHistory field outputs.",
                    self.identification,
                    1,
                )

        return U_n, V, P

    @performancetiming.timeit("elements")
    def computeElements(
        self,
        elements: list,
        U_np: DofVector,
        dU: DofVector,
        P: DofVector,
        timeStep: TimeStep,
    ) -> tuple[DofVector]:
        """Loop over all elements, and evalute them.
        Is called by solveIncrement() in each iteration.

        Parameters
        ----------
        elements
            The list of finite elements.
        U_n
            The current solution vector.
        dU
            The  solution increment vector.
        P
            The reaction vector.
        timeStep
            The time step.

        Returns
        -------
        tuple[DofVector,VIJSystemMatrix,DofVector]
            - The modified reaction vector.
            - The modified system matrix.
            - The modified accumulated flux vector.
        """

        time = timeStep.totalTime
        dT = timeStep.timeIncrement
        P[:] = 0.0
        psi = 0.0
        for el in elements.values():
            Pe = np.zeros(el.nDof)
            el.computeKernelsExplicit(Pe, U_np[el], dU[el], time, dT)
            psi += el.computeInternalEnergy()
            P[el] += Pe

        return P, psi

    def validateModelCapabilities(self, model: FEModel):
        """Refuse the model features this solver cannot integrate, on top of the base checks.

        Two beyond :meth:`~edelweissfe.solvers.base.nonlinearsolverbase.NonlinearSolverBase.validateModelCapabilities`:

        A **model modifier that would act after the analysis has started without configuring
        ``topology-check-frequency``.** Modifiers that act only at simulation start are served by the
        initial topology update before the increment loop. Modifiers that act mid-run require
        ``topology-check-frequency`` to be set to a non-zero multiple of ``output-frequency``;
        otherwise they are refused to prevent silently never adapting.

        A **constraint that introduces its own scalar variables** (a Lagrange multiplier, an
        indirect-control unknown). Those DOFs carry no inertia, so their inverse lumped mass is zero
        and the explicit update leaves them untouched forever -- the constraint would appear active
        and enforce nothing. Penalty-type constraints, which act through nodal forces alone, are
        supported: see :meth:`assembleConstraintForces`.

        Parameters
        ----------
        model
            The model tree.
        """

        super().validateModelCapabilities(model)

        # Modifiers acting after the start need topology-check-frequency set; it is not guessed.
        lateModifiers = sorted(
            name
            for name, modifier in model.modelModifiers.items()
            if modifier.initiatesTopologyChanges and not modifier.actsOnlyAtSimulationStart
        )
        topologyCheckFrequency = self.options["topology-check-frequency"]
        outputFrequency = self.options["output-frequency"]

        # The cadences are modulo operands: output-frequency must be >= 1, none may be negative.
        if outputFrequency is None or outputFrequency < 1:
            raise ValueError(
                "output-frequency ({:}) must be at least 1: it is the divisor of the reporting, "
                "field-output and topology-check cadences, so 0 does not disable them, it raises "
                "ZeroDivisionError on the first increment.".format(outputFrequency)
            )

        if topologyCheckFrequency is not None and topologyCheckFrequency < 0:
            raise ValueError(
                "topology-check-frequency ({:}) cannot be negative. Use 0 to run the topology "
                "update only once, before the increment loop.".format(topologyCheckFrequency)
            )

        contactUpdateFrequency = self.options["contact-update-frequency"]
        if contactUpdateFrequency is not None and contactUpdateFrequency < 0:
            raise ValueError(
                "contact-update-frequency ({:}) cannot be negative. Use 0 to disable the periodic "
                "connectivity update mid-run.".format(contactUpdateFrequency)
            )

        if lateModifiers and not topologyCheckFrequency:
            raise ValueError(
                "The model modifier(s) {:} act after the analysis has started, but "
                "topology-check-frequency is 0, so {:} would run the topology update only once and "
                "they would never act. Set topology-check-frequency to a multiple of "
                "output-frequency ({:}).".format(", ".join(lateModifiers), self.identification, outputFrequency)
            )

        if topologyCheckFrequency:
            if topologyCheckFrequency % outputFrequency:
                raise ValueError(
                    "topology-check-frequency ({:}) must be a multiple of output-frequency ({:}): a "
                    "marker refines on the last finalized field output, and field outputs are "
                    "finalized on the output-frequency cadence, so a check that does not land on one "
                    "would decide on stale results.".format(topologyCheckFrequency, outputFrequency)
                )
            if not lateModifiers:
                self.journal.message(
                    "topology-check-frequency is set but no model modifier acts after the start of "
                    "the analysis; the topology update will run once and the periodic check will "
                    "find nothing to do.",
                    self.identification,
                    1,
                )

        for constraintName, constraint in model.constraints.items():
            nScalarVariables = constraint.getNumberOfAdditionalNeededScalarVariables()
            if nScalarVariables:
                raise NotImplementedError(
                    f"Constraint '{constraintName}' introduces {nScalarVariables} additional scalar "
                    f"variable(s), which carry no inertia and which {self.identification} therefore "
                    "has no equation of motion for. Only constraints acting through nodal forces "
                    "(penalty formulations) are supported here."
                )

    @performancetiming.timeit("publish node fields")
    def publishNodeFields(self, model: FEModel, U, V, P):
        """Write the solution, velocity and force to the node fields and the scalar variables.

        Published every increment, not only on output increments, for two reasons: an h-adaptivity
        event can fall on any increment and its interpolation reads these entries, and a restart
        checkpoint written from the node fields is the only way an explicit run can resume with its
        kinetic state intact. It is an O(nDof) copy.

        Parameters
        ----------
        model
            The model tree.
        U
            The solution vector.
        V
            The velocity vector.
        P
            The net force vector.
        """

        for field in model.nodeFields.values():
            self.theDofManager.writeDofVectorToNodeField(U, field, "U")
            self.theDofManager.writeDofVectorToNodeField(P, field, "P")
            self.theDofManager.writeDofVectorToNodeField(V, field, "V")

        for variable in model.scalarVariables.values():
            variable.value = U[self.theDofManager.idcsOfScalarVariablesInDofVector[variable]]

    @performancetiming.timeit("constraint connectivity")
    def updateConstraintConnectivity(self, model: FEModel) -> bool:
        """Let the constraints whose connectivity is the outcome of a search re-run that search.

        Only the constraints in :attr:`_dynamicConnectivityConstraints` are ticked, and the caller
        ticks them only every ``contact-update-frequency`` increments. Both matter: a node-to-surface
        search is O(slaves x facets) in Python, which is affordable once per increment of an implicit
        analysis -- where it is amortised over a Newton loop and a linear solve -- and not affordable
        tens of thousands of times. What makes throttling defensible rather than merely cheap is that
        an explicit time step is tiny: between two searches a node moves ``V * dT * frequency``,
        orders of magnitude below a facet dimension. The caller reports the motion actually
        accumulated so that this can be checked against a given model instead of assumed.

        Parameters
        ----------
        model
            The model tree.

        Returns
        -------
        bool
            Whether any constraint's DOF footprint changed, i.e. whether the equation system has to
            be rebuilt.
        """

        return any([constraint.updateConnectivity(model) for constraint in self._dynamicConnectivityConstraints])

    def _operatorsReusable(self, model: FEModel, stepActions: dict) -> bool:
        """Whether the kept lumped operators still describe this model, so that a rebuild may keep
        them and re-locate the constraints alone.

        The operators are a function of the elements -- which carry the inertia and the damping --
        of the multi-point constraints, and of the degree-of-freedom layout the two are indexed by.
        A contact search changes none of those: it re-assigns which nodes a *constraint* couples,
        which moves the constraints' own DOF footprints and nothing else. A topology change does
        move them, and never reaches here: it passes no ``previous`` system and builds afresh.

        The multi-point constraints are safe to keep for a structural reason rather than a
        coincidental one, which is worth being explicit about. They are a separate hierarchy
        (:class:`~edelweissfe.constraints.base.multipointconstraintbase.MultiPointConstraintBase`)
        with no connectivity-update protocol at all -- ``updateConnectivity`` belongs to
        :class:`~edelweissfe.constraints.base.constraintbase.ConstraintBase`, and only those
        constraints are collected into ``_dynamicConnectivityConstraints`` and can ask for this
        rebuild -- and the dependency records they produce are a function of the node DOF indices,
        which a constraint refresh leaves untouched. Should a multi-point constraint ever gain a
        connectivity of its own, that argument fails and the transformation would have to be
        rebuilt here rather than kept.

        Parameters
        ----------
        model
            The model tree.
        stepActions
            The step's actions.

        Returns
        -------
        bool
            Whether the kept operators may be reused.
        """

        cache = self._reusableOperators
        if cache is None:
            return False

        # Inertia and damping come from the materials, so a material property change invalidates them.
        if stepActions["changematerialproperty"]:
            return False

        return (
            model.elements.keys() == cache.elementKeys
            and model.multiPointConstraints.keys() == cache.multiPointConstraintKeys
            and len(model.nodes) == cache.nNodes
            and len(model.scalarVariables) == cache.nScalarVariables
        )

    def _restoreLumpedOperators(self) -> tuple[DofVector, DofVector]:
        """Put the kept lumped operators back, into vectors carrying the refreshed entity mapping.

        The numbers are the ones :meth:`_assembleLumpedOperators` produced when it last ran. What
        that method additionally left on the solver -- which fields are first and which second
        order, and the DOF index sets of each -- is not restored because it was never invalidated:
        a constraint refresh leaves the fields and their indices exactly as they were, so those
        attributes still describe this system.

        Returns
        -------
        tuple[DofVector, DofVector]
            The vector the increment divides by, and its inverse.
        """

        cache = self._reusableOperators

        # The transformation is indexed by the DOF layout, which the refresh did not move.
        assert cache.mpcTransformation is None or cache.mpcTransformation.nDof == self.theDofManager.nDof
        self.mpcTransformation = cache.mpcTransformation

        M = self.theDofManager.constructDofVector()
        M[:] = cache.lumpedMass
        Minv = self.theDofManager.constructDofVector()
        Minv[:] = cache.inverseLumpedMass
        self._rawLumpedMass = self.theDofManager.constructDofVector()
        self._rawLumpedMass[:] = cache.rawLumpedMass
        self._dampingRate = self.theDofManager.constructDofVector()
        self._dampingRate[:] = cache.dampingRate
        self._lumpedMass = M

        return M, Minv

    def _assembleLumpedOperators(self, model: FEModel, step, verbosity: int) -> tuple[DofVector, DofVector]:
        """Assemble everything the explicit update divides by, and everything derived from it.

        The lumped inertia and the lumped damping are summed from the elements, checked, and used
        to decide which time derivative each field carries; the multi-point-constraint
        transformation is built alongside them because it folds the slave degrees of freedom's
        mass onto their masters. Beyond the two vectors returned, this leaves the derived
        classification, the raw (unfolded) mass, the damping rate and the field index sets on the
        solver.

        Parameters
        ----------
        model
            The model tree.
        step
            The step, for the actions the multi-point-constraint transformation consults.
        verbosity
            The journal level to report at.

        Returns
        -------
        tuple[DofVector, DofVector]
            The vector the increment divides by, and its inverse.
        """

        self.mpcTransformation = self.buildMPCTransformation(model, step.actions)
        self.checkMPCDirichletConflicts(self.mpcTransformation, step.actions)

        # The force plans are indexed by the old DofManager; stale ones would scatter to wrong DOFs.
        self._constraintForcePlans = {}

        # Elements without kernels (contact facets) are left out of the per-increment loop. The loop's
        # gather plan is keyed on this dict's identity.
        self._kernelElements = {number: element for number, element in model.elements.items() if element.hasKernels}

        # initialize mass and damping matrices
        M = self.theDofManager.constructDofVector()  # initialize lumped mass matrix
        Minv = self.theDofManager.constructDofVector()  # initialize inverse lumped mass matrix

        M[:] = 0.0
        for el in model.elements.values():
            Me = np.zeros(el.nDof)
            el.computeLumpedInertia(Me)
            M[el] += Me

        # Each field's first-derivative coefficient: zero mechanically, the non-local viscosity always.
        damping = self.theDofManager.constructDofVector()
        damping[:] = 0.0
        for el in model.elements.values():
            Ce = np.zeros(el.nDof)
            el.computeLumpedDamping(Ce)
            damping[el] += Ce

        # Checked separately: at second-order DOFs the damping enters only as the rate C/M, which the
        # inertia check below never sees, and a negative rate amplifies instead of damping.
        if not np.all(np.isfinite(damping)) or np.any(damping < 0.0):
            raise ValueError(
                "The assembled lumped damping is not a valid dissipation: {:} of {:} entries are "
                "negative and {:} are not finite (smallest: {:e}). A damping coefficient enters the "
                "second-order update as the rate C/M, where a negative value amplifies instead of "
                "damping, and a first-order field divides by it directly.".format(
                    int(np.count_nonzero(np.asarray(damping) < 0.0)),
                    damping.shape[0],
                    int(np.count_nonzero(~np.isfinite(np.asarray(damping)))),
                    np.nanmin(damping),
                )
            )

        # A field with an inertia is second order, one with only a damping first order.
        self.firstOrderFields, self.secondOrderFields = self._classifyFieldsByScheme(M, damping)

        self.ids_1st = self._dofIndicesOfFields(self.firstOrderFields)
        self.ids_2nd = self._dofIndicesOfFields(self.secondOrderFields)

        self.journal.message(
            "Time integration, derived from the assembled operators: central difference for {:}; "
            "forward Euler for {:}".format(
                ", ".join(self.secondOrderFields) or "(no field)",
                ", ".join(self.firstOrderFields) or "(no field)",
            ),
            self.identification,
            verbosity,
        )

        self._checkDerivedSchemeAgainstTheDeck()

        # Which second-order fields enter the linear momentum and the energy balance is a property of
        # the field, taken from the registry; see phenomena.inertiaKind.
        self.linearMomentumFields = [f for f in self.secondOrderFields if carriesLinearMomentum(f)]
        self.nonMechanicalSecondOrderFields = [f for f in self.secondOrderFields if not carriesKineticEnergy(f)]
        self.ids_mechanicalEnergy = self._dofIndicesOfFields(
            [f for f in self.secondOrderFields if carriesKineticEnergy(f)]
        )

        # First-order fields integrate C * rate = P, so from here on M is "the divisor": the inertia at
        # second-order DOFs, the damping at first-order ones.
        M[self.ids_1st] = damping[self.ids_1st]

        # Kept before folding, so the kinetic energy diagnostic accounts for the true velocities of
        # all nodes (including tied slaves) rather than master-placed folded mass.
        self._rawLumpedMass = M.copy()

        # compute inverses
        if np.any(M == 0.0):
            raise ValueError(
                "Zero found in the vector the increment divides by, at {:} of {:} degrees of freedom. "
                "Every FIELD is covered by the classification above, which refuses a field carrying "
                "neither coefficient, so what is left here are degrees of freedom belonging to no field "
                "-- scalar variables, which this solver has no equation of motion for.".format(
                    int(np.count_nonzero(np.asarray(M) == 0.0)), M.shape[0]
                )
            )

        # A negative lumped mass (e.g. row-summed quadratic elements) would integrate backwards in time.
        if np.any(M < 0.0):
            raise ValueError(
                "Negative coefficient found in {:} of {:} entries of the vector the increment "
                "divides by (smallest: {:e}). It is a lumped inertia at a second-order degree of "
                "freedom and a damping at a first-order one; either way a negative value makes the "
                "explicit update integrate backwards in time there.".format(
                    int(np.count_nonzero(M < 0.0)), M.shape[0], M.min()
                )
            )

        # Slave DOFs of multi-point constraints carry no own inertia: their mass is folded onto
        # their masters (row-sum lumping of T^T M T, mass-conserving), their Minv stays zero, and
        # their kinematics are assigned directly from the masters each increment.
        if self.mpcTransformation is not None:
            self.mpcTransformation.foldLumpedMass(M)
            # Folded like the inertia, so the ratio below is the damping rate of the folded system.
            self.mpcTransformation.foldLumpedMass(damping)

        Minv[M != 0.0] = 1.0 / M[M != 0.0]

        # Damping rate alpha = C / M at second-order DOFs with nonzero (folded) inertia; zero elsewhere.
        self._dampingRate = self.theDofManager.constructDofVector()
        self._dampingRate[:] = 0.0
        integrating = self.ids_2nd[M[self.ids_2nd] > 0.0]
        self._dampingRate[integrating] = damping[integrating] / M[integrating]

        # kept (instead of 1/Minv) for the kinetic energy: slave DOFs have Minv = 0
        self._lumpedMass = M

        return M, Minv

    @performancetiming.timeit("build equation system")
    def buildEquationSystem(self, model: FEModel, step, previous: ExplicitSystem = None) -> ExplicitSystem:
        """Build the equation system and everything sized by it.

        Called once before the increment loop, and again from inside it whenever a constraint reports
        that its DOF footprint changed -- one method for both, so the path every model takes and the
        path only a contact model takes cannot drift apart. On that second path the DofManager is
        refreshed rather than rebuilt and the lumped operators are kept; see
        :meth:`_operatorsReusable`.

        Parameters
        ----------
        model
            The model tree.
        step
            The step being solved; its actions are needed to check the multi-point constraints
            against the prescribed Dirichlet conditions.
        previous
            The system being replaced, when this is a rebuild rather than the initial build. Its
            solution, velocity and force are carried over verbatim rather than re-read from the node
            fields, which do not hold the velocity at all. A rebuild triggered by a constraint's
            connectivity leaves the mesh -- hence the DOF layout -- untouched, and that is checked
            rather than assumed: copying between two different layouts would mis-index every vector
            silently.

        Returns
        -------
        ExplicitSystem
            The freshly built system.
        """

        isRebuild = previous is not None
        verbosity = 2 if isRebuild else 0

        # A connectivity-only rebuild refreshes the DofManager and keeps the lumped operators; see
        # _operatorsReusable(). Anything else builds from scratch.
        reuseOperators = isRebuild and self._operatorsReusable(model, step.actions)

        if reuseOperators:
            self.journal.message(
                "Constraint connectivity changed: re-locating the constraints' degrees of freedom, "
                "keeping the lumped operators",
                self.identification,
                verbosity,
            )
            self.theDofManager.refreshConstraintIndices(model.constraints.values())
        else:
            self.journal.message("Creating monolithic equation system", self.identification, verbosity)
            # No system matrix, so neither the VIJ pattern nor the index-to-host-object map is needed.
            self.theDofManager = DofManager(
                model.nodeFields.values(),
                model.scalarVariables.values(),
                model.elements.values(),
                model.constraints.values(),
                model.nodeSets.values(),
                initializeVIJPattern=False,
                determiningIndexToHostObjectMapping=False,
            )
        self.journal.message(
            "total size of eq. system: {:}".format(self.theDofManager.nDof),
            self.identification,
            verbosity,
        )

        if not isRebuild:
            self.journal.printSeperationLine()

        # The force plans are indexed by the old constraint indices; stale ones would scatter to wrong DOFs.
        self._constraintForcePlans = {}

        if reuseOperators:
            M, Minv = self._restoreLumpedOperators()
        else:
            M, Minv = self._assembleLumpedOperators(model, step, verbosity)
            self._reusableOperators = _ReusableExplicitOperators(
                elementKeys=frozenset(model.elements.keys()),
                multiPointConstraintKeys=frozenset(model.multiPointConstraints.keys()),
                nNodes=len(model.nodes),
                nScalarVariables=len(model.scalarVariables),
                lumpedMass=np.array(M),
                rawLumpedMass=np.array(self._rawLumpedMass),
                inverseLumpedMass=np.array(Minv),
                dampingRate=np.array(self._dampingRate),
                mpcTransformation=self.mpcTransformation,
            )

        # Per-DOF factors of the velocity update, so that it runs on whole vectors.
        self._secondOrderMask = np.zeros(self._dampingRate.shape[0])
        if self.ids_2nd is not None:
            self._secondOrderMask[self.ids_2nd] = 1.0
        self._halfDampingRate = 0.5 * np.asarray(self._dampingRate) * self._secondOrderMask

        U = self.theDofManager.constructDofVector()  # initialize displacement vector
        dU = self.theDofManager.constructDofVector()  # initialize displacement vector
        V = self.theDofManager.constructDofVector()  # initilize velocity vector
        P = self.theDofManager.constructDofVector()  # initialize reaction vector

        if not isRebuild:
            for fieldName, field in model.nodeFields.items():
                U = self.theDofManager.writeNodeFieldToDofVector(U, field, "U")
                P = self.theDofManager.writeNodeFieldToDofVector(P, field, "P")

                # After an h-adaptivity event, the velocity interpolated onto the new nodes is read back
                # here (see hadaptivity.WARM_STARTED_NODE_FIELD_ENTRIES); it exists from the second build on.
                if "V" in field:
                    V = self.theDofManager.writeNodeFieldToDofVector(V, field, "V")

            for variable in model.scalarVariables.values():
                U[self.theDofManager.idcsOfScalarVariablesInDofVector[variable]] = variable.value
        else:
            if previous.U.shape != U.shape:
                raise RuntimeError(
                    "The equation system was rebuilt with {:} degrees of freedom instead of {:}. Only "
                    "a constraint's connectivity is expected to trigger a rebuild here, and that "
                    "cannot add or remove degrees of freedom -- so the solution and the velocity "
                    "cannot be carried across safely.".format(U.shape[0], previous.U.shape[0])
                )

            # Carried straight over, not re-read from the node fields: the velocity is not a node
            # field, so re-reading would silently resume from rest.
            U[:] = previous.U
            V[:] = previous.V
            P[:] = previous.P

        if isRebuild:
            # The mesh is unchanged, so the stable time increment is kept (recomputing costs an element pass).
            criticalTimeStep = previous.criticalTimeStep
        else:
            criticalTimeStep = self.options.get("courant-number") * self.getCriticalTimeStepForExplicitDynamics(
                model, U
            )
            self.journal.message(
                "Critical time step for explicit dynamics: {:e}".format(criticalTimeStep), self.identification, 1
            )

        return ExplicitSystem(
            Minv=Minv,
            U=U,
            dU=dU,
            V=V,
            P=P,
            criticalTimeStep=criticalTimeStep,
        )

    @performancetiming.timeit("assemble constraints")
    def assembleConstraintForces(
        self,
        constraints: dict,
        U_np: DofVector,
        dU: DofVector,
        P: DofVector,
        timeStep: TimeStep,
    ) -> DofVector:
        """Evaluate every constraint and add its nodal forces to the net force vector.

        The explicit counterpart of
        :meth:`~edelweissfe.solvers.nonlinearimplicitstatic.NIST.assembleConstraints`, and it shares
        that method's sign convention: a constraint writes what the implicit solver calls ``PExt``,
        which is why this is called after :meth:`assembleLoads`, on a ``P`` that already holds
        ``-P_internal``.

        No tangent is requested. An explicit increment solves no linear system, so a constraint's
        stiffness enters nothing, and
        :meth:`~edelweissfe.constraints.base.constraintbase.ConstraintBase.applyConstraintExplicit`
        is the entry point that evaluates the constraint forces directly without assembling a tangent.
        A constraint that could act *only* through its tangent would contribute nothing here; that is
        precisely the class :meth:`validateModelCapabilities` refuses.

        Parameters
        ----------
        constraints
            The constraints of the model, by name.
        U_np
            The current solution vector.
        dU
            The current solution increment.
        P
            The net force vector to be augmented.
        timeStep
            The current time step.

        Returns
        -------
        DofVector
            The augmented net force vector.
        """

        # Per-constraint buffers and index plans, built once per equation system.
        PPlain = P.asPlainArray()

        for name, constraint in constraints.items():
            plan = self._constraintForcePlans.get(name)
            if plan is None:
                indices = P.entitiesInDofVector[constraint]
                # np.add.at only if a DOF appears more than once (e.g. a slave node in its own master
                # facet), since fancy-indexed += would drop repeated contributions.
                plan = (np.zeros(constraint.nDof), indices, len(np.unique(indices)) != len(indices))
                self._constraintForcePlans[name] = plan

            Pc, indices, namesDofMoreThanOnce = plan

            # Reused, so it must be cleared: applyConstraintExplicit augments what it is handed.
            Pc[:] = 0.0

            constraint.applyConstraintExplicit(U_np[constraint], dU[constraint], Pc, timeStep)

            if namesDofMoreThanOnce:
                np.add.at(PPlain, indices, Pc)
            else:
                PPlain[indices] += Pc

        return P

    def secondOrderMomentum(self, mass: DofVector, V: DofVector, model: FEModel) -> np.ndarray:
        """The linear momentum of the fields whose inertia is a mass, per spatial component; see
        :func:`~edelweissfe.solvers.base.conservationchecks.linearMomentum`.

        Parameters
        ----------
        mass
            The lumped mass to weight with. Pass the *unfolded* mass: a multi-point-constraint slave
            carries real velocity, and folding its mass onto its masters would drop its momentum.
        V
            The velocity vector.
        model
            The model tree, for the fields' spatial dimension.

        Returns
        -------
        np.ndarray
            The momentum, one entry per spatial component.
        """

        return linearMomentum(np.asarray(mass) * np.asarray(V), self.theDofManager, self.linearMomentumFields, model)

    def _perFieldLumpedTotals(self) -> dict[str, float]:
        """The assembled lumped total of every first- and second-order field, each on its own.

        Never summed across fields. A conservation bug that halves one field's total can be
        diluted below detection by another field's total if the two differ enough in magnitude
        -- which happens routinely here, since a mechanical density and a non-local viscosity or
        non-mechanical inertia are not just different units, they are typically many orders of
        magnitude apart in value. That is true even between two fields of the SAME kind (two
        mechanical fields of very different density would have the same problem), so the fix is
        per field, not per "mechanical vs. not". The conservation check tests each field
        once with its own total, so a violation anywhere is visible regardless of what
        else is assembled alongside it.

        Returns
        -------
        dict[str, float]
            Field name to its total lumped mass, viscosity or non-mechanical inertia -- whichever
            that field's own coefficient is.
        """

        totals = {}
        for fieldName in self.firstOrderFields + self.secondOrderFields:
            indices = self.theDofManager.idcsOfFieldsInDofVector[fieldName]
            totals[fieldName] = float(np.sum(self._rawLumpedMass[indices]))
        return totals

    def _dofIndicesOfFields(self, fieldNames: list[str]) -> np.ndarray:
        """The degrees of freedom of the named fields, concatenated in the order given.

        Parameters
        ----------
        fieldNames
            Names of fields present in the current DofManager.

        Returns
        -------
        np.ndarray
            Their indices in the dof vector; empty (and of integer dtype, which matters -- numpy
            reads an index of None as np.newaxis) when no field is named.
        """

        indices = np.empty(0, dtype=int)
        for fieldName in fieldNames:
            indices = np.r_[indices, self.theDofManager.idcsOfFieldsInDofVector[fieldName]]
        return indices

    def _checkDerivedSchemeAgainstTheDeck(self):
        """Compare the derived time-integration scheme against what the deck says it expects.

        An ASSERTION, and deliberately nothing more. ``expect-second-order-fields`` and
        ``expect-first-order-fields`` are never consulted to decide how a field is integrated --
        :meth:`_classifyFieldsByScheme` has already done that from the assembled inertia and
        damping, which remain the single source of truth. This only refuses to continue when the
        two disagree.

        The distinction matters, because the deck used to *declare* the scheme and that is what was
        removed: a declaration can contradict the assembled operators, and the old one had a hole
        through which a field named in neither list was silently never integrated at all. An
        assertion cannot desynchronize from behaviour, because no behaviour reads it, and omitting
        it means "do not check" rather than "integrate nothing".

        What it buys is the one failure :meth:`_classifyFieldsByScheme` structurally cannot see. It
        refuses a field split across its own degrees of freedom, and a field carrying neither
        coefficient; it cannot refuse a field that is uniformly FIRST order when second order was
        intended. Such a run completes and reports success with different physics. That is a
        one-token slip: a non-local field is second order only if its material provides a
        micro-inertia, which for GCDP is optional material property 21 and legitimately defaults to
        zero, so forgetting it silently returns the field to the parabolic scheme -- whose own
        stability limit nothing checks.

        Raises
        ------
        ValueError
            If a field named in either list was derived as the other scheme, or names a field the
            model does not carry.
        """
        derived = {name: "second order" for name in self.secondOrderFields}
        derived.update({name: "first order" for name in self.firstOrderFields})

        for expected, declaredFields in (
            ("second order", self.options["expect-second-order-fields"]),
            ("first order", self.options["expect-first-order-fields"]),
        ):
            for fieldName in declaredFields:
                if fieldName not in derived:
                    raise ValueError(
                        "The deck expects field '{:}' to be integrated {:}, but this model carries "
                        "no such field. It carries: {:}.".format(
                            fieldName, expected, ", ".join(sorted(derived)) or "(no field)"
                        )
                    )

                if derived[fieldName] != expected:
                    raise ValueError(
                        "The deck expects field '{:}' to be integrated {:} in time, but it was "
                        "assembled {:}, so that is how it would have been integrated. The scheme "
                        "follows from the operators the elements report, never from this "
                        "declaration -- which exists precisely so that this disagreement stops the "
                        "run instead of quietly changing the answer.{:}".format(
                            fieldName,
                            expected,
                            derived[fieldName],
                            (
                                " A non-local field is second order only if its material provides a "
                                "micro-inertia; for GCDP that is optional material property 21, and "
                                "leaving it off legitimately means zero."
                                if expected == "second order"
                                else ""
                            ),
                        )
                    )

    def _classifyFieldsByScheme(self, inertia: DofVector, damping: DofVector) -> tuple[list[str], list[str]]:
        """Which fields are integrated second order in time and which first, read off the
        assembled operators rather than declared.

        The deck used to say this and had no freedom in what to say: an inertia is what a
        central-difference update divides by, so a field carrying one is second order and a field
        carrying none is not. Deriving it also closes a hole the declaration had -- a field named
        in neither list was silently never integrated at all.

        Per FIELD, not per degree of freedom: a multi-material mesh can give a micro-inertia to one
        material and not to another, so covering only some of the elements carrying a field is easy
        to do by accident, and integrating part of a field as a wave equation and the rest as a
        diffusion one is not a scheme anybody chose. Such a field is refused rather than split.

        Parameters
        ----------
        inertia
            The assembled lumped inertia, before the first-order entries are overwritten with
            their damping and before any multi-point-constraint fold -- a tied slave's own
            inertia has to still be there, or its field would look unintegrable.
        damping
            The assembled lumped damping, likewise.

        Returns
        -------
        tuple[list[str], list[str]]
            The first-order and the second-order field names, in the model's field order.

        Raises
        ------
        ValueError
            If any field carries a coefficient on some of its degrees of freedom and not on
            others, or carries neither coefficient anywhere.
        """

        firstOrderFields = []
        secondOrderFields = []

        for fieldName, indices in self.theDofManager.idcsOfFieldsInDofVector.items():
            carriesInertia = np.asarray(inertia[indices]) != 0.0
            carriesDamping = np.asarray(damping[indices]) != 0.0

            if carriesInertia.all():
                secondOrderFields.append(fieldName)
            elif carriesInertia.any():
                raise ValueError(
                    "Field {:} was assembled an inertia on {:} of its {:} degrees of freedom and "
                    "none on the rest, so it is neither second order in time nor first order. The "
                    "usual cause is a mesh whose materials disagree: for a non-local field the "
                    "micro-inertia is a material property, so a second material used by some of "
                    "the elements carrying the field and giving no micro-inertia splits it.".format(
                        fieldName, int(np.count_nonzero(carriesInertia)), carriesInertia.size
                    )
                )
            elif carriesDamping.all():
                firstOrderFields.append(fieldName)
            elif carriesDamping.any():
                raise ValueError(
                    "Field {:} carries no inertia, so it is integrated by its damping alone -- but "
                    "a damping was assembled on only {:} of its {:} degrees of freedom, and the "
                    "rest have nothing to divide by.".format(
                        fieldName, int(np.count_nonzero(carriesDamping)), carriesDamping.size
                    )
                )
            else:
                raise ValueError(
                    "Field {:} carries neither an inertia nor a damping, so there is no time "
                    "derivative for this solver to integrate it with. Its elements report both "
                    "through computeLumpedInertia and computeLumpedDamping; an element with zero "
                    "density or zero volume reports the first as zero.".format(fieldName)
                )

        return firstOrderFields, secondOrderFields

    def reportTopologyChangeConservation(
        self,
        lumpedTotalsBefore: dict[str, float],
        momentumBefore: np.ndarray,
        kineticBefore: float,
        V: DofVector,
        model: FEModel,
    ):
        """Report what a topology change did to the quantities that ought to survive it.

        Refinement interpolates the velocity onto new nodes and re-lumps every per-element
        scalar property, and the three invariants behave differently under that:

        * **Every field's own lumped total is conserved exactly**, by the same geometric
          identity regardless of what that field's coefficient physically is: the children of a
          refined element tile it and carry the same value. Checked per FIELD, not per family,
          via :class:`~edelweissfe.solvers.base.conservationchecks.ConservationCheck` -- summing across fields first, even ones
          that agree on units, would let a violation in a numerically small field hide inside a
          numerically large one. Violating any single field's total raises.
        * **Linear momentum is conserved exactly for a spatially uniform velocity field**, because
          the shape functions are a partition of unity and the child masses sum to the parent's. For
          a general field the discrepancy is second order in the velocity gradient across the parent:
          discretisation error, not a defect. Reported, not enforced. Restricted to the fields whose
          inertia is a mass, per :func:`~edelweissfe.config.phenomena.carriesLinearMomentum`: a
          viscosity or a non-mechanical inertia has no associated momentum, and a rotational inertia
          has an angular one that must not be added to a linear one. Across those fields the sum IS
          legitimate, because a real total system momentum is exactly the sum of its parts' momenta.
        * **Kinetic energy is not conserved** by interpolation plus re-lumping, and it is the most
          sensitive of the three, being quadratic in the interpolation error. Reported as a relative
          jump; more than roughly a percent is a reason to look at the transfer rather than to
          believe the physics. Restricted to the fields whose 0.5*m*v**2 is an energy -- the
          mass-carrying ones and the rotational ones, a wider set than momentum admits -- and
          reason, and summed across them for the same reason momentum is.

        Parameters
        ----------
        lumpedTotalsBefore
            Every first- and second-order field's own lumped total before the change, by field
            name; see :meth:`_perFieldLumpedTotals`.
        momentumBefore
            Per-component momentum before the change.
        kineticBefore
            Kinetic energy before the change.
        V
            The velocity vector of the rebuilt system.
        model
            The model tree.

        Raises
        ------
        RuntimeError
            If any field's own lumped total changed by more than its conservation tolerance.
        """

        lumpedTotalsAfter = self._perFieldLumpedTotals()
        momentumAfter = self.secondOrderMomentum(self._rawLumpedMass, V, model)
        kineticAfter = 0.5 * float(
            np.sum(self._rawLumpedMass[self.ids_mechanicalEnergy] * V[self.ids_mechanicalEnergy] ** 2)
        )

        tolerance = self.options["lumped-quantity-conservation-tolerance"]
        relativeChangeByField = {
            fieldName: self._conservationCheck.check(
                "lumped coefficient of field '{:}'".format(fieldName),
                before,
                lumpedTotalsAfter.get(fieldName, 0.0),
                tolerance,
            )
            for fieldName, before in lumpedTotalsBefore.items()
        }
        worstField, worstRelativeChange = (
            max(relativeChangeByField.items(), key=lambda item: item[1]) if relativeChangeByField else ("", 0.0)
        )

        # A tiny positive mass gives an enormous Minv; report the smallest integrating coefficient
        # relative to the median, so a collapsing mass is visible.
        integrating = self._lumpedMass[self._lumpedMass > 0.0]
        smallestMass = float(np.min(integrating)) if integrating.size else 0.0
        medianMass = float(np.median(integrating)) if integrating.size else 0.0
        massRatio = smallestMass / medianMass if medianMass > 0.0 else 0.0

        # Split by scheme: an inertia (central difference) and a damping (forward Euler) bound
        # different stability limits.
        def smallestOf(indices):
            if not indices.size:
                return 0.0, 0.0
            entries = self._lumpedMass[indices]
            entries = entries[entries > 0.0]
            if not entries.size:
                return 0.0, 0.0
            return float(np.min(entries)), float(np.median(entries))

        smallest2nd, median2nd = smallestOf(self.ids_2nd)
        smallest1st, median1st = smallestOf(self.ids_1st)

        self.journal.message(
            "Topology change: worst-conserved field '{:}' to {:.1e} relative; smallest integrating "
            "coefficient {:.3e} ({:.1e} of median) [2nd-order inertia {:.3e} of median {:.3e}; "
            "1st-order damping {:.3e} of median {:.3e}]; {:}".format(
                worstField,
                worstRelativeChange,
                smallestMass,
                massRatio,
                smallest2nd,
                median2nd,
                smallest1st,
                median1st,
                formatMomentumAndKineticEnergy(momentumBefore, momentumAfter, kineticBefore, kineticAfter),
            ),
            self.identification,
            1,
        )

    def getCriticalTimeStepForExplicitDynamics(self, model: FEModel, U: DofVector) -> float:
        """Compute the critical time step for explicit dynamics.

        Parameters
        ----------
        model
            The model tree.

        Returns
        -------
        float
            The critical time step for explicit dynamics.
        """
        minTimeStep = np.inf

        for element in model.elements.values():
            elementTimeStep = np.inf
            elementTimeStep = element.computeCriticalTimeStepForExplicitDynamics(U[element])
            if elementTimeStep < minTimeStep:
                minTimeStep = elementTimeStep

        return minTimeStep
