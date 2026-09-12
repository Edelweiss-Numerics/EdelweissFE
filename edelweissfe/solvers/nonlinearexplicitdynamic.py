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
from edelweissfe.numerics.dofmanager import DofManager, DofVector, VIJSystemMatrix
from edelweissfe.outputmanagers.base.outputmanagerbase import OutputManagerBase
from edelweissfe.solvers.base.nonlinearsolverbase import NonlinearSolverBase
from edelweissfe.stepactions.base.stepactionbase import StepActionBase
from edelweissfe.timesteppers.timestep import TimeStep
from edelweissfe.utils.exceptions import (
    ConditionalStop,
    CutbackRequest,
    ReachedMaxIncrements,
    ReachedMinIncrementSize,
    StepFailed,
)
from edelweissfe.utils.fieldoutput import FieldOutputController
from edelweissfe.utils.schema import schemaField

#: Tolerance on the relative change, across one topology change, of any row-sum-lumped
#: per-element quantity (a mass, a viscosity, a non-mechanical inertia). Children tile their
#: parent and carry its value, so conservation is a geometric identity -- but the assembly is
#: Gauss quadrature, exact only to a polynomial order, and a distorted hexa20's Jacobian is not
#: polynomial. Measured on the anchor pry-out, one live refinement moves the total mass by
#: 4.63e-08 relative; a real refinement or lumping error would be O(1).
_LUMPED_QUANTITY_CONSERVATION_TOLERANCE = 1e-6

#: Fractional margin by which the kinetic energy may exceed the external work before it is
#: reported as energy creation. KE <= W_ext is exact in the continuum, but the discrete run has
#: two legitimate sources of small violation: lumping the mass matrix, and the interpolate-then-
#: re-lump of a topology change, which does not conserve kinetic energy exactly (see
#: reportTopologyChangeConservation). One per cent is far above both and far below the runaway an
#: unstable time step produces -- v5 of the anchor pry-out reached 1e+38 mm.
_ENERGY_CREATION_TOLERANCE = 1e-2

#: Tolerance on the accumulated drift of any one lumped quantity over a whole step: a single
#: change being within tolerance does not bound a run with hundreds of refinements. At the
#: measured 4.63e-08 per change this permits over two thousand of them.
_CUMULATIVE_LUMPED_QUANTITY_DRIFT_TOLERANCE = 1e-4


@dataclass(frozen=True)
class NEDSchema:
    """The options of the ``*solver`` datalines and of an ``>>options`` block routed to this
    solver, owned by this module and never mutated from outside it.

    Mirrors :attr:`NED.NEDOptions` one-for-one; the plain ``self.options`` dict remains the actual
    source of truth consulted at runtime (see :class:`~edelweissfe.solvers.nonlinearimplicitstatic.NISTSchema`
    for why). The hyphenated option names are not valid Python identifiers, hence the
    ``optionName`` indirection.

    Every option here is a property of the ANALYSIS -- how far below the stability limit to run,
    how often to report, how often to let the mesh or a contact search change. Nothing here
    describes the model: which fields exist, which time derivative each carries, and what its
    coefficient means are all read from the model and from
    :mod:`edelweissfe.config.phenomena`, never asked of the deck. See
    :meth:`NED._classifyFieldsByScheme`.
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
    reportPerformance: bool = schemaField(
        description=(
            "Print a performance table on every progress report, covering the interval since the "
            "previous one. Off by default. An explicit run is millions of increments long, so the "
            "table printed at the end of the step is otherwise the only one anyone sees, and it "
            "averages the whole run into one row per phase -- this shows the cost structure as it "
            "is now, which is what reveals a refinement or a contact search that has become "
            "expensive while the analysis is still running."
        ),
        dtype=bool,
        default=False,
        optionName="report-performance",
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

    #: This solver supports model modifiers both before the increment loop (initialOnly modifiers)
    #: and mid-run on a periodic cadence via the ``topology-check-frequency`` solver option.
    supportsModelModifiers = True

    #: Option schema for this solver, per OptionSchemaProvider.
    schema = NEDSchema

    NEDOptions = {
        "courant-number": 0.8,
        "output-frequency": 1000,
        "contact-update-frequency": 100,
        "topology-check-frequency": 0,
        "report-performance": False,
    }

    def __init__(self, jobInfo, journal, **kwargs):
        self.journal = journal

        # Ensure mutable defaults (field lists) are isolated per solver instance.
        self.options = deepcopy(self.NEDOptions)
        self._updateOptions(kwargs, journal)
        #: Fields integrated first order (forward Euler) and second order (central difference) in
        #: time, and their degrees of freedom. All four are DERIVED from the assembled inertia and
        #: damping in :meth:`_classifyFieldsByScheme`, never declared.
        self.firstOrderFields = []
        self.secondOrderFields = []
        self.ids_1st = None
        self.ids_2nd = None
        #: Second-order DOFs whose 0.5*m*v**2 is a mechanical energy -- the mass-carrying ones and
        #: the rotational ones -- and equally the DOFs at which prescribed work is an energy. The
        #: energy balance uses these at both ends; see phenomena.carriesKineticEnergy.
        self.ids_mechanicalEnergy = None
        #: Second-order fields whose inertia is a mass, in declaration order; the per-component
        #: momentum diagnostic walks these.
        self.linearMomentumFields = []
        #: Second-order fields whose inertia is neither a mass nor a rotational inertia, in
        #: declaration order. Each is reported on its own line of the energy table and enters
        #: neither balance; see phenomena.inertiaKind.
        self.nonMechanicalSecondOrderFields = []
        #: Mass-proportional damping rate C/M per degree of freedom. Zero wherever the elements
        #: reported no damping, which is what makes the one damped update rule below reduce
        #: exactly to the undamped central difference there.
        self._dampingRate = None
        self._warnedAboutMissingInternalEnergy = False
        self._warnedAboutMissingExternalWork = False
        #: Summed relative drift of each lumped quantity (mass, first-order viscosity,
        #: non-mechanical inertia) over every topology change, keyed by name and checked
        #: against _CUMULATIVE_LUMPED_QUANTITY_DRIFT_TOLERANCE so that many
        #: individually-tolerable changes cannot silently add up to a meaningful one.
        self._cumulativeLumpedQuantityDrift = {}
        #: Whether the cumulative-drift warning has already fired for a given quantity name,
        #: so it is reported once per step rather than once per topology change.
        self._warnedAboutCumulativeDrift = {}
        #: Work done on the model by its prescribed degrees of freedom, accumulated every
        #: increment. Compared against the kinetic energy to detect energy creation; see
        #: _ENERGY_CREATION_TOLERANCE.
        self._externalWork = 0.0
        #: The external work a resumed checkpoint carried, handed to the next solveStep. Staged
        #: rather than assigned directly because readRestart necessarily runs before solveStep,
        #: which resets the live accumulator; see :meth:`readRestart`.
        self._resumedExternalWork = 0.0
        #: Per-constraint force buffer and scatter plan, by constraint name; see
        #: :meth:`assembleConstraintForces`. Cleared whenever the DofManager is rebuilt.
        self._constraintForcePlans = {}

    def _updateOptions(self, updatedOptions: dict, journal):
        """Update options of the solver using a string dict

        Parameters
        ----------
        updatedOptions
            The options dictionary.
        journal
            The journal module.
        """

        for k, v in updatedOptions.items():
            if k in self.NEDOptions:
                journal.message("Updating option {:}={:}".format(k, v), self.identification)
                if isinstance(self.NEDOptions[k], list):
                    for item in v.split(","):
                        self.options[k].append(item.strip())
                else:
                    self.options[k] = type(self.NEDOptions[k])(updatedOptions[k])
            else:
                raise AttributeError("Invalid option {:} for {:}".format(k, self.identification))

    def writeRestart(self, restartFile):
        """Persist the accumulated external work.

        It is an ACCUMULATOR, not a state that can be recomputed: it is summed increment by
        increment from the reaction forces at the prescribed degrees of freedom, so nothing in a
        converged solution reproduces it.

        A resumed run that started it at zero compared its kinetic energy against only the work
        done since the resume -- and that comparison is the check on whether the run is still
        quasi-static and whether energy is being created, so the one diagnostic that would flag a
        run going wrong instead read as though the model were mostly kinetic. The anchor pry-out
        run resumed at 22 % of its ramp reported kinetic energy at 41.7 % of external work for
        that reason alone.

        Parameters
        ----------
        restartFile
            The open checkpoint to write to.
        """
        restartFile.require_group("solver").attrs["externalWork"] = self._externalWork

    def _consumeResumedExternalWork(self) -> float:
        """The external work a resumed checkpoint carried, handed over exactly once.

        The energy balance is per step, so the step being resumed picks up where the checkpoint
        left off while any later step in the same job correctly starts from zero -- hence consumed
        rather than merely read. It is staged in the first place because
        :meth:`readRestart` necessarily runs BEFORE :meth:`solveStep`, which resets the live
        accumulator and would otherwise wipe the restored value before the first increment.
        """
        resumed = self._resumedExternalWork
        self._resumedExternalWork = 0.0
        return resumed

    def readRestart(self, restartFile):
        """Restore the accumulated external work; see :meth:`writeRestart`.

        Tolerates a checkpoint written before this state was carried, in which case the resumed
        step's energy balance is wrong in the old way rather than the run failing.

        Parameters
        ----------
        restartFile
            The open checkpoint to read from.
        """
        if "solver" not in restartFile or "externalWork" not in restartFile["solver"].attrs:
            return
        self._resumedExternalWork = float(restartFile["solver"].attrs["externalWork"])

    def solveStep(
        self,
        step,
        model: FEModel,
        fieldOutputController: FieldOutputController,
        outputmanagers: dict[str, OutputManagerBase],
    ):
        """Public interface to solve for a step.

        Parameters
        ----------
        stepNumber
            The step number.
        step
            The dictionary containing the step definition.
        stepActions
            The dictionary containing all step actions.
        model
            The  model tree.
        fieldOutputController
            The field output controller.
        """

        self.validateModelCapabilities(model)

        # Against this the timing table reports what it did *not* measure, so it has to span
        # everything this method does -- the initial topology refinement and the first equation
        # system included, since both are timed categories that would otherwise be subtracted from a
        # window they never ran in and drive the residue negative.
        stepWallClockTic = perf_counter()

        self._externalWork = self._consumeResumedExternalWork()
        self._cumulativeLumpedQuantityDrift = {}
        self._warnedAboutMissingInternalEnergy = False
        self._warnedAboutMissingExternalWork = False
        self._warnedAboutCumulativeDrift = {}

        # Constraints whose DOF footprint is the outcome of a search, i.e. contact. Collected once,
        # so a model without any pays nothing for the per-increment tick in the loop below.
        self._dynamicConnectivityConstraints = [
            constraint
            for constraint in model.constraints.values()
            if type(constraint).updateConnectivity is not ConstraintBase.updateConnectivity
        ]

        # Modifiers that can still act once the analysis is running. Collected once, so a model whose
        # refinement is all up-front pays nothing for the periodic check below.
        self._liveTopologyModifiers = [
            modifier
            for modifier in model.modelModifiers.values()
            if modifier.initiatesTopologyChanges and not modifier.actsOnlyAtSimulationStart
        ]

        # Step actions before the equation system, matching NIST: nothing they do depends on it.
        self.applyStepActionsAtStepStart(model, step.actions)

        # One topology update, here and nowhere else. Every modifier this solver accepts acts only at
        # the start of the analysis (validateModelCapabilities enforces that), and on its first call
        # hAdaptivity evaluates exactly its initialOnly markers -- so this reproduces what an
        # implicit run does on its own first pass. Running it before anything sized by the equation
        # system exists is what makes it both cheap and safe: the mesh is final before the lumped
        # mass, the multi-point-constraint condensation and the critical time step are derived from
        # it, and no velocity state exists yet that would have to be carried onto new nodes.
        self.updateTopologyAndConnectivity(model, step)

        theSystem = self.buildEquationSystem(model, step)

        Minv = theSystem.Minv
        U, dU, V, P = theSystem.U, theSystem.dU, theSystem.V, theSystem.P
        criticalTimeStep = theSystem.criticalTimeStep

        contactUpdateFrequency = self.options["contact-update-frequency"]
        topologyCheckFrequency = self.options["topology-check-frequency"]
        UAtLastConnectivitySearch = np.array(U)

        # The central-difference velocity update reads 0.5 * (dT + dT_prev). Leaving this None
        # makes the solver synthesise dT_prev = 0 further down, so the first increment gets dT/2 --
        # the half step that starts a leapfrog correctly on a COLD start. A resumed run must not
        # repeat that: the velocity in the checkpoint already carries the half-step offset, so
        # starting again applies one half-impulse too few. Measured on an anchor pry-out resume,
        # that alone left the final reaction force 2.34e-04 wrong while every other piece of state
        # restored correctly.
        restoredTimeIncrement = step.restoredTimeIncrement()
        prevTimeStep = (
            None if restoredTimeIncrement is None else TimeStep(0, 0.0, 0.0, restoredTimeIncrement, 0.0, model.time)
        )

        try:
            for timeStep in step.getTimeStep(enforcedTimeIncrement=criticalTimeStep):
                # only print for increments matching the configured output-frequency
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
                        # The *interval* table, not the cumulative one: what the solver has been
                        # doing since the previous report is what tells a running analysis whether
                        # its cost structure has moved -- a refinement that enlarged the mesh, a
                        # contact search that started admitting far more candidates, a material
                        # that entered a more expensive branch. The cumulative table, printed once
                        # at the end of the step, averages all of that away, and on a run of
                        # millions of increments it is also the only table anyone would ever see.
                        self.journal.printPrettyTable(
                            performancetiming.extractIncrementTimes(skipUnused=True),
                            self.identification,
                        )

                # The mid-run topology check at the end of this same increment re-runs the
                # connectivity search on EVERY constraint, these included. Searching here as well
                # would build and query the same k-d tree twice within one increment: the later
                # search is the one that has to happen, because a refinement in between invalidates
                # whatever this one found. Deferring to it costs one increment of staleness -- the
                # same staleness the configured frequency already accepts, and orders of magnitude
                # below a facet dimension at an explicit time step.
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
                    motionSinceLastSearch = float(np.max(np.abs(np.asarray(U) - UAtLastConnectivitySearch)))
                    UAtLastConnectivitySearch = np.array(U)

                    if connectivityChanged:
                        # The motion is reported rather than assumed: it is the upper bound on how
                        # far a slave node can have travelled relative to its master surface since
                        # the previous search, which is what says whether the configured frequency
                        # is defensible against this model's facet size.
                        self.journal.message(
                            "Constraint connectivity changed; largest nodal motion since the "
                            "previous search: {:e}".format(motionSinceLastSearch),
                            self.identification,
                            2,
                        )
                        theSystem = self.buildEquationSystem(model, step, previous=theSystem)

                        Minv = theSystem.Minv
                        U, dU, V, P = theSystem.U, theSystem.dU, theSystem.V, theSystem.P

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
                        prevTimeStep,
                    )

                except CutbackRequest as e:
                    # A cutback answers a CONVERGENCE failure, and an explicit scheme has no
                    # convergence to fail: its time step is dictated by stability, courant *
                    # dt_crit from the mesh and the wave speed. Shrinking it does nothing for a
                    # material that could not integrate, and doing so was actively destructive --
                    # discardAndChangeIncrement overwrites enforcedTimeIncrement with the reduced
                    # value, the generator reuses that value every iteration afterwards, and
                    # nothing raises it back (the critical step is enforced "lower only", and
                    # SimpleTimeStepper.preventIncrementIncrease is a no-op). One failed
                    # quadrature point permanently crippled the analysis: of three production runs
                    # of the anchor pry-out model, two cut back to minInc and died, and the third
                    # spent 291000 of 300000 increments at ~1e-14 s, covering 6e-09 s of loading.
                    #
                    # So the request is refused and the failure is surfaced where it happened.
                    for man in outputmanagers:
                        man.finalizeFailedIncrement(
                            statusInfoDict=None,
                        )
                    raise StepFailed(
                        "A material requested a cutback in increment {:}: {:}. The explicit time "
                        "step is set by stability, not by convergence, so it cannot be reduced in "
                        "response -- either the material cannot integrate at the stable step, or "
                        "the state reaching it is already wrong. Both need the material or the "
                        "model looked at, not a smaller step.".format(timeStep.number, e)
                    ) from e
                else:
                    # A zero increment is not a completed step: the generator yields one before the
                    # first real increment, and the velocity update returns early for it. Recording
                    # it here would overwrite the increment a RESUMED run was seeded with, putting
                    # the run back on the cold-start half step it must not repeat.
                    if timeStep.timeIncrement > 0.0:
                        prevTimeStep = timeStep

                    with performancetiming.timeit("publish node fields"):
                        for fieldName, field in model.nodeFields.items():
                            self.theDofManager.writeDofVectorToNodeField(U, field, "U")
                            self.theDofManager.writeDofVectorToNodeField(P, field, "P")

                            # Published every increment, not only on output increments, for two
                            # reasons: an h-adaptivity event can fall on any increment and its
                            # interpolation reads this entry, and a restart checkpoint written from a
                            # node field is the only way an explicit run can resume with its kinetic
                            # state intact. It is an O(nDof) copy.
                            self.theDofManager.writeDofVectorToNodeField(V, field, "V")

                        for variable in model.scalarVariables.values():
                            variable.value = U[self.theDofManager.idcsOfScalarVariablesInDofVector[variable]]

                    self.updateRigidBodies(model, timeStep)

                    # Timed because it is not what it looks like. FEModel.advanceToTime is a generic
                    # method shared with the implicit solvers, where it runs once per *converged*
                    # increment and is amortised over a Newton loop; here it runs on every one of
                    # millions of increments, and it is a serial Python loop over every element,
                    # constraint and multi-point constraint in the model.
                    with performancetiming.timeit("accept state"):
                        model.advanceToTime(timeStep.totalTime)

                    if timeStep.number % self.options["output-frequency"] == 0:
                        with performancetiming.timeit("finalize output"):
                            fieldOutputController.finalizeIncrement()
                            for man in outputmanagers:
                                man.finalizeIncrement(
                                    statusInfoDict=None,
                                )

                    # --- h-adaptivity, mid-run ------------------------------------------------
                    # Placed exactly here for three independent reasons. The marker refines on the
                    # last *finalized* field output, so anywhere earlier it would decide on stale
                    # results. The cutback path restores U/V/P from vectors sized by the old equation
                    # system, so a topology change interleaved with a cutback would restore the wrong
                    # length -- ending a successful increment keeps the two paths disjoint. And the
                    # pairing of U with the half-step-staggered V is unambiguous only between
                    # increments.
                    #
                    # Increment 0 is excluded deliberately. The time stepper yields a zero-length
                    # increment first, before it has even taken up the enforced time increment, and
                    # nothing has been solved at that point -- a live marker evaluated there would
                    # refine on the initial condition, and revising the time increment there raises.
                    if (
                        self._liveTopologyModifiers
                        and topologyCheckFrequency
                        and timeStep.number > 0
                        and timeStep.number % topologyCheckFrequency == 0
                    ):
                        lumpedTotalsBefore = self._perFieldLumpedTotals()
                        momentumBefore = self.secondOrderMomentum(self._rawLumpedMass, V, model)
                        kineticBefore = 0.5 * float(
                            np.sum(self._rawLumpedMass[self.ids_mechanicalEnergy] * V[self.ids_mechanicalEnergy] ** 2)
                        )

                        if self.updateTopologyAndConnectivity(model, step):
                            theSystem = self.buildEquationSystem(model, step)

                            Minv = theSystem.Minv
                            U, dU, V, P = theSystem.U, theSystem.dU, theSystem.V, theSystem.P
                            UAtLastConnectivitySearch = np.array(U)

                            # The net force is deliberately NOT re-evaluated on the new mesh. It
                            # could be, with one extra element pass -- but that would run the
                            # constitutive law off-cycle, with a zero strain increment, purely to
                            # obtain a force, and the material state is what that call writes into.
                            # Zeroing costs exactly one increment of force contribution to the
                            # velocity update: an O(dT) error confined to the increment following an
                            # event, after which it is computed normally. A bounded known error is
                            # preferable to an unbounded unknown one.
                            P[:] = 0.0

                            self.reportTopologyChangeConservation(
                                lumpedTotalsBefore, momentumBefore, kineticBefore, V, model
                            )

                            # Lower only. Refinement shrinks the smallest element and tightens the
                            # limit, which must be honoured; softening raises it, and taking that up
                            # mid-step would change the integrator's dispersion for no benefit.
                            if theSystem.criticalTimeStep < criticalTimeStep:
                                self.journal.message(
                                    "Refinement lowered the stable time increment from {:e} to "
                                    "{:e}".format(criticalTimeStep, theSystem.criticalTimeStep),
                                    self.identification,
                                    1,
                                )
                                criticalTimeStep = theSystem.criticalTimeStep
                                step.enforceTimeIncrement(criticalTimeStep)

        except ReachedMaxIncrements:
            self.applyStepActionsAtStepEnd(model, step.actions)

        except ReachedMinIncrementSize:
            self.journal.errorMessage("Incrementation failed", self.identification)
            raise StepFailed()

        except ConditionalStop:
            self.journal.message("Conditional Stop", self.identification)
            self.applyStepActionsAtStepEnd(model, step.actions)

        else:
            self.applyStepActionsAtStepEnd(model, step.actions)

        finally:
            prettyTable = performancetiming.makePrettyTable(wallTime=perf_counter() - stepWallClockTic)
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

        elements = model.elements
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

                # The work put into the model at a prescribed degree of freedom is the reaction force
                # there times the motion it is dragged through. At this point P holds the carried-over
                # net nodal force from the previous increment (external minus internal plus constraint
                # contributions), and the support reaction balancing it is its negative -- so this has
                # to be accumulated HERE, immediately before the zeroing below, which is the only
                # moment the reaction is available.
                #
                # Only where that product is an energy. A prescribed non-local damage, say, has a
                # conjugate "force" in the units its own balance equation carries, and adding it
                # here would put cubic millimetres into a total that is then compared against a
                # kinetic energy in newton-millimetres -- the same units error the kinetic sum
                # below excludes it from, at the other end of the same balance.
                if carriesKineticEnergy(dirichlet.field):
                    self._externalWork -= float(np.dot(P[dirichlet.constrainedDofIndices], prescribedIncrement))

                prescribedVelocity = prescribedIncrement / timeStep.timeIncrement

                P[dirichlet.constrainedDofIndices] = 0.0
                V[dirichlet.constrainedDofIndices] = prescribedVelocity
                prescribedVelocities.append((dirichlet.constrainedDofIndices, prescribedVelocity))

            if self.ids_1st is not None:
                V[self.ids_1st] = Minv[self.ids_1st] * P[self.ids_1st]
            if self.ids_2nd is not None:
                dtAverage = 0.5 * (timeStep.timeIncrement + prevTimeStep.timeIncrement)

                # Central difference with mass-proportional damping: the rate alpha = C/M enters
                # as one factor on each side rather than as an extra force evaluation, so a damped
                # degree of freedom costs what an undamped one does. At alpha = 0 it is
                # bit-identical to the undamped update, which is why there is no branch -- and why
                # mechanical Rayleigh damping would need no code here. The damping is not optional
                # for a hyperbolic non-local field: undamped, the transients minted at the start of
                # the step and at every refinement never decay, and the damage variable follows
                # every overshoot instead of the mean.
                halfRateStep = 0.5 * self._dampingRate[self.ids_2nd] * dtAverage
                V[self.ids_2nd] = (
                    (1.0 - halfRateStep) * V[self.ids_2nd] + Minv[self.ids_2nd] * P[self.ids_2nd] * dtAverage
                ) / (1.0 + halfRateStep)

            # A prescribed velocity is a boundary condition, not a solution, and both updates above
            # overwrote it: the damped one scales it by (1 - alpha dt/2)/(1 + alpha dt/2), the
            # first-order one loses it outright to Minv * 0. The undamped central difference added
            # Minv * 0 * dt and left it exactly, which is why this only needs saying now.
            for constrainedDofIndices, prescribedVelocity in prescribedVelocities:
                V[constrainedDofIndices] = prescribedVelocity

            # slave DOFs of multi-point constraints do not integrate their own equations of motion --
            # they ride along on their masters (Minv is zero there, so the updates above left them
            # untouched); displacements follow automatically via dU = V * dt
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
        P = self.assembleLoads(nodeforces, distributedLoads, bodyForces, U_n, P, timeStep)
        P = self.assembleConstraintForces(model.constraints, U_n, dU, P, timeStep)

        # fold the forces acting on slave DOFs onto their masters (action-reaction through the
        # rigid interpolation link); done here so the Dirichlet handling at the start of the next
        # increment operates on the already-folded vector
        if self.mpcTransformation is not None:
            with performancetiming.timeit("mpc force fold"):
                P[:] = self.mpcTransformation.foldExplicitForce(P)

        if timeStep.number % self.options["output-frequency"] == 0:
            Wint = psi

            # Only over the degrees of freedom where 0.5 * m * v**2 is an energy. Summing the whole
            # vector also collected the first-order fields, whose "mass" is a viscosity and whose
            # "velocity" is that field's rate -- no energy meaning, and orders of magnitude larger
            # than the real term (eta ~ 1e-4 against a density ~ 1e-9): every balance then read as
            # 100 % kinetic.
            Wkin = 0.5 * float(
                np.sum(self._rawLumpedMass[self.ids_mechanicalEnergy] * V[self.ids_mechanicalEnergy] ** 2)
            )

            Wext = self._externalWork

            # Everything the model absorbed that no reported term accounts for: strain energy the
            # material does not publish, plastic and viscous dissipation, and damage. It must stay
            # non-negative -- a negative value means the kinetic energy has overtaken the work put
            # in, i.e. energy is being created.
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

            # One row per second-order field that is not in the balance, never summed with another
            # or into the total. For a hyperbolic non-local field the inertia is a time squared, so
            # 0.5 * m * rate**2 is a volume rather than an energy -- kept on its own line because
            # it is the energy in the ringing the damping exists to remove.
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

            # KE <= W_ext is exact in the continuum, so a violation is energy from nowhere: for an
            # explicit scheme, a time step above the true stability limit. It catches what dt_crit
            # does not see -- the nonlocal field, contact penalty stiffness, a misjudged element
            # shape. The NaN case is checked first because every ordering against a NaN is False,
            # including the `Wext > 0.0` below, so a diverged run would pass silently and keep
            # writing NaN. It fails the step rather than reporting it: nothing after it is
            # meaningful, and an explicit scheme has no cutback to answer it with.
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

            # Said once, loudly, because a silent zero here reads as "no strain energy yet" rather
            # than "this quantity is not being reported", and the ratio above is the usual way one
            # decides whether an explicit run is quasi-static enough.
            # The energy-creation check above is gated on Wext > 0, and _externalWork only
            # accumulates at PRESCRIBED degrees of freedom. A model whose Dirichlet conditions are
            # all fixed and whose loading comes from body forces, node forces, a distributed load
            # or an initial velocity therefore has Wext identically zero, and the check silently
            # never fires -- on precisely the impact/drop class of problem explicit dynamics
            # exists for. Say so once rather than printing a table of dashes.
            #
            # NEGATIVE is the same case and must be caught by the same branch rather than falling
            # between the two: a model that has returned more work at its prescribed degrees of
            # freedom than was put in -- a rebounding or oscillating structure -- satisfies neither
            # Wext > 0 nor Wext == 0, so gating on equality left the diagnostic disabled with
            # nothing said at all. That is the failure this check exists to prevent.
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

    @performancetiming.timeit("distributed loads")
    def computeDistributedLoads(
        self,
        distributedLoads: list[StepActionBase],
        U_np: DofVector,
        PExt: DofVector,
        timeStep: TimeStep,
    ) -> DofVector:
        """Loop over all distributed loads acting on elements, and evaluate them.
        Assembles into the global external load vector.

        Parameters
        ----------
        distributedLoads
            The list of distributed loads.
        U_np
            The current solution vector.
        PExt
            The external load vector to be augmented.
        timeStep
            The current time step.

        Returns
        -------
        DofVector
            The augmented load vector.
        """

        time = timeStep.totalTime
        dT = timeStep.timeIncrement

        for dLoad in distributedLoads:
            load = dLoad.getCurrentLoad(timeStep)
            for faceID, elementSet in dLoad.surface.items():
                for el in elementSet:
                    Pe = np.zeros(el.nDof)
                    Ke = np.zeros((el.nDof, el.nDof)).ravel()
                    el.computeDistributedLoad(dLoad.loadType, Pe, Ke, faceID, load, U_np[el], time, dT)

                    PExt[el] += Pe

        return PExt

    @performancetiming.timeit("body forces")
    def computeBodyForces(
        self,
        bodyForces: list[StepActionBase],
        U_np: DofVector,
        PExt: DofVector,
        timeStep: TimeStep,
    ) -> DofVector:
        """Loop over all body forces loads acting on elements, and evaluate them.
        Assembles into the global external load vector and the system matrix.

        Parameters
        ----------
        distributedLoads
            The list of distributed loads.
        U_np
            The current solution vector.
        PExt
            The external load vector to be augmented.
        increment
            The increment.

        Returns
        -------
        tuple[DofVector,VIJSystemMatrix]
            The augmented load vector and system matrix.
        """

        time = timeStep.totalTime
        dT = timeStep.timeIncrement

        for bForce in bodyForces:
            force = bForce.getCurrentLoad(timeStep)
            for el in bForce.elementSet:
                Pe = np.zeros(el.nDof)
                Ke = np.zeros((el.nDof, el.nDof)).ravel()

                el.computeBodyForce(Pe, Ke, force, U_np[el], time, dT)

                PExt[el] += Pe

        return PExt

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
        Is is called by solveStep() in each iteration.

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

    @performancetiming.timeit("assemble loads")
    def assembleLoads(
        self,
        nodeForces: list[StepActionBase],
        distributedLoads: list[StepActionBase],
        bodyForces: list[StepActionBase],
        U_np: DofVector,
        PExt: DofVector,
        timeStep: TimeStep,
    ) -> tuple[DofVector, VIJSystemMatrix]:
        """Assemble all loads into a right hand side vector.

        Parameters
        ----------
        nodeForces
            The list of concentrated (nodal) loads.
        distributedLoads
            The list of distributed (surface) loads.
        bodyForces
            The list of body (volumetric) loads.
        U_np
            The current solution vector.
        PExt
            The external load vector.
        timeStep
            The current time step.

        Returns
        -------
        tuple[DofVector,VIJSystemMatrix]
            - The augmented external load vector.
            - The augmented system matrix.
        """
        for cLoad in nodeForces:
            PExt[
                self.theDofManager.idcsOfFieldsOnNodeSetsInDofVector[cLoad.field][cLoad.nodeSet]
            ] += cLoad.getCurrentLoad(timeStep).flatten()
        PExt = self.computeDistributedLoads(distributedLoads, U_np, PExt, timeStep)
        PExt = self.computeBodyForces(bodyForces, U_np, PExt, timeStep)

        return PExt

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

        # A modifier that acts only at the start of the analysis is served by the single topology
        # update before the increment loop. One that acts later needs the update re-run, which this
        # solver does -- but only if it has been told how often, because a marker reads the last
        # *finalized* field output and finalization is itself on a cadence. Refusing to guess is the
        # point: a silently-never-refining run looks exactly like a converged one.
        lateModifiers = sorted(
            name
            for name, modifier in model.modelModifiers.items()
            if modifier.initiatesTopologyChanges and not modifier.actsOnlyAtSimulationStart
        )
        topologyCheckFrequency = self.options["topology-check-frequency"]
        outputFrequency = self.options["output-frequency"]

        # All three cadence options are validated here, together, because each is used as a modulo
        # operand further down and none of them fails usefully on its own:
        #
        #  * output-frequency is a DIVISOR, not a flag: 0 does not mean "never report", it means
        #    ZeroDivisionError from a bare traceback on the first increment. It must be >= 1.
        #  * a NEGATIVE cadence is silently accepted otherwise. It is truthy, so it passes the
        #    "is it set" gates; and when its magnitude is a multiple of output-frequency it also
        #    passes the multiple check below (-20 % 20 == 0) and then fires at exactly the same
        #    increments as its absolute value (n % -20 == 0 for the same n as n % 20 == 0). When
        #    the magnitude is NOT a multiple, the multiple-check error blames the wrong thing.
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

    @performancetiming.timeit("topology update")
    def updateTopologyAndConnectivity(self, model: FEModel, step) -> bool:
        """Run the topology update, then let every mesh-dependent consumer catch up on it.

        The same two-phase sequence the implicit solver runs at the start of each of its increments
        (see :meth:`~edelweissfe.solvers.nonlinearimplicitstatic.NIST.solveStep`): the modifiers plan
        and apply to a fixed point inside one topology window, then the pure readers of a settled
        model -- surface facets, tie and contact connectivity -- catch up, once, on the net change.
        Both sweeps are materialised rather than short-circuited: neither may be skipped because the
        other already reported a change.

        Parameters
        ----------
        model
            The model tree.
        step
            The step being solved.

        Returns
        -------
        bool
            Whether anything changed, i.e. whether the equation system has to be built afresh. The
            only caller today builds it unconditionally right afterwards; the return value is what
            makes this reusable from inside an increment loop.
        """

        modelHasChanged = model.updateTopology(step, model.time)

        refreshed = model.refreshMeshDependents()
        ticked = any([constraint.updateConnectivity(model) for constraint in model.constraints.values()])

        return modelHasChanged or refreshed or ticked

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

    @performancetiming.timeit("build equation system")
    def buildEquationSystem(self, model: FEModel, step, previous: ExplicitSystem = None) -> ExplicitSystem:
        """Build the equation system and everything sized by it.

        Called once before the increment loop, and again from inside it whenever a constraint reports
        that its DOF footprint changed -- one method for both, so the path every model takes and the
        path only a contact model takes cannot drift apart.

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

        self.journal.message("Creating monolithic equation system", self.identification, verbosity)
        self.theDofManager = DofManager(
            model.nodeFields.values(),
            model.scalarVariables.values(),
            model.elements.values(),
            model.constraints.values(),
            model.nodeSets.values(),
        )
        self.journal.message(
            "total size of eq. system: {:}".format(self.theDofManager.nDof),
            self.identification,
            verbosity,
        )

        if not isRebuild:
            self.journal.printSeperationLine()

        # self.options already reflects every >>options, name=<this solver's name>, ... block applied
        # so far, applied as each block is constructed or re-declared; there is nothing to reset or
        # re-fetch here.

        self.mpcTransformation = self.buildMPCTransformation(model, step.actions)
        self.checkMPCDirichletConflicts(self.mpcTransformation, step.actions)

        # The constraint force buffers and their index plans belong to the DofManager that was just
        # (re)built: a refinement changes both a constraint's DOF count and where its DOFs sit, and
        # a stale plan would scatter forces to the wrong degrees of freedom silently.
        self._constraintForcePlans = {}

        # initialize mass and damping matrices
        M = self.theDofManager.constructDofVector()  # initialize lumped mass matrix
        Minv = self.theDofManager.constructDofVector()  # initialize inverse lumped mass matrix

        U = self.theDofManager.constructDofVector()  # initialize displacement vector
        dU = self.theDofManager.constructDofVector()  # initialize displacement vector
        V = self.theDofManager.constructDofVector()  # initilize velocity vector
        P = self.theDofManager.constructDofVector()  # initialize reaction vector

        M[:] = 0.0
        for el in model.elements.values():
            Me = np.zeros(el.nDof)
            el.computeLumpedInertia(Me)
            M[el] += Me

        # Each field's FIRST-derivative coefficient: zero mechanically, the non-local viscosity
        # always (whether or not that field also has an inertia -- see computeLumpedInertia()
        # above, the SECOND-derivative coefficient).
        damping = self.theDofManager.constructDofVector()
        damping[:] = 0.0
        for el in model.elements.values():
            Ce = np.zeros(el.nDof)
            el.computeLumpedDamping(Ce)
            damping[el] += Ce

        # Checked here, because the inertia check below never sees it at a second-order DOF: there
        # the divisor stays the positive inertia and the damping enters only as the rate C/M. A
        # negative rate amplifies the transient the damping exists to remove, and alpha*dt/2 = -1
        # makes the update's denominator exactly zero. Both are silent.
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

        # Which time derivative each field carries, read off the two vectors just assembled. An
        # inertia is what a central-difference update divides by, so a field with one is second
        # order and a field without one is not; no deck answer that disagreed could be honoured.
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

        # Which second-order fields may be summed into a linear momentum and which into the energy
        # balance. The assembled inertia cannot say -- a density, a rotational inertia and a
        # micro-inertia are all just positive numbers -- and it is a fact about the field rather
        # than about this analysis, so it comes from the registry. See phenomena.inertiaKind.
        self.linearMomentumFields = [f for f in self.secondOrderFields if carriesLinearMomentum(f)]
        self.nonMechanicalSecondOrderFields = [f for f in self.secondOrderFields if not carriesKineticEnergy(f)]
        self.ids_mechanicalEnergy = self._dofIndicesOfFields(
            [f for f in self.secondOrderFields if carriesKineticEnergy(f)]
        )

        # A first-order field integrates by forward Euler, C * rate = P, and needs the damping
        # computeLumpedDamping() reports here, not the (correctly zero) inertia. From here on this
        # vector is "the divisor", whichever of the two coefficients a degree of freedom's scheme
        # actually divides by.
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

        # A negative lumped mass is the classical failure mode of row-summing a quadratic element's
        # consistent mass matrix, and it is worse than a zero one: the update stays finite, the run
        # continues, and those degrees of freedom integrate backwards in time. The quadratic elements
        # here blend the linear shape functions in precisely to avoid it, which is exactly why this
        # is worth stating rather than trusting.
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
            # Folded with the same operator as the inertia it is divided by, so the ratio below is
            # the damping rate of the FOLDED system. Slaves of one material cancel exactly; slaves
            # of several blend by inertia, which is what the folded equation of motion has.
            self.mpcTransformation.foldLumpedMass(damping)

        Minv[M != 0.0] = 1.0 / M[M != 0.0]

        # Mass-proportional damping rate alpha = C / M, wherever an inertia is divided by and a
        # damping was assembled alongside it. Slave DOFs fold to zero inertia and integrate nothing,
        # so they stay at zero rather than dividing by it, and a DOF whose elements reported no
        # damping keeps the zero that makes the update exactly the undamped one. Second-order DOFs
        # only: a first-order one had its inertia overwritten with its damping, so C/M would be 1.0.
        self._dampingRate = self.theDofManager.constructDofVector()
        self._dampingRate[:] = 0.0
        integrating = self.ids_2nd[M[self.ids_2nd] > 0.0]
        self._dampingRate[integrating] = damping[integrating] / M[integrating]

        # kept (instead of 1/Minv) for the kinetic energy: slave DOFs have Minv = 0
        self._lumpedMass = M

        if not isRebuild:
            for fieldName, field in model.nodeFields.items():
                U = self.theDofManager.writeNodeFieldToDofVector(U, field, "U")
                P = self.theDofManager.writeNodeFieldToDofVector(P, field, "P")

                # The velocity entry exists only once this solver has published one, i.e. from the
                # second build onwards. Reading it back is what carries the kinetic state across an
                # h-adaptivity event: the modifier interpolated it onto the new nodes with the same
                # operator it used for U (see hadaptivity.WARM_STARTED_NODE_FIELD_ENTRIES), and there
                # is nowhere else it could come from -- a fresh vector would silently resume from
                # rest.
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
            # The mesh is unchanged (the layout check above establishes that), so the stable time
            # increment is unchanged too, and recomputing it would cost a full element pass -- the
            # material asks for its wave speed by evaluating its own tangent at every quadrature
            # point. Reusing it is also the conservative direction: as the material softens the true
            # limit only grows, and raising the time increment mid-run would change the integrator's
            # dispersion for no benefit.
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

        # Buffers and index plans, one per constraint, built once per equation system rather than
        # per increment -- this runs on the explicit hot path, tens of thousands of times. Both are
        # invalidated by exactly one event, a rebuild of the DofManager, which is where the cache is
        # cleared; nothing else can change a constraint's DOF count or its indices.
        PPlain = P.asPlainArray()

        for name, constraint in constraints.items():
            plan = self._constraintForcePlans.get(name)
            if plan is None:
                indices = P.entitiesInDofVector[constraint]
                # A constraint may name the same DOF more than once -- a slave node that also
                # appears in its own master facet's node list -- and += would then keep only the
                # last write instead of summing the contributions. That is the only reason
                # np.add.at is needed, and it is the rare case: decide once, here, instead of
                # paying its dispatch on every constraint of every increment.
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
        """The linear momentum of the second-order fields, per spatial component.

        Per component, not summed over the whole block: adding a momentum's x, y and z contributions
        together produces a number with no physical meaning and would hide a component-wise error
        behind a cancellation. A field occupies a contiguous slice of the dof vector, node-major with
        the component innermost -- that is what ``writeNodeFieldToDofVector``'s ``flatten()``
        establishes -- so reshaping the slice recovers the per-node vectors.

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

        total = None
        for fieldName in self.linearMomentumFields:
            indices = self.theDofManager.idcsOfFieldsInDofVector[fieldName]
            dimension = model.nodeFields[fieldName].dimension
            perNodeMass = np.asarray(mass[indices]).reshape((-1, dimension))
            perNodeVelocity = np.asarray(V[indices]).reshape((-1, dimension))
            contribution = np.sum(perNodeMass * perNodeVelocity, axis=0)

            # Two fields of different spatial dimension have no common momentum, and numpy would
            # not say so: adding a shape (1,) contribution to a shape (3,) total BROADCASTS it onto
            # every spatial component. The non-mechanical fields are excluded above; this is the rest.
            if total is not None and contribution.shape != total.shape:
                raise ValueError(
                    "Second-order field {:} has dimension {:} against {:} for the fields before it, "
                    "so their momenta have no common components to add.".format(
                        fieldName, contribution.shape[0], total.shape[0]
                    )
                )

            total = contribution if total is None else total + contribution

        return total if total is not None else np.zeros(0)

    def _perFieldLumpedTotals(self) -> dict[str, float]:
        """The assembled lumped total of every first- and second-order field, each on its own.

        Never summed across fields. A conservation bug that halves one field's total can be
        diluted below detection by another field's total if the two differ enough in magnitude
        -- which happens routinely here, since a mechanical density and a non-local viscosity or
        non-mechanical inertia are not just different units, they are typically many orders of
        magnitude apart in value. That is true even between two fields of the SAME kind (two
        mechanical fields of very different density would have the same problem), so the fix is
        per field, not per "mechanical vs. not". :meth:`_checkLumpedQuantityConserved` is called
        once per field with its own total, so a violation anywhere is visible regardless of what
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

    def _classifyFieldsByScheme(self, inertia: DofVector, damping: DofVector) -> tuple[list[str], list[str]]:
        """Which fields are integrated second order in time and which first, read off the
        assembled operators rather than declared.

        The deck used to say this and had no freedom in what to say: an inertia is what a
        central-difference update divides by, so a field carrying one is second order and a field
        carrying none is not. Deriving it also closes a hole the declaration had -- a field named
        in neither list was silently never integrated at all.

        Per FIELD, not per degree of freedom: ``*elementProperty`` takes an element set, so giving
        a micro-inertia to only some of the elements carrying a field is easy to do by accident,
        and integrating part of a field as a wave equation and the rest as a diffusion one is not
        a scheme anybody chose. Such a field is refused rather than split.

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
                    "usual cause is an *elementProperty assigning the inertia (for a non-local "
                    "field, 'nonlocal micro inertia') over an element set that does not cover "
                    "every element carrying the field.".format(
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

    def _checkLumpedQuantityConserved(self, label: str, before: float, after: float) -> float:
        """Check one row-sum-lumped per-element quantity for exact conservation across a
        topology change, and warn once per step if many individually-tolerable changes have
        accumulated into a meaningful one.

        Any quantity assigned as a per-element scalar and lumped with the same row-sum weights
        -- mass, a first-order field's viscosity, a second-order field's non-mechanical inertia
        -- is conserved by the SAME geometric identity: the children of a refined element tile
        it and carry the same value. What the quantity physically is plays no part in that;
        only how it is assembled does, which is why this one check serves all of them.

        Parameters
        ----------
        label
            Name of the quantity, used in the raised message and as the key for its own
            cumulative drift and warned-once state.
        before, after
            The total before and after the change.

        Returns
        -------
        float
            The relative change, for the caller to report.

        Raises
        ------
        RuntimeError
            If the relative change exceeds _LUMPED_QUANTITY_CONSERVATION_TOLERANCE.
        """

        relativeChange = abs(after - before) / before if before > 0.0 else 0.0
        self._cumulativeLumpedQuantityDrift[label] = (
            self._cumulativeLumpedQuantityDrift.get(label, 0.0) + relativeChange
        )

        if relativeChange > _LUMPED_QUANTITY_CONSERVATION_TOLERANCE:
            raise RuntimeError(
                "A topology change did not conserve the total lumped {:}: {:e} became {:e}, a "
                "relative change of {:e} against a tolerance of {:e}. The children of a refined "
                "element tile it and carry the same value, so it is conserved geometrically; the "
                "quadrature that assembles it is exact only up to a polynomial order, which "
                "admits a small change. A violation of this size is not quadrature -- it means "
                "the refinement or the lumping is wrong.".format(
                    label, before, after, relativeChange, _LUMPED_QUANTITY_CONSERVATION_TOLERANCE
                )
            )

        # Reported, not raised: each individual change was within the exact-conservation bound, so
        # what accumulates here is quadrature error rather than a violated invariant, and aborting a
        # multi-hour run on an accumulated heuristic is out of proportion. The per-change check raises.
        if self._cumulativeLumpedQuantityDrift[
            label
        ] > _CUMULATIVE_LUMPED_QUANTITY_DRIFT_TOLERANCE and not self._warnedAboutCumulativeDrift.get(label, False):
            self._warnedAboutCumulativeDrift[label] = True
            self.journal.message(
                "The accumulated relative {:} drift over this step has reached {:e}, above the "
                "tolerance of {:e}. Each individual topology change was within its own bound, so "
                "this is many small quadrature changes adding up rather than one bad refinement; "
                "the model's {:} is no longer the one the step started with.".format(
                    label,
                    self._cumulativeLumpedQuantityDrift[label],
                    _CUMULATIVE_LUMPED_QUANTITY_DRIFT_TOLERANCE,
                    label,
                ),
                self.identification,
                1,
            )

        return relativeChange

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
          via :meth:`_checkLumpedQuantityConserved` -- summing across fields first, even ones
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

        relativeChangeByField = {
            fieldName: self._checkLumpedQuantityConserved(fieldName, before, lumpedTotalsAfter.get(fieldName, 0.0))
            for fieldName, before in lumpedTotalsBefore.items()
        }
        worstField, worstRelativeChange = (
            max(relativeChangeByField.items(), key=lambda item: item[1]) if relativeChangeByField else ("", 0.0)
        )

        # Minv = 1/M is formed wherever M != 0.0 and only negative mass is rejected, so a master
        # DOF left with a tiny positive mass yields an enormous Minv and integrates itself to
        # infinity. Report the smallest inertia actually carried by an integrating DOF, and how far
        # it sits below the median, so a collapsing mass is visible when it appears.
        integrating = self._lumpedMass[self._lumpedMass > 0.0]
        smallestMass = float(np.min(integrating)) if integrating.size else 0.0
        medianMass = float(np.median(integrating)) if integrating.size else 0.0
        massRatio = smallestMass / medianMass if medianMass > 0.0 else 0.0

        # Split by integration scheme, because the two numbers mean different things: a second-order
        # DOF divides an inertia into a central-difference update, a first-order one divides a
        # damping into a forward-Euler update, and their stability limits are different expressions.
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

        momentumChange = float(np.max(np.abs(momentumAfter - momentumBefore))) if momentumBefore.size else 0.0
        momentumScale = float(np.max(np.abs(momentumBefore))) if momentumBefore.size else 0.0
        relativeKineticJump = abs(kineticAfter - kineticBefore) / kineticBefore if kineticBefore > 0.0 else 0.0

        self.journal.message(
            "Topology change: worst-conserved field '{:}' to {:.1e} relative; smallest integrating "
            "coefficient {:.3e} ({:.1e} of median) [2nd-order inertia {:.3e} of median {:.3e}; "
            "1st-order damping {:.3e} of median {:.3e}]; largest momentum component change "
            "{:.3e} (of {:.3e}); kinetic energy {:.6e} -> {:.6e} ({:+.2f} %)".format(
                worstField,
                worstRelativeChange,
                smallestMass,
                massRatio,
                smallest2nd,
                median2nd,
                smallest1st,
                median1st,
                momentumChange,
                momentumScale,
                kineticBefore,
                kineticAfter,
                relativeKineticJump * 100.0 * (1.0 if kineticAfter >= kineticBefore else -1.0),
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
