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


from copy import deepcopy
from dataclasses import dataclass

import numpy as np

import edelweissfe.utils.performancetiming as performancetiming
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


#: Tolerance on the relative change of the total lumped mass across a single topology
#: change. The children of a refined element tile it and carry the same density, so the
#: mass is a geometric identity -- but it is assembled by Gauss quadrature, which is exact
#: only up to a polynomial order. A distorted hexa20 has a non-polynomial Jacobian, so
#: repartitioning a parent into children changes the quadrature truncation error. Measured
#: on the anchor pry-out model, one live refinement moves the total mass by 4.63e-08
#: relative; this bound leaves more than an order of magnitude of headroom above that
#: while still catching a refinement or lumping error, which would be O(1), not O(1e-8).
_MASS_CONSERVATION_TOLERANCE = 1e-6

#: Tolerance on the accumulated relative mass drift over a whole step. A single change
#: being within tolerance does not bound a run with hundreds of refinements, so the drift
#: is summed and checked separately. At the measured 4.63e-08 per change, this permits
#: over two thousand refinements before tripping.
_CUMULATIVE_MASS_DRIFT_TOLERANCE = 1e-4


@dataclass(frozen=True)
class NEDSchema:
    """The options of the ``*solver`` datalines and of an ``>>options`` block routed to this
    solver, owned by this module and never mutated from outside it.

    Mirrors :attr:`NED.NEDOptions` one-for-one; the plain ``self.options`` dict remains the actual
    source of truth consulted at runtime (see :class:`~edelweissfe.solvers.nonlinearimplicitstatic.NISTSchema`
    for why). The ``*-fields``/``*-scheme``/``courant-number``/``output-frequency`` option names are
    not valid Python identifiers, hence the ``optionName`` indirection. ``firstOrderFields``/
    ``secondOrderFields`` are declared ``dtype=list`` to describe their real shape (a comma-separated
    list, appended to rather than replaced -- see :meth:`NED._updateOptions`), even though nothing
    coerces a raw string against this schema today.
    """

    firstOrderFields: list | None = schemaField(
        description="Fields integrated with a first-order (forward-Euler) time scheme.",
        dtype=list,
        default_factory=list,
        optionName="first-order-fields",
    )
    secondOrderFields: list | None = schemaField(
        description="Fields integrated with a second-order (central-difference) time scheme.",
        dtype=list,
        default_factory=list,
        optionName="second-order-fields",
    )
    firstOrderScheme: str | None = schemaField(
        description="The time integration scheme for first-order fields.",
        dtype=str,
        default="forward-euler",
        optionName="first-order-scheme",
    )
    secondOrderScheme: str | None = schemaField(
        description="The time integration scheme for second-order fields.",
        dtype=str,
        default="central-difference",
        optionName="second-order-scheme",
    )
    courantNumber: float | None = schemaField(
        description="The fraction of the critical time step actually used.",
        dtype=float,
        default=0.8,
        optionName="courant-number",
    )
    outputFrequency: int | None = schemaField(
        description="The increment interval at which progress is logged.",
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
            "search (contact) re-run that search."
        ),
        dtype=int,
        default=100,
        optionName="contact-update-frequency",
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
    M
        The lumped mass, with multi-point-constraint slave mass already folded onto its masters.
    Minv
        The inverse lumped mass. Zero on multi-point-constraint slave DOFs, which integrate no
        equation of motion of their own.
    U
        The solution vector.
    dU
        The solution increment of the increment being computed.
    V
        The velocity vector, staggered half an increment behind ``U``.
    P
        The net nodal force -- external minus internal -- which drives the velocity update.
    U_old
        ``U`` as of the start of the current increment, kept for the cutback path.
    V_old
        ``V`` as of the start of the current increment, kept for the cutback path.
    P_old
        ``P`` as of the start of the current increment, kept for the cutback path.
    criticalTimeStep
        The stable time increment for the current mesh, already scaled by the courant number.
    """

    M: DofVector
    Minv: DofVector
    U: DofVector
    dU: DofVector
    V: DofVector
    P: DofVector
    U_old: DofVector
    V_old: DofVector
    P_old: DofVector
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

    #: This solver runs the topology update exactly once, before its increment loop, which fully
    #: serves any modifier that only ever acts at the start of the analysis. One that would act later
    #: is refused by validateModelCapabilities rather than silently never running.
    supportsModelModifiers = True

    #: Option schema for this solver, per OptionSchemaProvider.
    schema = NEDSchema

    NEDOptions = {
        "first-order-fields": [],
        "second-order-fields": [],
        "first-order-scheme": "forward-euler",
        "second-order-scheme": "central-difference",
        "courant-number": 0.8,
        "output-frequency": 1000,
        "contact-update-frequency": 100,
        "topology-check-frequency": 0,
    }

    def __init__(self, jobInfo, journal, **kwargs):
        self.journal = journal

        # Ensure mutable defaults (field lists) are isolated per solver instance.
        self.options = deepcopy(self.NEDOptions)
        self._updateOptions(kwargs, journal)
        self.ids_1st = None
        self.ids_2nd = None
        self._warnedAboutMissingInternalEnergy = False
        #: Summed relative mass drift over every topology change, checked against
        #: _CUMULATIVE_MASS_DRIFT_TOLERANCE so that many individually-tolerable changes
        #: cannot silently add up to a meaningful one.
        self._cumulativeMassDrift = 0.0

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
        U_old, V_old, P_old = theSystem.U_old, theSystem.V_old, theSystem.P_old
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
                if (
                    self._dynamicConnectivityConstraints
                    and timeStep.number > 0
                    and timeStep.number % contactUpdateFrequency == 0
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
                        U_old, V_old, P_old = theSystem.U_old, theSystem.V_old, theSystem.P_old

                U_old[:] = U
                V_old[:] = V
                P_old[:] = P
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

                    for fieldName, field in model.nodeFields.items():
                        self.theDofManager.writeDofVectorToNodeField(U, field, "U")
                        self.theDofManager.writeDofVectorToNodeField(P, field, "P")

                        # Published every increment, not only on output increments, for two reasons:
                        # an h-adaptivity event can fall on any increment and its interpolation reads
                        # this entry, and a restart checkpoint written from a node field is the only
                        # way an explicit run can resume with its kinetic state intact. It is an
                        # O(nDof) copy.
                        self.theDofManager.writeDofVectorToNodeField(V, field, "V")

                    for variable in model.scalarVariables.values():
                        variable.value = U[self.theDofManager.idcsOfScalarVariablesInDofVector[variable]]

                    self.updateRigidBodies(model, timeStep)

                    model.advanceToTime(timeStep.totalTime)

                    if timeStep.number % self.options["output-frequency"] == 0:
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
                        massBefore = float(np.sum(self._rawLumpedMass))
                        momentumBefore = self.secondOrderMomentum(self._rawLumpedMass, V, model)
                        kineticBefore = 0.5 * float(np.sum(self._rawLumpedMass[self.ids_2nd] * V[self.ids_2nd] ** 2))

                        if self.updateTopologyAndConnectivity(model, step):
                            theSystem = self.buildEquationSystem(model, step)

                            Minv = theSystem.Minv
                            U, dU, V, P = theSystem.U, theSystem.dU, theSystem.V, theSystem.P
                            U_old, V_old, P_old = theSystem.U_old, theSystem.V_old, theSystem.P_old
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

                            self.reportTopologyChangeConservation(massBefore, momentumBefore, kineticBefore, V, model)

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
            prettyTable = performancetiming.makePrettyTable()
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

        # Enforce the Dirichlet boundary conditions on the constrained DOFs:
        # there is no free equilibrium there, so their force P is set to zero,
        # and their velocity is prescribed as (prescribed increment) / (time step).
        for dirichlet in dirichlets:
            P[dirichlet.constrainedDofIndices] = 0.0
            V[dirichlet.constrainedDofIndices] = (
                dirichlet.getPrescribedIncrement(timeStep).flatten() / timeStep.timeIncrement
            )

        if self.ids_1st is not None:
            V[self.ids_1st] = Minv[self.ids_1st] * P[self.ids_1st]
        if self.ids_2nd is not None:
            V[self.ids_2nd] += (
                Minv[self.ids_2nd] * P[self.ids_2nd] * 0.5 * (timeStep.timeIncrement + prevTimeStep.timeIncrement)
            )

        # slave DOFs of multi-point constraints do not integrate their own equations of motion --
        # they ride along on their masters (Minv is zero there, so the updates above left them
        # untouched); displacements follow automatically via dU = V * dt
        if self.mpcTransformation is not None:
            self.mpcTransformation.applySlaveKinematics(V)

        # update displacement increment vector
        np.multiply(V, timeStep.timeIncrement, out=dU)
        np.add(U_n, dU, out=U_n)

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
            P[:] = self.mpcTransformation.foldExplicitForce(P)

        if timeStep.number % self.options["output-frequency"] == 0:
            Wint = psi

            # Restricted to the second-order fields, which are the only ones carrying inertia in the
            # mechanical sense. Summing over the whole vector also collected the first-order fields,
            # whose "mass" is a viscosity (the gradient-enhanced nonlocal field's eta) and whose
            # "velocity" is that field's rate -- a product with no energy meaning, and one that can
            # be orders of magnitude larger than the real term (eta ~ 1e-4 against a density
            # ~ 1e-9). On any gradient-enhanced model that made the split read as 100 % kinetic
            # regardless of what the structure was actually doing.
            Wkin = 0.5 * float(np.sum(self._rawLumpedMass[self.ids_2nd] * V[self.ids_2nd] ** 2))

            W = Wint + Wkin
            if W > 0.0:
                self.journal.message(
                    "Internal energy: {:e} ({:.2f} %)".format(Wint, Wint / W * 100), self.identification, 2
                )
                self.journal.message(
                    "Kinetic energy:  {:e} ({:.2f} %)".format(Wkin, Wkin / W * 100), self.identification, 2
                )

            # Said once, loudly, because a silent zero here reads as "no strain energy yet" rather
            # than "this quantity is not being reported", and the ratio above is the usual way one
            # decides whether an explicit run is quasi-static enough.
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

        A **model modifier that would act after the analysis has started.** This solver runs the
        topology update once, before its increment loop, so a modifier whose every change happens
        there is fully served. One that would change the mesh later is not: the lumped mass, the
        multi-point-constraint condensation and the critical time step are all derived from the mesh
        at that single point, and -- unlike an implicit solver, which keeps everything it carries
        between increments in node fields -- the central-difference velocity is solver-local, so
        there is nothing to interpolate it onto newly created nodes with. Refusing is the honest
        outcome; ``PLAN_LIVE_AMR_EXPLICIT.md`` records what lifting this needs.

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

        presentVariableNames = list(self.theDofManager.idcsOfFieldsInDofVector.keys())

        if self.theDofManager.idcsOfScalarVariablesInDofVector:
            presentVariableNames += [
                "scalar variables",
            ]

        # self.options already reflects every >>options, name=<this solver's name>, ... block applied
        # so far, applied as each block is constructed or re-declared; there is nothing to reset or
        # re-fetch here.

        self.mpcTransformation = self.buildMPCTransformation(model, step.actions)
        self.checkMPCDirichletConflicts(self.mpcTransformation, step.actions)

        # initialize mass and damping matrices
        M = self.theDofManager.constructDofVector()  # initialize lumped mass matrix
        Minv = self.theDofManager.constructDofVector()  # initialize inverse lumped mass matrix

        U = self.theDofManager.constructDofVector()  # initialize displacement vector
        dU = self.theDofManager.constructDofVector()  # initialize displacement vector
        V = self.theDofManager.constructDofVector()  # initilize velocity vector
        P = self.theDofManager.constructDofVector()  # initialize reaction vector

        U_old = self.theDofManager.constructDofVector()  # initialize old displacement vector
        V_old = self.theDofManager.constructDofVector()  # initilize old velocity vector
        P_old = self.theDofManager.constructDofVector()  # initialize old reaction vector

        M[:] = 0.0
        for el in model.elements.values():
            Me = np.zeros(el.nDof)
            el.computeLumpedInertia(Me)
            M[el] += Me

        # compute inverses
        if np.any(M == 0.0):
            raise ValueError(
                "Zero mass found in mass vector. This can be caused by elements with zero density, or by elements with zero volume."
            )

        # A negative lumped mass is the classical failure mode of row-summing a quadratic element's
        # consistent mass matrix, and it is worse than a zero one: the update stays finite, the run
        # continues, and those degrees of freedom integrate backwards in time. The quadratic elements
        # here blend the linear shape functions in precisely to avoid it, which is exactly why this
        # is worth stating rather than trusting.
        if np.any(M < 0.0):
            raise ValueError(
                "Negative mass found in {:} of {:} entries of the lumped mass vector (smallest: "
                "{:e}). A negative lumped mass makes the explicit update integrate backwards in time "
                "at those degrees of freedom.".format(int(np.count_nonzero(M < 0.0)), M.shape[0], M.min())
            )

        # Kept before folding so the kinetic energy diagnostic accounts for the true velocities of
        # all nodes (including tied slaves) rather than master-placed folded mass.
        self._rawLumpedMass = M.copy()

        # Slave DOFs of multi-point constraints carry no own inertia: their mass is folded onto
        # their masters (row-sum lumping of T^T M T, mass-conserving), their Minv stays zero, and
        # their kinematics are assigned directly from the masters each increment.
        if self.mpcTransformation is not None:
            self.mpcTransformation.foldLumpedMass(M)

        Minv[M != 0.0] = 1.0 / M[M != 0.0]

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

        self.ids_1st = np.empty(0, dtype=int)
        self.ids_2nd = np.empty(0, dtype=int)

        # check if all fields are specified either in first-order-fields or second-order-fields
        isSpecified = {presentVariable: False for presentVariable in presentVariableNames}
        for fieldName in self.options["first-order-fields"] + self.options["second-order-fields"]:
            if fieldName not in presentVariableNames:
                raise ValueError(
                    "Field {:} specified in first-order-fields, but not present in model".format(fieldName)
                )
            if isSpecified[fieldName]:
                raise ValueError(
                    "Field {:} specified multiple times in first-order-fields and second-order-fields: {:}, {:}".format(
                        fieldName, self.options["first-order-fields"], self.options["second-order-fields"]
                    )
                )
            isSpecified[fieldName] = True

        # assign indices of fields to first-order and second-order update schemes
        for fieldName in self.options["first-order-fields"]:
            self.ids_1st = np.r_[self.ids_1st, self.theDofManager.idcsOfFieldsInDofVector[fieldName]]
        for fieldName in self.options["second-order-fields"]:
            self.ids_2nd = np.r_[self.ids_2nd, self.theDofManager.idcsOfFieldsInDofVector[fieldName]]

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
            M=M,
            Minv=Minv,
            U=U,
            dU=dU,
            V=V,
            P=P,
            U_old=U_old,
            V_old=V_old,
            P_old=P_old,
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
        :meth:`~edelweissfe.constraints.base.constraintbase.ConstraintBase.applyConstraintForcesOnly`
        is the entry point that says so. Its default still builds a throwaway container through the
        implicit system matrix' own protocol, so every constraint works unchanged; a constraint for
        which the tangent is a real cost overrides it, as the deformable-surface contact does -- there
        the per-slave outer product and block writes were 2.5 ms per increment on a 280k-dof model,
        and the container another 1.2 ms, all discarded. A constraint that could act *only* through
        its tangent would contribute nothing here; that is precisely the class
        :meth:`validateModelCapabilities` refuses.

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

        for constraint in constraints.values():
            Pc = np.zeros(constraint.nDof)

            constraint.applyConstraintForcesOnly(U_np[constraint], dU[constraint], Pc, timeStep)

            # np.add.at rather than +=: a constraint may name the same DOF more than once (a slave
            # node that also appears in its own master facet's node list), and += would keep only the
            # last write instead of summing the contributions.
            np.add.at(P, P.entitiesInDofVector[constraint], Pc)

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
        for fieldName in self.options["second-order-fields"]:
            indices = self.theDofManager.idcsOfFieldsInDofVector[fieldName]
            dimension = model.nodeFields[fieldName].dimension
            perNodeMass = np.asarray(mass[indices]).reshape((-1, dimension))
            perNodeVelocity = np.asarray(V[indices]).reshape((-1, dimension))
            contribution = np.sum(perNodeMass * perNodeVelocity, axis=0)
            total = contribution if total is None else total + contribution

        return total if total is not None else np.zeros(0)

    def reportTopologyChangeConservation(
        self,
        massBefore: float,
        momentumBefore: np.ndarray,
        kineticBefore: float,
        V: DofVector,
        model: FEModel,
    ):
        """Report what a topology change did to the quantities that ought to survive it.

        Refinement interpolates the velocity onto new nodes and re-lumps the mass, and the three
        invariants behave differently under that:

        * **Total mass is conserved exactly.** The children of a refined element tile it and carry the
          same density, so this is an identity rather than an approximation -- which makes it the
          cheapest correctness check on the entire transfer, and the one that catches a connectivity
          or geometry error nothing else would notice. Violating it raises.
        * **Linear momentum is conserved exactly for a spatially uniform velocity field**, because
          the shape functions are a partition of unity and the child masses sum to the parent's. For
          a general field the discrepancy is second order in the velocity gradient across the parent:
          discretisation error, not a defect. Reported, not enforced.
        * **Kinetic energy is not conserved** by interpolation plus re-lumping, and it is the most
          sensitive of the three, being quadratic in the interpolation error. Reported as a relative
          jump; more than roughly a percent is a reason to look at the transfer rather than to
          believe the physics.

        Parameters
        ----------
        massBefore
            Total lumped mass before the change.
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
            If the total lumped mass changed.
        """

        massAfter = float(np.sum(self._rawLumpedMass))
        momentumAfter = self.secondOrderMomentum(self._rawLumpedMass, V, model)
        kineticAfter = 0.5 * float(np.sum(self._rawLumpedMass[self.ids_2nd] * V[self.ids_2nd] ** 2))

        relativeMassChange = abs(massAfter - massBefore) / massBefore if massBefore > 0.0 else 0.0
        self._cumulativeMassDrift += relativeMassChange

        if relativeMassChange > _MASS_CONSERVATION_TOLERANCE:
            raise RuntimeError(
                "A topology change did not conserve the total lumped mass: {:e} became {:e}, a "
                "relative change of {:e} against a tolerance of {:e}. The children of a refined "
                "element tile it and carry the same density, so the mass is conserved geometrically; "
                "the quadrature that assembles it is exact only up to a polynomial order, which "
                "admits a small change. A violation of this size is not quadrature -- it means the "
                "refinement or the mass lumping is wrong.".format(
                    massBefore, massAfter, relativeMassChange, _MASS_CONSERVATION_TOLERANCE
                )
            )

        if self._cumulativeMassDrift > _CUMULATIVE_MASS_DRIFT_TOLERANCE:
            raise RuntimeError(
                "The accumulated relative lumped-mass drift over this step has reached {:e}, above "
                "the tolerance of {:e}. Each individual topology change was within its own bound, so "
                "this is many small quadrature changes adding up rather than one bad refinement; the "
                "model mass is no longer the one the step started with.".format(
                    self._cumulativeMassDrift, _CUMULATIVE_MASS_DRIFT_TOLERANCE
                )
            )

        # Minv = 1/M is formed wherever M != 0.0 and only negative mass is rejected, so a master
        # DOF left with a tiny positive mass yields an enormous Minv and integrates itself to
        # infinity. Report the smallest inertia actually carried by an integrating DOF, and how far
        # it sits below the median, so a collapsing mass is visible when it appears.
        integrating = self._lumpedMass[self._lumpedMass > 0.0]
        smallestMass = float(np.min(integrating)) if integrating.size else 0.0
        medianMass = float(np.median(integrating)) if integrating.size else 0.0
        massRatio = smallestMass / medianMass if medianMass > 0.0 else 0.0

        # Split by integration scheme. The second-order (displacement) field carries rho-based
        # inertia and integrates with central difference; the first-order (nonlocal damage) field
        # carries eta_nl-based inertia and integrates with forward Euler, whose stability limit is
        # a completely different expression. A collapsing mass means something different in each,
        # so the aggregate minimum above cannot be acted on without knowing which field it is in.
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
            "Topology change: mass conserved to {:.1e} relative; smallest integrating mass {:.3e} "
            "({:.1e} of median) [2nd-order {:.3e} of median {:.3e}; 1st-order {:.3e} of median "
            "{:.3e}]; largest momentum component change "
            "{:.3e} (of {:.3e}); kinetic energy {:.6e} -> {:.6e} ({:+.2f} %)".format(
                relativeMassChange,
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
