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

An increment solves no linear system. Second-order fields are advanced by central differences
(a leapfrog, with the velocity staggered half an increment behind the displacement), first-order
fields -- a nonlocal damage field, for instance -- by forward Euler, and both divide by a **lumped**
mass, so the cost of an increment is one element pass plus vector arithmetic. There is no linear
solver, no Newton loop, and no cutback: the time increment is set by stability, not by convergence,
so a material that fails to integrate at the time increment stability allows is reported as an error rather than
answered with a smaller step.

**Words, one per quantity.** The *critical time step* is the largest time increment the central difference method is
stable for, :math:`2/\\omega_\\mathrm{max}` of the assembled system; every part of the model gives an *estimate* of its
own, a lower bound of it (:doc:`/theory/criticaltimestep`). A *bound* is on the side of the eigenvalue only: an upper
bound of :math:`\\omega_\\mathrm{max}^2`, from which an estimate is made. The *time increment* is what the step takes:
at most the fraction ``courant-number`` of the critical time step. The *inertia* of a field is the coefficient of its
second time derivative -- a mass on the displacement, a micro-inertia on a nonlocal field --, its *damping* the
coefficient of the first; *lumped* means summed onto the diagonal, node by node.

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

**The critical time step.** The time increment is the fraction ``courant-number`` of the critical time step of the
model, the smallest of the critical time steps of its parts, each estimated by its own policy
(:mod:`~edelweissfe.solvers.base.criticaltimestep`, and :doc:`/theory/criticaltimestep`): the finite elements -- the
smallest of the elements' own estimates (``computeCriticalTimeStepForExplicitDynamics``; for a Marmot continuum element,
two over the square root of an upper bound of the largest eigenvalue of its stiffness scaled by its lumped mass) -- once
per mesh and at the start of every step, and deliberately *not* for a new numbering of the same mesh: an estimate costs
a full element pass, and the policy assumes that within a step neither the geometry nor the wave speed of an element
changes enough to use up the margin of the courant number (see
:class:`~edelweissfe.solvers.base.criticaltimestep.CriticalTimeStepOfTheElements` for where that fails); the entities
that carry their own inertia (below) when the margin of their last estimate may be used up. A penalty constraint (a
contact, a support or a tie held by a penalty) owns no degrees of freedom; it adds its stiffness to the frequencies of
the degrees of freedom it couples: right after every evaluation of the net force, and after a refinement, the row sums
of its stiffness, scaled by the lumped mass, are added to the parts' estimates there, from the stiffness it had in that
force, since the next increment integrates that force; stiffness that is new in a force is half an increment late
(:class:`~edelweissfe.solvers.base.criticaltimestep.CriticalTimeStepWithThePenaltyConstraints`). It is enforced where it
is below the largest increment the step allows, and a step that asks for smaller increments keeps them.

When the critical time step is estimated, and where ``NED`` does it:

* :meth:`NED.beginStep`: both parts and the penalty constraints forget their estimates (``reset``), since a material
  property may have changed with the step;
* :meth:`NED.onNewNumbering`: the elements are estimated, when the mesh is not the one of their last estimate (the
  first numbering of a step, a refinement);
* :meth:`NED.prepareIncrement`: in a model without entities that carry their own inertia, the penalty constraints at
  the state the step starts from, and after a refinement at the state on the new mesh; the critical time step is
  enforced at the start of a step (not in a resumed step, which keeps the time increment of its checkpoint), and after
  a refinement;
* :meth:`NED.attemptIncrement`: in a model without entities that carry their own inertia, the penalty constraints after
  every increment, right after its net force, and the critical time step is enforced at once;
* :meth:`NED._evaluateTheNetForceOnTheNewConnectivity`: the entities, after every increment, when their estimate is
  due, and the penalty constraints, right after the net force on the new connectivity; the critical time step is
  enforced at once;
* :meth:`NED._enforceTheCriticalTimeStep`: the one place where the parts are combined
  (:meth:`NED._criticalTimeStepOfTheModel`) and the time increment is set from them;
* :meth:`NED.endStep`: the report, how often each part was estimated in the step, and its last estimate.

**Lowered only, or followed.** By default, the time increment is lowered only, never raised within a step; with
``follow-critical-time-step``, it follows the critical time step up as well as down, at the cost of the safety margin
that lowering only keeps against a stiff state that returns. Both, and when to choose which, are explained under
:ref:`ned-time-increment-lowered-or-followed`.

**Three cadences, all in increments.** ``output-frequency`` is how often progress is logged and
field outputs are finalized (a divisor, so it must be at least 1). ``topology-check-frequency`` is
how often model modifiers may change the mesh mid-run, and must be a multiple of
``output-frequency`` because a marker reads the last finalized output; ``0`` runs the topology
update only once, before the increment loop. ``contact-update-frequency`` is how often constraints
whose connectivity is the outcome of a search re-run it; ``0`` disables that. The reasoning behind
the latter two, and what they cost in accuracy, is documented under
:doc:`/theory/adaptivity` and :doc:`/theory/contact` respectively.

**Constraints.** Multi-point constraints are enforced by folding a slave's mass and force onto its
masters (row-sum, mass-conserving) and slaving it kinematically, which adds no stiffness. Other
constraints -- contact above all -- are evaluated through
:meth:`~edelweissfe.constraints.base.constraintbase.ConstraintBase.computeKernelsExplicit`, which
asks for forces and no tangent, since a tangent assembled here would enter no system of equations; their stiffness
enters the critical time step only, through the bound of its row sums
(:meth:`~edelweissfe.constraints.base.constraintbase.ConstraintBase.addBoundOfAbsoluteStiffnessTimes`). A constraint
that can act *only* through its tangent contributes nothing to an explicit increment and is
refused at validation rather than silently ignored.

**The lumped mass is the mass of the start of the step.** The lumped operators are assembled once for the numbering,
at the first increment of every step, from the elements as they are then, and kept until the next numbering (the
entities that carry their own inertia excepted, see below). A change of
a material property during the step (:doc:`/analysis/stepactions`, ``changeMaterialProperty``) takes effect on the
mass at the next step, as it does on the critical time step.

**The discrete system.** It is built, like that of every solver, by
:meth:`~edelweissfe.solvers.base.nonlinearsolverbase.NonlinearSolverBase.updateDiscreteSystem`, from the
change the model reports (:class:`~edelweissfe.models.change.Change`). A new numbering -- the first one of a step, and
a refinement of the mesh -- gives the lumped operators, the classification of the fields by time derivative, the
multi-point-constraint condensation and the vectors of the increments, and the critical time step of the mesh
(:meth:`NED.onNewNumbering`). A contact search that couples other nodes gives new locations on the same numbering: the
lumped operators are functions of the elements and of the numbering and are kept, the vectors keep their values, and
only the locations they are wrapped with are new (:meth:`NED.onNewLocations`). The elements the increment evaluates are
the ones of the discrete system, those that carry kernels.

**Entities that carry their own inertia.** A model may hold, besides finite elements, entities that carry their state,
and with it their inertia, in themselves, and couple other nodes as they move: the RKPM particles of an
EdelweissMeshfree model, whose kernel functions move with them
(:attr:`~edelweissfe.models.discretesystem.DiscreteSystem.entitiesWithOwnInertia`). The nodes of their kernel functions
carry no total solution: their values are the coefficients of one increment, and no element assembles a mass there.
In NED, "own inertia" means that an entity forms its inertia itself, from its state, and lumps it onto the nodes it
couples: the solver integrates those nodes, as it integrates the nodes of the elements. (``NID``'s option
``inertia="entities"`` means more: there, the entities also integrate their inertia themselves.) What distinguishes
them in NED is that they are advanced on one connectivity and evaluated on the next. The leapfrog is the same, on every
node; an increment of such a model has three parts (the calls a Marmot particle receives in them, in their order, and
what each may change: Marmot's page "Meshfree Particles", section "The explicit life cycle of a particle"):

1. **advance**: the elements compute their internal force with the solution and the increment, and the entities take
   the increment on the kernel functions they were assigned at its start
   (:meth:`~edelweissfe.models.discretesystem.DiscreteSystem.advanceEntitiesWithOwnInertia`), and move;
2. **accept and reconnect**: the model accepts the state, assigns the entities to the kernel functions they reach now
   and says what that changed (:meth:`~edelweissfe.models.femodel.FEModel.updateConnectivity`), and the discrete system
   follows, through the one build path: new locations if the entities reach other nodes of the same numbering, a new
   numbering if the nodes of the active domain changed. The internal force of the elements is carried to a new
   numbering element by element; nothing is evaluated again;
3. **evaluate**: on the new kernel functions, the entities return their internal force, their lumped inertia and
   damping, their lumped momentum and, when its estimate is due, their critical time step
   (:meth:`~edelweissfe.models.discretesystem.DiscreteSystem.assembleInternalForceAndMomentumOfEntitiesWithOwnInertia`,
   :meth:`~edelweissfe.models.discretesystem.DiscreteSystem.assembleLumpedMassOfEntitiesWithOwnInertia`,
   :meth:`~edelweissfe.models.discretesystem.DiscreteSystem.assembleLumpedDampingOfEntitiesWithOwnInertia`,
   :meth:`~edelweissfe.models.discretesystem.DiscreteSystem.criticalTimeStepOfEntitiesWithOwnInertia`), and with the
   internal force of the elements and the forces of the loads and the constraints, they give the net force. The
   configuration is accepted, no increment is left: the loads and the constraints are evaluated with a zero
   increment, so only a constraint that measures its condition at the configuration acts here
   (:attr:`~edelweissfe.constraints.base.constraintbase.ConstraintBase.actsInExplicitAnalysis`).

The inertia and the damping of the entities are assembled anew in every increment -- they are theirs at the kernel
functions as they are at their current positions, which change in every increment, whether or not the locations do --
and enter the lumped operators at their nodes; those of the elements are assembled per numbering. The scheme of a
field follows from both, as for the elements: a field with an inertia is integrated by central differences, damped by
its damping, one with a damping alone by forward Euler -- e.g. a nonlocal field of gradient-enhanced particles, with or
without a micro-inertia --, also in a model of particles alone, whose fields are classified in the first evaluation of
a step. No node is both: a node the elements couple carries its total solution and their operators, and an entity that
lumps an inertia or a damping to it is refused. A node of the entities without a coefficient (their kernel function
reaches them only where it vanishes) has no equation of motion: it gets no velocity, and its ``P`` is its internal
force. The velocity of the nodes is the state the next increment
starts from, and the model holds it in the entry ``V`` of its node fields, for the nodes of the entities as for those of
the elements: it is written at acceptance and read back by node whenever the discrete system gets a new numbering, so a
node keeps its velocity while the active nodes change around it, and a node that has just become active starts at
rest. With ``project-velocity-every-increment``, the velocity of the nodes of the entities is projected from their
momentum in every increment instead, :math:`\boldsymbol{v} = \boldsymbol{M}^{-1} \boldsymbol{p}`, which dissipates
more; it is what the material point method does. The nodes of the elements are never projected.

The critical time step is the smaller of the elements' (of the mesh) and the estimate of the critical time step of the
entities' assembled system, which changes with their configuration: it is estimated anew in the zero increment, and
whenever the margin of its last estimate over the time increment in use may be used up -- by the fall of their lumped
inertia since, which is followed in every increment, and by the assumed fall per increment from their stiffness
(``critical-time-step-decrease-per-increment``) --, and lowers the time increment whenever it falls below it (or sets
it, with ``follow-critical-time-step``, see :ref:`ned-time-increment-lowered-or-followed`). The estimate is made from
a Collatz-Wielandt bound of the largest eigenvalue of the absolute value of the entities' mass-scaled stiffness, from
weights constant per field; with ``carry-critical-time-step-weights``, ``NED`` keeps the weights an estimate leaves and
hands them to the next, so that every estimate takes one more step of the power iteration towards the Perron vector,
where the bound is exact. An estimate below the one the weights started from starts them anew, and the larger of the
two is taken; the weights start anew also with every numbering and every step, and after a resume
(:class:`~edelweissfe.solvers.base.criticaltimestep.CriticalTimeStepOfEntitiesWithOwnInertia`, "Carried weights"). The
first increment of a step must be the zero increment: in it, the entities take an increment of zero length, from which
they evaluate the force, the inertia, the damping and the critical time step of the configuration the step starts from.
Where the model holds the motion of their nodes
(:meth:`~edelweissfe.models.discretesystem.DiscreteSystem.carriersOfTheMotion`) -- after a step of this solver, or of
``NID`` integrating the inertia at the nodes --, the step continues it as at the nodes of elements, from their velocity
and the net force of their acceleration: the force evaluated in the zero increment would not do, since the entities'
integration refers to the increment taken last. Otherwise -- at the start of an analysis, with the initial velocity
given to the particles, after a quasi-static step, or after ``NID`` integrating the inertia in the entities --, the
force evaluated in the zero increment is taken, and the velocity of their nodes is projected from their momentum. The
nodes of the elements always start from the model's velocity, which is how a projectile gets its initial velocity.
The kinetic energy of the balance, and the momentum of the diagnostics, are those of the nodal velocity with the
lumped mass, on the nodes of the entities as on those of the elements: the state the leapfrog integrates. The
checkpoint holds the inertia and the damping of the entities with the net force, and a resumed step finds their
connectivity again.
An entity that can integrate its own inertia by the Newmark scheme, as an implicit dynamic step asks of it, carries
none here: at the start of every step, this solver gives the entities zero Newmark parameters
(:meth:`~edelweissfe.models.femodel.FEModel.setNewmarkParametersOfEntities`), as every quasi-static solver does.
Model modifiers that act after the start of the analysis are not supported in such a model, and neither is a
multi-point constraint whose slave is a degree of freedom of the nodes of the entities: its inertia would be folded onto
its masters, and the velocity and the kinetic energy published for it would be zero. ``NED`` refuses such a constraint
when it builds the numbering.

**Energy balance.** Every reporting increment prints the energy the model held at the start of the
step, kinetic and internal (``E_0``), the external work accumulated at prescribed degrees of freedom
since, the kinetic energy, the internal (strain) energy, and what remains unaccounted. Its purpose is
one check that nothing else performs: the kinetic energy cannot exceed the energy at the start of the
step plus the external work, because the terms omitted from the balance are all non-negative, so a
violation means energy is appearing from nowhere and the time step is above the true stability limit
-- which the critical time step does not see in full: it may see a penalty contact late, or not at all
(:doc:`/theory/criticaltimestep`, "What the estimate does not see"), and the nonlocal field only where that field
carries a non-mechanical inertia. Such a field's
own ``0.5 m_k rate^2`` is reported on a separate line and kept out of the balance: with a
coefficient in seconds squared it has the units of a volume, not of an energy. ``E_0`` is read off
the model's state, nothing evaluated: zero for a step from rest in a model without strain energy,
and, for a step that continues another one, the energy that one left, its kinetic energy included.
The continued step's balance is therefore the one the two steps would report as one, less what the
first of them dissipated.

Two honest limits, both of which the solver states in the log rather than leaving to be
discovered. The external work accumulates only at *prescribed* degrees of freedom, so a step
driven by body forces, node forces, a distributed load or an initial velocity has none, and the
check cannot fire, whatever energy the step starts with; and the internal energy is whatever the
materials publish as a strain energy, which not every material does. When either is identically
zero -- or the external work is negative, a structure returning work at its boundary -- the balance
is not usable as a quasi-static criterion, and the honest substitute is to integrate a reaction
force against its prescribed displacement, both of which are available as ``saveHistory`` field
outputs.

**Diagnostics across a topology change.** Because the lumped mass is the operator an increment
divides by, a refinement's effect on it is reported: total mass, per-component linear momentum and
kinetic energy before and after, plus the relative mass drift accumulated over every such change,
so that many individually-tolerable changes cannot silently add up to a meaningful one. The
smallest coefficient the increment divides by is reported alongside -- an inertia at a
second-order degree of freedom, a damping at a first-order one -- since that is what bounds the
step.

**The force it publishes.** The node field result ``P`` is, as in every solver, the internal force
with the inertial and damping forces, :math:`\\boldsymbol{f}_\\mathrm{int} + \\boldsymbol{M}
\\boldsymbol{a} + \\boldsymbol{C} \\boldsymbol{v}`, without the loads and the constraints (see
:doc:`/output/nodefieldresults`). The central difference makes the inertial and damping
forces at the end of an increment exactly the net force :math:`\\boldsymbol{f}_\\mathrm{net}` the
next velocity update divides by -- that update solves :math:`\\boldsymbol{M} (\\boldsymbol{v}_{n+3/2}
- \\boldsymbol{v}_{n+1/2}) / \\Delta \\bar{t} + \\boldsymbol{C} (\\boldsymbol{v}_{n+3/2} +
\\boldsymbol{v}_{n+1/2}) / 2 = \\boldsymbol{f}_\\mathrm{net}` -- so ``P`` is
:math:`\\boldsymbol{f}_\\mathrm{int} + \\boldsymbol{f}_\\mathrm{net}` at every degree of freedom with
an equation of motion. Two kinds have none. At a degree of freedom with a prescribed motion, the
acceleration at the end of the increment is set by the prescribed velocity of the next increment,
which is not known when the increment is accepted, so ``P`` there is
:math:`\\boldsymbol{f}_\\mathrm{int}` alone: the reaction, without the inertia of the support's own
lumped mass. A multi-point-constraint slave carries no inertia of its own -- it is folded onto its
masters, whose ``P`` holds it -- so ``P`` there is its :math:`\\boldsymbol{f}_\\mathrm{int}` as well.

**The velocity and the acceleration it publishes.** Within a step, the velocity ``V`` the model
receives at every accepted increment is the leapfrog's own, :math:`\\boldsymbol{v}_{n+1/2}`, half an
increment behind the solution; on a field integrated by forward Euler, it is the rate of the
increment just taken. The acceleration ``A`` is the one at the end of the increment: the one with
which the velocity at that time, :math:`\\boldsymbol{v}_{n+1} = \\boldsymbol{v}_{n+1/2} + \\tfrac{1}{2}
\\Delta t \\, \\boldsymbol{a}_{n+1}`, satisfies the equation of motion, :math:`\\boldsymbol{M}
\\boldsymbol{a}_{n+1} + \\boldsymbol{C} \\boldsymbol{v}_{n+1} = \\boldsymbol{f}_\\mathrm{net}`, so

.. math::

    \\left( \\boldsymbol{M} + \\tfrac{1}{2} \\Delta t \\, \\boldsymbol{C} \\right) \\boldsymbol{a}_{n+1}
        = \\boldsymbol{f}_\\mathrm{net} - \\boldsymbol{C} \\boldsymbol{v}_{n+1/2},

which is :math:`\\boldsymbol{M}^{-1} \\boldsymbol{f}_\\mathrm{net}` without damping. On a field
integrated by forward Euler, which has no inertia, it is the change of the rate from the increment
just taken to the rate at the end of it. It is zero where the motion is prescribed, as the next
increment's prescribed velocity sets it, and a multi-point-constraint slave takes it from its
masters, as it takes its velocity. At the end of the step, the velocity is advanced by the half
increment it lags behind (:meth:`~edelweissfe.models.femodel.FEModel.advanceVelocity`), so that the
step leaves the velocity at its end time, as every solver does (see
:doc:`/output/nodefieldresults`).

**The net force is solver state, and a step starts from the model's state.** The net force lives
within one discrete system, and is never read back from the model. New locations after a change
of a constraint's connectivity carry it over, and a resumed step takes it from its
checkpoint. After a mesh change it starts at zero: re-evaluating it would run the constitutive law
off-cycle and write into the material state, and zero is an O(dT) error for one increment. At the
start of a step, it is the net force of the state the model holds, :math:`\\boldsymbol{f}_\\mathrm{net}
= \\boldsymbol{M} \\boldsymbol{a}_0 + \\boldsymbol{C} \\boldsymbol{v}_0`, formed from the model's
velocity and acceleration with the lumped operators, without evaluating anything (and zero where the
motion is prescribed), and the first increment is the leapfrog's half step from
:math:`\\boldsymbol{v}_0`. After a quasi-static step, which leaves the model at rest, it is zero: the
start state is taken as in equilibrium, exact after a quasi-static step that converged there (a
gravity preload, for instance), and a lag of one increment for a load applied suddenly at the start.
After an explicit step, it is the net force that step ended with, and the leapfrog continues as
though the two steps were one (with damping, to second order in the time increment). After an
implicit dynamic step, the step starts from the velocity and the acceleration that step ended with.

**Restart.** A resumed run must not repeat the half-step that starts a leapfrog: the velocity in
the checkpoint already carries the half-increment offset, so applying the startup again would apply
one half-impulse too few. The solver therefore takes the previous time increment and the net force
from the restart state rather than synthesising them, and the velocity field is checkpointed in its
own right, with the model -- an implicit solver reconstructs everything it needs from the
displacement, a central-difference scheme does not.
"""

from copy import deepcopy
from dataclasses import dataclass
from time import perf_counter

import numpy as np

import edelweissfe.utils.performancetiming as performancetiming
from edelweissfe.config.phenomena import carriesKineticEnergy, carriesLinearMomentum
from edelweissfe.models.change import Change
from edelweissfe.models.femodel import FEModel
from edelweissfe.numerics.dofvector import DofVector
from edelweissfe.outputmanagers.base.outputmanagerbase import OutputManagerBase
from edelweissfe.solvers.base.conservationchecks import (
    CONSERVATION_TOLERANCE,
    ConservationCheck,
    formatMomentumAndKineticEnergy,
    linearMomentum,
)
from edelweissfe.solvers.base.criticaltimestep import (
    CriticalTimeStepOfEntitiesWithOwnInertia,
    CriticalTimeStepOfTheElements,
    CriticalTimeStepWithThePenaltyConstraints,
)
from edelweissfe.solvers.base.nonlinearsolverbase import NonlinearSolverBase
from edelweissfe.timesteppers.timestep import TimeStep
from edelweissfe.utils.exceptions import CutbackRequest, StepFailed
from edelweissfe.utils.fieldoutput import FieldOutputController
from edelweissfe.utils.schema import schemaField

#: Fractional margin by which the kinetic energy may exceed the energy at the step start plus the external work
#: before it is reported as energy creation. KE <= E_0 + W_ext holds in the continuum; mass lumping and the re-lumping
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
    projectVelocityEveryIncrement: bool = schemaField(
        description=(
            "Project the velocity of the nodes of entities that carry their own inertia (the particles of a meshfree "
            "model) from their lumped momentum in every increment, not only at the start of a step whose model holds "
            "no motion at these nodes. Projecting dissipates more; it is what the material point method does. The "
            "nodes of finite elements are never projected."
        ),
        dtype=bool,
        default=False,
        optionName="project-velocity-every-increment",
    )
    criticalTimeStepDecreasePerIncrement: float | None = schemaField(
        description=(
            "The largest relative decrease per increment that the critical time step of the entities that carry their "
            "own inertia (the particles of a meshfree model) is assumed to have from their stiffness between two "
            "estimates; the decrease from their lumped inertia and damping is followed exactly in every increment. It "
            "sets how long an estimate is trusted while another part sets the time increment: an estimate is taken "
            "anew once its margin over the time increment in use may be used up at this rate. An assumption, measured "
            "on gradient-enhanced particles (see the theory page on the critical time step); 1 estimates in every "
            "increment."
        ),
        dtype=float,
        default=0.1,
        optionName="critical-time-step-decrease-per-increment",
    )
    followCriticalTimeStep: bool = schemaField(
        description=(
            "Let the time increment follow the critical time step up as well as down, at most the step's maxInc, each "
            "time the critical time step is taken; off by default: lowered only, never raised within a step. See 'The "
            "time increment: lowered only, or followed' in the documentation of NED."
        ),
        dtype=bool,
        default=False,
        optionName="follow-critical-time-step",
    )
    carryCriticalTimeStepWeights: bool = schemaField(
        description=(
            "Carry the weights of the estimate of the critical time step of the entities that carry their own inertia "
            "(the particles of a meshfree model) from one estimate to the next, and refine them by one "
            "Collatz-Wielandt step each time: the estimate tightens towards the spectral radius it bounds over a few "
            "estimates, in the pass the estimate takes anyway. An estimate below the one the weights started from "
            "(with weights constant per field) starts them anew, and so do every numbering and every step, and a "
            "resume."
        ),
        dtype=bool,
        default=False,
        optionName="carry-critical-time-step-weights",
    )


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
        "project-velocity-every-increment": False,
        "critical-time-step-decrease-per-increment": 0.1,
        "follow-critical-time-step": False,
        "carry-critical-time-step-weights": False,
        # Lists, so _updateOptions appends the comma-separated items. Empty means "assert nothing".
        "expect-second-order-fields": [],
        "expect-first-order-fields": [],
    }

    #: The last completed increment and the net force, both read by the central-difference velocity
    #: update of the next increment, the lumped inertia and damping of the entities that carry their own inertia (the
    #: particles of a meshfree model), which the next increment divides by, and the external work, an accumulator that
    #: cannot be reconstructed from the solution, with the energy at the step start it adds to. The velocity and the
    #: acceleration are the model's (entries ``V`` and ``A`` of its node fields).
    checkpointedState = {
        "prevTimeStep": TimeStep,
        "_netForce": np.ndarray,
        "_inertiaOfEntities": np.ndarray,
        "_dampingOfEntities": np.ndarray,
        "_numberingOfNetForce": str,
        "_externalWork": float,
        "_energyAtStepStart": float,
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
        #: time, derived from the mesh in :meth:`_classifyFieldsByScheme`, and their DOFs in the numbering.
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
        #: The DOFs whose inertia the entities that carry their own inertia lump to them (the nodes of the kernel
        #: functions of a meshfree model's particles), and no element; empty in a finite element model. Their inertia
        #: and damping are assembled in every increment (see :meth:`_takeLumpedOperatorsOfEntitiesWithOwnInertia`).
        self.ids_inertiaOfEntities = None
        #: The lumped inertia and damping of the elements, unfolded, vectors of the numbering, assembled for every
        #: numbering (see :meth:`_assembleLumpedOperators`).
        self._inertiaOfTheElements = None
        self._dampingOfTheElements = None
        #: The lumped inertia and damping of the entities that carry their own inertia at the configuration last
        #: accepted, vectors of the numbering; None before their first evaluation in a discrete system, and in a finite
        #: element model.
        self._inertiaOfEntities = None
        self._dampingOfEntities = None
        #: The version of the model's topology
        #: (:attr:`~edelweissfe.models.topologypipeline.TopologyPipeline.version`) the classification of the fields is
        #: derived for: a property of the mesh, derived again when it changed, and at the start of every step, not for a
        #: new numbering of the same mesh. None until it is first derived in a step.
        self._meshOfTheClassification = None
        #: The critical time steps of the two parts of a model, and of the parts with the penalty constraints, each
        #: estimated by its own policy (see :mod:`~edelweissfe.solvers.base.criticaltimestep`): the finite elements once
        #: per mesh, the entities that carry their own inertia when the margin of their last estimate may be used up,
        #: the parts with the penalty constraints after every evaluation of the net force.
        self._criticalTimeStepOfTheElements = CriticalTimeStepOfTheElements()
        self._criticalTimeStepOfTheEntities = CriticalTimeStepOfEntitiesWithOwnInertia(
            self.options["critical-time-step-decrease-per-increment"], self.options["carry-critical-time-step-weights"]
        )
        self._criticalTimeStepWithThePenaltyConstraints = CriticalTimeStepWithThePenaltyConstraints()
        #: The internal (strain) energy the elements publish at the end of the last increment, for the energy balance.
        self._internalEnergy = 0.0
        #: The operators of the explicit update, arrays of the length of the numbering, assembled by
        #: :meth:`_assembleLumpedOperators` for every new numbering; the elements and the numbering determine them,
        #: not the locations. The inverse lumped mass, zero on multi-point-constraint slave DOFs, which integrate no
        #: equation of motion of their own:
        self._Minv = None
        #: The lumped mass the increment divides by, with the multi-point-constraint fold, and the unfolded one. The
        #: diagnostics need the true mass of every node, a tied slave included.
        self._lumpedMass = None
        self._rawLumpedMass = None
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
        #: Work done by the prescribed DOFs in the step, accumulated every increment; with the energy at
        #: the step start, compared against the kinetic energy to detect energy creation.
        self._externalWork = 0.0
        #: The kinetic and internal energy the model held at the start of the step (see
        #: :meth:`energyOfTheModelsState`), which the external work of the step adds to.
        self._energyAtStepStart = 0.0
        #: The last completed increment; the velocity update reads 0.5 * (dT + dT_prev). None makes
        #: the first increment of a cold step the leapfrog's half step; a resumed step uses the
        #: checkpointed one, since the checkpointed velocity already carries the half-step offset.
        self.prevTimeStep = None
        #: The net force of the last completed increment, which the next velocity update divides by; a
        #: vector of the current discrete system. None between steps, so that the first build of a step
        #: can tell a resumed step (whose checkpoint holds it) from a cold one, which forms it from the
        #: state of the model (see :meth:`formNetForceOfTheModelsState`).
        self._netForce = None
        #: The digest of the numbering the net force is a vector of
        #: (:attr:`~edelweissfe.numerics.dofnumbering.DofNumbering.digest`), which the checkpoint holds with it: a
        #: resumed step takes the net force over only if its numbering is the one rebuilt from the resumed model.
        self._numberingOfNetForce = None
        #: The inertia and the damping of every degree of freedom of the current discrete system, the
        #: coefficients of its equation of motion :math:`M a + C v = f_\mathrm{net}`; see
        #: :meth:`_assembleLumpedOperators`.
        self._inertia = None
        self._damping = None
        #: The vectors the increments work on, of the current discrete system, wrapped with its locations. They are the
        #: solution, its increment, the velocity (staggered half an increment behind the solution) and the
        #: acceleration at the end of the last increment, which the model receives with the velocity (see
        #: :meth:`accelerationAtTheEndOfTheIncrement`); at the start of a step the one the model holds. The elements'
        #: sum of the increment, :math:`\boldsymbol{f}_\mathrm{int}`, from which the published force is formed at
        #: acceptance (see :meth:`internalInertialAndDampingForce`).
        self._U = None
        self._dU = None
        self._V = None
        self._A = None
        self._internalForce = None

    def beginStep(
        self,
        step,
        model: FEModel,
        fieldOutputController: FieldOutputController,
        outputmanagers: dict[str, OutputManagerBase],
    ):
        """Start the step; see
        :meth:`~edelweissfe.solvers.base.nonlinearsolverbase.NonlinearSolverBase.beginStep`. The discrete
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

        # The inertia of every entity is integrated at the nodes here. An entity that can integrate its own by the
        # Newmark scheme carries none: a step of an implicit solver may have left it its parameters.
        model.setNewmarkParametersOfEntities(0.0, 0.0)

        # The timing table's untimed residue is measured against this wall clock, so timers and
        # clock must start together, before any timed work of this step.
        performancetiming.reset()
        self._stepWallClockTic = perf_counter()

        self._externalWork = 0.0
        self._energyAtStepStart = 0.0
        self.prevTimeStep = None
        # The critical time steps and the classification of the fields are derived anew for every step: a change of a
        # material property takes effect at the next step.
        self._meshOfTheClassification = None
        self._criticalTimeStepOfTheElements.reset()
        self._criticalTimeStepOfTheEntities.reset()
        self._criticalTimeStepWithThePenaltyConstraints.reset()
        self._conservationCheck.reset()
        self._warnedAboutMissingInternalEnergy = False
        self._warnedAboutMissingExternalWork = False

        # Whether a constraint couples other nodes as the bodies move, e.g. contact, which the periodic update is for.
        self._modelConnectivityChangesWithMotion = any(
            constraint.connectivityChangesWithMotion for constraint in model.constraints.values()
        )

        # Modifiers that can still act once the analysis is running.
        self._liveTopologyModifiers = [
            modifier
            for modifier in model.modelModifiers.values()
            if modifier.initiatesTopologyChanges and not modifier.actsOnlyAtSimulationStart
        ]

        # Step actions before the discrete system, matching NIST: nothing they do depends on it.
        self.applyStepActionsAtStepStart(model, step)

        # The multi-point constraints are reconciled with the step's Dirichlet boundary conditions whenever the
        # discrete system gets a new numbering.
        self._stepActions = step.actions

        # The discrete system is built by the first prepareIncrement; a resumed step finds the net force of its
        # checkpoint there, as set by setRestartData.
        self.forgetDiscreteSystem()

    def prepareIncrement(self, step, model: FEModel, isRetry: bool):
        """The topology update, when due, and the discrete system; see
        :meth:`~edelweissfe.solvers.base.nonlinearsolverbase.NonlinearSolverBase.prepareIncrement`.

        The topology update is due at the start of the step, and after every
        ``topology-check-frequency``-th increment. At the start of the step it runs before anything
        sized by the discrete system exists, so the mesh is final before the lumped mass, the
        multi-point-constraint condensation and the critical time step are derived from it. It runs
        here, before the next increment is proposed, because a refinement may lower the critical time
        step, and with it the time increment that increment has to use.

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

        if self.system is None:
            # The net force, and the inertia and the damping of the entities that carry their own inertia, of the
            # checkpoint on a resumed step, None on a cold one.
            resumedNetForce, resumedInertiaOfEntities, resumedDampingOfEntities, numberingOfResumedNetForce = (
                self._netForce,
                self._inertiaOfEntities,
                self._dampingOfEntities,
                self._numberingOfNetForce,
            )
            if resumedInertiaOfEntities is not None:
                # A model with entities that carry their own inertia: the checkpoint does not hold their connectivity,
                # which follows from their positions; the resumed step finds it, as every increment does.
                model.updateConnectivity()
            self.updateDiscreteSystem(model, Change.NONE)
            self.journal.printSeperationLine()
            if resumedNetForce is not None:
                self.checkNumberingOfRestartedVectors(numberingOfResumedNetForce)
                self._netForce[:] = resumedNetForce
                if resumedInertiaOfEntities is not None:
                    self._takeLumpedOperatorsOfEntitiesWithOwnInertia(
                        resumedInertiaOfEntities, resumedDampingOfEntities
                    )
            elif not self.system.entitiesWithOwnInertia:
                # With entities that carry their own inertia, formed in the zero increment, once their inertia and
                # damping are evaluated (see _startTheMotionOfEntitiesWithOwnInertia).
                self.formNetForceOfTheModelsState(step.actions["dirichlet"].values())
                self._energyAtStepStart = self.energyOfTheModelsState()
            if not self.system.entitiesWithOwnInertia:
                # The penalty constraints at the state the step starts from, which the net force of the first increment
                # is of on a cold step, e.g. a contact closed at its start; a resumed step keeps its checkpointed time
                # increment, and its first increment estimates them anew. With entities that carry their own inertia,
                # in the zero increment, with their net force.
                self._estimateThePenaltyConstraintsAtTheModelsState(step, model)
            # The time stepper carries (and checkpoints) the time increment; a resumed step keeps its own.
            if step.timeStepper.enforcedTimeIncrement is None:
                self._enforceTheCriticalTimeStep(step, "The critical time step", 1)

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

        lumpedTotalsBefore = self._perFieldLumpedTotals()
        momentumBefore = self.secondOrderMomentum(self._rawLumpedMass, self._V, model)
        kineticBefore = self.kineticEnergy(self._V)

        # A change of the connectivity only moves the locations: the solution, the velocity and the net force carry
        # over. A change of the mesh gives a new numbering, whose solution and velocity are read from the model, and
        # whose net force starts at zero (see onNewNumbering).
        change = self.updateTopologyAndConnectivity(model, step)
        self.updateDiscreteSystem(model, change)

        if change == Change.MESH:
            self.reportTopologyChangeConservation(lumpedTotalsBefore, momentumBefore, kineticBefore, self._V, model)

            # Only a model without entities that carry their own inertia gets here (live modifiers are refused with
            # them, see onNewNumbering), so the entities' critical time step never enters it. The penalty constraints
            # are estimated at the state on the new mesh, as at the start of a step, so that the first increment on it
            # is within the critical time step of their springs on the new masses, not on the old ones.
            self._estimateThePenaltyConstraintsAtTheModelsState(step, model)
            self._enforceTheCriticalTimeStep(step, "Refinement", 1)

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

        if self.system.entitiesWithOwnInertia and self._inertiaOfEntities is None and timeStep.timeIncrement:
            raise ValueError(
                "The first increment of a step must be the zero increment for {:} in a model with entities that carry "
                "their own inertia (particles): in it, they evaluate the force and the mass of the configuration the "
                "step starts from, before anything moves.".format(self.identification)
            )

        # A model whose entities carry their own inertia is reconnected after every increment (see acceptIncrement),
        # with every connectivity search of its constraints: the periodic search would only repeat it.
        if (
            self._modelConnectivityChangesWithMotion
            and contactUpdateFrequency
            and timeStep.number > 0
            and timeStep.number % contactUpdateFrequency == 0
            and not topologyCheckDueThisIncrement
            and not self.system.entitiesWithOwnInertia
        ):
            # Throttled: a node-to-surface search is O(slaves x facets) in Python, which is affordable once per
            # increment of an implicit analysis -- amortised over a Newton loop and a linear solve -- and not tens
            # of thousands of times. An explicit time step is tiny: between two searches a node moves
            # ``V * dT * frequency``, orders of magnitude below a facet dimension.
            with performancetiming.timeit("constraint connectivity"):
                change = model.updateConnectivity()

            # Only the locations can change here: the search moves no node, and a constraint of this solver has no
            # scalar variable.
            if change > Change.LOCATIONS:
                raise RuntimeError(
                    "The connectivity search changed the numbering of the degrees of freedom ({:}). Only a change of "
                    "the locations is expected here, and that cannot add or remove degrees of freedom -- so the "
                    "solution, the velocity and the net force cannot be carried across safely.".format(change.name)
                )
            self.updateDiscreteSystem(model, change)

        U, dU, V, netForce, internalForce = self._U, self._dU, self._V, self._netForce, self._internalForce

        dU[:] = 0.0
        try:
            U, V, netForce = self.solveIncrement(
                U,
                dU,
                V,
                netForce,
                internalForce,
                self._Minv,
                step.actions,
                model,
                timeStep,
                self.prevTimeStep,
            )

        except CutbackRequest as e:
            # The explicit time step is dictated by stability, not convergence, so a cutback is refused:
            # a reduced increment would also stay enforced for the rest of the step, or, with follow-critical-time-step,
            # until the critical time step is taken next.
            raise StepFailed(
                "A material requested a cutback in increment {:}: {:}. The explicit time "
                "step is set by stability, not by convergence, so it cannot be reduced in "
                "response -- either the material cannot integrate at the time increment stability allows, or "
                "the state reaching it is already wrong. Both need the material or the "
                "model looked at, not a smaller step.".format(timeStep.number, e)
            ) from e

        self._U, self._V, self._netForce = U, V, netForce

        if timeStep.timeIncrement > 0.0 and not self.system.entitiesWithOwnInertia:
            # The penalty constraints with the stiffness they had in the net force just evaluated, which the next
            # increment integrates; with entities that carry their own inertia, the net force and its estimate follow
            # at acceptance, on the connectivity they reach.
            self._estimateThePenaltyConstraints(self._rowSumsOfThePenaltyConstraints(U, self._dU, timeStep))
            self._enforceTheCriticalTimeStep(step, "The penalty constraints", 2)

    def acceptIncrement(self, step, model: FEModel, timeStep: TimeStep):
        """Commit the increment to the model; see
        :meth:`~edelweissfe.solvers.base.nonlinearsolverbase.NonlinearSolverBase.acceptIncrement`. In a model with
        entities that carry their own inertia, the model is reconnected, and the net force evaluated on the new
        connectivity (see :meth:`_acceptReconnectAndEvaluate`).

        Parameters
        ----------
        step
            The step being solved.
        model
            The model tree.
        timeStep
            The increment.
        """

        U, dU, V = self._U, self._dU, self._V

        if timeStep.timeIncrement > 0.0:
            # The zero increment is not recorded, so the first real increment is the leapfrog's half step.
            self.prevTimeStep = timeStep

        if self.system.entitiesWithOwnInertia:
            self._acceptReconnectAndEvaluate(step, model, timeStep)
            return

        P, A = self._forceAndAccelerationToPublish(step, timeStep)
        self.writeResultsToModel(U, dU, P, V, A)
        self._acceptTheStateOfTheModel(model, timeStep)

    def _acceptTheStateOfTheModel(self, model: FEModel, timeStep: TimeStep):
        """Move the rigid bodies with the increment and let the model accept the state of its entities at the end of
        it (:meth:`~edelweissfe.models.femodel.FEModel.advanceToTime`): what both kinds of model do once the increment
        has reached the model, a finite element model last, a model with entities that carry their own inertia before
        it reconnects.

        Parameters
        ----------
        model
            The model tree.
        timeStep
            The increment.
        """

        self.updateRigidBodies(model, timeStep)

        # Timed: a serial Python loop over all elements and constraints, run every increment here.
        with performancetiming.timeit("accept state"):
            model.advanceToTime(timeStep.totalTime)

    def _forceAndAccelerationToPublish(self, step, timeStep: TimeStep) -> tuple[DofVector | None, DofVector | None]:
        """The force ``P`` and the acceleration ``A`` the model receives for the increment just completed (see
        :meth:`internalInertialAndDampingForce` and :meth:`accelerationAtTheEndOfTheIncrement`). The zero increment
        integrates nothing and publishes neither: the model keeps the force and the acceleration the last accepted
        increment published, of this step or of the step before.

        Parameters
        ----------
        step
            The step being solved.
        timeStep
            The increment.

        Returns
        -------
        tuple[DofVector | None, DofVector | None]
            The force and the acceleration; None for both in the zero increment.
        """

        if timeStep.timeIncrement == 0.0:
            return None, None

        dirichlets = step.actions["dirichlet"].values()
        return self.internalInertialAndDampingForce(dirichlets), self.accelerationAtTheEndOfTheIncrement(
            dirichlets, timeStep.timeIncrement
        )

    def _acceptReconnectAndEvaluate(self, step, model: FEModel, timeStep: TimeStep):
        """Accept an increment of a model with entities that carry their own inertia (see the module documentation,
        "Entities that carry their own inertia"): the increment, and the velocity it was taken with, reach the model on
        the connectivity of its start; the model accepts the state and reconnects the entities
        (:meth:`~edelweissfe.models.femodel.FEModel.updateConnectivity`), and the discrete system follows. On the new
        connectivity, the net force the next increment starts from is evaluated
        (:meth:`_evaluateTheNetForceOnTheNewConnectivity`), and the force, the velocity and the acceleration reach the
        model.

        Parameters
        ----------
        step
            The step being solved.
        model
            The model tree.
        timeStep
            The increment.
        """

        self.writeResultsToModel(self._U, self._dU, None, self._V, None)
        self._acceptTheStateOfTheModel(model, timeStep)

        numberingOfTheIncrement, internalForceOfTheElements = self.system.numbering, self._internalForce
        with performancetiming.timeit("reconnect"):
            change = model.updateConnectivity()
        self.updateDiscreteSystem(model, change)
        if self.system.numbering is not numberingOfTheIncrement:
            # The elements were computed on the numbering of the increment. Their nodes keep their degrees of freedom
            # in the new one, only with other indices: their force is carried over element by element.
            for element in self.system.elementsWithNodalInertia:
                self._internalForce[element] = internalForceOfTheElements[element]

        self._evaluateTheNetForceOnTheNewConnectivity(step, timeStep)

        P, A = self._forceAndAccelerationToPublish(step, timeStep)
        self.writeResultsToModel(self._U, None, P, self._V, A)

    @performancetiming.timeit("entities with own inertia")
    def _evaluateTheNetForceOnTheNewConnectivity(self, step, timeStep: TimeStep):
        """Evaluate, at the configuration just accepted and on the connectivity the entities that carry their own
        inertia reach now, what the next increment starts from: their internal force, their lumped momentum and their
        lumped inertia and damping (see :meth:`_takeLumpedOperatorsOfEntitiesWithOwnInertia`), their critical time step
        when its estimate is due
        (:meth:`~edelweissfe.solvers.base.criticaltimestep.CriticalTimeStepOfEntitiesWithOwnInertia.isDue`), and, with the internal force of the
        elements of the increment, the loads and the constraints, the net force (:meth:`assembleNetForce`). The configuration is
        accepted: no increment is left, so the loads and the constraints are evaluated with a zero increment, and the
        nodes of the entities carry no total solution.

        In the zero increment, the motion of the entities starts (:meth:`_startTheMotionOfEntitiesWithOwnInertia`);
        with ``project-velocity-every-increment``, the velocity of their nodes is projected from their momentum in every
        increment (:meth:`_projectTheVelocityOfEntitiesFromTheirMomentum`, the second-order fields only). A node of
        theirs without a positive mass has no equation of motion, so it has no velocity: not even the one it carried
        while it still had a mass, which the update would otherwise keep.

        Parameters
        ----------
        step
            The step being solved.
        timeStep
            The increment just accepted.
        """

        system = self.system
        dirichlets = step.actions["dirichlet"].values()
        self.locateConstrainedDofs(dirichlets)

        internalForceOfEntities, momentum = system.assembleInternalForceAndMomentumOfEntitiesWithOwnInertia(
            self.forEachChunk
        )
        self._takeLumpedOperatorsOfEntitiesWithOwnInertia(
            system.assembleLumpedMassOfEntitiesWithOwnInertia(self.forEachChunk),
            system.assembleLumpedDampingOfEntitiesWithOwnInertia(self.forEachChunk),
        )
        self._internalForce += internalForceOfEntities
        noIncrement = system.locations.constructDofVector()
        self._netForce = self.assembleNetForce(
            self._netForce, self._internalForce, self.collectLoads(step.actions), self._U, noIncrement, timeStep
        )

        if timeStep.timeIncrement == 0.0:
            self._startTheMotionOfEntitiesWithOwnInertia(momentum, dirichlets)
        elif self.options["project-velocity-every-increment"]:
            self._projectTheVelocityOfEntitiesFromTheirMomentum(momentum)

        ids = self.ids_inertiaOfEntities
        self._V[ids[self._Minv[ids] == 0.0]] = 0.0

        # The critical time steps, after the net force and the velocity. For the entities the order is not required:
        # their estimate needs only their lumped inertia and damping, just assembled, and nothing in the net force
        # depends on it. The penalty constraints are bounded with the stiffness they had in the net force, with the same
        # arguments, after the entities, since their estimate adds to the entities' last one. The time increment is set
        # once, from both: what they decide is the time increment of the next increment, which integrates that force.
        rowSumsOfThePenaltyConstraints = self._rowSumsOfThePenaltyConstraints(self._U, noIncrement, timeStep)
        criticalTimeStep = self._criticalTimeStepOfTheEntities
        entitiesAreDue = criticalTimeStep.isDue(
            system,
            timeStep.number,
            self._inertiaOfEntities,
            self._dampingOfEntities,
            self._timeIncrementInUse(step) / self.options["courant-number"],
            rowSumsOfThePenaltyConstraints,
        )
        if entitiesAreDue:
            with performancetiming.timeit("critical time step of entities"):
                criticalTimeStep.estimate(
                    system, self._inertiaOfEntities, self._dampingOfEntities, self.forEachChunk, timeStep.number
                )
        self._estimateThePenaltyConstraints(rowSumsOfThePenaltyConstraints)
        self._enforceTheCriticalTimeStep(
            step,
            (
                "The entities that carry their own inertia and the penalty constraints"
                if entitiesAreDue
                else "The penalty constraints"
            ),
            2,
        )

        if timeStep.timeIncrement > 0.0 and timeStep.number % self.options["output-frequency"] == 0:
            self.reportEnergyBalance(self._V, timeStep)

    def _startTheMotionOfEntitiesWithOwnInertia(self, momentum: DofVector, dirichlets):
        """Start the motion of the entities that carry their own inertia, in the zero increment, once their mass is
        evaluated (see the module documentation, "Entities that carry their own inertia"). The net force of the
        model's state (:meth:`formNetForceOfTheModelsState`) is formed again, now with their mass: the nodes of finite
        elements get the one they got when the discrete system was built. Where the model holds the motion of the
        entities' nodes (:meth:`~edelweissfe.models.discretesystem.DiscreteSystem.carriersOfTheMotion`), e.g. after a
        step of this solver, the step continues it from their velocity and acceleration, as at the nodes of finite
        elements. Otherwise, the force just evaluated is taken, and their velocity is projected from the entities'
        momentum, :math:`\boldsymbol{v} = \boldsymbol{M}^{-1} \boldsymbol{p}`. The energy at the start of the step
        follows, with their kinetic energy.

        Parameters
        ----------
        momentum
            The lumped momentum of the entities.
        dirichlets
            The Dirichlet boundary conditions of the step.
        """

        ids = self.ids_inertiaOfEntities
        evaluatedNetForce = np.array(self._netForce[ids])

        self.formNetForceOfTheModelsState(dirichlets)

        if "nodes" not in self.system.carriersOfTheMotion():
            self._netForce[ids] = evaluatedNetForce
            self._projectTheVelocityOfEntitiesFromTheirMomentum(momentum)

        self._energyAtStepStart = self.energyOfTheModelsState()

    def _projectTheVelocityOfEntitiesFromTheirMomentum(self, momentum: DofVector):
        """Project the velocity of the nodes of the entities that carry their own inertia from the entities' lumped
        momentum, :math:`\\boldsymbol{v} = \\boldsymbol{M}^{-1} \\boldsymbol{p}`, at the degrees of freedom of the
        second-order fields. A first-order field has no momentum: its rate is the one of the forward Euler update,
        :math:`\\boldsymbol{C}^{-1} \\boldsymbol{f}_\\mathrm{net}`, and is left as it is.

        Parameters
        ----------
        momentum
            The lumped momentum of the entities.
        """

        ids = self.ids_inertiaOfEntities
        secondOrder = ids[self._secondOrderMask[ids] != 0.0]
        self._V[secondOrder] = momentum[secondOrder] * self._Minv[secondOrder]

    def _takeLumpedOperatorsOfEntitiesWithOwnInertia(
        self, inertiaOfEntities: np.ndarray, dampingOfEntities: np.ndarray
    ):
        """Take the lumped inertia and the lumped damping of the entities that carry their own inertia at their nodes
        (:attr:`ids_inertiaOfEntities`), and derive the operators of the explicit update with them
        (:meth:`_deriveLumpedOperators`); in the first evaluation of a step, this classifies the fields, those only the
        entities carry included. They are those of the entities at their current positions, so they are taken in every
        increment, and from the checkpoint by a resumed step; the elements' are assembled per numbering.

        Parameters
        ----------
        inertiaOfEntities
            The lumped inertia of the entities, one value per degree of freedom
            (:meth:`~edelweissfe.models.discretesystem.DiscreteSystem.assembleLumpedMassOfEntitiesWithOwnInertia`).
        dampingOfEntities
            Their lumped damping, likewise
            (:meth:`~edelweissfe.models.discretesystem.DiscreteSystem.assembleLumpedDampingOfEntitiesWithOwnInertia`).

        Raises
        ------
        NotImplementedError
            If an entity lumps an inertia or a damping to a node of a finite element, which carries its total solution
            and the elements' operators.
        """

        ids = self.ids_inertiaOfEntities
        for operator in (np.asarray(inertiaOfEntities), np.asarray(dampingOfEntities)):
            if np.count_nonzero(operator[ids]) != np.count_nonzero(operator):
                raise NotImplementedError(
                    "An entity that carries its own inertia lumps its inertia or damping to a node of a finite "
                    "element. {:} integrates the nodes of finite elements, which carry their total solution and the "
                    "elements' operators, apart from the nodes of these entities, which carry the coefficients of one "
                    "increment: a node cannot be both.".format(self.identification)
                )

        self._inertiaOfEntities = inertiaOfEntities
        self._dampingOfEntities = dampingOfEntities
        self._deriveLumpedOperators()

    def _timeIncrementInUse(self, step) -> float:
        """The time increment the step takes next: the one enforced last, or else the step's own (``startInc``, at most
        ``maxInc``).

        Parameters
        ----------
        step
            The step being solved.

        Returns
        -------
        float
            The time increment.
        """

        timeStepper = step.timeStepper
        enforced = timeStepper.enforcedTimeIncrement
        return enforced if enforced is not None else timeStepper.increment * step.length

    def _rowSumsOfThePenaltyConstraints(self, U: DofVector, dU: DofVector, timeStep: TimeStep) -> np.ndarray | None:
        """The row sums of the absolute value of the penalty constraints' stiffness, scaled by the lumped mass on both
        sides, :math:`r_i = \\sum_j |K^\\mathrm{c}_{ij}| / \\sqrt{M_i M_j}`: what the constraints add to the squared
        frequency of every degree of freedom they couple (see :doc:`/theory/criticaltimestep`, "Penalty constraints").
        Every constraint of the discrete system bounds its own, at the state given, with the column weights
        :math:`1/\\sqrt{M_j}`
        (:meth:`~edelweissfe.constraints.base.constraintbase.ConstraintBase.addBoundOfAbsoluteStiffnessTimes`); the
        rows are then scaled by :math:`1/\\sqrt{M_i}`.

        The masses are those the increment divides by, on the degrees of freedom integrated by central differences; a
        degree of freedom without one, or with a prescribed motion, has no equation of motion and neither oscillates
        nor lets a spring oscillate it: its weight and its row are zero. A multi-point-constraint slave moves with its
        masters: its weight is the sum of theirs, by the absolute value of its coefficients, and its row is folded onto
        them likewise.

        Parameters
        ----------
        U
            The solution the constraints were evaluated with in the net force.
        dU
            The increment they were evaluated with.
        timeStep
            The time step they were evaluated with: the increment, or one of zero length at the state of the model
            (:meth:`_zeroLengthTimeStepAtTheModelsState`).

        Returns
        -------
        numpy.ndarray | None
            The row sums, one per degree of freedom; None if the discrete system has no constraints.
        """

        constraints = self.system.constraints
        if not constraints:
            return None

        scaling = np.sqrt(self._Minv) * self._secondOrderMask
        for dirichlet in self._stepActions["dirichlet"].values():
            scaling[dirichlet.constrainedDofIndices] = 0.0

        weights = scaling
        if self.mpcTransformation is not None:
            weights = self.mpcTransformation.slaveWeightsOfAbsoluteCoefficients(scaling)

        rowSums = np.zeros_like(scaling)
        with performancetiming.timeit("critical time step of penalty constraints"):
            for constraint in constraints:
                _, indices, _ = self._explicitEvaluationPlanOf(constraint)
                bound = np.zeros(constraint.nDof)
                constraint.addBoundOfAbsoluteStiffnessTimes(
                    U[constraint], dU[constraint], weights[indices], bound, timeStep
                )
                np.add.at(rowSums, indices, bound)

        if self.mpcTransformation is not None:
            rowSums = self.mpcTransformation.foldRowSumsOfAbsoluteCoefficients(rowSums)

        return scaling * rowSums

    def _zeroLengthTimeStepAtTheModelsState(self, model: FEModel) -> TimeStep:
        """A time step of zero length at the state the model holds, for what is evaluated there with no increment: the
        end of the last increment accepted in the step, or the start of the step if there is none.

        Parameters
        ----------
        model
            The model tree.

        Returns
        -------
        TimeStep
            The time step.
        """

        last = self.prevTimeStep
        if last is None:
            return TimeStep(0, 0.0, 0.0, 0.0, 0.0, model.time)
        return TimeStep(last.number, 0.0, last.stepProgress, 0.0, last.stepTime, last.totalTime)

    def _estimateThePenaltyConstraintsAtTheModelsState(self, step, model: FEModel):
        """Estimate the penalty constraints at the state the model holds, before an increment has evaluated them on the
        discrete system: at the start of a step, and after a refinement. They are evaluated there with no increment and
        with a time step of zero length (:meth:`_zeroLengthTimeStepAtTheModelsState`).

        Parameters
        ----------
        step
            The step being solved.
        model
            The model tree.
        """

        self.locateConstrainedDofs(step.actions["dirichlet"].values())
        self._estimateThePenaltyConstraints(
            self._rowSumsOfThePenaltyConstraints(
                self._U, self.system.locations.constructDofVector(), self._zeroLengthTimeStepAtTheModelsState(model)
            )
        )

    def _estimateThePenaltyConstraints(self, rowSumsOfThePenaltyConstraints: np.ndarray | None):
        """Estimate the critical time step of the parts with the penalty constraints, from the last estimates of the
        parts and the row sums of the constraints
        (:class:`~edelweissfe.solvers.base.criticaltimestep.CriticalTimeStepWithThePenaltyConstraints`); forget the
        last one if the discrete system has no constraints (any more), so that it does not hold the time increment down.

        Parameters
        ----------
        rowSumsOfThePenaltyConstraints
            The row sums of the constraints (:meth:`_rowSumsOfThePenaltyConstraints`); None without constraints.
        """

        if rowSumsOfThePenaltyConstraints is None:
            self._criticalTimeStepWithThePenaltyConstraints.forget()
            return
        self._criticalTimeStepWithThePenaltyConstraints.estimate(
            self.system,
            rowSumsOfThePenaltyConstraints,
            self._criticalTimeStepOfTheElements,
            self._criticalTimeStepOfTheEntities,
            self.mpcTransformation,
        )

    def _criticalTimeStepOfTheModel(self) -> float:
        """The critical time step of the model as last estimated: the smallest of the elements', the entities' and the
        one of the parts with the penalty constraints, which is not the constraints' own (see
        :class:`~edelweissfe.solvers.base.criticaltimestep.CriticalTimeStepWithThePenaltyConstraints` for why the
        minimum with the parts is kept).

        Returns
        -------
        float
            The critical time step, not scaled by the courant number; infinite if no part has one.
        """

        return min(
            self._criticalTimeStepOfTheElements.value,
            self._criticalTimeStepOfTheEntities.value,
            self._criticalTimeStepWithThePenaltyConstraints.value,
        )

    @property
    def _largestTimeIncrement(self) -> float:
        """The largest time increment the critical time step of the model allows: ``courant-number`` times it
        (:meth:`_criticalTimeStepOfTheModel`); infinite if no part has one."""

        return self.options["courant-number"] * self._criticalTimeStepOfTheModel()

    def _enforceTheCriticalTimeStep(self, step, cause: str, journalLevel: int):
        """Set the time increment of the step from the critical time step of the model: the largest time increment it
        allows (:attr:`_largestTimeIncrement`). Called each time a part is estimated. By default the increment is
        lowered only, to the largest one if it is below the increment the step would take now
        (:meth:`_timeIncrementInUse`); with ``follow-critical-time-step`` it becomes the largest one, at most the
        step's ``maxInc``. As long as no part has an estimate (the critical time step is infinite, e.g. at the start of
        a step of entities alone), the increment stays as it is, whether it follows or not. See
        :ref:`ned-time-increment-lowered-or-followed`.

        Parameters
        ----------
        step
            The step being solved.
        cause
            What changed the critical time step, for the journal.
        journalLevel
            The level of the journal message when an enforced increment changes.

        Raises
        ------
        ValueError
            If the critical time step of a part of the model is not a number.
        """

        # Python's min() keeps its first argument when the comparison with NaN is false, so each part is checked;
        # infinity is legal: a part without an estimate.
        parts = (
            self._criticalTimeStepOfTheElements,
            self._criticalTimeStepOfTheEntities,
            self._criticalTimeStepWithThePenaltyConstraints,
        )
        notANumber = [part for part in parts if np.isnan(part.value)]
        if notANumber:
            raise ValueError(
                "{:}: the critical time step is not a number ({:}): the stiffness or the mass of the model is not "
                "finite.".format(cause, ", ".join("of {:} {:e}".format(part.part, part.value) for part in notANumber))
            )
        largestTimeIncrement = self._largestTimeIncrement
        # No estimate yet: nothing to follow. Lowered only, this changes nothing anyway (no increment is above an
        # infinite one); followed, it keeps the step's own increment instead of jumping to maxInc.
        if np.isinf(largestTimeIncrement):
            return
        timeStepper = step.timeStepper
        enforced = timeStepper.enforcedTimeIncrement
        if self.options["follow-critical-time-step"]:
            timeIncrement = min(largestTimeIncrement, timeStepper.maxIncrement * step.length)
        else:
            inUse = self._timeIncrementInUse(step)
            timeIncrement = largestTimeIncrement if largestTimeIncrement < inUse else enforced
        if timeIncrement is None or timeIncrement == enforced:
            return
        if enforced is not None:
            self.journal.message(
                "{:} {:} the time increment from {:e} to {:e}".format(
                    cause, "lowered" if timeIncrement < enforced else "raised", enforced, timeIncrement
                ),
                self.identification,
                journalLevel,
            )
        timeStepper.enforceTimeIncrement(timeIncrement)

    def endStep(self, step, model: FEModel):
        """Leave the model's state at the end time of the step, and report the step's performance timing.

        The velocity the model holds lags half of the last increment behind the solution; it is advanced by that half
        increment with the acceleration (:meth:`~edelweissfe.models.femodel.FEModel.advanceVelocity`), so that the
        next step starts from the velocity at its start time (see the module documentation, "The velocity and the
        acceleration it publishes"). A step resumed at its very end does so as well: its checkpoint holds the
        staggered velocity and the last increment. This solver retries no increment, so its step never falls back to
        a checkpoint, which ends the abandoned attempt here too.

        Parameters
        ----------
        step
            The step that was solved.
        model
            The model tree.
        """

        if self.prevTimeStep is not None:
            model.advanceVelocity(0.5 * self.prevTimeStep.timeIncrement)

        for criticalTimeStep in (
            self._criticalTimeStepOfTheElements,
            self._criticalTimeStepOfTheEntities,
            self._criticalTimeStepWithThePenaltyConstraints,
        ):
            if criticalTimeStep.numberOfEstimates:
                self.journal.message(
                    "The critical time step of {:} was estimated {:} times in the step, last {:e}".format(
                        criticalTimeStep.part, criticalTimeStep.numberOfEstimates, criticalTimeStep.value
                    ),
                    self.identification,
                    1,
                )

        prettyTable = performancetiming.makePrettyTable(wallTime=perf_counter() - self._stepWallClockTic)
        self.journal.printPrettyTable(prettyTable, self.identification)
        performancetiming.reset()

    def accelerationAtTheEndOfTheIncrement(self, dirichlets, timeIncrement: float) -> DofVector:
        """The acceleration this solver publishes as the node field entry ``A`` of the increment just completed, the
        one at its end (see the module documentation, "The velocity and the acceleration it publishes"): with the
        velocity at that time, half an increment after the staggered one, it satisfies the equation of motion,
        :math:`\\left( \\boldsymbol{M} + \\tfrac{1}{2} \\Delta t \\, \\boldsymbol{C} \\right) \\boldsymbol{a} =
        \\boldsymbol{f}_\\mathrm{net} - \\boldsymbol{C} \\boldsymbol{v}_{n+1/2}`.

        On a degree of freedom with a prescribed motion, whose acceleration the next increment's prescribed velocity
        sets, it is zero. A multi-point-constraint slave, which integrates no equation of motion of its own, takes it
        from its masters.

        Parameters
        ----------
        dirichlets
            The Dirichlet boundary conditions of the step, located on the current discrete system.
        timeIncrement
            The time increment just completed.

        Returns
        -------
        DofVector
            The acceleration, the system's vector.
        """

        A = self._A
        A[:] = 0.0
        coefficient = self._inertia + (0.5 * timeIncrement) * self._damping
        hasEquationOfMotion = coefficient > 0.0
        forceNotSpentOnDamping = np.asarray(self._netForce) - self._damping * np.asarray(self._V)
        A[hasEquationOfMotion] = forceNotSpentOnDamping[hasEquationOfMotion] / coefficient[hasEquationOfMotion]
        for dirichlet in dirichlets:
            A[dirichlet.constrainedDofIndices] = 0.0
        if self.mpcTransformation is not None:
            self.mpcTransformation.applySlaveKinematics(A)

        return A

    def formNetForceOfTheModelsState(self, dirichlets):
        """Form the net force a cold step starts from: that of the state the model holds,
        :math:`\\boldsymbol{f}_\\mathrm{net} = \\boldsymbol{M} \\boldsymbol{a}_0 + \\boldsymbol{C} \\boldsymbol{v}_0`,
        from its velocity and acceleration, read with the solution when the discrete system was built, and the lumped
        operators; nothing is evaluated (see the module documentation, "The net force is solver state, and a step
        starts from the model's state"). Zero on a degree of freedom with a prescribed motion, which has no equation of
        motion, and on a multi-point-constraint slave, whose inertia and damping are folded onto its masters.

        Parameters
        ----------
        dirichlets
            The Dirichlet boundary conditions of the step.
        """

        netForce = self._netForce
        netForce[:] = self._inertia * np.asarray(self._A) + self._damping * np.asarray(self._V)
        self.locateConstrainedDofs(dirichlets)
        for dirichlet in dirichlets:
            netForce[dirichlet.constrainedDofIndices] = 0.0

    def energyOfTheModelsState(self) -> float:
        """The energy the model holds at the start of a cold step, which its energy balance starts from (see the module
        documentation, "Energy balance"): the kinetic energy of the model's velocity, read with the solution when the
        discrete system was built, and the internal (strain) energy the materials publish for the state the elements
        hold. Nothing is evaluated.

        Returns
        -------
        float
            The kinetic and the internal energy.
        """

        internalEnergy = sum(element.computeInternalEnergy() for element in self.system.elementsWithNodalInertia)

        return self.kineticEnergy(self._V) + internalEnergy

    def kineticEnergy(self, V: DofVector) -> float:
        """The kinetic energy :math:`\\tfrac{1}{2} \\sum m v^2` of a velocity, with the unfolded lumped mass, over the
        degrees of freedom where it is an energy (:attr:`ids_mechanicalEnergy`): not on a first-order field, whose
        coefficient is a viscosity, nor on a field whose inertia is not mechanical.

        Parameters
        ----------
        V
            The velocity vector.

        Returns
        -------
        float
            The kinetic energy.
        """

        ids = self.ids_mechanicalEnergy

        return 0.5 * float(np.sum(self._rawLumpedMass[ids] * V[ids] ** 2))

    def internalInertialAndDampingForce(self, dirichlets) -> DofVector:
        """The force this solver publishes as the node field result ``P`` of the increment just completed: the internal
        force with the inertial and damping forces, :math:`\\boldsymbol{f}_\\mathrm{int} + \\boldsymbol{M}
        \\boldsymbol{a} + \\boldsymbol{C} \\boldsymbol{v}`, without the loads and the constraints (see the module
        documentation, "The force it publishes").

        On a degree of freedom with an equation of motion, the inertial and damping forces are the net force, since the
        next velocity update solves exactly :math:`\\boldsymbol{M} \\boldsymbol{a} + \\boldsymbol{C}
        \\boldsymbol{v} = \\boldsymbol{f}_\\mathrm{net}` -- so :math:`\\boldsymbol{f}_\\mathrm{int} +
        \\boldsymbol{f}_\\mathrm{net}`, which is the force of the loads and the constraints there. On a degree of
        freedom with a prescribed motion, whose acceleration the next increment's prescribed velocity sets, it is the
        internal force alone. A multi-point-constraint slave has a zero net force (it is folded onto its masters), so
        it gets its internal force as well, and so does a node of an entity that carries its own inertia without a
        positive mass, which has no equation of motion.

        Parameters
        ----------
        dirichlets
            The Dirichlet boundary conditions of the step, located on the current discrete system.

        Returns
        -------
        DofVector
            The internal, inertial and damping force.
        """

        internalForce = self._internalForce
        force = self.system.locations.constructDofVector()
        np.add(internalForce, self._netForce, out=force)
        for dirichlet in dirichlets:
            force[dirichlet.constrainedDofIndices] = internalForce[dirichlet.constrainedDofIndices]
        withoutInertia = self.ids_inertiaOfEntities[self._Minv[self.ids_inertiaOfEntities] == 0.0]
        force[withoutInertia] = internalForce[withoutInertia]

        return force

    @performancetiming.timeit("increment")
    def solveIncrement(
        self,
        U_n: DofVector,
        dU: DofVector,
        V: DofVector,
        netForce: DofVector,
        internalForce: DofVector,
        Minv: np.ndarray,
        stepActions: list,
        model: FEModel,
        timeStep: TimeStep,
        prevTimeStep: TimeStep,
    ) -> tuple[DofVector, DofVector, DofVector]:
        """The central-difference update of an increment: the kinematic update, the elements' internal force at the end
        of it, and the entities that carry their own inertia advanced by it. In a finite element model, the net force
        of the increment follows here (:meth:`assembleNetForce`); in a model with entities that carry their own
        inertia, at acceptance, on the connectivity they reach (see :meth:`_acceptReconnectAndEvaluate`).

        Parameters
        ----------
        U_n
            The old solution vector, overwritten with the new one.
        dU
            The solution increment, overwritten.
        V
            The old velocity vector, overwritten with the new one.
        netForce
            The net force of the last increment, which accelerates the nodes in this one, overwritten with the net
            force of this increment.
        internalForce
            Overwritten with the elements' sum of this increment, the internal force.
        Minv
            The inverse lumped mass, one value per degree of freedom.
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
        tuple[DofVector,DofVector,DofVector]
            A tuple containing
                - the new solution vector
                - the new velocity vector
                - the net force of this increment
        """

        elements = self.system.elementsWithNodalInertia
        dirichlets = stepActions["dirichlet"].values()

        # Find which global DOFs the Dirichlet BCs constrain, once up front.
        self.locateConstrainedDofs(dirichlets)

        if timeStep.timeIncrement == 0.0:
            # Nothing moves in the zero increment. An entity that carries its own inertia takes it as an increment of
            # zero length, from which it evaluates the force of its state at acceptance.
            self._advanceEntitiesWithOwnInertia(dU, timeStep)
            return U_n, V, netForce

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
            # there is no free equilibrium there, so their net force is set to zero,
            # and their velocity is prescribed as (prescribed increment) / (time step).
            prescribedVelocities = []
            for dirichlet in dirichlets:
                prescribedIncrement = dirichlet.getPrescribedIncrementOfConstrainedDofs(timeStep)

                # External work = reaction (minus the carried-over net force) times the prescribed
                # increment. Must be accumulated before the net force is zeroed below. Only for fields where this
                # product is an energy (not, e.g., a nonlocal damage field).
                if carriesKineticEnergy(dirichlet.field):
                    self._externalWork -= float(np.dot(netForce[dirichlet.constrainedDofIndices], prescribedIncrement))

                prescribedVelocity = prescribedIncrement / timeStep.timeIncrement

                netForce[dirichlet.constrainedDofIndices] = 0.0
                V[dirichlet.constrainedDofIndices] = prescribedVelocity
                prescribedVelocities.append((dirichlet.constrainedDofIndices, prescribedVelocity))

            # Second-order DOFs: central difference with mass-proportional damping,
            # V = ((1 - h) V + Minv f_net dt) / (1 + h), h = alpha dt / 2, alpha = C/M; at alpha = 0 this is
            # exactly the undamped update. Written on whole vectors: h and the mask are zero off the
            # second-order DOFs. First-order DOFs: V = Minv f_net (forward Euler).
            dtAverage = 0.5 * (timeStep.timeIncrement + prevTimeStep.timeIncrement)
            halfRateStep = self._halfDampingRate * dtAverage
            forceOverMass = Minv * netForce.asPlainArray()

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

        # Every element adds its residual contribution, the internal force; it is kept for the force published at
        # acceptance.
        internalForce[:] = 0.0
        internalForce, self._internalEnergy = self.computeElements(elements, U_n, dU, internalForce, timeStep)

        self._advanceEntitiesWithOwnInertia(dU, timeStep)

        if self.system.entitiesWithOwnInertia:
            return U_n, V, netForce

        netForce = self.assembleNetForce(netForce, internalForce, self.collectLoads(stepActions), U_n, dU, timeStep)

        if timeStep.number % self.options["output-frequency"] == 0:
            self.reportEnergyBalance(V, timeStep)

        return U_n, V, netForce

    def assembleNetForce(
        self,
        netForce: DofVector,
        internalForce: DofVector,
        loads: list,
        U: DofVector,
        dU: DofVector,
        timeStep: TimeStep,
    ) -> DofVector:
        """The net force that accelerates the nodes,
        :math:`\\boldsymbol{f}_\\mathrm{net} = \\boldsymbol{f}_\\mathrm{ext} - \\boldsymbol{f}_\\mathrm{int}`: every
        element, entity, load and constraint adds its residual contribution, and the net force is the negative of their
        sum. The forces of the multi-point-constraint slaves are folded onto
        their masters, so that the next increment's Dirichlet handling sees the folded vector.

        Parameters
        ----------
        netForce
            Overwritten with the net force.
        internalForce
            The residual contributions of the elements and of the entities that carry their own inertia.
        loads
            The loads, see :meth:`~edelweissfe.solvers.base.nonlinearsolverbase.NonlinearSolverBase.collectLoads`.
        U
            The solution the loads and the constraints are evaluated with.
        dU
            The increment they are evaluated with.
        timeStep
            The increment.

        Returns
        -------
        DofVector
            The net force.
        """

        netForce[:] = internalForce
        netForce, _ = self.assembleLoads(loads, U, dU, netForce, None, timeStep)
        netForce = self.assembleConstraintForces(self.system.constraints, U, dU, netForce, timeStep)
        netForce[:] = -netForce[:]

        if self.mpcTransformation is not None:
            with performancetiming.timeit("mpc force fold"):
                netForce[:] = self.mpcTransformation.foldExplicitForce(netForce)

        return netForce

    def reportEnergyBalance(self, V: DofVector, timeStep: TimeStep):
        """Print the energy balance of a reporting increment (see the module documentation, "Energy balance"), and
        stop the step if it is no longer finite.

        Parameters
        ----------
        V
            The velocity.
        timeStep
            The increment.

        Raises
        ------
        StepFailed
            If the balance is no longer a finite number: the solution has diverged.
        """

        Wint = self._internalEnergy

        Wkin = self.kineticEnergy(V)

        Wext = self._externalWork

        # What the step can draw on: the energy the model held at its start, and the work done on it since.
        E0 = self._energyAtStepStart
        suppliedEnergy = E0 + Wext

        # Unpublished strain energy, dissipation and damage; negative means energy is being created.
        unaccounted = suppliedEnergy - Wkin - Wint

        def _share(value):
            return "{:7.2f} %".format(value / suppliedEnergy * 100) if suppliedEnergy > 0.0 else "      -- "

        energyRows = [
            ["energy", "value", "share of E_0 + W_ext"],
            ["at the step start E_0", "{:+.6e}".format(E0), _share(E0)],
            ["external work W_ext", "{:+.6e}".format(Wext), _share(Wext)],
            ["kinetic", "{:+.6e}".format(Wkin), _share(Wkin)],
            ["internal (strain)", "{:+.6e}".format(Wint), _share(Wint)],
            ["unaccounted", "{:+.6e}".format(unaccounted), _share(unaccounted)],
        ]

        # One row per non-mechanical second-order field, kept out of the total: for a hyperbolic
        # non-local field the inertia is a time squared, so 0.5 * m * rate**2 is a volume.
        for fieldName in self.nonMechanicalSecondOrderFields:
            indices = self.system.numbering.idcsOfFieldsInDofVector[fieldName]
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

        # KE <= E_0 + W_ext holds in the continuum; a violation means the time step is above the true
        # stability limit. NaN is checked first, since every comparison against NaN is False.
        if not (np.isfinite(Wext) and np.isfinite(Wkin)):
            raise StepFailed(
                "THE SOLUTION HAS DIVERGED in increment {:}: the energy balance is no longer a "
                "finite number (external work {:e}, kinetic {:e}), which means the state itself "
                "is not. The time step is above the true stability limit -- and dt_crit does not "
                "see all of it: it may see a penalty contact late, or not at all (see the theory page "
                "on the critical time step), and the nonlocal field only where that field carries a "
                "non-mechanical inertia, a first-order one having a forward-Euler limit that nothing "
                "checks. Resume from a checkpoint with a smaller courant-number.".format(timeStep.number, Wext, Wkin)
            )

        # Checked only while the prescribed degrees of freedom do work in this step: the work of the loads is not
        # accumulated, so a step they drive has no bound to check against, whatever energy it started with.
        if Wext > 0.0 and Wkin > suppliedEnergy * (1.0 + _ENERGY_CREATION_TOLERANCE):
            self.journal.message(
                "ENERGY IS BEING CREATED: the kinetic energy {:e} exceeds the energy at the start of "
                "the step plus the external work, {:e}, by {:.1f} %. That is impossible -- the "
                "unreported terms (strain energy, dissipation, damage) can only add to the balance. "
                "The time step is above the true stability limit, which dt_crit does not see in "
                "full: it ignores the nonlocal field entirely and may see a penalty contact late, or "
                "not at all (see the theory page on the critical time step). Reduce courant-number, or "
                "resume from a checkpoint with a smaller one.".format(
                    Wkin, suppliedEnergy, (Wkin / suppliedEnergy - 1.0) * 100
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

    def _advanceEntitiesWithOwnInertia(self, dU: DofVector, timeStep: TimeStep):
        """Advance the entities that carry their own inertia by the increment, on the connectivity of its start; see
        :meth:`~edelweissfe.models.discretesystem.DiscreteSystem.advanceEntitiesWithOwnInertia`.

        Parameters
        ----------
        dU
            The increment.
        timeStep
            The increment of time.
        """

        if self.system.entitiesWithOwnInertia:
            with performancetiming.timeit("entities with own inertia"):
                self.system.advanceEntitiesWithOwnInertia(dU, timeStep, self.forEachChunk)

    @performancetiming.timeit("elements")
    def computeElements(
        self,
        elements: list,
        U_np: DofVector,
        dU: DofVector,
        P: DofVector,
        timeStep: TimeStep,
    ) -> tuple[DofVector, float]:
        """Loop over all elements, and evaluate them: their residual contributions, summed, are the internal force.
        Called by :meth:`solveIncrement` once per increment.

        Parameters
        ----------
        elements
            The finite elements.
        U_np
            The current solution vector.
        dU
            The solution increment vector.
        P
            The vector the elements' residual contributions are added to; zero.
        timeStep
            The time step.

        Returns
        -------
        tuple[DofVector,float]
            - The internal force.
            - The internal (strain) energy the elements publish.
        """

        time = timeStep.totalTime
        dT = timeStep.timeIncrement
        psi = 0.0
        for el in elements:
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
        supported: see
        :meth:`~edelweissfe.solvers.base.nonlinearsolverbase.NonlinearSolverBase.assembleConstraintForces`.
        A constraint switched off from outside
        (:attr:`~edelweissfe.constraints.base.constraintbase.ConstraintBase.switchedOn`) is not refused: it takes no
        part in the step, and its scalar variables leave the discrete system with it. One that is switched on is
        refused even while it is out of reach: it may come within reach during the step.

        A **constraint that does not act in an explicit analysis**
        (:attr:`~edelweissfe.constraints.base.constraintbase.ConstraintBase.actsInExplicitAnalysis`): in a model with
        entities that carry their own inertia, the constraints are evaluated at the configuration just accepted, where
        the nodes of the entities carry neither a solution nor an increment, so a constraint on them that reads these
        would act as if nothing had moved (see the module documentation, "Entities that carry their own inertia").

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
            # a constraint switched off from outside takes no part in the step, nor do its scalar variables
            if not constraint.switchedOn:
                continue
            nScalarVariables = constraint.getNumberOfAdditionalNeededScalarVariables()
            if nScalarVariables:
                raise NotImplementedError(
                    f"Constraint '{constraintName}' introduces {nScalarVariables} additional scalar "
                    f"variable(s), which carry no inertia and which {self.identification} therefore "
                    "has no equation of motion for. Only constraints acting through nodal forces "
                    "(penalty formulations) are supported here."
                )
            if not constraint.actsInExplicitAnalysis:
                raise NotImplementedError(
                    f"Constraint '{constraintName}' ({type(constraint).__name__}) does not act in an explicit "
                    f"analysis: {self.identification} evaluates the constraints on the nodes of particles at the "
                    "configuration just accepted, where these nodes carry neither a solution nor an increment, so a "
                    "constraint that reads them would act as if nothing had moved. A constraint that acts here "
                    "measures its condition at the particles' current positions, see "
                    "ConstraintBase.actsInExplicitAnalysis."
                )

    @performancetiming.timeit("write results to nodes")
    def writeResultsToModel(self, U: DofVector, dU: DofVector, P: DofVector | None, V: DofVector, A: DofVector | None):
        """Write the results of an increment to the model, through the discrete system: the
        solution, its increment and the internal, inertial and damping force to the node fields, and the solution to the
        scalar variables (see :meth:`~edelweissfe.models.discretesystem.DiscreteSystem.writeResultsToModel`),
        then the velocity and the acceleration (see
        :meth:`~edelweissfe.models.discretesystem.DiscreteSystem.writeVelocityAndAccelerationToModel`); the
        incremental degrees of freedom are set to zero first, from where the next increment starts them (see
        :meth:`~edelweissfe.models.discretesystem.DiscreteSystem.zeroIncrementalCoefficients`).

        Written at every accepted increment, not only on output increments, for two reasons: an h-adaptivity
        event can fall on any increment and its interpolation reads these entries, and a restart
        checkpoint written from the node fields is the only way an explicit run can resume with its
        kinetic state intact. It is an O(nDof) copy.

        Parameters
        ----------
        U
            The solution vector.
        dU
            The solution increment vector of the increment.
        P
            The internal, inertial and damping force (see :meth:`internalInertialAndDampingForce`); None to leave the
            model's as it is.
        V
            The velocity vector.
        A
            The acceleration at the end of the increment (see :meth:`accelerationAtTheEndOfTheIncrement`); None to leave
            the model's as it is.
        """

        self.system.zeroIncrementalCoefficients(U)
        self.system.writeResultsToModel(U, dU, P)
        self.system.writeVelocityAndAccelerationToModel(V, A)

    def forgetDiscreteSystem(self) -> None:
        """Drop the discrete system, and everything that belongs to it, before the next one is built: the vectors, the
        lumped operators and the indices of the fields by time derivative; see
        :meth:`~edelweissfe.solvers.base.nonlinearsolverbase.NonlinearSolverBase.forgetDiscreteSystem`. Everything
        :meth:`onNewNumbering` builds is gone, so that no array of the previous numbering is left to be mistaken for
        one of the next. The net force and the mass and the damping of the entities that carry their own inertia, which
        a resumed step finds from its checkpoint, are dropped as well: whoever needs them takes them before the build (see
        :meth:`prepareIncrement`). What is derived from the mesh, the critical time step and the classification of
        the fields by time derivative, is no array of the numbering: it is derived again when the mesh changed (see
        :meth:`onNewNumbering`)."""

        super().forgetDiscreteSystem()
        self._U = self._dU = self._V = self._A = self._netForce = self._internalForce = None
        self._Minv = self._lumpedMass = self._rawLumpedMass = None
        self._dampingRate = self._halfDampingRate = self._secondOrderMask = None
        self._inertia = self._damping = None
        self.ids_1st = self.ids_2nd = self.ids_mechanicalEnergy = self.ids_inertiaOfEntities = None
        self._inertiaOfTheElements = self._dampingOfTheElements = None
        self._inertiaOfEntities = self._dampingOfEntities = None
        self._numberingOfNetForce = None

    def onNewNumbering(self):
        """Build what is sized by the numbering of the new discrete system: the multi-point-constraint condensation,
        the degrees of freedom whose inertia the entities that carry their own inertia lump to them, the lumped
        operators the increment divides by and the classification of the fields by time derivative that follows from
        them (see :meth:`_assembleLumpedOperators`), the vectors of the increments (the solution, the velocity and the
        acceleration read from the model, which holds the state a mesh change interpolated onto its new nodes; the net
        force and the internal force zero, see the module documentation, "The net force is solver state, and a step
        starts from the model's state"), and the critical time step of the new mesh.

        The lumped operators are functions of the elements and of the numbering: new locations on the same numbering
        keep them. What is derived from the mesh -- the critical time step, an element pass that evaluates the
        materials, and the classification of the fields -- is derived for the first numbering of a step and for a new
        mesh, not for a new numbering of the same mesh, e.g. when the particles of a meshfree model reach other nodes.

        See :meth:`~edelweissfe.solvers.base.nonlinearsolverbase.NonlinearSolverBase.onNewNumbering`.

        Raises
        ------
        NotImplementedError
            If the discrete system holds entities this solver does not integrate: material points, cells or cell
            elements of a meshfree model; or entities that carry their own inertia together with model modifiers that
            act after the start of the analysis, or with a multi-point constraint whose slave is one of their degrees
            of freedom (see :meth:`_refuseSlavesAmongTheDofsOfEntitiesWithOwnInertia`).
        """

        system = self.system

        integrated = len(system.elementsWithNodalInertia) + len(system.entitiesWithOwnInertia)
        if system.materialPoints or len(system.elements) != integrated:
            raise NotImplementedError(
                "The model has entities that {:} does not integrate: it integrates finite elements and entities that "
                "carry their own inertia (the particles of a meshfree model), not material points, background cells "
                "or cell elements.".format(self.identification)
            )
        if system.entitiesWithOwnInertia and self._liveTopologyModifiers:
            raise NotImplementedError(
                "The model modifier(s) {:} act after the start of the analysis, which {:} does not support in a model "
                "with entities that carry their own inertia.".format(
                    ", ".join(sorted(modifier.name for modifier in self._liveTopologyModifiers)), self.identification
                )
            )

        self.buildAndCheckMPCTransformation()

        self.ids_inertiaOfEntities = np.setdiff1d(
            self._dofIndicesOfFields(system.numbering.fields), system.dofsWithNodalInertia(system.numbering.fields)
        )
        self._refuseSlavesAmongTheDofsOfEntitiesWithOwnInertia()

        self._assembleLumpedOperators()

        locations = system.locations
        self._U = locations.constructDofVector()
        self._dU = locations.constructDofVector()
        self._V = locations.constructDofVector()
        self._A = locations.constructDofVector()
        self._netForce = locations.constructDofVector()
        self._internalForce = locations.constructDofVector()
        self._numberingOfNetForce = system.numbering.digest

        # After an h-adaptivity event, the solution, the velocity and the acceleration interpolated onto the new
        # nodes are read back here (see hadaptivity.WARM_STARTED_NODE_FIELD_ENTRIES).
        system.readStateFromModel(self._U, self._V, self._A)

        # Estimated only; the callers enforce it (see the module documentation, "The critical time step").
        criticalTimeStep = self._criticalTimeStepOfTheElements
        if criticalTimeStep.isDue(system):
            criticalTimeStep.estimate(system, self._U)
            self.journal.message(
                "The critical time step of the finite elements: {:e}".format(criticalTimeStep.value),
                self.identification,
                1,
            )

    def _refuseSlavesAmongTheDofsOfEntitiesWithOwnInertia(self):
        """Refuse a multi-point constraint whose slave is a degree of freedom of the entities that carry their own
        inertia (:attr:`ids_inertiaOfEntities`). Such a slave has no equation of motion of its own: the inertia the
        entities lump to it would be folded onto its masters, and the velocity this solver publishes for it at every
        accepted increment, and its kinetic energy, would be zero.

        Raises
        ------
        NotImplementedError
            If a multi-point constraint has a slave among the degrees of freedom of the entities; the message names
            the constraint.
        """

        if self.mpcTransformation is None:
            return

        slaves = np.intersect1d(self.mpcTransformation.slaveDofIndices, self.ids_inertiaOfEntities, assume_unique=True)
        if slaves.size == 0:
            return

        numbering = self.system.numbering
        names = [
            name
            for name, constraint in self.system.model.multiPointConstraints.items()
            if np.isin([slave for slave, _ in constraint.getMultiPointConstraints(numbering)], slaves).any()
        ]
        raise NotImplementedError(
            "The multi-point constraint(s) {:} make {:} degree(s) of freedom of the entities that carry their own "
            "inertia (the nodes of the particles of a meshfree model) their slaves, which {:} does not support: the "
            "velocity and the kinetic energy it publishes for such a slave would be zero.".format(
                ", ".join("'{:}'".format(name) for name in names), slaves.size, self.identification
            )
        )

    def onNewLocations(self):
        """Wrap the vectors of the increments with the new locations, their values kept: the increments continue from
        the solution, the velocity, the acceleration and the net force they have. The lumped operators and the critical
        time step are not touched: no element and no node moved.

        See :meth:`~edelweissfe.solvers.base.nonlinearsolverbase.NonlinearSolverBase.onNewLocations`.
        """

        locations = self.system.locations

        self._U = self._U.onLocations(locations)
        self._dU = self._dU.onLocations(locations)
        self._V = self._V.onLocations(locations)
        self._A = self._A.onLocations(locations)
        self._netForce = self._netForce.onLocations(locations)
        self._internalForce = self._internalForce.onLocations(locations)

    def _assembleLumpedOperators(self):
        """Assemble the lumped inertia and the lumped damping of the elements for the new numbering, and, in a model
        without entities that carry their own inertia, everything the explicit update divides by
        (:meth:`_deriveLumpedOperators`). The mass is that of the start of the step: nothing assembles it again before
        the next numbering, which every step starts with.

        In a model with entities that carry their own inertia, the operators are derived whenever their inertia and
        damping are taken, in every increment (:meth:`_takeLumpedOperatorsOfEntitiesWithOwnInertia`); the elements'
        part is the one of this numbering. The degrees of freedom of the entities (:attr:`ids_inertiaOfEntities`) get
        no element inertia or damping.
        """

        locations = self.system.locations
        nDof = self.system.numbering.nDof

        # The entities whose inertia is assembled element by element, as in every solver that does so (see
        # DiscreteSystem.elementsWithNodalInertia). Their locations are the same in every locations made from this
        # numbering, so the operators are the same too.
        M = np.zeros(nDof)
        damping = np.zeros(nDof)
        for el in self.system.elementsWithNodalInertia:
            Me = np.zeros(el.nDof)
            el.computeLumpedInertia(Me)
            M[locations[el]] += Me

            # Each field's first-derivative coefficient: zero mechanically, the non-local viscosity always.
            Ce = np.zeros(el.nDof)
            el.computeLumpedDamping(Ce)
            damping[locations[el]] += Ce

        self._inertiaOfTheElements = M
        self._dampingOfTheElements = damping

        if not self.system.entitiesWithOwnInertia:
            self._deriveLumpedOperators()

    def _classifyFieldsOfTheMesh(self, inertia: np.ndarray, damping: np.ndarray):
        """Classify the fields by time derivative (:meth:`_classifyFieldsByScheme`) for the current mesh, once per mesh
        and step (:attr:`_meshOfTheClassification`): a field with an inertia is second order, one with only a damping
        first order. Report the scheme, check it against the deck, and name the second-order fields that enter the
        linear momentum and the energy balance.

        Parameters
        ----------
        inertia
            The assembled lumped inertia of every degree of freedom, of the elements and of the entities.
        damping
            The assembled lumped damping, likewise.
        """

        self._meshOfTheClassification = self.system.model.topology.version
        self.firstOrderFields, self.secondOrderFields = self._classifyFieldsByScheme(inertia, damping)

        self.journal.message(
            "Time integration, derived from the assembled operators: central difference for {:}; "
            "forward Euler for {:}".format(
                ", ".join(self.secondOrderFields) or "(no field)",
                ", ".join(self.firstOrderFields) or "(no field)",
            ),
            self.identification,
            1,
        )

        self._checkDerivedSchemeAgainstTheDeck()

        # Which second-order fields enter the linear momentum and the energy balance is a property of
        # the field, taken from the registry; see phenomena.inertiaKind.
        self.linearMomentumFields = [f for f in self.secondOrderFields if carriesLinearMomentum(f)]
        self.nonMechanicalSecondOrderFields = [f for f in self.secondOrderFields if not carriesKineticEnergy(f)]

    def _deriveLumpedOperators(self):
        """Derive everything the explicit update divides by from the lumped inertia and damping of the elements and of
        the entities that carry their own inertia, once both are checked (a damping is finite and not negative, an
        inertia finite; a negative divisor is refused below): the classification of the fields by time derivative
        (:meth:`_classifyFieldsOfTheMesh`), a property of the mesh derived for the first numbering of a step and for a
        new mesh (see :meth:`onNewNumbering`), with the
        entities' operators once they are known; the index sets of the fields; and, as arrays of the length of the
        numbering, the inverse lumped mass the increment divides by, the lumped mass with and without the
        multi-point-constraint fold (:attr:`mpcTransformation`; the unfolded one for the diagnostics), the damping
        rate, the inertia and the damping of every degree of freedom, and the factors of the velocity update.
        """

        nDof = self.system.numbering.nDof
        ids = self.ids_inertiaOfEntities

        M = self._inertiaOfTheElements.copy()
        damping = self._dampingOfTheElements.copy()
        if self._inertiaOfEntities is not None:
            M[ids] = self._inertiaOfEntities[ids]
            damping[ids] = self._dampingOfEntities[ids]

        # The damping of the elements and of the entities, checked on its own: at second-order DOFs it enters only as
        # the rate C/M, which the check of the divisor below never sees, and a negative rate amplifies instead of
        # damping.
        if not np.all(np.isfinite(damping)) or np.any(damping < 0.0):
            raise ValueError(
                "The assembled lumped damping is not a valid dissipation: {:} of {:} entries are "
                "negative and {:} are not finite (smallest: {:e}). A damping coefficient enters the "
                "second-order update as the rate C/M, where a negative value amplifies instead of "
                "damping, and a first-order field divides by it directly.".format(
                    int(np.count_nonzero(damping < 0.0)),
                    damping.shape[0],
                    int(np.count_nonzero(~np.isfinite(damping))),
                    np.nanmin(damping),
                )
            )

        # A lumped inertia that is not a number passes every comparison below, and would reach every increment.
        if not np.all(np.isfinite(M)):
            raise ValueError(
                "The assembled lumped inertia is not finite at {:} of {:} degrees of freedom.".format(
                    int(np.count_nonzero(~np.isfinite(M))), M.shape[0]
                )
            )

        if self._meshOfTheClassification != self.system.model.topology.version:
            self._classifyFieldsOfTheMesh(M, damping)

        self.ids_1st = self._dofIndicesOfFields(self.firstOrderFields)
        self.ids_2nd = self._dofIndicesOfFields(self.secondOrderFields)
        self.ids_mechanicalEnergy = self._dofIndicesOfFields(
            [f for f in self.secondOrderFields if carriesKineticEnergy(f)]
        )

        # First-order fields integrate C * rate = f_net, so from here on M is "the divisor": the inertia at
        # second-order DOFs, the damping at first-order ones.
        M[self.ids_1st] = damping[self.ids_1st]

        # Kept before folding, so the kinetic energy diagnostic accounts for the true velocities of
        # all nodes (including tied slaves) rather than master-placed folded mass.
        self._rawLumpedMass = M.copy()

        # compute inverses; the entities that carry their own inertia give theirs in every increment
        withoutCoefficient = M == 0.0
        withoutCoefficient[self.ids_inertiaOfEntities] = False
        if np.any(withoutCoefficient):
            raise ValueError(
                "Zero found in the vector the increment divides by, at {:} of {:} degrees of freedom. "
                "Every FIELD is covered by the classification above, which refuses a field carrying "
                "neither coefficient, so what is left here are degrees of freedom belonging to no field "
                "-- scalar variables, which this solver has no equation of motion for.".format(
                    int(np.count_nonzero(withoutCoefficient)), M.shape[0]
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

        Minv = np.zeros(nDof)
        Minv[M != 0.0] = 1.0 / M[M != 0.0]
        self._Minv = Minv

        # Damping rate alpha = C / M at second-order DOFs with nonzero (folded) inertia; zero elsewhere.
        self._dampingRate = np.zeros(nDof)
        integrating = self.ids_2nd[M[self.ids_2nd] > 0.0]
        self._dampingRate[integrating] = damping[integrating] / M[integrating]

        # kept (instead of 1/Minv) for the kinetic energy: slave DOFs have Minv = 0
        self._lumpedMass = M

        # Per-DOF factors of the velocity update, so that it runs on whole vectors.
        self._secondOrderMask = np.zeros(nDof)
        self._secondOrderMask[self.ids_2nd] = 1.0
        self._halfDampingRate = 0.5 * self._dampingRate * self._secondOrderMask

        # The coefficients of every degree of freedom's equation of motion, M a + C v = f_net: by central differences,
        # the inertia and the damping (rate times inertia); by forward Euler, no inertia, and the damping the increment
        # divides by. Zero on a multi-point-constraint slave, whose inertia and damping are folded onto its masters.
        self._inertia = M * self._secondOrderMask
        self._damping = M * (self._dampingRate + (1.0 - self._secondOrderMask))

    def secondOrderMomentum(self, mass: np.ndarray, V: DofVector, model: FEModel) -> np.ndarray:
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

        return linearMomentum(np.asarray(mass) * np.asarray(V), self.system.numbering, self.linearMomentumFields, model)

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
            indices = self.system.numbering.idcsOfFieldsInDofVector[fieldName]
            totals[fieldName] = float(np.sum(self._rawLumpedMass[indices]))
        return totals

    def _dofIndicesOfFields(self, fieldNames: list[str]) -> np.ndarray:
        """The degrees of freedom of the named fields, concatenated in the order given.

        Parameters
        ----------
        fieldNames
            Names of fields present in the current numbering.

        Returns
        -------
        np.ndarray
            Their indices in the dof vector; empty (and of integer dtype, which matters -- numpy
            reads an index of None as np.newaxis) when no field is named.
        """

        indices = np.empty(0, dtype=int)
        for fieldName in fieldNames:
            indices = np.r_[indices, self.system.numbering.idcsOfFieldsInDofVector[fieldName]]
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
        micro-inertia, which for some materials is an optional material property that legitimately
        defaults to zero, so forgetting it silently returns the field to the parabolic scheme -- whose own
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
                                "micro-inertia; for some materials that is an optional material "
                                "property, and leaving it off legitimately means zero."
                                if expected == "second order"
                                else ""
                            ),
                        )
                    )

    def _classifyFieldsByScheme(self, inertia: np.ndarray, damping: np.ndarray) -> tuple[list[str], list[str]]:
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

        The degrees of freedom of the elements decide, and those of the entities that carry their own inertia
        (:attr:`ids_inertiaOfEntities`) wherever these give them a coefficient, an inertia or a damping: a node their
        kernel functions reach only where they vanish has neither, and no equation of motion. A field of the entities
        with a damping and no inertia, e.g. the nonlocal field of gradient-enhanced particles without a micro-inertia,
        is therefore first order, as it is for the elements. A field of the entities that gets no coefficient at any
        of its degrees of freedom, e.g. the displacement of particles without a density, is refused like a field of
        elements that carries neither.

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

        # The DOFs of the elements decide, and those of the entities that carry their own inertia where these give a
        # coefficient: a node their kernel functions reach only where they vanish has none, and no equation of motion.
        dofs = np.arange(self.system.numbering.nDof)
        decides = np.ones(dofs.size, dtype=bool)
        ids = self.ids_inertiaOfEntities
        decides[ids] = (np.asarray(inertia)[ids] != 0.0) | (np.asarray(damping)[ids] != 0.0)

        for fieldName, indicesOfField in self.system.numbering.idcsOfFieldsInDofVector.items():
            indices = dofs[indicesOfField][decides[indicesOfField]]
            carriesInertia = np.asarray(inertia[indices]) != 0.0
            carriesDamping = np.asarray(damping[indices]) != 0.0

            if not carriesInertia.any() and not carriesDamping.any():
                # Asked first: of a field that only the entities carry, no degree of freedom is left to decide then,
                # and "all of none" would hold.
                raise ValueError(
                    "Field {:} carries neither an inertia nor a damping, so there is no time "
                    "derivative for this solver to integrate it with. Its elements, and the entities that carry "
                    "their own inertia, report both (computeLumpedInertia and computeLumpedDamping); "
                    "a zero density or a zero volume gives a zero inertia.".format(fieldName)
                )
            elif carriesInertia.all():
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
            else:
                raise ValueError(
                    "Field {:} carries no inertia, so it is integrated by its damping alone -- but "
                    "a damping was assembled on only {:} of its {:} degrees of freedom, and the "
                    "rest have nothing to divide by.".format(
                        fieldName, int(np.count_nonzero(carriesDamping)), carriesDamping.size
                    )
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
            The velocity vector of the new discrete system.
        model
            The model tree.

        Raises
        ------
        RuntimeError
            If any field's own lumped total changed by more than its conservation tolerance.
        """

        lumpedTotalsAfter = self._perFieldLumpedTotals()
        momentumAfter = self.secondOrderMomentum(self._rawLumpedMass, V, model)
        kineticAfter = self.kineticEnergy(V)

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
