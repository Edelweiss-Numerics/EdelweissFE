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
"""The nonlinear implicit dynamic solver: Newmark-beta time integration on top of the Newton loop
of :class:`~edelweissfe.solvers.nonlinearimplicitstatic.NIST`.

An increment solves the semi-discrete equation of motion

.. math::

    \\boldsymbol{M} \\ddot{\\boldsymbol{u}}_{n+1} + \\boldsymbol{C} \\dot{\\boldsymbol{u}}_{n+1}
    + \\boldsymbol{P}_\\mathrm{int}(\\boldsymbol{u}_{n+1}) = \\boldsymbol{P}_\\mathrm{ext}(t_{n+1})

for the displacement at the end of the increment, with the velocity and the acceleration expressed
through the displacement increment :math:`\\Delta \\boldsymbol{u} = \\boldsymbol{u}_{n+1} - \\boldsymbol{u}_n`
by the two Newmark relations (Newmark 1959; Hughes, *The Finite Element Method*, ch. 9)

.. math::

    \\boldsymbol{u}_{n+1} &= \\boldsymbol{u}_n + \\Delta t \\, \\dot{\\boldsymbol{u}}_n
        + \\tfrac{\\Delta t^2}{2} \\left[ (1 - 2\\beta) \\, \\ddot{\\boldsymbol{u}}_n
        + 2\\beta \\, \\ddot{\\boldsymbol{u}}_{n+1} \\right], \\\\
    \\dot{\\boldsymbol{u}}_{n+1} &= \\dot{\\boldsymbol{u}}_n
        + \\Delta t \\left[ (1 - \\gamma) \\, \\ddot{\\boldsymbol{u}}_n + \\gamma \\, \\ddot{\\boldsymbol{u}}_{n+1} \\right].

Solved for the end-of-increment kinematics these read

.. math::

    \\ddot{\\boldsymbol{u}}_{n+1} &= \\frac{\\Delta \\boldsymbol{u} - \\Delta t \\, \\dot{\\boldsymbol{u}}_n
        - \\Delta t^2 \\left( \\tfrac{1}{2} - \\beta \\right) \\ddot{\\boldsymbol{u}}_n}{\\beta \\, \\Delta t^2}, \\\\
    \\dot{\\boldsymbol{u}}_{n+1} &= \\dot{\\boldsymbol{u}}_n
        + \\Delta t \\left[ (1 - \\gamma) \\, \\ddot{\\boldsymbol{u}}_n + \\gamma \\, \\ddot{\\boldsymbol{u}}_{n+1} \\right],

so that the effective residual and tangent handed to the Newton loop are

.. math::

    \\boldsymbol{R} &= \\boldsymbol{P}_\\mathrm{ext} - \\boldsymbol{P}_\\mathrm{int}
        - \\boldsymbol{M} \\ddot{\\boldsymbol{u}}_{n+1} - \\boldsymbol{C} \\dot{\\boldsymbol{u}}_{n+1}, \\\\
    \\boldsymbol{K}_\\mathrm{eff} &= \\boldsymbol{K} + \\frac{1}{\\beta \\, \\Delta t^2} \\boldsymbol{M}
        + \\frac{\\gamma}{\\beta \\, \\Delta t} \\boldsymbol{C}.

Everything else -- element evaluation, loads, constraints, multi-point-constraint condensation,
Dirichlet handling, the convergence test, cutbacks and the linear solver -- is the parent's, byte
for byte: the dynamics enter only as two terms in the residual and two in the tangent, inserted
where the parent forms ``R = P_ext - P_int`` and before the tangent is converted to CSR.

**Parameters.** The default :math:`\\beta = 1/4`, :math:`\\gamma = 1/2` is the average-acceleration
(trapezoidal) rule: second-order accurate, unconditionally stable for linear problems and free of
numerical damping, so the discrete energy of an undamped linear system is conserved exactly and
any drift in it points at a bug rather than at the scheme. :math:`\\gamma > 1/2` adds algorithmic
damping at the price of first-order accuracy; :math:`\\gamma \\geq 1/2` together with
:math:`\\beta \\geq (\\gamma + 1/2)^2 / 4` keeps the scheme unconditionally stable. A choice outside
that region is accepted with a warning, :math:`\\beta = 0` (the explicit central-difference member
of the family) is refused, because the formulation above divides by it.

**Mass.** The mass is the CONSISTENT one, :math:`\\int_\\Omega \\rho N^T N \\, dV`, assembled from
:meth:`~edelweissfe.elements.base.baseelement.BaseElement.computeConsistentInertia` into the very
sparsity pattern the stiffness uses -- both are scattered through the same VIJ layout of the
:class:`~edelweissfe.numerics.dofmanager.DofManager`, so adding :math:`\\boldsymbol{M}/(\\beta \\Delta t^2)`
to the tangent is an entry-wise addition of two value vectors, before the parent's in-place CSR
update, its multi-point-constraint condensation and its Dirichlet row replacement, all of which
therefore act on the effective matrix without knowing it is one. The damping :math:`\\boldsymbol{C}`
is the diagonal each element reports through ``computeLumpedDamping``, placed on the diagonal of
the same layout. Both are assembled when the equation system is (re)built -- on the first increment
and after a change of a constraint's connectivity -- not per Newton iteration; the one exception is
a step that changes a material property mid-step (``>>changematerialproperty``), which invalidates
the density and makes them reassemble every increment.

**Which fields.** Only the fields whose inertia is a mass
(:func:`~edelweissfe.config.phenomena.carriesLinearMomentum`, i.e. the displacement) are integrated
in time. Rows and columns of the mass and the damping belonging to any other field are zeroed, and
those fields keep the parent's quasi-static treatment. A model with no such field is refused.

**Zero-length and negligible increments.** The time stepper yields a zero-length increment before
the first real one of every step. A quasi-static solver equilibrates it; this solver skips it with
the state kept, because in zero time nothing can move: displacement and velocity are continuous,
and a load appearing at that instant is answered by the acceleration below, not by a displacement.
Solving it statically would put a suddenly loaded model at its static deflection before the first
real increment -- the whole dynamic response skipped. An increment shorter than :math:`10^{-10}` of
the time elapsed in the step is skipped for the same reason: it is the round-off remainder the
stepper's progress accumulation leaves at the end of a step (400 increments of 0.0025 leave 2e-14),
and the Newmark update would divide by its square.

**Initial acceleration.** At the start of a step the acceleration is computed from equilibrium,
:math:`\\boldsymbol{M} \\ddot{\\boldsymbol{u}}_0 = \\boldsymbol{P}_\\mathrm{ext}(t_0) - \\boldsymbol{P}_\\mathrm{int}(\\boldsymbol{u}_0)
- \\boldsymbol{C} \\dot{\\boldsymbol{u}}_0`, with the loads evaluated at the step's own start (an
``f(t)`` amplitude at :math:`t = 0`). Without it, a load applied suddenly at the start of a step
would enter the trapezoidal rule as though it were zero at :math:`t_0`, which costs one first-order
error in the momentum, and the scheme's second order is lost. A resumed run skips this: the
checkpoint carries the acceleration, and recomputing it would replace a consistent state with one
that agrees only to solver tolerance. Switch it off with ``computeInitialAcceleration=False`` to
continue a step from whatever acceleration was carried over.

**State and restart.** Velocity and acceleration are ordinary node-field entries, ``V`` and ``A``,
written after every converged increment alongside the parent's ``U``/``P``/``dU`` -- the same
mechanism the explicit solver uses for its velocity. That makes them part of every checkpoint the
``*output, type=restart`` manager writes and of every ``*fieldOutput`` (``result=V``,
``result=A``), with no solver-specific restart code beyond a marker that tells a resumed run not to
recompute the initial acceleration.

**Live h-adaptivity.** A ``*modelModifier, type=hAdaptivity`` refining the mesh mid-step is
supported. Three things make that work, and only the third is this solver's own:

* Being node-field entries, ``V`` and ``A`` are carried onto the nodes a refinement creates by the
  same isoparametric interpolation from the parent element that already carries ``U``
  (:data:`~edelweissfe.modelmodifiers.adaptivity.hadaptivity.WARM_STARTED_NODE_FIELD_ENTRIES`).
  Nothing here interpolates anything itself.
* The mass and the damping are not transferred at all: they are reassembled from the elements that
  now exist, because the parent rebuilds its
  :class:`~edelweissfe.numerics.dofmanager.DofManager` on a topology change and everything bound to
  it is rebuilt with it. There is therefore no re-lumping error of the kind an explicit solver has
  to account for -- the total mass is an assembly of the new mesh, not a redistribution of the old.
* The interpolated acceleration is only as good as an interpolation of the *previous* mesh's
  acceleration, and it is not in equilibrium with the operators that were just reassembled. So the
  increment after a topology change **recomputes the acceleration from equilibrium**, with the
  cold-start machinery a step start already uses (``computeInitialAcceleration``, which also
  switches this off). The displacement and the velocity are kept: they are the kinematic state, and
  interpolation is the best statement of them the new mesh admits. Only a change that the topology
  pipeline actually recorded re-arms it -- a mere change of a constraint's connectivity (a contact
  candidate list) rebuilds the equation system without moving a node, and must not restart the
  acceleration of every degree of freedom in the model.

What is exactly conserved across a refinement, and what is not, is reported at every event and
follows from the same partition-of-unity argument the explicit solver's own live-adaptivity notes
make: the assembled total mass is a geometric identity (children tile their parent at the same
density) and is *asserted*; the linear momentum and the kinetic energy are exact for a spatially
uniform velocity field and differ at second order in the velocity gradient across a refined parent
otherwise, which is discretisation error rather than a defect, and are reported only. Unlike an
explicit run, the increment that follows re-establishes equilibrium by Newton iteration, so an
imbalance the transfer leaves behind is absorbed rather than radiated.

**Not supported in this version.** Rayleigh or any other damping model beyond what the elements
report as their lumped damping.
"""

from dataclasses import dataclass

import numpy as np
from scipy.sparse import coo_matrix, csr_matrix, diags

import edelweissfe.utils.performancetiming as performancetiming
from edelweissfe.config.phenomena import carriesLinearMomentum
from edelweissfe.models.femodel import FEModel
from edelweissfe.numerics.dofmanager import DofManager, DofVector, VIJSystemMatrix
from edelweissfe.outputmanagers.base.outputmanagerbase import OutputManagerBase
from edelweissfe.solvers.base.dirichlet import applyDirichletToStiffness
from edelweissfe.solvers.nonlinearimplicitstatic import NIST, NISTSchema
from edelweissfe.timesteppers.timestep import TimeStep
from edelweissfe.utils.exceptions import DivergingSolution, ReachedMaxIterations
from edelweissfe.utils.fieldoutput import FieldOutputController
from edelweissfe.utils.schema import schemaField

#: Relative asymmetry above which an assembled consistent mass is rejected. It is the sum of
#: element matrices :math:`\int \rho N^T N`, each symmetric by construction, so any asymmetry
#: beyond round-off means an element wrote something that is not a mass matrix into its slot.
_MASS_SYMMETRY_TOLERANCE = 1e-10

#: Relative change of a field's total assembled mass above which a topology change is rejected.
#: The children of a refined element tile it and carry the same density, so the total is conserved
#: geometrically; the quadrature that assembles it is exact only up to a polynomial order, which
#: admits a small change on a distorted element. The same default, and the same reasoning, as the
#: explicit solver's ``lumped-quantity-conservation-tolerance``.
_MASS_CONSERVATION_TOLERANCE = 1e-6

#: An increment shorter than this fraction of the time elapsed in the step is the round-off
#: remainder of the time stepper's progress accumulation, not an increment, and is skipped with the
#: state kept -- see :meth:`NonlinearImplicitDynamic.solveIncrement`. Far below any increment a deck
#: can ask for (``minInc`` is a fraction of the step, typically 1e-8 at its smallest) and far above
#: the double-precision remainder (observed: 2e-14 of a step of 2.0).
_NEGLIGIBLE_INCREMENT_FRACTION = 1e-10


@dataclass(frozen=True)
class NIDSchema(NISTSchema):
    """:class:`~edelweissfe.solvers.nonlinearimplicitstatic.NISTSchema`'s options, plus the
    Newmark parameters and the initial-acceleration switch. Mirrors
    :attr:`NonlinearImplicitDynamic.SolverSpecificOptions` one-for-one; that dict remains the
    runtime source of truth, for the reasons given on the parent schema.
    """

    newmarkBeta: float | None = schemaField(
        description=(
            "Newmark parameter beta, weighting the end-of-increment acceleration in the displacement "
            "update. Must be > 0. 1/4 with gamma = 1/2 is the average-acceleration (trapezoidal) rule."
        ),
        dtype=float,
        default=0.25,
    )
    newmarkGamma: float | None = schemaField(
        description=(
            "Newmark parameter gamma, weighting the end-of-increment acceleration in the velocity "
            "update. 1/2 gives second-order accuracy and no algorithmic damping; larger values damp."
        ),
        dtype=float,
        default=0.5,
    )
    computeInitialAcceleration: bool | None = schemaField(
        description=(
            "Compute the acceleration at the start of every step from equilibrium with the loads at "
            "the step's start, instead of continuing from the carried-over acceleration. Skipped "
            "automatically when the step resumes from a restart checkpoint. Also re-armed for the "
            "increment following a topology change (h-adaptivity), whose interpolated acceleration "
            "is not in equilibrium with the operators reassembled on the refined mesh."
        ),
        dtype=bool,
        default=True,
    )
    massConservationTolerance: float | None = schemaField(
        description=(
            "Relative-change tolerance for the per-topology-change check on a dynamic field's total "
            "assembled mass. Children of a refined element tile it and carry the same density, so "
            "this is conserved geometrically; the default is already generous relative to floating-"
            "point precision and exists to catch a genuinely wrong refinement, not to absorb "
            "ordinary quadrature noise. The counterpart of the explicit solver's "
            "lumped-quantity-conservation-tolerance."
        ),
        dtype=float,
        default=_MASS_CONSERVATION_TOLERANCE,
    )


@dataclass
class _NewmarkSystem:
    """Everything the Newmark increment reads that is sized by the current equation system.

    Bound to the :class:`~edelweissfe.numerics.dofmanager.DofManager` it was assembled for: the
    moment the parent rebuilds that manager, every entry here is stale together, so the whole
    bundle is replaced rather than any part of it patched.

    Parameters
    ----------
    dofManager
        The manager these operators and vectors are indexed by.
    dynamicDofs
        The degrees of freedom integrated in time -- those of the fields carrying a mass.
    dynamicFields
        The names of those fields, in the model's field order.
    Mvij, Cvij
        The consistent mass and the (diagonal) damping as VIJ value vectors, in the stiffness'
        layout, ready to be scaled and added onto the tangent.
    M, C
        The same operators as CSR matrices, for the residual's matrix-vector products.
    V, A
        The velocity and acceleration at the last converged increment.
    """

    dofManager: DofManager
    dynamicDofs: np.ndarray
    dynamicFields: list
    Mvij: VIJSystemMatrix
    Cvij: VIJSystemMatrix
    M: csr_matrix
    C: csr_matrix
    V: DofVector
    A: DofVector


@dataclass(frozen=True)
class _ConservedQuantities:
    """What a topology change ought to leave alone, measured on one :class:`_NewmarkSystem`.

    Parameters
    ----------
    massByField
        Each dynamic field's own total assembled mass. Per field, never summed across them: a
        violation in a numerically small field would otherwise hide inside a large one.
    momentum
        The linear momentum :math:`M v`, per spatial component. Per component, not summed over
        them: adding a momentum's x, y and z contributions produces a number with no physical
        meaning and would hide a component-wise error behind a cancellation.
    kineticEnergy
        :math:`\\tfrac{1}{2} v^T M v`.
    """

    massByField: dict
    momentum: np.ndarray
    kineticEnergy: float


class NonlinearImplicitDynamic(NIST):
    """This is the Nonlinear Implicit Dynamic -- solver (``NID``), Newmark-beta time integration on
    top of the Newton loop of :class:`~edelweissfe.solvers.nonlinearimplicitstatic.NIST`.

    Parameters
    ----------
    jobInfo
        A dictionary containing the job information.
    journal
        The journal instance for logging.
    """

    identification = "NID"

    #: Live h-adaptivity is supported: the velocity and the acceleration ride the node-field warm
    #: start onto new nodes, the mass and the damping are reassembled on the refined mesh, and the
    #: acceleration is re-equilibrated on the increment that follows. See the module docstring.
    supportsModelModifiers = True

    #: Option schema for this solver, per OptionSchemaProvider.
    schema = NIDSchema

    SolverSpecificOptions = NIST.SolverSpecificOptions | {
        "newmarkBeta": 0.25,
        "newmarkGamma": 0.5,
        "computeInitialAcceleration": True,
        "massConservationTolerance": _MASS_CONSERVATION_TOLERANCE,
    }

    def __init__(self, jobInfo, journal, **kwargs):
        super().__init__(jobInfo, journal, **kwargs)

        #: The operators and kinematic state of the current equation system; None until the
        #: first increment of a step has built it. See :class:`_NewmarkSystem`.
        self._newmarkSystem = None
        #: Whether the next increment has to compute the acceleration from equilibrium first.
        #: Armed by :meth:`solveStep` for a step starting cold, disarmed by the increment that
        #: consumes it.
        self._initialAccelerationPending = False
        #: Staged by :meth:`readRestart`, which the driver calls BEFORE :meth:`solveStep` on the
        #: resumed step, and consumed there exactly once -- the same staging the explicit solver
        #: uses for its external work.
        self._resumedFromCheckpoint = False
        #: Length of ``model.topologyHistory`` when the current equation system was assembled.
        #: A change in it is what distinguishes a rebuild caused by the mesh actually changing
        #: from one caused by a constraint re-reporting its connectivity; see
        #: :meth:`_ensureNewmarkSystem`.
        self._topologyRecordsAtLastBuild = 0

        self._validateNewmarkParameters()

    def _validateNewmarkParameters(self):
        """Refuse a beta the implicit formulation divides by zero with, and warn about a pair
        outside the unconditionally stable region."""

        beta = self.options["newmarkBeta"]
        gamma = self.options["newmarkGamma"]

        if beta <= 0.0:
            raise ValueError(
                "newmarkBeta must be > 0 (got {:}): the implicit Newmark update divides by beta*dt^2; "
                "beta = 0 is the explicit central-difference member of the family, which this solver "
                "does not integrate -- use NED for that.".format(beta)
            )

        if gamma < 0.5 or beta < 0.25 * (gamma + 0.5) ** 2:
            self.journal.message(
                "Newmark parameters beta={:}, gamma={:} lie outside the unconditionally stable region "
                "(gamma >= 1/2 and beta >= (gamma + 1/2)^2 / 4); the time increment is then bounded by "
                "the highest frequency of the mesh, which nothing here checks.".format(beta, gamma),
                self.identification,
                0,
            )
        elif gamma != 0.5:
            self.journal.message(
                "Newmark gamma={:} differs from 1/2: the scheme is numerically damped and first-order "
                "accurate in time.".format(gamma),
                self.identification,
                1,
            )

    def writeRestart(self, restartFile):
        """Mark the checkpoint as carrying Newmark kinematics.

        The velocity and acceleration themselves need no solver code: they are node-field entries
        and go into the checkpoint with every other entry. This marker exists so that a resumed
        step knows the acceleration it finds there is the scheme's own consistent one and must not
        be replaced by a fresh equilibrium solve -- see :meth:`readRestart`.

        Parameters
        ----------
        restartFile
            The open checkpoint to write to.
        """

        restartFile.require_group("solver").attrs["newmarkKinematicsCheckpointed"] = True

    def readRestart(self, restartFile):
        """Note that the step about to be solved resumes from a checkpoint written by this solver.

        Tolerates a checkpoint written by another solver, or without the marker: the resumed step
        then starts as a cold one, computing its initial acceleration from equilibrium, which is
        the right answer for a state that carries no acceleration of its own.

        Parameters
        ----------
        restartFile
            The open checkpoint to read from.
        """

        if "solver" not in restartFile:
            return
        self._resumedFromCheckpoint = bool(restartFile["solver"].attrs.get("newmarkKinematicsCheckpointed", False))

    def solveStep(
        self,
        step,
        model: FEModel,
        fieldOutputController: FieldOutputController,
        outputmanagers: dict[str, OutputManagerBase],
    ):
        """Public interface to solve for a step; see the parent.

        Arms the initial-acceleration computation for a step starting cold and drops the operators
        of the previous step -- the parent rebuilds its equation system at the start of every step,
        and the operators follow that system. The velocity and acceleration are not dropped: they
        are re-read from the ``V``/``A`` node-field entries the previous step (or the checkpoint)
        left, which is what carries the kinematic state across steps.

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

        # >>options blocks may have changed them since construction.
        self._validateNewmarkParameters()

        self._newmarkSystem = None

        resumed = self._resumedFromCheckpoint
        self._resumedFromCheckpoint = False
        self._initialAccelerationPending = bool(self.options["computeInitialAcceleration"]) and not resumed

        self.journal.message(
            "Newmark-beta time integration, beta={:}, gamma={:}; consistent mass{:}".format(
                self.options["newmarkBeta"],
                self.options["newmarkGamma"],
                (
                    "; acceleration from the restart checkpoint"
                    if resumed
                    else (
                        "; initial acceleration from equilibrium"
                        if self._initialAccelerationPending
                        else "; carrying the acceleration over"
                    )
                ),
            ),
            self.identification,
            1,
        )

        return super().solveStep(step, model, fieldOutputController, outputmanagers)

    def solveIncrement(
        self,
        U_n: DofVector,
        dU: DofVector,
        P: DofVector,
        K: VIJSystemMatrix,
        stepActions: list,
        model: FEModel,
        timeStep: TimeStep,
        prevTimeStep: TimeStep,
        extrapolation: str,
        maxIter: int,
        maxGrowingIter: int,
    ) -> tuple[DofVector, DofVector, DofVector, int, dict]:
        """Newton-Raphson scheme on the Newmark-effective equation of motion of an increment.

        The parent's loop, with the inertia and damping terms inserted into the residual and the
        tangent -- see the module docstring. A zero-length increment -- the one the time stepper
        yields before the first real increment of a step -- has no equation of motion and is
        skipped with the state kept: displacement and velocity are continuous in time, and a load
        appearing at that instant is answered by the initial acceleration, not by a displacement.

        Parameters
        ----------
        U_n
            The old solution vector.
        dU
            The old solution increment.
        P
            The old reaction vector.
        K
            The system matrix to be used.
        stepActions
            The list of active step actions.
        model
            The model tree.
        timeStep
            The time step.
        prevTimeStep
            The previous time step.
        extrapolation
            The type of extrapolation to be used.
        maxIter
            The maximum number of iterations to be used.
        maxGrowingIter
            The maximum number of growing residuals until the Newton-Raphson is terminated.

        Returns
        -------
        tuple[DofVector,DofVector,DofVector,int,dict]
            A tuple containing
                - the new solution vector
                - the solution increment
                - the new reaction vector
                - the number of required iterations
                - the history of residuals per field
        """

        dT = timeStep.timeIncrement

        iterationCounter = 0
        incrementResidualHistory = dict.fromkeys(self.theDofManager.idcsOfFieldsInDofVector, (0.0, 0))

        if dT <= _NEGLIGIBLE_INCREMENT_FRACTION * abs(timeStep.stepTime):
            # A zero-length increment has no equation of motion. The displacement and the velocity
            # are continuous in time, so the state cannot change in zero time, and a load that
            # appears at this instant appears as an ACCELERATION -- which is exactly what the
            # equilibrium solve for the initial acceleration provides -- not as a displacement.
            # The parent's quasi-static solve would instead put the model at static equilibrium
            # with the new load in zero time; for a suddenly applied load that is the whole dynamic
            # response, skipped before it began. The time stepper yields one such increment before
            # the first real one of every step, so this is the ordinary path, not an edge case.
            #
            # An increment that is merely NEGLIGIBLE is treated the same way, for a different
            # reason: the time stepper closes a step with whatever step progress its accumulation
            # of increments left over, and 400 increments of 0.0025 leave a remainder of 2e-14 --
            # a 401st increment of that length. It carries no physics, but the Newmark update
            # divides by beta*dt^2 and would amplify the round-off of the linear solve by 1e28
            # into the acceleration and the velocity.
            self.applyStepActionsAtIncrementStart(model, timeStep, stepActions)
            self.journal.message(
                "{:} increment ({:e}): no equation of motion to integrate, state kept".format(
                    "zero-length" if dT <= 0.0 else "negligible", dT
                ),
                self.identification,
                2,
            )
            dU[:] = 0.0
            return U_n, dU, P, iterationCounter, incrementResidualHistory

        beta = self.options["newmarkBeta"]
        gamma = self.options["newmarkGamma"]
        # d(A_np)/d(dU) and d(V_np)/d(dU), the factors the mass and the damping enter the tangent with
        accelerationFactor = 1.0 / (beta * dT * dT)
        velocityFactor = gamma / (beta * dT)

        elements = model.elements
        constraints = model.constraints

        R = self.theDofManager.constructDofVector()
        F = self.theDofManager.constructDofVector()
        PExt = self.theDofManager.constructDofVector()
        U_np = self.theDofManager.constructDofVector()
        V_np = self.theDofManager.constructDofVector()
        A_np = self.theDofManager.constructDofVector()
        ddU = None

        dirichlets = stepActions["dirichlet"].values()
        nodeforces = stepActions["nodeforces"].values()
        distributedLoads = stepActions["distributedload"].values()
        bodyForces = stepActions["bodyforce"].values()

        # Find which global DOFs the Dirichlet BCs constrain, once up front.
        self.locateConstrainedDofs(dirichlets)

        self.applyStepActionsAtIncrementStart(model, timeStep, stepActions)

        # After the step actions, not before: a ``>>changematerialproperty`` acting on this
        # increment changes the density the mass is assembled from, and assembling first would use
        # the previous increment's.
        system = self._ensureNewmarkSystem(model, stepActions)

        # dT is fixed within the increment, so the two terms the dynamics add to the tangent are
        # too: form them once here rather than scaling and adding two full-length value vectors
        # onto K on every Newton iteration.
        KDynamic = accelerationFactor * np.asarray(system.Mvij) + velocityFactor * np.asarray(system.Cvij)

        if self._initialAccelerationPending:
            self._initialAccelerationPending = False
            self._computeInitialAcceleration(system, U_n, stepActions, model, timeStep)

        dU, isExtrapolatedIncrement = self.extrapolateLastIncrement(
            extrapolation, timeStep, dU, dirichlets, prevTimeStep, model
        )

        V_n = system.V
        A_n = system.A

        while True:
            for geostatic in stepActions["geostatic"].values():
                geostatic.applyAtIterationStart()

            U_np[:] = U_n
            U_np += dU

            P[:] = K[:] = F[:] = PExt[:] = 0.0

            P, K, F = self.computeElements(elements, U_np, dU, P, K, F, timeStep)
            PExt, K = self.assembleLoads(nodeforces, distributedLoads, bodyForces, U_np, PExt, K, timeStep)
            PExt, K = self.assembleConstraints(constraints, U_np, dU, PExt, K, timeStep)

            # --- the dynamics: the trial displacement increment fixes the trial kinematics ---
            self._newmarkKinematics(dU, V_n, A_n, dT, beta, gamma, system.dynamicDofs, V_np, A_np)
            PInertia = system.M @ np.asarray(A_np)
            PDamping = system.C @ np.asarray(V_np)

            R[:] = -P
            R += PExt
            R -= PInertia
            R -= PDamping

            # The inertia and damping forces are fluxes of the same field as the internal forces,
            # so they enter the reference scale of the relative flux tolerance too. Without them a
            # body in free flight -- no internal force at all -- would be held to an absolute 1e-7.
            F += np.abs(PInertia)
            F += np.abs(PDamping)

            # Same VIJ layout as K, so the effective tangent is an entry-wise sum, before the CSR
            # conversion, the MPC condensation and the Dirichlet row replacement below.
            K += KDynamic

            # Condense the residual BEFORE the Dirichlet handling below: T^T folds slave-row
            # residuals into their master rows, which may themselves carry a prescribed delta --
            # transforming afterwards would corrupt it.
            if self.mpcTransformation is not None:
                R[:] = self.mpcTransformation.transformResidual(R, dU)

            # Row-replacement Dirichlet handling, exactly as in the parent; see there.
            if iterationCounter == 0 and not isExtrapolatedIncrement and dirichlets:
                R = self.applyDirichletToResidual(timeStep, R, dirichlets)
            else:
                for dirichlet in dirichlets:
                    R[dirichlet.constrainedDofIndices] = 0.0

                converged, nodesWithLargestResidual = self.checkConvergence(
                    R, ddU, F, iterationCounter, incrementResidualHistory
                )

                if converged:
                    break

                if self.checkDivergingSolution(incrementResidualHistory, maxGrowingIter):
                    self.printResidualOutlierNodes(nodesWithLargestResidual)
                    raise DivergingSolution("Residual grew {:} times, cutting back".format(maxGrowingIter))

                if iterationCounter == maxIter:
                    self.printResidualOutlierNodes(nodesWithLargestResidual)
                    raise ReachedMaxIterations("Reached max. iterations in current increment, cutting back")

            K_ = self.assembleStiffnessCSR(K)

            if self.mpcTransformation is not None:
                K_ = self.mpcTransformation.transformSystemMatrix(K_)

            K_ = self.applyDirichletToStiffness(K_, dirichlets)  # zero rows, unit diagonal

            ddU = self.linearSolve(K_, R)
            dU += ddU
            iterationCounter += 1

        # Converged: the trial kinematics become the state. Committed here, and only here, because a
        # normal return IS the parent's acceptance of the increment -- every failure path above
        # leaves the loop by exception and keeps V_n/A_n for the cutback.
        system.V[:] = V_np
        system.A[:] = A_np
        self._publishKinematics(system, model)

        kineticEnergy = 0.5 * float(np.dot(np.asarray(system.V), system.M @ np.asarray(system.V)))
        self.journal.message("kinetic energy {:e}".format(kineticEnergy), self.identification, 2)

        return U_np, dU, P, iterationCounter, incrementResidualHistory

    @staticmethod
    def _newmarkKinematics(
        dU: DofVector,
        V_n: DofVector,
        A_n: DofVector,
        dT: float,
        beta: float,
        gamma: float,
        dynamicDofs: np.ndarray,
        V_np: DofVector,
        A_np: DofVector,
    ):
        """The end-of-increment velocity and acceleration implied by a displacement increment.

        The two Newmark relations solved for the new acceleration and velocity (see the module
        docstring), on the dynamic degrees of freedom; every other entry of the outputs is zero,
        matching the zero rows of the mass and the damping there.

        Parameters
        ----------
        dU
            The trial displacement increment.
        V_n, A_n
            The velocity and acceleration at the start of the increment.
        dT
            The time increment.
        beta, gamma
            The Newmark parameters.
        dynamicDofs
            The degrees of freedom integrated in time.
        V_np, A_np
            The vectors to write the trial velocity and acceleration into.
        """

        dUDynamic = np.asarray(dU)[dynamicDofs]
        VDynamic = np.asarray(V_n)[dynamicDofs]
        ADynamic = np.asarray(A_n)[dynamicDofs]

        ANew = (dUDynamic - dT * VDynamic - dT * dT * (0.5 - beta) * ADynamic) / (beta * dT * dT)
        VNew = VDynamic + dT * ((1.0 - gamma) * ADynamic + gamma * ANew)

        A_np[:] = 0.0
        V_np[:] = 0.0
        A_np[dynamicDofs] = ANew
        V_np[dynamicDofs] = VNew

    def _publishKinematics(self, system: _NewmarkSystem, model: FEModel):
        """Write the committed velocity and acceleration to the ``V`` and ``A`` entries of the
        dynamic fields' node fields -- what field outputs read, what a checkpoint stores, and what
        the next equation system reads them back from.

        Parameters
        ----------
        system
            The current Newmark system.
        model
            The model tree.
        """

        for fieldName in system.dynamicFields:
            field = model.nodeFields[fieldName]
            self.theDofManager.writeDofVectorToNodeField(system.V, field, "V")
            self.theDofManager.writeDofVectorToNodeField(system.A, field, "A")

    def _ensureNewmarkSystem(self, model: FEModel, stepActions: dict) -> _NewmarkSystem:
        """The Newmark operators and state for the parent's current equation system, assembled if
        that system is new and reused otherwise.

        The parent rebuilds its :class:`~edelweissfe.numerics.dofmanager.DofManager` at the start
        of a step and whenever the topology or a constraint's connectivity changes; the operators
        are bound to that manager by identity and follow it. The velocity and acceleration are then
        read back from the node fields, where the last converged increment (or the checkpoint, or
        nothing -- a start from rest) left them, and where a refinement has meanwhile interpolated
        them onto the nodes it created. A step changing a material property mid-step invalidates the
        density, so the operators are reassembled every increment of such a step.

        A rebuild that follows a recorded **topology change** additionally re-arms the
        initial-acceleration solve and reports what the change did to the conserved quantities --
        see the module docstring. Two conditions narrow that, and both matter:

        * ``system is not None``: the first build of a step is not it. A step boundary rebuilds the
          manager too, and :meth:`solveStep` has already decided there whether that step starts cold
          (its own ``computeInitialAcceleration``/restart logic) -- re-deciding it here would
          override that decision with a different one, for every existing multi-step deck.
        * a grown ``model.topologyHistory``: a rebuild triggered by a constraint re-reporting its
          connectivity (a contact candidate list, which can tick on any increment) has moved no
          node and interpolated nothing, so there is no stale acceleration to replace and no
          conservation statement to make. Restarting the acceleration of the whole model on it
          would be both wrong and, repeated per increment, expensive.

        Parameters
        ----------
        model
            The model tree.
        stepActions
            The step's actions.

        Returns
        -------
        _NewmarkSystem
            The system for the current DofManager.
        """

        system = self._newmarkSystem
        dofManagerChanged = system is None or system.dofManager is not self.theDofManager

        if not dofManagerChanged and not stepActions["changematerialproperty"]:
            return system

        # Safe to advance unconditionally here: a topology change always rebuilds the manager, so
        # it can never be missed by an increment that returned above.
        topologyRecords = len(model.topologyHistory)
        topologyChanged = topologyRecords != self._topologyRecordsAtLastBuild
        self._topologyRecordsAtLastBuild = topologyRecords

        dynamicDofs, dynamicFields = self._locateDynamicDofs()

        if dofManagerChanged:
            V = self.theDofManager.constructDofVector()
            A = self.theDofManager.constructDofVector()
            for fieldName in dynamicFields:
                field = model.nodeFields[fieldName]
                # The input-file driver creates both entries (zero) on every mass-carrying field
                # before anything runs, and a checkpoint restores them -- so on that path they are
                # always present, zero meaning a start from rest. Guarded all the same for a model
                # assembled programmatically, which never passes through that driver: a missing
                # entry is then the same start from rest, not an error.
                if "V" in field:
                    V = self.theDofManager.writeNodeFieldToDofVector(V, field, "V")
                if "A" in field:
                    A = self.theDofManager.writeNodeFieldToDofVector(A, field, "A")
        else:
            V, A = system.V, system.A

        Mvij, Cvij, M, C = self._assembleOperators(model, dynamicDofs, dynamicFields)

        self._newmarkSystem = _NewmarkSystem(
            dofManager=self.theDofManager,
            dynamicDofs=dynamicDofs,
            dynamicFields=dynamicFields,
            Mvij=Mvij,
            Cvij=Cvij,
            M=M,
            C=C,
            V=V,
            A=A,
        )

        if dofManagerChanged and system is not None and topologyChanged:
            self.reportTopologyChangeConservation(
                self._conservedQuantities(system, model),
                self._conservedQuantities(self._newmarkSystem, model),
            )
            self._initialAccelerationPending = bool(self.options["computeInitialAcceleration"])
            self.journal.message(
                (
                    "acceleration will be recomputed from equilibrium on the refined mesh"
                    if self._initialAccelerationPending
                    else "keeping the interpolated acceleration (computeInitialAcceleration is off)"
                ),
                self.identification,
                1,
            )

        return self._newmarkSystem

    def _conservedQuantities(self, system: _NewmarkSystem, model: FEModel) -> _ConservedQuantities:
        """Total mass, linear momentum and kinetic energy of one Newmark system.

        Read off that system's own :class:`~edelweissfe.numerics.dofmanager.DofManager`, so the
        same method measures the state before and after a rebuild without either one having to
        know about the other.

        A field occupies a contiguous slice of the dof vector, node-major with the component
        innermost -- what ``writeNodeFieldToDofVector``'s flatten establishes -- so reshaping the
        slice recovers the per-node vectors, and the row sums of the mass over the slice count
        every field component once, hence the division by the field's dimension.

        Parameters
        ----------
        system
            The system to measure.
        model
            The model tree, for the fields' spatial dimension.

        Returns
        -------
        _ConservedQuantities
            The three quantities.

        Raises
        ------
        ValueError
            If two dynamic fields differ in spatial dimension, so that their momenta have no
            common components to add.
        """

        V = np.asarray(system.V)
        MV = np.asarray(system.M @ V)
        rowSums = np.asarray(system.M.sum(axis=1)).ravel()

        massByField = {}
        momentum = None
        for fieldName in system.dynamicFields:
            indices = system.dofManager.idcsOfFieldsInDofVector[fieldName]
            dimension = model.nodeFields[fieldName].dimension
            massByField[fieldName] = float(np.sum(rowSums[indices])) / dimension

            contribution = MV[indices].reshape((-1, dimension)).sum(axis=0)
            if momentum is not None and contribution.shape != momentum.shape:
                raise ValueError(
                    "Dynamic field {:} has dimension {:} against {:} for the fields before it, so "
                    "their momenta have no common components to add.".format(
                        fieldName, contribution.shape[0], momentum.shape[0]
                    )
                )
            momentum = contribution if momentum is None else momentum + contribution

        return _ConservedQuantities(
            massByField=massByField,
            momentum=momentum if momentum is not None else np.zeros(0),
            kineticEnergy=0.5 * float(V @ MV),
        )

    def reportTopologyChangeConservation(self, before: _ConservedQuantities, after: _ConservedQuantities):
        """Report what a topology change did to the quantities that ought to survive it.

        The three behave differently, and saying which is which is the whole point of reporting
        them together:

        * **Every dynamic field's total mass is conserved exactly.** The children of a refined
          element tile it at the same density, and the mass is *reassembled* here rather than
          transferred, so this is an identity of the assembly and not a property of any transfer.
          It is the cheapest correctness check on the whole refinement, so it **raises**.
        * **Linear momentum is conserved exactly for a spatially uniform velocity field**, because
          the shape functions carrying the velocity onto new nodes are a partition of unity and the
          children's masses sum to the parent's. For a general field the discrepancy is second
          order in the velocity gradient across the parent: discretisation error, not a defect.
          Reported, not enforced.
        * **Kinetic energy is not conserved** by interpolation in general, and it is the most
          sensitive of the three, being quadratic in the interpolation error. Reported as a
          relative jump; more than roughly a percent is a reason to look at the transfer rather
          than to believe the physics.

        Parameters
        ----------
        before, after
            The quantities measured on the system before and on the one after the change.

        Raises
        ------
        RuntimeError
            If any dynamic field's total mass changed by more than ``massConservationTolerance``.
        """

        tolerance = self.options["massConservationTolerance"]
        worstField, worstRelativeChange = "", 0.0
        for fieldName, massBefore in before.massByField.items():
            massAfter = after.massByField.get(fieldName, 0.0)
            relativeChange = abs(massAfter - massBefore) / massBefore if massBefore > 0.0 else 0.0
            if relativeChange > tolerance:
                raise RuntimeError(
                    "A topology change did not conserve the total mass of field '{:}': {:e} became "
                    "{:e}, a relative change of {:e} against a tolerance of {:e}. The children of a "
                    "refined element tile it and carry the same density, and the mass is reassembled "
                    "from the elements that now exist -- so a change of this size means the "
                    "refinement itself is wrong, not that the quadrature moved.".format(
                        fieldName, massBefore, massAfter, relativeChange, tolerance
                    )
                )
            if relativeChange >= worstRelativeChange:
                worstField, worstRelativeChange = fieldName, relativeChange

        momentumChange = float(np.max(np.abs(after.momentum - before.momentum))) if before.momentum.size else 0.0
        momentumScale = float(np.max(np.abs(before.momentum))) if before.momentum.size else 0.0
        kineticJump = (
            abs(after.kineticEnergy - before.kineticEnergy) / before.kineticEnergy
            if before.kineticEnergy > 0.0
            else 0.0
        )

        self.journal.message(
            "Topology change: worst-conserved mass, field '{:}', to {:.1e} relative; largest momentum "
            "component change {:.3e} (of {:.3e}); kinetic energy {:.6e} -> {:.6e} ({:+.2f} %)".format(
                worstField,
                worstRelativeChange,
                momentumChange,
                momentumScale,
                before.kineticEnergy,
                after.kineticEnergy,
                kineticJump * 100.0 * (1.0 if after.kineticEnergy >= before.kineticEnergy else -1.0),
            ),
            self.identification,
            1,
        )

    def _locateDynamicDofs(self) -> tuple[np.ndarray, list]:
        """The degrees of freedom integrated in time: those of every field whose inertia is a mass.

        Returns
        -------
        tuple[np.ndarray, list]
            Their indices in the dof vector, and the names of their fields in the model's order.

        Raises
        ------
        ValueError
            If the model carries no such field.
        """

        isDynamic = np.zeros(self.theDofManager.nDof, dtype=bool)
        dynamicFields = []
        for fieldName, indices in self.theDofManager.idcsOfFieldsInDofVector.items():
            if carriesLinearMomentum(fieldName):
                isDynamic[indices] = True
                dynamicFields.append(fieldName)

        if not dynamicFields:
            raise ValueError(
                "No field of this model carries a mass (see phenomena.inertiaKind), so there is nothing "
                "for {:} to integrate in time; it carries: {:}.".format(
                    self.identification, ", ".join(self.theDofManager.idcsOfFieldsInDofVector) or "(no field)"
                )
            )

        return np.flatnonzero(isDynamic), dynamicFields

    @performancetiming.timeit("assemble mass and damping")
    def _assembleOperators(
        self, model: FEModel, dynamicDofs: np.ndarray, dynamicFields: list
    ) -> tuple[VIJSystemMatrix, VIJSystemMatrix, csr_matrix, csr_matrix]:
        """Assemble the consistent mass and the diagonal damping, in the stiffness' VIJ layout and
        as CSR matrices.

        Entries coupling to a degree of freedom that is not integrated in time are zeroed in both:
        the mass an element reports on a non-mechanical block (a micro-inertia) belongs to a field
        this solver keeps quasi-static. Elements without kernels -- contact facets -- carry geometry,
        not a material, and are skipped.

        Parameters
        ----------
        model
            The model tree.
        dynamicDofs
            The degrees of freedom integrated in time.
        dynamicFields
            Their fields, for the per-field mass report.

        Returns
        -------
        tuple[VIJSystemMatrix, VIJSystemMatrix, csr_matrix, csr_matrix]
            The mass and the damping as VIJ value vectors, and the same two as CSR matrices.

        Raises
        ------
        ValueError
            If the assembled mass is not finite, not symmetric, or leaves a dynamic degree of
            freedom without any mass.
        """

        Mvij = self.theDofManager.constructVIJSystemMatrix()
        Cvij = self.theDofManager.constructVIJSystemMatrix()

        for el in model.elements.values():
            if not el.hasKernels:
                continue

            el.computeConsistentInertia(Mvij[el])

            Ce = np.zeros(el.nDof)
            el.computeLumpedDamping(Ce)
            if np.any(Ce):
                CeView = Cvij[el]
                CeView += np.diagflat(Ce).reshape(CeView.shape)

        nDof = self.theDofManager.nDof
        I = self.theDofManager.I  # noqa: E741
        J = self.theDofManager.J

        isDynamic = np.zeros(nDof, dtype=bool)
        isDynamic[dynamicDofs] = True
        couplesDynamicOnly = isDynamic[I] & isDynamic[J]
        Mvij[~couplesDynamicOnly] = 0.0
        Cvij[~couplesDynamicOnly] = 0.0

        if not np.all(np.isfinite(Mvij)) or not np.all(np.isfinite(Cvij)):
            raise ValueError("The assembled consistent mass or damping contains non-finite entries.")

        M = coo_matrix((np.asarray(Mvij), (I, J)), shape=(nDof, nDof)).tocsr()
        C = coo_matrix((np.asarray(Cvij), (I, J)), shape=(nDof, nDof)).tocsr()

        massScale = float(np.max(np.abs(M.data))) if M.nnz else 0.0
        antisymmetricPart = M - M.T
        asymmetry = float(np.max(np.abs(antisymmetricPart.data))) if antisymmetricPart.nnz else 0.0
        if massScale > 0.0 and asymmetry > _MASS_SYMMETRY_TOLERANCE * massScale:
            raise ValueError(
                "The assembled consistent mass is not symmetric (largest asymmetry {:e} against entries "
                "up to {:e}); an element wrote something that is not a mass matrix.".format(asymmetry, massScale)
            )

        if np.any(np.asarray(C.data) < 0.0):
            raise ValueError("The assembled damping has negative entries; a damping must dissipate.")

        # A dynamic degree of freedom no element gave any mass -- a zero density, or a node carried
        # only by mass-less entities -- would leave the effective tangent with the stiffness alone
        # there while being integrated as though it had inertia.
        diagonal = M.diagonal()
        massless = dynamicDofs[diagonal[dynamicDofs] <= 0.0]
        if massless.size:
            raise ValueError(
                "{:} of {:} time-integrated degrees of freedom received no mass (first: dof {:}). Every "
                "element carrying a mass field must report a positive density.".format(
                    massless.size, dynamicDofs.size, int(massless[0])
                )
            )

        rowSums = np.asarray(M.sum(axis=1)).ravel()
        for fieldName in dynamicFields:
            indices = self.theDofManager.idcsOfFieldsInDofVector[fieldName]
            totalMass = float(np.sum(rowSums[indices])) / model.nodeFields[fieldName].dimension
            self.journal.message(
                "consistent mass assembled: field '{:}' carries a total mass of {:e}".format(fieldName, totalMass),
                self.identification,
                2,
            )

        return Mvij, Cvij, M, C

    @performancetiming.timeit("initial acceleration")
    def _computeInitialAcceleration(
        self, system: _NewmarkSystem, U_n: DofVector, stepActions: dict, model: FEModel, timeStep: TimeStep
    ):
        """Compute the acceleration from equilibrium at the start of an increment, and commit it.

        Solves :math:`M a_0 = P_\\mathrm{ext}(t_0) - P_\\mathrm{int}(u_0) - C v_0` on the dynamic,
        free degrees of freedom. The loads are evaluated with a synthetic time step at the start of
        the increment passed in (zero step-progress increment and zero time increment -- an ``f(t)``
        amplitude at that instant), the elements with a zero displacement increment, which leaves
        their state untouched. Prescribed degrees of freedom get a zero acceleration; degrees of
        freedom this solver does not integrate get a unit diagonal and a zero right-hand side so the
        system stays regular there.

        Called on the first increment of a cold-started step, where :math:`t_0` is the step's own
        start, and on the increment after a topology change, where it is that increment's start and
        the acceleration it replaces is the one a refinement interpolated -- see
        :meth:`_ensureNewmarkSystem`. The two are the same computation; only the instant differs.

        Parameters
        ----------
        system
            The current Newmark system; its acceleration is overwritten.
        U_n
            The displacement at the start of the increment.
        stepActions
            The step's actions.
        model
            The model tree.
        timeStep
            The increment about to be solved; the start is derived from it.
        """

        startTimeStep = TimeStep(
            timeStep.number,
            0.0,
            timeStep.stepProgress - timeStep.stepProgressIncrement,
            0.0,
            timeStep.stepTime - timeStep.timeIncrement,
            timeStep.totalTime - timeStep.timeIncrement,
        )

        dirichlets = stepActions["dirichlet"].values()
        nodeforces = stepActions["nodeforces"].values()
        distributedLoads = stepActions["distributedload"].values()
        bodyForces = stepActions["bodyforce"].values()

        P = self.theDofManager.constructDofVector()
        F = self.theDofManager.constructDofVector()
        PExt = self.theDofManager.constructDofVector()
        dU0 = self.theDofManager.constructDofVector()
        K = self.theDofManager.constructVIJSystemMatrix()

        P, K, F = self.computeElements(model.elements, U_n, dU0, P, K, F, startTimeStep)
        PExt, K = self.assembleLoads(nodeforces, distributedLoads, bodyForces, U_n, PExt, K, startTimeStep)
        PExt, K = self.assembleConstraints(model.constraints, U_n, dU0, PExt, K, startTimeStep)

        R = self.theDofManager.constructDofVector()
        R[:] = PExt
        R -= P
        R -= system.C @ np.asarray(system.V)

        nDof = self.theDofManager.nDof
        isStatic = np.ones(nDof, dtype=bool)
        isStatic[system.dynamicDofs] = False
        R[isStatic] = 0.0
        MEff = (system.M + diags(isStatic.astype(float), format="csr")).tocsr()

        if self.mpcTransformation is not None:
            R[:] = self.mpcTransformation.transformResidual(R, dU0)
            MEff = self.mpcTransformation.transformSystemMatrix(MEff)

        for dirichlet in dirichlets:
            R[dirichlet.constrainedDofIndices] = 0.0
        MEff = applyDirichletToStiffness(MEff, dirichlets)

        A0 = self.linearSolve(MEff, R)

        system.A[:] = 0.0
        system.A[system.dynamicDofs] = A0[system.dynamicDofs]
        self._publishKinematics(system, model)

        self.journal.message(
            "acceleration from equilibrium at t = {:g}: ||a0||inf = {:e}".format(
                startTimeStep.stepTime,
                float(np.max(np.abs(A0[system.dynamicDofs]))) if system.dynamicDofs.size else 0.0,
            ),
            self.identification,
            2,
        )
