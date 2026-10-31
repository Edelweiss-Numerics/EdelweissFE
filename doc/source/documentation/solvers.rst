Solvers
=======

.. automodule:: edelweissfe.config.solvers
   :members:

.. _residual-and-tangent-convention:

The residual and the tangent
----------------------------

Every entity that enters the equation system -- an element, a load, a constraint -- contributes two things: a
vector :math:`\boldsymbol{P}` to the residual, and its tangent
:math:`\boldsymbol{K} = \partial \boldsymbol{P} / \partial \boldsymbol{U}`. The rule is the same for all of them:

* An **element** adds its internal forces,
  :math:`\boldsymbol{P} = \boldsymbol{f}_\mathrm{int} = \int_\Omega \boldsymbol{B}^T \boldsymbol{\sigma} \, \mathrm{d}V`,
  and its stiffness matrix.
* A **load** adds the *negative* of the external load it applies: a node force :math:`\boldsymbol{F}` adds
  :math:`-\boldsymbol{F}`, a body force :math:`-\int_\Omega \boldsymbol{N}^T \boldsymbol{b} \, \mathrm{d}V`. A dead
  load has no tangent; a load that follows the deformation, such as a pressure on the current surface, has one.
* A **constraint** adds the negative of the force it exerts on its nodes -- just what an element of the same
  mechanics would add as its internal force. A penalty spring of stiffness :math:`k` that pulls a degree of freedom
  :math:`u` towards :math:`\bar{u}` adds :math:`P = k\,(u - \bar{u})` and :math:`K = k`; a Lagrange multiplier
  :math:`\boldsymbol{\lambda}` enforcing :math:`\boldsymbol{g}(\boldsymbol{u}) = \boldsymbol{0}` adds
  :math:`\boldsymbol{\lambda}^T \partial \boldsymbol{g} / \partial \boldsymbol{u}` to the rows of the nodes and
  :math:`\boldsymbol{g}` to the rows of the multipliers. See :doc:`constraints` for writing one.

A solver sums the contributions in three loops of every iteration: ``computeElements`` over the elements,
``assembleLoads`` over the loads -- every load assembles its own contribution, wherever it acts (see
:class:`~edelweissfe.stepactions.base.loadbase.LoadBase`) -- and ``assembleConstraints`` over the constraints that
are active (see :doc:`constraints`). Summed over all entities,

.. math::

    \boldsymbol{G}(\boldsymbol{U}) = \sum \boldsymbol{P} = \boldsymbol{f}_\mathrm{int} - \boldsymbol{f}_\mathrm{ext}

is the out-of-balance force, which equilibrium makes zero, and the summed :math:`\boldsymbol{K}` is its derivative.
Newton's method replaces :math:`\boldsymbol{G}(\boldsymbol{U} + \Delta \boldsymbol{U}) = \boldsymbol{0}` by its
linearization :math:`\boldsymbol{G} + \boldsymbol{K}\, \Delta \boldsymbol{U} = \boldsymbol{0}`, so the implicit
solvers solve

.. math::

    \boldsymbol{K}\, \Delta \boldsymbol{U} = \boldsymbol{R}, \qquad
    \boldsymbol{R} = -\boldsymbol{G} = \boldsymbol{f}_\mathrm{ext} - \boldsymbol{f}_\mathrm{int}

in every iteration. The right-hand side :math:`\boldsymbol{R}` is what the convergence criteria check, and what the
multi-point-constraint condensation and the Dirichlet boundary conditions act on. The other solvers use the same
sums:

* ``NID`` subtracts the inertia and damping forces from :math:`\boldsymbol{R}`, and adds their tangents to
  :math:`\boldsymbol{K}`.
* ``NED`` accelerates the nodes with the net force :math:`\boldsymbol{R}`, and needs no tangent.
* ``NISTPArcLength`` sums the loads twice, at the start of the step (the dead load, :math:`\boldsymbol{P}_0`) and
  at its end; their difference is the reference load, :math:`\boldsymbol{P}_\mathrm{ref}`. Scaled by the load
  parameter :math:`\lambda`, it enters
  :math:`\boldsymbol{R} = -(\boldsymbol{P}_0 + \lambda \boldsymbol{P}_\mathrm{ref} + \boldsymbol{P}_\mathrm{elements}
  + \boldsymbol{P}_\mathrm{constraints})`.

Why one rule? Because then :math:`\boldsymbol{K}` is always the derivative of exactly what was added to
:math:`\boldsymbol{P}`, for every entity alike. That can be checked by finite differences, entity by entity, and the
same check catches a contribution whose sign alone is wrong: :math:`\boldsymbol{K}` then comes out as
:math:`-\partial \boldsymbol{P} / \partial \boldsymbol{U}`. A contribution negated together with its tangent passes it,
so ``tests/test_residual_tangent_convention.py`` checks every constraint this way and also checks the direction of the
forces it exerts (see :doc:`constraints`); a dead load has no tangent, so there the sign of each element load is
checked against the load integrated independently.

The elements' sum alone, :math:`\boldsymbol{f}_\mathrm{int}`, is what the implicit solvers and ``NEST`` report as the
nodal result ``P`` of a field output: at a degree of freedom with a Dirichlet boundary condition and no load, it is the
reaction force. ``NED`` reports its net force :math:`\boldsymbol{R}` there.


``NIST`` - Nonlinear Implicit Static
------------------------------------

.. automodule:: edelweissfe.solvers.nonlinearimplicitstatic
   :members:

.. pprint:: solver:NIST


``NISTParallel`` - Nonlinear Implicit Static (parallel)
-------------------------------------------------------

.. automodule:: edelweissfe.solvers.nonlinearimplicitstaticparallel
   :members:

.. pprint:: solver:NISTParallel

``NISTPArcLength`` - Nonlinear Implicit Static - Arc length
-----------------------------------------------------------

.. automodule:: edelweissfe.solvers.nonlinearimplicitstaticparallelarclength
   :members:

.. pprint:: solver:NISTPArcLength

.. _arc-length-control:

Arc length control
~~~~~~~~~~~~~~~~~~

A load-controlled analysis cannot pass a limit point: beyond the peak of the load, no larger load is in equilibrium.
A displacement-controlled one cannot follow a path that snaps back, where the controlled displacement decreases
again. The arc length method (here the indirect displacement control of Jirásek and Bažant) controls neither the load
nor a displacement directly. The external load of a step becomes

.. math::

    \boldsymbol{f}_\mathrm{ext} = \boldsymbol{f}_0 + \lambda \, \boldsymbol{f}_\mathrm{ref},

the loads at the start of the step (the dead load :math:`\boldsymbol{f}_0`) plus the change of the loads over the
step (the reference load :math:`\boldsymbol{f}_\mathrm{ref}`) scaled by a load parameter :math:`\lambda`, which is an
unknown of every increment besides the solution. A controller adds the equation that determines it, typically that a
combination of displacements -- a crack opening, the mean contraction of a tunnel ring -- grows steadily over the
step. Every Newton iteration solves the linearized system for two right-hand sides at once, the out-of-balance force
and the reference load, and the controller combines the two corrections into the correction of the solution and of
the load parameter. Prescribed displacements are scaled by the load parameter like the loads.

The controller is a step action of the step, e.g. an ``indirectcontrol``, and the solver option
``arcLengthController`` names it (``off``, the default, solves the step without arc length control):

.. code-block:: edelweiss

    *step, solver=theSolver
    maxInc=1e-2, startInc=1e-2, minInc=1e-6, maxNumInc=1000, maxIter=25
    >>options, name=theSolver, arcLengthController=opening
    >>indirectcontrol, name=opening, dof1='model.nodeSets["IDC1"][0].fields["displacement"]', cVector1='-1, 0', dof2='model.nodeSets["IDC2"][0].fields["displacement"]', cVector2='1, 0', L=0.01

From Python, without an input file, the controller is a step action built through its typed constructor, and the
option is a constructor argument of the solver:

.. code-block:: python

    stepActions["indirectcontrol"]["opening"] = IndirectControl("opening", dof1, dof2, [-1.0, 0.0], [1.0, 0.0], 0.01, model, journal)
    solver = NISTPArcLength(jobInfo, journal, arcLengthController="opening")

Every controller derives from :class:`~edelweissfe.stepactions.base.arclengthcontrollerbase.ArcLengthControllerBase`,
which states what the solver asks of it.

The options of Newton's method act on the pair of the solution increment and the increment of the load parameter
:math:`(\Delta U, \Delta\lambda)`:

* ``extrapolation``: both are predicted from the same accepted increments with the same weights, linearly or
  quadratically, so that the prediction stays on one extrapolated path; ``retryWithoutExtrapolation`` retries a failed
  increment from a zero prediction of both.
* ``lineSearch``: a line search scales both corrections with the same step length, so that the trial states stay on
  the Newton direction of the coupled problem. The controller's equation, which a shorter step leaves partly
  unsatisfied, is enforced again in the next iteration.
* ``fallBackToQuasiNewton``: the tangent of the first iteration serves both right-hand sides.

The load parameter is part of the state of the model: every converged increment writes it to
``model.additionalParameters["arc length parameter"]``, where a later step continues from it, also with another
solver.

``NID`` - Nonlinear Implicit Dynamic (Newmark-beta)
----------------------------------------------------

.. automodule:: edelweissfe.solvers.nonlinearimplicitdynamic
   :members:

.. pprint:: solver:NID

The relevant blocks of a deck, abridged from the regression case ``testfiles/marmot/NID`` -- a bar
under a suddenly applied end load, oscillating about its static deflection:

.. code-block:: edelweiss

    *solver, solver=NID, name=theSolver
    newmarkBeta=0.25
    newmarkGamma=0.5

    *fieldOutput
    >>perNode, nSet=gen_right, field=displacement, result=U, name=tipU, f(x)='np.mean(x[:,0])', saveHistory=True
    >>perNode, nSet=gen_right, field=displacement, result=V, name=tipV, f(x)='np.mean(x[:,0])', saveHistory=True
    >>perNode, nSet=gen_right, field=displacement, result=A, name=tipA, f(x)='np.mean(x[:,0])', saveHistory=True

    *step, solver=theSolver
    stepLength=2.0, startInc=0.01, maxInc=0.01, minInc=1e-4, maxNumInc=1000, maxIter=25
    >>dirichlet, name=fixedEnd, nSet=gen_left, field=displacement, 1=0.0
    >>nodeforces, name=endLoad, nSet=gen_right, field=displacement, 1=0.0987, f(t)='1'

The time increment is the step's increment (``stepLength`` times the step-progress increment), so a
constant ``dt`` is obtained with ``startInc = maxInc``; a cutback shrinks it like any other implicit
increment. The velocity and acceleration are available to every ``*fieldOutput`` as ``result=V`` and
``result=A`` on the displacement field, and travel with the ``*output, type=restart`` checkpoints,
which is what makes a resumed run continue the same trajectory rather than restart it from rest.

``NIDParallel`` is the same solver with the element loop evaluated in parallel (see below). The
mass and the damping are assembled at a step's start and after every topology change; when a
contact constraint merely changes its connectivity, they are reused in the rebuilt equation
system instead, which matters for contact problems whose candidate lists change on nearly every
increment.

Reduced-integration elements (``C3D20R``, ``GC3D20R``, ...) need a Marmot whose consistent mass is
integrated with the full rule of the element shape (Marmot PR #101): with the element's own reduced
rule the mass is singular, and the equilibrium solve for the initial acceleration with it.

The material must report a density (for ``LinearElastic``, the third material parameter): the mass
is assembled from it, and a time-integrated degree of freedom that receives none is refused rather
than integrated as though it had inertia.

A ``*modelModifier, type=hAdaptivity`` refining the mesh mid-step is supported: ``V`` and ``A`` are
carried onto new nodes by the node-field warm start, the mass and damping are reassembled on the
refined mesh, and the acceleration is re-solved from equilibrium on the increment that follows --
see :doc:`modelmodifiers`, "Refinement under a dynamic solver", for what is conserved exactly across
such an event and what is not.

Verified in ``tests/test_nid_newmark.py``, ``tests/test_nid_restart.py`` and
``tests/test_nid_amr.py`` against the exact discrete Newmark recurrence of the same problem, against
the continuous closed form at second order, for exact conservation of the discrete energy, for
restart equivalence of ``U``, ``V`` and ``A``, and -- under live refinement -- for exact transfer of
a velocity field of degree at most one onto new nodes together with exact conservation of mass,
momentum and kinetic energy in that case.

``NIDParallel`` - Nonlinear Implicit Dynamic (Newmark-beta, parallel)
----------------------------------------------------------------------

.. automodule:: edelweissfe.solvers.nonlinearimplicitdynamicparallel
   :members:

.. pprint:: solver:NIDParallel

Prediction of an increment in the implicit solvers
--------------------------------------------------

Newton's method starts each increment from a prediction of its solution increment. The solver option
``extrapolation`` chooses it:

* ``off``: from zero, i.e., from the last converged state.
* ``linear`` (default): the last accepted increment, scaled to the size of the new one. The path of the solution is
  continued along its last secant.
* ``quadratic``: the parabola through the last three accepted states, evaluated at the end of the new increment, which
  continues the path along its curvature, too. For a solution that grows quadratically with the load, the prediction
  is exact, and the increment converges without a correction. While only one increment of the step of nonzero size has
  been accepted, the prediction is linear.

Sizes are measured in step progress. The prescribed degrees of freedom take their prescribed increment, so the first
iteration of a predicted increment already checks convergence. Under arc length control, the increment of the load
parameter is predicted from the same accepted increments with the same weights, see :ref:`arc-length-control`.

A prediction can lead Newton's method astray, e.g., when the path turns sharply. Two options decide what happens then
and around rebuilds of the equation system, both off by default:

* ``retryWithoutExtrapolation``: if an extrapolated increment fails, it is attempted once more from a zero prediction
  before it is cut back. A failed increment then keeps the accepted history, and the smaller increment after the
  cutback is extrapolated from it as well. Without this option, a failed increment drops the prediction, and the
  increments up to the next accepted one start from zero.
* ``extrapolateAcrossConnectivityChange``: a change of the connectivity, e.g., a contact search that couples other
  nodes, rebuilds the equation system, and the increment starts from zero. With this option, it is extrapolated
  nevertheless, as long as the rebuilt system numbers the degrees of freedom exactly as the last one
  (:meth:`~edelweissfe.numerics.dofmanager.DofManager.dofNumbering`): the last accepted increments are vectors of the
  same degrees of freedom then. A change of the topology, e.g., a mesh refinement, always starts from zero (see
  ``extrapolateAfterModelChange`` for the increment that follows it).

Both act on a prediction: a declaration of options -- the ``*solver`` datalines, or an ``>>options`` block -- that
switches either on with ``extrapolation=off`` is refused.

Line search and quasi-Newton fallback of the implicit solvers
-------------------------------------------------------------

Newton's method converges quadratically close to the solution, but far from it, or where the tangent changes abruptly
(e.g., material points switching between loading and unloading), a full correction can overshoot, and the iterations
oscillate or diverge. Two remedies, solver options of the implicit solvers, are off by default. Under arc length
control, both act on the pair of the solution and the load parameter, see :ref:`arc-length-control`.

**Line search** (``lineSearch``). The correction :math:`\Delta\Delta U` is scaled by a step length :math:`\alpha`:
the largest residual :math:`\| R(\Delta U + \alpha \, \Delta\Delta U) \|_\infty` (without the Dirichlet degrees of
freedom) is evaluated at the trial step lengths ``lineSearchStepLengths`` (default ``0.8, 1.0, 1.2``), a parabola is
fitted to these values, and :math:`\alpha` minimizes the parabola on :math:`[0.1, 1.5]`: its vertex, limited to that
range, or, for a parabola that does not open upwards, the end of the range where it is smaller. A trial at which the
residual is not finite fits no parabola: :math:`\alpha` is then the trial step length of the smallest finite residual,
and without one, the increment is cut back. Trial residuals that are equal to round-off do not tell the step lengths
apart, and the correction is taken in full. Each trial costs one evaluation of the model, so the line search is done
only after ``lineSearchAfterIterations`` iterations (default 5), and then in every iteration whose number is a multiple
of ``lineSearchEveryIterations`` (default 2). Small trial step lengths, e.g.
``lineSearchStepLengths='0.15, 0.4, 0.7, 1.0'``, damp an iteration that cycles between two states.

**Quasi-Newton fallback** (``fallBackToQuasiNewton``). An increment whose residual grew more often than the step
option ``maxGrowIter`` allows is cut back. With this option, it continues instead with the tangent of its first
iteration, i.e., of the converged state at the start of the increment, or of the prediction if the increment is
extrapolated. The convergence becomes linear, but a tangent that is not updated cannot switch abruptly either;
``maxIter`` still ends an increment that does not converge.

.. code-block:: edelweiss

    >>options, name=theSolver, lineSearch=True, lineSearchAfterIterations=2, lineSearchEveryIterations=1, fallBackToQuasiNewton=True

Convergence criteria of the implicit solvers
--------------------------------------------

The implicit solvers (``NIST``, ``NISTParallel``, ``NISTPArcLength``, ``NID``, ``NIDParallel``) select the
criterion of their Newton iterations with the solver option ``convergenceCriterion``, e.g.:

.. code-block:: edelweiss

    >>options, name=theSolver, convergenceCriterion=abaqus

The options of the ``abaqus`` criterion are named ``abaqus...``; a declaration of options -- the ``*solver`` datalines,
or an ``>>options`` block -- that sets one of them with ``convergenceCriterion=legacy`` is refused, since it would mean
nothing.

The ``legacy`` criterion is configured by tolerances per field and by solver options. EdelweissMeshfree's parameter
set, for instance, reads for a model with a displacement field

.. code-block:: edelweiss

    *updateConfiguration, configuration=fluxResidualTolerance
    displacement=1e-4
    *updateConfiguration, configuration=fluxResidualToleranceAlternative
    displacement=1e-3
    *updateConfiguration, configuration=absoluteFluxResidualTolerance
    displacement=1e-14
    *updateConfiguration, configuration=fieldCorrectionTolerance
    displacement=1e-14
    *updateConfiguration, configuration=relativeFieldCorrectionTolerance
    displacement=1e-3

together with

.. code-block:: edelweiss

    >>options, name=theSolver, iterationsForAlternativeTolerance=5, zeroFluxThreshold=1e-9, zeroIncrementThreshold=1e-13, minimumIterations=1, countFirstResidualAsGrowth=False, countGrowthOfConvergedFields=False

The last two options belong to the test for a diverging Newton iteration, which holds for either criterion: an
increment is abandoned, and retried smaller, once the flux residual of a field has grown more than ``maxGrowIter`` (a
step option) times in it. A growth is a flux residual larger than that of the previous checked iteration, and the two
options decide what else counts:

* ``countFirstResidualAsGrowth`` (default True): the residual history of a field starts at zero, so the first checked
  residual of an increment is a growth already. This is EdelweissFE's original counting; with False, counting starts
  with the second checked iteration.
* ``countGrowthOfConvergedFields`` (default True): a growth counts even in an iteration in which the field is
  converged, and only another field keeps the iterations going. With False, such growths are ignored: the residual of
  a converged field fluctuates at the level of its round-off, which says nothing about divergence.

Independent of the criterion, a residual that contains a NaN ends the increment at once.

.. automodule:: edelweissfe.solvers.base.convergencecriteria
   :members:

Iteration table of the implicit solvers
---------------------------------------

Every Newton iteration writes one row to the console: per field, the largest flux residual and the largest
correction, each marked ``✓`` once the criterion is met. With a linear solver that reports per-solve diagnostics
(e.g. ``blockamg`` at ``verbosity=info``), a ``linear solve`` column follows with the solver's iteration count
and achieved residual. The rows are never wrapped, so a model with many fields gives long lines rather than
broken ones.

.. code-block:: text

          displacement            nonlocal damage              linear solve
       ||R||∞     ||ddU||∞      ||R||∞     ||ddU||∞         iters        ‖r‖
       2.07e+01    8.35e-03     2.12e-02    8.15e-07          14        3.01e-04✓

.. automodule:: edelweissfe.solvers.base.convergencetable
   :members:

Performance table of the implicit solvers
-----------------------------------------

At the end of a step, and every ``report-performance-frequency`` increments if that option is set, the implicit
solvers print where the step's time went (:mod:`~edelweissfe.utils.performancetiming`): one row per timed part,
nested parts indented below their parent, each with its share of the step's wall clock, its number of calls and its
time per call. The table ends with the sum of the top-level rows, the wall clock since the start of the step, and the
share left ``unaccounted``, the time spent in code no timer covers. Every part of an increment is timed, so this share
stays small; a large one points at work that has no row yet.

Besides the element loop and the linear solve, the rows include the parts that cost a lot on large models although no
element is involved:

* ``topology update``: the model modifiers (e.g. ``AMR``), then the ties and the contact catching up;
* ``equation system rebuild``, after a change of the mesh or of a constraint's DOF footprint: ``dof manager``,
  ``linear solver setup``, ``system matrix pattern``, ``mpc transformation build`` and ``mpc dirichlet conflict
  check``;
* ``mpc transform system matrix``: condensing the slave DOFs of the multi-point constraints out of the system;
* ``inertia and damping terms`` (``NID``, ``NIDParallel``), per Newton iteration: ``newmark kinematics``,
  ``inertia and damping forces`` and ``effective tangent``;
* ``reset residual and tangent``, per Newton iteration;
* ``accept increment``, ``finalize increment`` and ``conservation report``, per accepted increment;
* ``output``, per output increment: ``field output`` and one row per output manager, under the name of its
  ``*output`` line.

The timers start from zero together with the step's wall clock, so the rows and the wall clock of a step's table cover
the same time, whatever ran before the step. A resumed run (``*restart, readFrom=...``) whose checkpoint replays a
topology history prints that replay in a table of its own (top-level ``AMR`` rows), before the step's table.

``NEST`` - Nonlinear Explicit Static
-------------------------------------

.. automodule:: edelweissfe.solvers.nonlinearexplicitstatic
   :members:

.. pprint:: solver:NEST

``NESTParallel`` - Nonlinear Explicit Static (parallel)
--------------------------------------------------------

.. automodule:: edelweissfe.solvers.nonlinearexplicitstaticparallel
   :members:

.. pprint:: solver:NESTParallel

``NED`` - Nonlinear Explicit Dynamic
-------------------------------------

.. automodule:: edelweissfe.solvers.nonlinearexplicitdynamic
   :members:

.. pprint:: solver:NED

Rebuilding the equation system during a step
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

An explicit analysis with contact rebuilds its equation system while it runs: a contact search
re-assigns which nodes a constraint couples, which changes that constraint's degree-of-freedom
footprint, and everything sized by the footprints has to follow.

Such a change is much narrower than it looks. It touches no node, no field, no scalar variable and
no element, so the degree-of-freedom numbering and every element's indices stay exactly as they
were -- and with them the lumped inertia, its inverse, the mass-proportional damping rate and the
multi-point-constraint transformation, none of which is a function of the constraints' footprints.
The solver therefore re-locates the constraints alone
(:meth:`~edelweissfe.numerics.dofmanager.DofManager.refreshConstraintIndices`) and keeps those
operators, rather than constructing a new :class:`~edelweissfe.numerics.dofmanager.DofManager` and
assembling them again over every element.

That shortcut is only valid while everything the operators *are* a function of is unchanged, so it
is taken only after checking the element set, the multi-point constraints, the node count and the
scalar variables against what was recorded when they were last assembled, and only when no step
action changes a material property mid-step. Anything that does not match falls back to a full
build. A topology change -- a mesh refinement -- is not a connectivity change and never takes this
path at all: it builds the system from scratch.

Work per increment outside the elements
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

An explicit increment is cheap, so the work around the element loop counts. Second-order degrees
of freedom are updated by the central difference with mass-proportional damping, first-order ones
by forward Euler. The damping rate and a second-order mask are formed once per equation system, so
the update runs on whole vectors instead of indexed subsets, and gives the same result to the bit.

``NEDParallel`` - Nonlinear Explicit Dynamic (parallel)
--------------------------------------------------------

.. automodule:: edelweissfe.solvers.nonlinearexplicitdynamicparallel
   :members:

.. pprint:: solver:NEDParallel
