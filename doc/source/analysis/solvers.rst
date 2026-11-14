Solvers
=======

.. automodule:: edelweissfe.config.solvers
   :members:

Every solver builds its discrete system from the model, and renews it when the model reports a change, e.g. after a
contact search or an h-adaptive refinement; what is renewed depends only on the kind of change. How this works, which
solver asks when, and what it asks of the model, is told once, for developers, in :doc:`/developer/architecture`; the
rule every contribution to the residual and the tangent follows is in :doc:`/developer/residualandtangent`. Below is
what the user needs per solver.

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
the load parameter. The Newton iterations are those of ``NIST``, into which the load parameter enters through the
hooks of the solver (see its class documentation above). Every arc length step scales its own reference load: its load
parameter starts from zero, where the external load is the dead load -- the loads the steps before it ended with. An
analysis that continues under arc length control in a later step declares there the change of the loads it applies on
top of them.

Prescribed displacements belong to the reference loading of the step, like the loads: a Dirichlet boundary condition
that prescribes a change :math:`\Delta\bar{\boldsymbol{u}}` over the step moves its degrees of freedom by
:math:`\lambda \, \Delta\bar{\boldsymbol{u}}`. They move with the load parameter, and only with it. The reference solve
carries :math:`\Delta\bar{\boldsymbol{u}}` in their rows, so a correction :math:`\Delta\Delta\lambda` moves them by
:math:`\Delta\Delta\lambda \, \Delta\bar{\boldsymbol{u}}`; eliminating their columns turns
:math:`\Delta\bar{\boldsymbol{u}}` into the forces it takes to impose it, which join the reference load. The
out-of-balance solve leaves them where they are. So the prescribed degrees of freedom are no unknowns of Newton's
method, and their rows, which hold no equation, are left out of the convergence check, as in ``NIST``. A predicted
increment places them where the predicted load parameter puts them.

At the end of an arc length step, every load keeps the value the step applied, :math:`f(0) + \lambda \, (f(1) - f(0))`,
from which the next step starts. With its amplitude :math:`a(t)`, a load keeps the change it declared for the step
scaled by :math:`a(0) + \lambda \, (a(1) - a(0))` (see :class:`~edelweissfe.stepactions.base.loadbase.LoadBase`): a
load ramped linearly over the step keeps :math:`\lambda` times its change, and a load the step holds at a constant
amplitude, such as a self-weight declared with ``f(t)='1'``, is part of the dead load and keeps its full value,
whatever the load parameter.

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

A step under arc length control may end before its time is up: the solver option ``stopCondition``, an expression of
the model and its field outputs, e.g. ``stopCondition='fieldOutputs["UTop"] < -1.0'``, is checked after every accepted
increment of nonzero size, and the step ends once it is true. A stop condition belongs to the controller it is
declared with: an ``>>options`` block that names ``arcLengthController`` (``off`` too) without a ``stopCondition``
drops the one of an earlier step, so that a later step does not stop by the condition that ended the step before it:

.. code-block:: edelweiss

    >>options, name=theSolver, arcLengthController=opening, stopCondition='fieldOutputs["UTop"] < -1.0'

A stop condition in the ``*solver`` datalines belongs to the solver: every step whose ``>>options`` name a controller
without a stop condition of their own returns to it. An ``>>options`` block that sets a ``stopCondition`` for a step
without arc length control is refused, since the condition would never be checked.

Every controller derives from :class:`~edelweissfe.stepactions.base.arclengthcontrollerbase.ArcLengthControllerBase`,
which states what the solver asks of it:

.. automodule:: edelweissfe.stepactions.base.arclengthcontrollerbase
   :members:

The controllers of EdelweissFE, ``indirectcontrol`` and ``indirectcontractioncontrol``, are indirect displacement
controls: their controlled quantity is a linear combination of degrees of freedom, and their target ``L`` is, with
``absolute`` (the default), the value it reaches at the end of the step, counted from its value in the model when the
controller is declared -- whether for the first time, in any step, or anew in a later one. Without ``absolute``, ``L``
is the change over the step:

.. automodule:: edelweissfe.stepactions.base.indirectdisplacementcontrolbase
   :members:

The options of Newton's method act on the pair of the solution increment and the increment of the load parameter
:math:`(\Delta U, \Delta\lambda)`:

* ``extrapolation``: both are predicted from the same accepted increments with the same weights, linearly or
  quadratically, so that the prediction stays on one extrapolated path; ``retryWithoutExtrapolation`` retries a failed
  increment from a zero prediction of both.
* ``lineSearch``: a line search scales the correction that the out-of-balance force causes, and the controller
  completes every trial state with the correction of the load parameter its equation asks for at that step length, as
  it completes the full correction. So every trial state satisfies the controller's equation, and so does the iterate
  the line search chooses: the line search cannot leave the equation unsatisfied, also not from an iterate that does
  not satisfy it yet.
* ``fallBackToQuasiNewton``: the tangent of the first iteration serves both right-hand sides.

Every accepted increment writes the load parameter of its step to the model, with its results, as
``model.additionalParameters["arc length parameter"]``, e.g. for a load-displacement curve. The solver checkpoints the
load parameter of the step and the increments of it its predictor extrapolates, the loads what they keep from step to
step, and the controller the change of its controlled quantity over the step, so an arc length controlled run resumes
exactly where it stopped. The load parameter advances only when an increment has converged, with its solution, so a
cut back increment starts again from the load parameter of the last accepted one.

The re-equilibration after a model change (``equilibrateAfterModelChange``, see :doc:`/analysis/modelmodifiers`) is, under arc
length control, an arc length increment of size zero: the controller asks for a change of zero, so the controlled
quantity stays where the last accepted increment left it, and the load parameter is solved for, and with it the loads
and the prescribed displacements it scales -- the counterpart of a zero-time increment under displacement control,
which, unlike a constant load, has a solution past a limit point. As for any increment, the stop condition is checked
before it, and the controller's ``finishIncrement`` is called once it has converged. The load parameter it reaches is
committed with its state, and the next increment starts from it.

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

``NIDParallel`` is the same solver with the element loop evaluated in parallel (see below). Both assemble the
system matrix either way (``systemMatrixAssembly=vij|direct``, see :ref:`system-matrix-assembly`): the mass is
assembled through the same assembly as the stiffness, kept on its sparsity pattern, and added to the effective tangent
while it is assembled, together with the lumped damping. The
mass and the damping are assembled at a step's start and after every change of the numbering; when a
contact constraint merely changes its connectivity, the new locations keep them (the mass copied onto the new
sparsity pattern), which matters for contact problems whose candidate lists change on nearly every increment.

Reduced-integration elements (``C3D20R``, ``GC3D20R``, ...) need a Marmot whose consistent mass is
integrated with the full rule of the element shape (Marmot PR #101): with the element's own reduced
rule the mass is singular, and the equilibrium solve for the initial acceleration with it.

The material must report a density (for ``LinearElastic``, the third material parameter): the mass
is assembled from it, and a time-integrated degree of freedom that receives none is refused rather
than integrated as though it had inertia.

A model whose entities carry their state, and with it their inertia, in themselves is solved through the same
discrete system: these entities integrate their own inertia with the solver's Newmark parameters (``inertia=entities``, the
default), or the solver integrates the nodes they couple with their lumped mass (``inertia=nodes``). The module
documentation above states what each one does; material points that belong to no element are refused.

A ``*modelModifier, type=hAdaptivity`` refining the mesh mid-step is supported: ``V`` and ``A`` are
carried onto new nodes by the node-field warm start, the mass and damping are reassembled on the
refined mesh, and the acceleration is re-solved from equilibrium on the increment that follows --
see :doc:`/analysis/modelmodifiers`, "Refinement under a dynamic solver", for what is conserved exactly across
such an event and what is not.

Verified in ``tests/test_nid_newmark.py``, ``tests/test_nid_restart.py`` and
``tests/test_nid_amr.py`` against the exact discrete Newmark recurrence of the same problem, against
the continuous closed form at second order, for exact conservation of the discrete energy, for
restart equivalence of ``U``, ``V`` and ``A``, and -- under live refinement -- for exact transfer of
a velocity field of degree at most one onto new nodes together with exact conservation of mass,
momentum and kinetic energy in that case. Every one of these tests runs with both ways of summing the system matrix,
and ``tests/test_nid_assemblies_agree.py`` compares the two to round-off.

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
and around rebuilds of the discrete system, both off by default:

* ``retryWithoutExtrapolation``: if an extrapolated increment fails, it is attempted once more from a zero prediction
  before it is cut back. A failed increment then keeps the accepted history, and the smaller increment after the
  cutback is extrapolated from it as well. Without this option, a failed increment drops the prediction, and the
  increments up to the next accepted one start from zero.
* ``extrapolateAcrossConnectivityChange``: a change of the connectivity, e.g., a contact search that couples other
  nodes, gives the discrete system new locations, and the increment starts from zero. With this option, it is
  extrapolated nevertheless, as long as the new discrete system numbers the degrees of freedom exactly as the one the
  last increments were accepted in -- the numbering is the same object for new locations, and a new numbering is
  compared by the digests of the two numberings, :attr:`~edelweissfe.numerics.dofnumbering.DofNumbering.digest`: the
  last accepted increments are vectors of the same degrees of freedom then. A change of the topology, e.g., a mesh
  refinement, always starts from zero (see ``extrapolateAfterModelChange`` for the increment that follows it).

Both act on a prediction: a declaration of options -- the ``*solver`` datalines, or an ``>>options`` block -- that
switches either on with ``extrapolation=off`` is refused.

The last accepted increments are checkpointed with the digest of their numbering. A step that continues from a
restart checkpoint, resumed or fallen back to (see :doc:`/developer/restart`), builds its discrete system at its first
increment, and extrapolates them only if the new numbering numbers the degrees of freedom alike: the numbering of a
meshfree model, e.g., follows its active cells, which may have changed since the checkpoint was written.

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
* ``discrete system rebuild``, after a change (:class:`~edelweissfe.models.change.Change`): ``numbering and locations``,
  and, for a new numbering, ``linear solver setup``, ``mpc transformation build`` and ``mpc dirichlet conflict check``;
  for new locations as well, ``system matrix pattern``;
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

Contact searches during a step
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

An explicit analysis with contact searches while it runs, every ``contact-update-frequency`` increments. A search
re-assigns which nodes a contact constraint couples, and touches no node, no field, no scalar variable and no element:
the change is ``LOCATIONS``, ``NED`` only wraps its vectors with the new locations, and the lumped inertia, the
mass-proportional damping rate, the multi-point-constraint condensation and the stable time increment are kept
(:meth:`~edelweissfe.solvers.nonlinearexplicitdynamic.NED.onNewLocations`, and :ref:`how-a-change-selects-the-path`).

A change of the mesh -- a refinement -- gives a new numbering
(:meth:`~edelweissfe.solvers.nonlinearexplicitdynamic.NED.onNewNumbering`): the operators are assembled anew over the
elements of the new discrete system, the solution, the velocity and the acceleration are read from the model, which
interpolated them onto the new nodes, and the stable time increment is computed again.

Work per increment outside the elements
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

An explicit increment is cheap, so the work around the element loop counts. Second-order degrees
of freedom are updated by the central difference with mass-proportional damping, first-order ones
by forward Euler. The damping rate and a second-order mask are formed once per discrete system, so
the update runs on whole vectors instead of indexed subsets, and gives the same result to the bit.

``NEDParallel`` - Nonlinear Explicit Dynamic (parallel)
--------------------------------------------------------

.. automodule:: edelweissfe.solvers.nonlinearexplicitdynamicparallel
   :members:

.. pprint:: solver:NEDParallel
