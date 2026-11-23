The solver base class
=====================

Every nonlinear solver derives from :class:`~edelweissfe.solvers.base.nonlinearsolverbase.NonlinearSolverBase`. It
runs the topology update, builds and renews the discrete system through the one path of
:ref:`how-a-change-selects-the-path`, and declares the two hooks, ``onNewNumbering`` and ``onNewLocations``, that every
solver implements for what it holds. It also assembles the loads and the constraints
(:ref:`residual-and-tangent-convention`), and checks convergence. A solver that holds more than the base class
extends ``forgetDiscreteSystem``.

.. _solver-discrete-system-updates:

When each solver updates its discrete system
--------------------------------------------

All solvers follow a change through the same method,
:meth:`~edelweissfe.solvers.base.nonlinearsolverbase.NonlinearSolverBase.updateDiscreteSystem`, but they run the
topology update at different times and use different parts of the discrete system:

.. list-table::
    :width: 100%
    :widths: 22 26 26 26
    :header-rows: 1

    * - Solver
      - System matrix
      - Linear solver
      - Updates its discrete system
    * - ``NIST``, ``NISTParallel``
      - tangent, ``vij`` or ``direct``
      - once per Newton iteration
      - at the start of every increment and of every retry, by the ``Change``
    * - ``NISTPArcLength``
      - tangent, plus the tangents of the dead and the reference load, scaled
      - for two right-hand sides
      - as ``NIST``
    * - ``NID``, ``NIDParallel``
      - tangent plus the mass (on the pattern of the tangent) and the damping
      - once per Newton iteration
      - as ``NIST``. New locations keep the mass and the damping, which the first increment on a new numbering
        assembles. With ``inertia=nodes``, the lumped mass of the entities that integrate their own inertia is
        assembled anew in every increment.
    * - ``NEST``, ``NESTParallel``
      - tangent, ``vij``
      - once per Runge--Kutta stage
      - once per step, in ``beginStep``; no topology update during the step
    * - ``NED``, ``NEDParallel``
      - none; lumped mass and damping
      - none
      - at the start of the step; after every ``topology-check-frequency``-th increment, if a model modifier acts
        after the start; a contact search every ``contact-update-frequency``-th increment; with entities that carry
        their own inertia, the connectivity after every increment

``NED`` runs the full topology update at the start of its step. During the step, it runs it again after every
``topology-check-frequency``-th increment, but only if a model modifier of the model acts after the start of the
analysis; the default of ``topology-check-frequency`` is zero, i.e., never. Every ``contact-update-frequency``-th
increment, it calls :meth:`~edelweissfe.models.femodel.FEModel.updateConnectivity` alone, but only if a constraint of
the model searches its partner as the bodies move
(:attr:`~edelweissfe.nodecouplingentity.base.nodecouplingentity.BaseNodeCouplingEntity.connectivityChangesWithMotion`,
e.g., a contact constraint). Such a contact search can only result in new locations, because ``NED`` refuses
constraints with scalar variables. In a model with entities that carry their own inertia, e.g. the particles of a
meshfree model, ``NED`` calls :meth:`~edelweissfe.models.femodel.FEModel.updateConnectivity` after every increment
instead, once the model has accepted it: the change may be a new numbering (the active nodes of the meshfree model
changed), which gives new vectors read from the model, and an element mass assembled anew, but neither a new stable
time increment nor a new classification of the fields, which belong to the mesh. ``NEST`` never updates the connectivity during a step, and refuses a constraint
whose connectivity changes with the motion.

``NIST`` predicts the first iterate of an increment by extrapolating the last accepted increments. Once per increment,
right after the update of the discrete system, it decides whether the prediction may be used
(:meth:`~edelweissfe.solvers.nonlinearimplicitstatic.NIST.predictionSurvives`). After a change of the mesh, the
prediction is always dropped. After any other change of the connectivity (``LOCATIONS`` or ``NUMBERING``), it is
dropped unless the option ``extrapolateAcrossConnectivityChange`` is set. With this option, it is used if the new
numbering numbers the degrees of freedom as the old one did, which new locations on the same numbering always do.

The class
---------

.. autoclass:: edelweissfe.solvers.base.nonlinearsolverbase.NonlinearSolverBase
   :members:
