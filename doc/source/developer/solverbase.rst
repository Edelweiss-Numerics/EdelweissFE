The solver base class
=====================

Every nonlinear solver derives from :class:`~edelweissfe.solvers.base.nonlinearsolverbase.NonlinearSolverBase`. It
runs the topology update, builds and renews the discrete system through the one path of
:ref:`how-a-change-selects-the-path`, and declares the two hooks, ``onNewNumbering`` and ``onNewLocations``, that every
solver implements for what it holds. It also assembles the loads and the constraints
(:ref:`residual-and-tangent-convention`), and checks convergence. A solver that holds more than the base class
extends ``forgetDiscreteSystem``.

.. autoclass:: edelweissfe.solvers.base.nonlinearsolverbase.NonlinearSolverBase
   :members:
