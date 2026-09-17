Solvers
=======

.. automodule:: edelweissfe.config.solvers
   :members:


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

``NEDParallel`` - Nonlinear Explicit Dynamic (parallel)
--------------------------------------------------------

.. automodule:: edelweissfe.solvers.nonlinearexplicitdynamicparallel
   :members:

.. pprint:: solver:NEDParallel
