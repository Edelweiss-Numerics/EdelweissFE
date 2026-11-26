Models
======

.. _model-discrete-system:

What the solver takes from the discrete system
----------------------------------------------

A solver does not look into the model. It asks the model for a discrete system
(:meth:`~edelweissfe.models.femodel.FEModel.buildDiscreteSystem`) and keeps it until the next change of the numbering,
or until new locations on the same numbering replace it (:ref:`how-a-change-selects-the-path`). From the discrete
system, the solver takes:

* the numbering of the degrees of freedom (:attr:`~edelweissfe.models.discretesystem.DiscreteSystem.numbering`). The
  numbering also gives the degrees of freedom of a field on a node set, e.g., those a Dirichlet boundary condition
  prescribes or a node force loads (:meth:`~edelweissfe.numerics.dofnumbering.DofNumbering.locateFieldOnNodeSet`);
* the locations of the entities (:attr:`~edelweissfe.models.discretesystem.DiscreteSystem.locations`), and the
  vectors created from them;
* the elements (:attr:`~edelweissfe.models.discretesystem.DiscreteSystem.elements`): everything the solver computes
  like an element in ``computeElements``. These are the finite elements that compute kernels; an entity that carries
  only geometry, such as a contact facet, is not included. This holds for every solver, ``NED`` included;
* the constraints that take part (:attr:`~edelweissfe.models.discretesystem.DiscreteSystem.constraints`): the active
  constraints, which the locations contain. The solver evaluates exactly these, in ``assembleConstraints`` (``NED``:
  ``assembleConstraintForces``, without the tangent). Every constraint the solver evaluates therefore has its rows in
  the system matrix, and the solver never asks a constraint whether it is active;
* the entities whose inertia is assembled at the nodes
  (:attr:`~edelweissfe.models.discretesystem.DiscreteSystem.elementsWithNodalInertia`). ``NID`` loops over them for
  the mass and the damping, ``NED`` for the lumped operators, the damping and the critical time step. ``NID`` also
  asks which degrees of freedom carry the mass of the nodes
  (:meth:`~edelweissfe.models.discretesystem.DiscreteSystem.dofsWithNodalInertia`), for the lumped mass of the
  entities that integrate their own inertia
  (:meth:`~edelweissfe.models.discretesystem.DiscreteSystem.assembleLumpedMassOfEntitiesWithOwnInertia`), and, at the
  start of a step, where the model holds their motion
  (:meth:`~edelweissfe.models.discretesystem.DiscreteSystem.carriersOfTheMotion`);
* the entities that carry their own inertia and couple other nodes as they move
  (:attr:`~edelweissfe.models.discretesystem.DiscreteSystem.entitiesWithOwnInertia`), which ``NED`` advances by the
  increment (:meth:`~edelweissfe.models.discretesystem.DiscreteSystem.advanceEntitiesWithOwnInertia`) and evaluates,
  after the model has reconnected them, for their internal force and their lumped momentum
  (:meth:`~edelweissfe.models.discretesystem.DiscreteSystem.assembleInternalForceAndMomentumOfEntitiesWithOwnInertia`)
  and their lumped mass;
* the state from which the solution vector (and the velocity and the acceleration of a dynamic solver) starts after a
  new numbering (:meth:`~edelweissfe.models.discretesystem.DiscreteSystem.readStateFromModel`), and the destination
  of the results of an accepted increment (:meth:`~edelweissfe.models.discretesystem.DiscreteSystem.writeResultsToModel`;
  the velocity and the acceleration through
  :meth:`~edelweissfe.models.discretesystem.DiscreteSystem.writeVelocityAndAccelerationToModel`, see
  :doc:`/output/nodefieldresults`);
* the model for which the linear solver is set up.

In every evaluation of the model, i.e., in every Newton iteration and in every trial state of a line search, ``NIST``
and the solvers derived from it call :meth:`~edelweissfe.models.discretesystem.DiscreteSystem.updateMaterialPoints`
right before they compute the elements (:meth:`~edelweissfe.solvers.nonlinearimplicitstatic.NIST.assembleInternalForces`).
``NED`` integrates no material points and does not call it. The loops over the entities remain with the solver:
``NIST`` computes the elements one after the other, ``NISTParallel`` on all threads, each element into its own part of
the residual and its own block of the system matrix (:ref:`system-matrix-assembly`). A discrete system that updates
material points, or advances and evaluates the entities that carry their own inertia, loops over them through the
solver's :meth:`~edelweissfe.solvers.base.nonlinearsolverbase.NonlinearSolverBase.forEachChunk`, so that they are
evaluated on the same threads.

At the start of every step, the implicit solvers and ``NEST`` give the entities of the model that integrate their own
inertia their Newmark parameters (:meth:`~edelweissfe.models.femodel.FEModel.setNewmarkParametersOfEntities`).
``NID`` gives its own :math:`\beta` and :math:`\gamma` with ``inertia=entities``, and zero with ``inertia=nodes``.
Every quasi-static solver gives zero, so that the entities are quasi-static as well, and so does ``NED``, which
integrates their inertia at the nodes. A step of ``NED`` starts from the
velocity and the acceleration the model holds, and leaves the velocity at its end time
(:meth:`~edelweissfe.models.femodel.FEModel.advanceVelocity`). The steps of a dynamic analysis therefore follow each
other, whichever solver solves them.

A finite element model answers as follows. Its constraints update the nodes they couple, e.g., a contact constraint
searches for the surface it touches. Its discrete system numbers all nodes and the active constraints with their
scalar variables. Its elements are the finite elements. Every node carries its total solution from increment to
increment. Nothing needs to be updated before the elements, which evaluate the material at their quadrature points
themselves.

A model of EdelweissMeshfree answers differently, and is solved by the same solvers:

* Its material points move through a fixed background grid of cells, and its particles carry their kernel functions
  along. Before an increment, it first assigns the material points to the cells they lie in and the particles to the
  kernel functions they reach, and then updates its constraints.
* Only the part of the grid that holds material points, and the nodes reached by the particles, form the discrete
  system (the active domain). A node set of the grid may therefore reach beyond the discrete system: a Dirichlet
  boundary condition then prescribes, and a node force loads, only the nodes of the set in the active domain.
* Its elements are its finite elements, the active cells and the particles. The entities whose inertia is assembled
  at the nodes are its finite elements only, since its cells and cell elements contribute none, and the particles
  integrate their own.
* Only the nodes of finite elements carry a total solution from increment to increment. The material point method and
  RKPM are incremental on the current configuration: the cells are reset, the kernel functions move with their
  particles, and the state lives in the material points and the particles. The values at the nodes of the grid and of
  the kernel functions are coefficients of one increment, and start at zero in every new discrete system.
* Before the cells are computed, each cell interpolates the solution increment to its material points, and the
  material points evaluate their material. The cells then integrate the result.
* Its particles are entities that carry their own inertia. ``NED`` advances them, lets the model reconnect them after
  every increment, and updates its discrete system on the kernel functions they reach then (see the documentation of
  :class:`~edelweissfe.solvers.nonlinearexplicitdynamic.NED`, "Entities that carry their own inertia").

``FEModel`` - A classical finite element model
----------------------------------------------

.. automodule:: edelweissfe.models.femodel
   :members:

``Change`` - What a model changed before an increment
-----------------------------------------------------

The answer of ``FEModel.updateConnectivity`` and of the solvers' topology update, ordered by how much of the discrete
system it invalidates. See :ref:`model-discrete-system`.

.. automodule:: edelweissfe.models.change
   :members:
   :member-order: bysource

``DiscreteSystem`` - What the discrete system of a model is made of
-------------------------------------------------------------------------

What a model answers a solver that (re)builds its discrete system: the numbering of the degrees of freedom, the
locations of the elements and constraints in it, and how the solution passes between the model and the solver's
vectors. See :ref:`model-discrete-system`.

.. automodule:: edelweissfe.models.discretesystem
   :members:

``TopologyPipeline`` - How the mesh may change during a run
-----------------------------------------------------------

Held by every model as ``model.topology``. The model is the mesh and its data; the pipeline owns the
topology window, the number allocators, the fixed-point rounds of the model modifiers, the change log
and mesh-dependent consumers, and the recorded history that a restart replays. See
:doc:`/developer/topologypipeline` for the design.

.. automodule:: edelweissfe.models.topologypipeline
   :members:

``ModelChange`` - The change log of the topology pipeline
---------------------------------------------------------

.. automodule:: edelweissfe.models.modelchange
   :members:

``MeshDependent`` - Consumers that cache mesh-derived state
-----------------------------------------------------------

.. automodule:: edelweissfe.models.meshdependent
   :members:

.. automodule:: edelweissfe.models.modelchangeobserver
   :members:
