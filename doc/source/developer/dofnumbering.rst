Degrees of freedom: numbering and locations
===========================================

The ``edelweissfe.numerics`` package provides core data structures for global discrete systems in EdelweissFE.

This page is the API reference of the numbering, the locations and what refers to them. The words *connectivity*,
*numbering*, *location* and *indices*, the lifetimes of the two objects and what a change of the model makes new are
told once, in :ref:`architecture-words` and :ref:`how-a-change-selects-the-path`. In short: the numbering is a
:class:`~edelweissfe.numerics.dofnumbering.DofNumbering`, the locations are a
:class:`~edelweissfe.numerics.doflocations.DofLocations`, and neither is ever changed: a change makes a new one
(:meth:`~edelweissfe.models.discretesystem.DiscreteSystem.withNewLocations` for new locations on the same numbering,
:meth:`~edelweissfe.models.femodel.FEModel.buildDiscreteSystem` for a new numbering).

Every entity of the discrete system is located by one protocol (:ref:`architecture-entities`): a finite element, a constraint, and,
in EdelweissMeshfree, a cell, a cell element or a particle. The protocol,
:class:`~edelweissfe.nodecouplingentity.base.nodecouplingentity.BaseNodeCouplingEntity`, declares what the locations
read: the nodes the entity couples (``nodes``), the fields it couples on each of them (``fields``), the permutation of
its local degrees of freedom (``dofIndicesPermutation``) and the scalar variables it owns (``scalarVariables``); and
whether its nodes change with the motion (``connectivityChangesWithMotion``), which decides what is counted again for
new locations. It does not declare inertia: which entities a solver asks for it is
:attr:`~edelweissfe.models.discretesystem.DiscreteSystem.elementsWithNodalInertia`. A test of each repository checks that
the compiled wrappers, which cannot derive from it, have every member. An entity that couples a field its node does not
have, or a scalar variable the numbering does not number, is an error of the model: it is reported, naming the entity
and what is missing, when the locations are made.

``locations[entity]`` is the one way to read the location of a single entity, ``entity in locations`` tells whether it
has one; the consumers that handle all entities at once (the system matrix assembly, the layout of the scatter
vectors) read the mapping of all of them, :attr:`~edelweissfe.numerics.doflocations.DofLocations.ofEntity`. The number
of accumulated fluxes per field, which the convergence criteria compare the residual with, is computed from the
entities of the locations when it is first asked for, so that new locations always have correct counts and a solver
without convergence criteria never pays; the entities whose nodes stay as they are (the finite elements) are counted
once for a numbering, only the constraints and the entities whose nodes change with the motion (particles) are counted
again for new locations. A vector (:class:`~edelweissfe.numerics.dofvector.DofVector`) refers to the
locations it was wrapped with; for new locations on the same numbering it is wrapped on them, its values kept
(:meth:`~edelweissfe.numerics.dofvector.DofVector.onLocations`).

The numbering knows nothing about the entities, and neither of the two knows anything about the system matrix. How
the blocks of the entities -- each given in the entity's own numbering by
:class:`~edelweissfe.numerics.systemmatrixcontributor.SystemMatrixContributorBase` -- become the global matrix is the
business of the system matrix assembly, which reads the locations, see :ref:`system-matrix-assembly`.

DofNumbering
------------
.. automodule:: edelweissfe.numerics.dofnumbering
   :members:
   :undoc-members:
   :show-inheritance:

DofLocations
------------
.. automodule:: edelweissfe.numerics.doflocations
   :members:
   :undoc-members:
   :show-inheritance:

The entities
------------
.. automodule:: edelweissfe.nodecouplingentity.base.nodecouplingentity
   :members:
   :undoc-members:
   :show-inheritance:

DofVector
---------
.. automodule:: edelweissfe.numerics.dofvector
   :members:
   :undoc-members:
   :show-inheritance:

ScatterDofVector
----------------
.. automodule:: edelweissfe.numerics.scatterdofvector
   :members:
   :undoc-members:
   :show-inheritance:

VIJSystemMatrix
---------------
.. automodule:: edelweissfe.numerics.vijsystemmatrix
   :members:
   :undoc-members:
   :show-inheritance:

SystemMatrixContributorBase
---------------------------
.. automodule:: edelweissfe.numerics.systemmatrixcontributor
   :members:
   :undoc-members:
   :show-inheritance:

Parallelization Utilities
-------------------------
.. automodule:: edelweissfe.numerics.parallelizationutilities
   :members:
   :undoc-members:
   :show-inheritance:
