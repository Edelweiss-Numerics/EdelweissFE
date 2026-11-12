Degrees of freedom: numbering and locations
===========================================

The ``edelweissfe.numerics`` package provides core data structures for global discrete systems in EdelweissFE.

A discrete system has two lifetimes, and each has its own immutable object. The words below are used in the code, the
docstrings and these pages in exactly this meaning.

.. list-table::
   :header-rows: 1
   :widths: 18 52 30

   * - Word
     - Meaning
     - Object
   * - connectivity
     - Which nodes an entity touches (for a contact: which it couples now). The *IEN* array of the textbooks.
     - the entities of the model
   * - numbering
     - The global index of every degree of freedom of a node field or a scalar variable. The *ID* array of Hughes --
       but every degree of freedom is numbered, a prescribed one included.
     - :class:`~edelweissfe.numerics.dofnumbering.DofNumbering`
   * - location of an entity
     - The global indices of the entity's local degrees of freedom, ``locations[entity]``: never a position in
       space. The *LM* array: the ID array applied to the IEN array.
     - :class:`~edelweissfe.numerics.doflocations.DofLocations`
   * - indices
     - Any other set of global indices: a node set's, the Dirichlet degrees of freedom, the slaves of a multi-point
       constraint.
     - whoever uses them

The numbering changes with the mesh, and when scalar variables (e.g. the Lagrange multipliers of constraints) enter or
leave the discrete system. The locations change when the connectivity changes: a contact search couples other nodes,
and the numbering stays. Neither object is ever changed: a change makes a new one. A new
:class:`~edelweissfe.numerics.doflocations.DofLocations` on the same
:class:`~edelweissfe.numerics.dofnumbering.DofNumbering` is what
:meth:`~edelweissfe.models.discretesystem.DiscreteSystem.withNewLocations` gives; a change of the numbering needs a
new discrete system, built by the model.

Every entity of the discrete system is located. An entity that couples a field its node does not have, or a scalar
variable the numbering does not number, is an error of the model: it is reported, naming the entity and what is
missing, when the locations are made.

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
