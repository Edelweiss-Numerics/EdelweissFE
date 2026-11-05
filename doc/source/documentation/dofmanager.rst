Degree of Freedom (DOF) Manager and Numerics
============================================

The ``edelweissfe.numerics`` package provides core data structures and managers for global equation systems in EdelweissFE.

The DOF manager numbers the degrees of freedom: every node field, scalar variable and entity gets its indices in the
global DOF vector. It knows nothing about the system matrix. How the blocks of the entities -- each given in the
entity's own numbering by :class:`~edelweissfe.numerics.systemmatrixcontributor.SystemMatrixContributorBase` -- become
the global matrix is the business of the system matrix assembly, see :ref:`system-matrix-assembly`.

DofManager
----------
.. automodule:: edelweissfe.numerics.dofmanager
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
