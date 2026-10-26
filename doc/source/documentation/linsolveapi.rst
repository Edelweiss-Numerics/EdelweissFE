Linear solvers: Python API
==========================

The classes and functions of the linear solvers. How to choose a linear solver is on :doc:`linsolvers`, how to compose
one on :doc:`composedlinsolvers`, the arguments and data lines at a glance on :doc:`composedlinsolversreference`.

The types and the configuration
-------------------------------

.. automodule:: edelweissfe.linsolve.types
    :members:

.. automodule:: edelweissfe.config.linsolve
    :members: getLinSolverByName, getDefaultLinSolver, createLinearSolver, linearSolverDefinitionsFromInputFile

.. automodule:: edelweissfe.linsolve.base
    :members: LinearSolver, DirectSolverBackend

The components
--------------

.. automodule:: edelweissfe.linsolve.composition.base
    :members:

.. automodule:: edelweissfe.linsolve.composition.krylov
    :members:

.. automodule:: edelweissfe.linsolve.composition.diagonalscaling
    :members:

.. automodule:: edelweissfe.linsolve.composition.amg
    :members:

.. automodule:: edelweissfe.linsolve.composition.twolevel
    :members:

.. automodule:: edelweissfe.linsolve.composition.schurcomplement
    :members:

.. automodule:: edelweissfe.linsolve.composition.direct
    :members:

.. automodule:: edelweissfe.linsolve.composition.combinators
    :members:

.. automodule:: edelweissfe.linsolve.composition.presets
    :members:

.. automodule:: edelweissfe.linsolve.composition.views
    :members:

The machinery
-------------

The adapter, the matrix the tree is set up on, the dumps, and the native cycles behind ``TwoLevel``.

.. automodule:: edelweissfe.linsolve.composition.systemmatrix
    :members:

.. automodule:: edelweissfe.linsolve.composition.solver
    :members:

.. automodule:: edelweissfe.linsolve.composition.solvedumps
    :members:

.. automodule:: edelweissfe.linsolve.composition.twolevelcycle
    :members:

.. automodule:: edelweissfe.linsolve.composition.rigidbodycoarselevel
    :members:

.. automodule:: edelweissfe.linsolve.composition.ptwogrid
    :members:
