Linear solvers
==============

A nonlinear solver solves one linear system per Newton iteration. Which linear solver it uses is its ``linsolver``
option: the name of a ``*linearSolver`` block, or a type of linear solver with its defaults.

.. code-block:: edelweiss

    *solver, solver=NIST, name=theSolver
    linsolver=pardiso

A ``*linearSolver`` block defines a linear solver with options, as ``*material`` defines a material: the keyword line
holds only its identity, ``name`` and ``type``; the data lines are the options of that type, one ``key=value`` per line
(an option the type does not have is an error):

.. code-block:: edelweiss

    *linearSolver, name=fast, type=blockamg
    rigidBodyCoarseLevel=false
    verbosity=info

    *solver, solver=NIST, name=theSolver
    linsolver=fast

A type name in ``linsolver=`` means that type with its defaults, so a block must not be named like a type.
``linsolver`` unset means ``pardiso`` if it was built, else ``superlu``. From Python, the same linear solver is
``getLinSolverByName("blockamg", {"rigidBodyCoarseLevel": False})`` (:mod:`edelweissfe.config.linsolve`).

.. list-table:: The types of linear solver
    :width: 100%
    :widths: 15 10 45
    :header-rows: 1

    * - Type
      - Direct
      - What it is
    * - ``superlu``
      - ✓
      - SciPy's sparse LU (SuperLU), always available
    * - ``umfpack``
      - ✓
      - UMFPACK through SciPy (SuperLU if ``scikit-umfpack`` is missing)
    * - ``pardiso``
      - ✓
      - Intel MKL PARDISO (``edelweissfe.linsolve.pardiso.pardiso``), the default
    * - ``panuapardiso``
      - ✓
      - Panua PARDISO (``edelweissfe.linsolve.panuapardiso.panuapardiso``)
    * - ``klu``
      - ✓
      - SuiteSparse's KLU (``edelweissfe.linsolve.klu.klu``)
    * - ``petsclu``
      - ✓
      - PETSc's LU, through ``petsc4py``
    * - ``mumps``
      - ✓
      - MUMPS, through its Python bindings
    * - ``blockamg``
      - ✗
      - the preset ``blockAMG()``: LGMRES around a block Gauss--Seidel sweep over the fields, with one AMG hierarchy
        (or two-level cycle) per field
    * - ``custom``
      - —
      - a composed solver defined by you, in a Python script (see :doc:`composedlinsolvers`)

Direct or iterative
-------------------

**Direct solvers** factorize the matrix and back-substitute: ``superlu`` and ``umfpack`` (SciPy, always available),
``pardiso`` (Intel MKL, the production default), ``panuapardiso``, ``klu``, ``petsclu`` and ``mumps``. Pick one of
these unless the factorization has become the run's bottleneck or exceeds available memory. A direct solver keeps its
backend for the whole step: PARDISO with ``reuseSymbolicFactorization=true`` keeps its reordering and symbolic
factorization while only the values of the matrix change. A singular matrix gives NaN, which the nonlinear solver
turns into a cutback.

**Iterative solvers** never factorize the full matrix, trading exactness for O(n) memory: past roughly a million DOFs
the fill-in of a direct factorization exceeds memory, whereas algebraic multigrid stays linear. ``blockamg`` is the
field-split solver for **large coupled multi-field systems** (e.g. displacement + gradient-enhanced damage): one AMG
hierarchy per physical field, combined by a block Gauss--Seidel sweep to precondition an outer LGMRES. The block
structure -- which DOFs belong to which field, and each field's dimension -- is discovered automatically from the
model. It requires the optional ``amgcl`` extension.

.. code-block:: edelweiss

    *linearSolver, name=fieldSplit, type=blockamg
    sweeps=1
    symmetric=false

    *solver, solver=NIST, name=theSolver
    linsolver=fieldSplit

A ``custom`` solver script composes any other tree: a Schur complement that condenses the dofs on which AMG alone
stalls (bars tied to their host elements by bond springs, a softening zone), other smoothers, other coarse spaces.
How to write one, and the tree of ``blockamg``, are on :doc:`composedlinsolvers`; why the field-split method works,
and each of its steps, on :doc:`blockamgtheory`.

One adapter for every type
--------------------------

Every type is a composed solver (:doc:`composedlinsolvers`) behind one adapter,
:class:`~edelweissfe.linsolve.composition.solver.ComposedLinearSolver`: ``blockamg`` is the tree of the ``blockAMG()``
preset, and a direct solver is a tree of one leaf, ``type=pardiso`` being ``DirectSolver("pardiso")``. What the adapter
does therefore works the same for every type, with the same data lines
(:class:`~edelweissfe.linsolve.types.LinearSolverSchema`, described in :ref:`composed-adapter`):

.. list-table:: The data lines of every type
    :width: 100%
    :widths: 25 20 55
    :header-rows: 1

    * - Data line
      - Default
      - What it sets
    * - ``verbosity``
      - ``warning``
      - ``silent``, ``warning``, ``info`` or ``debug`` (:ref:`composed-verbosity`)
    * - ``warnOuterIterations``
      - 100
      - a Krylov solve with more outer iterations is warned about, and is a slow solve
    * - ``residualGrowthFactor``
      - 4
      - a right-hand side this many times larger than the previous one marks a new increment
    * - ``dumps``, ``dumpDirectory``, ``maxDumps``, ``dumpContextSolves``
      - ``none``, ``linsolveDumps``, 10, 0
      - which solves are written to disk, as a sequence of authentic systems for an offline replay
        (:ref:`composed-solve-dumps`)

The options of each type
------------------------

.. pprint:: linsolver:pardiso
    :caption: ``pardiso``:

The other direct solvers take only the common data lines above.

.. pprint:: linsolver:blockamg
    :caption: ``blockamg``:

.. pprint:: linsolver:custom
    :caption: ``custom``:

The pages on linear solvers
---------------------------

.. toctree::
   :maxdepth: 1

   composedlinsolvers
   composedlinsolversreference
   blockamgtheory
   fieldsplitmeasurements
   linsolveapi
