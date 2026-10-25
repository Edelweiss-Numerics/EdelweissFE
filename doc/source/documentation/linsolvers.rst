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
        (or two-grid cycle) per field
    * - ``custom``
      - —
      - a composed solver defined by you, in a Python script (see :doc:`composedlinsolvers`)

One adapter for every type
--------------------------

Every type is a composed solver (:doc:`composedlinsolvers`) behind one adapter,
:class:`~edelweissfe.linsolve.composition.solver.ComposedLinearSolver`: ``blockamg`` is the tree of the ``blockAMG()``
preset, and a direct solver is a tree of one leaf, ``type=pardiso`` being ``DirectSolver("pardiso")``. What the adapter
does therefore works the same for every type, with the same data lines (:class:`~edelweissfe.linsolve.types.LinearSolverSchema`):

* ``verbosity``: ``warning`` (default; a warning for a slow or unconverged solve), ``silent``, ``info`` (one line per
  solve, the structure of the solver, and for a Krylov solver the "linear solve" column of the nonlinear solver's
  table) or ``debug``;
* ``warnOuterIterations``: a Krylov solve with more outer iterations than this (default 100) is warned about, and is a
  slow solve;
* ``residualGrowthFactor``: a right-hand side this many times larger than the previous one marks a new increment
  (default 4);
* ``dumps``: which solves are written to disk for an offline replay -- ``none`` (default), ``slow`` (above
  ``warnOuterIterations``, or stopped at the iteration cap short of the tolerance), ``every``, or the numbers of the
  solves separated by spaces, e.g. ``dumps=12 13 14``; with ``dumpDirectory`` (default ``linsolveDumps``), ``maxDumps``
  (default 10, a disk guard for the whole run) and ``dumpContextSolves`` (the solves before a dumped one, dumped with
  it). The format is described in :ref:`composed-solve-dumps`; ``scripts/benchmark_linsolve.py`` replays a dumped
  sequence through direct solver variants.

The dumps are what ``matrixdump`` did before (a sequence of authentic systems, replayed offline so that every solver
variant sees the same input), with the trigger of the former degradation dumps (``slow``).

A direct solver keeps its backend for the whole step: PARDISO with ``reuseSymbolicFactorization=true`` keeps its
reordering and symbolic factorization while only the values of the matrix change. A singular matrix gives NaN, which
the nonlinear solver turns into a cutback.

How the types relate
--------------------

**Direct solvers** factorize the matrix and back-substitute: ``superlu`` and ``umfpack`` (SciPy, always available),
``pardiso`` (Intel MKL, the production default), ``panuapardiso``, ``klu``, ``petsclu`` and ``mumps``. Pick one of
these unless the factorization has become the run's bottleneck or exceeds available memory.

**Iterative solvers** never factorize the full matrix, trading exactness for O(n) memory. ``blockamg`` is the
field-split solver for coupled multi-field models: one AMG hierarchy per physical field (e.g. displacement, nonlocal
damage), combined by a block Gauss--Seidel sweep to precondition an outer LGMRES. This is the route to problem sizes a
direct factorization cannot reach at all. A ``custom`` solver script composes any other tree: a Schur complement that
condenses the dofs on which AMG alone stalls (bars tied to their host elements by bond springs, a softening zone), other
smoothers, other coarse spaces.

The options of each type
------------------------

.. pprint:: linsolver:pardiso
    :caption: ``pardiso``:

The other direct solvers take only the common data lines above.

.. pprint:: linsolver:blockamg
    :caption: ``blockamg``:

.. pprint:: linsolver:custom
    :caption: ``custom``:

The ``blockamg`` solver
-----------------------

``blockamg`` is a field-split block-AMG solver for **large coupled multi-field systems** (e.g. displacement +
gradient-enhanced damage). It is the O(n)-memory route to problem sizes a direct factorization cannot reach -- past
roughly a million DOFs its fill-in exceeds memory, whereas algebraic multigrid stays linear.

Applied *monolithically*, AMG is ineffective on such a coupled system (a single hierarchy cannot represent the
disparate fields at once -- their physical scales and near-null-spaces differ). ``blockamg`` instead builds **one AMGCL
algebraic-multigrid hierarchy per field** and combines them with a **block Gauss--Seidel** sweep to precondition an
outer Krylov solve. Per solve it scales the system by its diagonal, splits it into field blocks, builds or reuses the
per-field hierarchies, and preconditions the outer solve with the block sweep. The block structure -- which DOFs belong
to which field, and each field's dimension -- is **discovered automatically** from the model. Requires the optional
``amgcl`` extension.

.. code-block:: edelweiss

    *linearSolver, name=fieldSplit, type=blockamg
    sweeps=1
    symmetric=false

    *solver, solver=NIST, name=theSolver
    linsolver=fieldSplit

Its tree, and how to change what its data lines do not reach, are on :doc:`composedlinsolvers`; the method, the
near-null-space construction, the adaptive outer tolerance, hierarchy reuse, the p-multigrid variant, the rigid-body
coarse level and the Schur complement condensation are explained in detail on :doc:`blockamgtheory`.

.. toctree::
   :maxdepth: 1

   composedlinsolvers
   blockamgtheory

Rigid-body coarse level (``blockamg``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The hierarchy of a 3D displacement field on node-wise blocks has a coarse space of the **translations** of every
aggregate of nodes, not their rotations. In a bending-dominated structure -- a slender beam, a tunnel lining, a plate --
the slowly converging error is bending, under which every aggregate rotates almost as a rigid body: neither that coarse
space nor the smoother reduces it, and the outer solve needs many iterations. So by default ``blockamg`` gives the 3D
displacement field a two-grid cycle whose coarse level is built from all rigid-body modes, translations and rotations
(``TwoGrid(coarseSpace=RigidBodyModes())``). The testfile deck
``testfiles/edelweiss-only/CantileverBeamHexa20BlockAMGRigidBody``, a slender cantilever, shows the effect: each linear
solve converges in some 20 outer iterations; with the coarse level switched off, every solve stops at the iteration
cap. On an elastic tunnel segment of quadratic hexahedra, it cut the outer iterations of the displacement block from 67
to 22.

Switch it off where the solves converge in a few outer iterations without it -- there it saves little, for a larger
setup (on an edge breakout model 22 against 20 iterations, at about the same cost):

.. code-block:: edelweiss

    *linearSolver, name=translationsOnly, type=blockamg
    rigidBodyCoarseLevel=false

A 2D displacement field, or any other vector field, gets the coarse level in a ``custom`` solver script,
``OnField("displacement", TwoGrid(coarseSpace=RigidBodyModes()))``.

Schur complement condensation (``custom``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Some dofs make the field-split AMG stall: the dofs of embedded bars with **bond-slip**, for instance, are tied to the
nodes of their host elements by stiff bond springs, :math:`u_{bar} \approx \sum_i N_i u_i`, a relation an aggregation
AMG coarse space cannot represent. (With perfect bond, the same relation is a multi-point constraint, eliminated before
the solve, and plain ``blockamg`` is the right choice.) A ``SchurComplement`` condenses such dofs :math:`b` exactly -- a
direct factorization of their block :math:`A_{bb}` -- and builds its child on the approximate Schur complement
:math:`\tilde S \approx A_{kk} - A_{kb} A_{bb}^{-1} A_{bk}` of all other dofs :math:`k`. The condensed dofs are chosen
with the **markers** of adaptive mesh refinement: the dofs of the interior nodes of the marked elements -- the nodes all
of whose elements are marked. It is a ``custom`` solver script; for embedded bars, mark the bar and the bond elements:

.. code-block:: python

    # solver.py
    from edelweissfe.config.markerlibrary import getMarkerClass
    from edelweissfe.linsolve.composition import (
        LGMRES, DiagonalScaling, Multiplicative, OnField, SchurComplement, gaussSeidelOrder,
    )
    from edelweissfe.linsolve.composition.presets import blockAMGFieldPreconditioner

    bars = getMarkerClass("elementSet").fromOptions({"elSet": "bars"})
    bonds = getMarkerClass("elementSet").fromOptions({"elSet": "bondElements"})


    def fieldCorrections(context):
        corrections = [OnField(f.name, blockAMGFieldPreconditioner(f, context)) for f in context.fields]
        return gaussSeidelOrder(corrections, sweeps=2)


    blockGaussSeidel = Multiplicative(fieldCorrections)
    linearSolver = LGMRES(SchurComplement(condense=[bars, bonds], preconditionS=DiagonalScaling(blockGaussSeidel)))

.. code-block:: edelweiss

    *linearSolver, name=barSchur, type=custom
    script=solver.py

    *solver, solver=NISTParallel, name=theSolver
    linsolver=barSchur

A ``fieldOutput`` marker condenses, e.g., a softening zone instead (testfile
``GradientEnhancedDamageSchurComplement``). The method and the caveats are on the :doc:`blockamgtheory` page (section
"Schur complement condensation").

Python API of the linear solvers
--------------------------------

.. automodule:: edelweissfe.linsolve.types
    :members:

.. automodule:: edelweissfe.config.linsolve
    :members: getLinSolverByName, getDefaultLinSolver, createLinearSolver, linearSolverDefinitionsFromInputFile

.. automodule:: edelweissfe.linsolve.base
    :members: LinearSolver, DirectSolverBackend, OneCallDirectSolverBackend
