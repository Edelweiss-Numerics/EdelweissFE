Linear solvers
==============

Linear solvers are defined in EdelweissFE after the ``*solver`` keyword using ``linsolver`` and an optional configuration file ``linsolverConfigFile`` as a data line.
The ``linsolverConfigFile`` needs to be in ``.json`` format.

Choose a linsolver after the ``*solver`` keyword:

.. code-block:: edelweiss

    *solver, solver=NIST, name=theSolver
    linsolver=gmres
    linsolverConfigFile=opt.json

.. list-table:: Currently available linear solvers
    :width: 100%
    :widths: 15 1 25
    :header-rows: 1

    * - Name
      - Direct solver
      - Relevant module
    * - ``superlu``
      - ✓
      - ``scipy.sparse.linalg.spsolve``
    * - ``umfpack``
      - ✓
      - ``scipy.sparse.linalg.spsolve``
    * - ``pardiso``
      - ✓
      - ``edelweissfe.linsolve.pardiso.pardiso``
    * - ``panuapardiso``
      - ✓
      - ``edelweissfe.linsolve.panuapardiso.panuapardiso``
    * - ``klu``
      - ✓
      - ``edelweissfe.linsolve.klu.klu``
    * - ``petsclu``
      - ✓
      - ``edelweissfe.linsolve.petsclu.petsclu``
    * - ``mumps``
      - ✓
      - ``edelweissfe.linsolve.mumps.mumps``
    * - ``gmres``
      - ✗
      - ``edelweissfe.linsolve.gmres.gmres``
    * - ``amgcl``
      - ✗
      - ``edelweissfe.linsolve.amgcl.amgcl``
    * - ``blockamg``
      - ✗
      - ``edelweissfe.linsolve.blockamg.blockamg``
    * - ``matrixdump``
      - —
      - ``edelweissfe.linsolve.matrixdump.matrixdump``

How the available solvers relate
--------------------------------

Every entry in the table above is selected the same way (``linsolver=<name>``) and presents the same
``(A, b) -> x`` interface, but they fall into three groups that differ in what they actually do:

**Direct solvers** factorize the matrix and back-substitute: ``superlu`` and ``umfpack`` (SciPy,
always available), ``pardiso`` (Intel MKL, the production default), ``panuapardiso``, ``klu``,
``petsclu`` and ``mumps``. They ignore ``linsolverConfigFile`` (except ``pardiso``, which reads a
single ``reuseSymbolicFactorization`` flag). Pick one of these unless the factorization has become
the run's bottleneck or exceeds available memory.

**Iterative solvers** never factorize the full matrix, trading exactness for O(n) memory:

* ``gmres`` -- GMRES preconditioned by a ``pyamg`` smoothed-aggregation hierarchy over the whole
  matrix.
* ``amgcl`` -- the AMGCL library's own solver/preconditioner combinations, configured through its
  JSON parameter tree.
* ``blockamg`` -- a *field-split* variant for coupled multi-field models: one AMG hierarchy per
  physical field (e.g. displacement, nonlocal damage), combined by a block Gauss--Seidel sweep to
  precondition an outer GMRES/LGMRES. This is the route to problem sizes a direct factorization
  cannot reach at all.

**A diagnostic wrapper**, which does not solve anything itself:

* ``matrixdump`` -- writes every :math:`(A, b)` pair it is handed to disk, then hands the solve to
  a real solver chosen with its ``delegate`` option, so the simulation proceeds normally. The point
  is to lift solver comparisons out of the finite element run: rerunning a simulation per solver
  variant is slow and not like-for-like (a variant that changes the Newton iterates changes the
  *sequence* of matrices being compared). Dump one authentic sequence once, then replay it offline
  with ``scripts/benchmark_linsolve.py`` so every variant sees byte-identical input. Use it to
  investigate solver performance, never in a production run.

Several linsolvers accept an optional configuration file ``linsolverConfigFile`` (a ``.json`` file), among them ``gmres``, ``amgcl``, ``blockamg`` and ``matrixdump``; the direct solvers ignore it (``pardiso`` additionally reads a single ``reuseSymbolicFactorization`` flag).

Choose the options for the linsolver (in this case ``gmres``) in an extra file:

.. code-block:: json

    	{
	"precondopts":
	{
	"presmoother": ["block_gauss_seidel", {"iterations": 15}],
	"postsmoother": ["block_gauss_seidel", {"iterations": 15}],
	},
	"linsolveopts": {"maxiter": 1, "restart": 1500}
	}


The ``blockamg`` solver
-----------------------

``blockamg`` is a field-split block-AMG solver for **large coupled multi-field systems** (e.g. displacement + gradient-enhanced damage). It is the O(n)-memory route to problem sizes a direct factorization cannot reach -- past roughly a million DOFs its fill-in exceeds memory, whereas algebraic multigrid stays linear.

Applied *monolithically*, AMG is ineffective on such a coupled system (a single hierarchy cannot represent the disparate fields at once -- their physical scales and near-null-spaces differ). ``blockamg`` instead builds **one AMGCL algebraic-multigrid hierarchy per field** and combines them with a **block Gauss--Seidel** sweep to precondition an outer Krylov solve. Per solve it equilibrates the system, splits it into field blocks, builds or reuses the per-field hierarchies, and preconditions the outer solve with the block sweep.

The block structure -- which DOFs belong to which field, and each field's dimension -- is **discovered automatically** from the ``DofManager`` and the live ``FEModel``, both handed to the solver via ``setModel()`` whenever the equation system is (re)built. A ``linsolverConfigFile`` is therefore optional and carries only solver knobs. Requires the optional ``amgcl`` extension.

.. code-block:: edelweiss

    *solver, solver=NIST, name=theSolver
    linsolver=blockamg
    linsolverConfigFile=blockamg.json

.. code-block:: json

    {
        "outerSolver": "amgcl_lgmres",
        "sweeps": 1,
        "symmetric": true
    }

The method, the near-null-space construction, the adaptive outer tolerance, hierarchy reuse, every configuration key with its default, and the experimental p-multigrid variant are documented in detail on the dedicated page linked at the bottom of this page.


The ``amgcl`` solver
--------------------

``amgcl`` is an iterative solver (Krylov method plus algebraic-multigrid or single-level preconditioner) built on the `AMGCL <https://github.com/ddemidov/amgcl>`_ library. Its ``linsolverConfigFile`` is forwarded as an AMGCL parameter tree; note that AMGCL silently ignores unknown parameter keys (warning only on stderr), so check its stderr if a configuration behaves unexpectedly.


The ``matrixdump`` diagnostic solver
------------------------------------

``matrixdump`` is not a solver but a diagnostic wrapper: it writes the equation systems it is handed to disk and then delegates the actual solve to a real linear solver, so a sequence of authentic ``(A, b)`` pairs can be replayed offline instead of by rerunning the simulation. Its ``linsolverConfigFile`` selects the ``delegate`` solver, the dump ``directory``, and which solves to capture (``dumpAt`` / ``skipFirst`` / ``maxDumps`` / ``instances``).


.. toctree::
   :maxdepth: 1

   blockamgtheory
