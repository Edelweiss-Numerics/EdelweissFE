Linear solvers
==============

Linear solvers are defined in EdelweissFE after the ``*solver`` keyword using ``linsolver`` and an optional configuration file ``linsolverConfigFile`` as a data line.
The ``linsolverConfigFile`` needs to be in ``.json`` format.

Choose a linsolver after the ``*solver`` keyword:

.. code-block:: edelweiss

    *solver, solver=NIST, name=theSolver
    linsolver=pardiso

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

* ``amgcl`` -- the AMGCL library's own solver/preconditioner combinations, configured through its
  JSON parameter tree.
* ``blockamg`` -- a *field-split* variant for coupled multi-field models: one AMG hierarchy per
  physical field (e.g. displacement, nonlocal damage), combined by a block Gauss--Seidel sweep to
  precondition an outer GMRES/LGMRES. This is the route to problem sizes a direct factorization
  cannot reach at all.
  With ``schurComplement``, it condenses selected dofs exactly first, for dofs on which AMG alone
  stalls, such as those of bars tied to their host elements by bond springs. With
  ``rigidBodyCoarseLevel``, a displacement field gets the rotations into its coarse space, for
  bending-dominated structures.

**A diagnostic wrapper**, which does not solve anything itself:

* ``matrixdump`` -- writes every :math:`(A, b)` pair it is handed to disk, then hands the solve to
  a real solver chosen with its ``delegate`` option, so the simulation proceeds normally. The point
  is to lift solver comparisons out of the finite element run: rerunning a simulation per solver
  variant is slow and not like-for-like (a variant that changes the Newton iterates changes the
  *sequence* of matrices being compared). Dump one authentic sequence once, then replay it offline
  with ``scripts/benchmark_linsolve.py`` so every variant sees byte-identical input. Use it to
  investigate solver performance, never in a production run.

Several linsolvers accept an optional configuration file ``linsolverConfigFile`` (a ``.json`` file), among them ``amgcl``, ``blockamg`` and ``matrixdump``; the direct solvers ignore it (``pardiso`` additionally reads a single ``reuseSymbolicFactorization`` flag). See each solver's own section below for its configuration format.


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
        "symmetric": false
    }

The method, the near-null-space construction, the adaptive outer tolerance, hierarchy reuse, every configuration key with its default, the experimental p-multigrid variant, the rigid-body coarse level and the Schur complement condensation are documented in detail on their own page:

.. toctree::
   :maxdepth: 1

   blockamgtheory


Schur complement condensation (``blockamg``)
---------------------------------------------

Some dofs make ``blockamg`` stall: the dofs of embedded bars with **bond-slip**, for instance, are tied to
the nodes of their host elements by stiff bond springs, :math:`u_{bar} \approx \sum_i N_i u_i`, a relation an
aggregation AMG coarse space cannot represent. (With perfect bond, the same relation is a multi-point constraint,
eliminated before the solve, and plain ``blockamg`` is the right choice.) The ``schurComplement`` option of ``blockamg``
condenses such dofs :math:`b` exactly -- a direct factorization of their block :math:`A_{bb}` -- and builds
the field-split AMG on the approximate Schur complement :math:`\tilde S \approx A_{kk} - A_{kb}
A_{bb}^{-1} A_{bk}` of all other dofs :math:`k`. Everything else is ``blockamg`` as usual.

The condensed dofs are chosen with the **markers** of adaptive mesh refinement (the options of a
``>>marker`` line): the dofs of the interior nodes of the marked elements -- the nodes all of whose
elements are marked -- are condensed. For embedded bars, mark the bar and the bond elements:

.. code-block:: edelweiss

    *solver, solver=NISTParallel, name=theSolver
    linsolver=blockamg
    linsolverConfigFile=blockamg.json

.. code-block:: json

    {
        "sweeps": 2,
        "schurComplement": {
            "markers": [
                {"type": "elementSet", "elSet": "bars"},
                {"type": "elementSet", "elSet": "bondElements"}
            ]
        }
    }

``bars`` and ``bondElements`` are the element sets of the bar and the bond elements. A ``fieldOutput`` marker
condenses, e.g., a softening zone instead (testfile ``GradientEnhancedDamageSchurComplement``). The method, every
key and the caveats are on the :doc:`blockamgtheory` page (section "Schur complement condensation").


Rigid-body coarse level (``blockamg``)
--------------------------------------

The hierarchy ``blockamg`` builds for a 2D or 3D displacement field stores the field in node-wise blocks, and its
coarse space holds the **translations** of every aggregate of nodes, not their rotations. In a bending-dominated
structure -- a slender beam, a tunnel lining, a plate -- the slowly converging error is bending, under which every
aggregate rotates almost as a rigid body: neither the coarse space nor the smoother reduces it, and the outer solve
needs many iterations. The ``rigidBodyCoarseLevel`` option of ``blockamg`` gives the fields it names a first coarse
level built from all rigid-body modes, translations and rotations, and keeps everything else as it is:

.. code-block:: edelweiss

    *solver, solver=NISTParallel, name=theSolver
    linsolver=blockamg
    linsolverConfigFile=blockamg.json

.. code-block:: json

    {
        "rigidBodyCoarseLevel": ["displacement"]
    }

The testfile deck ``testfiles/edelweiss-only/CantileverBeamHexa20BlockAMGRigidBody``, a slender cantilever, shows the
effect: with the option, each linear solve converges in some 20 outer iterations; without it, every solve stops at the
iteration cap. On an elastic tunnel segment of quadratic hexahedra, it cut the outer iterations of the displacement
block from 67 to 22; where the default already needs few iterations, or once softening dominates, it gains less. A
scalar field (e.g. nonlocal damage) has no rotations and is rejected. The method, its options and the measurements are
on the :doc:`blockamgtheory` page (section "Rigid-body coarse level").


The ``amgcl`` solver
--------------------

``amgcl`` is an iterative solver (Krylov method plus algebraic-multigrid or single-level preconditioner) built on the `AMGCL <https://github.com/ddemidov/amgcl>`_ library. Its ``linsolverConfigFile`` is forwarded as an AMGCL parameter tree; note that AMGCL silently ignores unknown parameter keys (warning only on stderr), so check its stderr if a configuration behaves unexpectedly.


The ``matrixdump`` diagnostic solver
------------------------------------

``matrixdump`` is not a solver but a diagnostic wrapper: it writes the equation systems it is handed to disk and then delegates the actual solve to a real linear solver, so a sequence of authentic ``(A, b)`` pairs can be replayed offline instead of by rerunning the simulation. Its ``linsolverConfigFile`` selects the ``delegate`` solver, the dump ``directory``, and which solves to capture (``dumpAt`` / ``skipFirst`` / ``maxDumps`` / ``instances``).

Its instance/dump-ordinal counters are process-wide and not part of a ``*restart`` checkpoint, so a resumed run starts them back at zero -- point a resumed analysis using ``matrixdump`` at a fresh ``directory`` if you need both runs' dumps kept, rather than the resumed run silently overwriting the interrupted one's.
