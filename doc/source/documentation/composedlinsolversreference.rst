Composed linear solvers: reference
==================================

The arguments, defaults and data lines of the components and of the adapter, for looking up; how to build a solver
from them is on :doc:`composedlinsolvers`, the method they implement on :doc:`blockamgtheory`, and the classes and
functions on :doc:`linsolveapi`.

.. _composed-adapter:

The adapter
-----------

:class:`~edelweissfe.linsolve.composition.solver.ComposedLinearSolver` is what the nonlinear solver calls,
``x = solver(A, b)``, once per Newton iteration, for every type of linear solver. It finds what changed since the last
call (the first solve, a new model, a resized system, a new sparsity pattern, a jump of the right-hand side that marks
a new increment), hands the matrix down the tree, applies the root (to each column of ``b``, if it has several), and
reports on the solve. Its settings are the data lines every type has (the table in :doc:`linsolvers`):

.. code-block:: edelweiss

    *linearSolver, name=mySolver, type=custom
    script=solver.py
    ** silent, warning (default), info, debug
    verbosity=info
    ** a solve with more outer iterations gets a warning (default 100)
    warnOuterIterations=60
    ** a right-hand side this many times the last one is a new increment (default 4)
    residualGrowthFactor=4.0
    ** none (default), slow, every, or the numbers of the solves: 12 13 14
    dumps=slow
    dumpDirectory=slowSolves
    maxDumps=10
    dumpContextSolves=2
    ** custom only: run the script again whenever it changes, during the run
    hotReload=true

In Python, the same adapter is ``ComposedLinearSolver(tree, verbosity=..., warnOuterIterations=...,
residualGrowthFactor=..., dumps=SolveDumps(directory, trigger, maxDumps=..., contextSolves=...))``, or
``ComposedLinearSolver.fromScript("solver.py", hotReload=True, ...)``; ``nodeCoordinates`` and ``p1Maps`` give node
coordinates and P1 maps per field instead of those of the model (an offline replay without a model).

.. _composed-verbosity:

Verbosity levels
~~~~~~~~~~~~~~~~

Messages go through the shared :class:`~edelweissfe.journal.journal.Journal`, and the data line ``verbosity`` decides
whether the solver attempts to log at all -- the Journal's own level then decides whether the attempt is shown.
All messages are nested inside the Newton iteration they belong to; warnings are marked ``⚠``.

``silent``
    Nothing at all.

``warning`` (default)
    Only abnormal conditions, so a healthy run stays quiet: an outer iteration count past
    ``warnOuterIterations`` (a slow solve), a true residual still above the requested tolerance after every
    continuation (saying whether an outer pass stopped at the iteration cap, or whether, on a scaled system, the scaled
    residual was met but not the true one), and any dump written. A pass that stopped at the cap but whose solve still met the tolerance, after a continuation,
    is not reported; the hierarchies are rebuilt all the same.

``info``
    Turns on the "linear solve" column of the nonlinear solver's per-iteration table: outer iteration
    count, achieved true residual (and whether it met the requested tolerance) and number of
    continuations of each solve; and shows the structure of the solver, in a frame, at its first solve. The frame
    stands above the header of the table, so that nothing the solver says interrupts the rows. This is the level to
    run at when first characterising a model on an iterative solver.

``debug``
    Adds, per solve, whether the hierarchies were rebuilt (and why, e.g. ``rebuilt(stale)``) or ``reused``, the
    requested tolerance and the wall-clock, in the "linear solve" column, and one line per continuation of a Krylov
    solve on a scaled system. After a rebuild, the notes of the components on what they built follow in the column:
    for a ``SchurComplement``, the condensed dofs and the nonzeros of :math:`\tilde S`.

The sizes of what a ``SchurComplement`` built are also recorded in its statistics (``condensed dofs``, ``S̃ nnz``), see
the run-statistics view.

Per-stage wall-clock -- equilibration, the equilibrated system (the scaled operator and the coupling
blocks between fields, built in one threaded pass), hierarchy build (per field, and per stage of a
p-multigrid build), near-null-space construction, outer solve, and continuations -- is recorded through
:mod:`~edelweissfe.utils.performancetiming` regardless of verbosity, and appears nested under "linear
solve" in the job's performance table. That table, not the log, is the first place to look when
deciding whether a run is dominated by hierarchy builds or by outer iterations.

.. _composed-solve-dumps:

Dumps of the solves
~~~~~~~~~~~~~~~~~~~

The data line ``dumps`` (in Python ``SolveDumps(directory, trigger, maxDumps, contextSolves)``) writes the raw systems
of chosen solves to ``dumpDirectory``, for an offline replay (``scripts/benchmark_linsolve.py``, which replays a
dumped sequence through direct solver variants, or a
``ComposedLinearSolver`` driven by hand with ``setFieldStructure`` and the dumped geometry as ``nodeCoordinates`` and
``p1Maps``). It works for every type, direct solvers included, and is the one way to capture a solve. A dump is
the raw ``(A, b)`` of a solve -- before any diagonal scaling -- with the field-block layout active for that solve,
which is what a replay needs in order to go back through the same tree without a live model: a sequence of
authentic systems, so that every solver variant replayed offline sees the same input. The trigger:

* ``slow``: a solve with more outer iterations than ``warnOuterIterations`` (so that a dump and its warning fire
  together), or one stopped at the iteration cap short of its tolerance. A preconditioner that degrades typically shows
  only tens of hours into a run, at a solve nobody can name in advance (a late-increment, heavily damaged Jacobian,
  say); unlike ``every`` or solve numbers, fixed *before* the run, this trigger decides by the solve's own outcome;
* ``every``: every solve, a sequence (whether a factorization or a preconditioner can be reused is a question about a
  sequence, not a single matrix);
* solve numbers, e.g. ``dumps=12 13 14``: these solves of each linear solver (counted from 1).

On disk:

* ``A_<solver>_<solve>.npz`` (SciPy, uncompressed), ``b_<solver>_<solve>.npy``, and with a Schur complement the
  condensed dofs, ``schurDofs_<solver>_<solve>.npy``; ``<solver>`` is the ordinal of the linear solver in the process
  (one per nonlinear solver, made anew only when a step names another linear solver), so the dumps of several
  solvers never overwrite each other, and the solve numbering continues across steps and a hot reload;
* ``geometry_<solver>_<solve>.npz``: what a replay of the tree needs of the model, the node coordinates of every vector
  field (``<field>_coordinates``, for its rigid-body modes) and the P1 maps the tree used (``<field>_isCorner``,
  ``<field>_edgeEndpoints``);
* one line per dumped system in ``manifest.jsonl``: ``solveCount``, ``role`` (``"trigger"`` or ``"context"``),
  ``triggerSolveCount``, ``contextOffset``, ``rows``, ``nnz``, ``matrixFile``, ``rhsFile``, ``geometryFile``, the field
  ``blocks`` (name, start, stop, dimension), ``patternChanged``, ``newIncrement``, and for a Krylov solve its
  ``SolveRecord`` -- ``iterations``, ``residual`` (the true relative residual), ``eta``, ``capHit``, ``continuations``,
  ``rebuildReason`` (``null``: the preconditioner was reused), ``scaledSystem`` -- and ``previousIterations``.

``dumpContextSolves`` also writes the solves *before* a dumped one, kept in memory as copies. A single snapshot cannot
distinguish "this operator is intrinsically hard" from "this solve's own carried-over state degraded independently
of the operator" -- such as a reused, now-stale hierarchy. Both look identical from the matrix alone, and a fresh
solver replaying the matrix on its own can converge in a fraction of the live iteration count, which is a genuinely
misleading result. Capturing the preceding sequence lets an offline replay feed one persistent solver instance the
same solves in the same order. The staleness question can often be answered from the manifest alone, without a
replay. ``maxDumps`` (default 10) bounds the systems written to one directory in the process, by all solvers that dump
there, context solves included. A sequence of every solve is ``dumps=every`` (or solve numbers).

.. _composed-hot-reload:

Hot reload of the solver script
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

With ``hotReload=true`` (``type=custom``), the adapter reads the script before every solve and, if it changed, runs
it again and continues with the tree it builds, set up on that solve (rebuild reason ``reload``). The settings of the
adapter stay those of the input file, and what the run has learnt stays too: the solve count, the residual history,
the numbering of the dumps. The journal says so at ``warning``, with the sha256 of the script and the new tree, so
that a reader of the log knows which solver was in force from which solve onwards. This makes solver settings
testable on a *live* run, at the state that was expensive to reach.

A bad edit can never end the run: a script that cannot be read or does not compile (e.g. half written by an editor) is
retried on the next solve; one that raises, or defines no composed solver, is reported (``REJECTED``) and the tree in
force kept until the script changes again; a new tree whose set-up fails (e.g. an AMGCL tree naming an unknown
smoother) is ``ROLLED BACK`` to the one in force. The contents themselves are the stamp, not the modification time,
which coarse file timestamps can leave unchanged across two quick edits.

Reuse and tolerance
-------------------

:class:`~edelweissfe.linsolve.composition.solver.ComposedLinearSolver` finds what changed since the last solve (the
first solve, a new model, a resized system, a new sparsity pattern, a jump of the right-hand side that marks a new
increment) and hands it down the tree as a ``MatrixChange``. ``LGMRES`` adds its own reasons -- a new increment, a
stale preconditioner (too many more iterations than the solve it was built for, or a cap hit), or ``reuse=False`` --
and every component decides in ``update`` what to keep: an ``AMG`` hierarchy is kept unless a rebuild is asked for,
a ``DiagonalScaling`` keeps its :math:`D`, a ``Multiplicative`` always takes the new couplings.

A ``DiagonalScaling`` rebuilds its :math:`D` together with what is below it, also for the reasons a Krylov solver below
it adds itself (``Preconditioner.withRebuildReasons``), so that ``DiagonalScaling(LGMRES(...))`` rebuilds as
``LGMRES(DiagonalScaling(...))`` does.

What is rebuilt is first forgotten (``Preconditioner.forget``): a Krylov solver that rebuilds its preconditioner drops
all of it before any part is built again, a ``SchurComplement`` drops :math:`\tilde S` and its child before it forms
the next :math:`\tilde S`, the blocks of the last matrix before it cuts out those of this one, and a direct solve its
factorization before it factorizes again; the adapter forgets the whole tree when the model changes (a new step, a
change of the mesh). So the old and the new hierarchies, factorizations and matrices are never held at once. What is
reused on purpose survives: the hierarchies while no rebuild is asked for, the selection of the condensed dofs, the
fine spectral radius of ``fineSpectralRadiusReuse``, a symbolic factorization (PARDISO's
``reuseSymbolicFactorization``), and what the Krylov solver has learnt (its reference iterations, the gap of a
``GapCompensation``, its scratch vectors).

Performance: a diagonal scaling does not form :math:`D A D`; the first component below it that needs a block forms
:math:`\hat A` and all field blocks in one threaded native pass, kept with the matrix for all others, which also keeps :math:`A` as the operator of the outer Krylov solve
(:class:`~edelweissfe.linsolve.composition.systemmatrix.SystemMatrix`).

.. _composed-scaled-krylov:

A Krylov solve on the scaled system
-----------------------------------

Where the diagonal scaling sits says what the Krylov solve converges. ``LGMRES(DiagonalScaling(M))`` (the default of
``blockAMG()``) solves :math:`A x = b`, preconditioned by :math:`D \hat M^{-1} D`: its residual is the true one,
:math:`\lVert b - A x \rVert / \lVert b \rVert`. ``DiagonalScaling(LGMRES(M))`` (``blockAMG(krylovOnScaledSystem=True)``)
solves :math:`\hat A \hat x = D b` with :math:`\hat A = D A D`, :math:`x = D \hat x`, and converges the scaled
residual :math:`\lVert D (b - A x) \rVert`, which weighs every row by its own diagonal -- a field with small residual
entries is resolved as well as the others -- but which can be smaller than the true one, on which the solve is accepted.

So a Krylov solver on a scaled matrix checks the true residual after its pass (in the physical variables of its
problem: :math:`b - A x = D^{-1} (\hat b - \hat A \hat x)`) and, while it misses :math:`\eta`, continues from its
iterate with a tighter tolerance, 0.01 times the last, at most ``continuations`` times (default 2). With
``gapCompensation=GapCompensation(safetyFactor=0.3, maxFactor=1e3)`` the first pass asks for
:math:`s \, \eta / g` instead, with :math:`g` the ratio of the true to the scaled residual learnt from the previous
solves, and a continuation tightens by the measured ratio; it is opt-in (:ref:`measured-gap-compensation`). On an unscaled matrix there are no
continuations: a pass that misses :math:`\eta` has hit the iteration cap, which a warm restart would only extend.
The method is :ref:`blockamg-true-residual`.

.. _composed-drop-tolerance:

A preconditioner on the filtered matrix
---------------------------------------

``AMG(dropTolerance=τ)`` and ``TwoLevel(..., dropTolerance=τ)`` build on the filtered block :math:`\tilde A`: the
off-diagonal entries with :math:`\lvert a_{ij} \rvert < \tau \sqrt{\lvert a_{ii} a_{jj} \rvert}` dropped, the
diagonal always kept. For operators with many entries per row that are numerically far sparser than structurally (e.g.
RKPM), smoothed aggregation then weighs only the entries that matter, and the Galerkin products carry fewer of them to
the coarse levels. Only the preconditioner sees :math:`\tilde A`; the Krylov solve applies :math:`A`, so the solution
is that of :math:`A`. ``lumpDroppedEntries=True`` adds the dropped entries of each row onto its diagonal, which keeps
the row sums (the "filtered matrix" of the AMG literature); for elasticity, whose off-diagonals are mostly negative,
this weakens the diagonal dominance. The run statistics record the stored entries before and after (``nnz``, ``nnz
kept``). ``blockAMG(dropTolerance=τ, lumpDroppedEntries=...)`` passes them to every field.

.. _composed-amg-backend:

The AMGCL backend of a field
----------------------------

``AMG`` and ``TwoLevel`` take the AMGCL tree and the backend apart (the method: :ref:`blockamg-per-field-amg`): the tree is what AMGCL reads, the backend how
EdelweissFE stores the field's matrix for it. The defaults are chosen in one place,
:func:`~edelweissfe.linsolve.composition.amg.backendOf`, from the field:

.. list-table::
    :header-rows: 1
    :widths: 30 70

    * - Keyword
      - Default
    * - ``backendBlockSize``
      - the field's dimension for a 2D or 3D field (node-wise blocks: less index and value traffic per product), else
        1 (scalar entries)
    * - ``backendPrecision``
      - ``"float"`` on 3x3 blocks (set up in double, only the matrices the cycle reads stored in float), else
        ``"double"``
    * - ``blockTranslationNullspace`` (``AMG`` only)
      - on a block backend, smoothed aggregation from the translations :math:`D^{-1} t` of the scaled block instead of
        AMGCL's constant blocks

So ``AMG(backendBlockSize=1)`` puts a 3D displacement field on the scalar backend, in double, where its hierarchy
takes all rigid-body modes as its near null-space. Rotations belong to the near null-space of a displacement field
only (by its phenomenon, :func:`~edelweissfe.linsolve.nullspace.carriesRigidBodyModes`): the rotation field of a
micropolar model gets its translations.

.. _composed-default-field-trees:

Default field trees
-------------------

``AMG(tree=...)`` and ``TwoLevel(..., tree=...)`` take a field's AMGCL parameter tree, forwarded to AMGCL verbatim. A
tree given replaces the default outright, so start from the one that applies
and change only what is intended. The defaults are, by the field's nodal dimension
(:func:`~edelweissfe.linsolve.composition.amg.defaultFieldTree` in the code):

.. code-block:: json

    {
        "3D vector field (e.g. displacement)": {
            "coarsening": {
                "type": "smoothed_aggregation",
                "aggr": {"eps_strong": 0.01},
                "estimate_spectral_radius": true,
                "power_iters": 20
            },
            "relax": {"type": "chebyshev", "degree": 16, "power_iters": 300, "lower": 0.01, "higher": 1.1},
            "npre": 0,
            "npost": 1
        },
        "2D vector field": {
            "coarsening": {
                "type": "smoothed_aggregation",
                "aggr": {"eps_strong": 0.01},
                "estimate_spectral_radius": true,
                "power_iters": 20
            },
            "relax": {"type": "chebyshev", "degree": 16, "power_iters": 300, "lower": 0.01, "higher": 1.1},
            "npre": 0,
            "npost": 1
        },
        "vector field of a dimension other than 2 or 3": {
            "coarsening": {
                "type": "smoothed_aggregation",
                "aggr": {"eps_strong": 0.01},
                "estimate_spectral_radius": true,
                "power_iters": 20
            },
            "relax": {"type": "chebyshev", "degree": 16, "power_iters": 300, "lower": 0.01, "higher": 1.1},
            "npre": 0,
            "npost": 1
        },
        "scalar field (e.g. nonlocal damage)": {
            "coarsening": {"type": "smoothed_aggregation"},
            "relax": {"type": "chebyshev"}
        }
    }

The backend the tree is built on is chosen apart from it, by the keywords ``backendBlockSize``, ``backendPrecision``
and (``AMG`` only) ``blockTranslationNullspace``, with defaults from the field
(:func:`~edelweissfe.linsolve.composition.amg.backendOf`): node-wise 3x3 blocks stored in float for a 3D field, 2x2
blocks in double for a 2D one (this wrapper has no float 2x2 backend), the scalar backend in double otherwise, and on
a block backend the translations :math:`D^{-1} t` as the near null-space. A vector field of a dimension the defaults
were never tuned for (anything but 2 or 3, e.g. a 4-DOF-per-node formulation) gets the same coarsening and
relaxation as the 2D and 3D fields, on the scalar backend. ``AMG(backendBlockSize=1)`` puts a 2D or 3D displacement
field on the scalar backend, the only one that takes the rotations (:ref:`blockamg-block-nullspace`); a tree with another coarsening than
smoothed aggregation on a block backend needs ``blockTranslationNullspace=False``, which is checked.

The 3D displacement field of the ``blockAMG()`` preset gets the rigid-body coarse level
(``TwoLevel(coarseSpace=RigidBodyModes())``), configured by the same tree: its ``relax`` and ``npre``/``npost`` are the
fine smoother of the rigid-body coarse level, its ``coarsening`` the aggregation of that level, and the whole tree the
AMGCL hierarchy below it. ``TwoLevel(coarseSpace=P1Interpolation())`` reads its fine smoother from the tree in the same
spelling. Only a genuinely scalar field (nodal dimension 1) gets the plain scalar tree, which predates the vector
recipe and was never re-tuned against it. Note that AMGCL silently ignores parameter keys it does not recognise
(warning only on stderr), so check its stderr if an override appears to do nothing.

.. _blockamg-rigid-body-coarse-level-configuration:

The rigid-body coarse level: ``RigidBodyModes`` and ``TwoLevel``
----------------------------------------------------------------

The ``blockAMG()`` preset (``type=blockamg``) gives the 3D displacement field the rigid-body coarse level; ``rigidBodyCoarseLevel=false``
switches it off. A 2D displacement field, or a 3D one with other options, gets it in a ``custom`` solver script,
``OnField(name, TwoLevel(coarseSpace=RigidBodyModes(...), ...))``. The field must be on a block backend
(``backendBlockSize`` 2 or 3, the default of a 2D or 3D field), and its tree must smooth with a Chebyshev polynomial
(``relax.type`` ``chebyshev``, without ``relax.scale``, ``relax.power_iters`` at least 1) and coarsen by smoothed
aggregation, as the default trees do; ``TwoLevel`` checks this when it meets its field, and the parameters of
``RigidBodyModes`` at construction. The option of ``RigidBodyModes``:

``prolongation`` (``"smoothed"``)
    ``"smoothed"``: :math:`P = (I - \omega\, \mathrm{diag}_B(\hat A_F)^{-1} \hat A_F) P^{\mathrm{tent}}`.
    ``"tentative"``:
    :math:`P^{\mathrm{tent}}` itself -- a cheaper setup and a sparser coarse operator, for more outer iterations (32
    instead of 22 on the elastic tunnel system).

and those of ``TwoLevel`` (the same for the p-two-grid):

``coarseHierarchyPrecision`` (``"double"``)
    The value type of the AMGCL hierarchy below: ``"double"``, or ``"float"`` (AMGCL's single-precision backend, its
    setup included) for half its memory, at the price of more outer iterations.
``fineSpectralRadiusReuse`` (``false``)
    Estimate the fine smoother's spectral radius in full only on a field's first build, and on every later rebuild
    only re-check it, as for the p-two-grid: 15 power iterations continued from the
    previous build's dominant vector, instead of the tree's ``relax.power_iters`` (300 in the default trees) from a
    fixed start vector. The full estimate is the largest part of the setup, 7.6 of 10.7 s on the elastic tunnel
    system. It is safe where the operator changes gradually between rebuilds, as over Newton iterations and
    increments: a re-check estimate more than 1% above the previous radius replaces it, and one beyond the smoother's
    10% margin (``relax.higher``) -- an operator that stiffened more than the margin covers -- triggers the full
    estimate again, because a radius too small for the operator would make the smoother amplify its highest modes. An
    operator that softened keeps the previous, larger radius, which costs the smoother some efficiency at the low end
    of its window. The cache is dropped when the model changes (e.g. after adaptive refinement) and when a hot reload
    replaces the tree.

.. _composed-schur-complement-arguments:

The arguments of ``SchurComplement``
------------------------------------

The method is :ref:`blockamg-schur-complement`, a complete script :ref:`composed-schur-complement-recipe`. The
arguments of :class:`~edelweissfe.linsolve.composition.schurcomplement.SchurComplement`:

``condense`` (required)
    A list of markers, created through the marker registry from the options of a ``>>marker`` line of
    ``*modelModifier, type=hAdaptivity``, e.g. ``getMarkerClass("fieldOutput").fromOptions({"fieldOutput":
    "damage", "threshold": 0.05, "halo": 1})``. Sets, surfaces and fieldOutputs they name must exist.
``preconditionS`` (required)
    The preconditioner of :math:`\tilde S`, e.g. ``DiagonalScaling(Multiplicative(...))``.
``approximation`` (``"local"``)
    ``"local"``: :math:`\tilde S` as in :ref:`blockamg-schur-complement`. ``"none"``: :math:`\tilde S = A_{kk}`, the coupling through :math:`b`
    ignored (for comparison).
``dropTolerance`` (``1e-4``)
    :math:`\tau`, the relative size below which entries of :math:`\tilde S` are dropped.
``windowNodes``, ``haloNodes`` (``50``, ``windowNodes``)
    The nodes of a window's core along a chain, and of the halo on either side. A chain no longer than one core with
    both halos is corrected exactly.
``blockFactors`` (``"U"``)
    ``"U"``: the block upper triangular factor, one solve with :math:`A_{bb}` per application. ``"LU"``: the block
    LU factorization, two solves.
``solveA_bb`` (``DirectSolver("superlu")``)
    The solver of :math:`A_{bb}`.
