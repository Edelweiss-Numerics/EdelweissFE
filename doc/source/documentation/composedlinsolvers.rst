Composed linear solvers
=======================

Every linear solver of EdelweissFE is a composed solver behind one adapter (:doc:`linsolvers`): ``type=blockamg`` is
the ``blockAMG()`` preset below, ``type=pardiso`` the one leaf ``DirectSolver("pardiso")``, and ``type=custom`` the
tree of a solver script of your own.

A composed linear solver is a **tree of small objects**, built with plain constructors in a Python script next to the
input file. Each object is one textbook component -- a Krylov solver, a diagonal scaling, a correction on one field, a
multiplicative combination of corrections, an AMG hierarchy, a Schur complement -- and its ``__call__`` is the formula
of the textbook. There is one way to build each method, and it is the fast one -- with one documented exception, the
two-level cycle (see :ref:`composed-two-level-two-ways`).

A solver script
---------------

.. code-block:: python

    # solver.py -- the linear solver of a displacement / nonlocal damage model
    from edelweissfe.linsolve.composition import (
        AMG, LGMRES, DiagonalScaling, Multiplicative, OnField, RigidBodyModes, TwoLevel,
    )

    # the displacement field: a two-level cycle on the rigid-body modes, AMG below the first coarse level
    displacement = OnField("displacement", TwoLevel(coarseSpace=RigidBodyModes()))
    # the nonlocal damage field: one AMG hierarchy
    damage = OnField("nonlocal damage", AMG())

    # both fields, one block Gauss-Seidel sweep: displacement first, then the nonlocal damage
    blockGaussSeidel = Multiplicative([displacement, damage])

    # accelerated by LGMRES
    linearSolver = LGMRES(DiagonalScaling(blockGaussSeidel))

and in the input file:

.. code-block:: edelweiss

    *linearSolver, name=mySolver, type=custom
    script=solver.py
    verbosity=info

    *solver, solver=NIST, name=theSolver
    linsolver=mySolver

The script defines the global ``linearSolver`` and holds only the math: the tree. What the adapter does around it --
the verbosity, the dumps, the hot reload -- are data lines of the block (a ``ComposedLinearSolver`` bound in the script
is refused). The journal logs the structure of the solver and a hash of the script at the first solve (with
``verbosity=info``).

The adapter
-----------

:class:`~edelweissfe.linsolve.composition.solver.ComposedLinearSolver` is what the nonlinear solver calls,
``x = solver(A, b)``, once per Newton iteration, for every type of linear solver. It finds what changed since the last
call (the first solve, a new model, a resized system, a new sparsity pattern, a jump of the right-hand side that marks
a new increment), hands the matrix down the tree, applies the root (to each column of ``b``, if it has several), and
reports on the solve. Its settings are the data lines every type has (see :doc:`linsolvers`):

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

* ``verbosity``: ``warning`` says nothing on a normal solve and warns about a slow one (more outer iterations than
  ``warnOuterIterations``) and one that missed its tolerance (stopped at the iteration cap, or on a scaled system with
  the scaled residual met but not the true one); ``info`` adds one line per solve, the structure of the solver at
  the first solve and the "linear solve" column of the nonlinear solver's table; ``debug`` adds each continuation of
  a Krylov solve on a scaled system, and in that column why the preconditioner was rebuilt (``rebuilt(stale)``) or
  ``reused``, with the tolerance and the time of the solve.
* ``dumps``: see :ref:`composed-solve-dumps`.
* ``hotReload``: see :ref:`composed-hot-reload`.

In Python, the same adapter is ``ComposedLinearSolver(tree, verbosity=..., warnOuterIterations=...,
residualGrowthFactor=..., dumps=SolveDumps(directory, trigger, maxDumps=..., contextSolves=...))``, or
``ComposedLinearSolver.fromScript("solver.py", hotReload=True, ...)``; ``nodeCoordinates`` and ``p1Maps`` give node
coordinates and P1 maps per field instead of those of the model (an offline replay without a model).

How to read the nesting
-----------------------

The script defines the components **bottom-up**, with named variables: first the basic method (the block Gauss--Seidel
sweep over the fields), then
what is wrapped around it, and the Krylov solver last -- the textbook order "build a basic iteration, then accelerate
it".

At run time the nesting means **delegation**: the outer component calls its children, many times, before and after its
own steps. LGMRES calls its preconditioner once per iteration; the diagonal scaling multiplies by :math:`D` before
*and* after the sweep; a Schur complement solves with :math:`A_{bb}` before and after its child. These are
sandwiches, not a pipeline -- which is why there is no ``|`` syntax.

Where a component sits in the tree says what it acts on. ``SchurComplement(..., DiagonalScaling(Multiplicative(...)))``
scales only what the sweep sees, while the outer Krylov solve still converges the true residual of :math:`A`.

Components
----------

.. list-table::
    :header-rows: 1
    :widths: 30 70

    * - Component
      - :math:`M^{-1}`
    * - ``LGMRES(preconditioner, tolerance, maxIterations, restart, augmentationVectors, rebuild, continuations, gapCompensation)``
      - the outer Krylov solve :math:`A x = b`, right-preconditioned (AMGCL); owns when the preconditioner is rebuilt
        (``RebuildPolicy(stalenessFactor, rebuildOnNewIncrement, reuse)``). Needs a fixed linear preconditioner, and
        refuses one that is not at construction. On a diagonally scaled matrix see :ref:`composed-scaled-krylov`.
    * - ``FGMRES(preconditioner, tolerance, maxIterations, restart)``
      - flexible GMRES (AMGCL): :math:`x = Z_m y_m` from the preconditioned vectors themselves, so the preconditioner
        may change between iterations -- e.g. an inner Krylov solve
    * - ``ForcingTolerance(minimum, maximum)``, ``FixedTolerance(eta)``
      - the tolerance per solve: Eisenstat--Walker, or fixed
    * - ``DiagonalScaling(inner)``
      - :math:`D \, \hat M^{-1} D`, :math:`\hat M` built on :math:`\hat A = D A D`,
        :math:`D = \operatorname{diag}(\lvert a_{ii} \rvert^{-1/2})`
    * - ``OnField(name, M)``
      - a correction on one field, :math:`R^T M^{-1} R`, :math:`M` built on the field's block :math:`A_{ff}`
    * - ``Multiplicative([...])``, ``Additive([...])``
      - successive corrections :math:`z \leftarrow z + M_k^{-1}(r - A z)`, or summed ones :math:`\sum_k M_k^{-1} r`;
        over ``OnField`` corrections, block Gauss--Seidel and block Jacobi. ``gaussSeidelOrder(corrections, sweeps,
        symmetric)`` gives the order of several or symmetric sweeps. The corrections may also be a function of the
        context, made at the first set-up (one per field, whatever the model)
    * - ``AMG(tree, backendBlockSize, backendPrecision, blockTranslationNullspace, rigidBodyNullspace, dropTolerance,
        lumpDroppedEntries)``
      - one AMGCL hierarchy on a field, one cycle per application. ``tree`` holds AMGCL's parameters only, forwarded
        verbatim; the backend (scalar entries or node-wise blocks, double or float, the block translations) is chosen
        by the keywords, the field's default for those not given (:ref:`composed-amg-backend`); with
        ``dropTolerance`` built on the filtered block (:ref:`composed-drop-tolerance`)
    * - ``TwoLevel(coarseSpace, tree, backendBlockSize, backendPrecision, coarseHierarchyPrecision,
        fineSpectralRadiusReuse, dropTolerance, lumpDroppedEntries)``
      - a two-level cycle with a coarse space of its own (a ``CoarseSpace``): all rigid-body modes per aggregate,
        ``RigidBodyModes(prolongation)``, or the P1 interpolant, ``P1Interpolation(coarsePrecond,
        useCoarseNullspace)``. Both read the fine Chebyshev smoother from the tree's ``relax``, ``npre`` and ``npost``
    * - ``SchurComplement(condense, preconditionS, solveA_bb, approximation, dropTolerance, windowNodes, haloNodes,
        blockFactors)``
      - block LU: :math:`z_k = M_{\tilde S}^{-1}(r_k - A_{kb} A_{bb}^{-1} r_b)`,
        :math:`z_b = A_{bb}^{-1}(r_b - A_{bk} z_k)`; ``solveA_bb`` is a ``DirectSolver``, set up on :math:`A_{bb}`
        on every solve
    * - ``DirectSolver(backend, **options)``
      - :math:`A^{-1}` by a registered direct solver backend (``superlu``, ``umfpack``, ``pardiso``, ``klu``,
        ``mumps``, ``petsclu``, ``panuapardiso``; registry category ``directsolver``): every backend factorizes once
        per set-up (a singular matrix is reported there) and back-substitutes in every application, so it is as
        cheap inside a Krylov tree as alone; ``type=pardiso`` is ``DirectSolver("pardiso",
        reuseSymbolicFactorization=...)`` alone
    * - ``CoarseCorrection(P, coarse)``, ``Jacobi(omega)``
      - :math:`P A_c^{-1} P^T` with :math:`A_c = P^T A P`, and :math:`\omega \operatorname{diag}(A)^{-1}`: the pieces
        of a two-level cycle spelled out, ``Multiplicative([Jacobi(), CoarseCorrection(P, coarse), Jacobi()])``
        (slow, for learning and new coarse spaces; see :ref:`composed-two-level-two-ways`)

.. _composed-amg-backend:

The AMGCL backend of a field
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``AMG`` and ``TwoLevel`` take the AMGCL tree and the backend apart: the tree is what AMGCL reads, the backend how
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

The combinators show that methods which look unrelated share one structure. Block Gauss--Seidel over the fields *is*
``Multiplicative([OnField("displacement", M_u), OnField("nonlocal damage", M_d)])``, and a two-level cycle *is*
``Multiplicative([smoother, CoarseCorrection(P, coarse), smoother])``. They are also fast: a combination asks each
correction which field it acts on, and forms the residual of a correction on field :math:`i` only on that field's
rows, :math:`r_i - \sum_{j: z_j \neq 0} A_{ij} z_j`, from the field blocks of the matrix (formed once, in one
threaded pass, for all corrections).

.. _composed-two-level-two-ways:

The two-level cycle: two ways, on purpose
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A two-level cycle can be written in two ways, and both are kept deliberately:

.. code-block:: python

    TwoLevel(coarseSpace=RigidBodyModes())                                  # the fast way
    Multiplicative([Jacobi(), CoarseCorrection(P, AMG()), Jacobi()])       # the spelled-out way

``TwoLevel`` is a leaf, like ``AMG``: pre-smoothing (Chebyshev), the coarse correction
:math:`P A_c^{-1} P^T (r - A z)` with :math:`A_c = P^T A P`, and post-smoothing run as one cycle in C++. Use it for
production runs; ``blockAMG()`` (``type=blockamg``) uses it for 3D displacement fields.

The spelled-out way builds the same cycle from general components, each of which can be replaced -- another smoother,
a coarse space of your own (any :math:`P`), a direct coarse solve. Use it to learn how a two-level cycle works and to
try a new coarse space or smoother. It is correct, but slow: on a 631k-dof displacement block it costs about twice as
much per application as ``TwoLevel`` (101 ms plus about 98 ms). The difference is where the work runs:

* ``TwoLevel`` keeps one compact copy of the field block (single precision, 3x3 node blocks) for the smoother and its
  residuals; the spelled-out components work on the general double-precision matrix and so read about twice the bytes
  per product;
* ``TwoLevel`` forms :math:`P x`, :math:`P^T r` and :math:`P^T A P` threaded in C++; ``CoarseCorrection`` uses SciPy;
* ``Multiplicative`` forms the residual :math:`r - A z` between its steps; ``TwoLevel`` takes it from the smoother.

There is no native Chebyshev component yet, so the spelled-out way cannot reproduce ``TwoLevel`` exactly. The exception
goes away once these three pieces exist natively: the compact field-block copy kept by the system matrix, a Chebyshev
smoother component, and threaded products with :math:`P` in ``CoarseCorrection`` (with ``RigidBodyModes`` and
``P1Interpolation`` building :math:`P`). Then ``TwoLevel`` is replaced by its spelled-out form.

Wrong compositions fail as early as possible: at construction when the tree alone shows it (a child that is not a
component, an empty combination, LGMRES around a Krylov solve, a Schur complement with nothing to condense), at the
first set-up otherwise, with a message that names the cause (a field the system does not have, a one-field component
such as ``AMG`` on several fields, rigid-body modes on a scalar field or without node coordinates).

A component of your own subclasses ``Preconditioner`` -- in the script itself, if you like --, calls
``super().__init__()`` in its constructor and implements the hooks ``_setUp(A, context)``, ``_apply(r)`` and, to reuse
what it built, ``_update(A, change)``; a component with children also implements ``children()``, the ``(role, child)``
pairs through which the tree is walked (the model, the end of a solve, whether it holds a Krylov solve, the views). It
then composes with every other component. The public ``setUp``, ``update`` and ``__call__`` of the base class record
each call and its time in the component's statistics and call the hook; a component calls its children through those
public methods.

Presets
-------

Presets are plain functions that return a tree. ``blockAMG()`` is the ``blockamg`` type, its arguments the data lines
of that type (``*linearSolver, type=blockamg``), and in a script ``linearSolver = blockAMG(...)``. It builds::

    LGMRES(
        DiagonalScaling(
            Multiplicative([OnField(f, M_f) for every field f])   # M_f: TwoLevel(RigidBodyModes()) for a
        )                                                         # 3D displacement field, else AMG()
    )

The corrections are made at the first set-up, for the fields of the model; ``sweeps`` and ``symmetric`` order them
with ``gaussSeidelOrder``. ``blockAMG(krylovOnScaledSystem=True)`` puts the scaling around LGMRES instead
(:ref:`composed-scaled-krylov`), ``continuations`` and ``gapCompensation`` go to LGMRES, and ``dropTolerance`` and
``lumpDroppedEntries`` to every field (:ref:`composed-drop-tolerance`).

Views
-----

Every component describes itself (``describe()``); the views in :mod:`edelweissfe.linsolve.composition.views` draw
the descriptions along the tree of ``children()``. For the ``blockAMG()`` preset on a displacement / nonlocal damage system, after two solves:

``print(solver)`` -- the structure::

    LGMRES(tolerance=ForcingTolerance(minimum=0.0001, maximum=0.001), maxIterations=100, restart=50, augmentationVectors=3)
    └─ DiagonalScaling(D=diag(1/√|aᵢᵢ|))
       └─ Multiplicative
          ├─ M1: OnField(field=displacement)
          │  └─ TwoLevel(coarse space=rigid-body modes, prolongation=smoothed, coarse solve=AMG V-cycle)
          └─ M2: OnField(field=nonlocal damage)
             └─ AMG(tree=default)

``views.formula(solver)`` -- the composed approximate inverse (``latex=True`` for LaTeX)::

    LGMRES(A, D · (Rᵀ AMG(nonlocal damage)⁻¹ R ∘ Rᵀ TwoLevel(displacement; rigid-body modes)⁻¹ R) · D)⁻¹

``views.runStatistics(solver)`` -- what each component recorded (inclusive times)::

    LGMRES(...)   updated 2× 0.07s · applied 2× 0.05s
    └─ DiagonalScaling(...)   updated 2× 0.07s · applied 18× 0.04s
       └─ Multiplicative   updated 2× 0.06s · applied 18× 0.04s
          ├─ M1: OnField(...)   set up 1× 0.06s · updated 1× 0.00s · applied 18× 0.03s
          │  └─ TwoLevel(...)   set up 1× 0.06s · ... · applied 18× 0.03s · rows 9555 · aggregates 198 · coarse dofs 1188
          └─ M2: OnField(...)   set up 1× 0.00s · updated 1× 0.00s · applied 18× 0.00s
             └─ AMG(...)   set up 1× 0.00s · ... · applied 18× 0.00s · rows 3185

The adapter only ever updates the root; a combination whose corrections are made per field sets them up at its first
update, and reuses them at the next.

``views.dataFlow(solver)`` -- one application as a graphviz graph (DOT, or SVG with the ``graphviz`` package): each
component a box around its steps, a step that delegates leading into its child's box. ``views.spy(solver, A)`` draws
the sparsity patterns of the matrices the components were built on, ``views.writeHtml(solver, "solver.html")`` writes
all views into one page, and in Jupyter a solver shows itself as that page.

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
solves, and a continuation tightens by the measured ratio; measured to save continuations per increment, but the
uncompensated scaled solve over-solves, which Newton relies on -- so it is opt-in. On an unscaled matrix there are no
continuations: a pass that misses :math:`\eta` has hit the iteration cap, which a warm restart would only extend.

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

.. _composed-solve-dumps:

Dumps of the solves
-------------------

The data line ``dumps`` (in Python ``SolveDumps(directory, trigger, maxDumps, contextSolves)``) writes the raw systems
of chosen solves to ``dumpDirectory``, for an offline replay (the workbench's ``bench.py``,
``scripts/benchmark_linsolve.py``). It works for every type, direct solvers included. The trigger:

* ``slow``: a solve with more outer iterations than ``warnOuterIterations`` (so that a dump and its warning fire
  together), or one stopped at the iteration cap short of its tolerance. A preconditioner that degrades typically shows
  only tens of hours into a run, at a solve nobody can name in advance; this trigger decides by the outcome;
* ``every``: every solve, a sequence (whether a factorization or a preconditioner can be reused is a question about a
  sequence, not a single matrix);
* solve numbers, e.g. ``dumps=12 13 14``: these solves of each linear solver (counted from 1).

On disk:

* ``A_<solver>_<solve>.npz`` (SciPy, uncompressed), ``b_<solver>_<solve>.npy``, and with a Schur complement the
  condensed dofs, ``schurDofs_<solver>_<solve>.npy``; ``<solver>`` is the ordinal of the linear solver in the process
  (one per step), so the dumps of several solvers never overwrite each other, and the solve numbering continues
  across a hot reload;
* one line per dumped system in ``manifest.jsonl``: ``solveCount``, ``role`` (``"trigger"`` or ``"context"``),
  ``triggerSolveCount``, ``contextOffset``, ``rows``, ``nnz``, ``matrixFile``, ``rhsFile``, the field ``blocks`` (name,
  start, stop, dimension), ``patternChanged``, ``newIncrement``, and for a Krylov solve its bookkeeping:
  ``outerIters``, ``trueResidual``, ``eta``, ``continuations``, ``info`` (1 for a cap hit), ``mustRefresh`` (the
  preconditioner was rebuilt for it), ``previousOuterIters``.

``dumpContextSolves`` also writes the solves *before* a dumped one, kept in memory as copies: one snapshot cannot tell
a hard operator from a solve made slow by a stale preconditioner, but a replay of the sequence through one persistent
solver can. ``maxDumps`` (default 10) bounds the systems written in the process, by all solvers, context solves
included. The format is that of the former ``blockamg`` degradation dumps; the former ``matrixdump`` linear solver is
``dumps=every`` (or solve numbers).

.. _composed-hot-reload:

Hot reload of the solver script
-------------------------------

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

From the former ``blockamg`` keys to a composed solver
------------------------------------------------------

``blockamg`` was a monolithic solver, ``BlockAMGSolver``, configured by a JSON ``linsolverConfigFile``; both are gone.
Every key it had, and where it lives now: a data line of ``*linearSolver, type=blockamg`` (an argument of the preset
``blockAMG(...)``) where one exists, else the component of a ``type=custom`` tree. The data lines every type has
(``verbosity``, ``warnOuterIterations``, ``residualGrowthFactor``, ``dumps``, ...) are named as such.

.. list-table::
    :header-rows: 1
    :widths: 32 68

    * - former ``blockamg`` key
      - data line of ``type=blockamg`` / component of a custom tree
    * - ``outerTol`` (a number)
      - data line ``tolerance``; ``LGMRES(..., tolerance=FixedTolerance(eta))``
    * - ``outerTol`` unset / ``"adaptive"``, ``etaMin``, ``etaMax``, ``ewGamma``, ``ewAlpha``
      - ``tolerance`` unset: the forcing tolerance with its defaults; else ``ForcingTolerance(minimum, maximum, gamma,
        alpha)``, the default tolerance of ``LGMRES``
    * - ``outerRestart``, ``outerMaxiter``
      - data line ``maxIterations`` (``outerRestart * outerMaxiter``); ``LGMRES(..., maxIterations)``
    * - ``outerSolver``
      - the class: ``LGMRES(...)``; ``FGMRES(...)`` for a variable preconditioner
    * - ``lgmresM``, ``lgmresK``
      - ``LGMRES(..., M, K)``
    * - ``lgmresAlwaysReset``, ``lgmresResetOnNewIncrement``
      - none: deprecated in ``blockamg``; every solve starts from an empty subspace
    * - ``sweeps``, ``symmetric``
      - data lines ``sweeps``, ``symmetric``; ``Multiplicative(gaussSeidelOrder([OnField(...), ...], sweeps, symmetric))``
    * - ``fieldPreconds``
      - ``OnField(name, AMG(tree))`` or ``OnField(name, TwoLevel(coarseSpace, tree))``; checked when the component meets
        its field
    * - ``useRigidBodyNullspace``
      - ``AMG(rigidBodyNullspace=...)``
    * - ``rigidBodyCoarseLevel`` (unset, ``false``, names)
      - data line ``rigidBodyCoarseLevel`` (``true`` for the 3D displacement field, ``false`` for none); other fields:
        ``OnField(name, TwoLevel(coarseSpace=RigidBodyModes()))``
    * - ``rigidBodyCoarseLevel`` options ``prolongation``, ``coarseHierarchyPrecision``, ``fineSpectralRadiusReuse``
      - ``RigidBodyModes(prolongation, coarseHierarchyPrecision, fineSpectralRadiusReuse)``
    * - ``p1FieldNames``
      - ``OnField(name, TwoLevel(coarseSpace=P1Interpolation(...)))``
    * - ``p1Nu``, ``p1NuPre``, ``p1NuPost``, ``p1FineDegree``, ``p1FineBlockSize``, ``p1FineValueType``,
        ``p1CoarseValueType``, ``p1CoarsePrecond``
      - ``P1Interpolation(nu, nuPre, nuPost, fineDegree, fineBlockSize, fineValueType, coarseValueType,
        coarsePrecond)``, checked at construction
    * - ``p1FineSpectralRadiusReuse``
      - ``P1Interpolation(fineSpectralRadiusReuse=True)``
    * - ``p1Maps``, ``nodeCoordinates`` (constructor only)
      - ``ComposedLinearSolver(..., p1Maps, nodeCoordinates)`` (Python)
    * - ``hierarchyDropTol``, ``hierarchyDropLumping``
      - data lines ``dropTolerance``, ``lumpDroppedEntries``; ``AMG(dropTolerance, lumpDroppedEntries)``,
        ``TwoLevel(..., dropTolerance, lumpDroppedEntries)``
    * - ``residualGrowthFactor``
      - data line ``residualGrowthFactor`` (every type)
    * - ``hierarchyStalenessFactor``, ``refreshOnNewIncrement``, ``reuseHierarchy``
      - ``LGMRES(..., rebuild=RebuildPolicy(stalenessFactor, rebuildOnNewIncrement, reuse))``
    * - ``krylovOnScaledSystem``
      - data line ``krylovOnScaledSystem``; the order of the components: ``DiagonalScaling(LGMRES(...))`` instead of
        ``LGMRES(DiagonalScaling(...))``
    * - ``trueResidualMaxContinuations``
      - data line ``continuations``; ``LGMRES(..., continuations)`` (only on a scaled matrix)
    * - ``gapCompensatedTolerance``, ``gapSafetyFactor``, ``gapMaxFactor``
      - data line ``gapCompensation`` (``true``: the default factors); ``LGMRES(...,
        gapCompensation=GapCompensation(safetyFactor, maxFactor))``
    * - ``verbosity``
      - data line ``verbosity`` (every type)
    * - ``warnOuterIterationsThreshold``
      - data line ``warnOuterIterations`` (every type)
    * - ``dumpOnDegradationDir``, ``dumpOnDegradationThreshold``, ``dumpOnDegradationMaxDumps``,
        ``dumpOnDegradationContextSolves``
      - data lines ``dumps=slow``, ``dumpDirectory``, ``maxDumps``, ``dumpContextSolves`` (every type), the same files;
        the threshold is ``warnOuterIterations``
    * - ``hotReloadConfigFile``
      - data line ``hotReload=true`` of ``type=custom``: the solver script is run again when it changes
    * - ``schurComplement``: ``markers``
      - ``SchurComplement(condense=[getMarkerClass(type).fromOptions(options), ...], ...)``
        (``edelweissfe.config.markerlibrary``)
    * - ``schurComplement``: ``approximation``, ``dropTolerance``, ``windowNodes``, ``haloNodes``, ``blockFactors``
      - ``SchurComplement(..., approximation, dropTolerance, windowNodes, haloNodes, blockFactors)``
    * - ``schurComplement``: ``directSolver``
      - ``SchurComplement(..., solveA_bb=DirectSolver(backend, **options))``
    * - ``condensation`` (constructor only)
      - ``SchurComplement(..., condensation=...)``

What ``blockamg`` did besides its keys, and where it is now:

* the forcing tolerance, the new-increment detection and the reuse/staleness decisions: ``ForcingTolerance``,
  the adapter (``residualGrowthFactor``) and ``RebuildPolicy`` in ``LGMRES``, with the same code;
* a non-finite right-hand side or diagonal declines the solve (NaN, a cutback), a zero diagonal raises:
  the adapter and ``DiagonalScaling``;
* the warnings (a slow solve, a missed tolerance), the "linear solve" column and its debug lines: the adapter;
* the Schur complement's warnings (a factorization of :math:`A_{bb}` or of :math:`\tilde S` that fails, every dof or
  more than half of them selected, non-positive diagonal entries of :math:`\tilde S`) and its info line:
  ``SchurComplement``, through the adapter's verbosity;
* the checks of the settings: each component checks its own parameters at construction, and what needs the fields
  (a field tree's backend against the field's dimension, the rigid-body coarse level on the scalar backend) when it
  meets its field. A component on a field the system does not have fails at the set-up, where ``blockamg`` only warned;
* the performance timers (``equilibration``, ``equilibrated system``, ``diagonal blocks``, ``hierarchy build``, the
  builds per field, ``outer Krylov solve``, ``true-residual continuations``, ``schur complement``, ``p1 topology``):
  the same names, recorded by the components;
* a default 3D displacement field without node coordinates gets ``AMG()`` on the translations, logged at ``info`` by
  ``blockAMG()``.

Python API of the components
----------------------------

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
