Composed linear solvers
=======================

.. note::

    A prototype. ``blockamg`` remains the production solver; the composed ``blockAMG()`` preset reproduces it (same
    outer iterations, same solution) and is meant to replace its internals.

A composed linear solver is a **tree of small objects**, built with plain constructors in a Python script next to the
input file. Each object is one textbook component -- a Krylov solver, a diagonal scaling, a correction on one field, a
multiplicative combination of corrections, an AMG hierarchy, a Schur complement -- and its ``__call__`` is the formula
of the textbook. There is one way to build each method, and it is the fast one -- with one documented exception, the
two-grid cycle (see :ref:`composed-two-grid-two-ways`).

A solver script
---------------

.. code-block:: python

    # solver.py -- the linear solver of a displacement / nonlocal damage model
    from edelweissfe.linsolve.composition import (
        AMG, LGMRES, ComposedLinearSolver, DiagonalScaling, Multiplicative, OnField, RigidBodyModes, TwoGrid,
    )

    # the displacement field: a two-grid cycle on the rigid-body modes, AMG below the first coarse level
    displacement = OnField("displacement", TwoGrid(coarseSpace=RigidBodyModes()))
    # the nonlocal damage field: one AMG hierarchy
    damage = OnField("nonlocal damage", AMG())

    # both fields, one block Gauss-Seidel sweep: displacement first, then the nonlocal damage
    blockGaussSeidel = Multiplicative([displacement, damage])

    # accelerated by LGMRES; the driver ComposedLinearSolver turns it into the linear solver of the input file
    linearSolver = ComposedLinearSolver(LGMRES(DiagonalScaling(blockGaussSeidel)))

and in the input file:

.. code-block:: edelweiss

    *solver, solver=NIST, name=theSolver
    linsolver=script:solver.py:linearSolver

(``script:solver.py`` alone takes the global ``linearSolver``.) The script binds the **driver**,
``ComposedLinearSolver``, around the tree; a bare tree is refused with the one-line fix,
``linearSolver = ComposedLinearSolver(<your tree>)``. The script runs with the global ``options`` set to the parsed
``linsolverConfigFile``, if the solver has one. The journal logs the structure of the solver and a hash of the script
at the first solve (with ``verbosity="info"``).

The driver
----------

``ComposedLinearSolver`` is what the nonlinear solver calls, ``x = solver(A, b)``, once per Newton iteration. It finds
what changed since the last call (the first solve, a new model, a resized system, a new sparsity pattern, a jump of the
right-hand side that marks a new increment), hands the matrix down the tree, applies the root, and reports on the solve.
Its settings, all optional:

.. code-block:: python

    linearSolver = ComposedLinearSolver(
        LGMRES(DiagonalScaling(blockGaussSeidel)),
        verbosity="info",                 # "silent", "warning" (default), "info", "debug"
        warnOuterIterations=60,           # a solve with more outer iterations gets a warning (default 100)
        residualGrowthFactor=4.0,         # a right-hand side this many times the last one is a new increment
        degradationDumps=DegradationDumps("slowSolves", threshold=80, maxDumps=10, contextSolves=2),
        hotReload=True,                   # run this script again whenever it changes, during the run
    )

* ``verbosity``: ``"warning"`` says nothing on a normal solve and warns about a slow one (more outer iterations than
  ``warnOuterIterations``) and one that missed its tolerance (stopped at the iteration cap, or on a scaled system with
  the scaled residual met but not the true one); ``"info"`` adds one line per solve, the structure of the solver at
  the first solve and the "linear solve" column of the nonlinear solver's table; ``"debug"`` adds each continuation of
  a Krylov solve on a scaled system, and in that column why the preconditioner was rebuilt (``rebuilt(stale)``) or
  ``reused``, with the tolerance and the time of the solve.
* ``degradationDumps``: see :ref:`composed-degradation-dumps`.
* ``hotReload``: see :ref:`composed-hot-reload`.
* ``nodeCoordinates``, ``p1Maps``: node coordinates and P1 maps per field, instead of those of the model (an offline
  replay without a model).

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
    * - ``LGMRES(preconditioner, tolerance, maxIterations, M, K, rebuild, continuations, gapCompensation)``
      - the outer Krylov solve :math:`A x = b`, right-preconditioned (AMGCL); owns when the preconditioner is rebuilt
        (``RebuildPolicy(stalenessFactor, rebuildOnNewIncrement, reuse)``). Needs a fixed linear preconditioner, and
        refuses one that is not at construction. On a diagonally scaled matrix see :ref:`composed-scaled-krylov`.
    * - ``FGMRES(preconditioner, tolerance, maxIterations, M)``
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
    * - ``AMG(tree, rigidBodyNullspace, dropTolerance, lumpDroppedEntries)``
      - one AMGCL hierarchy on a field, one cycle per application; with ``dropTolerance`` built on the filtered block
        (:ref:`composed-drop-tolerance`)
    * - ``TwoGrid(coarseSpace=RigidBodyModes() | P1Interpolation(), tree, dropTolerance, lumpDroppedEntries)``
      - a two-grid cycle with a coarse space of its own: all rigid-body modes per aggregate, or the P1 interpolant
    * - ``SchurComplement(condense, preconditionS, solveA_bb, blockFactors)``
      - block LU: :math:`z_k = M_{\tilde S}^{-1}(r_k - A_{kb} A_{bb}^{-1} r_b)`,
        :math:`z_b = A_{bb}^{-1}(r_b - A_{bk} z_k)`
    * - ``DirectSolver(linsolver)``
      - :math:`A^{-1}` by a registered direct solver (``superlu``, ``pardiso``, ...)
    * - ``CoarseCorrection(P, coarse)``, ``Jacobi(omega)``
      - :math:`P A_c^{-1} P^T` with :math:`A_c = P^T A P`, and :math:`\omega \operatorname{diag}(A)^{-1}`: the pieces
        of a two-grid cycle spelled out, ``Multiplicative([Jacobi(), CoarseCorrection(P, coarse), Jacobi()])``
        (slow, for learning and new coarse spaces; see :ref:`composed-two-grid-two-ways`)

The combinators show that methods which look unrelated share one structure. Block Gauss--Seidel over the fields *is*
``Multiplicative([OnField("displacement", M_u), OnField("nonlocal damage", M_d)])``, and a two-grid cycle *is*
``Multiplicative([smoother, CoarseCorrection(P, coarse), smoother])``. They are also fast: a combination asks each
correction which field it acts on, and forms the residual of a correction on field :math:`i` only on that field's
rows, :math:`r_i - \sum_{j: z_j \neq 0} A_{ij} z_j`, from the field blocks of the matrix (formed once, in one
threaded pass, for all corrections).

.. _composed-two-grid-two-ways:

The two-grid cycle: two ways, on purpose
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A two-grid cycle can be written in two ways, and both are kept deliberately:

.. code-block:: python

    TwoGrid(coarseSpace=RigidBodyModes())                                  # the fast way
    Multiplicative([Jacobi(), CoarseCorrection(P, AMG()), Jacobi()])       # the spelled-out way

``TwoGrid`` is a leaf, like ``AMG``: pre-smoothing (Chebyshev), the coarse correction
:math:`P A_c^{-1} P^T (r - A z)` with :math:`A_c = P^T A P`, and post-smoothing run as one cycle in C++. Use it for
production runs; ``blockAMG()`` uses it for 3D displacement fields.

The spelled-out way builds the same cycle from general components, each of which can be replaced -- another smoother,
a coarse space of your own (any :math:`P`), a direct coarse solve. Use it to learn how a two-grid cycle works and to
try a new coarse space or smoother. It is correct, but slow: on a 631k-dof displacement block it costs about twice as
much per application as ``TwoGrid`` (101 ms plus about 98 ms). The difference is where the work runs:

* ``TwoGrid`` keeps one compact copy of the field block (single precision, 3x3 node blocks) for the smoother and its
  residuals; the spelled-out components work on the general double-precision matrix and so read about twice the bytes
  per product;
* ``TwoGrid`` forms :math:`P x`, :math:`P^T r` and :math:`P^T A P` threaded in C++; ``CoarseCorrection`` uses SciPy;
* ``Multiplicative`` forms the residual :math:`r - A z` between its steps; ``TwoGrid`` takes it from the smoother.

There is no native Chebyshev component yet, so the spelled-out way cannot reproduce ``TwoGrid`` exactly. The exception
goes away once these three pieces exist natively: the compact field-block copy kept by the system matrix, a Chebyshev
smoother component, and threaded products with :math:`P` in ``CoarseCorrection`` (with ``RigidBodyModes`` and
``P1Interpolation`` building :math:`P`). Then ``TwoGrid`` is replaced by its spelled-out form.

Wrong compositions fail as early as possible: at construction when the tree alone shows it (a child that is not a
component, an empty combination, LGMRES around a Krylov solve, a Schur complement with nothing to condense), at the
first set-up otherwise, with a message that names the cause (a field the system does not have, a one-field component
such as ``AMG`` on several fields, rigid-body modes on a scalar field or without node coordinates).

A component of your own subclasses ``Preconditioner`` -- in the script itself, if you like -- and implements
``setUp(A, context)``, ``__call__(r)`` and, to reuse what it built, ``update(A, change)``. It then composes with every
other component.

Presets
-------

Presets are plain functions that return a tree (bound in a script as ``linearSolver =
ComposedLinearSolver(blockAMG())``). ``blockAMG()`` builds the equivalent of ``blockamg`` with its defaults::

    LGMRES(
        DiagonalScaling(
            Multiplicative([OnField(f, M_f) for every field f])   # M_f: TwoGrid(RigidBodyModes()) for a
        )                                                         # 3D displacement field, else AMG()
    )

The corrections are made at the first set-up, for the fields of the model; ``sweeps`` and ``symmetric`` order them
with ``gaussSeidelOrder``. ``blockAMG(krylovOnScaledSystem=True)`` puts the scaling around LGMRES instead
(:ref:`composed-scaled-krylov`), ``continuations`` and ``gapCompensation`` go to LGMRES, and ``dropTolerance`` and
``lumpDroppedEntries`` to every field (:ref:`composed-drop-tolerance`).

Views
-----

Every component describes itself (``describe()``); the views in :mod:`edelweissfe.linsolve.composition.views` draw
the description. For the ``blockAMG()`` preset on a displacement / nonlocal damage system, after two solves:

``print(solver)`` -- the structure::

    LGMRES(tolerance=ForcingTolerance(minimum=0.0001, maximum=0.001), maxIterations=100, M=50, K=3)
    └─ DiagonalScaling(D=diag(1/√|aᵢᵢ|))
       └─ Multiplicative
          ├─ M1: OnField(field=displacement)
          │  └─ TwoGrid(coarse space=rigid-body modes, prolongation=smoothed)
          └─ M2: OnField(field=nonlocal damage)
             └─ AMG(tree=default)

``views.formula(solver)`` -- the composed approximate inverse (``latex=True`` for LaTeX)::

    LGMRES(A, D · (Rᵀ AMG(nonlocal damage)⁻¹ R ∘ Rᵀ TwoGrid(displacement; rigid-body modes)⁻¹ R) · D)⁻¹

``views.runStatistics(solver)`` -- what each component recorded (inclusive times)::

    LGMRES(...)   updated 2× 0.07s · applied 2× 0.05s
    └─ DiagonalScaling(...)   updated 2× 0.07s · applied 18× 0.04s
       └─ Multiplicative   set up 1× 0.06s · updated 1× 0.00s · applied 18× 0.04s
          ├─ M1: OnField(...)   set up 1× 0.06s · ... · applied 18× 0.03s
          │  └─ TwoGrid(...)   set up 1× 0.06s · ... · applied 18× 0.03s · rows 9555 · aggregates 198 · coarse dofs 1188
          └─ M2: OnField(...)   set up 1× 0.00s · ... · applied 18× 0.00s
             └─ AMG(...)   set up 1× 0.00s · ... · applied 18× 0.00s · rows 3185

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

``AMG(dropTolerance=τ)`` and ``TwoGrid(..., dropTolerance=τ)`` build on the filtered block :math:`\tilde A`: the
off-diagonal entries with :math:`\lvert a_{ij} \rvert < \tau \sqrt{\lvert a_{ii} a_{jj} \rvert}` dropped, the
diagonal always kept. For operators with many entries per row that are numerically far sparser than structurally (e.g.
RKPM), smoothed aggregation then weighs only the entries that matter, and the Galerkin products carry fewer of them to
the coarse levels. Only the preconditioner sees :math:`\tilde A`; the Krylov solve applies :math:`A`, so the solution
is that of :math:`A`. ``lumpDroppedEntries=True`` adds the dropped entries of each row onto its diagonal, which keeps
the row sums (the "filtered matrix" of the AMG literature); for elasticity, whose off-diagonals are mostly negative,
this weakens the diagonal dominance. The run statistics record the stored entries before and after (``nnz``, ``nnz
kept``). ``blockAMG(dropTolerance=τ, lumpDroppedEntries=...)`` passes them to every field.

.. _composed-degradation-dumps:

Degradation dumps
-----------------

A preconditioner that degrades typically shows only tens of hours into a run, at a solve nobody can name in advance.
``ComposedLinearSolver(..., degradationDumps=DegradationDumps(directory, threshold, maxDumps, contextSolves))`` decides
by the outcome: a solve with more outer iterations than ``threshold`` (default: ``warnOuterIterations``, so that a dump
and its warning fire together), or one stopped at the iteration cap short of its tolerance, has its raw system written
to ``directory``, for an offline replay (the workbench's ``bench.py`` reads it unchanged):

* ``A_<solver>_<solve>.npz`` (SciPy, uncompressed), ``b_<solver>_<solve>.npy``, and with a Schur complement the
  condensed dofs, ``schurDofs_<solver>_<solve>.npy``; ``<solver>`` is the ordinal of the solver in the process, so the
  dumps of several solvers (one per step) never overwrite each other, and the solve numbering continues across a hot
  reload;
* one line per dumped system in ``manifest.jsonl``: ``solveCount``, ``role`` (``"trigger"`` or ``"context"``),
  ``triggerSolveCount``, ``contextOffset``, ``rows``, ``nnz``, ``matrixFile``, ``rhsFile``, the field ``blocks`` (name,
  start, stop, dimension), and the bookkeeping of the solve: ``outerIters``, ``trueResidual``, ``eta``,
  ``continuations``, ``info`` (1 for a cap hit), ``mustRefresh`` (the preconditioner was rebuilt for it),
  ``patternChanged``, ``newIncrement``, ``previousOuterIters``.

``contextSolves`` also writes the solves *before* the slow one, kept in memory as copies: one snapshot cannot tell a
hard operator from a solve made slow by a stale preconditioner, but a replay of the sequence through one persistent
solver can. ``maxDumps`` (default 10) bounds the systems written in the process, by all solvers, context solves
included. The format is that of ``blockamg``'s ``dumpOnDegradation*``, from the same code.

.. _composed-hot-reload:

Hot reload of the solver script
-------------------------------

With ``hotReload=True``, the driver reads its script before every solve and, if it changed, runs it again and
continues with the ``ComposedLinearSolver`` it builds -- the whole solver, its settings included -- set up on that solve
(rebuild reason ``reload``). What the run has learnt stays: the solve count, the residual history, the numbering of the
dumps. The journal says so at ``warning``, with the sha256 of the script and the new tree, so that a reader of the log
knows which solver was in force from which solve onwards. This makes solver settings testable on a *live* run, at the
state that was expensive to reach.

A bad edit can never end the run: a script that cannot be read or does not compile (e.g. half written by an editor) is
retried on the next solve; one that raises, or binds no ``ComposedLinearSolver``, is reported (``REJECTED``) and the
solver in force kept until the script changes again; a new solver whose set-up fails (e.g. an AMGCL tree naming an
unknown smoother) is ``ROLLED BACK`` to the one in force. The contents themselves are the stamp, not the modification
time, which coarse file timestamps can leave unchanged across two quick edits.

From ``blockamg`` to a composed solver
--------------------------------------

Every setting of ``blockamg`` and where it lives in a composed solver. ``blockAMG(...)`` is the preset that reproduces
``blockamg``; its keywords are listed where they exist.

.. list-table::
    :header-rows: 1
    :widths: 32 68

    * - ``blockamg`` key
      - composed equivalent
    * - ``outerTol`` (a number)
      - ``LGMRES(..., tolerance=FixedTolerance(eta))``; ``blockAMG(tolerance=...)``
    * - ``outerTol`` unset / ``"adaptive"``, ``etaMin``, ``etaMax``, ``ewGamma``, ``ewAlpha``
      - ``ForcingTolerance(minimum, maximum, gamma, alpha)``, the default tolerance of ``LGMRES``
    * - ``outerRestart``, ``outerMaxiter``
      - ``LGMRES(..., maxIterations=outerRestart * outerMaxiter)``
    * - ``outerSolver``
      - the class: ``LGMRES(...)``; ``FGMRES(...)`` for a variable preconditioner
    * - ``lgmresM``, ``lgmresK``
      - ``LGMRES(..., M, K)``
    * - ``lgmresAlwaysReset``, ``lgmresResetOnNewIncrement``
      - none: deprecated in ``blockamg``; every solve starts from an empty subspace
    * - ``sweeps``, ``symmetric``
      - ``Multiplicative(gaussSeidelOrder([OnField(...), ...], sweeps, symmetric))``; ``blockAMG(sweeps, symmetric)``
    * - ``fieldPreconds``
      - ``OnField(name, AMG(tree))`` or ``OnField(name, TwoGrid(coarseSpace, tree))``; checked when the component meets
        its field
    * - ``useRigidBodyNullspace``
      - ``AMG(rigidBodyNullspace=...)``
    * - ``rigidBodyCoarseLevel`` (unset, ``false``, names)
      - ``TwoGrid(coarseSpace=RigidBodyModes())`` on the field; ``blockAMG(rigidBodyCoarseLevel=False)`` for none
    * - ``rigidBodyCoarseLevel`` options ``prolongation``, ``coarseHierarchyPrecision``, ``fineSpectralRadiusReuse``
      - ``RigidBodyModes(prolongation, coarseHierarchyPrecision, fineSpectralRadiusReuse)``
    * - ``p1FieldNames``
      - ``OnField(name, TwoGrid(coarseSpace=P1Interpolation(...)))``
    * - ``p1Nu``, ``p1NuPre``, ``p1NuPost``, ``p1FineDegree``, ``p1FineBlockSize``, ``p1FineValueType``,
        ``p1CoarseValueType``, ``p1CoarsePrecond``
      - ``P1Interpolation(nu, nuPre, nuPost, fineDegree, fineBlockSize, fineValueType, coarseValueType,
        coarsePrecond)``, checked at construction
    * - ``p1FineSpectralRadiusReuse``
      - ``P1Interpolation(fineSpectralRadiusReuse=True)``
    * - ``p1Maps``, ``nodeCoordinates`` (constructor only)
      - ``ComposedLinearSolver(..., p1Maps, nodeCoordinates)``
    * - ``hierarchyDropTol``, ``hierarchyDropLumping``
      - ``AMG(dropTolerance, lumpDroppedEntries)``, ``TwoGrid(..., dropTolerance, lumpDroppedEntries)``;
        ``blockAMG(dropTolerance, lumpDroppedEntries)``
    * - ``residualGrowthFactor``
      - ``ComposedLinearSolver(..., residualGrowthFactor)``
    * - ``hierarchyStalenessFactor``, ``refreshOnNewIncrement``, ``reuseHierarchy``
      - ``LGMRES(..., rebuild=RebuildPolicy(stalenessFactor, rebuildOnNewIncrement, reuse))``
    * - ``krylovOnScaledSystem``
      - the order of the components: ``DiagonalScaling(LGMRES(...))`` instead of ``LGMRES(DiagonalScaling(...))``;
        ``blockAMG(krylovOnScaledSystem=True)``
    * - ``trueResidualMaxContinuations``
      - ``LGMRES(..., continuations)`` (only on a scaled matrix)
    * - ``gapCompensatedTolerance``, ``gapSafetyFactor``, ``gapMaxFactor``
      - ``LGMRES(..., gapCompensation=GapCompensation(safetyFactor, maxFactor))``
    * - ``verbosity``
      - ``ComposedLinearSolver(..., verbosity)``
    * - ``warnOuterIterationsThreshold``
      - ``ComposedLinearSolver(..., warnOuterIterations)``
    * - ``dumpOnDegradationDir``, ``dumpOnDegradationThreshold``, ``dumpOnDegradationMaxDumps``,
        ``dumpOnDegradationContextSolves``
      - ``ComposedLinearSolver(..., degradationDumps=DegradationDumps(directory, threshold, maxDumps, contextSolves))``,
        the same files
    * - ``hotReloadConfigFile``
      - ``ComposedLinearSolver(..., hotReload=True)``: the solver script is run again when it changes
    * - ``schurComplement``: ``markers``
      - ``SchurComplement(condense=[getMarkerClass(type).fromOptions(options), ...], ...)``
        (``edelweissfe.config.markerlibrary``)
    * - ``schurComplement``: ``approximation``, ``dropTolerance``, ``windowNodes``, ``haloNodes``, ``blockFactors``
      - ``SchurComplement(..., approximation, dropTolerance, windowNodes, haloNodes, blockFactors)``
    * - ``schurComplement``: ``directSolver``
      - ``SchurComplement(..., solveA_bb=DirectSolver(linsolver, options))``
    * - ``condensation`` (constructor only)
      - ``SchurComplement(..., condensation=...)``

What ``blockamg`` did besides its keys, and where it is now:

* the forcing tolerance, the new-increment detection and the reuse/staleness decisions: ``ForcingTolerance``,
  ``ComposedLinearSolver`` (``residualGrowthFactor``) and ``RebuildPolicy`` in ``LGMRES``, with the same code;
* a non-finite right-hand side or diagonal declines the solve (NaN, a cutback), a zero diagonal raises:
  ``ComposedLinearSolver`` and ``DiagonalScaling``;
* the warnings (a slow solve, a missed tolerance), the "linear solve" column and its debug lines: ``ComposedLinearSolver``;
* the Schur complement's warnings (a factorization of :math:`A_{bb}` or of :math:`\tilde S` that fails, every dof or
  more than half of them selected, non-positive diagonal entries of :math:`\tilde S`) and its info line:
  ``SchurComplement``, through the driver's verbosity;
* the checks of the settings: each component checks its own parameters at construction, and what needs the fields
  (a field tree's backend against the field's dimension, the rigid-body coarse level on the scalar backend) when it
  meets its field. A component on a field the system does not have fails at the set-up, where ``blockamg`` only warned;
* the performance timers (``equilibration``, ``equilibrated system``, ``diagonal blocks``, ``hierarchy build``, the
  builds per field, ``outer Krylov solve``, ``true-residual continuations``, ``schur complement``, ``p1 topology``):
  the same names, recorded by the components;
* a default 3D displacement field without node coordinates gets ``AMG()`` on the translations, logged at ``info`` by
  ``blockAMG()``.

API
---

.. automodule:: edelweissfe.linsolve.composition.base
    :members:

.. automodule:: edelweissfe.linsolve.composition.krylov
    :members:

.. automodule:: edelweissfe.linsolve.composition.diagonalscaling
    :members:

.. automodule:: edelweissfe.linsolve.composition.amg
    :members:

.. automodule:: edelweissfe.linsolve.composition.twogrid
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

.. automodule:: edelweissfe.linsolve.composition.degradationdumps
    :members:
