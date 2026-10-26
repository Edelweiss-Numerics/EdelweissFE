Composed linear solvers: reference
==================================

The arguments, defaults and data lines of the components and of the adapter, for looking up; how to build a solver from
them is on :doc:`composedlinsolvers`, the method they implement on :doc:`blockamgtheory`.

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

.. _composed-verbosity:

Verbosity levels
----------------

Messages go through the shared :class:`~edelweissfe.journal.journal.Journal`, and the data line ``verbosity`` decides
whether the solver attempts to log at all -- the Journal's own level then decides whether the attempt is shown.
All messages are nested inside the Newton iteration they belong to; warnings are marked ``⚠``.

``silent``
    Nothing at all.

``warning`` (default)
    Only abnormal conditions, so a healthy run stays quiet: an outer iteration count past
    ``warnOuterIterations``, a true residual still above the requested tolerance after every
    continuation (saying whether an outer pass stopped at the iteration cap), and any dump
    written. A pass that stopped at the cap but whose solve still met the tolerance, after a continuation,
    is not reported; the hierarchies are rebuilt all the same.

``info``
    Turns on the "linear solve" column of the nonlinear solver's per-iteration table: outer iteration
    count, achieved true residual (and whether it met the requested tolerance) and number of
    continuations of each solve; and the structure of the solver at its first solve. This is the level to run at
    when first characterising a model on an iterative solver.

``debug``
    Adds, per solve, whether the hierarchies were rebuilt (and why) or reused, the requested
    tolerance and the wall-clock, and one line per true-residual continuation attempt.

Per-stage wall-clock -- equilibration, the equilibrated system (the scaled operator and the coupling
blocks between fields, built in one threaded pass), hierarchy build (per field, and per stage of a
p-multigrid build), near-null-space construction, outer solve, and continuations -- is recorded through
:mod:`~edelweissfe.utils.performancetiming` regardless of verbosity, and appears nested under "linear
solve" in the job's performance table. That table, not the log, is the first place to look when
deciding whether a run is dominated by hierarchy builds or by outer iterations.

.. _composed-capturing-degraded-solves:

Capturing degraded solves
~~~~~~~~~~~~~~~~~~~~~~~~~

``dumps=slow`` writes the raw ``(A, b)`` of a solve -- before equilibration -- together with the field-block layout
active for that solve, which is what a replay needs in order to go back through this same block preconditioner without
a live model. Unlike ``dumps=every`` or a list of solve numbers, fixed *before* the run, ``slow`` triggers on the
solve's own outcome (more outer iterations than ``warnOuterIterations``, or a cap hit short of the tolerance), so a
pathological system (a late-increment, heavily damaged Jacobian, say) can be captured without knowing in advance which
solve will misbehave.

``dumpContextSolves`` exists because a single snapshot cannot distinguish "this operator
is intrinsically hard" from "this solve's own carried-over state degraded independently of the
operator" -- such as a reused, now-stale hierarchy. Both look identical from
the matrix alone, and a fresh solver replaying the matrix on its own can converge in a fraction of
the live iteration count, which is a genuinely misleading result. Capturing the preceding sequence
lets an offline replay feed one persistent solver instance the same solves in the same order.

A ``manifest.jsonl`` in the dump directory records, per dumped solve, its role (trigger or context),
its field blocks, and the solver state at that point: outer iterations, achieved true residual,
requested tolerance, continuations, and whether the solve refreshed, saw a pattern change, or was
detected as a new increment (:ref:`composed-solve-dumps`). The staleness question can often be answered from the
manifest alone, without a replay.

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
