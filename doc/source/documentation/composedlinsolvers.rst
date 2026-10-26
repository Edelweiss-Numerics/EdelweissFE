Composing a linear solver
=========================

Every linear solver of EdelweissFE is a composed solver behind one adapter (:doc:`linsolvers`): ``type=blockamg`` is
the ``blockAMG()`` preset below, ``type=pardiso`` the one leaf ``DirectSolver("pardiso")``, and ``type=custom`` the
tree of a solver script of your own.

A composed linear solver is a **tree of small objects**, built with plain constructors in a Python script next to the
input file. Each object is one textbook component -- a Krylov solver, a diagonal scaling, a correction on one field, a
multiplicative combination of corrections, an AMG hierarchy, a Schur complement -- and its ``__call__`` is the formula
of the textbook. There is one way to build each method, and it is the fast one -- with one documented exception, the
two-level cycle (see :ref:`composed-two-level-two-ways`).

This page teaches how to build a tree. The arguments and data lines of every component and of the adapter are on
:doc:`composedlinsolversreference`, the method of the field-split preconditioner on :doc:`blockamgtheory`, and the
Python API on :doc:`linsolveapi`.

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

``script`` is a path relative to the working directory, like the output files of the run (only ``*include`` is
relative to the including input file): ``script=solver.py`` finds a script next to the input file when the simulation
is started in the directory of the input file, as ``run_tests_edelweissfe`` does.

The script defines the global ``linearSolver`` and holds only the math: the tree. What the adapter does around it --
the verbosity, the dumps, the hot reload -- are data lines of the block (a ``ComposedLinearSolver`` bound in the script
is refused; see :ref:`composed-adapter`). The journal logs the structure of the solver and a hash of the script at the
first solve (with ``verbosity=info``).

How to read the nesting
-----------------------

The script defines the components **bottom-up**, with named variables: first the basic method (the block Gauss--Seidel
sweep over the fields), then what is wrapped around it, and the Krylov solver last -- the textbook order "build a basic
iteration, then accelerate it".

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

The combinators show that methods which look unrelated share one structure. Block Gauss--Seidel over the fields *is*
``Multiplicative([OnField("displacement", M_u), OnField("nonlocal damage", M_d)])``, and a two-level cycle *is*
``Multiplicative([smoother, CoarseCorrection(P, coarse), smoother])``. They are also fast: a combination asks each
correction which field it acts on, and forms the residual of a correction on field :math:`i` only on that field's
rows, :math:`r_i - \sum_{j: z_j \neq 0} A_{ij} z_j`, from the field blocks of the matrix (formed once, in one
threaded pass, for all corrections).

Wrong compositions fail as early as possible: at construction when the tree alone shows it (a child that is not a
component, an empty combination, LGMRES around a Krylov solve, a Schur complement with nothing to condense), at the
first set-up otherwise, with a message that names the cause (a field the system does not have, a one-field component
such as ``AMG`` on several fields, rigid-body modes on a scalar field or without node coordinates).

.. _composed-two-level-two-ways:

The two-level cycle: two ways, on purpose
-----------------------------------------

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

There is no native Chebyshev component yet, so the spelled-out way cannot reproduce ``TwoLevel`` exactly; a
``Chebyshev`` component is issue #237 (https://github.com/Edelweiss-Numerics/EdelweissFE/issues/237). The exception
goes away once these three pieces exist natively: the compact field-block copy kept by the system matrix, a Chebyshev
smoother component, and threaded products with :math:`P` in ``CoarseCorrection`` (with ``RigidBodyModes`` and
``P1Interpolation`` building :math:`P`). Then ``TwoLevel`` is replaced by its spelled-out form.

.. _composed-presets:

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

The 3D displacement field gets the rigid-body coarse level, ``TwoLevel(coarseSpace=RigidBodyModes())``: a first coarse
level built from all rigid-body modes, rotations included, which a bending-dominated structure needs
(:ref:`blockamg-rigid-body-coarse-level`). Switch it off where the solves converge in a few outer iterations without
it -- there it saves little, for a larger setup:

.. code-block:: edelweiss

    *linearSolver, name=translationsOnly, type=blockamg
    rigidBodyCoarseLevel=false

A 2D displacement field, or any other vector field, gets the coarse level in a ``custom`` solver script,
``OnField("displacement", TwoLevel(coarseSpace=RigidBodyModes()))``.

Everything else -- another tree for a field, the rebuild policy, the parameters of the forcing tolerance or of LGMRES,
the options of the rigid-body coarse level, the p-multigrid variant, a Schur complement -- is a ``custom`` solver
script: copy the preset (:mod:`edelweissfe.linsolve.composition.presets`, a plain function) into the script and change
what you need.

.. _composed-schur-complement-recipe:

A Schur complement
------------------

Some dofs make the field-split AMG stall: the dofs of embedded bars with **bond-slip**, for instance, tied to the
nodes of their host elements by stiff bond springs. A ``SchurComplement`` condenses such dofs exactly and builds its
child on the approximate Schur complement of all other dofs (the method: :ref:`blockamg-schur-complement`). The
condensed dofs are chosen with the **markers** of adaptive mesh refinement: the dofs of the interior nodes of the
marked elements -- the nodes all of whose elements are marked. For embedded bars, mark the bar and the bond elements:

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

The corrections here are a function of the context: they are made at the first set-up, one per field of the remaining
dofs, with the preconditioner the preset would give that field. A ``fieldOutput`` marker condenses, e.g., a softening
zone instead (testfile ``GradientEnhancedDamageSchurComplement``). The other arguments of ``SchurComplement`` are
listed in :ref:`composed-schur-complement-arguments`.

Views
-----

Every component describes itself (``describe()``); the views in :mod:`edelweissfe.linsolve.composition.views` draw
the descriptions along the tree of ``children()``. For the ``blockAMG()`` preset on a displacement / nonlocal damage
system, after two solves:

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
