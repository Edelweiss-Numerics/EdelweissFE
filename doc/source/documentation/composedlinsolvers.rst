Composed linear solvers
=======================

.. note::

    A prototype. ``blockamg`` remains the production solver; the composed ``blockAMG()`` preset reproduces it (same
    outer iterations, same solution) and is meant to replace its internals.

A composed linear solver is a **tree of small objects**, built with plain constructors in a Python script next to the
input file. Each object is one textbook component -- a Krylov solver, a diagonal scaling, a correction on one field, a
multiplicative combination of corrections, an AMG hierarchy, a Schur complement -- and its ``__call__`` is the formula
of the textbook. There is one way to build each method, and it is the fast one.

A solver script
---------------

.. code-block:: python

    # solver.py -- the linear solver of a displacement / nonlocal damage model
    from edelweissfe.linsolve.composition import (
        AMG, LGMRES, DiagonalScaling, ForcingTolerance, Multiplicative, OnField, RigidBodyModes, TwoGrid,
    )

    # the displacement field: a two-grid cycle on the rigid-body modes, AMG below the first coarse level
    displacement = OnField("displacement", TwoGrid(coarseSpace=RigidBodyModes()))
    # the nonlocal damage field: one AMG hierarchy
    damage = OnField("nonlocal damage", AMG())

    # both fields, one block Gauss-Seidel sweep: displacement first, then the nonlocal damage
    blockGaussSeidel = Multiplicative([displacement, damage])

    linearSolver = LGMRES(DiagonalScaling(blockGaussSeidel), tolerance=ForcingTolerance(maximum=1e-3))

and in the input file:

.. code-block:: edelweiss

    *solver, solver=NIST, name=theSolver
    linsolver=script:solver.py:linearSolver

(``script:solver.py`` alone takes the global ``linearSolver``.) The script runs with the global ``options`` set to the
parsed ``linsolverConfigFile``, if the solver has one. The journal logs the structure of the solver and a hash of the
script at the first solve (with ``verbosity="info"``).

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
    * - ``LGMRES(preconditioner, tolerance, maxIterations, M, K)``
      - the outer Krylov solve :math:`A x = b`, right-preconditioned (AMGCL); owns when the preconditioner is rebuilt
        (``RebuildPolicy``). Needs a fixed linear preconditioner, and refuses one that is not at construction.
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
    * - ``AMG(tree)``
      - one AMGCL hierarchy on a field, one cycle per application
    * - ``TwoGrid(coarseSpace=RigidBodyModes() | P1Interpolation())``
      - a two-grid cycle with a coarse space of its own: all rigid-body modes per aggregate, or the P1 interpolant
    * - ``SchurComplement(condense, preconditionS, solveA_bb, blockFactors)``
      - block LU: :math:`z_k = M_{\tilde S}^{-1}(r_k - A_{kb} A_{bb}^{-1} r_b)`,
        :math:`z_b = A_{bb}^{-1}(r_b - A_{bk} z_k)`
    * - ``DirectSolver(linsolver)``
      - :math:`A^{-1}` by a registered direct solver (``superlu``, ``pardiso``, ...)
    * - ``CoarseCorrection(P, coarse)``, ``Jacobi(omega)``
      - :math:`P A_c^{-1} P^T` with :math:`A_c = P^T A P`, and :math:`\omega \operatorname{diag}(A)^{-1}`: the pieces
        of a two-grid cycle spelled out, ``Multiplicative([Jacobi(), CoarseCorrection(P, coarse), Jacobi()])``

The combinators show that methods which look unrelated share one structure. Block Gauss--Seidel over the fields *is*
``Multiplicative([OnField("displacement", M_u), OnField("nonlocal damage", M_d)])``, and a two-grid cycle *is*
``Multiplicative([smoother, CoarseCorrection(P, coarse), smoother])``. They are also fast: a combination asks each
correction which field it acts on, and forms the residual of a correction on field :math:`i` only on that field's
rows, :math:`r_i - \sum_{j: z_j \neq 0} A_{ij} z_j`, from the field blocks of the matrix (formed once, in one
threaded pass, for all corrections). ``TwoGrid`` is the one exception still standing: its cycle runs in C++, on the
smoother's own (float, node-block) copy of the field block, and the spelled-out composition is not yet as fast.

Wrong compositions fail as early as possible: at construction when the tree alone shows it (a child that is not a
component, an empty combination, LGMRES around a Krylov solve, a Schur complement with nothing to condense), at the
first set-up otherwise, with a message that names the cause (a field the system does not have, a one-field component
such as ``AMG`` on several fields, rigid-body modes on a scalar field or without node coordinates).

A component of your own subclasses ``Preconditioner`` -- in the script itself, if you like -- and implements
``setUp(A, context)``, ``__call__(r)`` and, to reuse what it built, ``update(A, change)``. It then composes with every
other component.

Presets
-------

Presets are plain functions that return a tree. ``blockAMG()`` builds the equivalent of ``blockamg`` with its
defaults::

    LGMRES(
        DiagonalScaling(
            Multiplicative([OnField(f, M_f) for every field f])   # M_f: TwoGrid(RigidBodyModes()) for a
        )                                                         # 3D displacement field, else AMG()
    )

The corrections are made at the first set-up, for the fields of the model; ``sweeps`` and ``symmetric`` order them
with ``gaussSeidelOrder``.

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
    └─ DiagonalScaling(...)   set up 1× 0.06s · updated 2× 0.07s · applied 18× 0.04s
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

Performance: a diagonal scaling does not form :math:`D A D`; the first component below it that needs a block forms
:math:`\hat A` and all field blocks in one threaded native pass, kept with the matrix for all others, which also keeps :math:`A` as the operator of the outer Krylov solve
(:class:`~edelweissfe.linsolve.composition.systemmatrix.SystemMatrix`).

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
