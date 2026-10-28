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
        AMG, LGMRES, DiagonalScaling, Multiplicative, OnField, RigidBodyModesPerAggregate, TwoLevel,
    )

    # the displacement field: a two-level cycle on the rigid-body modes, AMG below the first coarse level
    displacement = OnField("displacement", TwoLevel(coarseSpace=RigidBodyModesPerAggregate()))
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

``LGMRES``'s defaults are the production ones: an Eisenstat--Walker tolerance per solve (``EisenstatWalkerTolerance``), a
``RebuildPolicy`` that builds the preconditioner anew for every solve (``RebuildPolicy(reusePreconditioner=True)``
keeps it across Newton iterations, :ref:`measured-rebuild-every-solve`), and the detection of a new increment
through the adapter's ``residualGrowthFactor``; for a plain solve to a fixed tolerance, write
``LGMRES(..., tolerance=FixedTolerance(1e-8))``.

``script`` is a path relative to the working directory, like the output files of the run (only ``*include`` is
relative to the including input file): ``script=solver.py`` finds a script next to the input file when the simulation
is started in the directory of the input file, as ``run_tests_edelweissfe`` does.

The script defines the global ``linearSolver`` and holds only the math: the tree. What the adapter does around it --
the verbosity, the dumps, the hot reload -- are data lines of the block (a ``ComposedLinearSolver`` bound in the script
is refused; see :ref:`composed-adapter`). The journal shows the structure of the solver and a hash of the script in a frame at the
first solve (with ``verbosity=info``), above the header of the iteration table.

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
    * - ``LGMRES(preconditioner, tolerance, maxIterations, restart, augmentationVectors, rebuild, trueResidualRestarts, residualGapCompensation)``
      - the outer Krylov solve :math:`A x = b`, right-preconditioned (AMGCL); owns when the preconditioner is rebuilt
        (``RebuildPolicy(stalenessFactor, rebuildOnNewIncrement, reusePreconditioner)``). Needs a fixed linear preconditioner, and
        refuses one that is not at construction. On a diagonally scaled matrix see :ref:`composed-scaled-krylov`.
    * - ``FGMRES(preconditioner, tolerance, maxIterations, restart)``
      - flexible GMRES (AMGCL): :math:`x = Z_m y_m` from the preconditioned vectors themselves, so the preconditioner
        may change between iterations -- e.g. an inner Krylov solve
    * - ``EisenstatWalkerTolerance(minimum, maximum)``, ``FixedTolerance(eta)``
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
        context, made at the first set-up (one per field, whatever the model). Every field must get a correction
        (else :math:`z = 0` there): the set-up raises for a field none acts on. A component is set up for one
        matrix, so one instance sits in one place of the tree (the first set-up raises otherwise); a combination may
        apply it several times
    * - ``AMG(amgclTree, backendBlockSize, backendPrecision, nearNullspace, dropTolerance, lumpDroppedEntries)``
      - one AMGCL hierarchy on a field, one cycle per application. ``amgclTree`` holds AMGCL's parameters only,
        forwarded verbatim; the backend (scalar entries or node-wise blocks, double or float) and the near null-space
        (``"rigidBodyModes"``, ``"translations"`` or ``None``) are chosen by the keywords, the field's default for
        those not given (:ref:`composed-amg-backend`); with ``dropTolerance`` built on the filtered block
        (:ref:`composed-drop-tolerance`)
    * - ``TwoLevel(coarseSpace, amgclTree, backendBlockSize, backendPrecision, dropTolerance, lumpDroppedEntries)``
      - a two-level cycle with a coarse space of its own (a ``CoarseSpace``): all rigid-body modes per aggregate,
        ``RigidBodyModesPerAggregate(prolongation, coarseHierarchyPrecision)``, or the interpolant from the corner
        nodes in the scaled variables (P1), ``CornerNodeInterpolation(coarseSolve)``, whose coarse solve is a
        component on the corner-node field (default: the coarse field solved like any field of the ``blockAMG()``
        preset under a diagonal scaling of its own, for a 3D displacement field
        ``DiagonalScaling(TwoLevel(coarseSpace=RigidBodyModesPerAggregate()))``, :ref:`blockamg-p-multigrid`). Both
        read the fine Chebyshev smoother from the tree's ``relax``, ``npre`` and ``npost``; without a tree, the default
        tree of the field (degree 16), and for ``CornerNodeInterpolation`` on a 3D displacement field the smoother of
        the corner-node two-level cycle (degree 8), on both of its levels (:ref:`composed-default-field-trees`)
    * - ``SchurComplement(condensedDofMarkers, schurComplementSolve, condensedBlockSolve, approximation, dropTolerance,
        windowNodes, haloNodes, blockFactors)``
      - block LU: :math:`z_k = M_{\tilde S}^{-1}(r_k - A_{kb} A_{bb}^{-1} r_b)`,
        :math:`z_b = A_{bb}^{-1}(r_b - A_{bk} z_k)`; ``condensedBlockSolve`` is a ``DirectSolver``, set up on :math:`A_{bb}`
        on every solve
    * - ``DirectSolver(package, **options)``
      - :math:`A^{-1}` by a registered direct solver package (``superlu``, ``umfpack``, ``pardiso``, ``klu``,
        ``mumps``, ``petsclu``, ``panuapardiso``; registry category ``directsolver``): every backend factorizes once
        per set-up (a singular matrix is reported there) and back-substitutes in every application, so it is as
        cheap inside a Krylov tree as alone; ``type=pardiso`` is ``DirectSolver("pardiso",
        reuseSymbolicFactorization=...)`` alone
    * - ``CoarseCorrection(P, coarseSolve)``, ``Jacobi(omega)``
      - :math:`P A_c^{-1} P^T` with :math:`A_c = P^T A P`, and :math:`\omega \operatorname{diag}(A)^{-1}`: the pieces
        of a two-level cycle spelled out, ``Multiplicative([Jacobi(), CoarseCorrection(P, coarseSolve), Jacobi()])``
        (about twice the wall time of ``TwoLevel``: for learning and new coarse spaces; see
        :ref:`composed-performance-of-forms`). ``P`` is a matrix, or a
        callable ``(A, context) -> (P, coarseContext)`` that also says what the coarse dofs are, e.g.
        ``context.ofCoarseNodes(field, isCorner)`` for the P1 interpolant: then the coarse solve may be any solver
        of a displacement field, ``DiagonalScaling(TwoLevel(coarseSpace=RigidBodyModesPerAggregate()))``

The combinators show that methods which look unrelated share one structure. Block Gauss--Seidel over the fields *is*
``Multiplicative([OnField("displacement", M_u), OnField("nonlocal damage", M_d)])``, and a two-level cycle *is*
``Multiplicative([smoother, CoarseCorrection(P, coarseSolve), smoother])``. They are also fast: a combination asks each
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

    TwoLevel(coarseSpace=RigidBodyModesPerAggregate())                 # the fast way
    Multiplicative([Jacobi(), CoarseCorrection(P, AMG()), Jacobi()])   # the spelled-out way

``TwoLevel`` is a leaf, like ``AMG``: pre-smoothing (Chebyshev), the coarse correction
:math:`P A_c^{-1} P^T (r - A z)` with :math:`A_c = P^T A P`, and post-smoothing run as one cycle in C++. Use it for
production runs; ``blockAMG()`` (``type=blockamg``) uses it for 3D displacement fields.

The spelled-out way builds the same cycle from general components, each of which can be replaced -- another smoother,
a coarse space of your own (any :math:`P`), a direct coarse solve. Use it to learn how a two-level cycle works and to
try a new coarse space or smoother. It is correct, but slower: on three bench systems it takes about twice the wall
time of ``TwoLevel``, with the same iterations (:ref:`measured-two-level-written-out`). Not because it is Python: its
products all run threaded in C++ -- ``Multiplicative`` forms its residuals in the threaded pass of the field split, and
``CoarseCorrection`` forms :math:`P^T r`, :math:`P z` and :math:`P^T A P` with AMGCL's threaded products -- but because
of the smoother. ``TwoLevel`` keeps one compact copy of the field block (single precision, 3x3 node blocks) for its
Chebyshev smoother, which reads half the bytes per product; a smoother component works on the general double-precision
matrix.

The testfile deck ``testfiles/edelweiss-only/TwoLevelSpelledOut`` runs the spelled-out cycle in a simulation, on
quadratic elements with the P1 interpolant as the coarse space. Its prolongation callable returns :math:`P` together
with the context of the coarse dofs, ``context.ofCoarseNodes(field, isCorner)`` -- the displacement field on the corner
nodes, with their coordinates -- so its coarse solve is the rigid-body coarse level,
``DiagonalScaling(TwoLevel(coarseSpace=RigidBodyModesPerAggregate()))``, the default of ``CornerNodeInterpolation``
for a 3D displacement field but for its smoother: this deck keeps the field's default tree (degree 16), the default
takes the weaker smoother of the corner-node two-level cycle (degree 8, :ref:`composed-default-field-trees`). (A callable whose
coarse dofs are no nodes returns ``context.ofCoarseDofs(nCoarse)``, one field of scalar dofs; a fixed matrix :math:`P`
gets that context.) :math:`P` acts on the matrix the correction is handed: the deck has no ``DiagonalScaling`` above
the cycle, so its variables are the displacements, and the plain weights :math:`\tfrac12, \tfrac12` are right. Under a
``DiagonalScaling`` the variables are :math:`D^{-1} u`, and the same interpolation of displacements is
:math:`D^{-1} P_{geo} D_c` -- the scaled weights of ``CornerNodeInterpolation`` (:ref:`blockamg-p-multigrid`), with :math:`D`
from ``context.scaling`` and :math:`D_c` from the coarse context's scaling. Unlike ``TwoLevel``, the deck's :math:`P`
keeps the Dirichlet rows, so a Dirichlet corner dof is no identity row of :math:`A_c`; the coarse solve treats it as
free, which is still a valid preconditioner. Its smoother is ``Jacobi(omega=0.4)``: damped Jacobi
damps the fast error modes only for :math:`\omega < 2 / \lambda_{max}(\operatorname{diag}(A)^{-1} A)`, and on
quadratic elements :math:`\lambda_{max}` is larger than for the Laplacian the default :math:`\omega = 2/3` is made
for (about 3.7 there), so the default amplifies them and the cycle stalls.

There is no native Chebyshev component yet, so the spelled-out way cannot reproduce ``TwoLevel`` exactly; a
``Chebyshev`` component is issue #237 (https://github.com/Edelweiss-Numerics/EdelweissFE/issues/237). The exception
goes away once the remaining two pieces exist natively: the compact field-block copy kept by the system matrix, and a
Chebyshev smoother component on it (``CoarseCorrection`` already multiplies with :math:`P` threaded). Then ``TwoLevel``
is replaced by its spelled-out form.

.. _composed-performance-of-forms:

Which form to use: what each costs
----------------------------------

Every component composes with every other, and many methods can be written in more than one way. The ways do not
cost the same. The rule: **a written-out form is for learning and for trying something new; for production runs,
take the native component where one exists.** What each form costs, as far as it was measured:

.. list-table::
    :header-rows: 1
    :widths: 32 38 30

    * - Written as
      - Costs
      - For production runs
    * - a two-level cycle spelled out, ``Multiplicative([smoother, CoarseCorrection(P, coarseSolve), smoother])``
      - about twice the wall time of ``TwoLevel``, with the same iterations (:ref:`measured-two-level-written-out`):
        a smoother component reads the double-precision matrix, ``TwoLevel`` a single-precision copy in 3x3 node
        blocks
      - ``TwoLevel(coarseSpace=RigidBodyModesPerAggregate())`` or ``TwoLevel(coarseSpace=CornerNodeInterpolation())``
    * - :math:`P` built by a callable of your own in NumPy, or in a Python loop over the nodes (the deck
        ``TwoLevelSpelledOut``)
      - one thread, on every set-up: finding the Dirichlet rows of a 0.6 -- 1.4 M dof block with NumPy alone took
        1.1 -- 2.4 s (:ref:`measured-two-level-written-out`), against a whole set-up of ``TwoLevel`` of 1 -- 4 s
      - ``CornerNodeInterpolation`` (or ``RigidBodyModesPerAggregate``), which build :math:`P` threaded in C++
    * - ``Jacobi`` as the smoother
      - a weaker smoother than Chebyshev, and only one that damps at all for :math:`\omega < 2 / \lambda_{max}`
        (0.4, not the textbook 2/3, on quadratic elements); not measured against Chebyshev on large systems
      - the Chebyshev smoother inside ``TwoLevel`` or ``AMG``
    * - ``AMG(backendBlockSize=1)`` or ``TwoLevel(..., backendBlockSize=1)`` on a 3D displacement field
      - double-precision scalar entries: the corner-node two-level cycle took 1.1 -- 2 times as long per application
        as on the default single-precision 3x3 blocks (:ref:`measured-two-level-written-out`)
      - the defaults of :ref:`composed-amg-backend`; the scalar backend only where its near null-space is needed
        (all rigid-body modes in one AMG hierarchy)
    * - a component of your own in NumPy or SciPy (:ref:`composed-own-component`)
      - NumPy and SciPy run on one thread: SciPy's sparse product too, whatever ``OMP_NUM_THREADS`` says
      - fine for trying an idea; keep its products with :math:`A` few, or make them native once it pays
    * - a preset copied into a solver script and changed
      - nothing: the components are the same as the preset's, and the composition of ``blockAMG()`` reproduced the
        former monolithic solver's iterations and, on two models, its wall time within 0.3 %
      - --

The combinators themselves cost nothing extra: ``Multiplicative`` and ``Additive`` form their residuals with the
threaded products of the field split, and ``CoarseCorrection`` its products with :math:`P` and :math:`P^T A P`
threaded too. What makes a written-out form slower is what it is written out *of*.

.. _composed-presets:

Presets
-------

Presets are plain functions that return a tree. ``blockAMG()`` is the ``blockamg`` type, its arguments the data lines
of that type (``*linearSolver, type=blockamg``), and in a script ``linearSolver = blockAMG(...)``. It builds::

    LGMRES(
        DiagonalScaling(
            Multiplicative([OnField(f, M_f) for every field f])   # M_f: TwoLevel(RigidBodyModesPerAggregate())
        )                                                         # for a 3D displacement field, else AMG()
    )

The corrections are made at the first set-up, for the fields of the model; ``sweeps`` and ``symmetric`` order them
with ``gaussSeidelOrder``. ``blockAMG(krylovOnScaledSystem=True)`` puts the scaling around LGMRES instead
(:ref:`composed-scaled-krylov`), ``trueResidualRestarts`` and ``residualGapCompensation`` go to LGMRES, and ``dropTolerance`` and
``lumpDroppedEntries`` to every field (:ref:`composed-drop-tolerance`).

Each field gets the preconditioner measured fastest for it (``blockAMGFieldPreconditioner``):

* a **quadratic** 3D displacement field (one with midside nodes) the two-level cycle on its corner nodes,
  ``TwoLevel(coarseSpace=CornerNodeInterpolation())`` (:ref:`blockamg-p-multigrid`): three levels, the quadratic field,
  its linear part on the corner nodes, and the rigid-body modes of aggregates of those;
* a **linear** 3D displacement field the rigid-body coarse level, ``TwoLevel(coarseSpace=RigidBodyModesPerAggregate())``:
  a first coarse level built from all rigid-body modes, rotations included, which a bending-dominated structure needs
  (:ref:`blockamg-rigid-body-coarse-level`);
* any other field one AMG hierarchy, ``AMG()``.

Each two-level cycle can be switched off; the rigid-body coarse level where the solves converge in a few outer
iterations without it -- there it saves little, for a larger setup:

.. code-block:: edelweiss

    *linearSolver, name=translationsOnly, type=blockamg
    rigidBodyModesPerAggregate=false
    cornerNodeInterpolation=false

A 2D displacement field, or any other vector field, gets the coarse level in a ``custom`` solver script,
``OnField("displacement", TwoLevel(coarseSpace=RigidBodyModesPerAggregate()))``.

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
    linearSolver = LGMRES(SchurComplement(condensedDofMarkers=[bars, bonds], schurComplementSolve=DiagonalScaling(blockGaussSeidel)))

.. code-block:: edelweiss

    *linearSolver, name=barSchur, type=custom
    script=solver.py

    *solver, solver=NISTParallel, name=theSolver
    linsolver=barSchur

The corrections here are a function of the context: they are made at the first set-up, one per field of the remaining
dofs, with the preconditioner the preset would give that field. A ``fieldOutput`` marker condenses, e.g., a softening
zone instead (testfile ``GradientEnhancedDamageSchurComplement``). The other arguments of ``SchurComplement`` are
listed in :ref:`composed-schur-complement-arguments`.

.. _composed-own-component:

Writing your own component
--------------------------

A component of your own subclasses :class:`~edelweissfe.linsolve.composition.base.Preconditioner` -- in the solver
script itself, if you like -- and calls ``super().__init__()`` in its constructor. It then composes with every other
component. The matrix it is given, ``A``, is a :class:`~edelweissfe.linsolve.composition.systemmatrix.SystemMatrix`,
not a SciPy matrix: ``A.matrix()`` gives the SciPy CSR matrix (the system matrix also forms the field blocks and the
scaled matrix for the components that need them).

.. note::

    A component of your own in NumPy or SciPy runs on one thread (SciPy's sparse products too): fine for trying an
    idea, but slower than the native components on a large model (:ref:`composed-performance-of-forms`).

Four hooks make a component:

``_setUp(A, context)``
    Build what the application needs for the matrix ``A``, e.g. the inverse of its diagonal. ``context`` (a
    :class:`~edelweissfe.linsolve.composition.base.SetUpContext`) holds the fields of the sub-problem and their node
    coordinates.
``_forget()``
    Drop what ``_setUp`` built: set each of those attributes to its value before the first set-up
    (``self.inverseDiagonal = None``). The constructor of the base class calls it, so this is where they are declared.
    The public ``setUp`` calls ``forget()`` (this hook and those of the children) before ``_setUp``: the old
    factorization or hierarchy is freed before the new one is built, never both at once, which on a large model is
    gigabytes. What is reused on purpose (a symbolic factorization, an estimate carried from one build to the next)
    is not forgotten.
``_apply(r)``
    Return :math:`z = M^{-1} r`.
``_update(A, change)``
    Take the matrix of the next Newton iteration. By default it builds anew (it calls ``self.setUp``); override it to
    reuse what you built, rebuilding (again by ``self.setUp``) only if ``change.requiresRebuild`` (a new increment, a
    new sparsity pattern, a stale preconditioner, as the ``MatrixChange`` and the Krylov solver above decide) or if
    nothing is built -- after ``forget()``, which a Krylov solver calls on its preconditioner before it rebuilds it,
    and the adapter on the whole tree when the model changes.

Then ``describe()`` returns a :class:`~edelweissfe.linsolve.composition.base.Description` -- the name, the formula
(unicode and LaTeX), the parameters and the steps of one application -- which the views show. The public
``setUp``, ``update`` and ``__call__`` of the base class record each call and its time in the component's statistics
and call the hook; a component calls its children through those public methods.

A complete component, run as a deck of the test suite (``testfiles/edelweiss-only/OwnComponentSolverScript``):

.. literalinclude:: ../../../testfiles/edelweiss-only/OwnComponentSolverScript/solver.py
   :language: python

The rest of the protocol of :class:`~edelweissfe.linsolve.composition.base.Preconditioner` has defaults that fit a
leaf; override a method only in the case named:

* ``children()``: a component with children returns its ``(role, child)`` pairs, in the order it uses them. The tree
  is walked through it (the model, the end of a solve, the questions below, the views), and the role is the
  placeholder of the child in the component's formula.
* ``fieldIndex`` and ``_applyRestricted(r_f)``: a component that acts on one field returns its index in the context;
  a combination then forms the residual of that field alone and calls ``applyRestricted`` with it.
* ``withRebuildReasons(change)``: a component that adds reasons for a rebuild of its own (a Krylov solver's
  ``RebuildPolicy``), or keeps something built for the matrix above such a child (the :math:`D` of a
  ``DiagonalScaling`` around a Krylov solve), so that it rebuilds together with the child.
* ``setModel(model, dofManager)``: a component that derives something from the model, e.g. which dofs a marker
  selects.
* ``finishSolve()``: a component that holds something only one solve needs (e.g. the field blocks of its matrix)
  releases it here, so that no copy of the last matrix is kept while the nonlinear solver assembles the next one.
* ``isFixedLinearOperator``: false for a component whose :math:`M^{-1} r` is not one fixed linear operator between two
  set-ups (a Krylov solve inside); ``LGMRES`` refuses such a preconditioner, ``FGMRES`` takes it.
* ``containsKrylovSolver``: true for a Krylov solver (the nonlinear solver then keeps the sparsity pattern stable,
  and the solve is recorded in the "linear solve" column).
* ``builtMatrices()``, ``solveRecord()``, ``condensedDofs()``: the matrices a component formed (for ``views.spy``),
  what its last Krylov solve did, and the dofs a Schur complement condensed (for the dumps).

Views
-----

Every component describes itself (``describe()``); the views in :mod:`edelweissfe.linsolve.composition.views` draw
the descriptions along the tree of ``children()``. For the ``blockAMG()`` preset on a displacement / nonlocal damage
system, after two solves:

``print(solver)`` -- the structure::

    LGMRES(tolerance=EisenstatWalkerTolerance(minimum=0.0001, maximum=0.001), maxIterations=100, restart=50, augmentationVectors=3)
    └─ DiagonalScaling(D=diag(1/√|aᵢᵢ|))
       └─ Multiplicative
          ├─ M1: OnField(field=displacement)
          │  └─ TwoLevel(coarse space=rigid-body modes, prolongation=smoothed, coarse solve=AMG V-cycle)
          └─ M2: OnField(field=nonlocal damage)
             └─ AMG(amgclTree=default)

``views.formula(solver)`` -- the composed approximate inverse (``latex=True`` for LaTeX)::

    LGMRES(A, D · (Rᵀ AMG(nonlocal damage)⁻¹ R ∘ Rᵀ TwoLevel(displacement; rigid-body modes)⁻¹ R) · D)⁻¹

``views.runStatistics(solver)`` -- what each component recorded (inclusive times)::

    LGMRES(...)   updated 2× 0.07s · applied 2× 0.05s
    └─ DiagonalScaling(...)   updated 2× 0.07s · applied 18× 0.04s
       └─ Multiplicative   set up 1× 0.06s · updated 2× 0.06s · applied 18× 0.04s
          ├─ M1: OnField(...)   set up 1× 0.06s · updated 1× 0.00s · applied 18× 0.03s
          │  └─ TwoLevel(...)   set up 1× 0.06s · ... · applied 18× 0.03s · rows 9555 · aggregates 198 · coarse dofs 1188
          └─ M2: OnField(...)   set up 1× 0.00s · updated 1× 0.00s · applied 18× 0.00s
             └─ AMG(...)   set up 1× 0.00s · ... · applied 18× 0.00s · rows 3185

The adapter only ever updates the root; a combination whose corrections are made per field sets them up at its first
update (a set-up within that update), and reuses them at the next.

``views.dataFlow(solver)`` -- one application as a graphviz graph (DOT, or SVG with the ``graphviz`` package): each
component a box around its steps, a step that delegates leading into its child's box. ``views.spy(solver, A)`` draws
the sparsity patterns of the matrices the components were built on, ``views.writeHtml(solver, "solver.html")`` writes
all views into one page, and in Jupyter a solver shows itself as that page.
