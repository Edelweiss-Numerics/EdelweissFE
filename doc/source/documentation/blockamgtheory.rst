Theory: the field-split AMG preconditioner
==========================================

This page explains a *field-split* algebraic-multigrid preconditioner for the coupled multi-field equation systems
EdelweissFE's implicit solvers produce -- displacement coupled to a scalar nonlocal (gradient-enhanced) damage field,
typically with penalty contact, tie constraints and adaptive refinement on top -- and the outer Krylov solve it
preconditions.

The method is a tree of components (:doc:`composedlinsolvers`), ``LGMRES(DiagonalScaling(Multiplicative([OnField(f,
M_f), ...])))``, and every step below is one of them, named in the heading of its section:
:class:`~edelweissfe.linsolve.composition.diagonalscaling.DiagonalScaling` (the equilibration),
:class:`~edelweissfe.linsolve.composition.combinators.OnField` (the field split),
:class:`~edelweissfe.linsolve.composition.amg.AMG` (one hierarchy per field),
:class:`~edelweissfe.linsolve.composition.combinators.Multiplicative` (the block Gauss--Seidel sweep),
:class:`~edelweissfe.linsolve.composition.krylov.LGMRES` with its
:class:`~edelweissfe.linsolve.composition.krylov.ForcingTolerance` and
:class:`~edelweissfe.linsolve.composition.krylov.RebuildPolicy` (the outer solve, its tolerance, the reuse of the
hierarchies), :class:`~edelweissfe.linsolve.composition.twolevel.TwoLevel` (the p-multigrid variant and the rigid-body
coarse level) and :class:`~edelweissfe.linsolve.composition.schurcomplement.SchurComplement` (the condensation). The
near null-spaces are built in :mod:`~edelweissfe.linsolve.nullspace`. The AMG hierarchies and the outer solve need the
optional ``amgcl`` extension (:mod:`edelweissfe.linsolve.amgcl.amgcl`).

The preset :func:`~edelweissfe.linsolve.composition.presets.blockAMG` builds this tree with its defaults; it is the
linear solver type ``blockamg`` (``linsolver=blockamg``, or a ``*linearSolver, type=blockamg`` block, see
:doc:`linsolvers`). Each setting below is named where it lives: an argument of a component in a ``custom`` solver
script, and for the settings the preset exposes, a data line of ``type=blockamg``
(:ref:`composed-presets`). The measurements behind the defaults are collected on their own page,
:doc:`fieldsplitmeasurements`, so that this page reads as the method alone; the arguments and defaults of the
components are on :doc:`composedlinsolversreference`.


Why a field-split preconditioner
--------------------------------

On large coupled fracture models a *direct* factorization eventually stops being an option: its
fill-in grows superlinearly, and past roughly :math:`10^6` DOFs it exceeds the available memory
regardless of how much time one is willing to spend. Algebraic multigrid (AMG) has :math:`O(n)`
memory and is the route to those sizes.

Applied **monolithically** to the coupled system, however, AMG does not work: a single hierarchy
cannot represent the disparate physics and scales of the fields at once. Measured directly on a real
coupled system, a monolithic hierarchy's residual reduction plateaus around 0.2 instead of
converging, and it does so regardless of how the near-null-space is chosen -- the missing ingredient
is the field-block structure itself, not a better single hierarchy.

The remedy, following the block-preconditioning strategy of Alkmim et al. (IJNME 2026), is to
precondition **blockwise**: build one AMG hierarchy per *physical field*, combine them with a block
Gauss--Seidel sweep, and use that sweep as the preconditioner of an outer Krylov solve over the full
coupled system. Each field's operator is individually AMG-friendly -- elasticity for a displacement
field, a screened-Poisson (Helmholtz) operator for a gradient-enhanced damage field (see
:doc:`adaptivitytheory` for that operator) -- even though their monolithic coupling is not.

The field-split preconditioner buys reach, not necessarily speed.
On the reference models AMGCL's smoothed aggregation converges, but not tightly, on the
non-symmetric (contact-condensed) displacement block, so the outer Krylov solve needs
many iterations -- with the former defaults typically :math:`O(100)`; with the current
defaults (see :ref:`measured-defaults`), mostly 5--50. The hardest damage-onset
solves stop at its cap of 100 iterations
(``maxIterations``), return their iterate, and the hierarchies are rebuilt for
the next solve. The cap is a deliberate budget: a solve that needs more is taken as a sign that the
hierarchies no longer fit the Jacobian, not given more iterations. That is an acceptable price where the
alternative is not fitting in memory at all; at problem sizes a direct
factorization still handles comfortably, a direct solver (e.g. ``pardiso``) can be competitive or
faster. Which one wins depends on problem size, conditioning, and
how severely damage and contact nonlinearity degrade the per-field hierarchies on a given increment.


Where the block structure comes from
------------------------------------

Nothing about the block layout is configured by hand. A sparse matrix does not carry the information
of which DOF belongs to which physical field, so it is pushed in from the model side:
:meth:`~edelweissfe.linsolve.base.LinearSolver.setModel` hands every registered linear solver the
live :class:`~edelweissfe.models.femodel.FEModel` and
:class:`~edelweissfe.numerics.dofmanager.DofManager`, and the base class derives from them the
ordered list of :class:`~edelweissfe.linsolve.base.FieldBlock` records -- field name, half-open DOF
range ``[start, stop)``, and nodal dimension (3 for a 3D displacement field, 1 for a scalar damage
field). The nonlinear solver calls it whenever the equation system is (re)built: on the first solve,
and again after every adaptive-refinement or connectivity change, so the layout tracks the mesh.

Two consequences worth knowing:

* The fields must **tile the DOF vector contiguously** in DOF order. They do, by construction, for
  every model the DOF manager builds; a gap or overlap raises an error rather than being silently
  worked around. DOFs past the last node field (scalar variables) are folded into one trailing scalar
  block named ``scalar variables``. In EdelweissFE these are at present the Lagrange multipliers of
  constraints (rigid bodies, equal-value constraints), whose rows have a zero diagonal: such a system is
  refused (see the equilibration step below), so the block only serves scalar variables with a diagonal
  of their own.
* Beyond the field ranges, the components read **node coordinates** from the model on their own, lazily,
  for the near-null-space construction described below (and, for the opt-in p-multigrid
  variant, the corner/midside element topology). Nothing has to be pushed in ahead of it.

:meth:`~edelweissfe.linsolve.base.LinearSolver.setFieldStructure` is a lower-level escape hatch for a
caller that knows the field blocks but has no model to hand over -- e.g. an offline script replaying
a captured system through the solver. On that path node coordinates are unavailable unless given
(:ref:`composed-adapter`), and an ``AMG`` hierarchy degrades gracefully to a translations-only near-null-space.


The solve pipeline
------------------

The linear solver sees only the condensed, Dirichlet-eliminated system: the nonlinear solver applies the
multi-point-constraint transform :math:`\hat K = T^\mathsf{T} K T + I_s` (hanging nodes, ties; the
identity :math:`I_s` fills the structurally zero slave rows and columns, and the slave DOFs are
recovered from their masters after the solve, so the transform keeps a symmetric :math:`K` symmetric
and adds no slave-to-master coupling that the equilibration below could blow up) and then
the symmetric Dirichlet elimination before calling the linear solver: each constrained row becomes
the identity row, and each constrained column is zeroed after its coupling to the prescribed value
has been moved to the right-hand side -- so a symmetric operator stays symmetric, as smoothed
aggregation and the Chebyshev smoother assume. Both steps are size-preserving and leave the DOF
manager's ordering intact, so the field blocks still describe the matrix that actually arrives.

The MPC condensation is skipped entirely while no multi-point constraint has a record -- e.g. an
``hAdaptivity`` before its first refinement, whose hanging-node constraint is registered up front.
This matters for hierarchy reuse: the condensed matrix would be pruned of exact zeros
(``pruneCondensedMatrixZeros``, a nonlinear-solver option), which changes its sparsity pattern from
one Newton iteration to the next, and a changed pattern forces a hierarchy rebuild (step (4) below).
Left unset, that option is automatic and leaves pruning off for a linear solver with a Krylov solve in its tree
(it reports :attr:`~edelweissfe.linsolve.base.LinearSolver.needsStableSparsityPattern`), so this cost is not paid
by default; an explicit ``pruneCondensedMatrixZeros = true`` in the nonlinear solver's own
configuration would still force it. The uncondensed matrix keeps the assembler's fixed pattern.

Beyond that matrix and right-hand side, the linear solver is told nothing about the Newton loop -- like
every registered linear solver it is called as ``(A, b) -> x``. Both of its stateful mechanisms
(adaptive tolerance and hierarchy reuse) therefore *reconstruct* the loop state from
:math:`\lVert b \rVert` and from the matrix's own bookkeeping across calls.

The symbols of this page, their variables in the worked example (:ref:`blockamg-worked-example`), and the
components that hold them:

.. list-table:: Notation
    :widths: 18 30 26 26
    :header-rows: 1

    * - Symbol
      - Meaning
      - Worked example
      - Component
    * - :math:`A, b, x`
      - the system as the nonlinear solver hands it over, and its solution
      - ``A``, ``b``, ``x``
      - the adapter's call ``x = solver(A, b)`` (``ComposedLinearSolver``)
    * - :math:`i = 1 \dots n`
      - the fields (DOF ranges, in DOF order)
      - ``fieldBlocks`` (``FieldBlock``), ``fields`` (their slices)
      - ``OnField(name, M)``, one per field
    * - :math:`D`
      - the diagonal scaling, entries :math:`\lvert a_{ii} \rvert^{-1/2}`
      - ``scaling`` (its diagonal), ``D``
      - ``DiagonalScaling``
    * - :math:`\hat A = D A D,\ \hat b = D b`
      - the scaled (equilibrated) system, :math:`x = D z`
      - ``A_scaled``, ``b_scaled``
      - formed for the components below a ``DiagonalScaling`` (``SystemMatrix``)
    * - :math:`\hat A_{ij}`
      - its field blocks
      - ``A_blocks[i][j]``
      - the field blocks of the ``SystemMatrix``, read by ``OnField`` and ``Multiplicative``
    * - :math:`M_i^{-1}`
      - one AMG cycle on :math:`\hat A_{ii}`
      - ``fieldAMG[i].applyPreconditioner``
      - ``AMG()`` (or ``TwoLevel(...)``) inside ``OnField``
    * - :math:`M^{-1}`
      - the block Gauss--Seidel sweep, :math:`v = M^{-1} r`
      - ``blockGaussSeidel``
      - ``Multiplicative([OnField(...), ...])``
    * - :math:`\hat A M^{-1} y = \hat b`
      - the outer Krylov solve, right-preconditioned
      - ``gmres(A_scaled_Minv, b_scaled, ...)``
      - ``LGMRES``
    * - :math:`\eta`
      - the requested relative tolerance of one solve
      - ``rtol`` of ``gmres``
      - ``ForcingTolerance`` (the default of ``LGMRES``), or ``FixedTolerance(eta)``

.. code-block:: text

    Newton iteration k, in the nonlinear solver
       assembled K (VIJ -> CSR)  and  residual R
                |
                |  MPC condensation      K <- T' K T + I_s     both size preserving:
                |  Dirichlet elimination  row, column i -> e_i  DOF ordering unchanged
                |
                |  setModel(model, dofManager)  -- on a (re)build only
                |      -> field blocks, nodal dimensions, node coordinates
                v
    (A, b) ---> the composed linear solver
                |
          (1)   Newton state, from ||b|| and the solver's own bookkeeping
                    ||b|| > residualGrowthFactor * ||b||_prev  ?
                        yes -> a new increment: eta = its maximum,
                               refresh (with rebuildOnNewIncrement)
                        no  -> eta from Eisenstat-Walker forcing
                    refresh the hierarchies?  first solve | block layout | size |
                        nnz differs from the rebuild's | new increment | setModel |
                        hot reload | previous solve stale or at the iteration cap
                |
          (2)   equilibration     Ahat = D A D,  bhat = D b,  D = diag(1/sqrt|A_ii|)
                                  (no refresh: the D of the rebuild solve)
                |
          (3)   field split       diagonal blocks Ahat_ii  +  couplings Ahat_ij
                |
          (4)   hierarchies       refresh -> build one AMG hierarchy per field
                                     (smoothed aggregation + Chebyshev,
                                      per-field near-null-space)
                                  otherwise -> reuse the standing ones
                |
          (5a)  preconditioner    M = block Gauss-Seidel sweep over the fields,
                                  one AMG cycle per diagonal block
                |
          (5b)  outer Krylov solve, preconditioned by M
                    krylovOnScaledSystem = true:  on Ahat z = bhat, x = D z
                    krylovOnScaledSystem = false: on A x = b, with D M^-1 D as the preconditioner
                |
          (6)   measure the TRUE residual  ||A x - b|| / ||b||
                    > eta and continuations left (scaled system only)?
                        yes -> warm restart at a tighter tolerance, back to (5b)
                        no  -> done
                |
          (7)   record ||b||, eta and the outer iteration count for the next solve
                (after a rebuild also nnz and the staleness reference count)
                v
    x ------> nonlinear solver
                |
                |  MPC slave recovery    x_s <- x_s + W x      (master-interpolated part)
                v
              the Newton correction

In matrix form
~~~~~~~~~~~~~~

With the fields numbered in DOF order, :math:`\hat A` splits into blocks :math:`\hat A_{ij}`. The
preconditioner of step (5a), :math:`v = M^{-1} r`, is one forward block Gauss--Seidel sweep with the
per-field AMG cycles :math:`M_i^{-1}` in place of exact block solves,

.. math::

   M^{-1} = (B + L)^{-1}, \qquad B = \operatorname{blockdiag}(M_1, \dots, M_n), \qquad
   L = \text{the blocks } \hat A_{ij},\ i > j,

applied by forward substitution, :math:`v_i = M_i^{-1} \bigl( r_i - \sum_{j < i} \hat A_{ij} v_j \bigr)`.
With exact :math:`M_i = \hat A_{ii}` and a block lower triangular :math:`\hat A`, it is :math:`\hat A^{-1}`.
The outer Krylov method (``LGMRES``, AMGCL's implementation) solves the right-preconditioned system for its own
unknown :math:`y`,

.. math::

   \hat A M^{-1} y = \hat b, \qquad z = M^{-1} y, \qquad x = D z,

whose residual :math:`\hat b - \hat A M^{-1} y = \hat b - \hat A z = D (b - A x)` is the scaled one --
hence the true-residual check of step (6). With
``krylovOnScaledSystem = false`` (the default, ``LGMRES(DiagonalScaling(...))`` rather than
``DiagonalScaling(LGMRES(...))``) the outer method solves the unscaled
system instead, with the same :math:`M` between two scalings,

.. math::

   A \, (D M^{-1} D) \, y = b, \qquad x = D M^{-1} D \, y,

and its own (right-preconditioned) residual :math:`b - A x` is the true one. Steps (1)--(7) are the components of
the tree, one section each below.


Diagonal equilibration: ``DiagonalScaling``
-------------------------------------------

A coupled multi-field system has a large diagonal dynamic range: elastic stiffnesses, a
Helmholtz-like damage operator and penalty contact stiffnesses sit in the same matrix, and the
Dirichlet-eliminated rows carry a unit diagonal among them. AMG's strength-of-connection test is
scale-sensitive, so every solve begins with a symmetric diagonal (Jacobi) equilibration

.. math::

   \hat A = D A D, \qquad \hat b = D b, \qquad x = D z,
   \qquad D = \operatorname{diag}\!\left( \lvert A_{ii} \rvert^{-1/2} \right),

(in the code, the diagonal of :math:`D` is the vector ``scaling``), a two-sided diagonal scaling:
:math:`\hat A z = \hat b` with :math:`x = D z` is the same system in other variables, whatever
:math:`D` is. That is what makes it possible to *reuse* an older :math:`D` together with a reused
hierarchy (see below): a stale scaling or hierarchy costs outer iterations, and the accuracy of a solve
only if it drives that solve to the iteration cap.

The field split: ``OnField``
----------------------------

Each ``OnField(name, M)`` acts on the dofs of one field, :math:`R^T M^{-1} R`. The matrix the tree is set up on
(:class:`~edelweissfe.linsolve.composition.systemmatrix.SystemMatrix`) splits
:math:`\hat A` along the field ranges into diagonal blocks :math:`\hat A_{ii}` (one per
field, the operators the AMG hierarchies are built for) and the off-diagonal coupling blocks
:math:`\hat A_{ij}`, which are needed on every solve regardless of whether the hierarchies are
rebuilt. All of these blocks, and the full-system operator used for the outer matrix-vector products
and the true-residual check, are handed to AMGCL's OpenMP-threaded sparse matrix-vector kernels:
SciPy's own sparse ``matvec`` is single-threaded C code irrespective of ``OMP_NUM_THREADS``, and on a
solver whose inner loop is nothing but sparse matrix-vector products that is a substantial share of
the wall-clock. They are rebuilt on every call, because the values of :math:`A` -- and with them those
of :math:`D A D` -- change on every Newton iteration; a stable sparsity pattern would only save the
counting pass.

.. _blockamg-per-field-amg:

Per-field AMG hierarchies: ``AMG``
----------------------------------

Each diagonal block gets its own AMGCL hierarchy, applied on every visit of the sweep to its field
(once per outer Krylov iteration with ``symmetric = false``; with ``symmetric = true`` the reverse sweep
revisits all fields but the last) and reused across Newton iterations while it still fits the Jacobian
(see the reuse rules below). The defaults differ by nodal dimension (:func:`~edelweissfe.linsolve.composition.amg.defaultFieldTree` in
the code): a **3D vector** field (e.g. displacement) is stored in :math:`3\times3` nodal blocks in
float, coarsened by smoothed aggregation with a strong-connection threshold of 0.01 and
``estimate_spectral_radius`` (20 power iterations) from the translations of its equilibrated block (see
:ref:`below <blockamg-block-nullspace>`), and smoothed by a degree-16 Chebyshev polynomial
(one post-smoothing sweep, no pre-smoothing) -- and its hierarchy starts with the rigid-body coarse level, a first
coarse level of all its rigid-body modes, rotations included (see :ref:`below <blockamg-rigid-body-coarse-level>`);
a **2D vector** field gets the same coarsening and
smoothing but in :math:`2\times2` blocks, in double (this wrapper has no float 2x2 backend); a vector
field of any other dimension gets the same recipe on the plain scalar backend -- the only backend that
takes the full rigid-body near-null-space (see below), which keeps that path available for a field
shape the block backend was never tuned for; a **scalar** field takes the plain scalar tree (smoothed
aggregation, Chebyshev at AMGCL's own defaults). Any field can be given another tree in a ``custom`` solver
script, ``OnField(name, AMG(tree=...))``; the default trees are listed in :ref:`composed-default-field-trees`.

The tree holds AMGCL parameters only. Three keyword arguments of ``AMG`` (the first two also of ``TwoLevel``) select
the backend the hierarchy is built on; left unset, :func:`~edelweissfe.linsolve.composition.amg.backendOf` chooses
them from the field:

``backendPrecision``
    ``"double"`` or ``"float"``; by default ``"float"`` on :math:`3\times3` blocks and ``"double"`` otherwise.
    A single-precision backend should in principle roughly
    halve the hierarchy's memory traffic, but measured across a full set of captured reference
    systems it inflated outer iteration counts and netted only a small gain in aggregate --
    plausibly because the Chebyshev smoother's own spectral-radius estimate loses accuracy in
    ``float32``, degrading smoother quality enough to eat the bandwidth saving. That holds for the
    scalar backend, which builds the whole hierarchy in single precision. With
    ``backendBlockSize = 3``, ``"float"`` instead sets the hierarchy up in double -- coarsening,
    Galerkin products, every spectral-radius estimate, the coarsest direct solve -- and stores only
    the level matrices the cycle reads in float, with double vectors and double accumulation.

``backendBlockSize``
    Defaults to the field's own nodal dimension, for a block-valued backend operating on
    :math:`B \times B` nodal blocks: ``3`` for a 3D vector field, ``2`` for a 2D one, ``1`` (scalar)
    otherwise. ``3`` for a displacement field was measured faster per outer iteration (less index
    traffic) and validated inside the block Gauss--Seidel sweep on captured and live runs. Note that a block
    hierarchy has the identity block as its near-null-space, unless ``blockTranslationNullspace`` says otherwise:
    constants per component *in the equilibrated variables*. Those are
    :math:`D`-weighted translations, :math:`D` times a constant, not the translations :math:`D^{-1} t`
    of :math:`\hat A` (see below); the two agree only where the diagonal is uniform, which it is not on
    quadratic elements (corner and midside nodes) or across a damage band. A block hierarchy also
    has no **rotational** rigid-body modes, which the scalar backend carries (described next);
    ``blockTranslationNullspace``, on by default on a block backend, gives it the translations, and the rigid-body
    coarse level (``TwoLevel(coarseSpace=RigidBodyModes())``, the default for the 3D displacement field) a first coarse
    level of all rigid-body modes (see :ref:`below <blockamg-rigid-body-coarse-level>`).

``blockTranslationNullspace``
    On by default on a block backend (``backendBlockSize`` 2 or 3): build smoothed aggregation with the translations
    :math:`D^{-1} t` of :math:`\hat A_{ii}` as its near-null-space, instead of AMGCL's identity blocks -- see
    :ref:`the near-null-space of a block backend <blockamg-block-nullspace>` below. The aggregation, the prolongation
    smoothing and the smoother stay AMGCL's. It needs smoothed aggregation, which is checked: a tree with another
    coarsening needs ``blockTranslationNullspace=False``. On the scalar backend, which takes the rigid-body
    modes, the translations among them, the keyword changes nothing and either value is accepted, so
    ``AMG(backendBlockSize=1)`` needs nothing else.

.. _blockamg-chebyshev:

Near-null-space and the Chebyshev smoother
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Smoothed aggregation and the Chebyshev smoother have to be understood together, because they play
deliberately complementary roles and only work if they agree on what "smooth error" means.

* The **smoother** damps error components whose eigenvalues lie inside an estimated window
  :math:`[\,\mathrm{lower},\ \mathrm{higher}\,]\cdot\rho` of the operator's spectral radius
  :math:`\rho`. With the vector-field default ``lower = 0.01``, eigenvalues below 1% of the spectral
  radius are by construction **not** targeted by the smoother at all.
* Everything below that window -- including the operator's **near-null-space**, the directions in
  which it has almost no stiffness -- is left to the coarse-grid correction, which removes it only to
  the extent that smoothed aggregation's prolongation operator can represent it. Smoothed aggregation
  builds that prolongation from a supplied near-null-space basis.

The Chebyshev smoother of degree :math:`k` is :math:`k` steps of the Chebyshev iteration on
:math:`\hat A_{ii} u = r_i` for the interval :math:`[\ell, h] = [\mathrm{lower}, \mathrm{higher}]\cdot\rho`,
with centre :math:`d = (h + \ell)/2` and half-width :math:`c = (h - \ell)/2`:

.. math::

   p_0 = \tfrac{1}{d}\, s_0, \qquad
   p_k = \alpha_k s_k + (\alpha_k d - 1)\, p_{k-1}, \qquad
   u_{k+1} = u_k + p_k, \qquad s_k = r_i - \hat A_{ii} u_k,

with :math:`\alpha_1 = 2d / (2d^2 - c^2)` and :math:`\alpha_k = (d - \alpha_{k-1} c^2 / 4)^{-1}`
(AMGCL's ``relaxation::chebyshev``). It damps every eigencomponent in :math:`[\ell, h]` by the
Chebyshev polynomial's bound, and each step costs one matrix-vector product with the block: ``degree``
16 with ``npre = 0``, ``npost = 1`` (the vector-field default) costs 16 products per cycle on the
finest level, ``degree`` 5 with ``npre = npost = 1`` (the former vector-field default) 10.

For 3D linear elasticity the physical near-null-space is the six-dimensional space of **rigid-body
motions**: three translations and three infinitesimal rotations, each of which costs zero elastic
strain energy. Of the whole, constrained operator they are not eigenvectors -- the Dirichlet rows hold
them -- but they are *locally* near-null: restricted to an aggregate away from the constrained boundary,
a rigid-body motion is almost free of energy, and that local property is what smoothed aggregation
needs. :func:`~edelweissfe.linsolve.nullspace.rigidBodyNullspace` builds that full basis from the
field's node coordinates, about the field's own coordinate centroid, node-major to match the DOF
vector's layout:

.. math::

   \boldsymbol{t}_1 = (1,0,0), \quad \boldsymbol{t}_2 = (0,1,0), \quad \boldsymbol{t}_3 = (0,0,1),

.. math::

   \boldsymbol{r}_x = (0,-z,y), \quad \boldsymbol{r}_y = (z,0,-x), \quad \boldsymbol{r}_z = (-y,x,0),

with :math:`(x,y,z)` the coordinates relative to the centroid. A 2D field gets two translations plus
the single in-plane rotation :math:`(-y,x)`; a field of any other nodal dimension has no rotational
rigid-body mode and takes the translations alone
(:func:`~edelweissfe.linsolve.nullspace.translationNullspace`). A scalar field, e.g. ``nonlocal
damage``, has no rigid-body rotation either and takes AMGCL's default constant.

Because the hierarchy is built for the *equilibrated* block, the basis has to be transformed
consistently: the near-null-space of :math:`D A D` is :math:`D^{-1}` times that of :math:`A`, which
is why both functions divide the raw modes by the field's own slice of the scaling vector.

Supplying only the translations -- the common simplification, since rotations need node coordinates
rather than just the DOF-block structure -- leaves the coarse grid unable to represent rotational
rigid-body error exactly, and by the construction above the Chebyshev smoother is not targeting that
error class either, so it has nowhere efficient to go.

The full basis was measured to need fewer outer iterations than translations alone, and to be robust to the thread
count (:ref:`measured-rigid-body-nullspace`).

That power-iteration setting is the smoother's second, independent subtlety. The smoother needs
:math:`\rho`. AMGCL's own default (``power_iters = 0``) bounds it by Gershgorin's theorem: safe and
deterministic, but pessimistic, which shifts the window upward; the scalar default tree (the damage
field's) keeps that. With ``power_iters > 0`` AMGCL estimates :math:`\rho` by that many steps of the power
iteration instead -- tighter, but from a start vector drawn from random streams seeded per OpenMP thread.
That estimate is deterministic for a given thread count but *different* at a different thread count,
because the number of independent random streams changes with it even though the operator does not. At a
short iteration budget it can be badly under-converged on a hard operator, which cripples the smoother.
Running the power iteration close to convergence -- the vector-field default does 300 steps -- removes the
practical thread-count sensitivity (:ref:`measured-power-iterations`), for a one-time cost per hierarchy build that is small against the
build itself. It does not make the hierarchy bitwise independent of the thread count: the estimate still
starts from per-thread random streams and agrees across thread counts only to its convergence (and its
reductions are summed in thread order, so even two runs at the same thread count can differ in the last
bits). Smoothed aggregation's ``estimate_spectral_radius`` (part of the vector-field default, with
only 20 power iterations) uses the same per-thread seeded estimate for the prolongation smoother's
weight, so a hierarchy built with it depends measurably on the thread count; results are reproducible at
a fixed thread count, up to that last-bit summation order.

.. _blockamg-block-nullspace:

The near-null-space of a block backend
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A block backend (``backendBlockSize`` 2 or 3, the default for 2D and 3D vector fields) stores a field's
block in :math:`B \times B` nodal blocks. Without a near-null-space, its tentative prolongation gives every
node :math:`i` of an aggregate :math:`a` the identity block, :math:`P^{\mathrm{tent}}_{ia} = I_B`, so the coarse
space holds the constants per component of the variables the hierarchy is built on. For the equilibrated block :math:`\hat A_{ii} = D A_{ii} D` these
are :math:`z = D^{-1} x`, and a constant :math:`z` is the displacement :math:`x = D\,c`: a
:math:`D`-weighted translation. The translations of :math:`\hat A_{ii}` are :math:`D^{-1} t` instead, the
same transformation the scalar backend applies to its rigid-body modes (see above).

``blockTranslationNullspace``, on by default on a block backend, builds the hierarchy from those translations
(``amgcl-block-translations.hpp`` of the AMGCL wrapper :mod:`edelweissfe.linsolve.amgcl.amgcl`). Per node
they form the diagonal block :math:`W_i = \operatorname{diag}(w_{i1}, \dots, w_{iB})`,
:math:`w_{ik} = 1/s_{ik}` with :math:`s` the equilibration
(:func:`~edelweissfe.linsolve.nullspace.translationNullspace`). Smoothed aggregation orthonormalizes its
near-null-space on every aggregate by a QR factorization of the stacked blocks. Column :math:`k` of that
stack is nonzero only in the rows of component :math:`k`, so the columns are orthogonal and the
factorization is a normalization per component:

.. math::

   P^{\mathrm{tent}}_{ia} = \operatorname{diag}\!\left( \frac{w_{ik}}{\lVert w_k \rVert_a} \right), \qquad
   R_a = \operatorname{diag}\!\left( \lVert w_k \rVert_a \right), \qquad
   \lVert w_k \rVert_a^2 = \sum_{j \in a} w_{jk}^2 .

So :math:`(P^{\mathrm{tent}})^\mathsf{T} P^{\mathrm{tent}} = I` on every aggregate, and
:math:`P^{\mathrm{tent}} R = W` reproduces the translations exactly; :math:`R` is the near-null-space of
the coarse level, which repeats the construction. Every block stays a diagonal :math:`B \times B` block,
so the hierarchy keeps AMGCL's block value type. The rotations do not fit that way (six modes on
:math:`3 \times 3` blocks would need :math:`6 \times 6` coarse blocks): this hierarchy takes the translations only
and refuses any other near-null-space, and the rotations stay with the scalar backend -- or with the rigid-body coarse
level, the default for the 3D displacement field, which builds the first coarse level of a block field itself (its
prolongation in :math:`B \times M` blocks, the Galerkin product in :math:`M \times M` blocks) and hands the unblocked
coarse operator to a scalar AMGCL hierarchy with ``aggr.block_size`` :math:`M` (see
:ref:`blockamg-rigid-body-coarse-level`, which also says why AMGCL's ``as_scalar`` coarsening, the route of AMGCL's
runtime coarsening for a near-null-space on a block backend, does not fit).
Everything else is AMGCL's smoothed aggregation, unchanged: the aggregation on :math:`\hat A_{ii}`, the
prolongation smoothing and the Galerkin product. Unit weights :math:`w = 1` give AMGCL's own hierarchy back,
up to a scaling of the coarse variables.

A hierarchy built on :math:`A_{ii}` itself, whose constant blocks are its translations :math:`t`, gives the
same coarse space: with the same aggregates, it and the hierarchy of :math:`\hat A_{ii}` on :math:`D^{-1} t`
are one hierarchy in other variables (a similarity transform),
:math:`M_{\hat A}^{-1} = D^{-1} M_A^{-1} D^{-1}`, provided the smoother is invariant under the scaling, as a
Jacobi-scaled Chebyshev smoother (``"relax": {"scale": true}``) is and an unscaled one is not. Only the
aggregates can differ: for blocks, AMGCL's strength test compares traces of block products,
:math:`\operatorname{tr}(\varepsilon^2 A_{ii} A_{jj}) < \operatorname{tr}(A_{ij} A_{ij})`, which a symmetric
diagonal scaling leaves unchanged only where :math:`D` is uniform within each node.

Measured offline and live, the translations need fewer outer iterations on most models; they help less once
softening makes the operator indefinite (:ref:`measured-block-translations`).


.. _blockamg-block-gauss-seidel:

The block Gauss--Seidel sweep: ``Multiplicative``
-------------------------------------------------

The preconditioner :math:`v = M^{-1} r` applied at each outer Krylov iteration is a block Gauss--Seidel
sweep over the fields. Starting from :math:`v_j = 0` for all fields, each field :math:`i` in turn
forms its local residual against the *current* parts of :math:`v` of all fields and corrects its own
part by one cycle of its own AMG hierarchy:

.. math::

   v_i \leftarrow v_i + M_i^{-1}\Bigl( r_i - \sum_{j} \hat A_{ij}\, v_j \Bigr),

where :math:`M_i^{-1}` denotes one AMG cycle on :math:`\hat A_{ii}`. On a field's first visit
:math:`v_i = 0`, and this is the plain :math:`v_i = M_i^{-1}(r_i - \sum_{j \neq i} \hat A_{ij} v_j)`;
the correction form matters on revisits, which an inexact :math:`M_i^{-1}` would otherwise throw
away. With ``symmetric = true`` (default false) each forward sweep is followed by a reverse-order sweep
of the same correction form, :math:`i = n-1, \dots, 1` -- starting one field before the one just
updated. With exact field solves that field has nothing left to correct; with AMG cycles it has, and
skipping it saves one cycle of that field. With exact :math:`M_i = \hat A_{ii}` this is the classic block
symmetric Gauss--Seidel preconditioner,

.. math::

   M_\mathrm{SGS}^{-1} = (B + U)^{-1}\, B\, (B + L)^{-1}, \qquad U = \text{the blocks } \hat A_{ij},\ i < j;

with inexact :math:`M_i`, a reverse sweep in correction form over *all* fields would give
:math:`(B + U)^{-1} (2B - \hat D) (B + L)^{-1}`, :math:`\hat D = \operatorname{blockdiag}(\hat A_{ii})`,
the classic operator for :math:`B = \hat D`; skipping the last field changes its last block row.

``sweeps`` repeats the (possibly reversed) sweep. Both are arguments of the ``blockAMG()`` preset (data lines of
``type=blockamg``); in a tree, ``Multiplicative(gaussSeidelOrder([OnField(...), ...], sweeps, symmetric))``
(:func:`~edelweissfe.linsolve.composition.combinators.gaussSeidelOrder`).
Neither is needed for the outer solve: right-preconditioned GMRES/LGMRES only needs :math:`M^{-1}`
to be a fixed linear operator, not a symmetric one -- the measured configurations use
``sweeps = 1``, ``symmetric = false``.

What that one forward sweep does depends on how the fields couple. On the gradient-enhanced damage
model of the reference runs the coupling is one-sided: :math:`K_{ud}` (displacement rows, damage
columns) is a noticeable fraction of :math:`K_{uu}`, while :math:`K_{du}` is many orders of magnitude
smaller than :math:`K_{dd}`.
With the displacement field first in DOF order, the only coupling product the forward sweep forms is
:math:`\hat A_{du} v_u`, which is numerically zero there: on that model the default sweep acts as
block Jacobi, and still pays that product on every outer iteration. Sweeping the damage field first
would instead make it the exact block back substitution of the (then block upper triangular) system,
up to the inexact field solves. Measured, the order made no difference to the outer iterations on the reference
models (:ref:`measured-sweep-order`).

This is deliberately only an *approximate* inverse of the coupled operator -- which is exactly why
the true-residual bookkeeping below exists.

The outer Krylov solve: ``LGMRES``
----------------------------------

The outer solve, :class:`~edelweissfe.linsolve.composition.krylov.LGMRES`, is AMGCL's own native
``amgcl::solver::lgmres``, driven through
:class:`~edelweissfe.linsolve.amgcl.amgcl.PyAMGCLKrylovSolver`; the block sweep is passed straight in
as the preconditioner callback. (:class:`~edelweissfe.linsolve.composition.krylov.FGMRES` is the alternative for a
variable preconditioner, one with a Krylov solve inside, which LGMRES refuses.) A former fallback on
:func:`scipy.sparse.linalg.gmres` was removed once the native path was gated live against
it at several thread counts and found to produce the same Newton trajectory. Orchestrating GMRES from
Python -- the Arnoldi steps, orthogonalization and restart bookkeeping -- was measured as a
substantial share of this solver's wall-clock on a reference model when run that way, and, being serial Python,
anti-scaled past one NUMA node on multi-socket hardware, which the native implementation does not.

``LGMRES(..., maxIterations)`` (the data line ``maxIterations``) bounds *all* Arnoldi steps of a solve, across all
its internal restarts.

LGMRES augments each restart with a few vectors carried over from the previous restart cycle of the same
solve. ``LGMRES(..., augmentationVectors)`` sets how many, and a restart cycle is ``restart + augmentationVectors``
iterations, every one of them a preconditioner application: ``restart`` Krylov steps plus one step per augmentation
vector (``amgcl/solver/lgmres.hpp``). The first cycle of a solve has no augmentation vectors yet and is plain
GMRES(``restart + augmentationVectors``). Every solve starts from an empty augmentation subspace. Keeping it alive
*across* solves ("recycling") sounds attractive -- a hard solve's converged subspace as a warm start for
the next -- but measured on captured systems it contributed nothing (bit-identical iteration counts), and
on a live run it was actively harmful: a solve that had already struggled left a poorly conditioned
subspace which compounded into a following solve needing far more iterations than a fresh one, and in
one run an outright ``NaN``. It was therefore removed.

.. _blockamg-forcing-tolerance:

Adaptive outer tolerance: ``ForcingTolerance``
----------------------------------------------

Most Newton iterates do not need their linear solve solved tightly, and loosening the requested
tolerance for those cuts outer iterations substantially. By default ``LGMRES`` therefore chooses
its own outer tolerance per solve by Eisenstat--Walker forcing ("choice 2"; ``ForcingTolerance(minimum, maximum,
gamma, alpha)``, :class:`~edelweissfe.linsolve.composition.krylov.ForcingTolerance`, the default tolerance of
``LGMRES``), from the ratio of successive residual norms:

.. math::

   \eta_k = \gamma \left( \frac{\lVert b_k \rVert}{\lVert b_{k-1} \rVert} \right)^{\alpha},
   \qquad \gamma = 0.9, \quad \alpha = \tfrac{1}{2}\bigl(1+\sqrt 5\bigr),

clamped to :math:`[\eta_\mathrm{min}, \eta_\mathrm{max}]` (``minimum``, ``maximum``). The classic safeguard against
over-loosening is applied: a large tightening step is only trusted if the previous tolerance was
already small, i.e. :math:`\eta_k` is raised back to :math:`\gamma \eta_{k-1}^{\alpha}` whenever that
quantity exceeds 0.1. At the classic parameters that needs :math:`\eta_{k-1} > 0.26`, so the safeguard
never acts with the default :math:`\eta_\mathrm{max}` (well below that). With no meaningful history to compare against -- the first solve, a residual of
exactly zero, or the solve right after a detected new increment, where the ratio across the
jump says nothing about Newton convergence -- the tolerance falls back to :math:`\eta_\mathrm{max}`.

A solve is treated as the first of a new increment when its residual norm exceeds
``residualGrowthFactor`` times the previous solve's. That single test drives both the tolerance reset
and a hierarchy refresh. It is a guess, and it has a blind spot: a cutback after a *diverged* attempt
usually starts from a residual *smaller* than the last, large one of that attempt, so the retry is not
recognized -- its first solve, typically the hardest, gets a tolerance from the ratio to the diverged
residual (clamped to :math:`\eta_\mathrm{min}`) and, unless the sparsity pattern changed, the
hierarchies of the discarded attempt. Only a cutback after an attempt whose residual had decreased is
detected. (A notification from the nonlinear solver at the start of every increment would replace the
guess; that is planned.)

A fixed relative tolerance -- the data line ``tolerance``, ``LGMRES(..., tolerance=FixedTolerance(eta))`` -- switches
the scheme off entirely. Adaptive forcing is safe to use as the default only because the requested tolerance is
enforced on the *true* residual, which is the subject of the next section: without that enforcement,
a loosened request combined with an imperfect preconditioner can quietly let the true residual run
looser still, which does change the Newton path.


.. _blockamg-true-residual:

True-residual enforcement: ``LGMRES`` on the scaled system
----------------------------------------------------------

On the scaled system (``krylovOnScaledSystem = true``; not the default), the outer Krylov method's own
stopping test is not on the true residual but on the residual of the *scaled* (equilibrated) system it
iterates on, :math:`\lVert \hat b - \hat A z \rVert / \lVert \hat b
\rVert`: AMGCL's LGMRES is right-preconditioned and tests it directly. The
scaled residual can differ substantially from the true relative residual (stiff rows are down-weighted
by :math:`D`), so "converged" can mean a true relative residual well above the requested :math:`\eta`;
measured on a reference model, one solve's achieved true residual was found far looser than what had
been asked for.

Every solve on the scaled system therefore measures the true relative residual :math:`\lVert A x - b \rVert / \lVert b
\rVert` explicitly (computed as :math:`D^{-1}(\hat A z - \hat b)`, which is exact and rides on the
threaded operator already built, rather than paying a second unscaled matrix-vector product). If it
still exceeds :math:`\eta`, the solve is **continued**: a warm restart from the current iterate at a
tighter requested tolerance, up to ``continuations`` times (the data line, ``LGMRES(..., continuations)``). By
default the tolerance
is tightened geometrically by a fixed factor of 0.01 per round -- deliberately expressed purely in
the units of the requested relative tolerance, which the Krylov solver interprets consistently by
construction, rather than being derived from the solver's internal residual bookkeeping. Setting
``continuations = 0`` disables enforcement: the outer solve's own test decides.

On the unscaled system (``krylovOnScaledSystem = false``) none of this is needed: the residual the
outer solve converges is the true one, so a pass that misses :math:`\eta` has stopped at its iteration
cap, and a continuation would only extend the cap (``maxIterations`` is the knob for more iterations). What
it trades is the norm itself. :math:`\lVert b - A x \rVert` sums the residuals of all
fields in their own units, while the scaled residual :math:`D (b - A x)` weighs every row by its own
diagonal. In a system whose fields differ widely in residual scale -- a displacement field in newtons
next to a damage field with entries of order one, say -- a field with small residual entries can thus
be left less resolved at the same :math:`\eta` on the unscaled system than on the scaled one. On the
reference models the unscaled system was the faster of the two, with no more Newton iterations (see
:ref:`measured-defaults`), but that is a measurement on those models, not a guarantee.

.. _blockamg-gap-compensation:

Gap-compensated tolerance: ``GapCompensation``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Because the system is equilibrated, the outer Krylov method minimises the *scaled* residual
:math:`\lVert \hat b - \hat A z \rVert` while acceptance is judged on the *true* relative residual.
The ratio between the two -- the **gap** -- has been measured to vary widely, from close to one on
some models to considerably larger on a contact model with multi-point constraints (stiff rows are
down-weighted by the equilibration), and it is why
almost every solve needs a continuation on the scaled system (the former default): on a reference
gradient-enhanced damage model, nearly every solve ran one, i.e. effectively every linear solve was
performed twice. A continuation is disproportionately expensive, because restarting discards most of
the accumulated Krylov subspace.

With gap compensation (the data line ``gapCompensation=true``; in a tree
``LGMRES(..., gapCompensation=GapCompensation(safetyFactor, maxFactor))``) the gap is measured on each pass (as the
achieved true
relative residual over the achieved scaled relative residual of the same solution -- not over the
tolerance the pass was given, which a pass stopped at the iteration cap does not reach; that version
inflated the gap by the unreached factor and made every following first pass ask for that much more
accuracy), smoothed across solves by an exponential moving
average, floored at 1 and capped at ``maxFactor``, and used to pre-compensate the *first* pass: it is asked for
``safetyFactor``:math:`\cdot\,\eta\,/\,\text{gap}` so that it lands on target in one pass. Any
continuation that is still needed then tightens by the *measured* gap rather than by the fixed factor
(never looser, and always at least a factor of two per round so the loop still terminates).

Gap compensation is opt-in, and need not be the default: the default already runs the outer solve on the unscaled
system (``krylovOnScaledSystem = false``), where there is no gap to compensate at all. Why it is not the default on
the scaled system either -- the uncompensated solve over-solves, and Newton relies on it -- is measured in
:ref:`measured-gap-compensation`.


Reuse across Newton iterations: ``RebuildPolicy``
-------------------------------------------------

Building the per-field hierarchies is a large fraction of a solve, and it is avoidable when the
Jacobian has barely moved since the last one. ``LGMRES`` therefore keeps the hierarchies below it (and the
equilibration they were built for) standing across solves by default; when they are rebuilt is the outer solve's
:class:`~edelweissfe.linsolve.composition.krylov.RebuildPolicy`,
``LGMRES(..., rebuild=RebuildPolicy(stalenessFactor, rebuildOnNewIncrement, reuse))``.

The outer Krylov solve always operates on the current matrix (equilibrated with the scaling the standing
hierarchies were built for, which is the same system whatever the scaling); only the preconditioner may be
stale. A stale preconditioner costs outer iterations and, if it drives a solve to the iteration cap, the
accuracy of that one Newton step; the next solve is then rebuilt (see below).

The hierarchies are rebuilt when

* ``reuse`` is off: then every solve builds its own, and none of the triggers below matters,
* there is nothing to reuse yet (the first solve),
* the field-block layout or the system size changed (e.g. an adaptive-refinement event),
* the number of stored nonzeros changed -- a free :math:`O(1)` proxy for "the sparsity pattern
  changed". It is not exact (two different patterns could coincidentally share an ``nnz``) but it
  errs safely, since an unnecessary refresh costs time and never correctness. A pattern change is not
  a graceful degradation: measured on one transition, a hierarchy built for the previous pattern
  needed far more outer iterations than a fresh one, an outright wall-clock regression,
* a residual jump marks a new increment (see ``residualGrowthFactor`` and its blind spot above) and
  ``rebuildOnNewIncrement`` is set (the default), or
* the previous solve's outer iteration count exceeded both ``stalenessFactor`` times and 15
  more than the count of the solve the hierarchies were built for -- a growing count being the signal
  that the reused hierarchies are drifting away from the current Jacobian. The reference is that
  first solve, not the previous one, so a slow drift across several solves is caught; the
  absolute margin keeps a factor alone from firing on noise at small counts, or
* the previous solve hit the outer iteration cap. A missed tolerance alone does not trigger a
  rebuild: on the scaled system, with continuations off, it typically comes from the gap between the
  scaled residual the outer solve converges and the true one (stiff rows down-weighted by the
  equilibration, e.g. on a model with multi-point constraints), which a fresh hierarchy cannot close --
  measured slower on such a model when it did trigger rebuilds; on the unscaled system (the
  default) a miss without a cap hit happens only at a rounding tie, since the outer solve then
  converges the true residual itself. A cap hit asks for a rebuild even when the capped solve had
  just rebuilt its hierarchies: not rebuilding again was measured neutral on a live run (the same
  Newton path, at a comparable cost), so the simpler rule stays.

On a model with contact or tie constraints the active constraint set can change between Newton
iterations, and then the sparsity pattern -- and with it the nonzero count -- changes and the
hierarchies are rebuilt. How often that happens depends on the model: the multi-point condensation
would otherwise drop exact zeros (``pruneCondensedMatrixZeros``, a nonlinear-solver option), which
changes the count on almost every Newton iteration even when the constraint set does not; left unset,
that option's "auto" default already leaves pruning off here, because a solver with a Krylov solve reports
:attr:`~edelweissfe.linsolve.base.LinearSolver.needsStableSparsityPattern`, so reuse pays on such
models without any extra configuration (on a contact model with many multi-point constraint slaves,
a configuration relying on it ran noticeably faster than with pruning on).

The native LGMRES instance is deliberately *not* torn down on a hierarchy refresh -- only on an
actual system-size change, which invalidates AMGCL's preallocated scratch vectors. (A hot reload of a ``custom``
solver script builds a new ``LGMRES``, and with it a new instance.)


.. _blockamg-p-multigrid:

p-multigrid: ``TwoLevel(coarseSpace=P1Interpolation())`` (opt-in)
-----------------------------------------------------------------

For a field discretized entirely with quadratic serendipity elements, a ``custom`` solver script can replace that
field's single-level hierarchy with a genuine two-level V-cycle, ``OnField(name,
TwoLevel(coarseSpace=P1Interpolation()))``
(:class:`~edelweissfe.linsolve.composition.ptwogrid.PTwoGridPreconditioner`): ``npre`` Chebyshev
sweeps on the field's own equilibrated block, restriction of the residual through
:math:`P^\mathsf{T}`, one AMGCL cycle on the Galerkin-projected coarse operator :math:`A_1 =
P^\mathsf{T} A P`, prolongation through :math:`P`, and ``npost`` more Chebyshev sweeps. The
coarse space is purely topological -- the mesh's corner nodes -- so no re-discretization is involved:
:math:`P` is the identity on corner nodes and the average of the two edge-endpoint corners on each
midside node. The coarse solve always runs on the plain scalar backend, so it is given the same
rigid-body near-null-space treatment the single-level default gives a scalar-backend field,
restricted to the corner-node subset (``P1Interpolation(useCoarseNullspace=True)``, the default) -- independent of
the backend the field's fine level is stored on (a block backend, the default for 2D/3D fields, takes the translations
at most; see ``backendBlockSize`` above).

Its preconditioner is rebuilt exactly when the single-level hierarchies are (see the reuse rules
above), and its coarse solve gets a near-null-space while the fine-level Chebyshev smoother, having
no coarsening step of its own, needs none. Measured against the default single-level
configuration with the same outer settings (fine operator in float, ``npre = 0``,
``npost = 1``, Chebyshev degree 16, the same coarse ``estimate_spectral_radius``), it was
slower on a cylinder sequence and level within noise on a contact sequence, with the cheaper
rebuild -- an alternative worth measuring on a model with expensive rebuilds, not the default.
Enable it per field with ``OnField(name, TwoLevel(coarseSpace=P1Interpolation(coarsePrecond, useCoarseNullspace),
tree=...))``; the topology map is computed lazily from the model and
recomputed whenever the model changes (``setModel``, e.g. after adaptive refinement). There is no
fallback to the single-level default: on linear elements every node is a corner, so the coarse
operator is the field's block itself and the cycle gains nothing, and an element topology the map
does not know raises. A node that two elements classify inconsistently (seen at non-conforming
refinement interfaces) is made a corner, with a warning.

The whole cycle, and every stage of its build, runs in C++ and is OpenMP-threaded. The Dirichlet
DOFs of the field's block -- rows whose only nonzero is the diagonal, which includes the identity
slave rows of the multi-point-constraint condensation -- are decoupled exactly rather than removed:
the fine right-hand side and iterate are kept zero on them, which a Dirichlet row
preserves under every residual and every smoothing step, and :math:`P` has no rows or columns for
them. This is the free-submatrix cycle up to floating-point summation order, without extracting the
free submatrix, and it allows ``TwoLevel(..., backendBlockSize=3)``: the fine operator is stored in node-wise
:math:`3\times 3` blocks even where a node is only partially constrained, which roughly halves the
memory traffic of the fine smoother and residual -- the dominant cost of the apply.
``backendPrecision="float"`` (the default on :math:`3\times 3` blocks) halves the bytes per stored entry once more.
Only the stored entries
are rounded: vectors and every sum stay double (a float matrix times a double vector), and the
spectral radius is estimated and re-checked on the double operator before it is rounded -- unlike
``AMG(backendPrecision="float")`` on the scalar backend, where AMGCL's own power iteration runs in single precision.

The fine Chebyshev smoother is the tree's ``relax`` (``degree``, ``lower``, ``higher``, ``power_iters``), applied
``npre`` times before and ``npost`` times after the coarse-grid correction -- the same spelling as for the rigid-body
coarse level. It uses AMGCL's polynomial, with the bounds of the vector-field default tree
(``lower`` 0.01, ``higher`` 1.1 times the spectral radius; AMGCL's own defaults are 1/30 and 1), but its
spectral-radius estimate -- 300 power iterations in the default tree, the most
expensive stage of the build -- starts from a deterministic, thread-count-independent vector, and can
be reduced to a re-check: with ``TwoLevel(..., fineSpectralRadiusReuse=True)`` a field's first build estimates it in
full, and every later rebuild continues the power iteration for 15 steps from the previous build's
dominant vector. A re-check estimate up to 1% above the previous radius keeps it (the full estimate is
itself converged only to about 1e-5); one further above replaces it; and one above it by more than the
10% margin of ``higher`` -- an operator that stiffened more than the smoother's bounds cover -- triggers
the full estimate again. The cache is dropped when the model
changes (e.g. after adaptive refinement).


.. _blockamg-rigid-body-coarse-level:

Rigid-body coarse level: ``TwoLevel(coarseSpace=RigidBodyModes())``
-------------------------------------------------------------------

By default, the hierarchy of the 3D displacement field starts with a coarse level built from **all** of its rigid-body
modes -- the translations and the rotations -- in place of AMGCL's hierarchy on the translations
(``TwoLevel(coarseSpace=RigidBodyModes())``, :mod:`~edelweissfe.linsolve.composition.rigidbodycoarselevel`); a 2D
displacement field gets it in a ``custom`` solver script. It keeps what makes the block backend fast, the fine level
in node-wise blocks stored in float, and adds what that backend cannot take, the rotations.

Why the translations are not enough
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The coarse space of a block hierarchy holds the translations of every aggregate (see :ref:`the near-null-space of a
block backend <blockamg-block-nullspace>`): displacements that are constant on each aggregate. Smoothed aggregation
represents a smooth error well only where it is close to such a combination on every aggregate, and the Chebyshev
smoother leaves smooth error alone by construction: its window starts at 1% of the spectral radius (see the
near-null-space and the Chebyshev smoother above). In a bending-dominated structure -- a slender beam, a tunnel
lining, a plate -- the smooth error is bending: every aggregate rotates almost as a rigid body, its displacement
varying linearly across it. Neither part of the cycle reduces such an error, and one application of the hierarchy
on the translations can even amplify it. Measured on the displacement blocks :math:`A_{uu}` of captured systems,
:math:`\lVert v - M A_{uu} v \rVert / \lVert v \rVert` after one application :math:`M` of the field's
preconditioner, for the rigid rotations :math:`v` about the field's centroid and for a smooth quadratic field
(:math:`u_x = y^2 + z^2` about the centroid), both zero on the Dirichlet rows:

.. list-table:: One application on smooth error, the hierarchy on the translations -> the rigid-body coarse level
    :width: 100%
    :header-rows: 1

    * - error
      - tunnel segment, elastic
      - tunnel segment, damaged
      - damage cylinder
      - edge breakout
    * - rotations about x, y, z
      - 1.12 2.06 0.88 -> 0.15 0.27 0.47
      - 1.21 0.97 0.81 -> 0.12 0.10 0.18
      - 0.28 0.28 0.80 -> 0.12 0.11 0.12
      - 0.19 0.37 0.50 -> 0.03 0.03 0.02
    * - quadratic field
      - 1.53 -> 0.62
      - 1.34 -> 0.46
      - 1.50 -> 1.06
      - 0.30 -> 0.17

A value above 1 means that one application amplifies that error. On the tunnel segment, a shell-like lining of
quadratic hexahedra, the hierarchy on the translations amplifies the rotations and the quadratic field, and the outer
solve pays with iterations: 67 of them, against 22 with the rotations in the coarse space (see
:ref:`measured-rigid-body-coarse-level`). Not every
error fares better: the rigid translations, which both coarse spaces hold, are reduced less on the elastic tunnel
system (0.11 0.04 0.20 -> 0.70 0.18 0.41). The outer iteration counts of
:ref:`measured-rigid-body-coarse-level` are the net effect.

How the coarse level is built
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

On the field's equilibrated block :math:`\hat A = D A D`, with its near-null-space :math:`B` -- the field's rigid-body
modes divided by :math:`D` (the near-null-space of :math:`D A D` is :math:`D^{-1}` times that of :math:`A`, as for the
translations), which Python builds with :func:`~edelweissfe.linsolve.nullspace.rigidBodyNullspace`, the one definition
of the modes, and hands to C++ -- in C++, ``amgcl-rigid-body-coarse-level.hpp``, one function per step:

#. **Aggregates.** AMGCL's plain aggregation of the nodes: the aggregates of AMGCL's own first level on the block
   backend, with its strength test on the :math:`B \times B` blocks (see :ref:`above <blockamg-block-nullspace>`) and
   the tree's ``coarsening.aggr.eps_strong``. A node without a strong connection -- a fully constrained one, for
   instance -- is in no aggregate and left to the smoother.
#. **Tentative prolongation.** Per aggregate :math:`a`, the rows of :math:`B` on its dofs, orthonormalized -- what
   AMGCL's own tentative prolongation does with a near-null-space:

   .. math::

      V_a = B \big|_a = \begin{bmatrix} b_1 & \cdots & b_M \end{bmatrix}_a = Q_a R_a,
      \qquad P^{\mathrm{tent}} \big|_a = Q_a ,

   with :math:`b_1, \dots, b_B` the translations and :math:`b_{B+1}, \dots, b_M` the rotations about the field's
   centroid. The rows of Dirichlet dofs and multi-point-constraint slaves are zero. The QR factorization (modified
   Gram--Schmidt with a second sweep) makes the columns of :math:`P^{\mathrm{tent}}` orthonormal, and
   :math:`P^{\mathrm{tent}} R_a` reproduces the modes on the aggregate exactly. A rotation about the field's centroid
   :math:`g` carries a translation of the order of :math:`|c_a - g|` across an aggregate of extent :math:`h_a` around
   its centroid :math:`c_a`; projecting the translations out loses about :math:`\log_{10}(|c_a - g| / h_a)` digits, 2 to
   4 on a mesh of elements a hundred to ten thousand times smaller than the structure, far from the threshold of the
   dependent modes below, so the modes need no recentring on every aggregate. The Dirichlet rows are zero because the
   cycle solves them exactly, apart from the coarse-grid correction (as the p-two-grid does), and a rigid-body mode is
   near-null only on the free dofs: so :math:`P^\mathsf{T} \hat A P = P_F^\mathsf{T} \hat A_{FF} P_F`.
#. **Dependent modes.** A mode the Gram--Schmidt sweeps reduce to at most :math:`\sqrt{\varepsilon_{\mathrm{mach}}}`
   (1.5e-8) of its norm lies in the span of the modes before it: the rotations of an aggregate of a single node, the
   rotation about the line through a collinear aggregate, a mode whose components are all constrained (an aggregate
   on a symmetry plane). Its normalized remainder would be rounding noise, not a direction of the aggregate, so it is
   dropped: its column of :math:`P^{\mathrm{tent}}` is zero, and its coarse dof gets a diagonal entry and nothing
   else. The threshold is not tuned: an exactly dependent mode leaves a remainder of about 1e-16 of its norm, an
   independent one of the order of :math:`h_a / |c_a - g|` (see above), or of the aggregate's aspect ratio.
#. **Smoothed prolongation.** :math:`P = (I - \omega\, \mathrm{diag}_B(\hat A_F)^{-1} \hat A_F) P^{\mathrm{tent}}`,
   AMGCL's own, shared with the translations of the block backend: :math:`\hat A_F` holds the strong connections, the
   weak ones lumped onto the diagonal blocks, :math:`\mathrm{diag}_B` takes its node-wise :math:`B \times B` diagonal
   blocks, and :math:`\omega` follows the tree's ``coarsening`` (``relax``, ``estimate_spectral_radius``,
   ``power_iters``). One damped block-Jacobi step widens every column by a layer of nodes and smooths away its jump at
   the aggregate's boundary. With the option ``"prolongation": "tentative"``, :math:`P = P^{\mathrm{tent}}`.
#. **Galerkin coarse operator.** :math:`A_c = P^\mathsf{T} \hat A P`, with :math:`M = B (B + 1) / 2` coarse dofs per
   aggregate, 6 in 3D and 3 in 2D, formed block by block, in :math:`M \times M` blocks, and handed to AMGCL unblocked
   to a scalar matrix: a dropped mode's coarse dof gets the mean of the kept diagonal entries as its diagonal entry,
   and the zeros of its row and column stay stored -- AMGCL's aggregation below takes the largest entry of every
   :math:`M \times M` block, which they do not change.
#. **The hierarchy below.** On aggregate :math:`a`, the field's rigid-body modes are :math:`V_a = P^{\mathrm{tent}} R_a`:
   their coefficients in the coarse basis, :math:`R_a`, are the near-null-space of an AMGCL hierarchy on
   :math:`A_c`, as in AMGCL's own recursion. The field's tree configures it as AMGCL would configure its next level:
   the strong-connection threshold halved, the :math:`M` dofs of an aggregate one point (``aggr.block_size``), in a
   scalar hierarchy.

What runs outside AMGCL is steps 2 to 5: the tentative prolongation (in :math:`B \times M` blocks), its smoothing
(AMGCL's own algorithm, on those blocks), the Galerkin product (in :math:`M \times M` blocks) and its unblocking; the
aggregates of step 1 and the hierarchy below are AMGCL's. AMGCL's ``as_scalar`` coarsening, which would run a scalar
smoothed aggregation with all modes on the block backend, is not used, for three reasons: it re-blocks the prolongation
into :math:`B \times B` blocks, which the :math:`M = 3` modes of a 2D field do not fit; its Householder QR keeps an
orthonormal completion of a dependent mode instead of dropping it; and its modes are not zero on the Dirichlet rows.
And AMGCL's sparse products take one block type, where the Galerkin product chains :math:`B \times B`,
:math:`B \times M` and :math:`M \times B` blocks: so the blocks have kernels of their own.

One application is the two-level cycle the p-two-grid uses too: restriction :math:`P^\mathsf{T} r`, one V-cycle of the
hierarchy below, prolongation, and the fine Chebyshev sweeps of the tree (``npre`` before, ``npost`` after the
coarse-grid correction: 0 and 1, degree 16, in the default trees), with the Dirichlet rows decoupled exactly.
:math:`P` and :math:`P^\mathsf{T}` are stored like the fine level (in float with the 3D default tree), every sum in
double. With the translations alone in place of all modes, the construction *is* AMGCL's first level on the block
backend -- the same aggregates, tentative prolongation, smoothing and Galerkin product, :math:`M = B` -- except on the
Dirichlet rows, where AMGCL's translations are not zero. A test of the suite checks that, on a system without Dirichlet rows,
the two preconditioners then agree up to the spectral-radius estimate of the fine smoother (AMGCL's power iteration
starts from random vectors, the cycle's from a deterministic one): what the rigid-body coarse level adds is the
rotations.

When it pays, and when to switch it off
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The coarse level pays where the hierarchy on the translations needs many outer iterations although the operator is
elastic or mildly damaged: typically a bending-dominated structure. It is the default for the 3D displacement field, of
any element type: the rigid-body modes are the near null-space of elasticity on the displacement field (the field of
type ``vector`` in :data:`~edelweissfe.config.phenomena.phenomena`). A rotation field -- of beams, or of the reference
point of a rigid body -- or any other vector field has another near null-space (its translations), and
``RigidBodyModes`` rejects it (:func:`~edelweissfe.linsolve.nullspace.carriesRigidBodyModes` decides, by the field's
phenomenon). A 2D displacement field gets the coarse level in a ``custom`` script, where every field's preconditioner
is written out, ``OnField(name, TwoLevel(coarseSpace=RigidBodyModes(), tree=...))``, and nothing is added to it. A 3D
displacement field without node coordinates (see below) keeps the hierarchy on the translations in the preset, logged
once at ``info``. The field's tree, the default one or one given to ``TwoLevel(tree=...)``, configures the smoother,
the aggregation and the hierarchy below.

Switch it off on a model whose solves converge in a few outer iterations with the hierarchy on the translations: the
coarse level saves little there, and its setup is larger wherever its coarse operator is -- up to 1.2 times that of
the hierarchy on the translations on the larger models of :ref:`measured-rigid-body-coarse-level`. On the edge breakout, where the translations need 22
and 23 outer iterations, the coarse level saves 2 and 1, at a setup of 0.94 and 0.99 times theirs and a solve time,
setup included, of 0.92 and 0.95 -- a wash. The data line ``rigidBodyCoarseLevel=false`` of ``type=blockamg``
switches it off; its options are those of a ``custom`` script (see :ref:`its arguments
<blockamg-rigid-body-coarse-level-configuration>`).

The modes are built from the node coordinates of the reference configuration, as the model holds them, which
the coarse level reads from the model: after adaptive refinement those of the new model, under a ``SchurComplement`` those
of the remaining nodes. Rigid-body modes about the reference configuration suit small rotations; under large rotations
they represent the rigid motions of the deformed structure less exactly, which costs convergence speed only, not
accuracy. Without a model -- a script driving the solver itself -- the coordinates must be given as
``nodeCoordinates`` (see :ref:`composed-adapter`): without them, ``TwoLevel(coarseSpace=RigidBodyModes())``
raises an error rather than switch off the coarse level it asks for, while the preset keeps the hierarchy on the
translations.

What it does not fix:

* **Softening.** The rigid-body modes are the near-null-space of the *elastic* operator. Once damage makes the
  displacement block indefinite, other directions of low energy appear that no coarse space of rigid bodies holds: on
  the damaged tunnel system (:ref:`measured-rigid-body-coarse-level`), the iterations fall from 61 to 36, against 67
  to 22 on the elastic one.
* **Scalar fields, and the scalar backend.** A scalar field has no rotations, and the scalar backend
  (``AMG(backendBlockSize=1)``) takes the rigid-body modes as its near-null-space itself
  (``AMG(rigidBodyNullspace=True)``, the default); ``TwoLevel(coarseSpace=RigidBodyModes())`` on either is rejected.


.. _blockamg-schur-complement:

Schur complement condensation: ``SchurComplement`` (opt-in)
-----------------------------------------------------------

Aggregation AMG fails on some dofs whatever the field split: the dofs of embedded bars with bond-slip are
the prominent case. A bar node is tied by bond springs to the nodes of the host element it lies in, so a
low-energy error has the bar follow the host, :math:`u_{bar} \approx \sum_i N_i u_i` -- an interpolation of
nodes at other positions, which an aggregate moving as a rigid body cannot represent. The bars are also
stiff 1D chains inside a softer 3D continuum, a strong anisotropy. The field-split preconditioner alone stalls on
such systems. (With perfect bond, the same relation is a multi-point constraint, eliminated before the solve, and
the field-split preconditioner without condensation is the right choice.)

``SchurComplement`` condenses these dofs exactly -- we call them :math:`b`, the condensed ("bar") dofs -- and leaves
AMG the remaining (kept) dofs :math:`k`:

.. math::

    \begin{bmatrix} A_{bb} & A_{bk} \\ A_{kb} & A_{kk} \end{bmatrix}
    \begin{bmatrix} x_b \\ x_k \end{bmatrix} =
    \begin{bmatrix} r_b \\ r_k \end{bmatrix},
    \qquad
    S = A_{kk} - A_{kb} A_{bb}^{-1} A_{bk}.

The Schur complement :math:`S` is the operator of the remaining dofs once :math:`x_b` is eliminated -- for
bars, the concrete stiffened by the bars as seen through their bond. It is again an operator of a continuum,
which is what the per-field hierarchies handle well. The preconditioner of the outer Krylov solve is the block
LU factorization,

.. math::

    y_b = A_{bb}^{-1} r_b, \qquad
    z_k = M_{\tilde S}^{-1} (r_k - A_{kb} y_b), \qquad
    z_b = A_{bb}^{-1} (r_b - A_{bk} z_k),

which is nothing but one symmetric block Gauss--Seidel sweep over the two blocks :math:`[b, k]`: the first and
the last step solve the :math:`b` block exactly (a direct factorization of :math:`A_{bb}`), the middle step
is the usual field-split sweep :math:`M_{\tilde S}^{-1}`, built on an approximation
:math:`\tilde S` of :math:`S` instead of on :math:`A`. With :math:`\tilde S = S` and exact field solves,
the preconditioner is :math:`A^{-1}`; the outer Krylov solve corrects for both approximations.

This is the inverse of :math:`A` written as the product of both block factors, with two solves with
:math:`A_{bb}`. By default (``blockFactors = "U"``) only the block upper triangular factor
:math:`U = \begin{bmatrix} A_{bb} & A_{bk} \\ 0 & S \end{bmatrix}` of
:math:`A = \begin{bmatrix} I & 0 \\ A_{kb} A_{bb}^{-1} & I \end{bmatrix} U` is inverted, with one solve:

.. math::

    z_k = M_{\tilde S}^{-1} r_k, \qquad z_b = A_{bb}^{-1} (r_b - A_{bk} z_k).

The remaining dofs are solved first, and the condensed ones follow them -- for bars, the bars follow the concrete,
which is exactly the relation AMG cannot represent.

Why one factor is enough: its last step solves the rows of :math:`b` exactly, so after the first outer iteration
the residual in these rows is zero, and from then on the outer Krylov solve iterates on the rows of :math:`k` alone,
with the operator :math:`S \tilde S^{-1}` -- the same as with the full block LU factorization, which saves one
iteration more. With :math:`\tilde S = S` and exact field solves, GMRES converges in two iterations instead of
one: :math:`A U^{-1}` is the unit block lower triangular factor :math:`L` of :math:`A = L U`, and
:math:`(L - I)^2 = 0`. With the approximations, the outer iterations were found to hardly grow, and every one saves
a solve with :math:`A_{bb}`. The block lower triangular factor (bars first, concrete held fixed) was tried too: its
last step leaves a residual in the rows of :math:`b`, coupled to :math:`k` by the stiff bond, and it needed
considerably more outer iterations.

**The approximate Schur complement.** :math:`S` is dense wherever :math:`A_{bb}^{-1}` is -- along a whole bar.
But the entries of the inverse of a banded, well-conditioned matrix decay exponentially away from the diagonal
(Demko, Moss and Smith 1984); physically, a bar on its bond is a beam on an elastic foundation, and a load on it
is felt only a few host elements away. :math:`\tilde S` therefore keeps only the significant entries:

* a banded component of :math:`A_{bb}` -- a chain: a bar, ordered along the bar by the reverse Cuthill--McKee
  ordering -- is cut into overlapping windows, cores of ``windowNodes`` nodes with halos of ``haloNodes`` nodes on
  either side; for the dofs of a core, :math:`A_{bb}^{-1}` is replaced by the inverse of the window's block, and
  only the core rows are kept (restricted additive Schwarz, Cai and Sarkis 1999). Windows count nodes, as the
  decay length is a length along the bar, whatever the dofs per node (a truss, a beam). A short component (a short
  bar) is a single window, which is exact;
* a component that is not banded (a zone of a 2D or 3D mesh, e.g. a softening zone) gets its exact correction from
  a factorization of its own block, solved for the coupled remaining dofs in batches of right-hand sides --
  windows along a chain would cut such a zone into pieces far smaller than the distance over which its inverse
  decays;
* entries :math:`|\tilde S_{ij}| < \tau \sqrt{|\tilde S_{ii} \tilde S_{jj}|}` are dropped (``dropTolerance``).

**Choosing the condensed dofs.** The markers of adaptive mesh refinement select elements
(:mod:`edelweissfe.adaptivity.marking`; any ``type`` with the options of its ``>>marker`` line), and :math:`b` holds
all dofs of the *interior nodes* of the marked elements: the nodes that have at least one element, all of them
marked. One rule serves every use::

    bar node:   bar, bond                -> all marked -> condensed
    host node:  bond, concrete, concrete -> not all    -> kept


* embedded bars: mark the bar elements and the bond elements (``elementSet`` markers for the bar and the bond
  element sets). A bar node belongs only to them; a host node also belongs to
  continuum elements, so it stays in :math:`k`;
* a softening zone: a ``fieldOutput`` marker on a ``perElement`` fieldOutput (damage, a hardening variable) with a
  threshold, and a ``halo`` so that the nodes of the zone itself become interior. All dofs of a node are condensed,
  e.g. displacement and nonlocal damage. A fieldOutput has no result before the first converged increment, so
  nothing is condensed until then.

The markers are evaluated again whenever what they read can change (``initialOnly`` is rejected): a fieldOutput
marker when its fieldOutput is updated, after a converged increment; a marker of sets or surfaces when the model
changes; a marker that cannot tell (e.g. ``recoveryError``) on every solve. A changed selection rebuilds
:math:`\tilde S` and the hierarchies. Should the markers select every
dof, nothing is condensed (with a warning): there would be nothing left for the field split, and a direct solver is
the better choice; more than half of the dofs is warned about too. Only the dofs of nodes can be condensed, and
always all fields of a node together: dofs that belong to no node (e.g. Lagrange multipliers) cannot be selected.

**Robustness.** If :math:`A_{bb}` cannot be factorized (a zone or a bond that lost all stiffness), or
:math:`\tilde S` cannot be formed (a singular window block of an :math:`A_{bb}` that factorizes as a whole), the solve
does without the condensation, for all dofs, with a warning: the condensed dofs come from the last converged
increment, so a cutback would meet the same block again. The condensation then keeps the state of its last successful
update. An approximate Schur complement with non-positive diagonal entries (the condensed zone does not hold all of
the softening, and AMG on the rest may struggle) is warned about as well.

**Within a solve.** Every solve cuts :math:`A_{bb}`, :math:`A_{bk}` and :math:`A_{kb}` out of the current matrix
(at positions found once per sparsity pattern) and factorizes :math:`A_{bb}`; the operator of its outer solve,
:math:`A` itself, is kept per sparsity pattern too, its values refreshed in place. :math:`\tilde S` is computed when
the hierarchies are rebuilt -- under the same reuse rules as without condensation -- and the hierarchies are
built on it: the field blocks are those of the remaining dofs (a field whose dofs are all condensed, such as the
rotations of embedded beams, has none), and the node coordinates of the near null-space and of the rigid-body
coarse level and the P1 maps of p-multigrid are restricted to the remaining nodes (a midside node whose edge endpoint is condensed becomes a
corner). Between rebuilds, the sweep keeps the couplings of the standing :math:`\tilde S`, while :math:`A_{bb}`,
:math:`A_{bk}` and :math:`A_{kb}` are those of the current matrix: unlike without condensation, where the couplings
between the fields are taken from the current matrix on every solve. The outer Krylov solve runs on :math:`A`
itself: the Schur complement sits under the outer Krylov solve, ``LGMRES(SchurComplement(...))``.

**Two sweeps.** The sweep in residual-correction form makes ``sweeps = 2`` exactly two steps of the stationary
iteration :math:`z \leftarrow z + M^{-1} (r - \tilde S z)` with the one-sweep preconditioner :math:`M^{-1}`,
which on bar systems was found a good balance between AMG cycles and outer iterations; ``sweeps = 1`` is
cheaper per iteration and needs more of them.

**The direct solver.** ``solveA_bb=DirectSolver(backend)`` takes any registered direct solver backend; ``superlu``
and ``pardiso`` factorize once for several right-hand sides, the others (``klu``, ``mumps``, ``petsclu``,
``umfpack``, ``panuapardiso``) factorize again for every one, which makes them a poor choice here. Every outer
iteration solves with :math:`A_{bb}`, so the cost of a back substitution counts most:
for the block-tridiagonal chains of bars, ``superlu`` (the default) is the cheaper one (with ``blockFactors =
"LU"``, twice per iteration). A large compact zone (a
softening zone of a 3D mesh) shifts the weight to the exact correction of :math:`\tilde S`, which solves for many
right-hand sides at once -- there a threaded factorization (``pardiso``) does better. For a softening zone, mind
that condensation pays only where AMG struggles with the zone: the field-split preconditioner alone handles a typical
gradient-enhanced damage band well, and the condensation then costs more per iteration and per rebuild of
:math:`\tilde S` (rebuilt whenever the zone grows) than it saves in iterations.

The condensation is a ``custom`` solver script, ``LGMRES(SchurComplement(condense=[...], preconditionS=...))``: a
complete one is :ref:`composed-schur-complement-recipe`, the arguments of
:class:`~edelweissfe.linsolve.composition.schurcomplement.SchurComplement` are listed in
:ref:`composed-schur-complement-arguments`.

.. _blockamg-worked-example:

A worked example
----------------

The whole method, written out with NumPy, SciPy and one AMGCL hierarchy per field on a small
displacement / damage system -- equilibration, field split, per-field AMG, one block Gauss--Seidel sweep as
the preconditioner, and the outer Krylov solve -- and the same solve as a composed solver. It is a test
of the test suite; ``displacementDamageSystem``, ``ExactFieldSolve`` and ``hierarchyLevels`` come from the
suite's ``tests/_linsolvesystems.py``:

.. literalinclude:: ../../../tests/test_blockamg_tutorial.py
   :start-after: # The worked example of the theory page.


Practical guidance
------------------

* Reach for the field-split preconditioner (``type=blockamg``) when a direct factorization no longer fits in memory, or when the
  factorization dominates the run. Below that, measure against ``pardiso`` rather than assuming.
* Run the first characterisation at ``verbosity=info`` and read the job's performance table.
  Outer iteration counts up to the low hundreds were the expected regime at the former defaults; with
  the current ones, mostly 5--50, with the hardest damage-onset solves reaching the cap; counts
  triggering the warning threshold, or cap hits, repeatedly point at the preconditioner, not the
  tolerance.
* Set ``OMP_NUM_THREADS`` for the AMGCL kernels as usual (see :doc:`parallelization`). Results are
  reproducible at a fixed thread count up to last-bit summation order; across thread counts the
  hierarchies differ slightly (see :ref:`blockamg-chebyshev`), and lowering the
  power-iteration setting of the vector-field default makes that difference large.
* Reach for the knobs in roughly this order: the tree of a field whose hierarchy is clearly the weak one (a
  ``custom`` script), then ``sweeps``/``symmetric`` for the sweep itself, then the tolerance.
  ``TwoLevel(coarseSpace=P1Interpolation())`` is an alternative per-field preconditioner, see
  :ref:`blockamg-p-multigrid`.
* The rigid-body coarse level is on for the 3D displacement field by default: give a 2D displacement field the
  coarse level in a ``custom`` script, and switch it off (``rigidBodyCoarseLevel=false``) on a model whose solves
  converge in a few outer iterations without it, see :ref:`blockamg-rigid-body-coarse-level`.
* When a run degrades reproducibly, capture it with ``dumps=slow`` plus a few ``dumpContextSolves`` rather than trying
  to reproduce it by rerunning the simulation (:ref:`composed-solve-dumps`).


Further reading
---------------

* S. C. Eisenstat, H. F. Walker, *Choosing the forcing terms in an inexact Newton method*, SIAM J.
  Sci. Comput. **17** (1996) 16--32 -- the adaptive outer tolerance, "choice 2".
* P. Vaněk, J. Mandel, M. Brezina, *Algebraic multigrid by smoothed aggregation for second and
  fourth order elliptic problems*, Computing **56** (1996) 179--196 -- the coarsening, and why it
  needs a near-null-space basis.
* A. H. Baker, E. R. Jessup, T. Manteuffel, *A technique for accelerating the convergence of
  restarted GMRES*, SIAM J. Matrix Anal. Appl. **26** (2005) 962--984 -- LGMRES and its recycled
  vectors.
* D. Demidov, *AMGCL: an efficient, flexible, and extensible algebraic multigrid implementation*,
  Lobachevskii J. Math. **40** (2019) 535--546 -- the library providing the hierarchies, the
  relaxation kernels and the native outer solver.
* N. Alkmim et al., Int. J. Numer. Methods Eng. (2026) -- the field-split block-preconditioning
  strategy for coupled multi-field fracture models that this solver follows.
* S. Demko, W. F. Moss, P. W. Smith, *Decay rates for inverses of band matrices*, Math. Comp. **43**
  (1984) 491--499 -- why the Schur complement of condensed bars is local.
* X.-C. Cai, M. Sarkis, *A restricted additive Schwarz preconditioner for general sparse linear
  systems*, SIAM J. Sci. Comput. **21** (1999) 792--797 -- the windows of the approximate Schur
  complement.
* H. C. Elman, D. J. Silvester, A. J. Wathen, *Finite Elements and Fast Iterative Solvers*, Oxford
  University Press (2014) -- block (Schur complement) preconditioners.
