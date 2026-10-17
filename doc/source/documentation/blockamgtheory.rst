Field-split block AMG: theory and configuration
================================================

This page documents the ``blockamg`` linear solver
(:mod:`~edelweissfe.linsolve.blockamg.blockamg`): a *field-split* algebraic-multigrid
preconditioner for the coupled multi-field equation systems EdelweissFE's implicit solvers produce
-- displacement coupled to a scalar nonlocal (gradient-enhanced) damage field, typically with
penalty contact, tie constraints and adaptive refinement on top. It is selected like any other
linear solver, ``linsolver=blockamg``, and is documented separately from the other entries in
:doc:`linsolvers` because it is the only one with a method of its own to explain rather than a
library call to name.

Everything on this page is implemented literally in
:mod:`~edelweissfe.linsolve.blockamg.blockamg`, :mod:`~edelweissfe.linsolve.nullspace` and
:mod:`~edelweissfe.linsolve.base`; the opt-in p-multigrid variant lives in
:mod:`~edelweissfe.linsolve.blockamg.ptwogrid`. The solver requires the optional ``amgcl``
extension (:mod:`edelweissfe.linsolve.amgcl.amgcl`) to be built.


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

``blockamg`` is a *feasibility-grade* solver in the sense that it buys reach, not necessarily speed.
On the reference models AMGCL's smoothed aggregation converges, but not tightly, on the
non-symmetric (contact-condensed) displacement block, so the outer Krylov solve needs
many iterations -- with the shipped defaults typically :math:`O(100)`; with the recommended
configuration below, mostly 5--60, and on the hardest damage-onset systems around 100. That is an
acceptable price where the alternative is not fitting in memory at all; at problem sizes a direct
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
  worked around. DOFs past the last node field (scalar variables, e.g. those introduced by indirect
  control) are folded into one trailing scalar block named ``scalar variables``.
* Beyond the field ranges, ``blockamg`` reads **node coordinates** from the model on its own, lazily,
  for the near-null-space construction described below (and, for the opt-in p-multigrid
  variant, the corner/midside element topology). Nothing has to be pushed in ahead of it.

:meth:`~edelweissfe.linsolve.base.LinearSolver.setFieldStructure` is a lower-level escape hatch for a
caller that knows the field blocks but has no model to hand over -- e.g. an offline script replaying
a captured system through the solver. On that path node coordinates are unavailable, and the solver
degrades gracefully to a translations-only near-null-space.


The solve pipeline
------------------

``blockamg`` sees only the condensed, Dirichlet-eliminated system: the nonlinear solver applies the
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
This matters for hierarchy reuse: the condensed matrix is pruned of exact zeros
(``pruneCondensedMatrixZeros``), which changes its sparsity pattern from one Newton iteration to
the next, and a changed pattern forces a hierarchy rebuild (step (4) below). The uncondensed
matrix keeps the assembler's fixed pattern.

Beyond that matrix and right-hand side, the solver is told nothing about the Newton loop -- like
every registered linear solver it is called as ``(A, b) -> x``. Both of its stateful mechanisms
(adaptive tolerance and hierarchy reuse) therefore *reconstruct* the loop state from
:math:`\lVert b \rVert` and from the matrix's own bookkeeping across calls.

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
    (A, b) ---> blockamg
                |
          (1)   Newton-state detection from ||b||
                    ||b|| > residualGrowthFactor * ||b||_prev  ?
                        yes -> new increment / cutback: eta = etaMax, force refresh
                        no  -> eta from Eisenstat-Walker forcing
                |
          (2)   equilibration     Ahat = D A D,  bhat = D b,  D = diag(1/sqrt|A_ii|)
                |
          (3)   field split       diagonal blocks Ahat_ii  +  couplings Ahat_ij
                |
          (4)   hierarchy state
                    refresh (first solve | block layout | size | nnz change |
                             new increment | staleness)
                        -> build one AMG hierarchy per field
                           (smoothed aggregation + Chebyshev, per-field near-null-space)
                    otherwise
                        -> reuse the standing hierarchies
                |
          (5)   outer Krylov solve on Ahat, preconditioned by
                    M = block Gauss-Seidel sweep, one AMG cycle per diagonal block
                |
          (6)   unscale  x = D z    and measure the TRUE residual  ||A x - b|| / ||b||
                    > eta and continuations left?
                        yes -> warm restart at a tighter tolerance, back to (5)
                        no  -> done
                |
          (7)   record ||b||, eta, nnz and the outer iteration count for the next solve
                v
    x ------> nonlinear solver
                |
                |  MPC slave recovery    x_s <- x_s + W x      (master-interpolated part)
                v
              the Newton correction


Anatomy of one solve
--------------------

Diagonal equilibration
~~~~~~~~~~~~~~~~~~~~~~

A coupled multi-field system has a large diagonal dynamic range: elastic stiffnesses, a
Helmholtz-like damage operator and penalty contact stiffnesses sit in the same matrix, and the
Dirichlet-eliminated rows carry a unit diagonal among them. AMG's strength-of-connection test is
scale-sensitive, so every solve begins with a symmetric diagonal (Jacobi) equilibration

.. math::

   \hat A = D A D, \qquad \hat b = D b, \qquad x = D z,
   \qquad D = \operatorname{diag}\!\left( \lvert A_{ii} \rvert^{-1/2} \right),

which is an exact diagonal similarity scaling: solving :math:`\hat A z = \hat b` and unscaling gives
the solution of :math:`A x = b`, whatever :math:`D` is. That is what makes it safe to *reuse* an
older :math:`D` together with a reused hierarchy (see below) -- a stale scaling can cost outer
iterations, never correctness.

Field split and off-diagonal couplings
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:math:`\hat A` is split along the field ranges into diagonal blocks :math:`\hat A_{ii}` (one per
field, the operators the AMG hierarchies are built for) and the off-diagonal coupling blocks
:math:`\hat A_{ij}`, which are needed on every solve regardless of whether the hierarchies are
rebuilt. All of these blocks, and the full-system operator used for the outer matrix-vector products
and the true-residual check, are handed to AMGCL's OpenMP-threaded sparse matrix-vector kernels:
SciPy's own sparse ``matvec`` is single-threaded C code irrespective of ``OMP_NUM_THREADS``, and on a
solver whose inner loop is nothing but sparse matrix-vector products that is a substantial share of
the wall-clock. On a model with contact or tie constraints the sparsity pattern changes on virtually
every Newton iteration, so there is no stable pattern to cache these threaded operators against and
they are rebuilt every call.

Per-field AMG hierarchies
~~~~~~~~~~~~~~~~~~~~~~~~~

Each diagonal block gets its own AMGCL hierarchy, built once per solve and applied many times (one
cycle per outer Krylov iteration). The defaults differ by nodal dimension: a **vector** field
(dimension > 1) is coarsened by smoothed aggregation with a strong-connection threshold of 0.01 and
smoothed by a degree-5 Chebyshev polynomial; a **scalar** field takes smoothed aggregation and
Chebyshev at AMGCL's own defaults. Both can be overridden per field with ``fieldPreconds``.

Two keys of a per-field parameter tree are not AMGCL parameters but select the wrapper's own backend
value type, and are consumed before the rest of the tree is forwarded verbatim:

``backendPrecision``
    ``"double"`` (default) or ``"float"``. A single-precision backend should in principle roughly
    halve the hierarchy's memory traffic, but measured across a full set of captured reference
    systems it inflated outer iteration counts by up to 30% and netted only about 3% in aggregate --
    plausibly because the Chebyshev smoother's own spectral-radius estimate loses accuracy in
    ``float32``, degrading smoother quality enough to eat the bandwidth saving. That holds for the
    scalar backend, which builds the whole hierarchy in single precision. With
    ``backendBlockSize = 3``, ``"float"`` instead sets the hierarchy up in double -- coarsening,
    Galerkin products, every spectral-radius estimate, the coarsest direct solve -- and stores only
    the level matrices the cycle reads in float, with double vectors and double accumulation.

``backendBlockSize``
    ``1`` (default, scalar) or the field's nodal dimension, for a block-valued backend operating on
    :math:`B \times B` nodal blocks. ``3`` for the displacement field is part of the recommended
    configuration below, measured faster per outer iteration (less index traffic) and validated
    inside the block Gauss--Seidel sweep on captured and live runs. Note that AMGCL's near-null-space
    path is unimplemented for block value types: a block hierarchy gets the identity block as its
    only near-null-space -- the translations -- and gives up the **rotational** rigid-body modes
    described next.

Near-null-space and the Chebyshev smoother
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Smoothed aggregation and the Chebyshev smoother have to be understood together, because they play
deliberately complementary roles and only work if they agree on what "smooth error" means.

* The **smoother** damps error components whose eigenvalues lie inside an estimated window
  :math:`[\,\mathrm{lower},\ \mathrm{higher}\,]\cdot\rho` of the operator's spectral radius
  :math:`\rho`. With the default ``lower = 0.01``, eigenvalues below 1% of the spectral radius are by
  construction **not** targeted by the smoother at all.
* Everything below that window -- including the operator's **near-null-space**, the directions in
  which it has almost no stiffness -- is left to the coarse-grid correction, which removes it only to
  the extent that smoothed aggregation's prolongation operator can represent it. Smoothed aggregation
  builds that prolongation from a supplied near-null-space basis.

For 3D linear elasticity the physical near-null-space is the six-dimensional space of **rigid-body
motions**: three translations and three infinitesimal rotations, each of which costs zero elastic
strain energy and therefore corresponds to a genuinely near-zero eigenvalue of the discretized
operator. :func:`~edelweissfe.linsolve.nullspace.rigidBodyNullspace` builds that full basis from the
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
error class either, so it has nowhere efficient to go. Measured on two real captured systems, the
full rigid-body basis needed about 28--31% fewer isolated per-field outer iterations than
translations alone, and -- unlike translations alone -- was robust to both thread count and the
Chebyshev power-iteration setting. (That was measured on the scalar backend; a block backend has no
way to take the rotations, see ``backendBlockSize``.)

That power-iteration setting is the smoother's second, independent subtlety, and the reason the
default is well above AMGCL's own. AMGCL estimates :math:`\rho` by a fixed number of power-iteration
steps whose starting vector is seeded per OpenMP worker thread. The estimate is deterministic for a
given thread count but *different* at a different thread count, because the number of independent
random streams changes with it even though the operator does not. At a short iteration budget the
estimate can be badly under-converged on a hard operator, which cripples the smoother: measured on
one captured solve, 1460 versus 43 outer iterations on the identical matrix with nothing differing
but ``OMP_NUM_THREADS`` (the true spectral radius, checked independently, barely moves between Newton
iterations there). Running the power iteration close to convergence removes the practical
thread-count sensitivity; the shipped default of 300 iterations measured 2.68x faster in aggregate
over ten captured degraded systems at 16 threads (811 s to 302 s), with no observed downside on
systems that were already fine. The one-time cost is paid per hierarchy build and is small against
the build itself. It does not make the hierarchy bitwise independent of the thread count: the
estimate still starts from per-thread random streams and agrees across thread counts only to its
convergence (and its reductions are summed in thread order, so even two runs at the same thread
count can differ in the last bits). Smoothed aggregation's ``estimate_spectral_radius`` (part of
the recommended configuration, with only 20 power iterations) uses the same per-thread seeded
estimate for the prolongation smoother's weight, so a hierarchy built with it depends measurably on
the thread count; results are reproducible at a fixed thread count, up to that last-bit summation
order.

The block Gauss--Seidel sweep
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The preconditioner :math:`M^{-1}` applied at each outer Krylov iteration is a block Gauss--Seidel
sweep over the fields. Starting from :math:`x_j = 0` for all fields, each field :math:`i` in turn
forms its local residual against the *current* estimates of all fields and corrects its own
estimate by one cycle of its own AMG hierarchy:

.. math::

   x_i \leftarrow x_i + M_i^{-1}\Bigl( r_i - \sum_{j} \hat A_{ij}\, x_j \Bigr),

where :math:`M_i^{-1}` denotes one AMG cycle on :math:`\hat A_{ii}`. On a field's first visit
:math:`x_i = 0`, and this is the plain :math:`x_i = M_i^{-1}(r_i - \sum_{j \neq i} \hat A_{ij} x_j)`;
the correction form matters on revisits, which an inexact :math:`M_i^{-1}` would otherwise throw
away. With ``symmetric = true`` (the default) each forward sweep is followed by a reverse-order sweep,
starting one field before the one just updated; ``sweeps`` repeats the (possibly reversed) sweep.
Neither is needed for the outer solve: right-preconditioned GMRES/LGMRES only needs :math:`M^{-1}`
to be a fixed linear operator, not a symmetric one -- the measured configurations use
``sweeps = 1``, ``symmetric = false``.

This is deliberately only an *approximate* inverse of the coupled operator -- which is exactly why
the true-residual bookkeeping below exists.

The outer Krylov solve
~~~~~~~~~~~~~~~~~~~~~~

With ``outerSolver = "amgcl_lgmres"`` (the default) the outer solve is AMGCL's own native
``amgcl::solver::lgmres``, driven through
:class:`~edelweissfe.linsolve.amgcl.amgcl.PyAMGCLLGMRESSolver`; the block sweep is passed straight in
as the preconditioner callback. The alternative, ``"scipy"``, uses
:func:`scipy.sparse.linalg.gmres` and is kept as a fallback. The native path exists because
orchestrating GMRES from Python -- the Arnoldi steps, orthogonalization and restart bookkeeping --
was measured as roughly 38% of this solver's wall-clock on a reference model, and, being serial
Python, anti-scales past one NUMA node on multi-socket hardware while the native implementation does
not. Both paths were gated live against each other at several thread counts and produce the same
Newton trajectory.

The two paths interpret the iteration budget differently. SciPy takes ``outerRestart`` as the restart
length and ``outerMaxiter`` as a number of restart *cycles*; AMGCL's ``lgmres`` bounds *all* Arnoldi
steps across all internal restarts with a single ``maxiter``, so their product is passed through as
one total-iteration budget. This is a generous translation, not an attempt to match SciPy's semantics
exactly.

LGMRES augments each restart with a few vectors carried over from the previous restart cycle of the
same solve. ``lgmresK`` sets how many, ``lgmresM`` the inner iterations per restart. Every solve
starts from an empty augmentation subspace. Keeping it alive *across* solves ("recycling") sounds
attractive -- a hard solve's converged subspace as a warm start for the next -- but measured on
captured systems it contributed nothing (bit-identical iteration counts), and on a live run it was
actively harmful: a solve that had already struggled left a poorly conditioned subspace which
compounded into a following solve needing 428 instead of 159 iterations, and in one run an outright
``NaN``. It was therefore removed; the former keys ``lgmresAlwaysReset`` and
``lgmresResetOnNewIncrement`` are still accepted at their non-recycling values ``true`` / ``false``,
with a deprecation note, and rejected otherwise.


Adaptive outer tolerance (Eisenstat--Walker forcing)
----------------------------------------------------

Most Newton iterates do not need their linear solve solved tightly, and loosening the requested
tolerance for those cuts outer iterations substantially. By default ``blockamg`` therefore chooses
its own outer tolerance per solve by Eisenstat--Walker forcing ("choice 2"), from the ratio of
successive residual norms:

.. math::

   \eta_k = \gamma \left( \frac{\lVert b_k \rVert}{\lVert b_{k-1} \rVert} \right)^{\alpha},
   \qquad \gamma = 0.9, \quad \alpha = \tfrac{1}{2}\bigl(1+\sqrt 5\bigr),

clamped to :math:`[\eta_\mathrm{min}, \eta_\mathrm{max}]`. The classic safeguard against
over-loosening is applied: a large tightening step is only trusted if the previous tolerance was
already small, i.e. :math:`\eta_k` is raised back to :math:`\gamma \eta_{k-1}^{\alpha}` whenever that
quantity exceeds 0.1. With no meaningful history to compare against -- the first solve, a residual of
exactly zero, or the solve right after a detected increment or cutback, where the ratio across the
jump says nothing about Newton convergence -- the tolerance falls back to :math:`\eta_\mathrm{max}`.

A solve is treated as the first of a new increment (or a cutback) when its residual norm exceeds
``residualGrowthFactor`` times the previous solve's. That single test drives both the tolerance reset
and a hierarchy refresh.

Pass an explicit ``outerTol`` to pin a fixed relative tolerance instead and switch the scheme off
entirely. Adaptive forcing is safe to use as the default only because the requested tolerance is
enforced on the *true* residual, which is the subject of the next section: without that enforcement,
a loosened request combined with an imperfect preconditioner can quietly let the true residual run
looser still, which does change the Newton path.


True-residual enforcement
-------------------------

The outer Krylov method's own stopping test is not on the true residual: AMGCL's LGMRES is
right-preconditioned and tests the residual of the *scaled* (equilibrated) system,
:math:`\lVert \hat b - \hat A z \rVert / \lVert \hat b \rVert`; SciPy's GMRES (``"scipy"``)
tests its preconditioned residual. Both can differ substantially from the true relative residual,
so "converged" can mean a true relative residual well above the requested :math:`\eta`; measured on
a reference model, one solve reached :math:`1.6\cdot 10^{-2}` where :math:`10^{-4}` had been asked
for.

Every solve therefore measures the true relative residual :math:`\lVert A x - b \rVert / \lVert b
\rVert` explicitly (computed as :math:`D^{-1}(\hat A z - \hat b)`, which is exact and rides on the
threaded operator already built, rather than paying a second unscaled matrix-vector product). If it
still exceeds :math:`\eta`, the solve is **continued**: a warm restart from the current iterate at a
tighter requested tolerance, up to ``trueResidualMaxContinuations`` times. By default the tolerance
is tightened geometrically by a fixed factor of 0.01 per round -- deliberately expressed purely in
the units of the requested relative tolerance, which the Krylov solver interprets consistently by
construction, rather than being derived from the solver's internal residual bookkeeping. Setting
``trueResidualMaxContinuations = 0`` disables enforcement: the outer solve's own test decides.

Gap-compensated tolerance
~~~~~~~~~~~~~~~~~~~~~~~~~

Because the system is equilibrated, the outer Krylov method minimises the *scaled* residual
:math:`\lVert \hat b - \hat A z \rVert` while acceptance is judged on the *true* relative residual.
The ratio between the two -- the **gap** -- has been measured between 1.0 and 3.5 on the first
reference model and up to about 40 on a contact model with multi-point constraints (stiff rows are
down-weighted by the equilibration), and it is why
almost every solve needs a continuation in the default configuration: on a reference
gradient-enhanced damage model, 4467 of 4499 solves ran one, i.e. effectively every linear solve was
performed twice. A continuation is disproportionately expensive, because restarting discards most of
the accumulated Krylov subspace.

With ``gapCompensatedTolerance = true`` the gap is measured on each pass (as the achieved true
relative residual over the achieved scaled relative residual of the same solution -- not over the
tolerance the pass was given, which a pass stopped at the iteration cap does not reach; that version
inflated the gap by the unreached factor and made every following first pass ask for that much more
accuracy), smoothed across solves by an exponential moving
average and floored at 1, and used to pre-compensate the *first* pass: it is asked for
``gapSafetyFactor``:math:`\cdot\,\eta\,/\,\text{gap}` so that it lands on target in one pass. Any
continuation that is still needed then tightens by the *measured* gap rather than by the fixed factor
(never looser, and always at least a factor of two per round so the loop still terminates).

.. note::
   **Why this is not the shipped default** (it is part of the recommended configuration below,
   together with ``trueResidualMaxContinuations = 0`` and ``gapMaxFactor = 10``). Per increment it is
   a clear win: on a live
   12-increment window of a reference model, continuations dropped from 126 to 1 and outer iterations
   from 8943 to 7124 (-20%), with no cutbacks or material return-mapping failures. But the default
   path happens to over-solve every system by roughly 30x, and the Newton iteration quietly relies on
   that: converging merely inside the requested tolerance costs about one extra Newton iteration per
   increment, which is enough to park the run above the adaptive time stepper's ``criticalIter``
   threshold and so stop it from growing the time increment. In the same window the default
   configuration grew its increment and covered more simulated time, while this one never grew.
   Judged per unit of *simulated time* rather than per increment, the per-increment saving is largely
   or wholly given back. ``gapSafetyFactor`` is the knob between the two regimes -- smaller values
   solve more accurately and behave more like the default -- and its optimum has not been explored.


Hierarchy reuse across Newton iterations
----------------------------------------

Building the per-field hierarchies is a large fraction of a solve, and it is avoidable when the
Jacobian has barely moved since the last one. ``blockamg`` therefore keeps its hierarchies (and the
equilibration they were built for) standing across solves by default.

This is safe in a way the adaptive tolerance is not. The outer Krylov solve always operates on the
current, freshly equilibrated matrix; only the preconditioner may be stale. A stale preconditioner
costs a few extra outer iterations at the *same* requested tolerance and cannot by itself change the
converged solution or the Newton path.

The hierarchies are rebuilt when

* there is nothing to reuse yet (the first solve),
* the field-block layout or the system size changed (e.g. an adaptive-refinement event),
* the number of stored nonzeros changed -- a free :math:`O(1)` proxy for "the sparsity pattern
  changed". It is not exact (two different patterns could coincidentally share an ``nnz``) but it
  errs safely, since an unnecessary refresh costs time and never correctness. A pattern change is not
  a graceful degradation: measured on one transition, a hierarchy built for the previous pattern
  needed 494 outer iterations against 94 for a fresh one, an outright wall-clock regression,
* a residual jump marks a new increment or a cutback (see ``residualGrowthFactor``) and
  ``refreshOnNewIncrement`` is set (the default), or
* the previous solve's outer iteration count exceeded both ``hierarchyStalenessFactor`` times and 15
  more than the count of the solve the hierarchies were built for -- a growing count being the signal
  that the reused hierarchies are drifting away from the current Jacobian. The reference is that
  first solve, not the previous one, so a slow drift (12, 17, 25, 37, 50 iterations) is caught; the
  absolute margin keeps a factor alone from firing on noise at small counts, or
* the previous solve hit the outer iteration cap. A missed tolerance alone does not trigger a
  rebuild: with continuations off it typically comes from the gap between the scaled residual the
  outer solve converges and the true one (stiff rows down-weighted by the equilibration, e.g. on a
  model with multi-point constraints), which a fresh hierarchy cannot close -- measured 18% slower on
  such a model when it did trigger rebuilds.

On a model with contact or tie constraints the active constraint set can change between Newton
iterations, and then the sparsity pattern -- and with it the nonzero count -- changes and the
hierarchies are rebuilt. How often that happens depends on the model: the multi-point condensation
used to drop exact zeros (``pruneCondensedMatrixZeros``), which changed the count on almost every
Newton iteration even when the constraint set did not; with that switched off, reuse pays on such
models too (on a contact model with 53,701 multi-point constraint slaves, a configuration relying on
it ran six increments in 917 s instead of 1328 s).

The native LGMRES instance is deliberately *not* torn down on a hierarchy refresh -- only on an
actual system-size change, which is the one condition that invalidates AMGCL's preallocated scratch
vectors.


Configuration keys
------------------

``blockamg`` is selected after the ``*solver`` keyword. A ``linsolverConfigFile`` is optional: the
block structure is discovered from the model, so the file carries only solver knobs, and every key is
optional. A key that is not listed in the tables below raises an error rather than being ignored.

.. code-block:: edelweiss

    *solver, solver=NIST, name=theSolver
    linsolver=blockamg
    linsolverConfigFile=blockamg.json

.. code-block:: json

    {
        "outerSolver": "amgcl_lgmres",
        "sweeps": 1,
        "symmetric": true,
        "verbosity": "info"
    }

Outer Krylov solve
~~~~~~~~~~~~~~~~~~

.. list-table:: Outer Krylov solve keys
    :width: 100%
    :widths: 20 10 45
    :header-rows: 1

    * - Key
      - Default
      - Meaning
    * - ``outerSolver``
      - ``"amgcl_lgmres"``
      - Which outer Krylov implementation to use: ``"amgcl_lgmres"`` for AMGCL's own native,
        OpenMP-threaded LGMRES, or ``"scipy"`` for :func:`scipy.sparse.linalg.gmres` as a fallback.
    * - ``outerTol``
      - unset
      - A fixed outer relative tolerance, switching off Eisenstat--Walker forcing. Unset, ``null``
        and the literal string ``"adaptive"`` all mean "use forcing"; pass a float to pin a
        tolerance.
    * - ``outerRestart``
      - ``100``
      - Restart length. With ``"scipy"`` this is the GMRES restart parameter; with
        ``"amgcl_lgmres"`` its product with ``outerMaxiter`` becomes one total Arnoldi-step budget.
    * - ``outerMaxiter``
      - ``8``
      - Maximum number of restart cycles (``"scipy"``), or a factor of the total iteration budget
        (``"amgcl_lgmres"``, see above).
    * - ``lgmresM``
      - ``30``
      - ``"amgcl_lgmres"`` only: inner iterations per outer restart.
    * - ``lgmresK``
      - ``3``
      - ``"amgcl_lgmres"`` only: number of augmentation vectors carried between the restart cycles of
        one solve.
    * - ``lgmresAlwaysReset``, ``lgmresResetOnNewIncrement``
      - ``true``, ``false``
      - Deprecated (cross-solve recycling was removed): accepted only at these values.

Block preconditioner and per-field hierarchies
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table:: Block preconditioner keys
    :width: 100%
    :widths: 20 10 45
    :header-rows: 1

    * - Key
      - Default
      - Meaning
    * - ``sweeps``
      - ``1``
      - Block Gauss--Seidel sweeps per preconditioner application.
    * - ``symmetric``
      - ``true``
      - Follow each sweep by a reverse-order sweep (starting one field before the one just
        updated). Not required by the GMRES-type outer solve.
    * - ``useRigidBodyNullspace``
      - ``true``
      - Give every vector field the full six-mode (3D) or three-mode (2D) rigid-body
        near-null-space. Falls back to translations only for a field whose node coordinates are
        unavailable. Set ``false`` to force translations only unconditionally -- worth roughly
        28--31% more per-field outer iterations, so only for comparison purposes.
    * - ``fieldPreconds``
      - ``{}``
      - Mapping of field name (e.g. ``"displacement"``, ``"nonlocal damage"``) to an AMGCL
        preconditioner parameter tree replacing the dimension-based default for that field,
        including ``backendPrecision`` / ``backendBlockSize`` and the Chebyshev
        ``relax.power_iters`` / ``relax.lower`` / ``relax.higher`` settings.
    * - ``p1FieldNames``
      - ``[]``
      - **Opt-in.** Vector fields to precondition with
        the p-multigrid two-grid variant instead of the single-level default; see below.
    * - ``p1Nu``
      - ``1``
      - p-multigrid: fine Chebyshev sweeps before and after the coarse-grid correction.
    * - ``p1NuPre``, ``p1NuPost``
      - ``p1Nu``
      - p-multigrid: separate pre- and post-smoothing sweep counts, e.g. ``0`` / ``1`` for a cycle
        that starts with the coarse-grid correction and ends on one smoothing sweep.
    * - ``p1FineDegree``
      - ``5``
      - p-multigrid: fine Chebyshev polynomial degree.
    * - ``p1FineBlockSize``
      - ``1``
      - p-multigrid: storage of the fine operator, ``1`` (scalar entries) or ``3`` (node-wise
        :math:`3\times 3` blocks, 3D fields only); see below.
    * - ``p1FineValueType``
      - ``"double"``
      - p-multigrid: value type of the fine operator for the smoother and residual, ``"double"`` or
        ``"float"``; see below.
    * - ``p1CoarseValueType``
      - ``"double"``
      - p-multigrid: value type of the coarse AMGCL hierarchy, ``"double"`` or ``"float"`` (AMGCL's
        single-precision backend, setup included).
    * - ``p1FineSpectralRadiusReuse``
      - ``false``
      - p-multigrid: estimate the fine smoother's spectral radius in full on a field's first build
        only, and only re-check it on later rebuilds; see below.
    * - ``p1CoarsePrecond``
      - ``null``
      - p-multigrid: the coarse level's AMGCL preconditioner tree. Unset, it is smoothed
        aggregation (``eps_strong`` 0.01) with a degree-4 Chebyshev smoother and ``npre`` =
        ``npost`` = 2.

Overriding a field's parameter tree replaces the default outright, so start from the shipped default
and change only what is intended. The defaults are

.. code-block:: json

    {
        "displacement": {
            "backendPrecision": "double",
            "backendBlockSize": 1,
            "coarsening": {"type": "smoothed_aggregation", "aggr": {"eps_strong": 0.01}},
            "relax": {"type": "chebyshev", "degree": 5, "power_iters": 300, "lower": 0.01},
            "npre": 1,
            "npost": 1
        },
        "nonlocal damage": {
            "backendPrecision": "double",
            "backendBlockSize": 1,
            "coarsening": {"type": "smoothed_aggregation"},
            "relax": {"type": "chebyshev"}
        }
    }

-- the first tree being the default for any field of nodal dimension > 1, the second for any scalar
field. Note that AMGCL silently ignores parameter keys it does not recognise (warning only on
stderr), so check its stderr if an override appears to do nothing.

Adaptive tolerance and true-residual enforcement
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table:: Tolerance keys
    :width: 100%
    :widths: 20 10 45
    :header-rows: 1

    * - Key
      - Default
      - Meaning
    * - ``etaMin``
      - ``1e-6``
      - Lower clamp on the Eisenstat--Walker forcing tolerance.
    * - ``etaMax``
      - ``3e-4``
      - Upper clamp on the forcing tolerance, and the tolerance used whenever there is no usable
        residual history (first solve, or the solve right after a detected increment or cutback).
    * - ``ewGamma``
      - ``0.9``
      - The forcing scheme's :math:`\gamma`.
    * - ``ewAlpha``
      - ``1.618033988749895``
      - The forcing scheme's :math:`\alpha`, the golden ratio (the classic "choice 2" value).
    * - ``trueResidualMaxContinuations``
      - ``2``
      - How many warm-restart continuations may be spent enforcing the requested tolerance on the
        true residual. ``0`` disables enforcement.
    * - ``gapCompensatedTolerance``
      - ``false``
      - **Opt-in.** Pre-compensate the first pass for the scaled-to-true residual gap so that it
        lands on target without a continuation. Read the note above before enabling it: the
        per-increment win is largely given back per unit of simulated time.
    * - ``gapSafetyFactor``
      - ``0.3``
      - Safety factor on the gap-compensated tolerance; only used when the above is enabled. Smaller
        values solve more accurately and behave more like the default configuration.

Hierarchy reuse
~~~~~~~~~~~~~~~

.. list-table:: Hierarchy reuse keys
    :width: 100%
    :widths: 20 10 45
    :header-rows: 1

    * - Key
      - Default
      - Meaning
    * - ``residualGrowthFactor``
      - ``4.0``
      - A solve whose :math:`\lVert b \rVert` exceeds this multiple of the previous solve's is
        treated as the first of a new increment (or a cutback): the forcing tolerance resets to
        ``etaMax`` and the hierarchies are refreshed.
    * - ``hierarchyStalenessFactor``
      - ``1.5``
      - Refresh the hierarchies before the *next* solve if this solve's outer iteration count
        exceeded both this factor times and 15 more than that of the solve the hierarchies were built
        for. A cap hit always refreshes them.
    * - ``refreshOnNewIncrement``
      - ``True``
      - Refresh the hierarchies on a residual jump (new increment or cutback). With ``False`` the
        first Newton iteration of an increment keeps the standing hierarchies; the layout/pattern
        checks and ``hierarchyStalenessFactor`` still apply.
    * - ``hierarchyDropTol``
      - ``0.0`` (off)
      - Build each field's AMG hierarchy from a *sparsified* copy of its diagonal block: drop
        off-diagonal :math:`a_{ij}` where :math:`|a_{ij}| < \text{tol}\cdot\sqrt{|a_{ii}| |a_{jj}|}`.
        Only the preconditioner is sparsified -- the operator the Krylov method applies is
        untouched -- so this cannot change the converged solution, only the iteration count needed
        to reach it. Worth up to **1.81x on the linear solve** on operators that store many
        numerically negligible entries (meshfree RKPM discretisations are the motivating case). Off
        by default because the useful range is operator-dependent.
    * - ``hierarchyDropLumping``
      - ``false``
      - When ``hierarchyDropTol`` drops an entry, add it onto its row's diagonal instead of
        discarding it, so row sums are preserved exactly and the constant near-null-space vector
        survives filtering -- the AMG literature's preferred filtered-matrix construction. Measured
        neutral at best on operators tested so far (no change at ``1e-4``, 8.1% *slower* at
        ``1e-2``, with outer GMRES iteration count unchanged either way): the whole penalty is the
        extra pass over the matrix. Off by default; plain truncation is simpler and faster.

Diagnostics
~~~~~~~~~~~

.. list-table:: Diagnostic keys
    :width: 100%
    :widths: 20 10 45
    :header-rows: 1

    * - Key
      - Default
      - Meaning
    * - ``verbosity``
      - ``"warning"``
      - One of ``"silent"``, ``"warning"``, ``"info"``, ``"debug"``; see below.
    * - ``warnOuterIterationsThreshold``
      - ``100``
      - Outer iteration count past which a solve emits a warning even at the default verbosity --
        a preconditioner-quality red flag.
    * - ``dumpOnDegradationDir``
      - unset
      - Directory (created if missing) to write degraded solves' systems to. Unset disables the
        whole mechanism, which then costs nothing.
    * - ``dumpOnDegradationThreshold``
      - ``warnOuterIterationsThreshold``
      - Outer iteration count above which a solve's system is dumped. Defaulting to the warning
        threshold makes a dump and its warning fire on exactly the same solves.
    * - ``dumpOnDegradationMaxDumps``
      - ``10``
      - Process-wide ceiling on the number of dumped systems, counting context solves. A disk-space
        guard: a badly degrading run could otherwise trigger on every remaining solve.
    * - ``dumpOnDegradationContextSolves``
      - ``0``
      - How many solves immediately *preceding* a degraded one to dump as well.

Verbosity levels
~~~~~~~~~~~~~~~~

Messages go through the shared :class:`~edelweissfe.journal.journal.Journal`, and this level decides
whether the solver attempts to log at all -- the Journal's own level then decides whether the attempt
is shown.

``"silent"``
    Nothing at all.

``"warning"`` (default)
    Only abnormal conditions, so a healthy run stays quiet: an outer iteration count past
    ``warnOuterIterationsThreshold``, a true residual still above the requested tolerance after every
    continuation, the outer solver reporting non-convergence, and any degradation dump written.

``"info"``
    Adds the field-block layout once, and turns on the "linear solve" column of the nonlinear solver's
    per-iteration table: outer iteration count, achieved true residual (and whether it met the
    requested tolerance) and number of continuations of each solve. This is the level to run at when
    first characterising a model on ``blockamg``.

``"debug"``
    Adds, per solve, whether the hierarchies were rebuilt (and why) or reused, the requested
    tolerance and the wall-clock, and one line per true-residual continuation attempt.

Per-stage wall-clock -- equilibration, the equilibrated system (the scaled operator and the coupling
blocks between fields, built in one threaded pass), hierarchy build (per field, and per stage of a
p-multigrid build), near-null-space construction, outer solve, and continuations -- is recorded through
:mod:`~edelweissfe.utils.performancetiming` regardless of verbosity, and appears nested under "linear
solve" in the job's performance table. That table, not the log, is the first place to look when
deciding whether a run is dominated by hierarchy builds or by outer iterations.

Capturing degraded solves
~~~~~~~~~~~~~~~~~~~~~~~~~

``dumpOnDegradationDir`` writes the raw ``(A, b)`` of a solve -- before equilibration, in the same
format the ``matrixdump`` solver uses, so the same offline tooling reads either -- together with the
field-block layout active for that solve, which is what a replay needs in order to go back through
this same block preconditioner without a live model. Unlike ``matrixdump``, which dumps at ordinals
fixed *before* the run, this triggers on the solve's own outcome, so a pathological system (a
late-increment, heavily damaged Jacobian, say) can be captured without knowing in advance which solve
will misbehave.

``dumpOnDegradationContextSolves`` exists because a single snapshot cannot distinguish "this operator
is intrinsically hard" from "this solve's own carried-over state degraded independently of the
operator" -- such as a reused, now-stale hierarchy. Both look identical from
the matrix alone, and a fresh solver replaying the matrix on its own can converge in a fraction of
the live iteration count, which is a genuinely misleading result. Capturing the preceding sequence
lets an offline replay feed one persistent solver instance the same solves in the same order.

A ``manifest.jsonl`` in the dump directory records, per dumped solve, its role (trigger or context),
its field blocks, and the solver state at that point: outer iterations, achieved true residual,
requested tolerance, continuations, and whether the solve refreshed, saw a pattern change, or was
detected as a new increment. The staleness question can often be answered from the manifest alone,
without a replay.


p-multigrid (opt-in)
--------------------

For a field discretized entirely with quadratic serendipity elements, ``blockamg`` can replace that
field's single-level hierarchy with a genuine two-grid V-cycle
(:class:`~edelweissfe.linsolve.blockamg.ptwogrid.PTwoGridPreconditioner`): :math:`\nu` Chebyshev
sweeps on the field's own equilibrated block, restriction of the residual through
:math:`P^\mathsf{T}`, one AMGCL cycle on the Galerkin-projected coarse operator :math:`A_1 =
P^\mathsf{T} A P`, prolongation through :math:`P`, and :math:`\nu` more Chebyshev sweeps. The
coarse space is purely topological -- the mesh's corner nodes -- so no re-discretization is involved:
:math:`P` is the identity on corner nodes and the average of the two edge-endpoint corners on each
midside node. The coarse solve is given the same rigid-body near-null-space treatment as the
single-level default, restricted to the corner-node subset.

Its preconditioner is rebuilt exactly when the single-level hierarchies are (see the reuse rules
above), and its coarse solve gets a near-null-space while the fine-level Chebyshev smoother, having
no coarsening step of its own, needs none. Measured against the recommended single-level
configuration below with the same outer settings (fine operator in float, ``p1NuPre = 0``,
``p1NuPost = 1``, fine degree 16, the same coarse ``estimate_spectral_radius``), it was about 12%
slower on a cylinder sequence and level within noise on a contact sequence, with the cheaper
rebuild -- an alternative worth measuring on a model with expensive rebuilds, not the default.
Enable it per field with ``p1FieldNames``; the topology map is computed lazily from the model and
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
free submatrix, and it allows ``p1FineBlockSize = 3``: the fine operator is stored in node-wise
:math:`3\times 3` blocks even where a node is only partially constrained, which roughly halves the
memory traffic of the fine smoother and residual -- the dominant cost of the apply.
``p1FineValueType = "float"`` halves the bytes per stored entry once more. Only the stored entries
are rounded: vectors and every sum stay double (a float matrix times a double vector), and the
spectral radius is estimated and re-checked on the double operator before it is rounded -- unlike
``backendPrecision = "float"``, where AMGCL's own power iteration runs in single precision.

The fine Chebyshev smoother uses AMGCL's polynomial, with the bounds of the vector-field default
(``lower`` 0.01, ``higher`` 1.1 times the spectral radius; AMGCL's own defaults are 1/30 and 1), but its
spectral-radius estimate -- 300 power iterations, the most
expensive stage of the build -- starts from a deterministic, thread-count-independent vector, and can
be reduced to a re-check: with ``p1FineSpectralRadiusReuse`` a field's first build estimates it in
full, and every later rebuild continues the power iteration for 15 steps from the previous build's
dominant vector. A re-check estimate up to 1% above the previous radius keeps it (the full estimate is
itself converged only to about 1e-5); one further above replaces it; and one above it by more than the
10% margin of ``higher`` -- an operator that stiffened more than the smoother's bounds cover -- triggers
the full estimate again. The cache is dropped when the model
changes (e.g. after adaptive refinement).


Recommended configuration
-------------------------

The shipped defaults are deliberately unchanged; the following configuration file -- the winner of a
systematic comparison on two captured models (a 0.84M-DOF gradient-enhanced damage cylinder through
its damage onset, and a 1.07M-DOF contact anchor model with 53,701 multi-point constraint slaves) --
is the recommended starting point for 3D displacement problems:

.. code-block:: json

    {
        "outerSolver": "amgcl_lgmres",
        "outerRestart": 50,
        "outerMaxiter": 1,
        "lgmresM": 50,
        "sweeps": 1,
        "symmetric": false,
        "etaMin": 1e-4,
        "etaMax": 1e-2,
        "hierarchyStalenessFactor": 3,
        "trueResidualMaxContinuations": 0,
        "gapCompensatedTolerance": true,
        "gapSafetyFactor": 0.3,
        "gapMaxFactor": 10.0,
        "fieldPreconds": {
            "displacement": {
                "backendPrecision": "float",
                "backendBlockSize": 3,
                "coarsening": {
                    "type": "smoothed_aggregation",
                    "aggr": {"eps_strong": 0.01},
                    "estimate_spectral_radius": true,
                    "power_iters": 20
                },
                "relax": {"type": "chebyshev", "degree": 16, "power_iters": 300, "lower": 0.01, "higher": 1.1},
                "npre": 0,
                "npost": 1
            }
        }
    }

What each part buys: 3x3 blocks stored in float (set up in double, see ``backendPrecision``) cut the
cost per outer iteration; one high-degree post-smoothing sweep instead of a symmetric V(1,1) cycle,
and smoothed aggregation's own spectral-radius estimate for its prolongation smoother, cut outer
iterations; ``lgmresM = 50`` restarts less often; a staleness factor of 3 reuses hierarchies longer.
On a model with multi-point constraints, also set ``pruneCondensedMatrixZeros = False`` in the
nonlinear solver, so that the hierarchies can be reused at all (see the reuse rules above).

Measured live, from the same checkpoints, against the shipped defaults: the contact model's six
increments in 676 s instead of 1328 s (-49%), and the damage cylinder at about 10.3 s instead of
25.5 s per Newton iteration (-60%). Offline, on 13 / 12 captured consecutive systems, the same
configuration with a 200-iteration cap took 91--93 s / 60--61 s against 107--112 s / 72--73 s with
the displacement blocks stored in double, at identical outer iteration counts.

Practical guidance
------------------

* Reach for ``blockamg`` when a direct factorization no longer fits in memory, or when the
  factorization dominates the run. Below that, measure against ``pardiso`` rather than assuming.
* Run the first characterisation at ``verbosity = "info"`` and read the job's performance table.
  Outer iteration counts up to the low hundreds are the expected regime at the shipped defaults,
  mostly below 60 with the recommended configuration; counts triggering the warning threshold
  repeatedly point at the preconditioner, not the tolerance.
* Set ``OMP_NUM_THREADS`` for the AMGCL kernels as usual (see :doc:`parallelization`). Results are
  reproducible at a fixed thread count up to last-bit summation order; across thread counts the
  hierarchies differ slightly (see the Chebyshev power-iteration discussion above), and lowering the
  shipped power-iteration setting makes that difference large.
* Reach for the knobs in roughly this order: ``fieldPreconds`` for a field whose hierarchy is clearly
  the weak one, then ``sweeps``/``symmetric`` for the sweep itself, then the tolerance keys.
  ``p1FieldNames`` is an alternative per-field preconditioner, see above.
* When a run degrades reproducibly, capture it with ``dumpOnDegradationDir`` plus a few
  ``dumpOnDegradationContextSolves`` rather than trying to reproduce it by rerunning the simulation.

.. note::
   The solver's constructor additionally accepts a ``p1Maps`` argument, taking precomputed
   p-multigrid topology maps directly. It is deliberately not reachable from a
   ``linsolverConfigFile`` -- it exists for a script that constructs
   :class:`~edelweissfe.linsolve.blockamg.blockamg.BlockAMGSolver` itself and already knows the
   topology, e.g. when replaying a captured system offline. Live runs use ``p1FieldNames``.


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
