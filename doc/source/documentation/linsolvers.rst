Linear solvers
==============

Linear solvers are defined in EdelweissFE after the ``*solver`` keyword using ``linsolver`` and an optional configuration file ``linsolverConfigFile`` as a data line.
The ``linsolverConfigFile`` needs to be in ``.json`` format.

Choose a linsolver after the ``*solver`` keyword:

.. code-block:: edelweiss

    *solver, solver=NIST, name=theSolver
    linsolver=gmres
    linsolverConfigFile=opt.json

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
    * - ``gmres``
      - ✗
      - ``edelweissfe.linsolve.gmres.gmres``
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

* ``gmres`` -- GMRES preconditioned by a ``pyamg`` smoothed-aggregation hierarchy over the whole
  matrix.
* ``amgcl`` -- the AMGCL library's own solver/preconditioner combinations, configured through its
  JSON parameter tree.
* ``blockamg`` -- a *field-split* variant for coupled multi-field models: one AMG hierarchy per
  physical field (e.g. displacement, nonlocal damage), combined by a block Gauss--Seidel sweep to
  precondition an outer GMRES/LGMRES. This is the route to problem sizes a direct factorization
  cannot reach at all.

**A diagnostic wrapper**, which does not solve anything itself:

* ``matrixdump`` -- writes every :math:`(A, b)` pair it is handed to disk, then hands the solve to
  a real solver chosen with its ``delegate`` option, so the simulation proceeds normally. The point
  is to lift solver comparisons out of the finite element run: rerunning a simulation per solver
  variant is slow and not like-for-like (a variant that changes the Newton iterates changes the
  *sequence* of matrices being compared). Dump one authentic sequence once, then replay it offline
  with ``scripts/benchmark_linsolve.py`` so every variant sees byte-identical input. Use it to
  investigate solver performance, never in a production run.

Several linsolvers accept an optional configuration file ``linsolverConfigFile`` (a ``.json`` file), among them ``gmres``, ``amgcl``, ``blockamg`` and ``matrixdump``; the direct solvers ignore it (``pardiso`` additionally reads a single ``reuseSymbolicFactorization`` flag).

Choose the options for the linsolver (in this case ``gmres``) in an extra file:

.. code-block:: json

    	{
	"precondopts":
	{
	"presmoother": ["block_gauss_seidel", {"iterations": 15}],
	"postsmoother": ["block_gauss_seidel", {"iterations": 15}],
	},
	"linsolveopts": {"maxiter": 1, "restart": 1500}
	}


The ``blockamg`` solver
-----------------------

``blockamg`` is a field-split block-AMG solver for **large coupled multi-field systems** (e.g. displacement + gradient-enhanced damage). It is the O(n)-memory route to problem sizes a direct factorization cannot reach — past roughly a million DOFs its fill-in exceeds memory, whereas algebraic multigrid stays linear.

Applied *monolithically*, AMG is ineffective on such a coupled system (a single hierarchy cannot represent the disparate fields at once — their physical scales and near-null-spaces differ). ``blockamg`` instead builds **one AMGCL algebraic-multigrid hierarchy per field** and combines them with a **block Gauss–Seidel** sweep to precondition an outer Krylov solve, following Alkmim et al. (IJNME 2026). Per solve it equilibrates the system (symmetric diagonal scaling, to tame the large dynamic range coupled multi-field systems typically have), splits it into field blocks, and preconditions the outer solve with the block sweep.

The block structure — which DOFs belong to which field, and each field's dimension — is **discovered automatically** from the ``DofManager`` and the live ``FEModel``, both handed to the solver by the nonlinear solver via ``setModel()`` whenever the equation system is (re)built (the first solve, and again after any adaptive-mesh-refinement or connectivity change). Nothing about the block layout, node coordinates, or mesh topology needed below is specified by hand. A ``linsolverConfigFile`` is therefore optional and carries only solver knobs. Requires the optional ``amgcl`` extension.

.. code-block:: edelweiss

    *solver, solver=NIST, name=theSolver
    linsolver=blockamg
    linsolverConfigFile=blockamg.json

.. code-block:: json

    {
        "outerSolver": "amgcl_lgmres",
        "sweeps": 1,
        "symmetric": true
    }

Near-null-space and the Chebyshev smoother
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Each field's AMG hierarchy uses **smoothed aggregation**, coarsened with a **Chebyshev polynomial smoother**. Both pieces need to be understood together, because they play deliberately complementary roles:

- The smoother damps error components whose eigenvalues lie in an estimated window ``[lower, higher] * rho`` of the operator's spectral radius ``rho`` (by default ``lower = 0.01``, i.e. eigenvalues below 1% of the spectral radius are, by design, **not** targeted by the smoother at all).
- Everything below that window — including the operator's *near-null-space*, i.e. directions in which the operator has almost no stiffness — is left for the coarse-grid correction to remove, via smoothed aggregation's own null-space-aware construction of the prolongation operator.

For 3D linear elasticity, the physical near-null-space is the 6-dimensional space of **rigid-body motions**: 3 translations plus 3 infinitesimal rotations, each of which costs zero elastic strain energy and therefore corresponds to a genuinely (near-)zero eigenvalue of the discretized stiffness operator. Standard AMG practice for elasticity is to hand smoothed aggregation the *full* 6-mode basis (translations **and** rotations) as the near-null-space, computed from each vector field's node coordinates about its own coordinate centroid (translations: unit displacement per component; rotations: the classic infinitesimal-rigid-rotation fields, e.g. about the z-axis, ``(-y, x, 0)``). A scalar field (e.g. ``nonlocal damage``) has no rotational rigid-body mode; its near-null-space is just the constant field.

Giving smoothed aggregation only the translations (a common simplification, since building the rotational modes needs node coordinates rather than just DOF-block structure) leaves the coarse grid unable to represent rotational rigid-body error exactly — and because the Chebyshev smoother is, by the construction above, *not* targeting that error class either, it has nowhere efficient to go, and outer-iteration counts suffer measurably (on the order of 30% more outer iterations on real coupled systems, isolated per field). The two AMG components genuinely need to agree on what "smooth error" means; a near-null-space that is missing part of the operator's true kernel is a gap neither component covers.

The Chebyshev smoother's own configuration has a second, independent subtlety: it estimates the operator's spectral radius via a fixed number of power-iteration steps rather than an exact eigenvalue computation. A short power-iteration budget can converge to a materially different (and sometimes badly under-converged) estimate depending on how many parallel worker threads are used to build the hierarchy — because a parallel implementation typically partitions the power-iteration's random starting vector across worker threads and seeds each thread's random-number generator independently, so the number of independent streams (and hence the effective starting vector) changes with thread count even though the true operator, and its true spectral radius, does not. A badly under-converged estimate degrades the smoother's own effectiveness and can defeat an otherwise-healthy near-null-space just as thoroughly as an incomplete one. Running the power iteration to convergence (i.e. increasing its iteration budget well past AMGCL's own low default) removes the thread-count sensitivity and is measurably worth the modest extra one-time hierarchy-build cost.

.. code-block:: edelweiss

    *solver, solver=NIST, name=theSolver
    linsolver=blockamg
    linsolverConfigFile=blockamg.json

Recognised keys, all optional:

.. list-table:: ``blockamg`` configuration keys
    :width: 100%
    :widths: 20 10 45
    :header-rows: 1

    * - Key
      - Default
      - Meaning
    * - ``outerSolver``
      - ``"amgcl_lgmres"``
      - The outer Krylov solve. ``"amgcl_lgmres"`` uses AMGCL's own native, OpenMP-threaded Loose GMRES implementation, which scales better across NUMA nodes than orchestrating the same iteration from Python; ``"scipy"`` uses SciPy's ``gmres`` instead, kept as a fallback.
    * - ``sweeps`` / ``symmetric``
      - ``1`` / ``true``
      - Number of block Gauss–Seidel sweeps per preconditioner application, and whether the sweep pattern is made symmetric (forward then backward), which keeps the preconditioner compatible with a symmetric outer Krylov method.
    * - ``useRigidBodyNullspace``
      - ``true``
      - Use the full 6-mode rigid-body near-null-space (translations and rotations, see above) for every vector field whose node coordinates are available; automatically falls back to translations-only for a field whose coordinates cannot be determined (e.g. a solver driven directly through the low-level ``setFieldStructure`` API instead of ``setModel``). Set to ``false`` to force translations-only unconditionally.
    * - ``fieldPreconds``
      - ``{}``
      - A mapping of field name (e.g. ``"displacement"``) to an AMGCL parameter tree overriding the dimension-based default for that field, including its own ``relax.power_iters``/``relax.lower``/``relax.higher`` Chebyshev settings.
    * - ``p1FieldNames``
      - ``[]``
      - **Experimental, opt-in only, not recommended as a default** — see "p-multigrid" below.
    * - ``dumpOnDegradationDir`` / ``dumpOnDegradationThreshold`` / ``dumpOnDegradationMaxDumps`` / ``dumpOnDegradationContextSolves``
      - ``None`` (off)
      - Diagnostic capture: when set, write the raw ``(A, b)`` system, its field-block layout, and (optionally) a rolling window of the preceding solves' own state, to disk for any solve whose outer-iteration count exceeds ``dumpOnDegradationThreshold`` — up to a process-wide cap of ``dumpOnDegradationMaxDumps`` — so a pathological live-run system can be captured for offline diagnosis without knowing in advance which solve will misbehave. Off by default; negligible cost when unused.
    * - ``etaMin`` / ``etaMax`` / ``ewGamma`` / ``ewAlpha``
      - ``1e-6`` / ``3e-4`` / ``0.9`` / ``1.618033988749895``
      - The same Eisenstat–Walker forcing-tolerance scheme, applied to the outer Krylov solve's own stopping tolerance.
    * - ``gapCompensatedTolerance`` / ``gapSafetyFactor``
      - ``false`` / ``0.3``
      - **Opt-in** — compensate the outer solve's stopping tolerance for the gap between the *scaled* residual the Krylov solver actually minimises and the *true* residual it is judged on, so most solves converge in a single pass instead of needing a warm-restart continuation. See "Gap-compensated tolerance" below, including why it is not the default.
    * - ``verbosity``
      - ``"warning"``
      - Log level (``"debug"``/``"info"``/``"warning"``/``"error"``) for this solver's own diagnostic output.

Gap-compensated tolerance (opt-in)
""""""""""""""""""""""""""""""""""

``blockamg`` equilibrates the system before solving, so the outer Krylov method minimises the
*scaled* residual :math:`\|D(b - Ax)\|`, while the stopping criterion is checked on the *true*
relative residual :math:`\|b - Ax\| / \|b\|`. The two differ by a factor — the *gap* — that grows
with the conditioning of the equilibration. When the true-residual check fails, the solve is
re-run as a warm restart at a tighter tolerance (a *continuation*).

In practice almost every solve needs one: on a reference gradient-enhanced damage model, 4467 of
4499 solves ran a continuation, i.e. effectively every linear solve was performed twice — and the
continuation is disproportionately expensive because restarting discards most of the accumulated
Krylov subspace.

With ``gapCompensatedTolerance = true`` the gap is measured, smoothed across solves, and used to
pre-compensate the *first* pass (asking it for ``gapSafetyFactor * eta / gap``); any continuation
that is still required then tightens by the measured gap rather than by a fixed factor.

.. note::
   **Why this is opt-in and not the default.** Measured on a 12-increment window of a reference
   model, enabling it cut continuations from 126 to 1 and outer iterations by 20% per increment.
   However, the default path happens to over-solve every system by roughly 30x, and the Newton
   iteration relies on that: converging merely inside the requested tolerance costs about one extra
   Newton iteration per increment, which is enough to keep the run above the adaptive time
   stepper's ``criticalIter`` threshold and so prevent it from growing the time increment. In the
   same window the default configuration grew its increment and covered 6.3% more simulated time.
   Normalised per unit of *simulated time* — the metric that matters — the net gain was 15% fewer
   iterations and 6.8% less wall-clock, roughly half the per-increment headline. ``gapSafetyFactor``
   is the knob between the two regimes (smaller values solve more accurately, closer to the
   default's Newton behaviour); its optimum has not been explored.

.. note::
   ``blockamg`` is the O(n)-memory route to the 1M+-DOF regime a direct factorization cannot reach, not necessarily the fastest solver at moderate problem sizes — a direct factorization (e.g. ``pardiso``) can still be competitive, or faster, on systems that comfortably fit its fill-in. Which is faster depends on problem size, conditioning, and how severely damage/contact nonlinearity degrades the per-field AMG hierarchies' convergence on a given increment.

p-multigrid (experimental, opt-in)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For a field discretized with quadratic (serendipity) elements, ``blockamg`` can optionally replace a field's single-level AMG hierarchy with a genuine **two-grid V-cycle**: a coarse level built purely from the mesh's corner nodes (a topological ``P1`` restriction/prolongation, requiring no re-discretization — corner values pass through unchanged, midside values are the average of their two edge-endpoint corners), with its own small AMGCL solve, sandwiched between Chebyshev pre- and post-smoothing sweeps on the full quadratic mesh.

The underlying two-grid algorithm is sound and does reduce the outer-iteration count on real coupled systems. However, it has two structural costs the single-level default does not pay, which can outweigh the iteration-count win depending on how AMG-friendly the operator otherwise is: a fixed per-solve setup cost (projecting the operator onto the coarse corner-node space via a sparse matrix triple product, plus building the coarse hierarchy itself), and a near-null-space handling gap — the coarse solve can be given the same rigid-body near-null-space treatment described above (restricted to the corner-node subset), but the fine-level Chebyshev smoother, unlike a full recursive AMG hierarchy, has no coarsening step of its own and is not given any near-null-space information at all. On at least one real reference model this made the two-grid variant measurably slower overall than the single-level default, on the range of operators tested (well short of severe convergence degradation). Whether it is worthwhile depends heavily on how badly a given field's single-level AMG hierarchy is already struggling — it is not recommended as a default, and is offered as an opt-in tool for a user who has independently confirmed it helps their own model.

Enable it per field via ``p1FieldNames`` (a list of field names to enable it for; the field's mesh must consist entirely of quadratic serendipity elements, or the solver falls back to the single-level default for that field with a warning).


The ``amgcl`` solver
--------------------

``amgcl`` is an iterative solver (Krylov method plus algebraic-multigrid or single-level preconditioner) built on the `AMGCL <https://github.com/ddemidov/amgcl>`_ library. Its ``linsolverConfigFile`` is forwarded as an AMGCL parameter tree; note that AMGCL silently ignores unknown parameter keys (warning only on stderr), so check its stderr if a configuration behaves unexpectedly.


The ``matrixdump`` diagnostic solver
------------------------------------

``matrixdump`` is not a solver but a diagnostic wrapper: it writes the equation systems it is handed to disk and then delegates the actual solve to a real linear solver, so a sequence of authentic ``(A, b)`` pairs can be replayed offline instead of by rerunning the simulation. Its ``linsolverConfigFile`` selects the ``delegate`` solver, the dump ``directory``, and which solves to capture (``dumpAt`` / ``skipFirst`` / ``maxDumps`` / ``instances``).
