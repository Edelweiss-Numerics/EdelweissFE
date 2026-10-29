Field-split AMG: measurements behind the defaults
=================================================

The defaults of the field-split preconditioner (:doc:`blockamgtheory`) -- those of the components and of the
``blockAMG()`` preset -- were chosen by measurement. This page collects those measurements, as the provenance of the
defaults; the theory page explains the method without them. Most were taken on two captured models, a gradient-enhanced
damage cylinder through its damage onset and a contact anchor model with many multi-point-constraint slaves, offline on
captured systems (dumped with ``dumps``, :ref:`composed-solve-dumps`) and live.

Many of them were taken before the composition existed, when the method was one monolithic solver; the composition
reproduces that solver's outer iterations, rebuild decisions and solutions, and replaced it.


.. _measured-rigid-body-nullspace:

The full rigid-body near-null-space
-----------------------------------

See :ref:`blockamg-chebyshev`.

On real captured systems the full rigid-body basis needed fewer isolated per-field
outer iterations than translations alone, and -- unlike translations alone -- was robust to both
thread count and the Chebyshev power-iteration setting. That was on the scalar backend; the block
backend's hierarchy takes the translations only (see ``backendBlockSize``).

.. _measured-power-iterations:

The spectral radius estimate of the Chebyshev smoother
------------------------------------------------------

See :ref:`blockamg-chebyshev`.

**Why it changed.** AMGCL estimated the spectral radius by power iteration from a start vector seeded per OpenMP
thread. With a short power iteration, one captured solve needed far more outer iterations at one thread count than at
another, nothing else differing, while the true spectral radius barely moved between the solves of that run. A long
power iteration removed that sensitivity, but made the estimate the most expensive stage of the setup.

**The estimate against a reference eigensolver.** On the scaled displacement blocks and the coarse operators of the
corner-node interpolation of captured systems, markedly nonsymmetric ones among them, Arnoldi and Lanczos from the same
row-seeded start vector agree to about three digits; Arnoldi is kept, since it holds for nonsymmetric matrices. With
the default number of steps, its estimate :math:`|\theta| + r` lay slightly above the spectral radius on every
operator, well within the smoother's margin; more steps bring it closer, and too few can leave it well below, where
the rigid-body coarse level of a small 2D beam of the test suite diverged.

**Outer iterations, setup and wall time**, on the displacement block alone of the captured systems of
:ref:`measured-scaled-corner-node-interpolation` and of the damage cylinder after its onset, for ``AMG()``, the
rigid-body coarse level and the scaled corner-node interpolation, at several thread counts: the default Arnoldi steps
needed about the outer iterations of the former long power iteration, for a fraction of its setup and clearly less wall
time, and their outer iterations no longer depend on the thread count at all. A start vector seeded per row alone
already removes the thread-count dependence of a power iteration. An adaptive number of steps, stopping once the
residual norm of the dominant Ritz pair is small against its Ritz value, needed the same outer iterations with more
setup on the small coarse AMG levels, so the estimate takes a fixed number of steps. The contact anchor systems the
former long power iteration was chosen on were not measured again.

.. _measured-block-translations:

The translations of a block backend
-----------------------------------

See :ref:`blockamg-block-nullspace`.

Displacement block alone, on captured systems of the damage cylinder: AMGCL's constant blocks of
:math:`\hat A_{uu}` leave row residuals far from exact, against near-exact residuals for the translations
:math:`D^{-1} t`.

Full system, offline, on the captured sequences and on isolated systems of both reference models, the
configuration with and without the translations as near-null-space (``nearNullspace``), outer iterations only: clearly
fewer on the damage cylinder, both in its onset sequence and on its isolated systems; on the contact
model about the same in its sequence, whose hierarchies were rebuilt more often, and fewer on its
isolated systems. A control with unit weights needed exactly the iterations of AMGCL's identity blocks,
system by system, on the cylinder's isolated systems. With ``"relax": {"scale": true}`` added, the
translations needed the same outer iterations as a hierarchy built on :math:`A_{uu}` itself, solve by
solve, on both sequences: the aggregates that differ between the two made no measurable difference. The
Jacobi-scaled smoother itself gave fewer outer iterations on the cylinder and more on the contact model,
where it also triggered more rebuilds; live, it made neither the L-shaped panel nor the damage cylinder
faster.

Live, the defaults with and without the translations, from the start of each run or over the same window,
judged per simulated time where the two runs took different load paths: on an L-shaped concrete panel
(with quadratic and with linear elements, and with adaptive refinement), a notched three-point bending
beam and an anchor pry-out, fewer outer iterations, fewer solves stopping at the iteration cap -- almost
none left on the panel and the beam -- and faster runs, with matching load-displacement curves where the
runs completed; on the damage cylinder fewer Newton iterations and a faster run. On the contact model
fewer Newton iterations but more outer iterations, level within the run-to-run noise. Hence the
default. The translations are the near-null-space of the elastic operator, so they help less once
softening makes the operator indefinite: on the damage cylinder after its damage onset, the outer
iterations per simulated time were the same with and without them, and more solves stopped at the cap
with them.

.. _measured-sweep-order:

The order of the block Gauss--Seidel sweep
------------------------------------------

See :ref:`blockamg-block-gauss-seidel`.

Damage first against displacement first, offline on the captured systems of the damage cylinder and
the contact model: identical outer iteration counts, and somewhat more time per outer iteration. On
these models the preconditioner's strength is its per-field AMG cycles, not the coupling.

.. _measured-gap-compensation:

Gap compensation
----------------

See :ref:`blockamg-gap-compensation`.

Gap compensation is a live option only on the scaled system. Per increment it
is a clear win there: on a live window of a reference model it substantially cut both the number of
true-residual restarts and the outer iterations needed, with no cutbacks or material return-mapping
failures. But the scaled system, left uncompensated, happens
to over-solve every system by a large factor, and the Newton iteration quietly relies on that:
converging merely inside the requested tolerance costs Newton extra iterations per
increment, which is enough to park the run above the adaptive time stepper's ``criticalIter``
threshold and so stop it from growing the time increment. In the same window the uncompensated
path grew its increment and covered more simulated time, while the compensated one never grew.
Judged per unit of *simulated time* rather than per increment, the per-increment saving is largely
or wholly given back. Hence this stays opt-in, for a model still run with
``krylovOnScaledSystem = true``; ``safetyFactor`` is the knob between the two regimes there --
smaller values solve more accurately and behave more like the uncompensated path -- and its
optimum has not been explored.

.. _measured-rigid-body-coarse-level:

The rigid-body coarse level
---------------------------

See :ref:`blockamg-rigid-body-coarse-level`.

Outer iterations of the field-split preconditioner to a relative residual of 1e-6 on the displacement block alone (one cycle per
outer iteration, the hierarchies built for the system): AMGCL's hierarchy on the translations (``false``), the
rigid-body coarse level (the default), and its control with the translations alone -- the same construction on the
translations, :math:`M = B`. The first system is the elastic stiffness of the testfile deck
``testfiles/edelweiss-only/CantileverBeamHexa20BlockAMGRigidBody``, a slender cantilever of quadratic hexahedra; the
others are captured systems of larger models.

.. list-table::
    :width: 100%
    :header-rows: 1

    * - system (unknowns)
      - translations
      - rigid-body coarse level
      - control
    * - slender cantilever, the testfile deck (3663)
      - 307
      - 18
      - 306
    * - tunnel segment, elastic (1.37 M)
      - 67
      - 22
      - 69
    * - tunnel segment, damaged (1.37 M)
      - 61
      - 36
      - 62
    * - damage cylinder, solve 16 (0.63 M)
      - 38
      - 28
      - 39
    * - damage cylinder, solve 26 (0.63 M)
      - 70
      - 59
      - 69
    * - edge breakout, solve 8 (0.81 M)
      - 22
      - 20
      - 22
    * - edge breakout, solve 20 (0.81 M)
      - 23
      - 22
      - 23

The control needs the iterations of the hierarchy on the translations within two: the gain is the rotations'. On
the larger models the setup took up to 1.2 times that of the hierarchy on the translations, and the solve, setup
included, 0.61 of its time on the elastic tunnel system and 0.84 to 0.95 on the others. In a simulation, the
testfile deck run with ``verbosity=info`` shows the same: each linear solve reaches its tolerance in some 20
outer iterations, and Newton converges in two; with the coarse level switched off, every solve stops at the
iteration cap with a relative residual of about 1e-2, and Newton needs five.

Live, from the start of each run on the same workstation, one run at a time and each capped at 10 minutes, the
hierarchy on the translations against the rigid-body coarse level for the displacement field:

* the L-shaped concrete panel (``examples/WinklerL``, quadratic elements, damage): with the rigid-body coarse level
  the run completed in 581 s, while the translations had covered 88% of the load path when stopped at 600 s;
  24.7 against 28.3 outer iterations per solve, at most 61 against one solve at the cap of 100. The
  load-displacement curves agree to 2e-4 of the peak load.
* a coarser tunnel segment with embedded bars with bond-slip (0.28 M unknowns, ``SchurComplement``, the rigid-body
  coarse level on the displacement field of the remaining nodes), run to its end: 46 against 65 s of computation,
  13 against 21 Newton iterations and linear solves, and 9.1 against 14.7 outer iterations per solve. The final
  load agrees to 2e-4.
* the damage cylinder (0.84 M unknowns, implicit dynamics): simulated time 0.215 against 0.185 in the same time,
  and 7.9 against 12.6 outer iterations per solve. The load agrees to 3e-4 at the same simulated times.

.. _measured-scaled-corner-node-interpolation:

The corner-node two-level cycle with scaled weights and a composed coarse solve
-------------------------------------------------------------------------------

See :ref:`blockamg-p-multigrid`.

Offline, outer iterations and solve time (setup included) to a relative residual of 1e-6 on the displacement block
alone, one cycle per outer iteration, on the captured systems of :ref:`measured-rigid-body-coarse-level`: the tunnel
segment, elastic and damaged, and sequences of the damage cylinder and of the edge breakout.

Against the former corner-node two-level cycle -- the plain weights :math:`\tfrac12, \tfrac12` and an AMG coarse solve
inside its C++ cycle -- the scaled weights with the coarse solve
``DiagonalScaling(TwoLevel(coarseSpace=RigidBodyModesPerAggregate()))`` needed many times fewer outer iterations on the
tunnel segment, several times fewer on the damage cylinder and fewer on the edge breakout, and less time on every
system. They also needed fewer outer iterations than the rigid-body coarse level on the whole field on every system;
with the fine spectral radius estimated by fewer power iterations than the default trees of the time, they were also
faster on almost every system, while those long power iterations, on the fine level of the cycle above all, took back
much of that gain in setup (the estimate that replaced them: :ref:`measured-power-iterations`). The iterations agree
with those of the offline prototype the method was developed in. Calling the coarse component back from the C++ cycle,
with the restriction and the prolongation, is a small part of an application next to the fine Chebyshev sweep.

Other coarse solves on the corner-node field: ``TwoLevel(coarseSpace=RigidBodyModesPerAggregate())`` without the
second scaling needed the same iterations where the scaling varies little along the edges (the tunnel, the cylinder),
and many more on the edge breakout, where midside diagonals are far smaller than their corners';
``DiagonalScaling(AMG(backendBlockSize=1))``, an AMG hierarchy with the rigid-body modes of the corners, needed as
many or more iterations and more time on every system.

These measurements were taken with the smoother of the time, degree 16 on both levels of the corner-node two-level
cycle; the default is now degree 8 (:ref:`measured-corner-node-two-level-smoother`).

.. _measured-rebuild-every-solve:

Rebuilding the preconditioner for every solve
---------------------------------------------

See :ref:`blockamg-per-field-amg` and the reuse rules on :doc:`blockamgtheory`.

Live, on the contact anchor model with many multi-point-constraint slaves, over a window of increments: two runs
from the same checkpoint, the reuse rules of ``RebuildPolicy`` against ``reusePreconditioner=False``, a new
preconditioner for every solve. The two runs took slightly different paths (other hierarchies, other linear solutions
within the tolerance), with about the same Newton iterations and simulated time reached.

A new preconditioner for every solve was faster over the window. A reused preconditioner needed several times the
outer iterations of a fresh one, and a share of its solves stopped at the iteration cap, against none with a new
preconditioner for every solve; the far cheaper Krylov solves outweighed the more frequent builds. Offline replays of
captured solves had shown the opposite, a reused preconditioner only a few iterations behind a fresh one and reuse
faster: they did not reproduce how far a live Jacobian moves between solves. Not yet measured live on a second model.

.. _measured-corner-node-two-level-smoother:

The smoother of the corner-node two-level cycle
-----------------------------------------------

See :ref:`blockamg-p-multigrid` and :ref:`composed-default-field-trees`.

The Chebyshev smoother -- its degree, the lower end of its interval relative to the spectral radius, and the cycle,
V(0,1) or V(1,1) -- was swept for the three displacement preconditioners, ``AMG()``,
``TwoLevel(coarseSpace=RigidBodyModesPerAggregate())`` and ``TwoLevel(coarseSpace=CornerNodeInterpolation())``,
offline on the displacement block alone: ``LGMRES(DiagonalScaling(.))`` to a true relative residual of 1e-6, setup
included, first on three systems (the damaged tunnel lining, a late solve of the damage cylinder and of the edge
breakout), then on the systems of :ref:`measured-scaled-corner-node-interpolation` and two post-onset cylinder solves.
Every solve reached the tolerance. The smoother of ``CornerNodeInterpolation`` was set on both of its levels (the
fine level and the rigid-body coarse level of its coarse solve), the others on every level they smooth.

* **For** ``AMG()`` **and the rigid-body coarse level, degree 16 is within the run-to-run scatter of the best
  setting.** The optimum is flat along a ridge: a wider interval needs a higher degree (degree 16 on 0.01 to 1.1,
  degree 8 on 0.03 to 1.1 and V(1,1) with degree 4 on 0.03 to 1.1 tie). A lower degree made the tunnel lining faster
  and the late cylinder solves slower, at more than twice the iterations, and moved no time overall. They keep
  degree 16.
* **For the corner-node two-level cycle, degree 16 is stronger than it pays.** The optimum is flat over degrees 6 to
  10 and lower ends 0.02 to 0.05; degree 8 on 0.03 to 1.1 was faster on both levels than on the fine level alone.
* A lower end of 0.1 lost everywhere; V(0,1) beat V(1,1) at the same number of polynomial applications for all three
  preconditioners.
* The outer iterations of the corner-node two-level cycle did not depend on the number of threads, and the gain was
  larger at fewer threads.

On all systems, the default ``TwoLevel(coarseSpace=CornerNodeInterpolation())`` (degree 8 on 0.03 to 1.1) against the
former default written out,
``TwoLevel(coarseSpace=CornerNodeInterpolation(DiagonalScaling(TwoLevel(coarseSpace=RigidBodyModesPerAggregate(),
amgclTree=t))), amgclTree=t)`` with ``t = defaultFieldTree(3)``, needed more outer iterations on every system, but
fewer than twice as many, at half the matrix products per smoothing sweep: it was faster or as fast on every system,
most on the tunnel lining and the damage cylinder; the edge breakout, whose time is mostly the setup, ties. The
iterations of ``AMG()`` and of ``TwoLevel(coarseSpace=RigidBodyModesPerAggregate())`` with their defaults are unchanged
by this default.

Not yet measured live: with Newton in the loop, and with the hierarchies reused over several solves.

.. _measured-two-level-written-out:

The two-level cycle written out
-------------------------------

See :ref:`composed-two-level-two-ways`.

The corner-node two-level cycle in C++, ``TwoLevel(coarseSpace=CornerNodeInterpolation(coarseSolve))``, against the
same cycle written out, ``Multiplicative([CoarseCorrection(cornerNodeInterpolation, coarseSolve), smoother])``: the same
:math:`P` (bit for bit), the same coarse solve, and as the smoother AMGCL's Chebyshev relaxation (degree 8, on the
scalar double-precision matrix: EdelweissFE has no Chebyshev component). Offline, on the displacement blocks of three
captured systems (a tunnel lining, a damage cylinder, an edge breakout), each inside ``LGMRES(DiagonalScaling(.))`` to
:math:`10^{-6}`, in repetitions.

Both forms need the same iterations on every system, and the written-out cycle takes markedly more wall time, in its
applications and in its set-up. Forming ``CoarseCorrection``'s products with :math:`P` and :math:`P^T A P` threaded,
instead of in SciPy, made each of them several times faster and the written-out cycle faster as a whole, most in the
set-up, but not as fast as the C++ cycle. In one application of the written-out cycle the smoother takes most of the
time; the residual of ``Multiplicative`` and the coarse correction take a small share each. The C++ cycle smooths on its
fine level stored in single-precision 3x3 node blocks, half the bytes per product; the same C++ cycle on
double-precision scalar entries (``backendBlockSize=1, backendPrecision="double"``) was slower per application on every
system. What remains of the difference is that smoother, and in the set-up the bench script's own search for the
Dirichlet rows in NumPy, on one thread.

.. _measured-defaults:

Why the defaults are what they are
----------------------------------

The defaults of the ``blockAMG()`` preset are not a separate "recommended configuration" layered
on top of different shipped defaults -- they *are* the code's defaults (those of
:func:`~edelweissfe.linsolve.composition.presets.blockAMG` and of its components). They are the winner of a systematic comparison, against the **former** defaults, on two
captured models (a gradient-enhanced damage cylinder through its damage onset, and a
contact anchor model with many multi-point-constraint slaves):

* **3x3 blocks stored in float** (set up in double, see ``backendPrecision``) cut the cost per outer
  iteration.
* **One high-degree post-smoothing sweep** (``npre = 0``, ``npost = 1``, Chebyshev degree 16, instead
  of a symmetric V(1,1) cycle at degree 5) and **smoothed aggregation's own spectral-radius estimate**
  for its prolongation smoother (``estimate_spectral_radius``, 20 power iterations) cut outer
  iterations for the same number of preconditioner applications.
* **The translations of the equilibrated block** as a block hierarchy's near-null-space
  (``nearNullspace``, the default on a block backend), instead of AMGCL's identity blocks, which are :math:`D`-weighted
  translations (see :ref:`the near-null-space of a block backend <blockamg-block-nullspace>`). Measured
  later than the rest, live on more models than these two, with and without them: fewer outer iterations
  and fewer solves stopping at the iteration cap on most models, with the same physical results. The
  exceptions are the damage cylinder after its damage onset, with more cap hits at the same outer iterations
  per simulated time, and the contact model, with more outer iterations at fewer Newton iterations; both
  ran no slower. The translations are the near-null-space of the elastic operator, and help less on a
  softening, indefinite one.
* **The rigid-body coarse level** for the 3D displacement field (``rigidBodyModesPerAggregate``), which holds the rotations
  the translations cannot: offline, on captured systems of three models and of the testfile deck, never more outer
  iterations than the hierarchy on the translations and up to three times fewer on a bending-dominated one, at a setup
  of up to 1.2 times; live, faster on all three models measured, with the same results (see
  :ref:`blockamg-rigid-body-coarse-level`).
* ``LGMRES(restart=50)`` matches the outer-iteration cap (``maxIterations = 100``), so
  LGMRES restarts at most once within it; ``augmentationVectors`` only allocates its augmentation vectors without
  ever exhausting them before the cap.
* ``RebuildPolicy(stalenessFactor=3)``, where hierarchies are reused (``reusePreconditioner=True``; since
  :ref:`measured-rebuild-every-solve` not the default), reuses them longer than the former factor of 1.5, which fired on
  noise at small iteration counts; with the cap of 100 it can fire only while the solve the
  hierarchies were built for took fewer than about 34 iterations -- beyond that, only a cap hit (or a
  new increment, a pattern change) rebuilds them.
* ``EisenstatWalkerTolerance(maximum=1e-3)`` is not about the linear solver at all: through a damage onset, Newton needs
  linear solves about ten times tighter than the Eisenstat--Walker forcing allowed at a maximum of 1e-2.
  ``minimum = 1e-4`` was measured against ``1e-3``: the looser value cost Newton extra
  iterations already before the damage onset, and more wall-clock to match.
* ``krylovOnScaledSystem = false`` converges the true residual directly, so the gap compensation and
  the true-residual restarts of the scaled system have nothing to do and are left at their own, otherwise
  inert, defaults.
* ``pruneCondensedMatrixZeros`` (a nonlinear-solver option, not a component's) now defaults to
  "off for a solver that needs the pattern stable", and a solver whose Krylov solve reuses its preconditioner reports
  exactly that -- so on a model with multi-point constraints the hierarchies can be reused
  (``reusePreconditioner=True``) without more settings; the former setup needed this set by hand (see the reuse rules
  on :doc:`blockamgtheory`).

Measured live, from the same checkpoints, the defaults against the former defaults (no per-model settings in
either run): the contact model's increments finished markedly faster; on the damage cylinder, the
former defaults had still not cleared the damage-onset increment in a comparable time budget, while
the current defaults carried the run through it and beyond.

How the settings got there
~~~~~~~~~~~~~~~~~~~~~~~~~~

How the settings got there, live against the production settings the reference runs had used: both
the contact model and the damage cylinder solved faster, judged per Newton iteration on the cylinder,
with a forcing maximum of 1e-2 and on the scaled system. On the
cylinder that setting took Newton more iterations per increment before the damage onset and failed
the onset increment outright (a cutback); a maximum of 1e-3 restores the fewer-iteration path,
converges the onset increment and reaches the same load, and is neutral on the contact model.
``krylovOnScaledSystem = false`` then saves a further amount on the cylinder, with fewer Newton
iterations, and on the contact model. Offline, on captured consecutive systems of both models, the
same configuration with a 200-iteration cap needed less wall-clock than with
the displacement blocks stored in double, at identical outer iteration counts.

All of the above, except the offline 200-iteration-cap comparison just above, was measured with the
outer-iteration cap at 50 (``maxIterations``). The default cap is 100 instead: more forgiving on
a model that has not yet been characterised on this solver, at the cost of a capped solve possibly
spending up to twice the iterations before its hierarchies are rebuilt.
