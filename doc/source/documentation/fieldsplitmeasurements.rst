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

Power iterations of the Chebyshev smoother
------------------------------------------

See :ref:`blockamg-chebyshev`.

With the former setting of 50 power iterations, one captured solve needed far more outer
iterations than the identical matrix solved again with nothing differing but ``OMP_NUM_THREADS``
(the true spectral radius, checked independently, barely moves between Newton iterations there) --
a sign of under-convergence at some thread counts. 300 iterations removed that sensitivity and were
faster in aggregate over a set of captured degraded systems, with no observed downside on systems
that were already fine.

.. _measured-block-translations:

The translations of a block backend
-----------------------------------

See :ref:`blockamg-block-nullspace`.

Displacement block alone, on captured systems of the damage cylinder: AMGCL's constant blocks of
:math:`\hat A_{uu}` leave row residuals far from exact, against near-exact residuals for the translations
:math:`D^{-1} t`.

Full system, offline, on the captured sequences and on isolated systems of both reference models, the
configuration with and without ``blockTranslationNullspace``, outer iterations only: clearly
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
continuations and the outer iterations needed, with no cutbacks or material return-mapping
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
  (``blockTranslationNullspace``), instead of AMGCL's identity blocks, which are :math:`D`-weighted
  translations (see :ref:`the near-null-space of a block backend <blockamg-block-nullspace>`). Measured
  later than the rest, live on more models than these two, with and without them: fewer outer iterations
  and fewer solves stopping at the iteration cap on most models, with the same physical results. The
  exceptions are the damage cylinder after its damage onset, with more cap hits at the same outer iterations
  per simulated time, and the contact model, with more outer iterations at fewer Newton iterations; both
  ran no slower. The translations are the near-null-space of the elastic operator, and help less on a
  softening, indefinite one.
* **The rigid-body coarse level** for the 3D displacement field (``rigidBodyCoarseLevel``), which holds the rotations
  the translations cannot: offline, on captured systems of three models and of the testfile deck, never more outer
  iterations than the hierarchy on the translations and up to three times fewer on a bending-dominated one, at a setup
  of up to 1.2 times; live, faster on all three models measured, with the same results (see
  :ref:`blockamg-rigid-body-coarse-level`).
* ``LGMRES(restart=50)`` matches the outer-iteration cap (``maxIterations = 100``), so
  LGMRES restarts at most once within it; ``augmentationVectors`` only allocates its augmentation vectors without
  ever exhausting them before the cap.
* ``RebuildPolicy(stalenessFactor=3)`` reuses hierarchies longer than the former factor of 1.5, which fired on
  noise at small iteration counts; with the cap of 100 it can fire only while the solve the
  hierarchies were built for took fewer than about 34 iterations -- beyond that, only a cap hit (or a
  new increment, a pattern change) rebuilds them.
* ``ForcingTolerance(maximum=1e-3)`` is not about the linear solver at all: through a damage onset, Newton needs
  linear solves about ten times tighter than the Eisenstat--Walker forcing allowed at a maximum of 1e-2.
  ``minimum = 1e-4`` was measured against ``1e-3``: the looser value cost Newton extra
  iterations already before the damage onset, and more wall-clock to match.
* ``krylovOnScaledSystem = false`` converges the true residual directly, so the gap compensation and
  the continuations of the scaled system have nothing to do and are left at their own, otherwise
  inert, defaults.
* ``pruneCondensedMatrixZeros`` (a nonlinear-solver option, not a component's) now defaults to
  "off for a solver that needs the pattern stable", and a solver with a Krylov solve reports exactly that -- so on a
  model with multi-point constraints the hierarchies are now reused automatically; the former setup
  needed this set by hand (see the reuse rules on :doc:`blockamgtheory`).

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
