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
thread. With 50 iterations one captured solve needed 1460 outer iterations at 16 threads and 43 at 4, nothing else
differing; the true spectral radius (ARPACK) barely moved between the solves of that run. 300 iterations removed the
sensitivity, at 1.5 to 4 s of setup per hierarchy on the systems below.

**The estimate against ARPACK.** Arnoldi and Lanczos from the same row-seeded start vector, on the scaled
displacement block :math:`D K D` and the re-scaled coarse operator of the corner-node interpolation of seven captured systems (asymmetry
:math:`\lVert A - A^T \rVert_F / \lVert A \rVert_F` of the free rows up to 0.21), agree to three digits: Arnoldi is
kept, since it holds for nonsymmetric matrices. Its estimate :math:`\theta + r`, relative to the spectral radius:

.. list-table::
    :width: 100%
    :header-rows: 1

    * - steps
      - 5
      - 8
      - 10
      - 15
      - 20
      - adaptive, tolerance 0.05
    * - :math:`(\theta + r) / \rho`, 14 operators
      - 0.92 -- 1.05
      - 1.00 -- 1.04
      - 1.005 -- 1.043
      - 1.001 -- 1.015
      - 0.993 -- 1.013
      - 1.003 -- 1.039, after 8 -- 11 steps

The adaptive estimate stops when :math:`r < 0.05\, \theta`, after at least 8 steps; it took 9.1 steps on average
here, a tolerance of 0.03 11.4 and of 0.02 14.2, without a better estimate; 0.1 stopped after 6 -- 8 steps, once at
:math:`0.975\,\rho`. The minimum of 8 steps: after 5, :math:`\theta + r` was down to :math:`0.92\,\rho` above, and
on the small coarse levels of a 2D beam of the test suite (650 unknowns) the rigid-body coarse level then diverged (301
GMRES iterations against 6 with 8 or more steps).

**Outer iterations, setup and wall time** (s, mean over 8, 16 and 32 threads, xeon) to 1e-6 on the displacement
block alone, the 14 systems of :ref:`measured-scaled-corner-node-interpolation` and two post-onset cylinder solves (21, 27), for the default
AMG tree (``AMG()``), the rigid-body coarse level and the scaled corner-node interpolation (with the smoother of the time, degree
16 on both levels, :ref:`measured-corner-node-two-level-smoother`); the sums over the systems:

.. list-table::
    :width: 100%
    :header-rows: 1

    * - estimate
      - ``AMG()``: its / setup / wall
      - ``RigidBodyModesPerAggregate``: its / setup / wall
      - ``CornerNodeInterpolation``: its / setup / wall
      - largest spread of the its over the thread counts
    * - power iteration, 300 (the former default)
      - 596 / 73.8 / 202
      - 455 / 82.7 / 177
      - 284 / 79.9 / 138
      - 1
    * - power iteration, 50
      - 593 / 25.3 / 154
      - 452 / 33.5 / 130
      - 284 / 28.1 / 87
      - 1
    * - power iteration, 50, start vector seeded per row
      - 591 / 24.6 / 152
      - 452 / 32.9 / 127
      - 285 / 28.2 / 87
      - 0
    * - Arnoldi, 10 steps (the default)
      - 600 / 16.4 / 144
      - 458 / 22.9 / 117
      - 286 / 20.0 / 79
      - 0
    * - Arnoldi, adaptive, tolerance 0.05, 8 to 30 steps
      - 600 / 19.4 / 151
      - 458 / 23.0 / 118
      - 286 / 19.9 / 80
      - 0

Against the former default, 10 Arnoldi steps need 0.7 % more outer iterations (the estimate :math:`\theta + r` lies slightly above the
radius, which shifts the smoother's interval upwards), 22 -- 28 % of the setup, 29 -- 43 % less wall time, and
show no thread-count dependence at all -- the estimate is the same, bit for bit, at any thread count. The adaptive
estimate needs the same outer iterations, but more setup with ``AMG()``, whose small coarse levels take more steps to
the tolerance: so 10 fixed steps are the default, and the adaptive estimate is an option, for an operator whose
estimate converges more slowly, or where the guarantee for symmetric operators matters. Not measured: the 10 AnchorPryOut systems behind the 300 iterations (the dumps no
longer exist). Scratch scripts and logs: xeon ``~/scratch_sr_2026-10-27`` (``sr_bench.py``, ``arnoldi_bench.py``,
``krylov_quality.py``, ``krylov_traj.py``).

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

Outer iterations and wall time in seconds (setup included) to a relative residual of 1e-6 on the displacement block
alone, one cycle per outer iteration, LGMRES (restart 50), 32 threads on a two-socket workstation, from the systems of
:ref:`measured-rigid-body-coarse-level`: the rigid-body coarse level (``TwoLevel(coarseSpace=RigidBodyModesPerAggregate())``); the
former corner-node two-level cycle with the plain weights :math:`\tfrac12, \tfrac12` and an AMG coarse solve fused into its C++
cycle (measured on the same machine with the code before the change); ``TwoLevel(coarseSpace=CornerNodeInterpolation())`` with its default coarse solve,
``DiagonalScaling(TwoLevel(coarseSpace=RigidBodyModesPerAggregate()))``, every fine spectral radius from 50 power iterations
(as the first two columns); and the same with the default trees of the time (300 power iterations; the estimate is
now :ref:`measured-power-iterations`).

.. list-table::
    :width: 100%
    :header-rows: 1

    * - system (unknowns)
      - rigid-body coarse level
      - former corner-node two-level cycle
      - scaled weights, 50 power iterations
      - scaled weights, default trees
    * - tunnel lining, elastic (1.37 M)
      - 22 / 9.2
      - 279 / 72.7
      - 14 / 6.5
      - 14 / 10.4
    * - tunnel lining, damaged (1.37 M)
      - 36 / 13.9
      - 253 / 63.5
      - 21 / 8.7
      - 21 / 12.9
    * - damage cylinder, solve 2 (0.63 M)
      - 23 / 3.9
      - 84 / 10.8
      - 10 / 2.1
      - 10 / 3.7
    * - damage cylinder, solve 16 (0.63 M)
      - 28 / 4.0
      - 95 / 11.3
      - 13 / 2.5
      - 13 / 4.1
    * - damage cylinder, solve 22 (0.63 M)
      - 36 / 5.1
      - 103 / 12.3
      - 24 / 3.2
      - 24 / 4.9
    * - damage cylinder, solve 26 (0.63 M)
      - 58 / 6.7
      - 128 / 13.8
      - 39 / 4.2
      - 39 / 5.7
    * - damage cylinder, solve 30 (0.63 M)
      - 60 / 7.3
      - 139 / 15.5
      - 40 / 4.5
      - 40 / 5.9
    * - edge breakout, solve 1 (0.81 M)
      - 13 / 3.6
      - 21 / 8.6
      - 9 / 3.7
      - 9 / 6.1
    * - edge breakout, solve 8 (0.81 M)
      - 20 / 4.8
      - 24 / 9.1
      - 13 / 4.3
      - 13 / 6.5
    * - edge breakout, solve 13 (0.81 M)
      - 21 / 4.8
      - 25 / 9.3
      - 13 / 4.3
      - 13 / 6.8
    * - edge breakout, solve 20 (0.81 M)
      - 22 / 4.9
      - 25 / 9.0
      - 13 / 4.2
      - 13 / 6.6
    * - edge breakout, solve 26 (0.81 M)
      - 24 / 5.0
      - 29 / 9.6
      - 14 / 4.5
      - 14 / 6.8

The iterations are those of the offline prototype the method was developed in, on every system; the wall times are
lower than the prototype's (its setup ran partly in single-threaded SciPy). The coarse component costs 10--45 ms per
application (the fine Chebyshev sweep 50--160 ms), and calling it back from the C++ cycle, with the restriction and the
prolongation, adds 1--3 ms. The 300 power iterations of the default trees cost 1.5--4 s of setup on these systems, on
the fine level of the cycle above all.

Other coarse solves, with 50 power iterations (iterations / wall s):

.. list-table::
    :width: 100%
    :header-rows: 1

    * - system
      - default: ``DiagonalScaling(TwoLevel(RigidBodyModesPerAggregate()))``
      - ``TwoLevel(RigidBodyModesPerAggregate())``, :math:`A_c` not scaled again
      - ``DiagonalScaling(AMG(backendBlockSize=1))``, the rigid-body modes of the corners
    * - tunnel lining, elastic
      - 14 / 6.5
      - 14 / 6.9
      - 18 / 8.8
    * - damage cylinder, solve 22
      - 24 / 3.2
      - 24 / 3.7
      - 24 / 4.1
    * - edge breakout, solve 1
      - 9 / 3.7
      - 18 / 5.3
      - 10 / 6.0
    * - edge breakout, solve 20
      - 13 / 4.2
      - 27 / 6.5
      - 14 / 6.9

Where the scaling varies little along the edges (the tunnel, the cylinder) the second scaling of :math:`A_c` changes
nothing; on the edge breakout, where midside diagonals are up to :math:`10^6` times smaller than their corners', it
halves the iterations.

These measurements were taken with the smoother of the time, degree 16 on both levels of the corner-node two-level cycle; the default
is now degree 8 (:ref:`measured-corner-node-two-level-smoother`).

.. _measured-rebuild-every-solve:

Rebuilding the preconditioner for every solve
---------------------------------------------

See :ref:`blockamg-per-field-amg` and the reuse rules on :doc:`blockamgtheory`.

Live, one model (the contact anchor model with many multi-point-constraint slaves, edge breakout, increments 230 to
255), two runs from the same checkpoint on two nodes of a cluster: the reuse rules of ``RebuildPolicy``, against
``reusePreconditioner=False``, a new preconditioner for every solve. The paths split at increment 233 (other hierarchies, other
linear solutions within the tolerance).

.. list-table::
    :width: 100%
    :header-rows: 1

    * -
      - reuse rules
      - a new preconditioner for every solve
    * - wall time of the window
      - 9,697 s
      - 8,579 s (-11.5 %)
    * - attempts / Newton iterations / simulated time reached
      - 34 / 165 / 0.378483
      - 36 / 169 / 0.378633
    * - linear solves, per solve
      - 14.4 s
      - 9.8 s (-25 %)
    * - outer Krylov solve, per solve
      - 9.25 s
      - 2.73 s
    * - outer iterations per solve
      - mean 21.5, median 15, 11 at the cap
      - mean 7.6, median 6, max 18, none at the cap
    * - preconditioner builds
      - 138, 9.38 s each
      - 301, 6.71 s each

The second node was 5-10 % faster on the rows no preconditioner touches, so part of the cheaper build is the node; the
Krylov solve, 70 % cheaper per solve, is not. A reused preconditioner needed about three times the outer iterations of
a fresh one here, while offline replays of captured solves had reused ones at only 4-6 more iterations and reuse 16-46 %
faster: they did not reproduce how far a live Jacobian moves between solves. Not yet measured live on a second model.

.. _measured-corner-node-two-level-smoother:

The smoother of the corner-node two-level cycle
-----------------------------------------------

See :ref:`blockamg-p-multigrid` and :ref:`composed-default-field-trees`.

The Chebyshev smoother -- its degree, the lower end of its interval relative to the spectral radius, and the cycle,
V(0,1) or V(1,1) -- was swept for the three displacement preconditioners, ``AMG()``,
``TwoLevel(coarseSpace=RigidBodyModesPerAggregate())`` and ``TwoLevel(coarseSpace=CornerNodeInterpolation())``, offline on the
displacement block alone: ``LGMRES(DiagonalScaling(.))`` to a true relative residual of 1e-6, setup included, 32
threads on a two-socket workstation (xeon), first on three systems (the damaged tunnel lining, solve 26 of the damage
cylinder, solve 20 of the edge breakout), then on 14 systems: the 12 of :ref:`measured-scaled-corner-node-interpolation` and two post-onset
cylinder solves (21, 27). Every solve reached the tolerance. Run-to-run scatter of the wall time was up to about 12 % on the three systems.

Wall s / outer iterations on the three systems; the smoother of ``CornerNodeInterpolation`` on both of its levels (the fine
level and the rigid-body coarse level of its default coarse solve), the others on every level they smooth:

.. list-table::
    :width: 100%
    :header-rows: 1

    * - smoother
      - tunnel lining, damaged
      - damage cylinder, solve 26
      - edge breakout, solve 20
      - sum of the wall times
    * - ``AMG()``, degree 16, lower 0.01 (its default)
      - 16.24 / 63
      - 6.94 / 70
      - 4.35 / 23
      - 27.54 (repeated: 29.17)
    * - ``AMG()``, degree 8, lower 0.03
      - 13.66 / 85
      - 9.23 / 133
      - 4.87 / 42
      - 27.76
    * - rigid-body coarse level, degree 16, lower 0.01 (its default)
      - 13.07 / 37
      - 6.13 / 59
      - 4.27 / 22
      - 23.46 (repeated: 23.41)
    * - rigid-body coarse level, degree 8, lower 0.03
      - 10.37 / 47
      - 8.06 / 132
      - 4.40 / 40
      - 22.84
    * - corner-node two-level cycle, degree 16, lower 0.01 (the former default)
      - 8.02 / 21
      - 4.19 / 39
      - 4.04 / 13
      - 16.16
    * - corner-node two-level cycle, degree 8, lower 0.03 (the default)
      - 6.65 / 27
      - 3.67 / 52
      - 3.92 / 21
      - 14.23

* **For** ``AMG()`` **and the rigid-body coarse level, degree 16 is within scatter of the best setting.** The optimum is
  flat along a ridge: a wider interval needs a higher degree (degree 16 on 0.01 to 1.1, degree 8 on 0.03 to 1.1 and
  V(1,1) with degree 4 on 0.03 to 1.1 tie). A lower degree makes the tunnel about 20 % faster and the cylinder 25--35
  % slower. They keep degree 16.
* **For the corner-node two-level cycle, degree 16 is stronger than it pays.** The optimum is flat over degrees 6 to 10 and lower ends
  0.02 to 0.05; degree 8 on 0.03 to 1.1 on the fine level alone took 14.93 s, on both levels 14.23 s.
* A lower end of 0.1 lost everywhere, by 10--30 %; V(0,1) beat V(1,1) at the same number of polynomial applications
  for all three preconditioners.
* The iterations of the corner-node two-level cycle were the same at 8, 16 and 32 threads, and the gain larger at fewer threads: 34.74
  against 27.80 s at 8 threads (-20 %), 21.71 against 17.18 s at 16 (-21 %), 16.16 against 14.23 s at 32 (-12 %).

On the 14 systems (outer iterations / wall s, the mean of two repetitions; the default
``TwoLevel(coarseSpace=CornerNodeInterpolation())``, against the former default written out,
``TwoLevel(coarseSpace=CornerNodeInterpolation(DiagonalScaling(TwoLevel(coarseSpace=RigidBodyModesPerAggregate(), amgclTree=t))), amgclTree=t)``
with ``t = defaultFieldTree(3)``):

.. list-table::
    :width: 100%
    :header-rows: 1

    * - system
      - degree 16, lower 0.01 (former)
      - degree 8, lower 0.03 (default)
      - wall time
    * - tunnel lining, elastic
      - 14 / 5.95
      - 17 / 4.99
      - -16 %
    * - tunnel lining, damaged
      - 21 / 7.94
      - 27 / 6.71
      - -16 %
    * - damage cylinder, solve 2
      - 10 / 1.97
      - 12 / 1.73
      - -12 %
    * - damage cylinder, solve 16
      - 13 / 2.21
      - 18 / 2.03
      - -8 %
    * - damage cylinder, solve 21
      - 24 / 3.05
      - 33 / 2.81
      - -8 %
    * - damage cylinder, solve 22
      - 25 / 3.07
      - 33 / 2.87
      - -7 %
    * - damage cylinder, solve 26
      - 39 / 4.06
      - 52 / 3.75
      - -8 %
    * - damage cylinder, solve 27
      - 38 / 4.11
      - 51 / 3.84
      - -6 %
    * - damage cylinder, solve 30
      - 40 / 4.28
      - 53 / 3.74
      - -13 %
    * - edge breakout, solve 1
      - 9 / 3.54
      - 14 / 3.40
      - -4 %
    * - edge breakout, solve 8
      - 13 / 4.08
      - 22 / 4.03
      - -1 %
    * - edge breakout, solve 13
      - 13 / 4.02
      - 22 / 4.00
      - 0 %
    * - edge breakout, solve 20
      - 13 / 4.09
      - 21 / 3.96
      - -3 %
    * - edge breakout, solve 26
      - 14 / 4.24
      - 23 / 4.16
      - -2 %
    * - sum
      - 56.6
      - 52.0
      - -8 %

The default needs more outer iterations on every system, but fewer than twice as many, at half the matrix products per
smoothing sweep: faster or the same on all 14, by 16 % on the tunnel lining, 6--13 % on the cylinder; the edge
breakout, whose wall time is mostly the setup, ties. The sweep had shown -9.6 % on the same systems (59.14 and 56.43 s
against 52.25 s in two repetitions each). On these systems, the rigid-body coarse level alone with degree 8 only moved
time around (76.43 against 76.33 s): the tunnel lining about 20 % faster, the late cylinder solves 20--35 % slower at
2.2--2.5 times the iterations (solve 30: 61 against 155). The iterations of ``AMG()`` and of
``TwoLevel(coarseSpace=RigidBodyModesPerAggregate())`` with their defaults are unchanged by this default.

Not yet measured live: with Newton in the loop, and with the hierarchies reused over several solves.

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
  "off for a solver that needs the pattern stable", and a solver with a Krylov solve reports exactly that -- so on a
  model with multi-point constraints the hierarchies can be reused (``reusePreconditioner=True``) without more settings; the former
  setup needed this set by hand (see the reuse rules on :doc:`blockamgtheory`).

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
