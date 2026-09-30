# NID performance: mass/damping assembly and per-increment cost (xeon, 2026-09-26/27)

Branch `fix/nid-mass-assembly-memory` (EdelweissFE, off origin/next_v26.11 f38b05de), pushed to `mn`, no PR.
Scratch on xeon: `~/nidmem/` (`phases.py` = per-phase wall/RSS instrumentation, `cootime.py`, `asym.py`, run dirs).

## Library
Phase 2 ran against a separate Marmot worktree `~/nidmem/marmot-src` = origin/next_v26.11 **20dc4ad5**, installed in
`~/nidmem/marmot-prefix`. The private modules came from the main xeon checkout, which has no nested repos:
GCDP and CDP copied in; GosfordSandstone/GMNeoHooke/GMDamagedShearNeoHooke dropped for missing deps.
The Cython `.so` files RPATH conda's `lib` ahead of `LD_LIBRARY_PATH`, so I patchelf'ed the three Marmot-linking
worktree `.so` files to `marmot-prefix/lib` first. patchelf was installed into env next_v2611 for this.
`/proc/self/maps` shows `~/nidmem/marmot-prefix/lib/libMarmot.so.1.0.0` loaded. The WIP Marmot checkout was not touched.
Phase-1 numbers ("baseline" below) used the old Sep-15 lib (bulk-viscosity branch).

## c1_50 s2s deck: 193,486 dof, VIJ 71.1 M, nnz(M) 40.9 M, 18.5k elements after the initial AMR
Setup: 6 increments, OMP=16, MKL_CBWR=AUTO,STRICT, ensight output off. The live-refinement threshold was lowered to
1e-3, but no live refinement fired in 6 increments. So the per-refinement cost is the step-start one:
after a live refinement exactly `_assembleMassAndDamping` + `_computeInitialAcceleration` run again
(the reuse path is for constraint-connectivity-only rebuilds).

| phase | NIST | NID before | NID after (3becd500) |
|---|---|---|---|
| total wall (7 increments incl. inc 0) | 284 s | 206 s | 192 s |
| RSS peak | 7.17 GiB | 10.37 GiB | 9.09 GiB |
| Newton iterations / linear solves | 34 | 27 | 27 |
| linear solve (blockamg setup+GMRES) | 207 s (6.1 s/solve) | 121 s (4.5 s/solve) | 119 s |
| mass assembly (per refinement) | - | 9.9 s, +2.77 GiB | 5.9 s, +1.45 GiB |
| initial acceleration (per refinement; blockamg solve on M) | - | 3.06 s, +1.58 GiB | 2.92 s |
| dynamicStiffness forming (initializeIncrement minus the two above), per increment | - | 0.43 s | 0.16 s |
| residual inertia/damping + K += dynamicStiffness (assembleAdditionalTerms), per iteration | 0 | 0.25 s | 0.15 s |
| Newmark predictor/corrector (_newmarkVelocityAndAcceleration), per iteration | - | 4 ms | 4 ms |
| reuse path | - | not triggered | not triggered |
| elements / constraints / CSR / Dirichlet | 7.1 / 11.1 / 3.9 / 3.3 s | 5.8 / 7.1 / 2.8 / 2.5 s | same |

The "after" run shared xeon with two OMP=1 bitwise runs, so its wall times are an upper bound.

**What NID adds over NIST:**
- Wall: about 13 s per refinement (mass 9.9 -> 5.9 s, initial acceleration 3 s), plus ~0.2 s per increment and ~0.15 s per iteration.
- Memory: +3.2 GiB RSS peak before the fixes, +1.9 GiB after. Retained per system before: Mvij 0.53, Cvij 0.53, M 0.46, C 0.46, dynamicStiffness 0.53 GiB.
- Net, on this deck NID is *faster* than NIST: its linear solves are cheaper (4.5 vs 6.1 s) and it needs fewer iterations (27 vs 34). The mass regularizes the tangent.
- Scaling to c1_100 (VIJ ~200 M, ~2.8x) gives about 36 s per refinement and 5.3 GiB extra RSS before the fixes. The c1_100 deck (on xeon under `~/nidmem/c1_100`) was NOT run.

## Symmetry check verdict
Removed from runtime, both the global and the per-element version.
- It is not harmless to skip entirely: an injected ±5 % antisymmetric mass still converges and silently gives a wrong
  tip displacement (-1.1e-4 vs 1.36e-6; `asym.py`).
- So it is now pinned per element type in `tests/test_nid_element_mass.py`. Types covered: C3D8, C3D20, C3D20R, GC3D8, GC3D20R.
- Checked per type: symmetry, PSD, total mass rho*V per displacement component, no cross-component/field coupling, and the textbook C3D8 matrix.

## Implemented (bitwise identical on NID, NIDParallel, NIDLiveAMR testfiles, MKL_CBWR=AUTO,STRICT, both libs)
1. 5fa1808d perf: in-place dynamicStiffness; finite-value check on the CSR data, not the VIJ vectors; per-element symmetry check.
2. e5ffdd72 refactor: removes the runtime symmetry check and adds the element mass tests. Also adds tests that the retained
   checks fire (non-finite mass, non-finite damping, negative damping, total-mass report; the massless-dof test already existed).
3. 3becd500 perf: damping kept as its nonzero VIJ diagonal entries (`dampingVIJIndices/Values`, `_collectLumpedDamping`).
   - Finding: every current Marmot element reports damping only on non-mechanical fields, which NID discards.
   - So Cvij and C were 1 GiB of zeros that were multiplied every iteration.
   - Effect: mass assembly 9.9 -> 5.9 s, RSS peak -1.3 GiB.

c1_50 at OMP=1, 3 increments (4 Newton iterations), e5ffdd72 vs 3becd500: RF_loading, U_loading, maxDamage **bitwise
identical** (`~/nidmem/bit_base` vs `bit_fix`). The 5fa1808d ops were verified on the NID testfiles only.

## Ranked open fixes (measured or strongly supported)
1. **Assemble M through the solver's existing `csrGenerator` instead of scipy COO->CSR**:
   - Measured 3.55 s -> 0.29 s on c1_50, same nnz; the rest of the assembly is the element loop plus masks (~2.4 s).
   - NOT bitwise: max |dM| = 1.4e-17 from a different summation order. It changes numbers at round-off, so it is flagged rather than landed.
   - Also makes M share K's pattern, so `K += dynamicStiffness` is already aligned.
2. **Initial acceleration after refinement** (2.9 s, +1.6 GiB, a full blockamg setup on M):
   - Option a: keep the interpolated acceleration (`computeInitialAcceleration=False` exists already).
   - Option b: a lumped-M solve (diagonal, ~0 s).
   - Either changes the numbers, so it needs an accuracy A/B (energy/momentum report right after a refinement). Not measured.
3. **dynamicStiffness VIJ vector** (0.53 GiB retained, per increment):
   - Could be dropped by adding `a*Mvij` into K in place each iteration (one fused multiply-add, no storage).
   - Or kept as CSR once 1. lands.
4. **M·a with a CSR copy**: M is already CSR and there is no copy per product; nothing to gain.
5. **Incremental M update after refinement**:
   - Only the refined elements' blocks change, but the VIJ layout is rebuilt.
   - The reuse path would need a slot map, old element to new element. Worth it only after 1.; the loop costs ~2.4 s of 5.9 s.
6. **Remaining Python full-array temporaries**: `isDynamic[I] & isDynamic[J]` (3 bool temporaries of 71 MB), `~couplesDynamicOnly`, the discarded mask. Minor.

## Not measured
- The c1_100 deck.
- py-spy flame graphs: ptrace_scope=1 blocks attach. I used method-level wall/RSS instrumentation instead.
- blockamg-internal copies.
- A live-refinement increment itself (expected equal to the step-start cost above).

## Phase 3 (2026-09-27): fixes 1-3, initial-acceleration A/B, c1_100 through live refinements

### Commits (each checked with pre-commit and `git show`)
- **8b2a86dd** perf: the mass matrix is summed into CSR form by the solver's own `csrGenerator` (K's pattern).
  - Scipy's COO->CSR step drops from 3.55 s to 0.29 s.
  - **Not bitwise, by design**: duplicates are summed in a different order.
  - Measured differences:
    - max |dM| = 1.4e-17 on c1_50.
    - NID and NIDParallel U: identical.
    - NIDLiveAMR: max |dU| = 1.8e-18, against |U| up to 1e-2.
    - c1_50 at OMP=1, 3 increments: RF max |d| = 1.6e-13 (rel. 2e-15), U max |d| = 1.6e-19, maxDamage identical.
  - All testfiles pass against their committed U.ref; none regenerated.
- **2d0cd3e3** perf: the stored `dynamicStiffness` VIJ vector is gone.
  - Each iteration does `K += massFactor * Mvij`, then adds `dampingFactor * damping` at its diagonal slots.
  - Saves 0.53 GiB (c1_50) or about 1.5 GiB (c1_100) of retained memory. The scaled mass is a transient temporary.
  - Bitwise identical on the NID testfiles.
  - The order of the two additions differs only on damping slots, and no current element has any undiscarded damping there.
- **0825f313** docs: new "Checks on the mass" section in the module docs (Sphinx automodule); the storage of M and C is documented; assembly comments tidied.
  - No option or default changed.

### Initial acceleration after a live refinement: A/B (not landed; default unchanged)
Fixture: c1_50 at OMP=6, 13 increments. The first live refinement comes at increment 11 (195,206 dof). The three variants
are identical before it. Harness: `~/nidmem/abvariant.py`; runs in `~/nidmem/ab_*`.

| variant | cost after refinement | KE, inc 11 / 12 | max\|A\|, inc 11 | RF at t = 0.055 / 0.060 | Newton iterations, incs 11-13 |
|---|---|---|---|---|---|
| consistent solve (current) | 14.3 s | 1.014 / 1.562 | 7170 | 1565.6 / 1807.3 | 6, 11, 10 |
| (a) keep interpolated | 0 s | 1.106 / 1.518 | 147 | 1474.9 (-5.8 %) / 1771.0 (-2.0 %) | 9, 10, 8 |
| (b) lumped (HRZ) solve | 1.3 s | 2.704 / 2.083 | 5.0e5 | 907 (-42 %) / 1505 (-17 %) | 10, 8, … |

NIDLiveAMR (C3D20R bar, 10 increments):
- (a) tracks the consistent solve to about 1e-7 in KE, but carries a constant acceleration offset (max|A| off by 0.07 of 1.8).
- (b) triples max|A| immediately and drifts KE by 0.5 %.

Verdict: neither variant is clearly equivalent.
- Lumped is wrong for the 20-node serendipity mass, because the HRZ diagonal is a poor inverse.
- Interpolated moves RF by 2-6 % in the first increments after a refinement, and there is no reference that says which is closer.
- So `computeInitialAcceleration=True` stays the default and nothing is landed.
- The consistent solve does produce a local max|A| spike (69 -> 7170), which is worth a look (see Open 2).

### c1_100 with the fixed branch
Setup: OMP=16, MKL_CBWR=AUTO,STRICT, ensight/restart off. Code snapshot `~/nidmem/fe_c100` = 2d0cd3e3; run in `~/nidmem/ph_c100`.

Model after the initial refinement: 455k dof, VIJ about 200 M.

Per refinement (identical for each of the initial and live refinements measured):

| phase | wall | RSS above entry |
|---|---|---|
| AMR apply | 2.5 s | +0.06 GiB |
| mass assembly | 7.2 s | +3.4 GiB |
| initial acceleration | 15 s | +3.4 GiB |
| linear solve (~12 s each, ~7 per increment) | - | +4.6 GiB |

**Survival and headroom.** It survives the reassembly after three live refinements, including the equivalent of the
step LEO4 died in. The LEO4 failing allocation was the removed M - M.T array.
RSS peak 32.0 GiB at 3460 s (22 increments incl. cutbacks), against 56 GB on LEO4 (at 28 threads; xeon ran 16). The run was still going at hand-off: `~/nidmem/ph_c100/phases.txt` (rewritten after every call > 0.5 s).

**NEW FINDING (open): RSS grows at every live refinement and never comes back down.**

| stage | steady RSS between increments |
|---|---|
| before any live refinement | 18.8 GiB |
| after the 1st | 23.1 GiB |
| after the 2nd | 25.3 GiB |
| after the 3rd | 29.7 GiB |
| later, no further refinement (constraint-connectivity rebuilds?) | 32.0 GiB |

- That is +2 to +4.4 GiB per refinement, far more than the ~100 new elements account for.
- Something from the old equation system survives each rebuild. Candidates, all unverified: the old DofManager/CSRGenerator (int32 gather map plus I/J, about 2.4 GiB at c1_100), blockamg's old hierarchy, or glibc arena fragmentation under free-threading.
- Extrapolated: 28 GB (x9) or 42 GB (LEO4, after 4 live refinements), consistent with LEO4's 41.6 GB maxvmem.
- **This, not the mass assembly, sets the headroom**: at about 3.5 GiB per refinement, 56 GB is reached after about 8 live refinements, whatever the solver.
- Next step: `gc.get_referrers` / tracemalloc snapshot diff across one refinement on c1_50, with MALLOC_ARENA_MAX=2 as a control.

### Open
1. The RSS growth per refinement above; not diagnosed.
2. The consistent initial-acceleration spike after refinement (max|A| x100); compare against a no-AMR, pre-refined control.
3. Incremental M reuse after refinement: not attempted. Mass assembly is now 7 s against 15 s for the initial acceleration and ~12 s per linear solve, so it is low priority.

## Phase 4 (2026-09-27): the RSS growth per equation-system rebuild. Diagnosed partly, no fix.

### c1_100 run result (Phase 3 run, `~/nidmem/ph_c100/phases.txt`)
- It finished at my `maxNumInc=24` cap: 4116 s, t = 0.0975.
- **It did not reach a 4th live refinement.** The cap counts cutback attempts, and the marker was still deferring (9 marked < 10).
- So the step LEO4 died on has NOT been reproduced.

| stage | steady RSS |
|---|---|
| before any live refinement | 18.8 GiB |
| after the 1st | 21.0 GiB |
| after the next rebuild (reuse path) | 23.1 GiB |
| after the 2nd refinement | 25.3 GiB |
| after the 3rd | 29.7 GiB |
| after a reuse-path rebuild | 32.0 GiB |
| end | 34.1 GiB (peak 34.12) |

### Where the growth happens (from the per-call RSS log)
- Every step up comes at the **first linear solve after a new DofManager**, whichever way it was rebuilt.
- Live refinement: the solve at t = 1286 s adds +2.1 to 3 GiB.
- Reuse path (constraint connectivity only, no element touched): t = 1454 s adds +2.1 GiB, t = 3325 s adds +2.1 GiB.
- The mass assembly and the initial-acceleration solve do not step it up.
- Answer to question 4: reuse-path rebuilds grow memory on their own, by the same ~2 GiB.

### What was ruled out
1. **Python-level owners.**
   - Method: weakrefs on every DofManager, CSR generator (via its `csrMatrix`; the Cython object is not weak-referenceable), `_NewmarkSystem` and `Mvij`, checked after `gc.collect()` at each rebuild.
   - Fixtures: NIDLiveAMR (refinement) and `tests/test_nid_contact.py` (4 reuse rebuilds). Scripts: `~/nidmem/leakprobe.py`, `reuseprobe.py`.
   - Result: the old system lives exactly one call longer, held by `_currentIncrement.system` until the next increment's `_NewmarkIncrement` replaces it. Everything is then collected; at the end only the current set is alive.
   - This transient overlap adds to the peak, not to the steady state.
2. **Allocator fragmentation, and growth on c1_50.**
   - Method: glibc `mallinfo2` after each increment, on c1_50 forced to refine every increment (nodeSet marker on `front_support`, maxLevel = 4; nDof 193k -> 200k -> 221k -> 289k -> 539k).
   - Controls: plain at OMP = 6 and 16, `MALLOC_ARENA_MAX=2`, and `malloc_trim(0)` after every increment. Runs in `~/nidmem/lk_*`, script `memprobe.py`.
   - Result: RSS stays proportional to nDof, 31-33 KB/dof throughout (6.4 GiB at 193k, 16.3 GiB at 539k).
   - Free memory inside the arenas stays at 0.3-0.5 GiB. Neither arena capping nor trimming changes anything beyond 0.3 GiB.
   - So c1_50 shows **no leak and no fragmentation**, and **does not reproduce** the c1_100 behaviour.
   - tracemalloc was tried first, but on c1_50 it slowed the run over 10x (still setting up after 9 min), so the snapshot-diff-by-traceback analysis was not done.
3. **blockamg Python caches** are cleared in `setModel`. The AMGCL C++ objects are held by `unique_ptr` and freed in `__dealloc__`. Both were checked by reading the code only.

### Remaining suspects (c1_100-specific)
- Native memory allocated in the first blockamg solve on a new pattern, when that solve does not converge. Every refinement on c1_100 was followed by "GMRES itself did not converge within outerMaxiter x outerRestart"; c1_50 had no such warning.
- The dump-on-degradation machinery. The deque has maxlen 0 and no dumps were written, so this is unlikely.
- Next experiment: the c1_100 deck with `mallinfo2` probing (`memprobe.py`), starting from the first live refinement (restart files would help). Plus one run with blockamg swapped for direct pardiso across a single rebuild, to split solver-native memory from the rest.

### No fix committed
Nothing was proven, so nothing was changed. A weakref test would pass today: old systems are collected.
