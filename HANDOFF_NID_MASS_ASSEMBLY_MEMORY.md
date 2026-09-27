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

c1_50 at OMP=1, 3 increments, base vs fix: runs in `~/nidmem/bit_base` / `bit_fix` were still going at hand-off.
Compare `RF_loading.csv`, `U_loading.csv`, `maxDamage.csv` once each has a `DONE` file.

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
