# PERF: the explicit element loop — measurements, and three dead ends

**Branch:** `perf/explicit-element-loop`, off `feat/ned-explicit-dynamics`. **Machine:** rabbit
(AMD Ryzen Threadripper PRO 5975WX, 32 physical cores, SMT to 64, one NUMA node).
**Benchmark:** `examples/AnchorPryOut/perfbench/` — the explicit pry-out at 280k dof, 200 increments,
Ensight output removed, contact search and topology check disabled so the timing isolates the
increment. `PYTHON_GIL=0` throughout; GIL confirmed disabled through the `marmotelement` import.

> Written as the measurements came in, including the ones that went nowhere. The dead ends are the
> point of the document: each one is an hour someone else does not have to spend.

---

## Where the time goes (64 threads)

| phase | per increment | scales? |
|:--|--:|:--|
| `increment` | 68 ms | 1.23x from 8 to 64 threads |
| &nbsp;&nbsp;`elements` | 43 ms | 1.40x — the main puzzle |
| &nbsp;&nbsp;`assemble constraints` | 18 ms | **1.00x, perfectly flat** |
| &nbsp;&nbsp;remainder | ~7 ms | |

`assemble constraints` is 26 % of the increment and does not thread at all. It is a per-slave Python
loop inside `constraints/nodetodeformablesurfacepenalty.py::applyConstraint`, reached from
`NED.assembleConstraintForces`.

---

## Dead end 1: the serial scatter/reduce is not the problem

`computeElementsInParallelForExplicit` allocates a scatter buffer and reduces it with `np.bincount`
after the threaded region, both single-threaded. That looked like the obvious Amdahl term. Measured
at the model's sizes (1e6 scatter entries, 280,155 dof):

| | |
|:--|--:|
| `np.zeros(totalSize)` | 0.15 ms |
| `np.bincount(idx, weights, minlength=nDof)` | 1.79 ms |
| `target += <fresh nDof array>` | 0.11 ms |
| **total** | **2.04 ms** |

**2 ms of a 43 ms phase.** Not the cause. Do not spend time threading the reduction.

## Dead end 2: no NumPy-level replacement for the per-element gather is faster

The worker does `Ue = Un1[element]` and `dUe = dU[element]`, each a fancy-index gather that allocates
a fresh array — 28,072 allocations per increment. Replacing the allocation looked free. It is not:

| pattern | ms | vs current |
|:--|--:|--:|
| current, `U[idx]` | 22.08 | 1.00x |
| `np.take(U, idx, out=buffer)` | 38.28 | **0.58x** |
| `U.take(idx, out=buffer)` | 22.38 | 0.99x |
| ... with a per-element buffer slice | 26.51 | 0.83x |
| ... plus a precomputed `(elem, idcs)` list | 22.05 | 1.00x |

At 11.5 ns per element read this is a **memory-latency-bound gather** (1.9M random reads per
increment), not allocation overhead. It is irreducible at the NumPy level; only a different data
layout would help.

## Dead end 3: `computeInternalEnergy()` per element costs nothing

It is one extra Cython call per element whose result the journal reads only every
`output-frequency` increments, so skipping it on the other increments looked like free money.
Implemented, measured against baseline on the real model:

| | 8 threads | 64 threads |
|:--|--:|--:|
| baseline `elements` | 0.060 s | 0.043 s |
| with the call deferred | 0.060 s | 0.044 s |

**No difference.** Reverted rather than landed — a branch and two worker variants for zero gain is
exactly the speculative complexity that should not enter the codebase.

---

## What the isolated worker pattern actually does

Threaded microbenchmark of the worker's own bookkeeping, 14,036 entities, `PYTHON_GIL=0`:

| threads | full pattern | bookkeeping only (no gather) | gather only |
|--:|--:|--:|--:|
| 1 | 29.06 ms | 3.67 ms | 21.93 ms |
| 8 | 7.22 ms | 3.26 ms | 5.44 ms |
| 64 | 5.77 ms (**5.0x**) | 4.08 ms (**0.9x**) | 5.49 ms (**4.0x**) |

Two things fall out:

- **The gather scales** (4x), saturating memory bandwidth by 8 threads. More threads buy nothing
  because the bottleneck is bandwidth, not concurrency.
- **The pure Python bookkeeping is flat** — dict lookups, slice views, item writes on shared objects.
  This *is* free-threading reference-count contention, and it is real, but it is only ~3.7 ms.

Together the whole pattern costs ~5.8 ms at 8+ threads against a measured `elements` phase of 43 ms.
**So roughly 37 ms of the phase is the Marmot kernel itself**, and none of the Python-side theories
account for it. That is where the remaining investigation has to go.

## For comparison: what EdelweissMeshfree does

Its *explicit* solver uses the **same** free-threaded `ThreadPoolExecutor` — it even imports
`chunked_iterable` from EdelweissFE. The Cython `prange`/`nogil` routines in
`solvers/base/parallelization.pyx` belong to the implicit/quasi-static path, not the explicit one, so
"Meshfree solved this with Cython" is wrong.

What its explicit worker does differently is the *body*:

- its force pass reads **no global vector at all** — `computePhysicsKernelsExplicit(PP)` /
  `computeLumpedInertia(MP)` / `computeLumpedMomentum(MVP)` write into scatter views and take
  neither `U` nor `dU`, so it allocates nothing per particle;
- the state update is a **separate pass** reading only `dU[particle]` — one gather, not two;
- no per-entity energy call.

Given dead end 2, the interesting part of that design is not the avoided allocation but the avoided
*gather*: one fewer random-access read of the global vector per entity.

---

## What was actually fixed, and by how much

### 1. The connectivity search: 7.05x (committed)

`updateConnectivity` swept every slave against every facet, one clamped closest-point call per pair —
778,000 calls, **5.98 s per call**. A `cKDTree` over facet centroids now supplies candidates, under a
bound that makes the result bit-identical rather than merely close: a triangle's centroid lies inside
its own closed domain, so `d* <= d0`, hence `|x - C*| <= d* + r* <= d0 + rMax`. Every facet that could
win or tie has its centroid in that ball.

**5.98 s -> 0.848 s per call.** All 13 contact test cases unchanged, including the implicit ones that
run this every increment.

The point is not only the time. At the frequency the explicit input had to use (500) the search cost
12 ms per increment; it is now 1.7 ms — and the *default* frequency of 100 costs 8.5 ms, less than the
old compromise did, so the compromise can be dropped whenever the contact resolution matters more than
the last 7 ms.

### 2. A force-only constraint path: 19.4 % of the constraint assembly (committed)

An explicit increment assembles no system matrix, so every tangent a constraint computes is discarded.
Measured on this model, that waste was 2.5 ms of tangent arithmetic (`np.outer(w, w)` plus four block
writes per slave) and 1.2 ms of container construction.

`ConstraintBase.applyConstraintForcesOnly` now says so. Its default still builds a throwaway container
through the implicit matrix' own protocol, so no constraint had to change; the deformable-surface
contact overrides it by running its single loop with `K=None` and guarding the tangent — one loop, so
the implicit and explicit paths cannot drift apart.

**`assemble constraints` 18.93 -> 15.25 ms, increment 66.4 -> 63.9 ms**, exactly the 3.7 ms predicted.

---

## Dead end 4: load imbalance is not the problem either

The chunk factor gives 4 chunks per thread. With GCDP costing wildly different amounts per element
(elastic vs return mapping), imbalance looked plausible. Swept at 64 threads:

| chunks per thread | `elements` |
|--:|--:|
| **4 (current)** | **0.0436 s** |
| 16 | 0.0445 s |
| 64 | 0.0716 s |
| 256 | 0.2890 s |

The current value is already optimal; finer chunking is catastrophic (6.6x worse at 256), dominated by
dispatch. Do not touch it.

---

## The element loop: what is left, and what it is not

`elements` is 43 ms at 64 threads and **scales 6.44x from 1 to 64 threads** — near-linear to 4 threads,
saturating at 8. An earlier note in this document quoted "1.42x", which was the 8-to-64 range only,
i.e. the already-saturated tail. The loop is not broken; it is saturated.

Ruled out, each by measurement:

| hypothesis | verdict |
|:--|:--|
| serial scatter buffer + `np.bincount` reduction | 2.04 ms of 43 ms |
| per-element gather allocation | no NumPy alternative is faster; `np.take(out=)` is 0.58x |
| per-element `computeInternalEnergy()` | zero measurable cost |
| Cython call / memoryview overhead | ruled out by the above — 14,036 fewer calls changed nothing |
| load imbalance / chunk granularity | current factor already optimal |

Of the 43 ms, roughly 6 ms is the worker's own bookkeeping and gather (measured in isolation, and the
gather part does scale 4x). **The remaining ~37 ms is the Marmot C++ kernel**, and its memory traffic
(~17 MB per increment of quadrature-point state) is far below this machine's bandwidth, so the
saturation is not obviously bandwidth either.

Pursuing it needs C++-level profiling (`perf record` on the element pass), not more Python-side
theories. That is the next step, and it is a different kind of investigation from everything above.

## Remaining, with measured potential

| item | worth |
|:--|--:|
| vectorise the contact force loop across slaves (frozen `w`, indices, gaps) | ~15 ms of 64 ms |
| profile the Marmot kernel's thread scaling in C++ | up to ~37 ms, unknown |
| early termination in the candidate loop (needs explicit tie-breaking to stay bit-identical) | ~1 ms |

---

# 2026-08-30: the section above had the attribution backwards

Everything up to here concluded that ~37 ms of the 43 ms element phase was the Marmot C++
kernel and that its saturation was unexplained. **That was wrong, and the experiment that
settled it took four minutes**: run the loop with the kernel call removed and keep only the
Python gathers.

| loop | 8 threads, 1 CCD | 8 threads, 4 CCDs |
| --- | --- | --- |
| gather only (no kernel) | 0.03765 s | 0.08788 s |
| full loop | 0.05834 s | 0.10510 s |
| **C++ kernel (difference)** | **0.02069 s** | **0.01722 s** |

The Marmot kernel scales **7.8x on 8x threads** and is ~6 % of the element phase at 64
threads. The Python gather layer is the rest, and it had *negative* thread scaling
(0.03765 s at 8 threads, 0.04049 s at 64). Every thread-scaling number in the sections
above was measuring Python bookkeeping, not physics.

## 5. The entity lookup returned a numpy subclass: 1.48x (committed 207260eb)

`DofVector[entity]` cost 0.78 us per call. Broken down on a 280155-dof model:

| component | cost | share |
| --- | --- | --- |
| numpy subclass wrapping (`__array_finalize__`) | ~0.43 us | 54 % |
| the gather itself | 0.205 us | 26 % |
| the `isinstance` chain | 0.137 us | 17 % |
| entity dict lookup | 0.030 us | 4 % |

Trying the dict first and returning a plain ndarray view removes the first and third:
0.78 -> 0.26 us, element phase 0.04315 -> 0.02722 s, increment 0.06394 -> 0.04335 s.
**Bit-identical** on the full model (KE and RF matched to every digit at increment 10000).

## 6. One gather per chunk instead of two per element: 1.13x (committed c81c118c)

The win is reduced allocator contention, not fewer numpy calls: batching is worth 1.05x
single-threaded (the gather is memory bound at ~3 ns/dof) but 1.76x at 64 threads and
4.79x at 8. **Always test an allocation hypothesis with threads** -- single-threaded it is
invisible. Element phase 0.02742 -> 0.02344 s; increment 0.04379 -> 0.03893 s.

The plan is cached and a stale one would gather the wrong DOFs *silently*. Invalidation is
keyed on the identity of the entity mapping, which the DofManager replaces wholesale on
every topology change, so live h-adaptivity invalidates it by construction.

## Dead end 5: CPU pinning gains nothing (and hurts at 16 threads)

CPU placement matters enormously when threads are *deliberately* scattered -- the same 8
threads cost 0.05834 s packed onto one CCD and 0.10510 s spread over four (1.80x; 1.40x
after 207260eb). It is tempting to conclude the pool should pin its workers. It should not:

| config | elements | increment |
| --- | --- | --- |
| 64 threads, pinned per cache domain | 0.02787 s | 0.04363 s |
| 64 threads, unpinned | 0.02742 s | 0.04379 s |
| 16 threads, pinned | 0.03951 s | 0.05494 s |
| 16 threads, unpinned | 0.03579 s | 0.05100 s |

The Linux scheduler already packs threads, so explicit pinning recovers nothing and only
removes its freedom -- which is why 16 threads confined to one CCD (8 physical + SMT) lose
to 16 threads on 16 physical cores. Note also that most of the 1.80x was shared-object
contention removed by 207260eb, not cache geometry. **An effect measured on an old build
is not evidence about a new one.**

## Cumulative

| | s/increment |
| --- | --- |
| before any of this work | 0.0786 |
| after cKDTree + force-only path | 0.06394 |
| after the plain-ndarray entity lookup | 0.04379 |
| after the batched per-chunk gather | **0.03893** |

~2.0x, all of it bit-identical or reference-verified.

## What is left

The gather is still the bulk of the element phase and is now close to memory bound.
Moving it into Marmot C++ was measured as **not** worth it: C++ faces identical memory
traffic and its only gain is the allocation reduction the batching already captures.
