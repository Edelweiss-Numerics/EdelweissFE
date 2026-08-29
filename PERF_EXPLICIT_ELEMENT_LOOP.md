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
