# Winkler L-panel

The L-shaped concrete panel of Winkler et al. under an upward displacement of its right arm, the
classic test for cracking from a re-entrant corner. Three variants of the same model show EdelweissFE's
field-split block-AMG linear solver (`blockamg`) and its live h-adaptivity (AMR):

| File | Linear solver | Mesh | What it shows |
|---|---|---|---|
| `WinklerL_pardiso.inp` | PARDISO (direct) | fixed | the reference solution |
| `WinklerL_blockamg.inp` | blockamg | fixed | same result as the reference, faster |
| `WinklerL_blockamg_amr.inp` | blockamg | refined live along the crack | adaptive refinement of the crack band |

Variants 1 and 2 differ only in their `linsolver` lines, so they give the same load-displacement
curve. Variant 3 refines the mesh where the concrete starts to yield, which resolves the crack band
more finely and gives a lower curve.

## Requirements

- EdelweissFE built with its Marmot extensions (`pip install -v -e .` against a Marmot install).
- Marmot with the **GCDP** material module (gradient-enhanced concrete damage plasticity). GCDP is a
  private Marmot module, not part of the public Marmot repository.

## Running

Run each variant on its own; they are sized for one workstation:

```bash
export PYTHON_GIL=0 OMP_NUM_THREADS=32
edelweissfe WinklerL_pardiso.inp
edelweissfe WinklerL_blockamg.inp
edelweissfe WinklerL_blockamg_amr.inp
```

Never set `OMP_PROC_BIND` or `OMP_PLACES`: the element loop runs on Python threads, and pinning the
initial OpenMP thread confines all of them to a single core.

## Files

| File | Content |
|---|---|
| `model.inp` | the shared model: units, geometry, material, sections, field outputs (included by every variant) |
| `mesh.inp` | 20-node hexahedral mesh: GC3D20R concrete (displacement + nonlocal damage), C3D20R steel |
| `mesh_hex8.inp`, `hex8_to_hex20.py` | the linear source mesh and the script that generated `mesh.inp` from it |
| `blockamg.json` | every blockamg setting, defaults included (see below) |

To regenerate the mesh: `python hex8_to_hex20.py mesh_hex8.inp mesh.inp`.

## The model

Units are mm, N and MPa. The concrete panel spans x 0..500 mm and y 100..500 mm, without the cut-out
x > 250, y < 250, and is 50 mm thick. A steel support block (x 0..250, y 0..100) sits under the left
leg; a steel load block (x 400..500, y 250..350) sits at the underside of the right arm.

- The support block is clamped at its bottom face (`setBottom`).
- The complete back face z = 0 is supported in z (`setBack`, u_z = 0), holding the plate against
  out-of-plane bending.
- The load line in the load block (`setLoad`) is raised by 1 mm over one step.

The load peaks at about 0.1 mm; the panel then softens as a crack grows from the re-entrant corner
(250, 250) to the left free edge.

## Expected results

Measured on an AMD Ryzen Threadripper PRO 5995WX, 32 threads, one run at a time (EdelweissFE
`next_v26.11`, September 2026):

| Variant | Run time | Linear solves | Time per solve | Peak load | Load at 1 mm |
|---|---|---|---|---|---|
| pardiso | 2174 s | 537 | 3.89 s | 3072 N at 0.110 mm | 132 N |
| blockamg | 2020 s | 538 | 3.59 s | 3072 N at 0.110 mm | 132 N |
| blockamg + AMR | 6027 s | 1091 | 4.47 s | 2773 N at 0.095 mm | 108 N |

Variant 3 refines 21 times; the model grows from 218,564 to 286,856 degrees of freedom. At this
model size a direct solver is still competitive: blockamg's advantage grows with the problem size.

## Viewing the results

Each variant writes `WinklerL_<variant>.case` (EnSight Gold) with the displacement, the nonlocal
damage field, the damage `omega` and the plastic hardening variable `alphaP` (largest value per
element), plus the reaction force and load-line displacement histories as `RF_<variant>.csv` and
`U_<variant>.csv`.

The case files hold two time sets: the geometry (time set 1; one step, or one step per refinement
pass in variant 3) and the variables (time set 2; one step per increment). ParaView handles this
itself. In pyvista, select the variables' time set before reading:

```python
import pyvista as pv

reader = pv.get_reader("WinklerL_blockamg_amr.case")
reader.set_active_time_set(1)  # the variables' time set
reader.set_active_time_value(reader.time_values[-1])
mesh = reader.read()
```

## The blockamg settings

`blockamg.json` lists every setting blockamg accepts, defaults included, so that each one is visible.
The values that differ from blockamg's own current defaults are marked **bold** -- blockamg's shipped
defaults changed after this file was tuned and smoke-tested, so several keys below are now pinned
explicitly to the *former* defaults rather than adopt the new, unverified trajectory; those are bold
too, even though they were not bold when this file was first written.

| Setting | Value | Meaning |
|---|---|---|
| `outerSolver` | `amgcl_lgmres` | the outer Krylov solver: AMGCL's LGMRES (the only accepted value) |
| `outerTol` | `adaptive` | the outer tolerance: Eisenstat-Walker forcing between `etaMin` and `etaMax`, or a fixed number |
| `outerRestart` | **50** | outer Krylov restart length (default 100) |
| `outerMaxiter` | **3** | outer restart cycles, so at most `outerRestart * outerMaxiter` iterations (default 1) |
| `lgmresM` | **30** | LGMRES inner subspace size (default 50) |
| `lgmresK` | 3 | number of LGMRES augmentation vectors (default 3) |
| `sweeps` | 1 | block Gauss-Seidel sweeps over the fields per preconditioner application |
| `symmetric` | false | no reverse sweep after the forward one: same iteration count, ~30 % cheaper here (now also the default) |
| `krylovOnScaledSystem` | **true** | iterate the outer Krylov solve on the equilibrated system `D A D` rather than `A` itself (default false) |
| `fieldPreconds` | `{"displacement": {...}}` | **pins the displacement field's former default tree** (plain scalar backend, Chebyshev degree 5): the current default builds a node-wise 3x3 float-block hierarchy instead, not yet verified against this model |
| `useRigidBodyNullspace` | true | give the displacement AMG the six rigid-body modes as near null-space (only takes effect on the scalar backend the pinned `fieldPreconds` tree above uses) |
| `p1FieldNames` | `[]` | fields preconditioned by p-multigrid through their corner nodes (experimental); empty here, so every other `p1*` tuning key (`p1Nu`, `p1NuPre`, `p1NuPost`, `p1FineDegree`, `p1FineBlockSize`, `p1FineValueType`, `p1CoarseValueType`, `p1FineSpectralRadiusReuse`, `p1CoarsePrecond`) is unset and moot |
| `hierarchyDropTol` | 0.0 | drop small entries before building each AMG hierarchy (0: keep all) |
| `etaMin` | 1e-4 | lower bound of the adaptive tolerance: late Newton iterations need no more accuracy (now also the default) |
| `etaMax` | **3e-4** | upper bound of the adaptive tolerance (default 1e-3) |
| `ewGamma`, `ewAlpha` | 0.9, 1.618 | Eisenstat-Walker forcing parameters |
| `residualGrowthFactor` | 4.0 | a jump of the right-hand side by this factor marks a new increment (tolerance and hierarchies reset) |
| `refreshOnNewIncrement` | true | refresh the AMG hierarchies whenever a residual jump marks a new increment (default) |
| `hierarchyStalenessFactor` | **1.5** | rebuild the AMG hierarchies when a solve needs this many times (and 15 more than) the outer iteration count of the solve the hierarchies were built for (default 3) |
| `trueResidualMaxContinuations` | **0** | warm restarts to reach the tolerance on the true residual (default 2); only on the scaled system, so only relevant together with `krylovOnScaledSystem` above |
| `gapCompensatedTolerance` | **true** | tighten the first pass by the measured gap between scaled and true residual (default false); only on the scaled system |
| `gapSafetyFactor` | 0.3 | safety factor of that tightening |
| `gapMaxFactor` | 1e3 | upper clamp on the measured gap's exponential moving average (default) |
| `verbosity` | **info** | `silent`, `warning` (default), `info` or `debug` |
| `warnOuterIterationsThreshold` | 100 | warn about solves that need more outer iterations |
| `dumpOnDegradationThreshold`, `dumpOnDegradationMaxDumps`, `dumpOnDegradationContextSolves` | 100, 10, 0 | capture degraded solves for offline diagnosis (only active with `dumpOnDegradationDir`) |

Two settings have no value to write down, because they are off by default: `dumpOnDegradationDir`
(a directory to enable the capture above) and `hotReloadConfigFile` (a JSON file re-read during the
run, to change settings on the fly). `lgmresAlwaysReset` and `lgmresResetOnNewIncrement` are no longer
set here either: both are deprecated and accepted only at their one remaining value (discard LGMRES's
recycled vectors before every solve, never carry them across an increment), so leaving them unset and
pinning them explicitly no longer differ.

The two settings that matter most for this model are `symmetric` and `etaMin` -- both now also
blockamg's own defaults, but pinned here regardless so this file keeps stating, explicitly, the
configuration it was tuned and smoke-tested against. A key blockamg does not know -- a typo, say --
stops the run with an error that lists the recognized keys.
