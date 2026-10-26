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

Every variant gives the same load-displacement curve up to the AMR one:

| Variant | Peak load | Load at 1 mm |
|---|---|---|
| pardiso | 3072 N at 0.110 mm | 132 N |
| blockamg (defaults) | 3072 N at 0.110 mm | 132 N |
| blockamg, former settings (for comparison) | 3072 N at 0.110 mm | 132 N |
| blockamg + AMR (former settings) | 2773 N at 0.095 mm | 108 N |

With its defaults, blockamg's looser linear tolerance can cost a few Newton iterations compared with PARDISO, but
each solve is much cheaper, and the defaults were measured faster end to end than both PARDISO and blockamg's
former settings. Variant 3 refines repeatedly along the crack band, and the model grows by roughly a third.

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

The `*linearSolver, type=blockamg` block sets only `verbosity=info` (one log line per linear solve, with its outer
iteration count): every other setting is the default of the `blockAMG()` preset. The table lists its data lines with
their defaults; `doc/source/documentation/composedlinsolvers.rst` shows the tree they build and
`doc/source/documentation/fieldsplitmeasurements.rst` explains why the defaults are what they are.

| Data line | Default | Meaning |
|---|---|---|
| `tolerance` | adaptive | a fixed relative tolerance; unset: Eisenstat-Walker forcing between 1e-4 and 1e-3 |
| `maxIterations` | 100 | the outer-iteration cap of one solve; a solve that reaches it rebuilds the preconditioner |
| `krylovOnScaledSystem` | false | the outer solve runs on `A` itself, preconditioned by `D M^-1 D`, so its residual is the true one |
| `sweeps`, `symmetric` | 1, false | one forward block Gauss-Seidel sweep over the fields per preconditioner application |
| `rigidBodyCoarseLevel` | true | displacement (3D): a two-level cycle on its rigid-body modes; else (and for the scalar nonlocal damage) one smoothed-aggregation AMG hierarchy |
| `dropTolerance`, `lumpDroppedEntries` | 0, false | build the field preconditioners on filtered blocks |
| `verbosity` | **info** (default `warning`) | `silent`, `warning`, `info` or `debug` |

The rebuild policy (rebuild when a solve needs 3 times the outer iterations of the solve the preconditioner was built
for, and on a new increment) and the AMGCL settings of each field are those of the preset; to change them, write the
tree in a `type=custom` solver script.
