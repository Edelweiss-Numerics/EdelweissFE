# Bonded anchor pry-out, full scale — implicit against explicit

A real 280 000-dof engineering model: a bonded anchor pulled sideways out of a concrete slab, with
gradient-enhanced damage in the concrete, five tie constraints, two contact pairs, and a mesh that
refines itself where the concrete yields.

Two variants, differing only in how time is integrated:

| | Gauss-point-to-segment contact |
|---|---|
| **implicit static** | `pryout_implicit_gpts.inp` |
| **explicit dynamic** | `pryout_explicit_gpts.inp` |

```
edelweissfe pryout_implicit_gpts.inp
edelweissfe pryout_explicit_gpts.inp
```

## Cost — read this before starting one

**This is not a smoke test.** The reference explicit run of this model took **32.2 hours on 64
cores** and 1 832 580 increments. The configuration here is cheaper — the mass scaling is ×100
rather than the ×10 that run used, which raises the stable increment by √10 — but you should still
budget hours, on a machine you are not otherwise using.

If you want the same physics in half an hour, run
[`../AnchorPryOutCoarse`](../AnchorPryOutCoarse/) instead. It is the same model on a deliberately
coarse mesh, and it additionally offers node-to-surface contact variants for comparison. This
directory exists for the case where the answer has to be mesh-credible rather than quick.

## Where this model comes from

It is a translation of a real Abaqus/Standard model, `AnchorPryOut.inp`, and the translation made
seven modelling decisions that are not free parameters — the concrete element, the steel element,
what happened to the cohesive mortar layer, the node numbering, and why the Abaqus `*Dynamic` step
became a static one here. Those are recorded in
[`TRANSLATION_NOTES.md`](TRANSLATION_NOTES.md), together with the reasoning behind the `blockamg`
configuration. **Read it before changing the physics.**

## What it exercises

- **GCDP** gradient-enhanced concrete damage-plasticity, in a genuinely softening regime
- **Live h-adaptivity** driven by the material state, with hanging-node constraints
- **Integrated (GPTS) contact** on quadratic element faces
- **Five tie constraints**, including one whose slave nodes are contested by the hanging-node
  constraints — which is why the two claims have to be arbitrated centrally
- For the implicit variant, **`blockamg`** on a hard, ill-conditioned, path-dependent problem
- For the explicit variant, **mass scaling, energy balance and a stable-increment audit**

At the first topology update both variants report the same thing, which is the quickest check that
your run is set up correctly:

```
AMR ModelModifier: marked 110, refined -> active elements 11176 -> 11946, 574 hanging nodes
eliminating 8804 slave DOF(s) via multi-point constraints
```

## How the files fit together

A variant is only the three lines that make it that variant. Everything else is shared, literally —
the same files, not copies that have to be kept in step.

```
pryout_implicit_gpts.inp ─┐
pryout_explicit_gpts.inp ─┤
                          ├─ materials_implicit.inc   /  materials_explicit.inc   <- differs
                          ├─ model.inc                                            <- shared
                          ├─ contact_gpts.inc                                     <- shared
                          ├─ adaptivity.inc                                       <- shared
                          ├─ outputs.inc                                          <- shared
                          └─ steps_implicit.inc       /  steps_explicit.inc       <- differs
```

`model.inc` holds the mesh includes, the complete z-symmetry node set, all five sections, all
fourteen facet-surface generators and all five ties. It was factored out only after checking that
those sections were identical, line for line, in the two decks it came from.

`mesh/` is generated, and nothing else has to be. In particular the z-symmetry node sets come from
the mesh itself and are complete: of the 56 881 nodes, 5 259 lie on the z = 0 plane and all 5 259
are members of `z_symm_concrete` / `z_symm_steel`. An earlier revision of this mesh was short by
516 nodes and had to be patched with a generated side file; that mesh has since been regenerated,
the patch became a byte-for-byte duplicate of what the mesh already declares, and it is gone.
`model.inc` records the check.

## Reading the two solvers against each other

Both ramp the plate to the same 5.0 mm and refine on the same criterion, so they are comparable at
equal displacement rather than at equal cost.

| | implicit | explicit | why |
|---|---|---|---|
| densities | present but unused | **×100 mass scaling** | a static solve builds no mass matrix; the explicit stable increment goes with 1/√ρ, so the scaling is what makes the run finish. It costs inertia: watch the energy table, not the wall clock |
| predictor | **`extrapolation=off`** | — | the linear predictor overshoots through contact-status flips and localizing damage. Measured on this exact model: ON gave a cutback spiral (three >100-iteration solves in one increment); OFF ran 230 increments past where ON died |
| amplitude | linear in step progress | **quintic smoothstep** | the plate must start from rest with zero velocity *and* zero acceleration, or the first increment shocks the structure. Harmless under statics |
| courant number | — | **0.1**, not the 0.8 default | `l/c` is a linear-element formula and a quadratic element's highest eigenfrequency is several times what it predicts |
| AMR marker | `alphaP >= 1` | `alphaP >= 1` | deliberately identical, so the comparison is not confounded. `alphaP` is monotone; a stress threshold carries the full wave content and, with no coarsening anywhere in the code, ratchets towards refining everything |
| `splitFactor` | 2 | 2 | 3 is free implicitly and **not** free explicitly — it takes the refined concrete below the anchor's stable increment. 2 is what makes the meshes match |

## The reference result

The best-characterised run of this model to date — same mesh, same GPTS contact, ×10 mass scaling
and an initial cone-shaped refinement instead of the live marker used here — peaked at

**63 268 N at U = 3.045 mm**, descending to 50 567 N by U = 4.986 mm.

That is a genuine post-peak branch: the drop is 12.7 kN against a frame-to-frame wobble of about
±1.5 kN. Earlier runs of this model were repeatedly stopped on the rising branch and their
"capacities" retracted, so treat any number from a run that did not reach a peak as no number at
all.

## Where to take it next

- `maxLevel=2` lets `alphaP` refine the borehole layer that the initial marker already refined. It
  is not free for the explicit variant: refined concrete reaches ~0.62 mm, where it starts competing
  with the anchor's 1.25 mm elements for the stable increment.
- The Duvaut-Lions viscosity is `0` here. `AnchorPryOutCoarse` uses `1e-6` implicitly, and what the
  right value is for a softening run at this scale is genuinely open.
- `report-performance=True` is set on the explicit variant. It prints the cost structure per
  reporting interval, which is how you see a refinement or a contact search becoming expensive while
  the run is still going rather than afterwards.
