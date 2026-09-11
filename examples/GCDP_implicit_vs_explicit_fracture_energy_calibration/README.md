# GCDP fracture energy: implicit against explicit, all variants

One uniaxial tension bar, one loading history, five ways of integrating it. The calibrated quantity
is the dissipated work per unit fracture area, `G_f`, and the question the study answers is narrow:

> Does buying an explicit time step — with a micro-inertia, with artificial bulk viscosity, or both
> — change the fracture energy the model delivers?

## The variants

| variant | non-local field | mechanical field | stable `dt` scaling |
|---|---|---|---|
| `implicit` | elliptic, Newton–Raphson | — | unconstrained |
| `explicit-parabolic` | 1st order in time (viscosity as the mass), forward Euler | central difference | `h²`, **checked by nothing** |
| `explicit-hyperbolic` | 2nd order in time (micro-inertia as the mass, viscosity as the damping) | central difference | `h` |
| `explicit-parabolic-bv` | as above | + artificial bulk viscosity | `h²` |
| `explicit-hyperbolic-bv` | as above | + artificial bulk viscosity | `h` |

Held fixed across all five, so the comparison means something: geometry, material, the weakened
central element, and the **total prescribed elongation of 0.2 mm** — the same endpoint
`examples/AlphaP_AMR_Study` uses, so `G_f` is directly comparable to its reference value of
**0.098 N/mm**. The Duvaut-Lions viscosity is zero everywhere, which is what makes implicit and
explicit the same constitutive problem rather than two different ones.

## The two parameters are not free

- **`m_k = η²/4 = 2.5e-11 s²`.** Requiring the zeroth-order reaction mode not to ring bounds the
  micro-inertia at `η²/4`, and the largest admissible value is the best one because `dt ∝ √m_k`. So
  it follows from the viscosity the deck already had — one parameter, not two. Going below it lands
  in the overdamped branch where `dt ≈ 2m_k/η → 0`; measured directly, `m_k = 1e-14` against
  `η = 1e-4` gives `dt = 1.999951e-11 s`.
- **`b1 = 0.06, b2 = 1.2`**, the Abaqus/Explicit defaults for bulk viscosity. Defaults, not
  recommendations: `b1` is sized to damp the highest resolvable frequency of the mesh, not to
  represent physical dissipation.

## Running it

```
python run_calibration.py                                   # all five, nX = 20..160
python run_calibration.py --quick                            # nX = 20, 40
python run_calibration.py --variants implicit,explicit-hyperbolic
python run_calibration.py --meshes 40,160 --threads 16
```

Each case lands in `run_<variant>_nX<n>/` and is skipped if its `RF.csv` exists. Requires a Marmot
carrying both the bulk viscosity and the micro-inertia, with the EdelweissFE extensions rebuilt
against it — adding a virtual to `MarmotElement` changes the vtable, so a stale `libMarmot` is not
merely out of date, it is wrong.

## Reading the table

Two columns exist to stop the headline number being believed too easily:

- **`F_end`** — the force still standing at the endpoint, as a share of peak. `G_f` is an integral
  over a softening tail, so it has measured all of the dissipation only where this is small. This is
  a property of the loading history, not of the integration scheme, which is why every variant here
  shares one endpoint.
- **`T/W_ext`** — the kinetic share of the external work at the last reporting increment, i.e. how
  quasi-static the explicit run actually was. Measured, not assumed.

`DIVERGED` is expected for `explicit-parabolic` once the mesh is fine enough: its own stability
limit falls off with `h²` and nothing checks it, so the run reports a finite `dt`, goes to NaN, and
keeps going. On a solver without the NaN check in the energy guard it looks like a success — hence
the force being used as the backstop for the status column.

## Results

The full run -- the derivation of `eta`/`m_k`/the mass scaling, the results table, and what it
establishes about agreement, stability, and the cost of bulk viscosity -- is written up separately
rather than kept here.

## Watch for, in the explicit variants

Mass-proportional damping controls the low modes: at `m_k = η²/4` the reaction mode is critically
damped and the shortest-wavelength modes are barely touched. GCDP's damage rides an accumulating
internal variable, so an overshoot of `ε̄` above `ε̃` is written in irreversibly. Two things in the
log are worth reading before trusting a curve:

- the `non-mechanical-inertia (not an energy)` row of the energy table — the energy in the ringing; it
  should decay after the initial transient, not persist;
- `alphaP` in the Ensight output away from the crack — a rising value where nothing is loading is
  ringing being integrated into damage.

If either shows up, the answer is a stiffness-proportional damping term, which is not implemented.
