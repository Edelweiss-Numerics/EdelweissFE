# Hyperbolic gradient damage: calibration and validation

An explicit solver cannot solve the gradient-enhanced field's elliptic equation, so it is made
parabolic with the non-local viscosity `eta` and marched with forward Euler. That limit falls off
with the **square** of the element size,

    dt <= 2 eta / (1 + C l^2/h^2)   ->   2 eta h^2 / (C l^2)     for h << l,

and nothing in the code checks it: `computeCriticalTimeStepForExplicitDynamics` covers the
mechanical field only. The only knob it offers is `eta` itself, which is the artificial delay
between the damage front and the strain driving it — so on a refined mesh stability is bought by
making the regularisation lag.

A **micro-inertia** `m_k` makes the field second order in time,

    m_k d2(eps_bar)/dt2 + eta d(eps_bar)/dt + eps_bar - c grad^2(eps_bar) = eps_tilde,

which is a damped wave equation whose stable increment falls off **linearly** with `h`. The
viscosity keeps its meaning and changes role: it is now the damping. See
`Marmot/doc/pages/features/nonlocalmicroinertia.rst` for the derivation, the parameter choice, and
what this does *not* fix.

## The model

The bar of `examples/AlphaP_AMR_Study` — 100 x 5 x 5 mm, `GC3D20R`, GCDP, `l = 5 mm`, with a
weakened central element (`ftu` 2.68 against 2.70 MPa) to localise the crack — which is the
calibrated fracture-energy benchmark for this material. Reusing it means the load-displacement
curve and the dissipated energy can be checked against a reference established independently of
anything measured here. The explicit runs add the density and the non-local viscosity, which an
implicit run does not need.

## The parameter is not free

Requiring that the zeroth-order reaction mode not ring gives `m_k <= eta^2/4`, and the largest
admissible value is the best one because `dt` grows with `sqrt(m_k)`. So there is **one** parameter,
not two: pick `eta` for the lag you accept — exactly as before — and take `m_k = eta^2/4`. For this
deck's `eta = 1e-5 s` that is `m_k = 2.5e-11 s^2`, a damage wave speed of `2l/eta = 1e6 mm/s`, which
crosses the `2*pi*l = 31 mm` process zone in 31 microseconds against a 1 ms ramp.

## Running it

    python run_study.py                  # the full matrix, nX = 20, 40, 80, 320
    python run_study.py --quick          # coarse meshes, short ramp
    python run_study.py --meshes 20,320  # just the two that matter

Each case lands in `run_<scheme>_nX<n>/` and is skipped if its `RF.csv` is already there.

Requires a Marmot built with the micro-inertia (`feat/hyperbolic-gradient-damage` or later) and the
EdelweissFE Cython extensions rebuilt against it — adding a virtual to `MarmotElement` changes the
vtable, so a stale `libMarmot` is not merely out of date, it is wrong.

## What the result has to show

**Agreement** — the validation:

- peak force and fracture energy `G_f` match the parabolic run on the same mesh, and both approach
  the implicit reference of `AlphaP_AMR_Study` (`G_f ~ 0.098 N/mm`) as the mesh is refined;
- the localisation band stays `l`-controlled, i.e. its width does not change with the scheme.

**Difference** — the point:

- the parabolic scheme's own limit crosses the mechanical one somewhere between `nX = 80` and
  `nX = 320` (`h = 1.25` to `0.31 mm` against `l = 5 mm`), and since nothing checks it, that run is
  expected to trip the solver's energy guard — `ENERGY IS BEING CREATED` — rather than fail cleanly;
- the hyperbolic run at the same mesh stays governed by the mechanical limit, so its increment count
  tracks `1/h` instead of `1/h^2`.

The `dt`-scaling claim itself does not need these runs: it is measured directly, per element, by
`TestGeneralGradientEnhancedDisplacementFiniteElement` in Marmot, which finds `dt(h/2)/dt(h) =
0.500676` against the parabolic scheme's 0.25. What these runs establish is that buying that
scaling did not change the physics.

## Watch the diagnostics, not just the curve

Mass-proportional damping controls the *low* modes: at `m_k = eta^2/4` the reaction mode is
critically damped and the shortest-wavelength modes are barely touched. Since GCDP's damage rides an
accumulating internal variable, an overshoot of `eps_bar` above `eps_tilde` is written in
irreversibly. Two things in the log are worth reading:

- the `micro-inertial (not an energy)` row of the energy table — that is the energy in the ringing;
  it should decay after the initial transient and after every refinement, not persist;
- `alphaP` in the Ensight output away from the crack — a rising value where nothing is loading is
  the signature of ringing being integrated into damage.

If either shows up, the fix is a stiffness-proportional damping term, which is not implemented; see
the limitations section of the Marmot doc page.
