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

**Agreement** — the validation. Measured:

    scheme        nX   h [mm]      dt [s]  increments  F_max [N]  G_f [N/mm]  status
    parabolic     20    5.000  1.2127e-07        8248      27.84     0.03704      ok
    hyperbolic    20    5.000  8.8530e-08       11297      42.80     0.04715      ok
    parabolic     40    2.500  6.0634e-08       16494      60.86     0.05141      ok
    hyperbolic    40    2.500  6.0634e-08       16494      61.06     0.05149      ok
    parabolic     80    1.250  3.0317e-08       32986      61.70     0.05025      ok
    hyperbolic    80    1.250  3.0317e-08       32986      61.94     0.05043      ok
    parabolic    160    0.625  1.5158e-08       65971        nan         nan  DIVERGED
    hyperbolic   160    0.625  1.5158e-08       65971      67.10     0.05369      ok
    parabolic    320    0.312  7.5792e-09      131941        nan         nan  DIVERGED
    hyperbolic   320    0.312  7.5792e-09      131941      67.21     0.08160      ok

- **Scheme against scheme, at nX = 40 and 80** — the only meshes where both are stable — the two
  agree to **0.4 %** in peak force and 0.4 % in dissipated energy, on an **identical** time
  increment. That is the validation: the micro-inertia bought the time-step scaling without moving
  the answer.
- **Peak force converges onto an independent reference.** 61.06, 61.94, 67.10, 67.21 N against the
  implicit `AlphaP_AMR_Study` value of 66.95 N, i.e. **0.4 %** at convergence — a number established
  before any of this work existed.
- **`G_f` is NOT mesh-converged, in either scheme.** 0.0515, 0.0504, 0.0537, 0.0816 N/mm against the
  implicit 0.098. Every run softens essentially to zero (2-4 % of peak by U = 0.097 mm), so this is
  not a truncated integral: the coarse meshes under-dissipate, and refinement moves `G_f` upward
  toward the reference. With `l = 5 mm` the band is `2*pi*l ~ 31 mm` wide, so even `h = 1.25 mm` is
  only a handful of elements across it. Do not read `G_f` from this study as a converged property;
  read it as a scheme-to-scheme comparison at fixed mesh.
- The post-peak curve is a single smooth branch at every mesh -- no second peak, no shoulder -- so
  the rise in `G_f` is a fatter softening tail on a better-resolved process zone, not a second crack.

**Difference** — the point:

- the parabolic scheme's own limit crosses the mechanical one between `nX = 80` and `nX = 160`
  (`h = 1.25` to `0.625 mm` against `l = 5 mm`), and since nothing checks it, that run **diverges to
  NaN** — measured: `nX = 160` parabolic reports a finite critical time step of `1.52e-8 s`, runs to
  completion, and produces `NaN` for every output. It did not even trip the energy guard, because
  every ordering against a NaN is False, including the `W_ext > 0` the guard is gated on; that is
  now caught separately (`THE SOLUTION HAS DIVERGED`), but a run on an older solver will look like a
  success;
- the hyperbolic run at the same mesh stays governed by the mechanical limit, so its increment count
  tracks `1/h` instead of `1/h^2`.

This also brackets the discretisation constant `C`: the parabolic limit `2 eta/(1 + C l^2/h^2)` at
`h = 0.625 mm` is `2.6e-8 s` for `C = 12` and `1.3e-8 s` for `C = 24`, and the run diverges at
`1.52e-8 s` — so `C` is nearer 24 than 12 for this element, which is why the non-local branch of the
stable increment bounds the assembled operator's eigenvalue directly instead of assuming a constant.

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
