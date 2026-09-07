**=================================================================================================
** SOLVER AND STEPS
**=================================================================================================
** 5. Solver: Abaqus uses an implicit *Dynamic step mostly as a numerical-damping stabilizer for
**    the concrete's softening response under displacement control, not because true inertial
**    dynamics matter over its (dimensionless) 1-time-unit step. EdelweissFE has no implicit-
**    dynamic (Newmark/HHT) solver, only implicit-static (NIST/NISTParallel) and explicit
**    (NEST/NED, ...static/dynamic). We use NISTParallel with fine automatic sub-stepping instead
**    (mirroring Abaqus' maxInc/minInc/maxNumInc); the GCDP material's own Duvaut-Lions
**    viscoplastic regularization (already present in the source UEL properties, "viscosity"
**    below) provides the same kind of numerical robustness Abaqus was leaning on the dynamic
**    stabilization for. This also sidesteps needing a density for the concrete (see point 1).
**
**
**=================================================================================================
** LINEAR SOLVER -- why these settings, measured on this model
**=================================================================================================
** A real, full-scale (280k-dof) engineering model -- concrete via a gradient-enhanced-damage
** Marmot material, steel fastener parts, 5 *TIE constraints, 2 penalty contact pairs, and
** geometry-driven AMR -- used here to demonstrate this repo's currently best-known linear-solver
** setup for exactly this kind of hard, ill-conditioned, path-dependent problem
** (`linsolver=blockamg`, `blockamg.json` in this directory):
**
** - `outerSolver=amgcl_lgmres`: AMGCL's own native outer Krylov solve in place of
** `scipy.sparse.linalg.gmres`, with `lgmresAlwaysReset=true` -- both now the `blockamg` default
** in this codebase, live-gated at 8/16/32 threads, faster than the previous scipy-based default
** at every thread count tested and increasingly so as thread count grows. `lgmresAlwaysReset`
** matters here specifically: the opposite setting was found (live, on this exact model) to let a
** struggling solve's poorly-conditioned recycled Krylov subspace compound into a much more
** expensive, or once, NaN, subsequent solve.
** - `useAmgclMPCCondensation=True` (`>>options` on the loading step below): threads the multi-
** point-constraint condensation this model's 5 ties require (`T^T K T + C`, every solve) via
** AMGCL's own OpenMP-threaded `product()`/`sum()` instead of single-threaded SciPy CSR routines.
** Measured ~2.4-2.6x faster than the plain expression offline, ~7.3% faster end-to-end live on
** top of the `lgmres` default above.
** - Deliberately *not* enabled: p-two-grid (`p1FieldNames`) -- measured at parity with the shipped
** default on this model (not a real margin), not worth the added complexity here.
**
** A previous attempt at this file also set a fixed, tight `outerTol=1e-8` on the theory that it was
** just an accuracy choice independent of the solver settings above -- a live smoke test on this
** exact model caught that it was not: with `outerTol=1e-8`, the first real cutback needed 963 outer
** GMRES iterations and did not converge. `outerTol` is left unset here (the adaptive Eisenstat-
** Walker forcing default, `etaMax=3e-4`) -- the same smoke test then reproduces the expected
** trajectory exactly (125 outer iterations at that same cutback). If your own model needs a tighter
** fixed tolerance, re-verify the load path against a reference run before trusting it with
** `outerSolver=amgcl_lgmres` -- this combination had not been tested at all before that finding.
**
** Smoke-tested (this exact file): both the trivial preload step and the loading step's first full
** cutback sequence (0.005 -> 0.0025 -> 0.00125 -> 0.000625, converging and advancing to the next
** increment) reproduce the expected trajectory, no NaN. Not run to completion -- maxNumInc=100000
** below is a real, unbounded run; verify the full load path yourself before relying on it.
**
** Source: AnchorPryOut.inp (Abaqus, concrete via GCDP UEL U004/C3D20R, steel via C3D8I, mortar via
** COH3D8 cohesive elements, 5 *TIE constraints, 2 penalty *Contact pairs).
**
** SIMPLIFICATIONS / MODELLING DECISIONS made during translation (please review):
*solver, solver=NISTParallel, name=theSolver
linsolver=blockamg
linsolverConfigFile=blockamg.json

**=================================================================================================
** STEPS
**=================================================================================================
** Step 1 ("zero" in the source model): trivial warm-up step at zero load, establishing consistent
** tie/contact state before loading (mirrors the source model's zero-length *Static warm-up step).
*step, solver=theSolver
maxInc=1, minInc=1, maxNumInc=1, maxIter=25, stepLength=1
>>dirichlet, name=fixLeft,        nSet=left,           field=displacement, 1=0, 2=0, 3=0
>>dirichlet, name=fixRight,       nSet=right,          field=displacement, 1=0, 2=0, 3=0
>>dirichlet, name=fixBack,        nSet=back,           field=displacement, 1=0, 2=0, 3=0
>>dirichlet, name=fixBottom,      nSet=bottom,         field=displacement, 1=0, 2=0, 3=0
>>dirichlet, name=zSymmConcrete,  nSet=z_symm_concrete, field=displacement, 3=0
>>dirichlet, name=zSymmSteel,     nSet=z_symm_steel,    field=displacement, 3=0
>>dirichlet, name=plateXFix,      nSet=plate_left_loading, field=displacement, 1=0

** Step 2 ("loading" in the source model): ramp the plate's x-displacement to 5.0 mm using an
** Abaqus SMOOTH STEP-equivalent amplitude (quintic smoothstep, zero slope/curvature at both ends).
** Increment control mirrors the source model's *Dynamic data line (5e-3, 1, 1e-08, 5e-3).
*step, solver=theSolver
maxInc=5e-3, minInc=1e-8, maxNumInc=100000, maxIter=25, criticalIter=10, stepLength=1
>>options, name=theSolver, extrapolation=off, useAmgclMPCCondensation=True
>>dirichlet, name=fixLeft,        nSet=left,           field=displacement, 1=0, 2=0, 3=0
>>dirichlet, name=fixRight,       nSet=right,          field=displacement, 1=0, 2=0, 3=0
>>dirichlet, name=fixBack,        nSet=back,           field=displacement, 1=0, 2=0, 3=0
>>dirichlet, name=fixBottom,      nSet=bottom,         field=displacement, 1=0, 2=0, 3=0
>>dirichlet, name=zSymmConcrete,  nSet=z_symm_concrete, field=displacement, 3=0
>>dirichlet, name=zSymmSteel,     nSet=z_symm_steel,    field=displacement, 3=0
>>dirichlet, name=plateXFix,      nSet=plate_left_loading, field=displacement, 1=5.0
