# Translation notes — Abaqus `AnchorPryOut.inp` to EdelweissFE

This example is a translation of a real Abaqus/Standard model. What follows is the provenance and
the modelling decisions taken during that translation, carried over verbatim from the monolithic
input file this example replaced, so they are not lost now that the deck is split into includes.

Read this before changing the physics. Several of the choices below are not free parameters.

```
 EdelweissFE translation of AnchorPryOut.inp (Abaqus/Standard, UEL)

 A real, full-scale (280k-dof) engineering model -- concrete via a gradient-enhanced-damage
 Marmot material, steel fastener parts, 5 *TIE constraints, 2 penalty contact pairs, and
 geometry-driven AMR -- used here to demonstrate this repo's currently best-known linear-solver
 setup for exactly this kind of hard, ill-conditioned, path-dependent problem
 (`linsolver=blockamg`, `blockamg.json` in this directory):

  - `outerSolver=amgcl_lgmres`: AMGCL's own native outer Krylov solve in place of
    `scipy.sparse.linalg.gmres`, with `lgmresAlwaysReset=true` -- both now the `blockamg` default
    in this codebase, live-gated at 8/16/32 threads, faster than the previous scipy-based default
    at every thread count tested and increasingly so as thread count grows. `lgmresAlwaysReset`
    matters here specifically: the opposite setting was found (live, on this exact model) to let a
    struggling solve's poorly-conditioned recycled Krylov subspace compound into a much more
    expensive, or once, NaN, subsequent solve.
  - `useAmgclMPCCondensation=True` (`>>options` on the loading step below): threads the multi-
    point-constraint condensation this model's 5 ties require (`T^T K T + C`, every solve) via
    AMGCL's own OpenMP-threaded `product()`/`sum()` instead of single-threaded SciPy CSR routines.
    Measured ~2.4-2.6x faster than the plain expression offline, ~7.3% faster end-to-end live on
    top of the `lgmres` default above.
  - Deliberately *not* enabled: p-two-grid (`p1FieldNames`) -- measured at parity with the shipped
    default on this model (not a real margin), not worth the added complexity here.

 A previous attempt at this file also set a fixed, tight `outerTol=1e-8` on the theory that it was
 just an accuracy choice independent of the solver settings above -- a live smoke test on this
 exact model caught that it was not: with `outerTol=1e-8`, the first real cutback needed 963 outer
 GMRES iterations and did not converge. `outerTol` is left unset here (the adaptive Eisenstat-
 Walker forcing default, `etaMax=3e-4`) -- the same smoke test then reproduces the expected
 trajectory exactly (125 outer iterations at that same cutback). If your own model needs a tighter
 fixed tolerance, re-verify the load path against a reference run before trusting it with
 `outerSolver=amgcl_lgmres` -- this combination had not been tested at all before that finding.

 Smoke-tested (this exact file): both the trivial preload step and the loading step's first full
 cutback sequence (0.005 -> 0.0025 -> 0.00125 -> 0.000625, converging and advancing to the next
 increment) reproduce the expected trajectory, no NaN. Not run to completion -- maxNumInc=100000
 below is a real, unbounded run; verify the full load path yourself before relying on it.

 Source: AnchorPryOut.inp (Abaqus, concrete via GCDP UEL U004/C3D20R, steel via C3D8I, mortar via
 COH3D8 cohesive elements, 5 *TIE constraints, 2 penalty *Contact pairs).

 SIMPLIFICATIONS / MODELLING DECISIONS made during translation (please review):

  1. Concrete: Abaqus UEL U004 (C3D20R host, GCDP material) -> Marmot element GC3D20R (gradient-
     enhanced hex20, reduced integration) with Marmot's native "gcdp" material. Property order
     verified against Marmot/modules/materials/GCDP/src/GCDP.cpp (19 properties; the Abaqus UEL's
     20th/21st properties, viscosity and density, collide at material property index 18 in the
     current Marmot GCDP source -- see the comment above the material block below).

  2. Steel (anchor/plate/washer/nut): Abaqus C3D8I (incompatible-modes hex8) -> Marmot C3D8
     (standard, fully-integrated hex8). Marmot does not currently register an incompatible-modes/
     enhanced-assumed-strain hex8 under any element name. This is slightly stiffer in bending than
     the source model; not expected to matter here since these parts stay close to linear-elastic
     and are not the focus of the analysis, but worth knowing if plate bending becomes significant.

  3. Mortar bond layer: Abaqus models this with COH3D8 cohesive elements, response=TRACTION
     SEPARATION, an elastic-only (no debonding) traction law, and a cylindrical stack direction.
     EdelweissFE has no cohesive element. Inspecting the actual mesh geometry shows the COH3D8
     "thickness"/stack direction here is circumferential, not radial (the mortar annulus is
     meshed as many thin circumferential wedges) -- so in the source model the mortar's own bulk
     stiffness contributes essentially nothing to the radial/axial anchor-concrete load path,
     which is instead carried entirely by the ANCHOR_TO_MORTAR / MORTAR_TO_CONCRETE ties. We
     therefore replace the COH3D8 wedges with ordinary linear-elastic C3D8 solid elements (same
     connectivity) using generic grout/mortar properties (E=30000, nu=0.2, comparable to the
     concrete). This is very likely a *more* compliant/realistic bulk representation than the
     source model's cohesive-only element, not less -- but it is a genuine judgement call, so
     please swap in a real mortar/grout modulus if you have one.

  4. Node numbering: EdelweissFE has no Abaqus Part/Instance/Assembly concept -- one flat node and
     element label space. Concrete (mesh/concrete.inp, nodes 1-58569) and steel (mesh/steel.inp,
     nodes 4489-19732) node numbers collide, so mesh/steel_edelweissfe.inp is a copy of
     mesh/steel.inp with every node label (and every reference to one, in *ELEMENT connectivity
     and *NSET lists) offset by +100000. Element labels do NOT collide (concrete: 4004-15575,
     steel: 1-4003) and were left untouched. *ELSET/*SURFACE blocks reference element labels only,
     also untouched. The *NSET "z_symm" is defined in both parts and was renamed to
     "z_symm_concrete"/"z_symm_steel" to avoid one silently overwriting the other. See
     mesh/concrete_edelweissfe.inp / mesh/steel_edelweissfe.inp for the exact (small, mechanical)
     diffs against the original Abaqus mesh files -- geometry/connectivity/sets are otherwise
     byte-for-byte identical.

  5. Solver: Abaqus uses an implicit *Dynamic step mostly as a numerical-damping stabilizer for
     the concrete's softening response under displacement control, not because true inertial
     dynamics matter over its (dimensionless) 1-time-unit step. EdelweissFE has no implicit-
     dynamic (Newmark/HHT) solver, only implicit-static (NIST/NISTParallel) and explicit
     (NEST/NED, ...static/dynamic). We use NISTParallel with fine automatic sub-stepping instead
     (mirroring Abaqus' maxInc/minInc/maxNumInc); the GCDP material's own Duvaut-Lions
     viscoplastic regularization (already present in the source UEL properties, "viscosity"
     below) provides the same kind of numerical robustness Abaqus was leaning on the dynamic
     stabilization for. This also sidesteps needing a density for the concrete (see point 1).

  6. The *SURFACE blocks below are the original Abaqus S1..S6 face-numbered elsets, imported
     unmodified and fed straight into "*modelGenerator, generator=surfaceElementGenerator".
     EdelweissFE's own face-numbering convention for this generator is documented as NOT being
     the general Abaqus S1..S6 convention (see edelweissfe/generators/surfaceelementgenerator.py)
     -- checked empirically (facet span/shape sanity check + closest-point-distance profiling
     for every tie pair, see point 7): all 5 generated facet surfaces are geometrically sane
     (small, local, correctly-shaped facets; matching bounding boxes / radii between tie pairs),
     so this coincidence holds for this mesh's Marmot-compatible node ordering.

  7. HISTORY (resolved): concrete_to_mortar (the concrete borehole wall) originally extended the
     full 240 mm slab height in the exported mesh, instead of stopping at the real 80 mm bonded
     embedment depth (mortar_to_concrete's extent) -- a Cubit geometry-generation bug (see
     mesh/Pryout_elastic_anchorwashernut.py: a webcut used to extend the borehole's radial
     partition through the solid concrete below the hole, for meshing reasons, ended up
     coincident with the real borehole wall over the region where they overlap, and ACIS fused
     them into one over-extended, mis-scoped surface). Fixed at the source and the mesh
     re-exported; concrete_to_mortar now matches mortar_to_concrete's 80 mm extent exactly, and
     tieMortarToConcrete correctly ties every one of its 373 slave nodes (0 untied).

     This originally exposed a real, separate bug in EdelweissFE's tie constraint too (fixed
     upstream, see EdelweissFE/edelweissfe/constraints/tie.py): omitting positionTolerance used
     to tie every slave node unconditionally, however far the closest master point was, which
     force-snapped the (at the time, spuriously unbonded) borehole's slave nodes onto the mortar
     surface's edge and tore the surrounding concrete elements. That fix is a real improvement
     independent of the Cubit fix (it also matters for any genuinely partial-bond-length model,
     just not this one anymore) and is why no explicit positionTolerance appears on any *constraint,
     type=tie below: the default (positionToleranceFactor=0.25 times the master surface's mean
     facet size, independent of adjust) is enough on its own.

     VERIFY: every *constraint, type=tie below publishes "<name>_tied" / "<name>_untied" node
     sets, which show up automatically as their own Ensight parts (NSET_..., no *fieldOutput
     needed) -- with the fix, none of the 5 ties should have a published "_untied" set at all
     (empty ones are deliberately not published, see edelweissfe/constraints/tie.py).
```
