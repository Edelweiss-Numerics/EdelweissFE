# PLAN — Adaptive Mesh Refinement with Hanging Nodes for HEX20 Multifield

**Status:** design / scoping · **Date:** 2026-07-24 · **Target branch:** off `next_v26.11`

## 1. Goal

Adaptive `h`-refinement with hanging nodes for **HEX20 (20-node serendipity)** elements in the
**small-strain** regime, for **multifield** simulations — specifically **gradient-enhanced damage**
(`GC3D20` / `GC3D20R`, i.e. Marmot's `GeneralGradientEnhancedDisplacementFiniteElement<3,20>`).

The coupling of hanging nodes to their coarse neighbours is done with **exact linear multipoint
constraints (MPC)** — *not* mortar. State-variable transfer to refined children is *not* projected
in the first step. The bulk element and its Marmot material are **never touched**; the whole feature
is additive.

## 2. Why MPC and not mortar (justification)

The premise "serendipity is polynomial-incomplete, so we need mortar" does not hold for structured
`h`-AMR:

- An AMR interface is non-conforming but **nested**. Nestedness (coarse trace ⊆ fine trace), not
  conformity, is what an MPC needs to be *exact*.
- The QUAD8 face-trace space `span{1, x, y, x², xy, y², x²y, xy²}` is **invariant under the
  axis-aligned affine sub-maps of octree refinement** (a factor that is degree-1 in `y` can never
  generate `x²y²`). The missing `x²y²` term is missing *symmetrically on both sides*, so it never
  needs to be matched. Edges are 1D quadratics, trivially nested.
- Therefore the coarse face shape is exactly representable on the refined side, and pinning each
  hanging DOF to the coarse interpolation (`u_hanging = Σ Nₐ(ξ) uₐ`) is **exact** — this pinning *is*
  the MPC.

Mortar is only needed when exact matching is impossible: genuinely non-matching meshes, different
element orders across the interface (`hp`), or non-nested refinement — none of which occur here.

MPC is also **purely kinematic**: it needs no knowledge of the material or PDE and applies
identically to the `displacement` and `nonlocal field` DOFs. (A Nitsche coupling, by contrast, would
need the conormal flux → the constitutive tangent → per-field physics wiring.)

### Literature anchors
- Nestedness / constrained-approximation for high-order H1 in 3D: Červený, Dobrev & Kolev, *SIAM J.
  Sci. Comput.* (MFEM), 2019; Šolín et al., *J. Comput. Appl. Math.* (arbitrary-level hanging nodes,
  hp-FEM), 2014.
- Serendipity + mortar (only if we ever went that route): Lamichhane & Wohlmuth, *ESAIM: M2AN*,
  2004; Lamichhane, Stevenson & Wohlmuth, *Numer. Math.*, 2005.
- Gradient/nonlocal damage + AMR + hanging nodes (application precedent, 2D Q8/Q4): Sarkar et al.,
  *Eng. Fract. Mech.*, 2019.

## 3. Grounding facts (from the current code)

### 3.1 Element (Marmot)
`Marmot/modules/elements/GeneralGradientEnhancedDisplacementFiniteElement/include/Marmot/GeneralGradientEnhancedDisplacementFiniteElement.h`
- `template < int nDim, int nNodes, int nNonlocalVariables = 1, int nNonLocalNodes = nNodes >`.
- Registered 3D: `GC3D8`, `GC3D20` (27 QP), `GC3D20R` (8 QP) — all with `nNonLocalNodes = nNodes`
  (**equal-order serendipity** for both fields).
- Node fields: `["displacement", "nonlocal field", "strain symmetric"]`; `nDofPerNodeU = nDim`,
  `nDofPerNodeK = nNonlocalVariables (=1)`. DOFs interleaved per node via
  `getDofIndicesPermutationPattern()`.
- ⇒ every node (corner + mid-edge) carries `u` (3) + `k` (1); a hanging node's MPC uses the **same**
  QUAD8/edge weights for both fields.

### 3.2 State-variable storage (EdelweissFE wrapper)
`edelweissfe/elements/marmotelement/element.pyx`
- Single flat Marmot-owned buffer: `nStateVars = getNumberOfRequiredStateVars()`, `_stateVars`
  (converged) + `_stateVarsTemp` (trial); `assignStateVars(&_stateVarsTemp[0], nStateVars)`.
- Read: `getResultArray(result, quadraturePoint)` → `getStateView(name, qp)` (per-QP named views).
- **Write: only `setInitialCondition(stateType, values)`** → `marmotElement.setInitialConditions(...)`.
  No generic "set arbitrary internal state at QP k" API is currently exposed.
- QPs live at Gauss abscissae in the element interior (2×2×2 for `GC3D20R`, one per octant),
  independent of the serendipity node layout.

### 3.3 Constraints (EdelweissFE)
`edelweissfe/constraints/base/constraintbase.py`, `equalvaluelagrangian.py`
- Mechanism is **Lagrange-multiplier augmentation**: a constraint requests extra scalar variables via
  `getNumberOfAdditionalNeededScalarVariables()` / `assignAdditionalScalarVariables()` and augments
  the system in `applyConstraint(U_np, dU, PExt, K, timeStep)`; multipliers are appended after nodal
  DOFs.
- **There is no DOF-elimination / condensation path in this mainline** (that lives on a separate
  surface-tie branch).
- `equalvaluelagrangian` (ties `nNodes` values equal via `nNodes−1` multipliers, `g = u_slave −
  u_master`) is the direct template for a weighted hanging-node MPC (`g = u_slave − Σ wₐ u_masterₐ`).

### 3.4 Surfaces & sets (EdelweissFE)
`edelweissfe/surfaces/entitybasedsurface.py`
- `EntityBasedSurface(dict)` = `{ localFaceId : <element set> }` — Abaqus element-based surface,
  i.e. a bag of `(element, local-face-id)` pairs.
- Consumed by `edelweissfe/stepactions/distributedload.py` (iterates faces → elements →
  `computeDistributedLoad(faceId, …)`); produced by generators (`boxgen`, `pipegen`,
  `planerectquad`) and `*surface` in `abqmodelconstructor.py`. Stored in `models/femodel.py`
  (`self.surfaces`, `self.elementSets`, `self.nodeSets`).
- ⇒ refining an element invalidates every surface/set that references it.

## 4. Locked-in design decisions

1. **Coupling:** exact linear **MPC** via a new Lagrange-multiplier `Constraint` (template:
   `equalvaluelagrangian.py`). Condensation → SPD is a deferred optimization (§6, WS-J), not first cut.
2. **Refinement:** octree bisection, **2:1 balance**, **refine-only** first (coarsening later).
3. **State transfer:** tiered — virgin init first, admissible nearest-QP block-copy as the real
   target (§5.6). No field projection unless a benchmark demands it.
4. **Non-intrusive:** bulk element / material untouched; all logic additive in a new
   `edelweissfe/adaptivity/` package plus one new constraint module.

## 5. Workstreams

### WS-A — Mesh hierarchy & refinement data structure *(new `edelweissfe/adaptivity/`)*
Element tree (parent ↔ 8 children, level), node registry (mint/dedupe shared nodes), hanging-node
registry. Consumes/extends `models/femodel.py`.

### WS-B — HEX20 subdivision
One HEX20 → 8 children; new node coordinates via the **parent isoparametric map**
`x = Σ Nₐ(ξ) xₐ` (honours curved edges/faces). Emits the fixed **node-** and **face-inheritance
tables** used by WS-D and WS-K.

### WS-C — Hanging-node classification + 2:1 balance
Classify each new node: interior (free DOF), **edge-hanging** (masters = 3 coarse edge nodes),
**face-hanging** (masters = 8 coarse QUAD8 face nodes). Enforce 2:1 balance so every master is a
genuine free node ⇒ **flat constraints, no chains**.

### WS-D — MPC generation (field-aware; uniform for `GC3D20`)
Per hanging node, emit `g = u_slave − Σ Nₐ(ξ_slave) u_masterₐ`. For `GC3D20/20R` the same weights
cover `displacement` and `nonlocal field` (equal-order), so one constraint per hanging node covers
all its DOFs. Keep the generator field-aware to support a future reduced-nonlocal variant.

### WS-E — Constraint wiring
New `edelweissfe/constraints/hangingnode.py` subclassing `ConstraintBase`; reuse the scalar-variable
allocation + `applyConstraint` path (already solved by existing solvers). Feeds
`numerics/dofmanager.py` and CSR assembly.

### WS-F — State-variable transfer (tiered) — see §5.6
- **F0 (bootstrap):** virgin init via existing `setInitialCondition`.
- **F1 (target):** admissible nearest-QP block-copy (new element hook + Marmot buffer-layout probe).
- **F2 (optional):** field projection, only if F1 proves too rough.

### WS-G — Marking / driver *(expression on fieldOutputs)*
The refinement criterion is a **user math expression evaluated on fieldOutputs**, reusing the
existing stack — *not* a hard-coded threshold:
- `edelweissfe/utils/fieldoutput.py` already provides `perElement` field outputs (`elSet`, `result`,
  `quadraturePoint`, optional `f(x)`), which yield a per-element array (e.g. damage / nonlocal
  equivalent strain).
- `edelweissfe/utils/math.py` (`createMathExpression` / `createFunction`, symbol `x`) is the
  expression engine; `outputmanagers/conditionalstop.py` ("stop when an expression becomes true") and
  `monitor.py` / `meshplot.py` (`entry["f(x)"] = createMathExpression(fx)`) are direct precedents.

Design: a new marker keyword references one or more `perElement` fieldOutputs and a
`createMathExpression`; elements where the expression is truthy / over threshold are marked for
refinement. The marker maps the fieldOutput's `elSet` back to element handles. Mirrors
`conditionalstop` (expression → bool) + `monitor` (fieldOutput + `f(x)`). Refine-only first;
a-posteriori error estimator later.

### WS-H — Solver-loop integration *(de-risked — reuse the contact-branch rebuild)*
Adapt **between increments** (mesh frozen within a Newton solve). The mid-run equation-system rebuild
**already exists** on `matthias/deformable_surface_contact` (merging soon) and is model-wide, not
contact-specific:
- `ConstraintBase.updateConnectivity(model) -> bool` is called for every constraint at the **start of
  each increment**; if any returns `True` (or first increment), `nonlinearimplicitstatic.py` (NIST)
  rebuilds `DofManager(model.nodeFields, model.scalarVariables, model.elements, model.constraints,
  model.nodeSets)`, the VIJ/CSR structure, and `U/P/dU`, then repopulates `U` from
  `model.nodeFields`; extrapolation is suppressed that increment (`prevTimeStep = None`). Comment
  notes it mirrors EdelweissMeshfree's `NonlinearQuasistaticSolver`.
- `ConstraintBase.acceptLastState()` (called from `femodel.advanceToTime`) lets stateful entities
  promote converged state; our kinematic MPCs use the no-op default.

**AMR integration:**
- *Quickest (no solver change):* register the AMR manager as a **pseudo-constraint** in
  `model.constraints`; its `updateConnectivity` marks → refines → mutates the model → returns `True`.
- *Cleaner (small solver addition, recommended post-merge):* a sibling adaptivity-manager call
  alongside the constraint loop (AMR isn't semantically a constraint).

**Strict ordering inside `updateConnectivity` (before returning `True`):** mark (on last-converged
fieldOutputs) → subdivide → create nodes → **write interpolated nodal values into `model.nodeFields`**
→ transfer QP state (WS-F) → update sets/surfaces/constraints (WS-K) → initialize children
(material/section). The solver reads `model.nodeFields` into `U` *immediately after* the rebuild, so
all of the above must be done first.

**Dependencies / gaps:**
- **Branch dependency:** build AMR on top of `matthias/deformable_surface_contact` (stacked on
  `discrete_rigid_body_contact`), or wait for its merge into `next_v26.11` — not off current
  `next_v26.11`.
- **Solver coverage:** only NIST (implicit static) has the rebuild; arc-length / parallel / dynamic
  do not. Gradient-damage softening may want arc-length → replicate the NIST diff there (follow-on).
- **fieldOutput access:** `updateConnectivity(model)` gets only `model`; the marker (WS-G) needs
  fieldOutput values (fieldOutputController) — wire them through or stash needed per-element results
  on the model.

### WS-I — Input API + Ensight time-varying geometry
- Keyword to enable adaptivity (max level, marker expression, cadence) via
  `utils/inputfileparser.py`.
- **Ensight changing geometry** — the machinery already exists; the current output just doesn't
  exercise it. `outputmanagers/ensight.py` already uses **two time sets**:
  `writeGeometryTrendChunk(..., timeAndFileSetNumber=1)` for geometry and
  `writeVariableTrendChunk(..., timeAndFileSetNumber=2)` for variables, via `EnsightGeometryTrend` +
  `EnsightTimeSet`. But `initializeJob()` writes geometry **once** to a *static* timeset
  (`self.staticTAndFSetNumber`), while variables are written every increment in
  `finalizeIncrement()`.
  - **Plan:** give geometry its **own advancing timeset** (the "geometry time series"); on each
    adaptation event, rebuild geometry parts from the current mesh (`_createGeometryParts`) and emit
    a fresh `.geo` chunk via `writeGeometryTrendChunk` at the current geometry-time. Between
    adaptations the geometry timeset holds; variables (timeset 2) advance every increment and resolve
    against the most recent geometry (standard Ensight Gold "changing connectivity" mode).
  - **Ordering constraint:** after an adaptation the node/element enumeration changes, so the
    variable trends written thereafter must use the new geometry's ordering — sequence
    geometry-chunk-then-variables at each adaptation step.

Test placement: per-feature standalone tests under `testfiles/marmot/` (see §6.1).

### WS-K — Topological consistency under refinement *(prerequisite for M0)*
Route all container updates through one consistency manager, driven by WS-B's inheritance tables:

| Container | Rule on `refine(E)` |
|---|---|
| `model.elements` | remove `E`, add 8 children |
| element sets (⇒ material/section) | children **inherit** `E`'s memberships (else children are inert) |
| surfaces | parent `(E, f)` → 4 child `(child, f')` pairs (reverse index `element → surfaces`) |
| node sets | new edge/face nodes inherit membership from the parent entity they subdivide |
| sections | re-resolved via element-set inheritance |

Distributed load on refined child faces is **statically equivalent** to the parent face (children are
full HEX20 with their own `computeDistributedLoad`); loads landing on hanging (slave) nodes are
transmitted to masters through the multipliers. Coarsening later reuses this machinery inverted.

### WS-J — *(deferred)* DOF condensation
Optional elimination path for an SPD, smaller system (port the surface-tie elimination approach).

## 5.6 State-variable transfer detail

Octree makes each child exactly one octant of the parent parametric cube, so every child QP has a
known location in the parent's parametric space.

- **F0 — virgin:** children created like fresh elements (`initializeElement` → zeroed →
  `setInitialCondition("sdvini", …)`). Discards history ⇒ sound only if refining *ahead of* the
  process zone. Zero new API.
- **F1 — nearest-QP block copy (recommended):** parent and children share element type + material, so
  each QP state block has identical size `s`. Copy the parent QP block **verbatim** into child QP
  blocks under the octree QP-index map. For `GC3D20R` (8 QP, one per octant) this is a clean
  broadcast: parent octant-QP → all 8 QPs of the corresponding child (piecewise constant). Copies an
  **already-admissible** state (no off-yield / inconsistent-loading artefacts, unlike interpolation).
  Needs: a small element hook "assign state from parent with QP remap" + confirm Marmot's flat buffer
  is `nQP × s` (+ optional element-level tail to skip).
- **F2 — projection:** extrapolate parent QP values to a field, resample at child QPs; expensive and
  admissibility-risky for internal variables. Last resort.

Note: **nodal DOF values** (`u`, `k`) for new nodes come from parent isoparametric interpolation —
exact, continuous, free, admissible — and are a *separate* transfer from the QP internal state.

## 6. Milestones

| # | Deliverable | Verification |
|---|---|---|
| **M0** | Single-field (`u`-only) HEX20, one-level refine, linear static; WS-A/B/C/D/E/K | **Patch test:** quadratic field reproduced exactly across the interface (MPC exactness). **Distributed load on a refined surface** reproduces exact stress state (WS-K + hanging-node loads). |
| **M1** | 2:1 balance + multi-level | Balance invariant holds; patch test exact at level jumps |
| **M2** | Two-field gradient damage (`GC3D20R`); state F0 | Patch test on `u` *and* `k`; single per-node constraint (equal-order) verified |
| **M3** | Damage-driven adaptation across load steps; state F1 | **Adaptive vs. uniform-fine** benchmark (notched specimen): load–displacement + damage profile match within tol |
| **M4** *(opt.)* | Coarsening, F2 projection, WS-J condensation | Reduced sensitivity to refinement timing; system-size/conditioning gains |

## 6.1 Testing — one dedicated standalone test per feature

Every feature gets its **own** self-contained test (not only milestone-level checks), following the
`run_tests_edelweissfe` convention: a directory under `testfiles/marmot/<Feature>/` with `test.inp`
+ reference data (`U.ref`; or an exported fieldOutput/mesh artifact where displacement is not the
natural quantity). Each test must run and pass in isolation.

| Feature (WS) | Standalone test | What it pins |
|---|---|---|
| Subdivision (B) | `AMR_Hex20Subdivide` | 8 children, correct shared-node dedupe + isoparametric coords |
| Hanging classification + 2:1 (C) | `AMR_Balance2to1` | edge/face classification; balance propagation mesh |
| MPC exactness — 1 field (D/E) | `AMR_PatchTestU` | quadratic `u` reproduced to machine precision across a 2:1 interface |
| MPC exactness — multifield (D) | `AMR_PatchTestUK` | `u` *and* nonlocal `k` exact (equal-order, single per-node constraint) |
| Surface consistency (K) | `AMR_RefinedSurfaceLoad` | pressure on a refined face → exact stress; loads on slave nodes transmitted via multipliers |
| Set/section inheritance (K) | `AMR_SectionInheritance` | children carry parent material/section (would fail if inert) |
| Marker expression (G) | `AMR_MarkerExpression` | fieldOutput-expression marks the intended elements only |
| State transfer F0 (F) | `AMR_StateVirgin` | children initialise to virgin state |
| State transfer F1 (F) | `AMR_StateBlockCopy` | nearest-QP block copy reproduces parent history admissibly |
| Ensight geometry trend (I) | `AMR_EnsightChangingGeo` | `.case` + per-adaptation `.geo` chunks on the geometry timeset; variables resolve to correct geometry |
| End-to-end adaptation (H) | `AMR_AdaptiveVsUniform` | adaptive run matches uniform-fine reference |

New Marmot-side hooks (F1 block copy, any element API additions) additionally get a Marmot ctest
under the relevant module's `tests/`.

## 7. Risks (ranked)

1. ~~Dynamic DOF layout (WS-H)~~ **RESOLVED** — the model-wide mid-run rebuild exists on
   `matthias/deformable_surface_contact` (`updateConnectivity` → full `DofManager` rebuild in NIST;
   see WS-H). Residual risk is now just the *dependencies/gaps* noted there (branch merge, solver
   coverage beyond NIST, fieldOutput wiring), not the core capability.
2. **Saddle-point growth** — one multiplier per constrained hanging DOF; correct but watch
   conditioning/solver cost at high refinement. Mitigation = WS-J.
3. **Material/section inheritance (WS-K)** — not optional polish: children are inert until element-set
   membership propagates ⇒ WS-K is a prerequisite for M0.
4. **2:1 balance guarantees flat constraints** — verify the balancer never leaves a master that is
   itself a slave.
5. **Marmot state buffer layout** — F1 assumes clean `nQP × s` blocking; confirm and expose block
   boundaries.

## 8. Validation spine

The **patch test** is the gate: a manufactured quadratic (`u`) / quadratic (`k`, equal-order) field
reproduced to machine precision across a refined interface ⇒ MPC exactness proven, the rest is
engineering. Then the loaded-refined-surface stress test (WS-K), and finally adaptive-vs-uniform-fine
physics agreement on a gradient-damage benchmark.

## 9. Remaining decisions

- **Refinement generality:** commit to structured octree + 2:1 first (assumed), or design for
  arbitrary non-matching from the start? *(Recommend: structured first.)*
- **Marking criterion:** damage/nonlocal threshold (assumed first cut) vs. an a-posteriori error
  estimator.
- **Condensation (WS-J):** implement early for SPD, or defer until saddle-point conditioning bites?
  *(Recommend: defer.)*
- **First verification vehicle:** which existing `testfiles/marmot/` gradient-damage case to fork for
  the M3 benchmark.
