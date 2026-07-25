# Review Report: Adaptive Mesh Refinement (AMR) with Hanging Nodes for HEX20 Elements

**Repository / Branch:** `EdelweissFE` / `feat/amr-hanging-nodes`
**Date:** July 25, 2026
**Status:** In Development (Milestone M3 Prototype)
**Target Application:** Small-strain multifield fracture simulations (gradient-enhanced damage `GC3D20` / `GC3D20R`)

---

## 1. Executive Summary

An in-depth code review was conducted for the adaptive mesh refinement (AMR) feature implemented on branch `feat/amr-hanging-nodes`. The implementation provides an end-to-end framework for dynamic $h$-refinement of 20-node serendipity hexahedral (`HEX20`) elements in small-strain multifield (displacement + nonlocal damage) finite-element formulations.

The core architecture follows the design specification outlined in [`PLAN_AMR.md`](file:///home/matthias/constitutive_modeling/next_v2611/EdelweissFE/PLAN_AMR.md):
- **Exact Kinematic Multi-Point Constraints (MPC):** Non-conforming 2:1 octree boundaries are coupled using exact serendipity trace interpolation (QUAD8 face traces and 3-node quadratic edge traces). Constraints are enforced via **master-slave DOF elimination**, avoiding Lagrange multiplier growth and saddle-point ill-conditioning.
- **8-Way Octree Bisection & 2:1 Balancing:** Implements recursive 8-child subdivision for HEX20 elements and enforces 2:1 face-adjacency balance via a spatial broad phase.
- **Flexible Marking:** Expression-based element marking (`markElements`) driven by user-defined math expressions evaluated on quadrature-point results (e.g. nonlocal damage or equivalent strain).
- **Admissible History Transfer (Tier F1):** Nearest-quadrature-point block copy (`transferStateNearestQp`) for material internal state buffers.
- **Dynamic Solver Integration:** Zero-DOF `Constraint` manager (`adaptivity.py`) hooked into `updateConnectivity(model)` to trigger equation system rebuilds mid-simulation.

**Overall Rating:** The implementation is well-structured, mathematically sound, and non-intrusive. However, **two critical bugs** (missing nodal value interpolation on new nodes and un-synchronized model surfaces) and **three major design refinements** (edge curvature handling, state transfer coordinates, and boundary BC set updates) must be addressed before merging into `next_v26.11`.

---

## 2. Architectural & Technical Assessment

### 2.1 Strengths & Core Highlights

1. **Nested Serendipity Exactness (Mathematical Premise Verified):**
   The QUAD8 face-trace polynomial space $\text{span}\{1, x, y, x^2, xy, y^2, x^2y, xy^2\}$ is invariant under 2D axis-aligned affine sub-maps of octree bisection. The missing $x^2y^2$ term is missing symmetrically on both sides of the interface. Thus, kinematic MPC pinning ($u_s = \sum_a N_a(\xi_s) u_{m_a}$) is mathematically **exact**, requiring no mortar integration or conormal flux calculations.

2. **Master-Slave DOF Elimination:**
   Multi-level hanging node chains are pre-flattened into independent master nodes in `AdaptiveMesh.hanging_mpc_records()` by recursive linear weight substitution. This satisfies the strict non-chained requirement of `MultiPointConstraintBase`, maintaining symmetric positive-definite (SPD) global system matrices without adding scalar multiplier variables.

3. **Non-Intrusive Material & Bulk Element Architecture:**
   Bulk elements and Marmot constitutive models are untouched. State variable buffers are exposed cleanly via abstract `getStateVars()` and `setStateVars()` methods on `BaseElement` and Cython `Element` wrappers.

4. **Multi-Field Support (`u` + `k` Equal-Order Serendipity):**
   For equal-order elements like `GC3D20R`, displacement ($u_x, u_y, u_z$) and nonlocal damage ($k$) share identical nodal locations. `HangingNodeConstraint` automatically detects all active fields on master/slave nodes and applies the same kinematic weights across all field components.

---

## 3. Findings, Bugs & Deficiencies

The findings below are categorized by severity.

### 3.1 Critical Bugs (Must Fix for Correctness & Performance)

#### Finding 1: Missing Nodal Field Interpolation for Newly Minted Nodes
* **Location:** [`edelweissfe/constraints/adaptivity.py`](file:///home/matthias/constitutive_modeling/next_v2611/EdelweissFE/edelweissfe/constraints/adaptivity.py#L177-L236) (`_materialize()`)
* **Description:** When an element is subdivided, new interior and hanging nodes are instantiated and added to `model.nodes`. However, their nodal field solution entries ($u_x, u_y, u_z, k$) in `model.nodeFields` are never populated with interpolated values from the parent element. When `model._prepareVariablesAndFields()` is called, new node entries default to **zero**.
* **Impact:** At the start of the increment following refinement, newly created nodes possess $u=0$ and $k=0$, creating a severe artificial displacement/damage jump. In `examples/WinklerL_AMR/amr_run.log`, every refinement event causes a huge residual spike ($\|R\|_\infty > 25.0$) requiring 8–10 Newton iterations to resolve.
* **Remedy:** In `_materialize()`, interpolate nodal values for all new nodes using the parent isoparametric map ($u_{\text{new}} = \sum_a N_a(\xi_{\text{new}}) u_{\text{parent}, a}$) and write them into `model.nodeFields` before rebuilding equation system vectors.

#### Finding 2: `model.surfaces` Is Not Synchronized After Element Subdivision
* **Location:** [`edelweissfe/constraints/adaptivity.py`](file:///home/matthias/constitutive_modeling/next_v2611/EdelweissFE/edelweissfe/constraints/adaptivity.py#L177-L236) (`_materialize()`) and [`edelweissfe/adaptivity/refinement.py`](file:///home/matthias/constitutive_modeling/next_v2611/EdelweissFE/edelweissfe/adaptivity/refinement.py#L280-L286)
* **Description:** `AdaptiveMesh.refine()` updates its internal surface registry `self.surfaces` (replacing parent element-face pairs `(peid, faceID)` with child pairs `(child_eid, faceID)`). However, `_materialize()` in `adaptivity.py` never updates `model.surfaces` in the live `FEModel`.
* **Impact:** Any surface load (`*dload` / `distributedload.py`) or surface output manager referencing a surface whose elements were refined will attempt to access deleted parent element IDs in `model.elements`, raising a runtime `KeyError`.
* **Remedy:** In `_materialize()`, rebuild `model.surfaces` by mapping updated surface pairs from `self._mesh.surfaces`.

---

### 3.2 Major Deficiencies (Numerical & Geometric Issues)

#### Finding 3: Linear Edge Assumption for Curved / Distorted Elements
* **Location:** [`edelweissfe/adaptivity/refinement.py`](file:///home/matthias/constitutive_modeling/next_v2611/EdelweissFE/edelweissfe/adaptivity/refinement.py#L124-L135) (`_lies_on_segment`) and [`hanging_weights`](file:///home/matthias/constitutive_modeling/next_v2611/EdelweissFE/edelweissfe/adaptivity/refinement.py#L54-L66)
* **Description:** `_lies_on_segment` determines whether a candidate node lies on an element edge by checking collinearity with the two corner nodes `ca` and `cb`. `hanging_weights` for edge hanging nodes projects points linearly onto the chord `[ca, cb]`.
* **Impact:** For elements with genuine quadratic edge curvature (where the midside node `em` is offset from the corner chord), candidate edge hanging nodes will fail classification or produce inaccurate MPC weights.
* **Remedy:** Extend edge containment and parameter extraction to 3-node quadratic 1D geometry taking midside node `em` into account.

#### Finding 4: Euclidean Spatial Distance vs. Reference Parametric Coordinates in State Transfer
* **Location:** [`edelweissfe/adaptivity/statetransfer.py`](file:///home/matthias/constitutive_modeling/next_v2611/EdelweissFE/edelweissfe/adaptivity/statetransfer.py#L81) (`transferStateNearestQp`)
* **Description:** Child quadrature point states are inherited from the parent QP that minimizes Euclidean physical distance $\|x_{\text{child\_qp}} - x_{\text{parent\_qp}}\|$.
* **Impact:** For elements with high aspect ratios (e.g. 10:1 elongated hexes) or skewed geometry, Euclidean distance can select a parent QP from a neighboring octant rather than the geometrically corresponding octant.
* **Remedy:** Map child QP positions into the parent element's reference parametric cube $[-1, 1]^3$ (or use octant index mapping) to select the parent QP deterministically.

#### Finding 5: Boundary Dirichlet BC Node Set Tracking
* **Location:** [`edelweissfe/constraints/adaptivity.py`](file:///home/matthias/constitutive_modeling/next_v2611/EdelweissFE/edelweissfe/constraints/adaptivity.py#L96-L103) and [`L214-L219`](file:///home/matthias/constitutive_modeling/next_v2611/EdelweissFE/edelweissfe/constraints/adaptivity.py#L214-L219)
* **Description:** Step actions (e.g. `DirichletBC`) cache target node/DOF indices during step initialization. When `_materialize()` updates `model.nodeSets` with new boundary nodes, step actions are not notified.
* **Impact:** If refinement reaches a boundary with active Dirichlet BCs, newly created boundary nodes will not have boundary conditions applied unless step actions re-initialize their target node/DOF indices upon `updateConnectivity()` rebuild.
* **Remedy:** Trigger a step action re-indexing hook when `updateConnectivity()` returns `True`.

---

### 3.3 Minor Code Quality & Robustness Issues

#### Finding 6: Silent Non-Convergence in `bilinear_inverse`
* **Location:** [`edelweissfe/adaptivity/geometry.py`](file:///home/matthias/constitutive_modeling/next_v2611/EdelweissFE/edelweissfe/adaptivity/geometry.py#L88-L103) (`bilinear_inverse`)
* **Description:** If Newton iteration in `bilinear_inverse` does not reach `tol=1e-13` within `itmax=50`, the loop exits silently without raising an error or warning.
* **Remedy:** Raise a `RuntimeError` if Newton iteration fails to converge.

#### Finding 7: Redundant State Buffer Allocation in Cython Wrapper
* **Location:** [`edelweissfe/elements/marmotelement/element.pyx`](file:///home/matthias/constitutive_modeling/next_v2611/EdelweissFE/edelweissfe/elements/marmotelement/element.pyx#L321) (`getStateVars`)
* **Description:** `getStateVars()` returns `np.asarray(self._stateVars).copy()`. In `statetransfer.py`, this copy is modified and immediately passed to `setStateVars()`, creating unnecessary allocations per child element.

---

## 4. Summary of Verification & Test Coverage

| Test Case | Location | Status | Scope / Verification |
|---|---|---|---|
| `AMR_Hex20Subdivide` | `testfiles/marmot/AMR_RefineHex20/` | **Pass** | 8-child subdivision & coordinate deduplication |
| `AMR_Balance2to1` | `testfiles/marmot/AMR_Balance2to1/` | **Pass** | 2:1 face-balance propagation |
| `AMR_PatchTestU` | `testfiles/marmot/AMR_PatchTestU/` | **Pass** | Single-field displacement MPC exactness |
| `AMR_TwoFieldGC3D20R` | `testfiles/marmot/AMR_TwoFieldGC3D20R/` | **Pass** | Multi-field ($u$ and $k$) equal-order MPC |
| `AMR_RefinedSurfaceLoad` | `testfiles/marmot/AMR_RefinedSurfaceLoad/` | **Pass** | Load transmission through hanging multipliers |
| `WinklerL_AMR` | `examples/WinklerL_AMR/` | **Functional** | End-to-end dynamic fracture simulation (suffers from Finding 1 residual spikes) |

---

## 5. Recommended Action Plan

1. **Fix Critical Nodal Field Interpolation (Finding 1):**
   Implement shape function interpolation for $u$ and $k$ on new nodes in `_materialize()`. This will immediately eliminate Newton iteration spikes in dynamic AMR runs.
2. **Synchronize Model Surfaces (Finding 2):**
   Add surface dictionary updates in `_materialize()` to prevent runtime crashes during surface load evaluations.
3. **Refine State Transfer to Parametric Coordinates (Finding 4):**
   Replace physical distance matching with parametric octant indexing.
4. **Generalize Edge Curved Geometry (Finding 3):**
   Support 3-node quadratic edge parameterization in edge classification and MPC weight routines.
5. **Add Benchmark Regression Tests:**
   Add an automated reference check comparing uniform-fine vs. adaptive mesh load-displacement responses on the Winkler L-shaped panel.
