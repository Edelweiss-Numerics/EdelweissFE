---
name: ew-add-module
description: >-
  Universal workflow and architectural lifecycle for adding or extending any kind of functionality in EdelweissFE.
  Use when implementing new solvers, linear solvers, step actions, constraints, output managers, generators, analytical fields, or when routing to specialized skills.
---

# Module Development Lifecycle in EdelweissFE

Universal entry point for adding/extending any subsystem feature.

## 1. Routing to Dedicated Skills

| Task | Skill |
| :--- | :--- |
| Materials (elastic, plastic, damage) | [`ew-add-material`](../ew-add-material/SKILL.md) |
| Elements (continuum, structural, mixed) | [`ew-add-element`](../ew-add-element/SKILL.md) |
| Regression test decks (`test.inp` + `U.ref`) | [`ew-create-regression-test`](../ew-create-regression-test/SKILL.md) |
| Sphinx docs & keyword reference | [`ew-documentation`](../ew-documentation/SKILL.md) |
| QA & PR code review | [`ew-code-review`](../ew-code-review/SKILL.md) |

---

## 2. Subsystem Directory & Plugin Registry Map

| Subsystem | Directory | Registry (`edelweissfe/config/`) |
| :--- | :--- | :--- |
| Nonlinear Solvers | `edelweissfe/solvers/` | `solvers.py::solverLibrary` |
| Linear Solvers | `edelweissfe/linsolve/` | `linsolve.py::getLinSolverByName` |
| Step Actions (loads, BCs, controllers) | `edelweissfe/stepactions/` | `stepactions.py` |
| Constraints & MPCs | `edelweissfe/constraints/` | `constraints.py` |
| AMR Markers | `edelweissfe/adaptivity/` | `amrmarkers.py` |
| Output Managers | `edelweissfe/outputmanagers/` | `outputmanagers.py` |
| Mesh Generators | `edelweissfe/generators/` | `generators.py` |
| Analytical Fields | `edelweissfe/analyticalfields/` | `analyticalfields.py` |
| Sections | `edelweissfe/sections/` | `sections.py` |

---

## 3. Implementation Checklist

1. **Inheritance & Code Reuse**: Inherit from subsystem base class (`LinearSolver`, `StepActionBase`, `ConstraintBase`). Reuse shared math/tensor routines in `edelweissfe/numerics/`.
2. **Free-Threading Safety**: Never import third-party C-extensions without `Py_MOD_GIL_NOT_USED` (e.g. `gstools`) at top level. Import lazily inside methods.
3. **Lazy Registration**: In `edelweissfe/config/<subsystem>.py`, import class lazily inside condition:
   ```python
   if strCaseCmp(name, "myFeature"):
       from edelweissfe.subsystem.myfeature import MyFeature

       return MyFeature
```
4. **Input DSL Schema**: Register keyword handler via `InputSystemRegistry` in `edelweissfe/utils/inputlanguage.py`.
5. **Regression Test**: Follow [`ew-create-regression-test`](../ew-create-regression-test/SKILL.md) to generate `test.inp` + `U.ref`.
6. **Documentation**: Follow [`ew-documentation`](../ew-documentation/SKILL.md) to update `.rst` files in `doc/source/documentation/`.
7. **Review**: Check with [`ew-code-review`](../ew-code-review/SKILL.md) (`pre-commit run --all-files`, clean commit message).
