---
name: ew-code-review
description: >-
  Quality assurance, static check, and architectural review checklist for EdelweissFE changes and pull requests.
  Use when reviewing code, checking pre-commit compliance, inspecting free-threading safety, or preparing PR submissions.
---

# Code Review & QA Checklist

## 1. Free-Threading & CSR Performance
- [ ] **No GIL Re-enabling Imports**: Lazy import non-nogil C-extensions (e.g. `gstools`). Never at module top level.
- [ ] **Cython Directives**: Ensure `freethreading_compatible: True` in `setup.py`.
- [ ] **CSR Sparsity**: In-place matrix assembly (`updateInPlace`), retain Dirichlet explicit off-diagonal zeros.

## 2. Architecture & Code Reuse
- [ ] **Code Reuse**: Subclass base classes (`BaseHypoElasticMaterial`, `DisplacementElement`, `LinearSolver`). Reuse `edelweissfe/numerics/` and `voigtnotation`.
- [ ] **Minimal State Variables**: Materials must store only strictly path-dependent history variables in `stateVars` (never instantaneous or algebraically computable quantities, unless explicitly requested).
- [ ] **Lazy Plugin Registrations**: Register in `edelweissfe/config/<subsystem>.py` with lazy imports.
- [ ] **Input System Registry**: Register keyword schemas via `InputSystemRegistry`.

## 3. Formatting, Linting & Hygiene
```bash
pre-commit run --all-files
bash scripts/format_cython_files.sh  # if Cython linting issues occur
```
- [ ] **No Stray Agent Artifacts**: Zero comments referencing agent plans, chats, prompts, or temp notes.
- [ ] **No Debug Print/Dumps**: Clean up temporary print statements or scratch files.

## 4. Tests & Documentation
- [ ] **Regression Tests**: Fast test deck (< 1s) under `testfiles/edelweiss-only/` using `*modelGenerator`.
- [ ] **Run Tests**: `run_tests_edelweissfe ./testfiles/edelweiss-only/` passes.
- [ ] **Sphinx Docs**: `sphinx-build ./doc/source/ ./docs -b html` builds without warnings.

## 5. Maintenance & PR Targeting
- [ ] **Agent Guidance**: Update `AGENTS.md`, `CONTRIBUTING.md`, or `.agents/skills/` if workflows change.
- [ ] **Conventional Commits**: `<type>(<scope>): <summary>` (`feat(...)`, `fix(...)`).
- [ ] **PR Target**: `master` for bug fixes; `next_v<YY>.<MM>` for features/enhancements.
