# AGENTS.md

This file provides guidance to AI Agents when working with code in this repository.

## What this is

EdelweissFE is a light-weight, platform-independent, parallel finite element framework (Unit of Strength of
Materials and Structural Analysis, University of Innsbruck). Python handles non performance-critical code;
Cython/C/C++ handle performance-critical routines (element loops, CSR matrix assembly, Dirichlet BC
application, linear solver interfaces). It optionally links against the sister C++ library
[Marmot](https://github.com/MAteRialMOdelingToolbox/Marmot/) for element/material formulations, but also
ships pure-Python/Cython element and material implementations that work standalone.

Requires Python >= 3.14. The pip-only environment (`pip_requirements.txt` / `conda_requirements.txt`
"without Marmot") targets the free-threaded ("nogil") CPython build (`python-freethreading`); see the
`freethreading_compatible` Cython directive in `setup.py` — importing a non-freethreading-safe extension
would silently re-enable the GIL process-wide and disable the thread-parallel element loops.

## Build & install

Cython extensions are compiled at install time via `setup.py` (`cythonize`, `-O3 -march=native`, with optional extensions skipped if native libraries are missing — see `edelweissfe/built_extensions.log`).

```console
# Standalone (pure Python/Cython elements & materials only)
mamba install --file conda_requirements.txt
pip install -r pip_requirements.txt
pip install .

# With Marmot (native C++ element/material formulations):
pip install -v .
```

See [CONTRIBUTING.md](CONTRIBUTING.md#environment-setup--installation) for full environment prerequisites (Eigen, autodiff, Fastor, AMGCL, Marmot) and build overrides (`MARMOT_INSTALL_DIR`, `MKL_INCLUDE_DIR`, `EIGEN_INCLUDE_DIR`, `EDELWEISSFE_ARCH_FLAGS`).

## Running tests

Tests are full finite-element input-deck regression tests driven by the `run_tests_edelweissfe` console script (`edelweissfe/_cli/_run_tests_edelweissfe.py`):

```console
run_tests_edelweissfe ./testfiles/edelweiss-only/      # tests that don't need Marmot
run_tests_edelweissfe ./testfiles/marmot/              # tests that need Marmot
run_tests_edelweissfe ./testfiles/marmot/ --tests ADVonMises,ADLinearElastic
run_tests_edelweissfe ./testfiles/marmot/ --create     # (re)generate U.ref reference solutions
pytest edelweissfe/numerics/test_numerics.py           # unit tests
```

See [CONTRIBUTING.md](CONTRIBUTING.md#adding--running-tests) for test creation guidelines, and [CONTRIBUTING.md](CONTRIBUTING.md#pull-requests) for CI Marmot branch resolution and PR target branch conventions (`master` for bugfixes, `next_v<YY>.<MM>` for features).

## Linting, formatting & commit conventions

Formatting and static checks are enforced by pre-commit hooks (`autoflake`, `black --line-length 120`, `isort`, `flake8`, `cython-lint`, `clang-format`). All commits must adhere to Conventional Commits. A helper script for formatting and resolving `cython-lint` issues is available at `scripts/format_cython_files.sh`.

See [CONTRIBUTING.md](CONTRIBUTING.md#pre-commit-hooks) for hook installation and tool flags, and [CONTRIBUTING.md](CONTRIBUTING.md#conventional-commits) for commit message types and subsystem scopes.

## Workspace Skills (`.agents/skills/`)

Specialized runbooks and checklists for development workflows are available under `.agents/skills/`. Agents should activate and follow the corresponding `SKILL.md` when tasked with:

- **[`ew-add-module`](.agents/skills/ew-add-module/SKILL.md)**: Universal entry point and architectural lifecycle for adding or extending any kind of functionality (solvers, linear solvers, step actions, constraints, AMR markers, output managers, generators, analytical fields), and routing to specialized skills.
- **[`ew-add-material`](.agents/skills/ew-add-material/SKILL.md)**: Use when creating, refactoring, or registering a new material model (elastic, hyperelastic, hypoelastic, plastic, damage). Follow this skill to reuse tensor utilities, inherit from base classes, register lazily in `materiallibrary.py`, and document in Sphinx.
- **[`ew-add-element`](.agents/skills/ew-add-element/SKILL.md)**: Use when implementing new finite element formulations (continuum, structural, mixed/EAS, Marmot wrappers). Follow this skill for node numbering conventions, quadrature/shape function reuse, `*elementproperty` support, and registration in `elementlibrary.py`.
- **[`ew-create-regression-test`](.agents/skills/ew-create-regression-test/SKILL.md)**: Use when creating or updating regression test decks (`test.inp` + `U.ref`) for bug fixes, new features, or solver benchmarking. Follow this skill for fast-running mesh guidelines, running `run_tests_edelweissfe --create`, and residual checks.
- **[`ew-documentation`](.agents/skills/ew-documentation/SKILL.md)**: Use when writing, extending, or verifying Sphinx documentation (`.rst` files), input keyword catalogs, mathematical equations, and NumPy-style docstrings.
- **[`ew-code-review`](.agents/skills/ew-code-review/SKILL.md)**: Use when reviewing code, checking pre-commit compliance, inspecting free-threading safety, or preparing PR submissions. Follow this skill to verify zero-GIL re-enabling imports, in-place CSR assembly, documentation updates, and agent guidance maintenance.

## Architecture

### The `*keyword`-based input file DSL

Simulations are defined in a custom text format (`*.inp`), parsed by `edelweissfe/utils/inputfileparser.py`
against a schema in `edelweissfe/utils/inputlanguage.py` using the `InputSystemRegistry`. Every
`*someKeyword, option=value, ...` block is parsed into a dict entry; `>>subKeyword` lines nest sub-definitions
(e.g. `>>dirichlet`, `>>nodeforces` inside `*step`; `>>perNode` inside `*fieldOutput`); indented, unmarked
lines below a keyword are "datalines" (e.g. node/element connectivity tables). `edelweissfe -k` / `--keywords`
prints the full keyword reference; `--doc=<module>` prints a module's docstring-based docs.

A minimal job looks like:

```
*material, name=linearelastic, id=linearelastic, provider=edelweiss
30000.0, 0.15

*section, name=section1, material=linearelastic, type=solid
all

*job, name=myjob, domain=3d
*solver, solver=NIST, name=theSolver

*node
      1,    0., 0., 0.
      ...
*element, type=C3D8, provider=edelweiss
      1,    1, 2, 3, 4, 5, 6, 7, 8

*nSet, nSet=left
      1, 2, 3, ...

*fieldOutput
>>perNode, elSet=all, field=displacement, result=U, name=displacement

*output, type=ensight, name=esExport
>>perNode, fieldOutput=displacement

*step, solver=theSolver
maxInc=1.0, minInc=1e-8, maxNumInc=1000, maxIter=100
>>dirichlet, name=left, nSet=left, field=displacement, 1=0, 2=0, 3=0
```

Full keyword/syntax reference: `doc/source/documentation/syntax.rst` and `doc/source/documentation/keywords.rst`.

### Plugin-registry pattern for extensibility

Almost every pluggable concept (elements, materials, solvers, linear solvers, generators, step-actions,
output managers, constraints, sections, analytical fields, AMR markers) follows the same pattern: a small
`edelweissfe/config/<thing>.py` module maps a user-facing name (from the input file, e.g. `type=C3D8` or
`solver=NISTParallel`) to a submodule path, then `importlib`-imports and returns the class lazily (so unused
optional deps — e.g. Marmot-backed elements, AMGCL, or external libraries like `gstools` — are never imported
unless actually requested, avoiding unnecessary memory overhead and preventing unintentional GIL re-enabling).
Example: `getElementClass()` in `edelweissfe/config/elementlibrary.py` dispatches on `provider=` (`edelweiss`
vs `marmot` vs `marmotsingleqpelement`); `edelweissfe/config/solvers.py::solverLibrary` maps
`NIST`/`NISTParallel`/etc. to solver modules. When adding a new element/material/solver/etc., register it in
the corresponding `edelweissfe/config/*.py` file rather than importing it eagerly elsewhere.

### Simulation flow & Core Subsystems

Entry point: `edelweissfe/drivers/inputfiledrivensimulation.py::finiteElementSimulation()` (called by both
the `edelweissfe` CLI and `run_tests_edelweissfe`):

1. **Model Assembly**: Parse the input file → build an `FEModel` (`edelweissfe/models/femodel.py`), the model
   tree holding nodes, elements, node/element sets, sections, surfaces, constraints, contact pairs, materials,
   analytical fields, scalar variables, and rigid bodies (`edelweissfe/helpers/inputfilehelpers.py::fillFEModelFromInputFile`).
2. **Controller & Output Setup**: Build the field output controller, plotter, output managers, and named solvers
   from the input file.
3. **Step Execution**: A `StepManager` (from `*step` blocks) yields `Step` objects one at a time; each step collects
   its step-actions (`>>dirichlet`, `>>nodeforces`, `>>options`, ...) and calls `step.solve()` against its assigned solver.
4. **Finalization**: Field outputs and output managers (Ensight, Paraview, CSV, matplotlib, ...) are finalized after
   all steps, or on failure (`StepFailed`) / `KeyboardInterrupt`.

#### Solvers & Parallelization
- **Nonlinear Solvers** (`edelweissfe/solvers/`): Implicit/explicit static/dynamic solvers (`NIST`, `NEST`, `NED`),
  each with serial and `...Parallel` thread-parallel variants (e.g. `NISTParallel`), plus arc-length methods
  (`NISTPArcLength`).
- **Thread Parallelism & Free-Threading**: Element loops dispatch element chunks across persistent thread pools
  (`edelweissfe/numerics/parallelizationutilities.py::getThreadPool`). `ScatterDofVector` uses precomputed layout
  templates (`ScatterDofVectorTemplate`) and `np.bincount` for lock-free parallel accumulation. Third-party C
  extensions lacking free-threading tags (e.g. `gstools`) are imported lazily to keep the GIL disabled.
- **CSR Matrix Assembly**: High-performance assembly via `CSRGenerator` and OpenMP-accelerated `CSRGeneratorV2`
  (`edelweissfe/numerics/`). In-place assembly (`updateInPlace`) avoids reallocating sparsity patterns across iterations.
- **Dirichlet BC Enforcement**: Optimized in `edelweissfe/solvers/base/dirichlet.pyx`. Off-diagonal entries are
  zeroed explicitly to preserve the CSR sparsity structure (avoiding costly `eliminate_zeros()` rebuilds) and allow
  linear solvers to reuse symbolic factorizations. Dirichlet indices are memoized per BC/DofManager.

#### Linear Solvers (`edelweissfe/linsolve/`)
All linear solvers adhere to the unified `LinearSolver` base contract (`edelweissfe.linsolve`):
- **MKL Pardiso**: Direct sparse solver (`PardisoSolver`) with multi-RHS support, bounds checking, and optional
  symbolic factorization reuse (`reuseSymbolicFactorization=False` by default for numerical safety in coupled DOFs).
- **Panua Pardiso**: Alternative Pardiso interface (`optional=True`).
- **AMGCL**: Algebraic multigrid preconditioners paired with Krylov methods (BiCGStab, CG) and outer LGMRES solvers.
- **KLU**: SuiteSparse direct sparse solver for unsymmetric/sparse systems.
- **Block-AMG (`BlockAMGSolver`)**: Field-split block algebraic multigrid solver for multi-physics/coupled problems.
- **Diagnostics & Benchmarking**: Offline matrix dump (`matrixdump`) and solver benchmarking utilities (`bench_linsolve`).

#### Contact, Rigid Bodies & Constraints (`edelweissfe/constraints/`)
- **Node-to-Surface & Contact with Facets**: Penalty contact formulations (`NodeToDeformableSurfacePenaltyConstraint`,
  `NodeToDiscreteRigidBodyPenaltyConstraint`).
- **Surface Tie Constraints**: Multi-point constraint tying non-conforming interface surfaces.
- **Rigid Bodies & MPCs**: Linearized 2D/3D rigid bodies (`*rigidBody`, `LinearizedRigidBodyConstraint`), Lagrangian
  and penalty equal-value constraints (`EqualValuePenaltyConstraint`), and penalty-based indirect displacement control.

#### Adaptivity (AMR) & Topology Pipeline
- **Adaptive Mesh Refinement**: AMR with hanging nodes (`AMRHangingNodes`), 2:1 balancing rules, and ZZ (Zienkiewicz-Zhu)
  recovery-based error estimators / Superconvergent Patch Recovery (`SPR`). Refinement can be batched using `minMarkedElements`.
- **Topology Pipeline**: Model-modifier pipeline with a monotonic element/node allocator for dynamic mesh modifications.

#### Checkpoint & Restart (`*restart`)
- State snapshotting and resumption via `*restart` and `RestartManager`.
- Restart-aware output managers (preserving Ensight transient sequences and geometry ring buffers) and adaptive time steppers.

#### Elements & Materials
- **Elements**: 2D/3D continuum (e.g. `C3D8`, `C3D20`, `CPS4`, `CPS8R`, mixed EAS elements), structural/truss/beam elements,
  and Marmot-backed elements. Node numbering follows standard Abaqus/Marmot conventions (e.g. Hexa8/Hexa20). Elements support
  named string-keyed properties (`*elementproperty`).
- **Multi-Field Phenomena**: Registered scalar fields (e.g. `pressure`, `jacobi`) for mixed $u$-$p$-$J$ formulations.
- **Materials**: Linear elastic (isotropic, orthotropic, transverse isotropic), hyperelastic (consolidated Neo-Hookean formulations
  Wa/Wb/Wc, Pence-Gou), hypoelastic, plasticity (von Mises, Drucker-Prager, Concrete Damage Plasticity), and user/Marmot materials.

### Documentation

Sphinx docs live in `doc/source/documentation/*.rst`, one file per subsystem (elements, materials, solvers,
linsolvers, constraints, contacttheory, rigidbodies, steps, stepactions, generators, fields, dofmanager,
parallelization, etc.) — check the relevant `.rst` file first when working on an unfamiliar subsystem,
since these describe intended design/usage, not just API.

