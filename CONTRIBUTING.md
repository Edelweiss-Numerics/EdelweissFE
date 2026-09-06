# Contributing to EdelweissFE

We welcome pull requests of all sizes, bug reports, and enhancements. This guide explains how to set up your development environment, follow our commit conventions, run tests, write documentation, and open high-quality issues and pull requests.

---

## Code of Conduct
By participating in this project, you agree to uphold a respectful, inclusive environment. Be kind, be constructive, and assume good intent. If you encounter any problems, please open an issue.

## Ways to Contribute
- Report bugs and propose enhancements via GitHub Issues.
- Improve documentation, docstrings, and input-deck examples.
- Add test decks (`test.inp`), improve test coverage, or benchmark performance.
- Implement new elements, material models, solvers, linear solvers, constraints, or step-actions.
- Optimize performance-critical Cython, C++, or free-threaded Python routines.

> **Pull requests are welcome!** See [Pull Requests](#pull-requests).

---

## Environment Setup & Installation

EdelweissFE requires **Python >= 3.14** and targets the free-threaded ("nogil") CPython build (`python-freethreading`).

### 1. Standalone / Without Marmot (Pure Python/Cython only)
```bash
mamba install --file conda_requirements.txt
pip install -r pip_requirements.txt
pip install .
```

### 2. With Marmot (Native C++ Element & Material Formulations)
Requires Eigen, autodiff, Fastor, AMGCL, and [Marmot](https://github.com/MAteRialMOdelingToolbox/Marmot/) installed into `$CONDA_PREFIX`:
```bash
pip install -v .
```

### Build Environment Overrides
- `MARMOT_INSTALL_DIR`: Path to Marmot installation prefix (defaults to `$CONDA_PREFIX` / `sys.prefix`).
- `MKL_INCLUDE_DIR`: Path to Intel MKL headers.
- `EIGEN_INCLUDE_DIR`: Path to Eigen3 headers.
- `EDELWEISSFE_ARCH_FLAGS`: Override default compiler optimization flags (defaults to `-march=native`).

---

## Pre-commit Hooks

We use **pre-commit** to enforce code formatting and static checks before each commit.

### Install and Run
```bash
# Install the pre-commit framework (one-time)
pip install pre-commit

# Install git hook scripts
pre-commit install

# Run checks against all files
pre-commit run --all-files
```

Formatting and linting tools configured in `.pre-commit-config.yaml`:
- `autoflake`: Removes unused imports.
- `black`: Formats Python code (`--line-length 120`, target Python 3.12+).
- `isort`: Sorts imports (`--profile black`).
- `flake8`: Style and lint checks (`--max-line-length 120`, ignores `E203,E501`, extends ignore `W503`).
- `cython-lint`: Lints Cython `.pyx` files (`--max-line-length=120 --ignore=E741`).
- `clang-format`: Formats C and C++ source files.

> **Tip**: An automated helper script is available at `scripts/format_cython_files.sh` to iteratively format Cython `.pyx` and `.pxd` files and resolve common `cython-lint` issues.

---

## Conventional Commits

All commits **must** follow the [Conventional Commits](https://www.conventionalcommits.org) specification. This maintains a clean Git history and facilitates changelog generation.

### Format
```
<type>(<optional-scope>): <description>
```

### Types
- **feat**: A new feature or capability
- **fix**: A bug fix
- **docs**: Documentation-only changes
- **test**: Adding, updating, or fixing tests
- **refactor**: Code changes that neither fix a bug nor add a feature
- **perf**: Performance improvements
- **build**: Build system, setup scripts, or external dependencies
- **ci**: CI workflows and configuration
- **chore**: Routine maintenance tasks

### Common Scopes
`(solvers)`, `(linsolve)`, `(elements)`, `(materials)`, `(constraints)`, `(contact)`, `(adaptivity)`, `(topology)`, `(restart)`, `(numerics)`, `(inputlanguage)`, `(outputmanagers)`

### Examples
```
feat(linsolve): add field-split block-AMG solver
fix(solvers): guard CSR matrix against structural mutation
perf(numerics): use persistent thread pool for element parallel loops
docs(constraints): document surface tie multipoint constraint
test(marmot): add regression test for ADVonMises
```
> Keep the commit subject line concise (≤72 characters). Use the body to explain **what** and **why**, referencing issues where appropriate (e.g., `Fixes #42`).

---

## Opening Issues

Before opening a new issue, search existing issues and pull requests to avoid duplicates. When filing a bug report, please provide:
- **Environment**: OS, Python version (standard vs free-threaded build), compiler version, and Marmot version/branch (if applicable).
- **Installed Extensions**: The contents of `edelweissfe/built_extensions.log` if build or solver-related.
- **Minimal Reproducible Example (MRE)**: A self-contained input deck (`test.inp`) or minimal Python script demonstrating the failure.
- **Expected vs Actual Behavior**: Expected results vs observed error tracebacks or discrepancies.

---

## Pull Requests

We follow the GitHub flow: **fork → branch → PR → review → merge**.

### Target Branches
- **Bug fixes (`fix`)**: Open pull requests targeting the `master` branch.
- **Features, enhancements, and refactoring (`feat`, `refactor`, `perf`, etc.)**: Open pull requests targeting the upcoming release branch: `next_v<YY>.<MM>` (e.g., `next_v26.11`).

### Workflow
1. **Fork** the repository and create a feature/bugfix branch from the appropriate target base (`master` for fixes, `next_v<YY>.<MM>` for features):
   ```bash
   # For a bug fix:
   git checkout -b fix/<short-scope>-<concise-topic> origin/master

   # For a new feature / refactoring:
   git checkout -b feat/<short-scope>-<concise-topic> origin/next_v26.11
   ```
2. **Develop & Format**: Make your changes and verify that `pre-commit run --all-files` passes locally.
3. **Build & Test**: Ensure the package builds cleanly (`pip install -v .`) and all relevant tests pass (`run_tests_edelweissfe`).
4. **Open a PR**: Target the correct branch (`master` for bug fixes, `next_v<YY>.<MM>` for features/enhancements), provide a clear title following Conventional Commits, and link relevant issues.

### Synchronizing with Marmot
If your changes depend on features or fixes in [Marmot](https://github.com/MAteRialMOdelingToolbox/Marmot/), ensure the Marmot-side branch is named identically to your EdelweissFE branch. CI automatically resolves and checks out matching Marmot branches during test runs.

### PR Checklist
- [ ] PR targets the correct branch (`master` for bugfixes, `next_v<YY>.<MM>` for features/enhancements).
- [ ] PR title follows Conventional Commits format.
- [ ] `pre-commit run --all-files` passes cleanly.
- [ ] Project builds cleanly via `pip install .` (and `pip install -v .` if using Marmot).
- [ ] All tests pass locally via `run_tests_edelweissfe ./testfiles/edelweiss-only/` (and `./testfiles/marmot/`).
- [ ] New features, keywords, or options are documented in `doc/source/documentation/`.
- [ ] New features or bug fixes include a regression test deck (`test.inp` + `U.ref`).

---

## Documentation

Documentation is built with **Sphinx** and hosted on GitHub Pages. Sources are located under `doc/source/`.

### Local Documentation Build
```bash
# Build HTML documentation into ./docs
sphinx-build ./doc/source/ ./docs -b html
```

### Documenting New Features
When adding new functionality:
- **Subsystem documentation**: Add or update the corresponding topic page in `doc/source/documentation/*.rst` (e.g. `elements.rst`, `materials.rst`, `solvers.rst`, `linsolvers.rst`, `constraints.rst`, `contacttheory.rst`).
- **Input language & keywords**: Update `doc/source/documentation/keywords.rst` and provide clear docstrings on keyword handlers registered with `InputSystemRegistry`.
- **CLI module documentation**: Ensure docstrings are descriptive; users can query module documentation using `edelweissfe --doc=<module>`.

---

## Adding & Running Tests

EdelweissFE uses input-deck regression tests driven by the `run_tests_edelweissfe` CLI script.

### Running Existing Tests
```bash
# Run standalone tests (does not require Marmot)
run_tests_edelweissfe ./testfiles/edelweiss-only/

# Run tests requiring Marmot installation
run_tests_edelweissfe ./testfiles/marmot/

# Run specific named test cases
run_tests_edelweissfe ./testfiles/marmot/ --tests ADVonMises,ADLinearElastic

# Run Python unit tests (e.g., numerics unit tests)
pytest edelweissfe/numerics/test_numerics.py
```

### Creating a New Test Case
1. Create a subdirectory under `testfiles/edelweiss-only/` (or `testfiles/marmot/` if Marmot is required).
2. Add a minimal, deterministic `test.inp` file defining the simulation deck.
3. Generate the reference solution file (`U.ref`):
   ```bash
   run_tests_edelweissfe ./testfiles/edelweiss-only/ --tests MyNewTest --create
   ```
4. Verify that running without `--create` passes with a maximum residual below `1e-6`.

---

## Architectural & Coding Guidelines

- **Free-Threading Safety**: EdelweissFE runs under Python 3.14 free-threaded builds (`nogil`). All Cython extensions must specify `"freethreading_compatible": True` in directives. Ensure any native routines or thread-parallel loops are safe to execute concurrently without the GIL.
- **Lazy Dependency Loading**: Optional or heavy external libraries (such as `gstools`, `marmot`, or native solver wrappers) must be imported lazily via the plugin registry in `edelweissfe/config/<subsystem>.py` to avoid memory overhead and prevent third-party C-extensions from unintentionally re-enabling the GIL.
- **CSR Matrix Sparsity Preservation**: Avoid structural reallocations or destructive modifications to CSR matrices during Newton iterations. Zeroed off-diagonals (e.g. during Dirichlet boundary condition application) should be stored explicitly so that linear solvers can reuse symbolic factorizations.
- **Plugin Registries**: Extend elements, materials, solvers, and step-actions by registering lazy loaders in `edelweissfe/config/` rather than importing them eagerly across modules.

---

## License

By contributing to EdelweissFE, you agree that your contributions will be licensed under the **GNU Lesser General Public License v2.1** (LGPL-2.1). See [LICENSE.md](LICENSE.md) for full details.
