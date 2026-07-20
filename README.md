[![documentation](https://github.com/EdelweissFE/EdelweissFE/actions/workflows/sphinx.yml/badge.svg)](https://edelweiss-numerics.github.io/EdelweissFE)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![DOI](https://zenodo.org/badge/1095513352.svg)](https://doi.org/10.5281/zenodo.17603044)

# EdelweissFE: A light-weight, platform-independent, parallel finite element framework.

<p align="center">
  <img width="512" height="512" src="./doc/source/borehole_damage_lowdilation.gif">
</p>

See the [documentation](https://edelweiss-numerics.github.io/EdelweissFE).

EdelweissFE aims at an easy to understand, yet efficient implementation of the finite element method.
Some features are:

 * Python for non performance-critical routines
 * Cython for performance-critical routines
 * Parallelization
 * Modular system, which is easy to extend
 * Output to Paraview, Ensight, CSV, matplotlib
 * Interfaces to powerful direct and iterative linear solvers

EdelweissFE makes use of the [Marmot](https://github.com/MAteRialMOdelingToolbox/Marmot/) library for finite element and constitutive model formulations.

## Installation

The quickest path is the bootstrap script, which creates a fresh conda environment and
builds the complete stack (including Marmot and its dependencies) in one go:

```console
bash scripts/bootstrap_stack.sh            # full stack, environment name "edelweissfe"
ENV_NAME=myenv bash scripts/bootstrap_stack.sh
```

The manual installation paths below assume that you are in the repository root and that
your conda environment is active.

### Working installation without Marmot

Step 1: Install the required conda packages.

```console
mamba install --file conda_requirements.txt
```

Step 2: Install the additional pip packages.

```console
pip install -r pip_requirements.txt
```

Step 3: Install EdelweissFE. The build dependencies (Cython, numpy) are already provided
by the conda environment, so disable pip's build isolation to compile against them.

```console
pip install --no-build-isolation .
```

The build prints a summary of the compiled extensions; optional ones (Marmot wrappers,
linear-solver interfaces) may fail without aborting the installation, and the result is
recorded in `edelweissfe/built_extensions.log`.

Step 4: Validate the EdelweissFE-only installation.

```console
run_tests_edelweissfe ./testfiles/edelweiss-only/
```

### Working installation with Marmot

Step 1: Install the required conda packages (this includes Eigen, which Marmot uses).

```console
mamba install --file conda_requirements.txt
```

Step 2: Install the additional pip packages.

```console
pip install -r pip_requirements.txt
```

Step 3: Install autodiff.

```console
cd ..
git clone --branch v1.1.0 https://github.com/autodiff/autodiff.git
cd autodiff
mkdir build
cd build
cmake -DAUTODIFF_BUILD_TESTS=OFF -DAUTODIFF_BUILD_PYTHON=OFF -DAUTODIFF_BUILD_EXAMPLES=OFF -DAUTODIFF_BUILD_DOCS=OFF -DCMAKE_INSTALL_PREFIX=$CONDA_PREFIX ..
make install
cd ../..
```

Step 4: Install Fastor.

```console
git clone https://github.com/romeric/Fastor.git
cd Fastor
mkdir build
cd build
cmake -DBUILD_TESTING=OFF -DCMAKE_INSTALL_PREFIX=$CONDA_PREFIX ..
make install
cd ../..
```

Step 5: Install AMGCL.

```console
git clone --branch 1.4.7 --depth 1 https://github.com/ddemidov/amgcl.git
cd amgcl
mkdir build
cd build
cmake -DCMAKE_INSTALL_PREFIX=$CONDA_PREFIX ..
make install
cd ../..
```

Step 6: Install Marmot from the master branch.

```console
git clone --branch master --recurse-submodules https://github.com/MAteRialMOdelingToolbox/Marmot/
cd Marmot
mkdir build
cd build
cmake -DCMAKE_INSTALL_PREFIX=$CONDA_PREFIX ..
make install
cd ../../EdelweissFE
```

Step 7: Install EdelweissFE with Marmot available.

```console
pip install --no-build-isolation -v .
```

Step 8: Validate the Marmot-enabled installation.

```console
run_tests_edelweissfe ./testfiles/marmot/
run_tests_edelweissfe ./testfiles/edelweiss-only/
```

### Verifying free-threading

EdelweissFE targets the free-threading (no-GIL) build of CPython. All Cython extensions
declare themselves free-threading compatible; if any imported extension or third-party
package has not, CPython silently re-enables the GIL and the parallel solvers lose their
element-loop speedup. Verify your installation keeps the GIL disabled:

```console
python -c "import sys; import edelweissfe.numerics.csrgeneratorv2, edelweissfe.utils.elementresultcollector, edelweissfe.solvers.base.dirichlet; assert not sys._is_gil_enabled(), 'GIL was re-enabled!'; print('free-threading OK')"
```

As a stopgap, `PYTHON_GIL=0 edelweissfe ...` forces the GIL off regardless.
