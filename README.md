[![documentation](https://github.com/EdelweissFE/EdelweissFE/actions/workflows/sphinx.yml/badge.svg)](https://edelweiss-numerics.github.io/EdelweissFE)
[![codecov](https://codecov.io/gh/EdelweissFE/EdelweissFE/graph/badge.svg)](https://codecov.io/gh/EdelweissFE/EdelweissFE)
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

Everything, including the free-threaded Python interpreter, comes from conda: from
[conda-forge](https://conda-forge.org/), plus the
[`matthiasneuner/edelweiss`](https://prefix.dev/channels/edelweiss) channel for the few packages conda-forge lacks
(a free-threaded `vtk`, and `autodiff`, `fastor`, `amgcl` for building Marmot).
Linux and macOS are supported.

```console
mamba env create -f environment.yml
mamba activate edelweissfe
mamba install mkl mkl-include   # Linux only, optional: enables the PARDISO solver

pip install --no-deps --no-build-isolation .
run_tests_edelweissfe ./testfiles/edelweiss-only/
```

To use the Marmot-backed elements and materials, build [Marmot](https://github.com/MAteRialMOdelingToolbox/Marmot/)
into the same environment before installing EdelweissFE (all its dependencies are already in the environment):

```console
git clone --recurse-submodules --branch next_v26.11 https://github.com/MAteRialMOdelingToolbox/Marmot/ ../Marmot
cmake -S ../Marmot -B ../Marmot/build -DCMAKE_INSTALL_PREFIX=$CONDA_PREFIX -DCMAKE_PREFIX_PATH=$CONDA_PREFIX
cmake --build ../Marmot/build -j && cmake --install ../Marmot/build

pip install -v --no-deps --no-build-isolation .
run_tests_edelweissfe ./testfiles/marmot/
```

See the [installation documentation](doc/source/installation.rst) for details and troubleshooting.

