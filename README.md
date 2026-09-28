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

EdelweissFE is developed in a conda environment, declared in `environment.yml` and pinned exactly for every platform
by the lockfile `conda-lock.yml`. Everything, including the free-threaded Python interpreter, comes from conda
packages: from [conda-forge](https://conda-forge.org/), plus the
[`matthiasneuner/edelweiss`](https://prefix.dev/channels/edelweiss) channel for the few packages conda-forge lacks.
Linux and macOS 14+ are supported. Requires conda (e.g. Miniforge) and `conda-lock` (`mamba install -n base conda-lock`).
Use a dedicated environment for EdelweissFE, never `base` or one shared with other projects: it runs on the free-threaded
Python build, its versions are pinned by the lockfile, and it can be deleted and recreated from the lockfile at any time.

```console
conda-lock install -n edelweissfe conda-lock.yml
conda activate edelweissfe
pip install --no-build-isolation -e .
PYTHON_GIL=0 run_tests_edelweissfe ./testfiles/edelweiss-only/
```

To use the Marmot-backed elements and materials, build [Marmot](https://github.com/MAteRialMOdelingToolbox/Marmot/)
into the same environment first (all its dependencies are already in it):

```console
git clone --recurse-submodules --branch next_v26.11 https://github.com/MAteRialMOdelingToolbox/Marmot/ ../Marmot
cmake -S ../Marmot -B ../Marmot/build -DCMAKE_INSTALL_PREFIX=$CONDA_PREFIX -DCMAKE_PREFIX_PATH=$CONDA_PREFIX
cmake --build ../Marmot/build -j && cmake --install ../Marmot/build
pip install -v --no-build-isolation -e .
PYTHON_GIL=0 run_tests_edelweissfe ./testfiles/marmot/
```

See the [installation documentation](doc/source/installation.rst) for details, changing dependencies and troubleshooting.
