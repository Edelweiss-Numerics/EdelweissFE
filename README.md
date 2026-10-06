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

EdelweissFE runs on Linux, macOS (version 14 or newer) and Windows. Installing it takes four steps and about
15 minutes, most of it waiting for downloads.

### 1. Install conda

EdelweissFE and everything it needs (Python, numerical libraries, compilers) are installed with **conda**, a
package manager. If you do not have conda yet, install **Miniforge**, a free distribution of conda:

- **Linux or macOS:** open a terminal and run
  ```console
  curl -L -O "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"
  bash Miniforge3-$(uname)-$(uname -m).sh
  ```
  then close and reopen the terminal.
- **Windows:** download and run the installer from [conda-forge.org/download](https://conda-forge.org/download/).
  Windows also needs Microsoft's C++ compiler, which conda cannot install: install the free
  [Visual Studio 2022 Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/) and select
  *Desktop development with C++*. Afterwards, run all commands below in the **Miniforge Prompt** (find it in the
  Start menu).

### 2. Create the EdelweissFE environment

An *environment* is a separate folder that contains everything EdelweissFE needs, in exactly the versions it is
tested with, without touching anything else on your computer. Create it with:

```console
conda create -n edelweissfe -c https://repo.prefix.dev/matthiasneuner/edelweiss -c conda-forge edelweissfe-dev=2026.10.04
```

Always use this separate environment for EdelweissFE. If it ever breaks, delete it
(`conda env remove -n edelweissfe`) and run the command again.

### 3. Download and install EdelweissFE

```console
conda activate edelweissfe
git clone --branch next_v26.11 https://github.com/Edelweiss-Numerics/EdelweissFE.git
cd EdelweissFE
pip install -e .
```

`conda activate edelweissfe` switches to the environment. You need it again in every new terminal before using
EdelweissFE.

### 4. Check that it works

Run the test suite (it takes a few minutes and should end with `Tests failed: 0`):

- **Linux or macOS:**
  ```console
  PYTHON_GIL=0 run_tests_edelweissfe ./testfiles/edelweiss-only/
  ```
- **Windows:**
  ```console
  set PYTHON_GIL=0
  run_tests_edelweissfe .\testfiles\edelweiss-only\
  ```

`PYTHON_GIL=0` lets EdelweissFE compute in parallel on several processor cores.

### Running a simulation

```console
conda activate edelweissfe
edelweissfe my_simulation.inp
```

### Optional: Marmot

[Marmot](https://github.com/MAteRialMOdelingToolbox/Marmot/) adds further elements and material models. Build it into the same environment, then reinstall
EdelweissFE (on Windows, see the [installation documentation](doc/source/installation.rst) for the commands):

```console
conda activate edelweissfe
git clone --recurse-submodules --branch next_v26.11 https://github.com/MAteRialMOdelingToolbox/Marmot/ ../Marmot
cmake -S ../Marmot -B ../Marmot/build -DCMAKE_INSTALL_PREFIX=$CONDA_PREFIX -DCMAKE_PREFIX_PATH=$CONDA_PREFIX
cmake --build ../Marmot/build -j && cmake --install ../Marmot/build
pip install -e .
PYTHON_GIL=0 run_tests_edelweissfe ./testfiles/marmot/
```

More details, including how the environment is pinned, how to change dependencies and troubleshooting, are in the
[installation documentation](doc/source/installation.rst).
