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

The following installation paths assume that you are in the repository root and that your conda environment is active.

### Working installation without Marmot

Step 1: Install the required conda packages.

```console
mamba install --file conda_requirements.txt
```

Step 2: Install the additional pip packages.

```console
pip install -r pip_requirements.txt
```

Step 3: Install EdelweissFE.

```console
pip install .
```

Step 4: Validate the EdelweissFE-only installation.

```console
run_tests_edelweissfe ./testfiles/edelweiss-only/
```

### Working installation with Marmot

Step 1: Install the required conda packages.

```console
mamba install --file conda_requirements.txt
```

Step 2: Install the additional pip packages.

```console
pip install -r pip_requirements.txt
```

Step 3: Install Eigen.

```console
cd ..
git clone --branch 3.4.0 https://gitlab.com/libeigen/eigen.git
cd eigen
mkdir build
cd build
cmake -DBUILD_TESTING=OFF -DINCLUDE_INSTALL_DIR=$CONDA_PREFIX/include -DCMAKE_INSTALL_PREFIX=$CONDA_PREFIX ..
make install
cd ../..
```

Step 4: Install autodiff.

```console
git clone --branch v1.1.0 https://github.com/autodiff/autodiff.git
cd autodiff
mkdir build
cd build
cmake -DAUTODIFF_BUILD_TESTS=OFF -DAUTODIFF_BUILD_PYTHON=OFF -DAUTODIFF_BUILD_EXAMPLES=OFF -DAUTODIFF_BUILD_DOCS=OFF -DCMAKE_INSTALL_PREFIX=$CONDA_PREFIX ..
make install
cd ../..
```

Step 5: Install Fastor.

```console
git clone https://github.com/romeric/Fastor.git
cd Fastor
mkdir build
cd build
cmake -DBUILD_TESTING=OFF -DCMAKE_INSTALL_PREFIX=$CONDA_PREFIX ..
make install
cd ../..
```

Step 6: Install AMGCL.

```console
git clone --branch 1.4.7 --depth 1 https://github.com/ddemidov/amgcl.git
cd amgcl
mkdir build
cd build
cmake -DCMAKE_INSTALL_PREFIX=$CONDA_PREFIX ..
make install
cd ../..
```

Step 7: Install Marmot from the master branch.

```console
git clone --branch master --recurse-submodules https://github.com/MAteRialMOdelingToolbox/Marmot/
cd Marmot
mkdir build
cd build
cmake -DCMAKE_INSTALL_PREFIX=$CONDA_PREFIX ..
make install
cd ../../EdelweissFE
```

Step 8: Install EdelweissFE with Marmot available.

```console
pip install -v .
```

Step 9: Validate the Marmot-enabled installation.

```console
run_tests_edelweissfe ./testfiles/marmot/
run_tests_edelweissfe ./testfiles/edelweiss-only/
```
