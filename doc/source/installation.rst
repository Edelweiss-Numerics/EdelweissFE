Installation
============

EdelweissFE is installed into a conda environment that contains every dependency, including the free-threaded
(``cp314t``) Python interpreter. Nothing is installed with ``pip`` except EdelweissFE itself.

All dependencies come from `conda-forge <https://conda-forge.org/>`_, except for the few packages conda-forge does not
provide yet, which come from the `matthiasneuner/edelweiss <https://prefix.dev/channels/edelweiss>`_ channel on
prefix.dev:

* ``vtk`` built for free-threaded Python (conda-forge only builds it for the regular interpreter),
* ``autodiff`` 1.1.2 with Eigen 5 support, ``fastor`` and ``amgcl`` (header-only C++ libraries required to build Marmot).

The recipes of that channel are maintained at
`matthiasneuner/edelweiss-conda-channel <https://github.com/matthiasneuner/edelweiss-conda-channel>`_.

Supported platforms are Linux (x86-64) and macOS (arm64 and x86-64).

Get Miniforge
*************

If you do not have a conda installation yet, install `Miniforge <https://conda-forge.org/download/>`_:

.. code-block:: console

    curl -L -O "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"
    bash Miniforge3-$(uname)-$(uname -m).sh

Create the environment
**********************

From the EdelweissFE repository root:

.. code-block:: console

    mamba env create -f environment.yml
    mamba activate edelweissfe

On Linux, additionally install Intel MKL to enable the PARDISO direct solver:

.. code-block:: console

    mamba install mkl mkl-include

MKL is not available on macOS. PARDISO is optional: without MKL, the PARDISO extension is simply not built and the
default linear solver falls back to SciPy's SuperLU.

.. note::

    Keep the default (flexible) channel priority. The environment does not solve with
    ``--strict-channel-priority``.

Installation without Marmot
***************************

.. code-block:: console

    pip install --no-deps --no-build-isolation .
    run_tests_edelweissfe ./testfiles/edelweiss-only/

``--no-build-isolation`` builds against the environment's Cython, NumPy and setuptools instead of letting pip fetch
them from PyPI, and ``--no-deps`` keeps pip from installing anything else.

This installation is sufficient for the EdelweissFE-only elements, materials and tests.

Installation with Marmot
************************

`Marmot <https://github.com/MAteRialMOdelingToolbox/Marmot/>`_ provides the Marmot-backed elements and constitutive
models. All of its dependencies (Eigen, autodiff, Fastor) are already in the environment, so only Marmot itself is
built from source, into the environment's prefix:

.. code-block:: console

    cd ..
    git clone --recurse-submodules https://github.com/MAteRialMOdelingToolbox/Marmot/
    cd Marmot
    git checkout next_v26.11
    cmake -S . -B build -DCMAKE_INSTALL_PREFIX=$CONDA_PREFIX -DCMAKE_PREFIX_PATH=$CONDA_PREFIX
    cmake --build build -j
    cmake --install build
    cd ../EdelweissFE

Then build EdelweissFE, which picks up Marmot automatically, and validate the installation:

.. code-block:: console

    pip install -v --no-deps --no-build-isolation .
    run_tests_edelweissfe ./testfiles/marmot/
    run_tests_edelweissfe ./testfiles/edelweiss-only/

Marmot is found in ``$CONDA_PREFIX`` by default; set ``MARMOT_INSTALL_DIR`` if it is installed elsewhere.

Running with free-threading
***************************

Disable the GIL and set the number of threads explicitly:

.. code-block:: console

    PYTHON_GIL=0 OMP_NUM_THREADS=8 edelweissfe input.inp

Troubleshooting
***************

* **CMake finds an unexpected Eigen or other package.** CMake also searches its user package registry
  (``~/.cmake/packages``), to which some projects register their *build* trees. If a package from the environment is
  rejected (e.g. by a version check), CMake silently falls back to such an entry. Configure with
  ``-DCMAKE_FIND_USE_PACKAGE_REGISTRY=OFF`` to rule this out; the configure output prints the Eigen version and location
  that was found.
* **The GIL is re-enabled at runtime.** Importing any extension module that does not declare free-threading support
  re-enables the GIL for the whole process, with a ``RuntimeWarning``. Do not install such packages (e.g. ``pyamg``)
  into the environment.
