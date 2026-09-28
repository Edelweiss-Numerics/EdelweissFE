Installation
============

EdelweissFE is developed in a conda environment. ``environment.yml`` declares it, and the committed lockfile
``conda-lock.yml`` pins it exactly: every package, with version, build and checksum, for every supported platform.
Installing from the lockfile gives the identical environment on every machine and in CI, and nothing changes until
the lockfile is deliberately updated. Everything, including the free-threaded (``cp314t``) Python interpreter, comes
from conda packages; nothing is installed with ``pip`` except EdelweissFE itself.

The packages come from `conda-forge <https://conda-forge.org/>`_, except for the few conda-forge does not provide
yet, which come from the `matthiasneuner/edelweiss <https://prefix.dev/channels/edelweiss>`_ channel on prefix.dev:

* ``vtk`` built for free-threaded Python (conda-forge only builds it for the regular interpreter),
* ``autodiff`` 1.1.2 with Eigen 5 support, ``fastor`` and ``amgcl`` (header-only C++ libraries).

The recipes of that channel are maintained at
`matthiasneuner/edelweiss-conda-channel <https://github.com/matthiasneuner/edelweiss-conda-channel>`_.

Supported platforms are Linux (x86-64) and macOS 14 or newer (arm64 and x86-64).

Prerequisites
*************

A conda installation, e.g. `Miniforge <https://conda-forge.org/download/>`_, and
`conda-lock <https://conda.github.io/conda-lock/>`_:

.. code-block:: console

    curl -L -O "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"
    bash Miniforge3-$(uname)-$(uname -m).sh
    mamba install -n base conda-lock

Create the environment
**********************

From the EdelweissFE repository root, install the environment **from the lockfile**:

.. code-block:: console

    conda-lock install -n edelweissfe conda-lock.yml
    conda activate edelweissfe

.. note::

    Do not create the environment from ``environment.yml`` directly (``mamba env create -f environment.yml``):
    that re-solves it against whatever packages are newest today, which is exactly what the lockfile avoids.

On Linux the environment includes Intel MKL, which enables the PARDISO direct solver. MKL does not exist for macOS;
there the PARDISO extension is simply not built and the default linear solver falls back to SciPy's SuperLU.

Installation without Marmot
***************************

.. code-block:: console

    pip install --no-deps --no-build-isolation .
    PYTHON_GIL=0 run_tests_edelweissfe ./testfiles/edelweiss-only/

``--no-build-isolation`` builds against the environment's Cython, NumPy and setuptools instead of letting pip fetch
them from PyPI, and ``--no-deps`` keeps pip from installing anything else.

This installation is sufficient for the EdelweissFE-only elements, materials and tests.

Installation with Marmot
************************

`Marmot <https://github.com/MAteRialMOdelingToolbox/Marmot/>`_ provides the Marmot-backed elements and constitutive
models. All of its dependencies (Eigen, autodiff, Fastor) are already in the environment, so only Marmot itself is
built from source, into the environment:

.. code-block:: console

    git clone --recurse-submodules --branch next_v26.11 https://github.com/MAteRialMOdelingToolbox/Marmot/ ../Marmot
    cmake -S ../Marmot -B ../Marmot/build -DCMAKE_INSTALL_PREFIX=$CONDA_PREFIX -DCMAKE_PREFIX_PATH=$CONDA_PREFIX \
          -DCMAKE_FIND_USE_PACKAGE_REGISTRY=OFF
    cmake --build ../Marmot/build -j
    cmake --install ../Marmot/build

Then build EdelweissFE, which picks up Marmot automatically, and validate the installation:

.. code-block:: console

    pip install -v --no-deps --no-build-isolation .
    PYTHON_GIL=0 run_tests_edelweissfe ./testfiles/marmot/
    PYTHON_GIL=0 run_tests_edelweissfe ./testfiles/edelweiss-only/

Marmot is found in ``$CONDA_PREFIX`` by default; set ``MARMOT_INSTALL_DIR`` if it is installed elsewhere.

Developing
**********

Edit the sources of EdelweissFE (or Marmot), then rebuild with ``pip install --no-deps --no-build-isolation .``
(after rebuilding and installing Marmot for Marmot changes). Further:

* ``PYTHON_GIL=0 pytest tests``: the pytest suite,
* ``sphinx-build -b html doc/source doc/build/html``: this documentation.

Changing dependencies
*********************

Edit ``environment.yml``, then re-lock and update your environment:

.. code-block:: console

    conda-lock lock -f environment.yml --virtual-package-spec virtual-packages.yml
    conda-lock install -n edelweissfe conda-lock.yml

Commit ``environment.yml`` and ``conda-lock.yml`` together; CI fails if the lockfile is out of date with
``environment.yml``. ``virtual-packages.yml`` tells conda-lock which system properties (e.g. the minimum macOS
version) to assume for each platform. Platform-specific dependencies use selectors, e.g. ``- mkl  # [linux64]``.

A weekly CI job re-locks against the newest packages and opens a pull request, whose CI tests the updated stack
before it is merged.

Running with free-threading
***************************

Disable the GIL and set the number of threads explicitly:

.. code-block:: console

    PYTHON_GIL=0 OMP_NUM_THREADS=8 edelweissfe input.inp

Troubleshooting
***************

* **The environment does not solve, or behaves differently than on other machines.** Make sure it was installed with
  ``conda-lock install`` from ``conda-lock.yml``, not created from ``environment.yml``. With mamba, keep the default
  (flexible) channel priority: this environment does not solve with ``--strict-channel-priority`` in mamba.
* **CMake finds an unexpected Eigen or other package.** CMake also searches its user package registry
  (``~/.cmake/packages``), to which some projects register their *build* trees. If a package from the environment is
  rejected (e.g. by a version check), CMake silently falls back to such an entry. Configure with
  ``-DCMAKE_FIND_USE_PACKAGE_REGISTRY=OFF`` (as above); the configure output prints the Eigen version and location
  that was found.
* **The GIL is re-enabled at runtime.** Importing any extension module that does not declare free-threading support
  re-enables the GIL for the whole process, with a ``RuntimeWarning``. Do not add such packages (e.g. ``pyamg``) to the
  environment.
