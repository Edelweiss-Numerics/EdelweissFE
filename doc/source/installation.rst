Installation
============

EdelweissFE's development environment is defined by ``pixi.toml`` and pinned exactly, for every supported platform,
by the committed lockfile ``pixi.lock``. Everything, including the free-threaded (``cp314t``) Python interpreter,
comes from conda packages; nothing is installed with ``pip`` except EdelweissFE itself.

The packages come from `conda-forge <https://conda-forge.org/>`_, except for the few conda-forge does not provide
yet, which come from the `matthiasneuner/edelweiss <https://prefix.dev/channels/edelweiss>`_ channel on prefix.dev:

* ``vtk`` built for free-threaded Python (conda-forge only builds it for the regular interpreter),
* ``autodiff`` 1.1.2 with Eigen 5 support, ``fastor`` and ``amgcl`` (header-only C++ libraries).

The recipes of that channel are maintained at
`matthiasneuner/edelweiss-conda-channel <https://github.com/matthiasneuner/edelweiss-conda-channel>`_.

Supported platforms are Linux (x86-64) and macOS 14 or newer (arm64 and x86-64).

Get pixi
********

`pixi <https://pixi.sh>`_ is a package manager for conda packages that works per project, with lockfiles:

.. code-block:: console

    curl -fsSL https://pixi.sh/install.sh | sh

Create the environment
**********************

From the EdelweissFE repository root:

.. code-block:: console

    pixi install

This creates the environment in ``.pixi/`` exactly as pinned by ``pixi.lock``. Run commands in it with
``pixi run <command>``, or open a shell in it with ``pixi shell``.

On Linux the environment includes Intel MKL, which enables the PARDISO direct solver. MKL does not exist for macOS;
there the PARDISO extension is simply not built and the default linear solver falls back to SciPy's SuperLU.

Installation without Marmot
***************************

.. code-block:: console

    pixi run install
    pixi run test

``pixi run install`` runs ``pip install --no-deps --no-build-isolation .``: it builds against the environment's
Cython, NumPy and setuptools instead of letting pip fetch them from PyPI, and installs nothing else.

This installation is sufficient for the EdelweissFE-only elements, materials and tests.

Installation with Marmot
************************

`Marmot <https://github.com/MAteRialMOdelingToolbox/Marmot/>`_ provides the Marmot-backed elements and constitutive
models. All of its dependencies (Eigen, autodiff, Fastor) are already in the environment, so only Marmot itself is
built from source, into the environment:

.. code-block:: console

    git clone --recurse-submodules --branch next_v26.11 https://github.com/MAteRialMOdelingToolbox/Marmot/ ../Marmot
    pixi run build-marmot            # or: pixi run build-marmot /path/to/Marmot

Then build EdelweissFE, which picks up Marmot automatically, and validate the installation:

.. code-block:: console

    pixi run install
    pixi run test-marmot
    pixi run test

Marmot is found in the environment's prefix by default; set ``MARMOT_INSTALL_DIR`` if it is installed elsewhere.

Developing
**********

The environment is meant for development: edit the sources of EdelweissFE (or Marmot), then rebuild with
``pixi run install`` (after ``pixi run build-marmot`` for Marmot changes). Further tasks:

* ``pixi run pytest``: the pytest suite,
* ``pixi run docs``: this documentation, into ``doc/build/html``.

To change a dependency, edit ``pixi.toml``; pixi updates ``pixi.lock`` on the next ``pixi install`` or ``pixi run``.
Commit both files together. A weekly CI job re-solves ``pixi.lock`` against the newest packages and opens a pull
request, whose CI tests the updated stack before it is merged.

Running with free-threading
***************************

Disable the GIL and set the number of threads explicitly:

.. code-block:: console

    PYTHON_GIL=0 OMP_NUM_THREADS=8 pixi run edelweissfe input.inp

The ``test`` tasks already set ``PYTHON_GIL=0``.

Using conda or mamba instead of pixi
************************************

pixi can export the environment for conda/mamba, per platform:

.. code-block:: console

    pixi workspace export conda-environment --platform linux-64 environment.yml
    mamba env create -f environment.yml

The exported file is not pinned like ``pixi.lock``. With mamba, keep the default (flexible) channel priority: the
environment does not solve with ``--strict-channel-priority`` in mamba.

Troubleshooting
***************

* **CMake finds an unexpected Eigen or other package.** CMake also searches its user package registry
  (``~/.cmake/packages``), to which some projects register their *build* trees. If a package from the environment is
  rejected (e.g. by a version check), CMake silently falls back to such an entry. ``pixi run build-marmot``
  therefore configures with ``-DCMAKE_FIND_USE_PACKAGE_REGISTRY=OFF``; the configure output prints the Eigen version and
  location that was found.
* **The GIL is re-enabled at runtime.** Importing any extension module that does not declare free-threading support
  re-enables the GIL for the whole process, with a ``RuntimeWarning``. Do not add such packages (e.g. ``pyamg``) to the
  environment.
