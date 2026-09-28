Installation
============

EdelweissFE is developed in a dedicated conda environment. ``environment.yml`` declares it, and the committed lockfile
``conda-lock.yml`` pins it exactly: every package, with version, build and checksum, for every supported platform.
The same pinned environment is published as the conda package ``edelweissfe-dev``, versioned by date (the current
version is in ``conda/edelweissfe-dev/VERSION``), so it installs with a single ``conda create``, identically on every
machine and in CI. Nothing changes until a new version is published. Everything, including the free-threaded
(``cp314t``) Python interpreter, comes from conda packages; ``pip`` only builds and installs EdelweissFE itself.

The packages come from `conda-forge <https://conda-forge.org/>`_, except for the few conda-forge does not provide
yet, which come from the `matthiasneuner/edelweiss <https://prefix.dev/channels/edelweiss>`_ channel on prefix.dev:

* ``vtk`` built for free-threaded Python (conda-forge only builds it for the regular interpreter),
* ``autodiff`` 1.1.2 with Eigen 5 support, ``fastor`` and ``amgcl`` (header-only C++ libraries).

The recipes of that channel are maintained at
`matthiasneuner/edelweiss-conda-channel <https://github.com/matthiasneuner/edelweiss-conda-channel>`_.

Supported platforms are Linux (x86-64), macOS 14 or newer (arm64 and x86-64), and Windows (x64, without Marmot).

Prerequisites
*************

A conda installation, e.g. `Miniforge <https://conda-forge.org/download/>`_. On Linux and macOS:

.. code-block:: console

    curl -L -O "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"
    bash Miniforge3-$(uname)-$(uname -m).sh

On Windows, use the Miniforge installer (``Miniforge3-Windows-x86_64.exe``) and run the commands below in the
*Miniforge Prompt*. Windows additionally needs the Microsoft C++ compiler: install the free
`Build Tools for Visual Studio <https://visualstudio.microsoft.com/visual-cpp-build-tools/>`_ with the workload
*Desktop development with C++*. The environment's ``compilers`` package only activates it; it cannot install it.

Create a dedicated environment
******************************

Install EdelweissFE into an environment of its own, never into conda's ``base`` environment or one shared with other
projects:

* **It needs a special interpreter.** EdelweissFE runs on the free-threaded (``cp314t``) Python build, and many
  packages built for the regular interpreter cannot be installed alongside it.
* **Its package versions are pinned.** The environment is installed exactly as ``conda-lock.yml`` specifies. Installing
  other packages into it changes those versions, and a single package that is not free-threading safe (e.g.
  ``pyamg``) silently re-enables the GIL for the whole process, disabling the thread-parallel element loops.
* **It can be rebuilt at any time.** A dedicated environment can simply be deleted and recreated from the lockfile,
  which reliably fixes a broken installation; ``base`` cannot be recreated that way.
* **Marmot is built into it.** ``cmake --install`` writes Marmot's libraries and headers into the environment, where
  they must not mix with other projects' builds.

Create it with the pinned environment package:

.. code-block:: console

    conda create -n edelweissfe -c https://repo.prefix.dev/matthiasneuner/edelweiss -c conda-forge edelweissfe-dev=2026.09.28.2
    conda activate edelweissfe

This installs exactly the environment of ``conda-lock.yml`` for your platform.

**Alternatively, from the lockfile.** The same environment can be installed from ``conda-lock.yml`` in the
repository root with `conda-lock <https://conda.github.io/conda-lock/>`_, which itself gets a small environment of its
own (not ``base``). This route does not include ``edelweissfe-dev``'s activation script, so set
``PIP_NO_BUILD_ISOLATION`` yourself (last line; see below):

.. code-block:: console

    conda create -n conda-lock -c conda-forge conda-lock
    conda run -n conda-lock conda-lock install -n edelweissfe conda-lock.yml
    conda env config vars set -n edelweissfe PIP_NO_BUILD_ISOLATION=0
    conda activate edelweissfe

.. note::

    Do not create the environment from ``environment.yml`` directly (``conda env create -f environment.yml``): that
    re-solves it against whatever packages are newest today, which is exactly what the pinned environment avoids.

On Linux and Windows the environment includes Intel MKL, which enables the PARDISO direct solver. MKL does not exist for
macOS; there the PARDISO extension is simply not built and the default linear solver falls back to SciPy's SuperLU.

The commands on this page use Linux/macOS shell syntax. On Windows, set environment variables separately in the
Miniforge Prompt (cmd.exe), e.g. ``set PYTHON_GIL=0`` before ``run_tests_edelweissfe .\testfiles\edelweiss-only\``.

Installation without Marmot
***************************

.. code-block:: console

    pip install -e .
    PYTHON_GIL=0 run_tests_edelweissfe ./testfiles/edelweiss-only/

pip only builds and installs EdelweissFE itself; all dependencies are already in the environment. ``-e`` (editable)
makes changes to Python files take effect immediately; rerun the command after changing Cython or C++ sources.

pip builds against the environment's own setuptools, Cython and NumPy: activating the environment sets
``PIP_NO_BUILD_ISOLATION=0`` (pip reads it inverted; ``0`` disables build isolation). Without that, pip would fetch
its own copies from PyPI into a temporary build environment and compile the Cython extensions against those, which
can mismatch the environment's NumPy at runtime. An environment installed from the lockfile needs the
``conda env config vars set`` step shown above, or ``pip install --no-build-isolation -e .``.

This installation is sufficient for the EdelweissFE-only elements, materials and tests.

Installation with Marmot
************************

Not yet supported on Windows.

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

    pip install -v -e .
    PYTHON_GIL=0 run_tests_edelweissfe ./testfiles/marmot/
    PYTHON_GIL=0 run_tests_edelweissfe ./testfiles/edelweiss-only/

Marmot is found in ``$CONDA_PREFIX`` by default; set ``MARMOT_INSTALL_DIR`` if it is installed elsewhere.

Developing
**********

Python changes take effect immediately (editable install). After changing Cython sources, or after rebuilding and
installing Marmot, rerun ``pip install -e .``. The editable install belongs to one checkout: in a
second checkout or git worktree, use a separate environment, or it silently runs the first checkout's code. Further:

* ``PYTHON_GIL=0 pytest tests``: the pytest suite,
* ``sphinx-build -b html doc/source doc/build/html``: this documentation.

Changing dependencies
*********************

With conda-lock in its own environment (``conda create -n conda-lock -c conda-forge conda-lock``), edit
``environment.yml``, re-lock, and update your environment from the lockfile:

.. code-block:: console

    conda run -n conda-lock conda-lock lock -f environment.yml --virtual-package-spec virtual-packages.yml
    conda run -n conda-lock conda-lock install -n edelweissfe conda-lock.yml

Then set a new version, today's date, in ``conda/edelweissfe-dev/VERSION`` (append ``.1``, ``.2``, ... for further
changes on the same day) and in the ``conda create`` commands of this page, the README, CONTRIBUTING.md and AGENTS.md. Commit everything together;
CI fails if the lockfile is out of date with ``environment.yml``, if it changed without a new version, or if the
documented commands do not show the current version. Once merged into ``next_v26.11``, CI publishes the new
``edelweissfe-dev``.

``virtual-packages.yml`` tells conda-lock which system properties (e.g. the minimum macOS version) to assume for each
platform. Platform-specific dependencies use selectors, e.g. ``- mkl  # [linux64]``.

A weekly CI job re-locks against the newest packages and opens a pull request with a new version, whose CI tests the
updated environment before it is merged.

Running with free-threading
***************************

Disable the GIL and set the number of threads explicitly:

.. code-block:: console

    PYTHON_GIL=0 OMP_NUM_THREADS=8 edelweissfe input.inp

Troubleshooting
***************

* **The environment behaves differently than on other machines.** Make sure it was created from ``edelweissfe-dev``
  (or with ``conda-lock install``), not from ``environment.yml``, and that ``conda list edelweissfe-dev`` shows the
  version documented above.
* **CMake finds an unexpected Eigen or other package.** CMake also searches its user package registry
  (``~/.cmake/packages``), to which some projects register their *build* trees. If a package from the environment is
  rejected (e.g. by a version check), CMake silently falls back to such an entry. Configure with
  ``-DCMAKE_FIND_USE_PACKAGE_REGISTRY=OFF`` (as above); the configure output prints the Eigen version and location
  that was found.
* **The GIL is re-enabled at runtime.** Importing any extension module that does not declare free-threading support
  re-enables the GIL for the whole process, with a ``RuntimeWarning``. Do not add such packages (e.g. ``pyamg``) to the
  environment.
