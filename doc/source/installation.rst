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

Supported platforms are Linux (x86-64), macOS 14 or newer (arm64 and x86-64), and Windows (x64).

Ready-to-run package
********************

To run simulations without developing EdelweissFE, install the conda package ``edelweissfe`` (released versions,
with Marmot's public elements and materials and all linear solvers; same platforms as below):

.. code-block:: console

    conda create -n edelweissfe -c https\://repo.prefix.dev/matthiasneuner/edelweiss -c conda-forge edelweissfe

It depends on the matching ``marmot`` package, pinned exactly: compiled C++ code has no stable interface across
Marmot versions. Both are built with portable compiler flags (no ``-march=native``).

Everything else on this page is for **developing** EdelweissFE or Marmot, adding (private) Marmot modules, or
compiling for your own CPU. Do that in a separate environment created from ``edelweissfe-dev``, never in one with the
``edelweissfe`` or ``marmot`` package installed: a self-built Marmot installed there overwrites files conda manages,
and the packaged EdelweissFE would then run against a library it was not compiled for. Likewise, ``pip install -e .``
on top of the ``edelweissfe`` package leaves two copies of EdelweissFE in the environment.

The recipe is ``conda/edelweissfe/recipe.yaml``; the ``conda`` workflow builds and tests it (against Marmot's own
recipe, built from the matching Marmot branch) on every pull request and uploads it for a release tag.

Get conda
*********

A conda installation, e.g. `Miniforge <https://conda-forge.org/download/>`_. On Linux and macOS:

.. code-block:: console

    curl -L -O "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"
    bash Miniforge3-$(uname)-$(uname -m).sh

On Windows, use the Miniforge installer (``Miniforge3-Windows-x86_64.exe``) and run the commands below in the
*Miniforge Prompt*. Windows additionally needs the Microsoft C++ compiler: the free
`Visual Studio 2022 Build Tools <https://visualstudio.microsoft.com/visual-cpp-build-tools/>`_ (MSVC v143 toolset)
with the workload *Desktop development with C++*. The environment's ``compilers`` package only activates it; it cannot
install it. Both can also be installed from a terminal:

.. code-block:: console

    winget install CondaForge.Miniforge3
    winget install Microsoft.VisualStudio.2022.BuildTools --override "--wait --passive --add Microsoft.VisualStudio.Workload.VCTools --includeRecommended"

Activating the environment on Windows prints a long list of Visual Studio setup commands; that is the compiler
activation and normal.

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

.. parsed-literal::

    conda create -n edelweissfe -c https\://repo.prefix.dev/matthiasneuner/edelweiss -c conda-forge edelweissfe-dev=\ |edelweissfe_dev_version|
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

Without Marmot, the build output (``pip install -v``) shows a warning that Marmot was not found, followed by compiler
errors (e.g. ``fatal error C1083`` on Windows) and ``[FAIL]`` lines for the three Marmot extensions
(``marmotelement.element``, ``marmothypoelastic``, ``marmotgradientenhancedhypoelastic``). That is expected: these
extensions are optional and skipped. ``edelweissfe/built_extensions.log`` lists the extensions that were built.

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

On Windows, in the *Miniforge Prompt*, conda's headers and libraries are under ``%CONDA_PREFIX%\Library``, and
Marmot is built in the ``Release`` configuration:

.. code-block:: console

    git clone --recurse-submodules --branch next_v26.11 https://github.com/MAteRialMOdelingToolbox/Marmot/ ..\Marmot
    cmake -S ..\Marmot -B ..\Marmot\build -DCMAKE_INSTALL_PREFIX=%CONDA_PREFIX%\Library ^
          -DCMAKE_PREFIX_PATH=%CONDA_PREFIX%\Library -DCMAKE_FIND_USE_PACKAGE_REGISTRY=OFF
    cmake --build ..\Marmot\build --config Release --parallel
    cmake --install ..\Marmot\build --config Release

Then build EdelweissFE, which picks up Marmot automatically, and validate the installation:

.. code-block:: console

    pip install -v -e .
    PYTHON_GIL=0 run_tests_edelweissfe ./testfiles/marmot/
    PYTHON_GIL=0 run_tests_edelweissfe ./testfiles/edelweiss-only/

Marmot is found in ``$CONDA_PREFIX`` (Windows: ``%CONDA_PREFIX%\Library``) by default; set ``MARMOT_INSTALL_DIR``
if it is installed elsewhere. The build records the Marmot installation it used (``edelweissfe/marmot_install_dir.txt``):
on Linux and macOS, the Marmot extensions find the library there through their embedded library path; on Windows,
``import edelweissfe`` adds its ``bin`` directory to the DLL search path. So the extensions always load the Marmot they
were built against, whatever the runtime environment. Rebuild EdelweissFE after moving or reinstalling Marmot
elsewhere.

The extensions are compiled with ``-march=native`` by default (no architecture flags on Windows), so a build only runs
on processors like the one it was compiled on. For builds that must run elsewhere, e.g. container images or packages,
set ``EDELWEISSFE_ARCH_FLAGS`` (e.g. ``-march=x86-64-v3``, or empty for none). The compile flags are defined in
``edelweissfe.utils.extensionbuild``, which downstream packages (EdelweissMeshfree) use for their own extensions.

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
changes on the same day) and in the ``conda create`` command of the README (this page reads it from ``VERSION``). Commit everything together;
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
