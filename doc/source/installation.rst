Installation
============

The currently maintained installation recipes reflect the supported installation paths for EdelweissFE.
They assume that you are in the EdelweissFE repository root and that a conda environment is active,
so that ``$CONDA_PREFIX`` points to the installation prefix used by the build steps.

Common base setup
*****************

Both supported installation paths start with the same package installation steps:

.. code-block:: console

    mamba install --file conda_requirements.txt
    pip install -r pip_requirements.txt

Working installation without Marmot
***********************************

Build a working EdelweissFE installation without Marmot support as follows:

.. code-block:: console

    mamba install --file conda_requirements.txt
    pip install -r pip_requirements.txt
    pip install .

Validate that installation with the same command used in CI:

.. code-block:: console

    run_tests_edelweissfe ./testfiles/edelweiss-only/

This installation path is sufficient for the EdelweissFE-only examples and tests. Marmot-backed elements and material models
require the additional dependencies described below.

Working installation with Marmot
********************************

Extend the base setup with the external libraries needed for Marmot-enabled builds.

Install Eigen:

.. code-block:: console

    cd ..
    git clone --branch 3.4.0 https://gitlab.com/libeigen/eigen.git
    cd eigen
    mkdir build
    cd build
    cmake -DBUILD_TESTING=OFF -DINCLUDE_INSTALL_DIR=$CONDA_PREFIX/include -DCMAKE_INSTALL_PREFIX=$CONDA_PREFIX ..
    make install
    cd ../..

Install autodiff:

.. code-block:: console

    git clone --branch v1.1.0 https://github.com/autodiff/autodiff.git
    cd autodiff
    mkdir build
    cd build
    cmake -DAUTODIFF_BUILD_TESTS=OFF \
      -DAUTODIFF_BUILD_PYTHON=OFF \
      -DAUTODIFF_BUILD_EXAMPLES=OFF \
      -DAUTODIFF_BUILD_DOCS=OFF \
      -DCMAKE_INSTALL_PREFIX=$CONDA_PREFIX \
      ..
    make install
    cd ../..

Install Fastor:

.. code-block:: console

    git clone https://github.com/romeric/Fastor.git
    cd Fastor
    mkdir build
    cd build
    cmake -DBUILD_TESTING=OFF -DCMAKE_INSTALL_PREFIX=$CONDA_PREFIX ..
    make install
    cd ../..

Build Boost from source:

``boost-cpp``'s only available conda-forge build (1.85.0) pins ``icu<76``, which
conflicts with the ``icu>=78.3`` that the only free-threaded-Python-compatible
``matplotlib`` build needs, so it can't be installed into the same environment
as the rest of ``conda_requirements.txt``. Build the specific Boost libraries
AMGCL needs (``program_options``, ``serialization``, ``test``) from source
instead, into the same conda prefix as everything else:

.. code-block:: console

    curl -L -o boost_1_85_0.tar.gz https://archives.boost.io/release/1.85.0/source/boost_1_85_0.tar.gz
    tar xzf boost_1_85_0.tar.gz
    cd boost_1_85_0
    ./bootstrap.sh --prefix=$CONDA_PREFIX --with-libraries=program_options,serialization,test
    ./b2 install
    cd ..

Install AMGCL:

Even installed into the same prefix, AMGCL's ``cmake`` won't reliably find
Boost without an explicit hint (a plain, unhinted ``cmake ..`` here can
silently pick up an unrelated Boost from elsewhere on the system instead —
verified, not hypothetical), so point it there explicitly:

.. code-block:: console

    git clone --branch 1.4.7 --depth 1 https://github.com/ddemidov/amgcl.git
    cd amgcl
    mkdir build
    cd build
    cmake -DCMAKE_INSTALL_PREFIX=$CONDA_PREFIX -DBOOST_ROOT=$CONDA_PREFIX -DCMAKE_PREFIX_PATH=$CONDA_PREFIX ..
    make install
    cd ../..

Install Marmot:

.. code-block:: console

    git clone --branch master --recurse-submodules https://github.com/MAteRialMOdelingToolbox/Marmot/
    cd Marmot

Then build and install Marmot:

.. code-block:: console

    mkdir build
    cd build
    cmake -DCMAKE_INSTALL_PREFIX=$CONDA_PREFIX ..
    make install
    cd ../../EdelweissFE

Build EdelweissFE with Marmot available:

.. code-block:: console

    pip install -v .

Validate that installation with the same CI commands:

.. code-block:: console

    run_tests_edelweissfe ./testfiles/marmot/
    run_tests_edelweissfe ./testfiles/edelweiss-only/

TLDR
****

Assuming that you are in an empty directory,
you can quickly get a working version of EdelweissFE in a Linux based
environment:

Installation steps without Marmot
_________________________________

If necessary, get `Anaconda <https://www.anaconda.com/>`_

The example below uses the Linux ``aarch64`` Miniforge installer. If you are on a different platform,
choose the matching installer from the Miniforge releases page.

.. code-block:: console
   :caption: Step 1

    curl -L -O \
        https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-aarch64.sh
    bash Miniforge3-Linux-aarch64.sh -b -p ./miniforge3

Add mamba to your environment:

.. code-block:: console
   :caption: Step 2

    export EWROOT=$PWD
    export PATH=$EWROOT/miniforge3/bin:$PATH
    conda init --all
    exit

Restart shell and activate mamba

.. code-block:: console
   :caption: Step 3

    export EWROOT=$PWD
    conda activate

Get EdelweissFE:

.. code-block:: console
   :caption: Step 4

    git clone https://github.com/EdelweissFE/EdelweissFE.git

Install the required conda packages:

.. code-block:: console
   :caption: Step 5

    mamba install --file EdelweissFE/conda_requirements.txt

Install the additional pip packages:

.. code-block:: console
   :caption: Step 6

    pip install -r EdelweissFE/pip_requirements.txt

Build and test EdelweissFE without Marmot:

.. code-block:: console
   :caption: Step 7

    cd $EWROOT/EdelweissFE
    pip install .
    run_tests_edelweissfe ./testfiles/edelweiss-only/

Installation steps with Marmot
______________________________

Install Eigen:

.. code-block:: console
   :caption: Step 8

    cd $EWROOT
    git clone --branch 3.4.0 https://gitlab.com/libeigen/eigen.git
    cd eigen
    mkdir build
    cd build
    cmake \
        -DBUILD_TESTING=OFF \
        -DINCLUDE_INSTALL_DIR=$CONDA_PREFIX/include \
        -DCMAKE_INSTALL_PREFIX=$CONDA_PREFIX \
        ..
    make install

Install autodiff:

.. code-block:: console
   :caption: Step 9

    cd $EWROOT
    git clone --branch v1.1.0 https://github.com/autodiff/autodiff.git
    cd autodiff
    mkdir build
    cd build
    cmake \
        -DAUTODIFF_BUILD_TESTS=OFF \
        -DAUTODIFF_BUILD_PYTHON=OFF \
        -DAUTODIFF_BUILD_EXAMPLES=OFF \
        -DAUTODIFF_BUILD_DOCS=OFF \
        -DCMAKE_INSTALL_PREFIX=$CONDA_PREFIX \
        ..
    make install

Install Fastor:

.. code-block:: console
   :caption: Step 10

    cd $EWROOT
    git clone https://github.com/romeric/Fastor.git
    cd Fastor
    mkdir build
    cd build
    cmake -DBUILD_TESTING=OFF -DCMAKE_INSTALL_PREFIX=$CONDA_PREFIX ..
    make install

Build Boost from source (``boost-cpp``'s only conda-forge build conflicts
with the free-threaded-Python-compatible ``matplotlib`` build — see above):

.. code-block:: console
   :caption: Step 11

    cd $EWROOT
    curl -L -o boost_1_85_0.tar.gz https://archives.boost.io/release/1.85.0/source/boost_1_85_0.tar.gz
    tar xzf boost_1_85_0.tar.gz
    cd boost_1_85_0
    ./bootstrap.sh --prefix=$CONDA_PREFIX --with-libraries=program_options,serialization,test
    ./b2 install

Install AMGCL (pointing its ``cmake`` explicitly at the Boost just built —
a plain, unhinted ``cmake ..`` here can silently pick up an unrelated Boost
from elsewhere on the system instead):

.. code-block:: console
   :caption: Step 12

    cd $EWROOT
    git clone --branch 1.4.7 --depth 1 https://github.com/ddemidov/amgcl.git
    cd amgcl
    mkdir build
    cd build
    cmake -DCMAKE_INSTALL_PREFIX=$CONDA_PREFIX -DBOOST_ROOT=$CONDA_PREFIX -DCMAKE_PREFIX_PATH=$CONDA_PREFIX ..
    make install

Install Marmot from the master branch:

.. code-block:: console
   :caption: Step 13

    cd $EWROOT
    git clone --branch master --recurse-submodules https://github.com/MAteRialMOdelingToolbox/Marmot/
    cd Marmot
    mkdir build
    cd build
    cmake -DCMAKE_INSTALL_PREFIX=$CONDA_PREFIX ..
    make install

Build and test EdelweissFE with Marmot:

.. code-block:: console
   :caption: Step 14

    cd $EWROOT/EdelweissFE
    pip install -v .
    run_tests_edelweissfe ./testfiles/marmot/
    run_tests_edelweissfe ./testfiles/edelweiss-only/

Build the documentation
***********************

The documentation workflow builds the HTML output with:

.. code-block:: console

    sphinx-build ./doc/source/ ./docs -b html
