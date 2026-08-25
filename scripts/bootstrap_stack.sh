#!/usr/bin/env bash
#  ---------------------------------------------------------------------
#
#  Bootstrap the complete EdelweissFE stack into a fresh conda environment:
#
#    conda packages (incl. Eigen) -> autodiff -> Fastor -> AMGCL -> Marmot
#    -> EdelweissFE (with Marmot support)
#
#  Usage (from the EdelweissFE repository root):
#
#    bash scripts/bootstrap_stack.sh
#
#  Configuration via environment variables:
#
#    ENV_NAME       name of the conda environment to create   (default: edelweissfe)
#    MARMOT_BRANCH  Marmot branch to build                    (default: master)
#    BUILD_DIR      where the C++ dependencies are cloned     (default: ../edelweissfe_stack)
#    NPROC          parallel build jobs                       (default: nproc)
#
#  The script is re-runnable: existing clones and the existing environment
#  are reused.
#
#  ---------------------------------------------------------------------
set -euo pipefail

ENV_NAME="${ENV_NAME:-edelweissfe}"
MARMOT_BRANCH="${MARMOT_BRANCH:-master}"
BUILD_DIR="${BUILD_DIR:-$(pwd)/../edelweissfe_stack}"
NPROC="${NPROC:-$(nproc)}"

REPO_ROOT="$(pwd)"
if [[ ! -f "$REPO_ROOT/conda_requirements.txt" ]]; then
    echo "ERROR: run this script from the EdelweissFE repository root." >&2
    exit 1
fi

if command -v mamba &>/dev/null; then
    CONDA_CMD=mamba
elif command -v conda &>/dev/null; then
    CONDA_CMD=conda
else
    echo "ERROR: neither mamba nor conda found on PATH." >&2
    exit 1
fi

echo "==> Creating/updating conda environment '$ENV_NAME'"
if ! $CONDA_CMD env list | grep -qE "^$ENV_NAME\s"; then
    $CONDA_CMD create -y -n "$ENV_NAME" --file conda_requirements.txt
else
    $CONDA_CMD install -y -n "$ENV_NAME" --file conda_requirements.txt
fi

eval "$(conda shell.bash hook)"
conda activate "$ENV_NAME"

echo "==> Installing pip packages"
pip install -r pip_requirements.txt

mkdir -p "$BUILD_DIR"

clone_if_missing() {
    local url="$1" dir="$2"
    shift 2
    if [[ ! -d "$BUILD_DIR/$dir" ]]; then
        git clone "$@" "$url" "$BUILD_DIR/$dir"
    else
        echo "==> Reusing existing clone $BUILD_DIR/$dir"
    fi
}

cmake_install() {
    local dir="$1"
    shift
    cmake -S "$BUILD_DIR/$dir" -B "$BUILD_DIR/$dir/build" -DCMAKE_INSTALL_PREFIX="$CONDA_PREFIX" "$@"
    cmake --build "$BUILD_DIR/$dir/build" --parallel "$NPROC" --target install
}

echo "==> Building autodiff"
clone_if_missing https://github.com/autodiff/autodiff.git autodiff --branch v1.1.0 --depth 1
cmake_install autodiff \
    -DAUTODIFF_BUILD_TESTS=OFF -DAUTODIFF_BUILD_PYTHON=OFF \
    -DAUTODIFF_BUILD_EXAMPLES=OFF -DAUTODIFF_BUILD_DOCS=OFF

echo "==> Building Fastor"
clone_if_missing https://github.com/romeric/Fastor.git Fastor --depth 1
cmake_install Fastor -DBUILD_TESTING=OFF

echo "==> Building AMGCL"
clone_if_missing https://github.com/ddemidov/amgcl.git amgcl --branch 1.4.7 --depth 1
cmake_install amgcl

echo "==> Building Marmot (branch: $MARMOT_BRANCH)"
clone_if_missing https://github.com/MAteRialMOdelingToolbox/Marmot/ Marmot --branch "$MARMOT_BRANCH" --recurse-submodules
cmake_install Marmot

echo "==> Installing EdelweissFE"
cd "$REPO_ROOT"
pip install --no-build-isolation -v .

echo "==> Verifying free-threading"
python -c "
import sys
import edelweissfe.numerics.csrgeneratorv2
import edelweissfe.utils.elementresultcollector
import edelweissfe.solvers.base.dirichlet
assert not sys._is_gil_enabled(), 'GIL was re-enabled by an extension!'
print('free-threading OK')
"

echo "==> Running test suites"
run_tests_edelweissfe ./testfiles/edelweiss-only/
run_tests_edelweissfe ./testfiles/marmot/

echo "==> Done. Activate the environment with: conda activate $ENV_NAME"
