# Installed by edelweissfe-dev into $CONDA_PREFIX/etc/conda/activate.d; runs on `conda activate`.
# Build EdelweissFE (pip install -e .) against this environment's setuptools, Cython and NumPy, not against copies
# pip would otherwise fetch from PyPI into an isolated build environment. pip reads this variable inverted: 0
# disables build isolation.
if [ -n "${PIP_NO_BUILD_ISOLATION+x}" ]; then
    export _EDELWEISSFE_DEV_SAVED_PIP_NO_BUILD_ISOLATION="$PIP_NO_BUILD_ISOLATION"
fi
export PIP_NO_BUILD_ISOLATION=0
