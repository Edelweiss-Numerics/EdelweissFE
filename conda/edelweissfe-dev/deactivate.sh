# Installed by edelweissfe-dev into $CONDA_PREFIX/etc/conda/deactivate.d; restores what activate.sh changed.
if [ -n "${_EDELWEISSFE_DEV_SAVED_PIP_NO_BUILD_ISOLATION+x}" ]; then
    export PIP_NO_BUILD_ISOLATION="$_EDELWEISSFE_DEV_SAVED_PIP_NO_BUILD_ISOLATION"
    unset _EDELWEISSFE_DEV_SAVED_PIP_NO_BUILD_ISOLATION
else
    unset PIP_NO_BUILD_ISOLATION
fi
