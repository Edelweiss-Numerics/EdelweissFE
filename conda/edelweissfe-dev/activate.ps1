# Installed by edelweissfe-dev into $env:CONDA_PREFIX\etc\conda\activate.d; runs on `conda activate` (PowerShell).
# See activate.sh: make `pip install -e .` build against this environment (0 disables build isolation).
if (Test-Path Env:PIP_NO_BUILD_ISOLATION) { $env:_EDELWEISSFE_DEV_SAVED_PIP_NO_BUILD_ISOLATION = $env:PIP_NO_BUILD_ISOLATION }
$env:PIP_NO_BUILD_ISOLATION = "0"
