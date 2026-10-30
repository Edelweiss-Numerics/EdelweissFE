# Installed by edelweissfe-dev into $env:CONDA_PREFIX\etc\conda\deactivate.d; restores what activate.ps1 changed.
if (Test-Path Env:_EDELWEISSFE_DEV_SAVED_PIP_NO_BUILD_ISOLATION) {
    $env:PIP_NO_BUILD_ISOLATION = $env:_EDELWEISSFE_DEV_SAVED_PIP_NO_BUILD_ISOLATION
    Remove-Item Env:_EDELWEISSFE_DEV_SAVED_PIP_NO_BUILD_ISOLATION
} else {
    Remove-Item Env:PIP_NO_BUILD_ISOLATION -ErrorAction SilentlyContinue
}
