@REM Installed by edelweissfe-dev into %CONDA_PREFIX%\etc\conda\deactivate.d; restores what activate.bat changed.
@if defined _EDELWEISSFE_DEV_SAVED_PIP_NO_BUILD_ISOLATION (
    @set "PIP_NO_BUILD_ISOLATION=%_EDELWEISSFE_DEV_SAVED_PIP_NO_BUILD_ISOLATION%"
    @set "_EDELWEISSFE_DEV_SAVED_PIP_NO_BUILD_ISOLATION="
) else (
    @set "PIP_NO_BUILD_ISOLATION="
)
