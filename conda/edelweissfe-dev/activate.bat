@REM Installed by edelweissfe-dev into %CONDA_PREFIX%\etc\conda\activate.d; runs on `conda activate` (cmd.exe).
@REM See activate.sh: make `pip install -e .` build against this environment (0 disables build isolation).
@if defined PIP_NO_BUILD_ISOLATION set "_EDELWEISSFE_DEV_SAVED_PIP_NO_BUILD_ISOLATION=%PIP_NO_BUILD_ISOLATION%"
@set "PIP_NO_BUILD_ISOLATION=0"
