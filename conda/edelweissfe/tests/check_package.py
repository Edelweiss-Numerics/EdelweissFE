"""Package test of edelweissfe: every expected extension imports, the GIL stays off, and two test cases pass."""

import importlib
import os
import shutil
import subprocess
import sys

extensions = [
    "edelweissfe.elements.marmotelement.element",
    "edelweissfe.materials.marmot.marmothypoelastic",
    "edelweissfe.materials.marmot.marmotgradientenhancedhypoelastic",
    "edelweissfe.utils.elementresultcollector",
    "edelweissfe.numerics.csrgenerator",
    "edelweissfe.numerics.csrgeneratorv2",
    "edelweissfe.solvers.base.dirichlet",
    "edelweissfe.linsolve.amgcl.amgcl",
    "edelweissfe.linsolve.klu.klu",
]
if sys.platform != "darwin":  # PARDISO needs Intel MKL, which does not exist for macOS
    extensions.append("edelweissfe.linsolve.pardiso.pardiso")

for extension in extensions:
    importlib.import_module(extension)
    print("imported", extension)

# An extension not declared free-threading compatible silently re-enables the GIL process-wide.
assert not sys._is_gil_enabled(), "importing the extensions re-enabled the GIL"

runner = shutil.which("run_tests_edelweissfe")
assert runner, "run_tests_edelweissfe is not on PATH"
# The runner's output is captured and printed here: on Windows, the build tool's log otherwise loses it.
env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUTF8="1")
# The installed launcher itself must start (on Windows a broken one exits 1 silently).
subprocess.run([runner, "--help"], check=True, env=env)
# Each case must PASS: the runner exits 0 for a SKIPPED case (e.g. an element or material the packaged Marmot lacks).
failed = []
for directory, case in [("testfiles/edelweiss-only", "CantileverBeamQuad4"), ("testfiles/marmot", "CPS4")]:
    result = subprocess.run(
        [runner, directory], capture_output=True, text=True, encoding="utf-8", errors="replace", env=env
    )
    print(result.stdout)
    print(result.stderr, file=sys.stderr)
    passed = any(case in line and "PASSED" in line for line in result.stdout.splitlines())
    if result.returncode != 0 or not passed:
        failed.append(case)
assert not failed, f"test cases did not pass: {failed}"
