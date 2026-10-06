"""Generate the recipe of the ``edelweissfe-dev`` meta-package for one platform from ``conda-lock.yml``.

``edelweissfe-dev`` contains only two activation scripts (see activate.sh). It depends on every package of the locked EdelweissFE development
environment, each pinned to its exact version and build, so that

    conda create -n edelweissfe -c https://repo.prefix.dev/matthiasneuner/edelweiss -c conda-forge edelweissfe-dev=<VERSION>

creates exactly the environment ``conda-lock.yml`` describes for that platform. The version is the date in
``conda/edelweissfe-dev/VERSION``.

Usage: python generate_recipe.py <platform> <output directory>
"""

import pathlib
import shutil
import sys
import urllib.parse

import yaml

root = pathlib.Path(__file__).resolve().parents[2]
platform, output = sys.argv[1], pathlib.Path(sys.argv[2])

version = (root / "conda" / "edelweissfe-dev" / "VERSION").read_text().strip()
lock = yaml.safe_load((root / "conda-lock.yml").read_text())
packages = sorted(
    (package for package in lock["package"] if package["platform"] == platform and package["manager"] == "conda"),
    key=lambda package: package["name"],
)
if not packages:
    sys.exit(f"conda-lock.yml has no packages for platform {platform}")


def build_string(package):
    """Return the build string of a locked package, taken from its file name (<name>-<version>-<build>.conda)."""
    # Unquote first: versions with an epoch (e.g. x264 1!164.3095) appear URL-encoded (1%21164.3095).
    filename = urllib.parse.unquote(package["url"].rsplit("/", 1)[1])
    stem = filename.removesuffix(".conda").removesuffix(".tar.bz2")
    return stem.removeprefix(f"{package['name']}-{package['version']}-")


def pin(package):
    """Return the exact requirement for a locked package.

    CPU-level marker packages (e.g. _x86_64-microarch-level) are the exception: they have one build per CPU name
    (x86_64_v3, zen3, icelake, ...), each requiring exactly that name from the machine, so pinning the locked build
    would only install on CPUs reported by that name. Leave them unpinned; the packages that need a minimum level
    (e.g. SuiteSparse built for x86-64-v3) still require it themselves.
    """
    if package["name"].endswith("-microarch-level"):
        return package["name"]
    return f"{package['name']} =={package['version']} {build_string(package)}"


recipe = {
    "package": {"name": "edelweissfe-dev", "version": version},
    "build": {
        "number": 0,
        # The only files: activation scripts that make a plain `pip install -e .` build against this environment
        # (PIP_NO_BUILD_ISOLATION=0), see activate.sh.
        # (Built on Linux for every platform; Windows additionally gets cmd.exe and PowerShell variants.)
        "script": [
            f"install -D -m 644 $RECIPE_DIR/{script} $PREFIX/etc/conda/{phase}.d/edelweissfe-dev{suffix}"
            for phase in ("activate", "deactivate")
            for script, suffix in (
                [(f"{phase}.sh", ".sh")]
                + ([(f"{phase}.bat", ".bat"), (f"{phase}.ps1", ".ps1")] if platform.startswith("win-") else [])
            )
        ],
    },
    "requirements": {
        "run": [pin(package) for package in packages],
    },
    "about": {
        "homepage": "https://github.com/Edelweiss-Numerics/EdelweissFE",
        "license": "LGPL-2.1-only",
        "summary": "The complete, pinned conda environment for developing EdelweissFE (generated from conda-lock.yml).",
    },
}

output.mkdir(parents=True, exist_ok=True)
for script in ("activate.sh", "deactivate.sh", "activate.bat", "deactivate.bat", "activate.ps1", "deactivate.ps1"):
    shutil.copy(root / "conda" / "edelweissfe-dev" / script, output / script)
(output / "recipe.yaml").write_text(yaml.safe_dump(recipe, sort_keys=False, width=200))
print(f"edelweissfe-dev {version} for {platform}: {len(packages)} pinned packages")
