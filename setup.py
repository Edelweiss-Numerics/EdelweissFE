# -*- coding: utf-8 -*-
#  ---------------------------------------------------------------------
#
#  _____    _      _              _         _____ _____
# | ____|__| | ___| |_      _____(_)___ ___|  ___| ____|
# |  _| / _` |/ _ \ \ \ /\ / / _ \ / __/ __| |_  |  _|
# | |__| (_| |  __/ |\ V  V /  __/ \__ \__ \  _| | |___
# |_____\__,_|\___|_| \_/\_/ \___|_|___/___/_|   |_____|
#
#
#  Unit of Strength of Materials and Structural Analysis
#  University of Innsbruck,
#  2017 - today
#
#  Matthias Neuner matthias.neuner@uibk.ac.at
#
#  This file is part of EdelweissFE.
#
#  This library is free software; you can redistribute it and/or
#  modify it under the terms of the GNU Lesser General Public
#  License as published by the Free Software Foundation; either
#  version 2.1 of the License, or (at your option) any later version.
#
#  The full text of the license can be found in the file LICENSE.md at
#  the top level directory of EdelweissFE.
#  ---------------------------------------------------------------------
"""
Created on Thu May 21 14:23:14 2015

@author: c8441141
"""
import os
import pathlib
import shutil
import sys
from os.path import expanduser, join

import numpy
from Cython.Build import build_ext, cythonize
from setuptools import setup
from setuptools.extension import Extension

# The platform-dependent build settings are shared with downstream packages (EdelweissMeshfree), which import
# them from the installed package. Here, they are imported from the source tree.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from edelweissfe.utils.extensionbuild import (  # noqa: E402
    compile_flags,
    get_arch_flags,
    is_windows,
    link_flags,
    native_prefix,
    runtime_library_dirs,
)

directives = {
    "boundscheck": False,
    "wraparound": False,
    "nonecheck": False,
    "initializedcheck": False,
    # Declare all extensions safe for free-threading CPython builds. Without this,
    # importing any of them silently re-enables the GIL process-wide, which disables
    # the thread-parallel element loops of the parallel solvers.
    "freethreading_compatible": True,
}

print("*" * 80)
print("EdelweissFE setup")
print("System prefix: " + sys.prefix)
print("*" * 80)

marmot_dir = expanduser(os.environ.get("MARMOT_INSTALL_DIR", native_prefix))
# Name of the generated package file recording marmot_dir, read by edelweissfe/__init__.py.
marmot_install_dir_file = "marmot_install_dir.txt"
mkl_include = expanduser(os.environ.get("MKL_INCLUDE_DIR", join(native_prefix, "include")))
eigen_include = expanduser(os.environ.get("EIGEN_INCLUDE_DIR", join(native_prefix, "include", "eigen3")))
# AMGCL specifically defaults to no arch flags (see the comment at its Extension below) but
# still honors an explicit EDELWEISSFE_ARCH_FLAGS override, consistent with every other
# extension above -- only the *default* differs, not the override mechanism.
amgcl_arch_flags = get_arch_flags(default="")
arch_flags = get_arch_flags()
print("Marmot install directory (overwrite via environment var. MARMOT_INSTALL_DIR):")
print(marmot_dir)
print("MKL include directory (overwrite via environment var. MKL_INCLUDE_DIR):")
print(mkl_include)
print("Eigen include directory (overwrite via environment var. EIGEN_INCLUDE_DIR):")
print(eigen_include)
print("Architecture compile flags (overwrite via environment var. EDELWEISSFE_ARCH_FLAGS):")
print(arch_flags)
print("AMGCL architecture compile flags (overwrite via the same environment variable; defaults to none, see below):")
print(amgcl_arch_flags)

# Extension build failures are tolerated by optional_build_ext below, so a wrong include
# directory would otherwise only show up as a missing module much later. The header is looked
# for in every directory which ends up on the include path of the respective extension, since
# Eigen is sometimes installed next to the Marmot headers rather than into its own eigen3
# subdirectory.
for description, header, searchPath, variable in [
    ("Marmot", join("Marmot", "MarmotMaterialHypoElastic.h"), [join(marmot_dir, "include")], "MARMOT_INSTALL_DIR"),
    ("Eigen", join("Eigen", "Dense"), [eigen_include, join(marmot_dir, "include")], "EIGEN_INCLUDE_DIR"),
]:
    if not any(os.path.exists(join(candidate, header)) for candidate in searchPath):
        print("!" * 80)
        print(
            "WARNING: {:} was not found ({:} is in none of {:}).\n"
            "         Extensions depending on it will NOT be built.\n"
            "         Set the environment variable {:} to the correct location.".format(
                description, header, ", ".join(searchPath), variable
            )
        )
        print("!" * 80)

print("*" * 80)


print("Gather the extension for the MarmotElement base element, linked to the Marmot library")
extensions = [
    Extension(
        "*",
        sources=["edelweissfe/elements/marmotelement/element.pyx"],
        include_dirs=[join(marmot_dir, "include"), numpy.get_include()],
        libraries=["Marmot"],
        library_dirs=[join(marmot_dir, "lib")],
        runtime_library_dirs=runtime_library_dirs(join(marmot_dir, "lib")),
        language="c++",
        extra_compile_args=compile_flags(arch=arch_flags),
    )
]

print("Gather the extensions for the point-wise Marmot material interfaces")
marmot_material_dir = join("edelweissfe", "materials", "marmot")
for marmot_material_source in [
    "marmothypoelastic.pyx",
    "marmotgradientenhancedhypoelastic.pyx",
]:
    extensions += [
        Extension(
            "*",
            sources=[join(marmot_material_dir, marmot_material_source)],
            include_dirs=[
                join(marmot_dir, "include"),
                numpy.get_include(),
                eigen_include,
                # for the C++ shim living next to the sources
                marmot_material_dir,
            ],
            libraries=["Marmot"],
            library_dirs=[join(marmot_dir, "lib")],
            runtime_library_dirs=runtime_library_dirs(join(marmot_dir, "lib")),
            language="c++",
            extra_compile_args=compile_flags(cxx20=True),
        )
    ]

print("Gather the extension for the fast element result collector")
extensions += [
    Extension(
        "*",
        ["edelweissfe/utils/elementresultcollector.pyx"],
        include_dirs=[numpy.get_include()],
        language="c++",
        extra_compile_args=compile_flags(arch=arch_flags),
    )
]

print("Gather the extension for the fast CSR matrix generator")
extensions += [
    Extension(
        "*",
        ["edelweissfe/numerics/csrgenerator.pyx"],
        include_dirs=[numpy.get_include()],
        language="c++",
    )
]

print("Gather the extension for the even faster CSR matrix v2 generator")
extensions += [
    Extension(
        "*",
        ["edelweissfe/numerics/csrgeneratorv2.pyx"],
        include_dirs=[numpy.get_include()],
        language="c++",
        extra_compile_args=compile_flags(cxx20=True, openmp=True, arch=arch_flags),
        extra_link_args=link_flags(openmp=True),
    )
]

print("Gather the extensions for fast dirichlet application")
extensions += [
    Extension(
        "*",
        sources=["edelweissfe/solvers/base/dirichlet.pyx"],
        include_dirs=[numpy.get_include()],
        language="c++",
        extra_compile_args=compile_flags(openmp=True, arch=arch_flags, gcc_only=["-Wno-maybe-uninitialized"]),
        extra_link_args=link_flags(openmp=True),
    )
]

print("Gather the pardiso interface")
extensions += [
    Extension(
        "*",
        sources=[
            "edelweissfe/linsolve/pardiso/pardiso.pyx",
        ],
        include_dirs=[
            numpy.get_include(),
            mkl_include,
        ],
        # On Windows, MKL's single dynamic library (mkl_rt) selects threading and interface at runtime.
        libraries=(
            ["mkl_rt"]
            if is_windows
            else [
                "mkl_gnu_thread",
                "mkl_core",
                "mkl_rt",
                "mkl_gf_lp64",
                "iomp5",
            ]
        ),
        library_dirs=[join(native_prefix, "lib")],
        language="c++",
    )
]


print("Gather the Panua pardiso interface")
extensions += [
    Extension(
        "*",
        sources=[
            "edelweissfe/linsolve/panuapardiso/panuapardiso.pyx",
        ],
        include_dirs=[
            numpy.get_include(),
        ],
        libraries=[
            "pardiso",
        ],
        language="c++",
        extra_link_args=["-fopenmp", "-lgfortran", "-lpthread", "-lm"],
        optional=True,
    )
]
if is_windows:  # Panua PARDISO's link line is GCC/Linux-specific
    extensions.pop()

print("Gather the AMGCL interface")
# No arch flags by default: -march=native measured ~40% SLOWER here on Skylake-SP, where AMGCL's
# sustained 512-bit inner loops trigger that generation's package-wide AVX-512 downclock. Set
# EDELWEISSFE_ARCH_FLAGS explicitly to opt in.
extensions += [
    Extension(
        "*",
        sources=["edelweissfe/linsolve/amgcl/amgcl.pyx"],
        include_dirs=[numpy.get_include(), join(native_prefix, "include"), "."],
        language="c++",
        # MSVC's default language standard (C++14) already covers AMGCL's C++11 requirement.
        extra_compile_args=([] if is_windows else ["-std=c++11"]) + compile_flags(openmp=True, arch=amgcl_arch_flags),
        extra_link_args=link_flags(openmp=True),
    )
]

print("Gather the KLU interface")
extensions += [
    Extension(
        "*",
        sources=[
            "edelweissfe/linsolve/klu/klu.pyx",
            "edelweissfe/linsolve/klu/kluInterface.c",
        ],
        include_dirs=[
            numpy.get_include(),
            join(native_prefix, "include"),
            join(native_prefix, "include", "suitesparse"),
        ],
        libraries=[
            "klu",
            "btf",
            "amd",
            "colamd",
            "metis",
            "cholmod",
            "camd",
            "ccolamd",
            *([] if is_windows else ["iomp5"]),
            "suitesparseconfig",
        ],
        library_dirs=[join(native_prefix, "lib")],
        language="c",
        extra_compile_args=compile_flags(optimize=False, openmp=True, gcc_only=["-Wno-maybe-uninitialized"]),
        extra_link_args=link_flags(openmp=True),
    )
]

print("Now compile!")


class optional_build_ext(build_ext):
    def build_extensions(self):
        self.successful_extensions = []

        for ext in self.extensions:
            try:
                self.build_extension(ext)
                self.successful_extensions.append(ext.name)
                print(f"[OK] Built extension: {ext.name}")
            except Exception as e:
                print(f"[FAIL] Could not build {ext.name}: {e}")

        self.write_package_file("built_extensions.log", "\n".join(self.successful_extensions) + "\n")
        # The Marmot installation the extensions were built against. On Windows, edelweissfe/__init__.py adds its
        # bin directory to the DLL search path, the counterpart of the runtime library path baked in elsewhere.
        self.write_package_file(marmot_install_dir_file, os.path.abspath(marmot_dir) + "\n")

    def write_package_file(self, name, content):
        """Write a generated file into the package, both in the source tree and in the build directory."""
        source_file = pathlib.Path("edelweissfe") / name
        source_file.parent.mkdir(parents=True, exist_ok=True)
        source_file.write_text(content, encoding="utf-8")

        build_file = pathlib.Path(self.build_lib) / "edelweissfe" / name
        build_file.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_file, build_file)

        print(f"Wrote {source_file} and {build_file}")


setup(
    cmdclass={"build_ext": optional_build_ext},
    ext_modules=cythonize(extensions, compiler_directives=directives, annotate=True, language_level=3),
    include_package_data=True,
    package_data={
        "edelweissfe": ["built_extensions.log", marmot_install_dir_file],
        # Downstream packages (e.g. EdelweissFD) compile their own Cython extensions against
        # the point-wise Marmot material interfaces, so the declarations and C++ shims they
        # cimport/include have to be part of the installed distribution, not just the source
        # checkout.
        "edelweissfe.materials.marmot": ["*.pxd", "*.h"],
    },
)

print("Finish!")
