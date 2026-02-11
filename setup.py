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
import sys
from os.path import expanduser, join

import numpy
from Cython.Build import build_ext, cythonize
from setuptools import setup
from setuptools.extension import Extension

directives = {
    "boundscheck": False,
    "wraparound": False,
    "nonecheck": False,
    "initializedcheck": False,
}

default_install_prefix = sys.prefix
print("*" * 80)
print("EdelweissFE setup")
print("System prefix: " + sys.prefix)
print("*" * 80)

marmot_dir = expanduser(os.environ.get("MARMOT_INSTALL_DIR", default_install_prefix))
mkl_include = expanduser(os.environ.get("MKL_INCLUDE_DIR", join(default_install_prefix, "include")))

print("Marmot install directory (overwrite via environment var. MARMOT_INSTALL_DIR):")
print(marmot_dir)
print("MKL include directory (overwrite via environment var. MKL_INCLUDE_DIR):")
print(mkl_include)
print("*" * 80)

print("Gather the extension for the MarmotElement base element, linked to the Marmot library")
extensions = [
    Extension(
        "*",
        sources=["edelweissfe/elements/marmotelement/element.pyx"],
        include_dirs=[join(marmot_dir, "include"), numpy.get_include()],
        libraries=["Marmot"],
        library_dirs=[join(marmot_dir, "lib")],
        runtime_library_dirs=[join(marmot_dir, "lib")],
        language="c++",
    )
]

# print(
#     "Gather the extension for the single quadrature point element using MarmotMaterials, linked to the Marmot library"
# )
# extensions += [
#     Extension(
#         "*",
#         sources=[
#             "edelweissfe/elements/marmotsingleqpelement/marmot.pyx",
#         ],
#         include_dirs=[join(marmot_dir, "include"), numpy.get_include()],
#         libraries=["Marmot"],
#         library_dirs=[join(marmot_dir, "lib")],
#         runtime_library_dirs=[join(marmot_dir, "lib")],
#         language="c++",
#         extra_compile_args=["-O3", "-std=c++20"],
#     )
# ]

extensions += [
    Extension(
        "*",
        sources=[
            "edelweissfe/elements/marmotsingleqpelement/marmotmaterialhypoelasticwrapper.pyx",
        ],
        include_dirs=[join(marmot_dir, "include"), numpy.get_include()],
        libraries=["Marmot"],
        library_dirs=[join(marmot_dir, "lib")],
        runtime_library_dirs=[join(marmot_dir, "lib")],
        language="c++",
        extra_compile_args=["-O3", "-std=c++20"],
    )
]

# extensions += [
#     Extension(
#         "*",
#         sources=[
#             "edelweissfe/elements/marmotsingleqpelement/marmotmaterialgradientenhancedhypoelasticwrapper.pyx",
#         ],
#         include_dirs=[join(marmot_dir, "include"), numpy.get_include()],
#         libraries=["Marmot"],
#         library_dirs=[join(marmot_dir, "lib")],
#         runtime_library_dirs=[join(marmot_dir, "lib")],
#         language="c++",
#     )
# ]

print("Gather the extension for the fast element result collector")
extensions += [
    Extension(
        "*",
        ["edelweissfe/utils/elementresultcollector.pyx"],
        include_dirs=[numpy.get_include()],
        language="c++",
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
        extra_compile_args=["-O3", "-std=c++20", "-march=native", "-fopenmp"],
        extra_link_args=["-fopenmp"],
    )
]

print("Gather the extension for the NISTParallel solver")
extensions += [
    Extension(
        "*",
        sources=["edelweissfe/solvers/nonlinearimplicitstaticparallelmk2.pyx"],
        include_dirs=[numpy.get_include()],
        language="c++",
        extra_compile_args=[
            "-fopenmp",
            "-Wno-maybe-uninitialized",
        ],
        extra_link_args=["-fopenmp"],
    )
]

print("Gather the extension for the NISTParallel (MarmotElements only) solver")
extensions += [
    Extension(
        "*",
        sources=["edelweissfe/solvers/nonlinearimplicitstaticparallel.pyx"],
        include_dirs=[numpy.get_include()] + [join(marmot_dir, "include")],
        language="c++",
        extra_compile_args=[
            "-fopenmp",
            "-Wno-maybe-uninitialized",
        ],
        extra_link_args=["-fopenmp"],
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
        libraries=[
            "mkl_gnu_thread",
            "mkl_core",
            "mkl_rt",
            "mkl_gf_lp64",
            "iomp5",
        ],
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

print("Gather the AMGCL interface")
extensions += [
    Extension(
        "*",
        sources=["edelweissfe/linsolve/amgcl/amgcl.pyx"],
        include_dirs=[numpy.get_include(), join(default_install_prefix, "include"), "."],
        language="c++",
        extra_compile_args=["-std=c++11", "-fopenmp", "-O3"],
        extra_link_args=["-fopenmp"],
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
            "iomp5",
            "suitesparseconfig",
        ],
        language="c",
        extra_compile_args=[
            "-fopenmp",
            "-Wno-maybe-uninitialized",
        ],
        extra_link_args=["-fopenmp"],
    )
]

print("Now compile!")


# overwrite build_ext to make all extensions optional
class optional_build_ext(build_ext):
    def build_extension(self, ext):
        try:
            super().build_extension(ext)
        except Exception as e:
            print(f"Extension {ext.name} could not be built:")
            print(e)


setup(
    cmdclass={"build_ext": optional_build_ext},
    ext_modules=cythonize(extensions, compiler_directives=directives, annotate=True, language_level=3),
)

print("Finish!")
