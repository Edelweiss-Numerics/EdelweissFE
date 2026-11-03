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
"""Platform-dependent settings for compiling the Cython extensions of EdelweissFE.

Used by EdelweissFE's ``setup.py`` and by the ``setup.py`` of downstream packages (EdelweissMeshfree), so that
the whole stack is compiled with the same flags and the platform logic has a single implementation.

This module must only depend on the standard library: it is imported at build time, before anything else is
guaranteed to be installed.
"""

import os
import sys
from os.path import join

#: True for builds with MSVC on Windows; GCC or Clang elsewhere.
is_windows = sys.platform == "win32"

#: The prefix holding C/C++ headers and libraries of the environment. On Windows, conda installs them under
#: ``<prefix>\\Library``, not ``<prefix>``.
native_prefix = join(sys.prefix, "Library") if is_windows else sys.prefix


def get_arch_flags(default: str | None = None) -> list[str]:
    """Return the architecture compile flags, as set by the environment variable ``EDELWEISSFE_ARCH_FLAGS``.

    Without the variable, the default is ``-march=native`` (none on Windows, since MSVC has no equivalent).
    Set the variable when the build must run on other machines than the one compiling it, e.g. container images
    or conda packages (``EDELWEISSFE_ARCH_FLAGS="-march=x86-64-v3"``), or to an empty string for no flags.

    Parameters
    ----------
    default
        The flags used when the variable is not set; overrides the platform default.
    """
    if default is None:
        default = "" if is_windows else "-march=native"
    return os.environ.get("EDELWEISSFE_ARCH_FLAGS", default).split()


def compile_flags(*, optimize=True, cxx20=False, openmp=False, arch=(), gcc_only=()) -> list[str]:
    """Return the compile flags for the platform's compiler: MSVC on Windows, GCC/Clang elsewhere.

    ``gcc_only`` flags (e.g. warning switches) are dropped for MSVC. MSVC uses its classic OpenMP runtime
    (/openmp, vcomp140.dll): /openmp:llvm links Microsoft's copy of the LLVM runtime (libomp140), which aborts the
    process ("OMP: Error #15") next to conda's own LLVM runtime (libomp.dll) that other packages load. vcomp
    implements OpenMP 2.0, which covers the constructs used here (`omp simd` is excluded under MSVC).
    """
    if is_windows:
        return [
            *(["/O2"] if optimize else []),
            *(["/std:c++20"] if cxx20 else []),
            *(["/openmp"] if openmp else []),
            *arch,
        ]
    return [
        *(["-O3"] if optimize else []),
        *(["-std=c++20"] if cxx20 else []),
        *(["-fopenmp"] if openmp else []),
        *arch,
        *gcc_only,
    ]


def link_flags(*, openmp=False) -> list[str]:
    """Return the link flags for OpenMP: MSVC links its OpenMP runtime implicitly."""
    return ["-fopenmp"] if openmp and not is_windows else []


def runtime_library_dirs(*dirs) -> list[str]:
    """Return a runtime library search path; MSVC cannot embed one (Windows finds DLLs on PATH)."""
    return [] if is_windows else list(dirs)
