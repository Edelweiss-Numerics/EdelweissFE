#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#  ---------------------------------------------------------------------
#  EdelweissFE - Numerics Module
#  ---------------------------------------------------------------------

"""Numerics: DOF management and sparse system-matrix assembly."""

from os.path import abspath, dirname


def get_include() -> str:
    """Directory holding the C++ headers of this package.

    ``_csrcore.h`` declares ``CSRDirectAssembler``, whose ``scatterBlock`` is called from inside the
    threaded entity loops of downstream packages (EdelweissMeshfree). They add this directory to their
    extension ``include_dirs`` so that the scatter has exactly one implementation, rather than each
    consumer reimplementing the addressing against raw memoryviews.

    Mirrors the convention of ``numpy.get_include()``.
    """
    return dirname(abspath(__file__))
