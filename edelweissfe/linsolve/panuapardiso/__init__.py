#!/usr/bin/env python3
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

"""Registry-facing factory for the Panua PARDISO solver.

The implementation lives in the Cython extension
:mod:`edelweissfe.linsolve.panuapardiso.panuapardiso`, which is optional -- it needs the Panua
PARDISO library at build time. The factory below therefore imports it lazily; see
:func:`createSolver`.
"""

from collections.abc import Callable


def createSolver(opts) -> Callable:
    """Create a Panua-PARDISO-backed linear solver.

    The factory the ``linsolver`` registry category resolves for the name ``panuapardiso``: every ``linsolve``
    subpackage exposes this one signature, so that a third party can contribute a linear solver through an entry
    point.

    Parameters
    ----------
    opts
        The linear-solver options parsed from the solver's ``linsolverConfigFile``. Not consulted:
        this backend takes no options. It is still accepted so that every ``createSolver`` presents
        the identical signature.

    Returns
    -------
    Callable
        :func:`~edelweissfe.linsolve.panuapardiso.panuapardiso.panuaPardisoSolve`, i.e. a stateless
        ``(A, b) -> x`` function.

    Raises
    ------
    ImportError
        If the optional ``panuapardiso`` extension was not built.
    """

    # Imported inside the function body, not at module scope: this extension is optional and
    # genuinely absent in most installs, so at module scope its absence would break anyone who
    # merely resolves a `linsolver` registry name rather than this one.
    from edelweissfe.linsolve.panuapardiso.panuapardiso import panuaPardisoSolve

    return panuaPardisoSolve
