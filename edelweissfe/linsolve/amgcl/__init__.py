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

"""Registry-facing factory for the AMGCL algebraic-multigrid solver.

The implementation lives in the Cython extension :mod:`edelweissfe.linsolve.amgcl.amgcl`, which is
optional -- it needs the AMGCL headers at build time. The factory below therefore imports it lazily;
see :func:`createSolver`.
"""

from collections.abc import Callable, Mapping


def createSolver(opts) -> Callable:
    """Create an AMGCL-backed linear solver.

    The factory the ``linsolver`` registry category resolves for the name ``amgcl``: every ``linsolve`` subpackage
    exposes this one signature, so that a third party can contribute a linear solver through an entry point.

    Parameters
    ----------
    opts
        The linear-solver options parsed from the solver's ``linsolverConfigFile``. Copied into a
        plain ``dict`` and handed to :class:`~edelweissfe.linsolve.amgcl.amgcl.PyAMGCLSolver`, which
        forwards it to AMGCL's own runtime parameter tree (solver/preconditioner selection and their
        tolerances). A non-mapping ``opts`` -- e.g. the ``""`` the implicit-static solver passes when
        no configuration file is given -- becomes an empty ``dict``, i.e. AMGCL's defaults.

    Returns
    -------
    Callable
        The bound :meth:`~edelweissfe.linsolve.amgcl.amgcl.PyAMGCLSolver.solve` method of a
        :class:`~edelweissfe.linsolve.amgcl.amgcl.PyAMGCLSolver` configured from ``opts``, i.e. a
        ``(A, b) -> x`` callable holding onto those options.

    Raises
    ------
    ImportError
        If the optional ``amgcl`` extension was not built.
    """

    # Imported inside the function body, not at module scope: this extension is optional, so at
    # module scope its absence would break anyone who merely resolves a `linsolver` registry name
    # rather than this one.
    from edelweissfe.linsolve.amgcl.amgcl import PyAMGCLSolver

    amgclOpts = dict(opts) if isinstance(opts, Mapping) else {}

    return PyAMGCLSolver(amgclOpts).solve
