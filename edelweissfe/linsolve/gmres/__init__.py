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

"""Registry-facing factory for the preconditioned GMRES solver.

The implementation lives in :mod:`edelweissfe.linsolve.gmres.gmres`, which is pure Python but
imports ``pyamg`` at module scope -- an optional dependency. The factory below therefore imports it
lazily; see :func:`createSolver`.
"""

from collections.abc import Callable


def createSolver(opts) -> Callable:
    """Create a GMRES-backed linear solver.

    The factory the ``linsolver`` registry category resolves for the name ``gmres`` (see
    ``PLAN_INPUT_SYSTEM.md`` §9): every ``linsolve`` subpackage exposes this one signature, so that
    a third party can contribute a linear solver through an entry point.

    Parameters
    ----------
    opts
        The linear-solver options parsed from the solver's ``linsolverConfigFile``, handed to
        :class:`~edelweissfe.linsolve.gmres.gmres.Gmres` unchanged. It reads ``opts["precondopts"]``
        (the ``pyamg`` smoothed-aggregation preconditioner options) and ``opts["linsolveopts"]``
        (the :func:`scipy.sparse.linalg.gmres` options), each with its own default, and tolerates a
        falsy ``opts`` -- including the ``""`` the implicit-static solver passes when no
        configuration file is given.

    Returns
    -------
    Callable
        The bound :meth:`~edelweissfe.linsolve.gmres.gmres.Gmres.gmresSolve` method of a
        :class:`~edelweissfe.linsolve.gmres.gmres.Gmres` configured from ``opts``, i.e. a
        ``(A, b) -> x`` callable holding onto those options.

    Raises
    ------
    ImportError
        If the optional ``pyamg`` dependency is not installed.
    """

    # Imported inside the function body, not at module scope: `pyamg` is an optional dependency, so
    # at module scope its absence would break anyone who merely resolves a `linsolver` registry name
    # rather than this one.
    from edelweissfe.linsolve.gmres.gmres import Gmres

    return Gmres(opts).gmresSolve
