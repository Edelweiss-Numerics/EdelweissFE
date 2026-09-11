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

"""Registry-facing factory for the diagnostic matrix-dumping linear solver.

The implementation lives in :mod:`edelweissfe.linsolve.matrixdump.matrixdump`; see there for what
this is for.
"""

from collections.abc import Callable, Mapping

#: How many dumps are written when the configuration does not say. One contiguous Newton sequence is
#: the smallest useful capture (a single system cannot answer any sequential question), and six
#: iterations is a typical increment of a converging nonlinear analysis.
_DEFAULT_MAX_DUMPS = 6


def createSolver(opts) -> Callable:
    """Create a matrix-dumping linear solver wrapping a real one.

    The factory the ``linsolver`` registry category resolves for the name ``matrixdump``: every
    ``linsolve`` subpackage exposes this one signature.

    Parameters
    ----------
    opts
        The linear-solver options parsed from the solver's ``linsolverConfigFile``. Recognized keys,
        all optional:

        ``directory``
            Where to write the dumps; default ``"linsolveDumps"``.
        ``delegate``
            The name of the ``linsolver`` that performs the actual solves; default ``"pardiso"``.
            Resolved through the same registry as any other named solver, so the delegate can be any
            registered linear solver -- including, in principle, another wrapper.
        ``delegateOpts``
            The option mapping handed to the delegate's own factory; default ``{}``.
        ``dumpAt``
            Explicit zero-based solve ordinals to dump; default ``[]``, meaning use
            ``skipFirst``/``maxDumps``.
        ``skipFirst``
            Solves to pass through undumped when ``dumpAt`` is empty; default ``0``.
        ``maxDumps``
            Process-wide hard ceiling on dumps written; default ``6``.
        ``instances``
            Which solver instances -- i.e. which analysis steps, since a fresh linear solver is built
            per step -- are permitted to dump; default ``[]``, meaning all of them. ``[1]`` captures
            the second step only.

        As with the other factories, a non-mapping ``opts`` is tolerated (the implicit-static solver
        passes ``""`` when no configuration file is given), in which case every default applies --
        though dumping with all defaults is rarely what you want, since the interesting solves are
        not the first ones.

    Returns
    -------
    Callable
        A :class:`~edelweissfe.linsolve.matrixdump.matrixdump.MatrixDumpSolver` instance, callable as
        ``(A, b) -> x``.
    """

    # Imported inside the function body to match every other factory in this package: `getLinSolver`
    # is imported here rather than at module scope because `config.linsolve` imports the registry,
    # and the registry is what resolves *this* module -- a module-scope import would close that loop.
    from edelweissfe.config.linsolve import getLinSolverByName
    from edelweissfe.linsolve.matrixdump.matrixdump import MatrixDumpSolver

    optionMap = opts if isinstance(opts, Mapping) else {}

    delegate = getLinSolverByName(
        optionMap.get("delegate", "pardiso"),
        optionMap.get("delegateOpts", {}),
    )

    return MatrixDumpSolver(
        directory=optionMap.get("directory", "linsolveDumps"),
        delegate=delegate,
        dumpAt=[int(ordinal) for ordinal in optionMap.get("dumpAt", [])],
        skipFirst=int(optionMap.get("skipFirst", 0)),
        maxDumps=int(optionMap.get("maxDumps", _DEFAULT_MAX_DUMPS)),
        instances=[int(instance) for instance in optionMap.get("instances", [])],
    )
