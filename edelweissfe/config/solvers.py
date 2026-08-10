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
# Created on Fri Feb 10 19:20:25 2017

# @author: Matthias Neuner
"""
Currently, EdelweissFE provides

 * a nonlinear implicit static solver (NIST),
 * a nonlinear explicit static solver (NEST),
 * a nonlinear explicit dynamic solver (NED),
 * a parallel nonlinear implicit static solver (NISTParallel),
 * a parallel nonlinear explicit static solver (NESTParallel),
 * a parallel nonlinear explicit dynamic solver (NEDParallel),
 * and a parallel arc length solver (NISTPArcLength).

Choose the solver in the ``*solver`` definition:

.. code-block:: edelweiss

    *solver, name=mySolver, solver=NISTParallel
"""

from edelweissfe.config import registry


def getSolverByName(name: str) -> type:
    """Get the class type of the requested solver.

    Resolved through the L3 registry (``solver`` category) rather than through a hand-maintained
    table private to this module. Such a table could only ever list solvers living *inside* this
    package, so an external package -- EdelweissMeshfree, a plugin -- had no way to contribute one;
    going through the registry means a built-in, an entry point and an in-process
    :func:`~edelweissfe.config.registry.register` call are all equally reachable here. An unknown
    name now raises :class:`~edelweissfe.config.registry.RegistryLookupError` naming the available
    solvers, instead of a ``KeyError``.

    **Solver names are now case-insensitive, deliberately.** This resolver was the one
    case-*sensitive* registry in the codebase: it indexed a table with CamelCase keys and then read
    the class off the module under the *same* string, so the name doubled as the class
    attribute name and e.g. ``"nist"`` failed twice over, while 12 of the 13 legacy ``config/*.py``
    registries already casefolded the name at the resolver. That audit amends rule (c) to sanction
    this: a name must not resolve differently depending on which front-end it arrived through, and
    the registry is reached by callers with no ``.inp`` parser in the loop. The change is strictly
    more permissive, so no existing input file changes meaning.

    Parameters
    ----------
    name
        The name of the solver to load (case insensitive).

    Returns
    -------
    type
        The solver class type.
    """

    solverClass, _ = registry.lookup("solver", name)

    return solverClass
