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
 * a nonlinear implicit dynamic solver (NID, Newmark-beta time integration),
 * a parallel nonlinear implicit static solver (NISTParallel),
 * a parallel nonlinear explicit static solver (NESTParallel),
 * a parallel nonlinear explicit dynamic solver (NEDParallel),
 * a parallel nonlinear implicit dynamic solver (NIDParallel),
 * and a parallel arc length solver (NISTPArcLength).

Choose the solver in the ``*solver`` definition:

.. code-block:: edelweiss

    *solver, name=mySolver, solver=NISTParallel
"""

from edelweissfe.config import registry


def getSolverByName(name: str) -> type:
    """Get the class type of the requested solver.

    Resolved through the registry (``solver`` category), so a solver contributed by another package
    (an entry point, or an in-process :func:`~edelweissfe.config.registry.register` call) is found
    like a built-in one. An unknown name raises
    :class:`~edelweissfe.config.registry.RegistryLookupError` naming the available solvers.

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
