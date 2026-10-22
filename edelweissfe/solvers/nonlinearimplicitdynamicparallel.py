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
"""
Parallel implementation of the NID solver.

The dynamics add nothing to the element loop, so the parallel variant is the serial one with
:class:`~edelweissfe.solvers.nonlinearimplicitstaticparallel.NISTParallel`'s thread-parallel
``computeElements`` mixed in. The method resolution order ``NIDParallel -> NID -> NISTParallel ->
NIST`` routes every ``super()`` call of NID through NISTParallel, so both the Newton loop and the
initial-acceleration solve evaluate the elements in parallel.
"""

from edelweissfe.solvers.nonlinearimplicitdynamic import NonlinearImplicitDynamic
from edelweissfe.solvers.nonlinearimplicitstaticparallel import NISTParallel


class NIDParallel(NonlinearImplicitDynamic, NISTParallel):
    """This is the parallel Nonlinear Implicit Dynamic -- solver (``NIDParallel``).

    Parameters
    ----------
    jobInfo
        A dictionary containing the job information.
    journal
        The journal instance for logging.
    """

    identification = "NIDPSolver"
