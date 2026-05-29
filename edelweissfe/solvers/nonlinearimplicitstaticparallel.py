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
Parallel implementation of the NIST solver.
"""

import edelweissfe.utils.performancetiming as performancetiming
from edelweissfe.numerics.parallelizationutilities import getNumberOfThreads
from edelweissfe.solvers.base.parallelelementcomputation import (
    computeElementsInParallel,
)
from edelweissfe.solvers.nonlinearimplicitstatic import NIST


class NISTParallel(NIST):

    identification = "NISTPSolver"

    def solveStep(self, step, model, fieldOutputController, outputmanagers):

        self.journal.message("Using {:} threads".format(getNumberOfThreads()), self.identification)
        return super().solveStep(step, model, fieldOutputController, outputmanagers)

    @performancetiming.timeit("elements")
    def computeElements(self, elements, Un1, dU, P, K, F, timeStep):
        return computeElementsInParallel(elements, Un1, dU, P, K, F, timeStep)
