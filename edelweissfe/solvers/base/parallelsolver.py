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
#  Alexander Dummer alexander.dummer@uibk.ac.at
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
What every parallel solver adds to its serial one, besides computing the elements on the threads.
"""

from edelweissfe.numerics.parallelizationutilities import (
    getNumberOfThreads,
    mapChunksInParallel,
    reportThreadAvailability,
)


class ParallelSolver:
    """The part the parallel solvers share: they report the threads they have at the start of every step, and their
    loop over independent entities other than the elements (:meth:`forEachChunk`) runs on the threads, as the elements
    are computed. It is inherited before the serial solver, e.g. ``class NEDParallel(ParallelSolver, NED)``, so that
    its methods come first and hand on to the serial solver's.
    """

    def beginStep(self, step, model, fieldOutputController, outputmanagers):
        """Report the threads available, then begin the step as the serial solver does.

        Parameters
        ----------
        step
            The step to begin.
        model
            The model.
        fieldOutputController
            The field output controller.
        outputmanagers
            The output managers.
        """

        reportThreadAvailability(getNumberOfThreads(), self.journal, self.identification)
        return super().beginStep(step, model, fieldOutputController, outputmanagers)

    def forEachChunk(self, entities: list, work):
        """Call ``work(chunk)`` for chunks of ``entities`` on the available threads, as the elements are computed (see
        :func:`~edelweissfe.numerics.parallelizationutilities.mapChunksInParallel`); otherwise as
        :meth:`~edelweissfe.solvers.base.nonlinearsolverbase.NonlinearSolverBase.forEachChunk`.

        Parameters
        ----------
        entities
            The entities, a list.
        work
            Called for each chunk; it must write only to what belongs to the entities of its chunk.
        """

        mapChunksInParallel(entities, work)
