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
# Created on Thu May 12 18:35:44 2022

# @author: Matthias Neuner

import numpy as np

from edelweissfe.stepactions.base.stepactionbase import StepActionBase
from edelweissfe.timesteppers.timestep import TimeStep
from edelweissfe.utils.inputlanguage import InputLanguage

"""
Indirect (displacement) controller for the NISTArcLength solver
uses a ring to control the contraction, e.g., for tunneling simulations.

Currently 2D only!

The center is autotically computed from the bounding node coordinates.
"""


inputLanguage = InputLanguage()

# Register this step action for all available step types. This requires the step type
# modules to be imported before the step actions, as done in the input file parser.
modules = inputLanguage["step"].modules if "step" in inputLanguage else []

documentation = []

for module in modules:
    kw = module.addOptionalKeyword(
        "indirectcontractioncontrol",
        "Indirect (displacement) controller for the NISTArcLength solver using a ring to control the contraction, e.g., for tunneling simulations.",
    )
    kw.addRequiredArg("name", "Name of the step action.", str)
    kw.addRequiredArg("contractionNSet", "The node set defining the contraction ring", str)
    kw.addRequiredArg("L", "Final distance (e.g. crack opening)", float)
    kw.addOptionalArg("exportCVector", "File to export the computed c vector", str, None)
    kw.addOptionalArg("absolute", "Use absolute formulation", bool, True)

    documentation.append(kw)


class StepAction(StepActionBase):
    identification = "IndirectControl"

    def __init__(self, name, action, jobInfo, model, fieldOutputController, journal):
        self.name = name
        self.journal = journal

        self.currentL0 = 0.0
        self._currentL = 0.0

        self.L = action["L"]

        self.generateCVector(action, jobInfo, model, fieldOutputController, journal)

        if action["exportCVector"] is not None:
            np.savetxt(action["exportCVector"] + ".csv", self.cVector)

        self.absolute = action["absolute"]

    def _getIdcsInDofVector(self, dofManager) -> np.ndarray:
        """Determine the indices of the contraction ring displacements in the dof vector.

        Parameters
        ----------
        dofManager
            The dof manager of the current equation system.

        Returns
        -------
        np.ndarray
            The indices in the dof vector.
        """

        return np.hstack(
            [dofManager.idcsOfFieldVariablesInDofVector[n.fields["displacement"]][:2] for n in self.contractionNSet]
        )

    def computeDDLambda(self, dU, ddU_0, ddU_f, timeStep: TimeStep, dofManager):
        idcs = self._getIdcsInDofVector(dofManager)

        dL = timeStep.stepProgressIncrement * self.L

        ddLambda = (dL - self.cVector.dot(dU[idcs] + ddU_0[idcs])) / self.cVector.dot(ddU_f[idcs])
        return ddLambda

    def finishIncrement(self, U, dU, dLambda, timeStep: TimeStep, dofManager):
        idcs = self._getIdcsInDofVector(dofManager)
        self._currentL = self.cVector.dot(U[idcs] + dU[idcs])

    def applyAtStepEnd(self, model):
        self.currentL0 = self._currentL

    def updateStepAction(self, action, jobInfo, model, fieldOutputController, journal):
        if self.absolute:
            self.L = action["L"] - self.currentL0
        else:
            self.L = action["L"]

        self.generateCVector(action, jobInfo, model, fieldOutputController, journal)

    def generateCVector(self, action, jobInfo, model, fieldOutputController, journal):
        contractionNSet = model.nodeSets[action["contractionNSet"]]

        nNodes = len(contractionNSet)

        allCoordinates = np.array([n.coordinates for n in contractionNSet])

        x_min = np.min(allCoordinates[:, 0])
        x_max = np.max(allCoordinates[:, 0])
        y_min = np.min(allCoordinates[:, 1])
        y_max = np.max(allCoordinates[:, 1])

        x_center = 0.5 * (x_max + x_min)
        y_center = 0.5 * (y_max + y_min)

        cVector = []

        for n in contractionNSet:
            vec_n_to_center = np.array([x_center - n.coordinates[0], y_center - n.coordinates[1]])
            norm_vec_n_to_center = np.linalg.norm(vec_n_to_center)

            vec_n_to_center_normalized = vec_n_to_center / norm_vec_n_to_center

            cVector.append(vec_n_to_center_normalized)

        self.cVector = np.hstack(cVector)

        # dividing c vector to make 'average' contraction of ring:
        self.cVector *= 1.0 / nNodes

        self.contractionNSet = contractionNSet
