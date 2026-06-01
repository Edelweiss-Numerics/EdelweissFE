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
# Created on Mon Jan 23 13:03:09 2017

# @author: Matthias Neuner

import numpy as np
import sympy as sp

from edelweissfe.config.phenomena import getFieldSize
from edelweissfe.stepactions.base.dirichletbase import DirichletBase
from edelweissfe.steps.adaptivestep import InputLanguage
from edelweissfe.timesteppers.timestep import TimeStep

"""
Standard Dirichlet boundary condition.
If not modified in subsequent steps, the BC is held constant.
"""

inputLanguage = InputLanguage()

modules = [
    inputLanguage["step"].getModule("adaptive"),
    inputLanguage["step"].getModule("adaptiveForExplicitSimulations"),
]

documentation = []

for module in modules:
    kw = module.addOptionalKeyword("dirichlet", "Standard Dirichlet boundary condition.")
    kw.addRequiredArg("name", "Name of the step action.", str)
    kw.addRequiredArg("nSet", "The node set for application of the boundary condition.", str)
    kw.addRequiredArg("field", "Field for which the boundary condition is active.", str)

    kw.addOptionalArg("1", "Prescribe first component of field.", float, None)
    kw.addOptionalArg("2", "Prescribe second component of field.", float, None)
    kw.addOptionalArg("3", "Prescribe third component of field.", float, None)
    kw.addOptionalArg("4", "Prescribe fourth component of field.", float, None)
    kw.addOptionalArg("5", "Prescribe fifth component of field.", float, None)
    kw.addOptionalArg("6", "Prescribe sixth component of field.", float, None)

    kw.addOptionalArg(
        "components",
        "Prescribe values using a numpy ndarray for representation; use 'x' for ignored values.",
        str,
        None,
    )
    kw.addOptionalArg("analyticalField", "Scales the defined boundary condition", str, None)
    kw.addOptionalArg("f(t)", "Define an amplitude in the step progress interval [0...1]", str, None)

    documentation.append(kw)

    kw = module.addOptionalKeyword("updateDirichlet", "Update a previously defined dirichlet definition.")
    kw.addRequiredArg("name", "Name of the step action to update.", str)
    # kw.addRequiredArg("nSet", "The node set for application of the boundary condition.", str)
    # kw.addRequiredArg("field", "Field for which the boundary condition is active.", str)

    kw.addOptionalArg("1", "Prescribe first component of field.", float, None)
    kw.addOptionalArg("2", "Prescribe second component of field.", float, None)
    kw.addOptionalArg("3", "Prescribe third component of field.", float, None)
    kw.addOptionalArg("4", "Prescribe fourth component of field.", float, None)
    kw.addOptionalArg("5", "Prescribe fifth component of field.", float, None)
    kw.addOptionalArg("6", "Prescribe sixth component of field.", float, None)

    kw.addOptionalArg(
        "components",
        "Prescribe values using a numpy ndarray for representation; use 'x' for ignored values.",
        str,
        None,
    )
    kw.addOptionalArg("analyticalField", "Scales the defined boundary condition", str, None)
    kw.addOptionalArg("f(t)", "Define an amplitude in the step progress interval [0...1]", str, None)

    documentation.append(kw)


class StepAction(DirichletBase):
    """Dirichlet boundary condition, based on a node set"""

    def __init__(self, name, action, jobInfo, model, fieldOutputController, journal):
        self.name = name

        self.field = action["field"]

        self.action = action
        self.nSet = model.nodeSets[action["nSet"]]
        self.fieldSize = getFieldSize(self.field, model.domainSize)
        self.possibleComponents = [str(i + 1) for i in range(self.fieldSize)]

        self._components = None

        self.updateStepAction(action, jobInfo, model, fieldOutputController, journal)

    @property
    def components(
        self,
    ):
        return self._components

    def applyAtStepEnd(self, model):
        self.active = False

    def updateStepAction(self, action, jobInfo, model, fieldOutputController, journal):
        self.active = True

        self.action = action

        if action["components"] is not None:
            action = self._getDirectionsFromComponents(action)

        components = [i for i, direction in enumerate(self.possibleComponents) if action[direction] is not None]

        values = {
            i: float(action[direction])
            for i, direction in enumerate(self.possibleComponents)
            if action[direction] is not None
        }

        self.delta = np.tile(list(values.values()), (len(self.nSet), 1))

        # for i, node in enumerate(self.nSet):
        if action["analyticalField"] is not None:
            self.analyticalField = model.analyticalFields[action["analyticalField"]]
            for i, node in enumerate(self.nSet):
                self.delta[i, :] *= self.analyticalField.evaluateAtCoordinates(node.coordinates)[0][0]

        self._components = components

        self.amplitude = self._getAmplitude(action)

    def getDelta(self, timeStep: TimeStep):
        if self.active:
            return self.delta * (
                self.amplitude(timeStep.stepProgress)
                - (self.amplitude(timeStep.stepProgress - timeStep.stepProgressIncrement))
            )
        else:
            return self.delta * 0.0

    def _getDirectionsFromComponents(self, action: dict) -> dict:
        """Determine the direction components from a numpy array representation.

        Parameters
        ----------
        action
            The dictionary defining this step action.

        Returns
        -------
        dict
            The updated dictionary defining this step action containing the directional definitions.
        """

        components = np.array(eval(action["components"].replace("x", "np.nan")), dtype=float)

        for i, t in enumerate(components):
            if not np.isnan(t):
                action[str(i + 1)] = t

        return action

    def _getAmplitude(self, action: dict) -> callable:
        """Determine the amplitude for the step, depending on a potentially specified function.

        Parameters
        ----------
        action
            The dictionary defining this step action.

        Returns
        -------
        callable
            The function defining the amplitude depending on the step propress.
        """

        if action["f(t)"] is not None:
            t = sp.symbols("t")
            amplitude = sp.lambdify(t, sp.sympify(action["f(t)"]), "numpy")
        else:

            def amplitude(x):
                return x

        return amplitude
