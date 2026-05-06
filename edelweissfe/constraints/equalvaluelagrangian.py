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
# Created on Fri Sep 9 11:34:35 2022

# @author: Matthias Neuner

import numpy as np

from edelweissfe.config.phenomena import getFieldSize
from edelweissfe.constraints.base.constraintbase import ConstraintBase
from edelweissfe.models.femodel import FEModel
from edelweissfe.timesteppers.timestep import TimeStep
from edelweissfe.utils.caseinsensitivedict import CaseInsensitiveDict
from edelweissfe.utils.inputlanguage import InputLanguage
from edelweissfe.utils.misc import (
    caseInsensitiveKwargsChecker,
    castKwargsValuesAndAddDefaults,
)

"""
A lagrangian multiplier based constraint used for constraining nodal values
of a node set to be equal.
"""

# documentation = {
#     "field": "The field this constraint acts on.",
#     "component": "The component of the field.",
#     "nSet": "The node set to be constrained.",
# }

inputLanguage = InputLanguage()
module = inputLanguage["constraint"].addModule(
    "equalvaluelagrangian",
    "A lagrangian multiplier based constraint used for constraining nodal values of a node set to be equal.",
)

module.addRequiredArg("field", "The field this constraint acts on.", str)
module.addRequiredArg("component", "The component of the field.", int)
module.addRequiredArg("nSet", "The node set to be constrained.", str)

documentation = [module]


class Constraint(ConstraintBase):
    @caseInsensitiveKwargsChecker([kw.name for kw in module.requiredArgs], [kw.name for kw in module.optionalArgs])
    @castKwargsValuesAndAddDefaults(module)
    def __init__(self, name: str, model: FEModel, *args, **kwargs):
        super().__init__(name, model, *args, **kwargs)

        kwargs = CaseInsensitiveDict(kwargs)

        theField = kwargs["field"]
        self.sizeField = getFieldSize(theField, model.domainSize)
        self.component = kwargs["component"]
        self._name = name
        self._nodes = model.nodeSets[kwargs["nSet"]]
        self.nNodes = len(self._nodes)
        self.nMultipliers = len(self._nodes) - 1

        self._nDof = self.sizeField * self.nNodes + self.nMultipliers

        self.index_master = self.component
        self.indices_slaves = slice(
            self.sizeField + self.component,
            self.sizeField * self.nNodes,
            self.sizeField,
        )
        self.indices_multipliers = slice(self.sizeField * self.nNodes, self._nDof)

        self._fieldsOnNodes = [
            [
                theField,
            ]
        ] * self.nNodes

        self.active = True

    @property
    def nodes(self) -> list:
        return self._nodes

    @property
    def fieldsOnNodes(self) -> list:
        return self._fieldsOnNodes

    @property
    def nDof(self) -> int:
        return self._nDof

    def getNumberOfAdditionalNeededScalarVariables(self):
        return self.nNodes - 1

    def applyConstraint(
        self,
        U_np: np.ndarray,
        dU: np.ndarray,
        PExt: np.ndarray,
        K: np.ndarray,
        timeStep: TimeStep,
    ):
        if not self.active:
            return

        val_master = U_np[self.index_master]
        vals_slaves = U_np[self.indices_slaves]
        multipliers = U_np[self.indices_multipliers]

        gs = vals_slaves - val_master

        PExt[self.index_master] -= -np.sum(multipliers)
        PExt[self.indices_slaves] -= multipliers
        PExt[self.indices_multipliers] -= gs

        K[self.index_master, self.indices_multipliers] += -1.0

        diag_sm = np.diag(K[self.indices_slaves, self.indices_multipliers])
        diag_sm.setflags(write=True)  # bug in numpy
        diag_sm[:] += 1.0

        K[self.indices_multipliers, self.index_master] = K[self.index_master, self.indices_multipliers].T
        K[self.indices_multipliers, self.indices_slaves] = K[self.indices_slaves, self.indices_multipliers].T
