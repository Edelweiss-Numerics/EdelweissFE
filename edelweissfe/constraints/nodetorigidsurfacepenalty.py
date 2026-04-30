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
#  Konstantin Basche konstantin.basche@uibk.ac.at
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
# Created on Thu Mar 26 10:21:35 2026

# @author: Konstantin Basche

import numpy as np

from edelweissfe.config.phenomena import getFieldSize
from edelweissfe.constraints.base.constraintbase import ConstraintBase
from edelweissfe.models.femodel import FEModel
from edelweissfe.timesteppers.timestep import TimeStep
from edelweissfe.utils.misc import convertLinesToStringDictionary

"""
A penalty based unilateral constraint used for preventing the nodes of a node set from penetrating a defined rigid boundary.
"""

documentation = {
    "field": "The field this constraint acts on.",
    "component": "The component of the field.",
    "penalty": "The numerical penalty value.",
    "nSet": "The node set to be constrained.",
    "value": "The prescribed distance to the rigid boundary. A value of 0.0 implies no initial gap between the node set and the boundary.",
    "direction": "The normal direction outward from the continuum towards the boundary (1.0 or -1.0).",
    "type": "The formulation type: 'linear' (linear force, constant stiffness with jump) or 'quadratic' (quadratic force, linear stiffness).",
}


class Constraint(ConstraintBase):
    def __init__(self, name: str, definitionLines: list, model: FEModel):
        super().__init__(name, definitionLines, model)

        definition = convertLinesToStringDictionary(definitionLines)

        theField = definition["field"]
        self.sizeField = getFieldSize(theField, model.domainSize)
        self.component = int(definition["component"])
        self.penalty = float(definition["penalty"])
        self.value = float(definition.get("value", 0.0))
        self.direction = float(definition.get("direction", 1.0))
        self._nodes = model.nodeSets[definition["nSet"]]
        self._nNodes = len(self._nodes)
        self._nDof = self.sizeField * self._nNodes

        self.type = definition.get("type", "linear").lower()
        if self.type not in ["linear", "quadratic"]:
            raise ValueError(f"Constraint type '{self.type}' is not supported. Use 'linear' or 'quadratic'.")

        self.indices_component = np.arange(self.component, self._nDof + self.component, self.sizeField)

        self._fieldsOnNodes = [
            [
                theField,
            ]
        ] * self._nNodes

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

        values = U_np[self.indices_component]
        gap = (values - self.value) * self.direction

        active_mask = gap > 0

        if not np.any(active_mask):
            return

        active_indices = self.indices_component[active_mask]
        active_gaps = gap[active_mask]

        if self.type == "linear":
            force_magnitude = self.penalty * active_gaps
            stiffness = self.penalty
        elif self.type == "quadratic":
            force_magnitude = 0.5 * self.penalty * active_gaps**2
            stiffness = self.penalty * active_gaps

        PExt[active_indices] -= force_magnitude * self.direction
        K[active_indices, active_indices] += stiffness
