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
#  Paul Hofer Paul.Hofer@uibk.ac.at
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
"""Define a field using a scalar expression."""

from typing import Callable

import numpy as np

from edelweissfe.analyticalfields.base.analyticalfieldbase import (
    AnalyticalField as AnalyticalFieldBase,
)
from edelweissfe.utils.inputlanguage import InputLanguage
from edelweissfe.utils.math import createModelAccessibleFunction
from edelweissfe.utils.misc import (
    caseInsensitiveKwargsChecker,
    castKwargsValuesAndAddDefaults,
)

inputLanguage = InputLanguage()
module = inputLanguage["analyticalField"].addModule(
    "scalarExpression", "Define an analytical field using a scalar expression."
)
module.addRequiredArg(
    "f(x,y,z)",
    "Python expression using variables x, y, z (coordinates); dictionaries contained in model can be accessed",
    str,
)

documentation = [module]


@caseInsensitiveKwargsChecker([kw.name for kw in module.requiredArgs], [kw.name for kw in module.optionalArgs])
@castKwargsValuesAndAddDefaults(module)
def analyticalFieldFactory(name, FEModel, **kwargs):
    expression = createModelAccessibleFunction(kwargs["f(x,y,z)"], FEModel, *"xyz")

    return AnalyticalField(name, FEModel, expression)


class AnalyticalField(AnalyticalFieldBase):
    def __init__(self, name: str, FEModel, expression: Callable):
        self.name = name
        self.type = "scalarExpression"

        self.domainSize = FEModel.domainSize

        self.expression = expression

        return

    def evaluateAtCoordinates(self, coords):
        coords = np.array(coords)

        if coords.ndim == 1:
            coords = np.expand_dims(coords, 0)
        coords = np.c_[coords, np.zeros((coords.shape[0], 3 - coords.shape[-1]))]

        return np.expand_dims(np.array([float(self.expression(*coords_)) for coords_ in coords]), 1)
