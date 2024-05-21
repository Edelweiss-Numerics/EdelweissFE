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
"""Define a random field using the GSTools library.
"""

import gstools
import numpy as np

from edelweissfe.analyticalfields.base.analyticalfieldbase import (
    AnalyticalField as AnalyticalFieldBase,
)
from edelweissfe.utils.misc import kwargsChecker

documentation = {
    "required": dict(),
    "optional": dict(
        model="Gaussian",
        mean=0.0,
        variance=1.0,
        lengthScale=10.0,
        seed=0,
    ),
}


@kwargsChecker(documentation["required"], documentation["optional"])
def analyticalFieldFactory(name, FEModel, **kwargs):
    modelType = kwargs.get("model", "Gaussian")
    mean = float(kwargs.get("mean", 0.0))
    variance = float(kwargs.get("variance", 1.0))
    lengthScale = float(kwargs.get("lengthScale", 10.0))
    seed = int(kwargs.get("seed", 0))

    return AnalyticalField(name, FEModel, modelType, mean, variance, lengthScale, seed)


class AnalyticalField(AnalyticalFieldBase):
    def __init__(
        self,
        name: str,
        FEModel,
        modelType="Gaussian",
        mean: float = 0,
        variance: float = 1.0,
        lengthScale: float = 10.0,
        seed: int = 0,
    ):
        self.name = name
        self.type = "randomScalar"

        self.domainSize = FEModel.domainSize

        modelMethod = getattr(gstools, modelType)
        model = modelMethod(
            dim=self.domainSize,
            var=variance,
            len_scale=lengthScale,
        )
        self.srf = gstools.SRF(model, seed=seed, mean=mean)

        return

    def evaluateAtCoordinates(self, coords):
        coords = np.array(coords)

        if coords.ndim == 1:
            coords = np.expand_dims(coords, 0)

        return np.expand_dims(np.array([self.srf(coords_)[0] for coords_ in coords]), 1)
