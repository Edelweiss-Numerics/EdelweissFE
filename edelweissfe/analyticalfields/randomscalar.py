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
"Müller, S., Schüler, L., Zech, A., and Heße, F.: GSTools v1.3: a toolbox for geostatistical modelling in Python, Geosci. Model Dev., 15, 3161–3182, https://doi.org/10.5194/gmd-15-3161-2022, 2022."
"""

import gstools
import numpy as np

from edelweissfe.analyticalfields.base.analyticalfieldbase import (
    AnalyticalField as AnalyticalFieldBase,
)
from edelweissfe.utils.inputlanguage import InputLanguage
from edelweissfe.utils.misc import caseInsensitiveKwargsChecker

inputLanguage = InputLanguage()
module = inputLanguage["*analyticalField"].addModule("randomScalar", "input language for randomscalar module")

module.addOptionalArg("model", "Covariance Model of the spatial random field", str, "Gaussian")
module.addOptionalArg("mean", "Mean of the spatial random field", float, 0.0)
module.addOptionalArg("variance", "Variance of the model", float, 1.0)
module.addOptionalArg("lengthScale", "Length scale of the model", float, 10.0)
module.addOptionalArg("seed", "Seed of the random number generator", int, 0)


@caseInsensitiveKwargsChecker([kw.name for kw in module.requiredArgs], [kw.name for kw in module.optionalArgs])
def analyticalFieldFactory(name, FEModel, **kwargs):
    modelType = module["model"].getValueFromKwargs(kwargs)
    mean = module["mean"].getValueFromKwargs(kwargs)
    variance = module["variance"].getValueFromKwargs(kwargs)
    lengthScale = module["lengthScale"].getValueFromKwargs(kwargs)
    seed = module["seed"].getValueFromKwargs(kwargs)

    return AnalyticalField(name, FEModel, modelType, mean, variance, lengthScale, seed)


class AnalyticalField(AnalyticalFieldBase):
    def __init__(
        self,
        name: str,
        FEModel,
        modelType=module["model"].default,
        mean: float = module["mean"].default,
        variance: float = module["variance"].default,
        lengthScale: float = module["lengthScale"].default,
        seed: int = module["seed"].default,
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
