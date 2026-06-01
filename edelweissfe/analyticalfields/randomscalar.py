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
from edelweissfe.utils.caseinsensitivedict import CaseInsensitiveDict
from edelweissfe.utils.inputlanguage import InputLanguage, Module
from edelweissfe.utils.misc import (
    caseInsensitiveKwargsChecker,
    castKwargsValuesAndAddDefaults,
    strCaseCmp,
)

module = Module("randomScalar", "Define a random field using the GSTools library.")

inputLanguage = InputLanguage()

keyword = "analyticalField"
if keyword in inputLanguage:
    inputLanguage[keyword].addModule(module)

module.addOptionalArg("model", "Covariance Model of the spatial random field", str, "Gaussian")
module.addOptionalArg("mean", "Mean of the spatial random field", float, 0.0)
module.addOptionalArg("variance", "Variance of the model", float, 1.0)
module.addOptionalArg("lengthScale", "Length scale of the model", float, 10.0)
module.addOptionalArg("nu", "Smoothness parameter for Matern covariance function", float, 1.0)
module.addOptionalArg("seed", "Seed of the random number generator", int, 0)

documentation = [module]


@caseInsensitiveKwargsChecker([kw.name for kw in module.requiredArgs], [kw.name for kw in module.optionalArgs])
@castKwargsValuesAndAddDefaults(module)
def analyticalFieldFactory(name, FEModel, **kwargs):
    kwargs = CaseInsensitiveDict(kwargs)

    modelType = kwargs["model"]
    mean = kwargs["mean"]
    variance = kwargs["variance"]
    lengthScale = kwargs["lengthScale"]
    nu = kwargs["nu"]
    seed = kwargs["seed"]

    return AnalyticalField(name, FEModel, modelType, mean, variance, lengthScale, nu, seed)


class AnalyticalField(AnalyticalFieldBase):
    def __init__(
        self,
        name: str,
        FEModel,
        modelType=module["model"].default,
        mean: float = module["mean"].default,
        variance: float = module["variance"].default,
        lengthScale: float = module["lengthScale"].default,
        nu: float = module["nu"].default,
        seed: int = module["seed"].default,
    ):
        self.name = name
        self.type = "randomScalar"

        self.domainSize = FEModel.domainSize

        if strCaseCmp(modelType, "Gaussian"):
            # modelMethod = getattr(gstools, modelType)
            model = gstools.Gaussian(
                dim=self.domainSize,
                var=variance,
                len_scale=lengthScale,
            )
        elif strCaseCmp(modelType, "Matern"):
            # modelMethod = getattr(gstools, modelType)
            model = gstools.covmodel.Matern(
                dim=self.domainSize,
                var=variance,
                len_scale=lengthScale,
                nu=nu,
            )
        else:
            raise NotImplementedError(f"Model type {modelType} not implemented.")

        self.srf = gstools.SRF(model, seed=seed, mean=mean)

        return

    def evaluateAtCoordinates(self, coords):
        coords = np.array(coords)

        if coords.ndim == 1:
            coords = np.expand_dims(coords, 0)

        return np.expand_dims(np.array([self.srf(coords_)[0] for coords_ in coords]), 1)
