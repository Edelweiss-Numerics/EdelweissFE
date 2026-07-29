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

from dataclasses import dataclass

import numpy as np

from edelweissfe.analyticalfields.base.analyticalfieldbase import (
    AnalyticalField as AnalyticalFieldBase,
)
from edelweissfe.utils.inputlanguage import InputLanguage, Module
from edelweissfe.utils.misc import strCaseCmp
from edelweissfe.utils.schema import schemaField

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


@dataclass(frozen=True)
class RandomScalarSchema:
    """L2: the options this analytical field accepts, owned by this module and never mutated from
    outside it.

    Mirrors the ``module.addOptionalArg(...)`` declarations above one-for-one. The two declarations
    coexist while the migration is in progress; the ``Module`` one goes away with the
    ``InputLanguage`` singleton in P5.
    """

    model: str = schemaField(description="Covariance Model of the spatial random field", dtype=str, default="Gaussian")
    mean: float = schemaField(description="Mean of the spatial random field", dtype=float, default=0.0)
    variance: float = schemaField(description="Variance of the model", dtype=float, default=1.0)
    lengthScale: float = schemaField(description="Length scale of the model", dtype=float, default=10.0)
    nu: float = schemaField(description="Smoothness parameter for Matern covariance function", dtype=float, default=1.0)
    seed: int = schemaField(description="Seed of the random number generator", dtype=int, default=0)


class AnalyticalField(AnalyticalFieldBase):
    """Define a random field using the GSTools library."""

    #: L2 schema declared for the L3 registry, per OptionSchemaProvider.
    schema = RandomScalarSchema

    def __init__(self, name: str, FEModel, *, configuration: RandomScalarSchema = RandomScalarSchema()):
        """L1: constructible standalone, with no ``InputLanguage``/``Module``/parser involvement.

        Parameters
        ----------
        name
            The name of this analytical field.
        FEModel
            The model tree.
        configuration
            The options this analytical field accepts; defaults to all-defaults.
        """
        # gstools is imported lazily: its Cython extension does not declare
        # free-threading support, so importing it re-enables the GIL process-wide
        # and would disable thread-parallel computations for ALL simulations,
        # even those not using random fields.
        try:
            import gstools
        except ModuleNotFoundError as e:
            raise ModuleNotFoundError(
                "the 'randomScalar' analytical field requires the 'gstools' package "
                "(install via 'pip install gstools' or 'mamba install -c conda-forge gstools')"
            ) from e

        self.name = name
        self.type = "randomScalar"

        self.domainSize = FEModel.domainSize

        modelType = configuration.model
        variance = configuration.variance
        lengthScale = configuration.lengthScale

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
                nu=configuration.nu,
            )
        else:
            raise NotImplementedError(f"Model type {modelType} not implemented.")

        self.srf = gstools.SRF(model, seed=configuration.seed, mean=configuration.mean)

        return

    def evaluateAtCoordinates(self, coords):
        coords = np.array(coords)

        if coords.ndim == 1:
            coords = np.expand_dims(coords, 0)

        return np.expand_dims(np.array([self.srf(coords_)[0] for coords_ in coords]), 1)
