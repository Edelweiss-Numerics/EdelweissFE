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

from dataclasses import dataclass

import numpy as np

from edelweissfe.analyticalfields.base.analyticalfieldbase import (
    AnalyticalField as AnalyticalFieldBase,
)
from edelweissfe.utils.inputlanguage import InputLanguage, Module
from edelweissfe.utils.math import createModelAccessibleFunction
from edelweissfe.utils.schema import schemaField

module = Module("scalarExpression", "Define an analytical field using a scalar expression.")

inputLanguage = InputLanguage()

keyword = "analyticalField"
if keyword in inputLanguage:
    inputLanguage[keyword].addModule(module)

module.addRequiredArg(
    "f(x,y,z)",
    "Python expression using variables x, y, z (coordinates); dictionaries contained in model can be accessed",
    str,
)

documentation = [module]


@dataclass(frozen=True)
class ScalarExpressionSchema:
    """L2: the options this analytical field accepts, owned by this module and never mutated from
    outside it.

    Mirrors the ``module.addRequiredArg(...)`` declaration above. The two declarations coexist
    while the migration is in progress; the ``Module`` one goes away with the ``InputLanguage``
    singleton in P5.

    ``f_x_y_z`` is declared ``required=True`` explicitly, mirroring ``addRequiredArg`` above, but is
    still given a ``default=None`` so that ``ScalarExpressionSchema()`` remains constructible for
    the L1 constructor's default argument; the L4 adapter (``buildSchemaFromOptions``) still
    enforces that an ``.inp`` file supplies it.
    """

    f_x_y_z: str | None = schemaField(
        description="Python expression using variables x, y, z (coordinates); dictionaries "
        "contained in model can be accessed",
        dtype=str,
        default=None,
        required=True,
        optionName="f(x,y,z)",
    )


class AnalyticalField(AnalyticalFieldBase):
    """Define an analytical field using a scalar expression."""

    #: L2 schema declared for the L3 registry, per OptionSchemaProvider.
    schema = ScalarExpressionSchema

    def __init__(self, name: str, FEModel, *, configuration: ScalarExpressionSchema = ScalarExpressionSchema()):
        """L1: constructible standalone, with no ``InputLanguage``/``Module``/parser involvement.

        Parameters
        ----------
        name
            The name of this analytical field.
        FEModel
            The model tree.
        configuration
            The options this analytical field accepts; ``f_x_y_z`` is still required, see
            :class:`ScalarExpressionSchema`.
        """
        self.name = name
        self.type = "scalarExpression"

        self.domainSize = FEModel.domainSize

        self.expression = createModelAccessibleFunction(configuration.f_x_y_z, FEModel, *"xyz")

        return

    def evaluateAtCoordinates(self, coords):
        coords = np.array(coords)

        if coords.ndim == 1:
            coords = np.expand_dims(coords, 0)
        coords = np.c_[coords, np.zeros((coords.shape[0], 3 - coords.shape[-1]))]

        return np.expand_dims(np.array([float(self.expression(*coords_)) for coords_ in coords]), 1)
