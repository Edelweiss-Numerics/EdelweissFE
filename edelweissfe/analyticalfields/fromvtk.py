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
"""Use PyVista to interpolate from vtk data."""

from dataclasses import dataclass

import numpy as np
import pyvista

from edelweissfe.analyticalfields.base.analyticalfieldbase import (
    AnalyticalField as AnalyticalFieldBase,
)
from edelweissfe.utils.schema import schemaField


@dataclass(frozen=True)
class FromVtkSchema:
    """The options this analytical field accepts, owned by this module and never mutated from
    outside it.

    Both fields are ``required=True``, but are still given a ``default=None``/``default=""`` so
    that ``FromVtkSchema()`` remains constructible as the constructor's default argument;
    ``buildSchemaFromOptions`` still enforces that an ``.inp`` file supplies ``file``. ``result``
    may legitimately be supplied empty (see ``AnalyticalField.__init__``'s auto-pick fallback), so
    an empty string -- not ``None`` -- is its default, matching what an ``.inp`` file writing
    ``result=`` produces.
    """

    file: str | None = schemaField(description="path to database file", dtype=str, default=None, required=True)
    result: str = schemaField(description="result name in database", dtype=str, default="", required=True)


class AnalyticalField(AnalyticalFieldBase):
    """Use PyVista to interpolate from vtk data."""

    #: Option schema for this analytical field, per OptionSchemaProvider.
    schema = FromVtkSchema

    def __init__(self, name: str, FEModel, *, configuration: FromVtkSchema = FromVtkSchema()):
        """Constructible standalone, with no parser involvement.

        Parameters
        ----------
        name
            The name of this analytical field.
        FEModel
            The model tree.
        configuration
            The options this analytical field accepts; ``file`` is still required, see
            :class:`FromVtkSchema`.
        """
        self.name = name
        self.type = "fromVtk"

        self.domainSize = FEModel.domainSize

        reader = pyvista.get_reader(configuration.file)
        self.data = reader.read()

        availableResults = self.data.array_names

        if not len(availableResults) > 0:
            raise ValueError("Database does not contain at least one result.")

        result = configuration.result
        if not result:
            if len(availableResults) == 1:
                result = self.data.array_names[0]
            else:
                raise ValueError("Database contains multiple results. Specify result with option result=...")
        else:
            if result not in availableResults:
                raise KeyError(f"Specified result '{result}' not available. Available results: {availableResults}")

        self.result = result

        return

    def evaluateAtCoordinates(self, coords):
        coords = np.array(coords)

        interpolatedData = pyvista.PointSet(coords).sample(self.data)
        interpolatedResult = interpolatedData[self.result]

        return np.expand_dims(interpolatedResult, 1)
