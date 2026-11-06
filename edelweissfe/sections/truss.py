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
"""
A section for truss (bar) elements, e.g., the Marmot truss elements ``T2D2``, ``T3D2``, ``T3D2CR``, ...,
assigning the cross section area.

.. code-block:: edelweiss
    :caption: Example:

    *material, name=VonMises, id=steel, provider=marmot
        200000, 0.3, 500, 1000, 0, 1, 7.85e-9

    *section, name=rebarSection, type=truss, area=113.1, material=steel
        rebars
"""

from dataclasses import dataclass

import numpy as np

from edelweissfe.sections.base.sectionbase import MaterialParameterFromFieldSchema
from edelweissfe.sections.base.sectionbase import Section as SectionBase
from edelweissfe.sections.base.sectionbase import WriteMaterialPropertiesToFileSchema
from edelweissfe.sets.elementset import ElementSet
from edelweissfe.utils.schema import datalineField, schemaField, subKeywordField


@dataclass(frozen=True)
class TrussSectionSchema:
    """The options this section accepts, owned by this module and never mutated from outside it.

    ``area`` is declared ``required=True``, but is still given a ``default=None`` so that
    ``TrussSectionSchema()`` remains constructible on its own.
    """

    area: float | None = schemaField(description="cross section area", dtype=float, default=None, required=True)
    materialParameterFromField: tuple[MaterialParameterFromFieldSchema, ...] = subKeywordField(
        description="use material properties given by an analytical field",
        schema=MaterialParameterFromFieldSchema,
    )
    writeMaterialPropertiesToFile: tuple[WriteMaterialPropertiesToFileSchema, ...] = subKeywordField(
        description="export material properties to file",
        schema=WriteMaterialPropertiesToFileSchema,
    )
    elementSets: str | None = datalineField(
        description="elementSets as comma separated list of element sets for this section", required=True
    )


class Section(SectionBase):
    """A section for truss (bar) elements, assigning the cross section area."""

    #: Option schema for this section, per OptionSchemaProvider.
    schema = TrussSectionSchema

    def __init__(
        self,
        name,
        model,
        material: dict,
        elementSets: list[ElementSet],
        *,
        configuration: TrussSectionSchema = TrussSectionSchema(),
    ):
        """Constructible standalone, with no parser involvement.

        Parameters
        ----------
        name
            The name of this section.
        model
            The model tree.
        material
            The material (or marmot material provider dict) assigned to this section.
        elementSets
            The element sets this section is applied to.
        configuration
            The options this section accepts; ``area`` is still required, see :class:`TrussSectionSchema`.
        """
        super().__init__(
            name,
            model,
            material,
            elementSets,
            configuration.materialParameterFromField,
            configuration.writeMaterialPropertiesToFile,
        )
        if configuration.area is None or configuration.area <= 0:
            raise ValueError(f"Truss section {name}: a positive cross section area is required.")
        self.area = configuration.area

    def assignSectionPropertiesToElement(self, element, **kwargs):
        material = kwargs.get("material", self.material)

        if element.ensightType not in ("bar2", "bar3"):
            raise Exception(f"Truss section is incompatible with '{element.ensightType}' elements.")

        element.setProperties(np.array([self.area], dtype=float))
        element.initializeElement()
        if not isinstance(material, dict):
            element.setMaterial(material)
        else:
            try:  # for Marmot
                element.setMaterial(material["name"], material["properties"])
            except TypeError:
                raise Exception("Material provider and element are not compatible!")
