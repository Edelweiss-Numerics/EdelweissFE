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
#  Paul Hofer paul.hofer@uibk.ac.at
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
# Created on Tue Jan  17 19:10:42 2017
#
# @author: Matthias Neuner, Paul Hofer

from dataclasses import dataclass

import numpy as np

from edelweissfe.sections.base.sectionbase import MaterialParameterFromFieldSchema
from edelweissfe.sections.base.sectionbase import Section as SectionBase
from edelweissfe.sections.base.sectionbase import WriteMaterialPropertiesToFileSchema
from edelweissfe.sets.elementset import ElementSet
from edelweissfe.utils.schema import datalineField, schemaField, subKeywordField


@dataclass(frozen=True)
class PlaneSectionSchema:
    """The options this section accepts, owned by this module and never mutated from outside it.

    ``thickness`` is declared ``required=True``, but is still given a ``default=None`` so that
    ``PlaneSectionSchema()`` remains constructible on its own; ``buildSchemaFromOptions`` still
    enforces that an ``.inp`` file supplies it.

    ``elementSets`` is a :func:`~edelweissfe.utils.schema.datalineField`, additive-only: it
    documents the dataline payload's presence for the grammar surface, but is excluded from
    :func:`~edelweissfe.utils.schema.optionNames`/``buildSchemaFromOptions`` and is not read by
    this section's constructor -- the element-set datalines are interpreted elsewhere.
    """

    thickness: float | None = schemaField(description="thickness", dtype=float, default=None, required=True)
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
    """This section represents a classical plane solid material section."""

    #: Option schema for this section, per OptionSchemaProvider.
    schema = PlaneSectionSchema

    def __init__(
        self,
        name,
        model,
        material: dict,
        elementSets: list[ElementSet],
        *,
        configuration: PlaneSectionSchema = PlaneSectionSchema(),
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
            The options this section accepts; ``thickness`` is still required, see
            :class:`PlaneSectionSchema`.
        """
        super().__init__(
            name,
            model,
            material,
            elementSets,
            configuration.materialParameterFromField,
            configuration.writeMaterialPropertiesToFile,
        )
        self.thickness = configuration.thickness

    def assignSectionPropertiesToElement(self, element, **kwargs):
        material = kwargs.get("material", self.material)

        nSpatialDimensions = element.nSpatialDimensions
        if nSpatialDimensions != 2:
            raise Exception(f"Plane section is incompatible with {nSpatialDimensions}-dimensional finite elements.")

        thickness = self.thickness

        elProperties = np.array([thickness], dtype=float)

        element.setProperties(elProperties)
        element.initializeElement()
        # to make sure all elProviders work
        if not isinstance(material, dict):
            element.setMaterial(material)
        else:
            try:  # for Marmot
                element.setMaterial(material["name"], material["properties"])
            except TypeError:
                raise Exception("Material provider and element are not compatible!")
