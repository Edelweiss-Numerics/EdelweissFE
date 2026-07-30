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


from dataclasses import dataclass

from edelweissfe.sections.base.sectionbase import MaterialParameterFromFieldSchema
from edelweissfe.sections.base.sectionbase import Section as SectionBase
from edelweissfe.sections.base.sectionbase import WriteMaterialPropertiesToFileSchema
from edelweissfe.sets.elementset import ElementSet
from edelweissfe.utils.inputlanguage import InputLanguage, Module
from edelweissfe.utils.schema import datalineField, subKeywordField

module = Module("solid", "This section represents a classical solid materal section.")

inputLanguage = InputLanguage()

keyword = "section"
if keyword in inputLanguage:
    inputLanguage[keyword].addModule(module)

module.addRequiredDatalines("elementSets as comma separated list of element sets for this section", str)

kw = module.addOptionalKeyword("materialParameterFromField", "use material properties given by an analytical field")
kw.addRequiredArg("index", "index of material parameter", int)
kw.addRequiredArg("field", "name of analytical field", str)
kw.addRequiredArg("type", "either 'setToValue' or 'scale'", str)
kw.addOptionalArg("f(p,f)", "p...value of parameter from material definition; f...value of analytical field", str, "f")

kw = module.addOptionalKeyword("writeMaterialPropertiesToFile", "export material properties to file")
kw.addRequiredArg("filename", "file name for material property export", str)

required = [kw.name for kw in module.requiredArgs]
required += [kw.name for kw in module.requiredKeywords]

optional = [kw.name for kw in module.optionalArgs]
optional += [kw.name for kw in module.optionalKeywords]

documentation = [module]


@dataclass(frozen=True)
class SolidSectionSchema:
    """L2: the options this section accepts, owned by this module and never mutated from outside
    it.

    Mirrors the ``module.addOptionalKeyword(...)``/``module.addRequiredDatalines(...)``
    declarations above one-for-one. The two declarations coexist while the migration is in
    progress; the ``Module`` one goes away with the ``InputLanguage`` singleton in P5.

    ``elementSets`` is a :func:`~edelweissfe.utils.schema.datalineField`, additive-only: it
    documents the dataline payload's presence for the grammar surface, but is excluded from
    :func:`~edelweissfe.utils.schema.optionNames`/``buildSchemaFromOptions`` and is not read by
    this section's constructor -- the actual element-set datalines are still interpreted by the
    U3-scoped construction path.
    """

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
    """This section represents a classical solid material section."""

    #: L2 schema declared for the L3 registry, per OptionSchemaProvider.
    schema = SolidSectionSchema

    def __init__(
        self,
        name,
        model,
        material: dict,
        elementSets: list[ElementSet],
        *,
        configuration: SolidSectionSchema = SolidSectionSchema(),
    ):
        """L1: constructible standalone, with no ``InputLanguage``/``Module``/parser involvement.

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
            The options this section accepts; defaults to all-defaults.
        """
        super().__init__(
            name,
            model,
            material,
            elementSets,
            configuration.materialParameterFromField,
            configuration.writeMaterialPropertiesToFile,
        )

    def assignSectionPropertiesToElement(self, element, material=None):
        if not material:
            material = self.material

        nSpatialDimensions = element.nSpatialDimensions
        if nSpatialDimensions < 3 and nSpatialDimensions != 0:
            raise Exception(f"Solid section is incompatible with {nSpatialDimensions}-dimensional finite elements.")

        element.initializeElement()

        # to make sure all elProviders work
        if not isinstance(material, dict):
            element.setMaterial(material)
        else:
            try:  # for Marmot
                element.setMaterial(material["name"], material["properties"])
            except TypeError:
                raise Exception("Material provider and element are not compatible!")
