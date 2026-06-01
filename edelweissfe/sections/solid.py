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


from edelweissfe.sections.base.sectionbase import Section as SectionBase
from edelweissfe.sets.elementset import ElementSet
from edelweissfe.utils.caseinsensitivedict import CaseInsensitiveDict
from edelweissfe.utils.inputlanguage import InputLanguage, Module
from edelweissfe.utils.misc import (
    caseInsensitiveKwargsChecker,
    castKwargsValuesAndAddDefaults,
    splitLinesAtCommas,
)

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


@caseInsensitiveKwargsChecker(required, optional)
@castKwargsValuesAndAddDefaults(module)
def sectionFactory(name, FEModel, materialName: str, datalines: list[str], moduleOptions, **kwargs):
    kwargs = CaseInsensitiveDict(kwargs)

    elementSetNames = splitLinesAtCommas(datalines)

    materialParameterFromFieldDefs = moduleOptions.get("materialParameterFromField", [])
    writeMaterialPropertiesToFileDefs = moduleOptions.get("writeMaterialPropertiesToFile", [])

    return Section(
        name,
        FEModel,
        FEModel.materials[materialName],
        [FEModel.elementSets[name] for name in elementSetNames],
        materialParameterFromFieldDefs,
        writeMaterialPropertiesToFileDefs,
    )


class Section(SectionBase):
    def __init__(
        self,
        name,
        model,
        material: dict,
        elementSets: list[ElementSet],
        materialParameterFromFieldDefs: list[dict],
        writeMaterialPropertiesToFileDefs: list[dict],
        # expression: Callable = None,
    ):
        super().__init__(
            name, model, material, elementSets, materialParameterFromFieldDefs, writeMaterialPropertiesToFileDefs
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
