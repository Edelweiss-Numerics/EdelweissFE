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
#
# @author: Matthias Neuner, Paul Hofer

from abc import ABC, abstractmethod

import numpy as np

from edelweissfe.sets.elementset import ElementSet
from edelweissfe.utils.caseinsensitivedict import CaseInsensitiveDict
from edelweissfe.utils.math import createFunction
from edelweissfe.utils.misc import strCaseCmp


class Section(ABC):
    def __init__(
        self,
        name,
        model,
        material: dict,
        elementSets: list[ElementSet],
        materialParameterFromFieldDefs: list[CaseInsensitiveDict],
        writeMaterialPropertiesToFileDefs: list[CaseInsensitiveDict],
        # expression: Callable = None,
    ):
        self.material = material
        self.elSets = elementSets

        self.materialParameterFromFieldDefs = materialParameterFromFieldDefs
        self.writeMaterialPropertiesToFileDefs = writeMaterialPropertiesToFileDefs

        for definition in materialParameterFromFieldDefs:
            if not any(
                [strCaseCmp(definition["type"], implementedType) for implementedType in ["setToValue", "scale"]]
            ):
                raise ValueError(
                    f"{name}: {definition['type']} is not a known type; currently available types: 'setToValue', 'scale'"
                )

            definition["expression"] = createFunction(definition["f(p,f)"], "p", "f", model=model)

        if len(self.writeMaterialPropertiesToFileDefs) > 1:
            raise ValueError("Too many definitions for writeMaterialPropertiesToFile")

        self.writeMaterialPropertiesToFile = False
        for definition in self.writeMaterialPropertiesToFileDefs:
            self.writeMaterialPropertiesToFile = True
            self.materialPropertiesFileName = definition["filename"]

    def assignSectionPropertiesToModel(self, model):
        if any(self.materialParameterFromFieldDefs):
            for elSet in self.elSets:
                for el in elSet:
                    if isinstance(self.material, dict):  # for marmotmaterial provider
                        modifiedMaterial = self.material.copy()
                        modifiedMaterial["properties"] = self.propertiesFromField(el, self.material, model, True)
                    else:  # for edelweissmaterial provider
                        materialType = type(self.material)
                        modifiedProperties = self.propertiesFromField(el, self.material, model, False)
                        modifiedMaterial = materialType(modifiedProperties)
                        if hasattr(self.material, "_materialEnergy"):  # for autodiff materials
                            modifiedMaterial.setEnergyFunction(self.material._materialEnergy)
                    self.assignSectionPropertiesToElement(el, material=modifiedMaterial)
        else:
            for elSet in self.elSets:
                for el in elSet:
                    self.assignSectionPropertiesToElement(el)

        if self.writeMaterialPropertiesToFile:
            self.exportMaterialPropertiesToFile(self.elSets)

        return model

    @abstractmethod
    def assignSectionPropertiesToElement(self, element, material):
        pass

    def propertiesFromField(self, el, material, model, isMarmotMaterial):
        coordinatesAtCenter = el.getCoordinatesAtCenter()
        materialProperties = np.copy(material["properties"]) if isMarmotMaterial else material.materialProperties.copy()
        isCustomMaterial = isinstance(materialProperties, dict)

        for definition in self.materialParameterFromFieldDefs:
            index = int(definition["index"]) if not isCustomMaterial else definition["index"]
            fieldValue = model.analyticalFields[definition["field"]].evaluateAtCoordinates(coordinatesAtCenter)[0][0]
            parameterValue = materialProperties[index]
            if strCaseCmp(definition["type"], "setToValue"):
                materialProperties[index] = definition["expression"](parameterValue, fieldValue)
            elif strCaseCmp(definition["type"], "scale"):
                materialProperties[index] *= definition["expression"](parameterValue, fieldValue)

        return materialProperties

    def exportMaterialPropertiesToFile(self, elSets):
        with open("{:}.csv".format(self.materialPropertiesFileName), "w+") as f:
            for elSet in elSets:
                for el in elSet:
                    f.write("{:}".format(el.elNumber))
                    [f.write("{:} ".format(matprop)) for matprop in el._materialProperties]
                    f.write("\n")
