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
# Created on Tue Apr  3 09:44:10 2018

# @author: Matthias Neuner
"""
The default way to create finite element meshes
is using the keywords

 * ``*node``
 * ``*element``
 * ``*nset``
 * ``*elset``
 * ``*surface``

employing an Abaqus-like syntax.
"""

import numpy as np

from edelweissfe.config.analyticalfields import getAnalyticalFieldFactoryByName
from edelweissfe.config.constraints import getConstraintClass
from edelweissfe.config.elementlibrary import getElementClass
from edelweissfe.config.materiallibrary import getMaterialClass
from edelweissfe.config.sections import getSectionFactoryByName
from edelweissfe.models.femodel import FEModel
from edelweissfe.points.node import Node
from edelweissfe.sets.elementset import ElementSet
from edelweissfe.sets.nodeset import NodeSet
from edelweissfe.utils.caseinsensitivedict import CaseInsensitiveDict
from edelweissfe.utils.inputlanguage import (
    keywordIdentifier,
    moduleLevelKeywordIdentifier,
)
from edelweissfe.utils.misc import (
    convertLinesToFlatArray,
    convertLinesToMixedDictionary,
    convertLinesToStringDictionary,
    isInteger,
    splitLineAtCommas,
)

# isort: off
from edelweissfe.utils.inputfileparser import inputLanguage  # noqa: F811

from edelweissfe.analyticalfields.randomscalar import inputLanguage  # noqa: F811
from edelweissfe.analyticalfields.fromvtk import inputLanguage  # noqa: F811
from edelweissfe.analyticalfields.scalarexpression import inputLanguage  # noqa: F811

from edelweissfe.sections.solid import inputLanguage  # noqa: F811
from edelweissfe.sections.plane import inputLanguage  # noqa: F811

# isort: on


class AbqModelConstructor:
    def __init__(self, journal):
        self.journal = journal

    def createGeometryFromInputFile(self, model: FEModel, inputFile: dict) -> dict:
        """Collects nodes, elements, node sets and element sets from
        the input file.

        Parameters
        ----------
        model
            A dictionary containing the model tree.
        inputFile
            A dictionary contaning the input file tree.

        Returns
        -------
        dict
            The updated model tree.
        """

        domainSize = model.domainSize

        # returns an dict of {node label: node}
        nodeDefinitions = model.nodes
        for nodeDefs in inputFile["node"]:
            currNodeDefs = {}
            for line in nodeDefs["datalines"]:
                defLine = splitLineAtCommas(line)

                label = int(defLine[0])
                coordinates = np.zeros(domainSize)
                coordinates[:] = defLine[1:]
                currNodeDefs[label] = Node(
                    label,
                    coordinates,
                )
            nodeDefinitions.update(currNodeDefs)

            if nodeDefs["nSet"] is not None:
                setName = nodeDefs["nset"]
                model.nodeSets[setName] = NodeSet(setName, [nodeDefinitions[x] for x in currNodeDefs.keys()])

        # returns an dict of {element Label: element}
        elements = model.elements

        for elDefs in inputFile["element"]:
            elementType = elDefs["type"]
            elementProvider = elDefs.get("provider")
            ElementClass = getElementClass(elementType, elementProvider)

            currElDefs = {}
            for line in elDefs["datalines"]:
                defLine = [int(i) for i in splitLineAtCommas(line)]

                label = defLine[0]
                # store nodeObjects in elNodes list
                elNodes = [nodeDefinitions[n] for n in defLine[1:]]
                newEl = ElementClass(elementType, label)
                newEl.setNodes(elNodes)
                currElDefs[label] = newEl
            elements.update(currElDefs)

            if elDefs["elSet"] is not None:
                setName = elDefs["elset"]
                model.elementSets[setName] = ElementSet(setName, currElDefs.values())

        # generate dictionary of elementObjects belonging to a specified elementset
        # or generate elementset by generate definition in inputfile
        elementSets = model.elementSets

        for elSetDefinition in inputFile["elSet"]:
            name = elSetDefinition["elSet"]

            data = [splitLineAtCommas(line) for line in elSetDefinition["datalines"]]
            # decide if entries are labels or existing nodeSets:
            if isInteger(data[0][0]):
                elNumbers = [int(num) for line in data for num in line]

                if elSetDefinition.get("generate", False):
                    generateDef = elNumbers[0:3]
                    els = [
                        elements[n]
                        for n in np.arange(
                            generateDef[0],
                            generateDef[1] + 1,
                            generateDef[2],
                            dtype=int,
                        )
                    ]

                elif elSetDefinition.get("boolean", False):
                    booleanDef = elSetDefinition.get("boolean")
                    if booleanDef == "difference":
                        els = [n for n in elementSets[name] if n.elNumber not in elNumbers]

                    elif booleanDef == "union":
                        els = [n for n in elementSets[name]]
                        els += [elements[n] for n in elNumbers]

                    elif booleanDef == "intersection":
                        elNumbersBase = [n.elNumber for n in elementSets[name]]
                        els = [elements[n] for n in list(set(elNumbers).intersection(elNumbersBase))]
                    else:
                        raise Exception("Undefined boolean operation!")

                    if elSetDefinition.get("newElSet") != name:
                        name = elSetDefinition.get("newElSet")
                    else:
                        del elementSets[name]
                else:
                    els = [elements[elNum] for elNum in elNumbers]
                elementSets[name] = ElementSet(name, set(els))
            else:
                elementSets[name] = []
                for line in data:
                    for elSet in line:
                        elementSets[name] += elementSets[elSet]

        # generate dictionary of nodeObjects belonging to a specified nodeset
        # or generate nodeset by generate definition in inputfile
        nodeSets = model.nodeSets
        for nSetDefinition in inputFile["nSet"]:
            name = nSetDefinition["nSet"]

            data = [splitLineAtCommas(line) for line in nSetDefinition["datalines"]]
            if isInteger(data[0][0]):
                nodes = [int(n) for line in data for n in line]
                if nSetDefinition.get("generate", False):
                    generateDef = nodes  # nSetDefinition['data'][0][0:3]
                    nodes = [
                        nodeDefinitions[n]
                        for n in np.arange(
                            generateDef[0],
                            generateDef[1] + 1,
                            generateDef[2],
                            dtype=int,
                        )
                    ]
                else:
                    nodes = [nodeDefinitions[n] for n in nodes]
                nodeSets[name] = NodeSet(name, nodes)
            else:
                nSetLabels = [nSetLabel for line in data for nSetLabel in line]
                nodes = [n for nSetLabel in nSetLabels for n in nodeSets[nSetLabel].nodes]
                nodeSets[name] = NodeSet(name, nodes)

        model.nodeSets["all"] = NodeSet("all", model.nodes.values())
        model.elementSets["all"] = ElementSet("all", model.elements.values())

        # generate surfaces sets
        for surfaceDef in inputFile["surface"]:
            name = surfaceDef["name"]
            sType = surfaceDef.get("type", "element").lower()
            surface = {}
            if sType == "element":
                data = [splitLineAtCommas(line) for line in surfaceDef["datalines"]]
                for line in data:
                    elSet, faceNumber = line
                    faceNumber = int(faceNumber.replace("S", ""))
                    surface[faceNumber] = model.elementSets[elSet]

            model.surfaces[name] = surface

        return model

    def createMaterialsFromInputFile(self, model, inputFile):
        """Collects material defintions from the input file.
        Creates instances of materials.

        Parameters
        ----------
        model
            A dictionary containing the model tree.
        inputFile
            A dictionary contaning the input file tree.

        Returns
        -------
        dict
            The updated model tree.
        """

        for materialDef in inputFile["material"]:
            materialName = materialDef["name"]
            if "advanced" in materialName:
                raise Exception("Please use the *advancedmaterial keyword for advanced materials!")
            materialProvider = materialDef.get("provider", None)
            materialID = materialDef.get("id", materialName)

            materialProperties = convertLinesToFlatArray(materialDef["datalines"], dtype=float)
            materialClass = getMaterialClass(materialName, materialProvider)

            if materialClass is None:  # for Marmot
                model.materials[materialID] = {
                    "name": materialName,
                    "properties": materialProperties,
                }
            else:  # for DisplacementElement
                model.materials[materialID] = materialClass(materialProperties)

        return model

    def createAdvancedMaterialsFromInputFile(self, model, inputFile):
        """Collects advanced material defintions from the input file.
        Creates instances of advanced materials.

        Parameters
        ----------
        model
            A dictionary containing the model tree.
        inputFile
            A dictionary contaning the input file tree.

        Returns
        -------
        dict
            The updated model tree.
        """

        for materialDef in inputFile["advancedmaterial"]:
            materialName = materialDef["name"]
            if "advanced" not in materialName:
                raise Exception("The keyword *advancedmaterial only allows the use of advanced materials!")
            materialProvider = materialDef.get("provider", None)
            materialID = materialDef.get("id", materialName)

            materialProperties = convertLinesToMixedDictionary(materialDef["datalines"])
            materialClass = getMaterialClass(materialName, materialProvider)

            if materialClass is None:  # for Marmot
                raise Exception("Advanced materials are currently not possible with this material provider!")
            else:  # for DisplacementElement
                model.materials[materialID] = materialClass(materialProperties)

        return model

    def createConstraintsFromInputFile(self, model, inputFile):
        """Collects constraint defintions from the input file.

        Parameters
        ----------
        model
            A dictionary containing the model tree.
        inputFile
            A dictionary contaning the input file tree.

        Returns
        -------
        dict
            The updated model tree.
        """

        for definition in inputFile["constraint"]:
            constraintKwArgs = CaseInsensitiveDict(definition.copy())

            name = constraintKwArgs.pop("name")
            constraintType = constraintKwArgs.pop("type")
            data = constraintKwArgs.pop("datalines")

            module = inputLanguage["constraint"].getModule(constraintType)

            args, kwargs = module.parseDatalines(data)

            constraint = getConstraintClass(constraintType)(name, model, **kwargs)
            model.constraints[name] = constraint

        return model

    def createSectionsFromInputFile(self, model, inputFile):
        """Collects section defintions from the input file.
        Assigns properties and section properties to all elements by
        the given section definitions.

        Parameters
        ----------
        model
            A dictionary containing the model tree.
        inputFile
            A dictionary contaning the input file tree.

        Returns
        -------
        dict
            The updated model tree.
        """

        for definition in inputFile["section"]:
            sectionKwArgs = CaseInsensitiveDict(definition.copy())

            name = sectionKwArgs.pop("name")
            sectionType = sectionKwArgs.pop("type")
            materialName = sectionKwArgs.pop("material")
            data = sectionKwArgs.pop("datalines")
            moduleOptions = sectionKwArgs.pop("moduleOptions")

            sectionKwArgs.pop("inputfile")

            if name in model.sections:
                raise Exception(f"Section with name {name} already exists")

            module = inputLanguage["section"].getModule(sectionType)

            args, kwargs = module.parseDatalines(data)
            # sectionKwArgs.update(kwargs)

            for elSet in args:
                if elSet not in model.elementSets:
                    raise Exception(
                        f"During parsing of keyword {keywordIdentifier}section: Element set {elSet} does not exist."
                    )

            if kwargs:
                raise Exception(
                    f"During parsing of keyword {keywordIdentifier}section: Unexpected keyword arguments. Use module level keyword identifier {moduleLevelKeywordIdentifier} instead."
                )

            sectionFactory = getSectionFactoryByName(sectionType)

            try:
                section = sectionFactory(name, model, materialName, args, moduleOptions, **sectionKwArgs)
            except ValueError as e:
                e.args = (
                    f"Error during parsing of keyword {keywordIdentifier}section (type={sectionType}): " + e.args[0],
                )
                raise e

            model.sections[name] = section

        return model

    def createAnalyticalFieldsFromInputFile(self, model, inputFile):
        """Collects field defintions from the input file.

        Parameters
        ----------
        model
            A dictionary containing the model tree.
        inputFile
            A dictionary contaning the input file tree.

        Returns
        -------
        dict
            The updated model tree.
        """

        for definition in inputFile["analyticalField"]:
            analyticalFieldName = definition["name"]
            analyticalFieldType = definition["type"]
            data = definition["datalines"]

            if analyticalFieldName in model.analyticalFields:
                raise Exception(f"AnalyticalField with name {analyticalFieldName} already exists")

            # analytical fields accept no module level keywords
            analyticalFieldKwargs = convertLinesToStringDictionary(data)

            analyticalFieldFactory = getAnalyticalFieldFactoryByName(analyticalFieldType)
            analyticalField = analyticalFieldFactory(analyticalFieldName, model, **analyticalFieldKwargs)

            model.analyticalFields[analyticalFieldName] = analyticalField

        return model

    def createElementPropertiesFromInputFile(self, model, inputFile):
        """Collects element property definitions from the input file and stores them in the model.

        Parameters
        ----------
        model
            The model object.
        inputFile
            The input file dictionary.

        Returns
        -------
        model
            The updated model object.
        """
        from edelweissfe.elements.elementproperty import ElementProperty

        for definition in inputFile.get("elementproperty", []):
            elSetName = definition["elSet"]
            propertyName = definition["propertyName"]
            data = definition["datalines"]

            values_str = " ".join(data).replace(",", " ")
            values = np.array([float(x) for x in values_str.split()], dtype=float)

            elementProperty = ElementProperty(elSetName, propertyName, values)
            model.elementProperties.append(elementProperty)

        return model
