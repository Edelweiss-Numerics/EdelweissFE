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
# Created on Sun Jan 15 14:22:48 2017

# @author: Matthias Neuner

import datetime
import os
from collections import defaultdict
from dataclasses import dataclass
from io import TextIOBase

import numpy as np

from edelweissfe.models.femodel import FEModel
from edelweissfe.outputmanagers.base.outputmanagerbase import OutputManagerBase
from edelweissfe.points.node import Node
from edelweissfe.rigidbodies.rigidbody import RigidBody
from edelweissfe.sets.elementset import ElementSet
from edelweissfe.sets.nodeset import NodeSet
from edelweissfe.utils.fieldoutput import (
    ElementFieldOutput,
    NodeFieldOutput,
    _FieldOutputBase,
)
from edelweissfe.utils.meshtools import disassembleElsetToEnsightShapes
from edelweissfe.utils.schema import schemaField, subKeywordField

"""
Output manager for Ensight exports.
If loaded, it automatically exports all elSets as Ensight parts.
For each part, perNode and perElement results can be exported, which are imported from fieldOutputs.

"""


@dataclass(frozen=True)
class EnsightPerNodeSchema:
    """The options of a single ``>>perNode`` block."""

    fieldOutput: str | None = schemaField(
        description="Name of the result, defined on an elSet (also for perNode results!)",
        dtype=str,
        default=None,
        required=True,
    )


@dataclass(frozen=True)
class EnsightPerElementSchema:
    """The options of a single ``>>perElement`` block."""

    fieldOutput: str | None = schemaField(
        description="Name of the result, defined on an elSet (also for perNode results!)",
        dtype=str,
        default=None,
        required=True,
    )


@dataclass(frozen=True)
class EnsightConfigurationSchema:
    """The options of a single ``>>configuration`` block.

    These defaults are also what an ensight export uses when no ``>>configuration`` block is given
    at all. Note that ``overwrite`` defaults to ``False``, i.e. an export directory is by default
    suffixed with a timestamp rather than overwritten.
    """

    overwrite: bool = schemaField(description="Overwrite results.", dtype=bool, default=False)
    intermediateSaveInterval: int | None = schemaField(
        description="Set intermediate save interval.", dtype=int, default=10
    )
    elSet: str | None = schemaField(description="Element set.", dtype=str, default=None)
    nSet: str | None = schemaField(description="Node set.", dtype=str, default=None)
    transient: bool = schemaField(description="Set transient ensight output.", dtype=bool, default=True)


@dataclass(frozen=True)
class EnsightSchema:
    """The options this output manager accepts, owned by this module and never mutated from
    outside it.

    Ensight's grammar is not a flat option list: it is a set of repeatable ``>>`` sub-keyword
    blocks, mirrored one-for-one here via :func:`edelweissfe.utils.schema.subKeywordField`. Each
    field therefore holds a *tuple* of per-block schema instances, in file order. ``configurations``
    answers to the sub-keyword ``>>configuration`` (singular) but is a tuple like the others, since
    repeating the block is not forbidden.

    ``intermediateSaveInterval``/``minDTForOutput`` are not read at construction time: they exist so
    a later ``>>options, name=<this export's name>, ...`` block (``stepactions/options.py``) has
    something to validate against and :meth:`OutputManager.applyOptionsOverride` to apply --
    adjusting the running export mid-job without repeating its full ``>>configuration``. Not writing
    either leaves whatever the manager was already configured with in place, so both are marked
    :attr:`~edelweissfe.utils.schema.SchemaFieldMeta.optionsOverrideOnly`: reachable through
    ``>>options`` but not part of this keyword's own line/``>>``-block grammar.
    """

    perNode: tuple[EnsightPerNodeSchema, ...] = subKeywordField(
        description="Node-based Ensight export.", schema=EnsightPerNodeSchema
    )
    perElement: tuple[EnsightPerElementSchema, ...] = subKeywordField(
        description="Element-based Ensight export.", schema=EnsightPerElementSchema
    )
    configurations: tuple[EnsightConfigurationSchema, ...] = subKeywordField(
        description="", schema=EnsightConfigurationSchema, optionName="configuration"
    )
    intermediateSaveInterval: int | None = schemaField(
        description="Set the intermediate save interval for the Ensight export. Not writing it "
        "leaves whatever the '>>configuration' block set (or its own default) in place.",
        dtype=int,
        default=None,
        optionsOverrideOnly=True,
    )
    minDTForOutput: float | None = schemaField(
        description="Set the minimum time between two Ensight exports. Not writing it leaves no " "minimum in place.",
        dtype=float,
        default=None,
        optionsOverrideOnly=True,
    )


def writeCFloat(f, ndarray):
    np.asarray(ndarray, dtype=np.float32).tofile(f)


def writeCInt(f, ndarray):
    np.asarray(ndarray, dtype=np.int32).tofile(f)


def writeC80(f, string):
    np.asarray(string, dtype="S80").tofile(f)


ensightPerNodeVariableTypes = {
    1: "scalar per node",
    3: "vector per node",
    6: "tensor symm per node",
    9: "tensor asym per node",
}

ensightPerElementVariableTypes = {
    1: "scalar per element",
    3: "vector per element",
    6: "tensor symm per element",
    9: "tensor asym per element",
}


class EnsightUnstructuredPart:
    """Represents an unstructured ENsight part, defined by a list of nodes and a dictionary of elements.
    Each dictionary entry consists of a list of tuples of elementlabel and nodelist:

    Parameters
    ----------
    description
        A string describing the name of this part
    partNumber
        A unique integer identifying this part
    nodes
        The list of nodes in this part
    elementTree
        A dictionary, with entries for
            - each element shape,
            - containing a list of elements
                - defined by a tuple of
                    - a label and
                    - the node index list.
    """

    def __init__(
        self,
        description: str,
        partNumber: int,
        nodes: list[Node],
        elementTree: dict[str, dict[int, list[Node]]],
    ):
        self.structureType = "coordinates"
        self.elementTree = elementTree

        self.nodes = nodes
        self.nodeLabels = np.asarray([node.label for node in nodes], np.int32)

        self.description = description  # string, describing the part; max. 80 characters
        self.partNumber = partNumber
        self.nodeCoordinateArray = np.asarray([node.coordinates for node in nodes])

    def writeToFile(
        self,
        binaryFileHandle=TextIOBase,
        printNodeLabels: bool = True,
        printElementLabels: bool = True,
    ):
        """
        Write the part to a file.

        Parameters
        ----------
        binaryFileHandle
            The file handle for writing.
        printNodeLabels
            Write the node labels.
        printElementLabels
            Write the element labels.
        """

        if len(self.nodeCoordinateArray.shape) > 1 and self.nodeCoordinateArray.shape[1] < 3:
            extendTo3D = True
        else:
            extendTo3D = False

        nNodes = self.nodeCoordinateArray.shape[0]
        f = binaryFileHandle

        writeC80(f, "part")
        writeCInt(f, self.partNumber)
        writeC80(f, self.description)
        writeC80(f, "coordinates")
        writeCInt(f, nNodes)

        if printNodeLabels and self.nodeLabels is not None:
            writeCInt(f, self.nodeLabels)

        writeCFloat(f, self.nodeCoordinateArray.T)

        if extendTo3D:
            writeCFloat(
                f,
                np.zeros(self.nodeCoordinateArray.shape[0] * (3 - self.nodeCoordinateArray.shape[1])),
            )

        for elemType, elements in self.elementTree.items():
            writeC80(f, elemType)
            writeCInt(f, len(elements))
            if printElementLabels:
                writeCInt(f, np.asarray([elNumber for elNumber in elements.keys()], np.int32))

            for nodeIndices in elements.values():
                writeCInt(f, np.asarray(nodeIndices, np.int32)[:] + 1)


class EnsightTimeSet:
    """Represents a set which may be used by EnsightGeometry, EnsightStructuredPart, EnsightUnstructuredPart and is written into the case file.

    Parameters
    ----------
    number
        The unique number of this time set.
    description
        A description of this time set.
    fileNameStartNumber
        Counter start of the multiple files counter.
    fileNameNumberIncrement
        Counter increment of the multpple files counter.
    timeValues
        A list of time values.
    """

    def __init__(
        self,
        number: int = 1,
        description: str = "timeStepDesc",
        fileNameStartNumber: int = 0,
        fileNameNumberIncrement: int = 1,
        timeValues: list = None,
    ):
        self.number = number
        self.description = description
        self.fileNameStartNumber = fileNameStartNumber
        self.fileNameNumberIncrement = fileNameNumberIncrement
        self.timeValues = timeValues if timeValues is not None else []


class EnsightGeometry:
    """Container class for one or more EnsightParts at a certain time state, handles also the file writing operation.

    Parameters
    ----------
    name
        The name.
    descriptionLine1
        Description line 1.
    descriptionLine2
        Description line 2.
    ensightPartList
        The list of unstructured parts in this geometry.
    nodeIdOption
        Ensight option for handling the node ids.
    elementIdOption
        Ensight option for handling the elemnt ids.
    """

    def __init__(
        self,
        name: str = "geometry",
        descriptionLine1: str = "",
        descriptionLine2: str = "",
        ensightPartList: list[EnsightUnstructuredPart] = None,
        nodeIdOption: str = "given",
        elementIdOption: str = "given",
    ):
        self.name = name
        self.descLine1 = descriptionLine1
        self.descLine2 = descriptionLine2
        self.partList = ensightPartList if ensightPartList is not None else []
        self.nodeIdOption = nodeIdOption
        self.elementIdOption = elementIdOption

    def writeToFile(self, fileHandle: TextIOBase):
        """Write the variable to a file."

        Parameters
        ----------
        fileHandle
            The file handle for writing the file.
        """
        f = fileHandle
        writeC80(f, self.descLine1)
        writeC80(f, self.descLine2)
        writeC80(f, "node id " + self.nodeIdOption)
        writeC80(f, "element id " + self.elementIdOption)

        if self.nodeIdOption == "given" or self.nodeIdOption == "ignore":
            printNodeLabels = True
        else:  # assign or off
            printNodeLabels = False

        if self.elementIdOption == "given" or self.nodeIdOption == "ignore":
            printElementLabels = True
        else:  # assign or off
            printElementLabels = False

        for part in self.partList:
            part.writeToFile(f, printNodeLabels, printElementLabels)


class EnsightGeometryTrend:
    """Container class for the time dependent evolution of the geometry,
    establishes the connection between the geometry entities and a EnsightTimeSet.

    Parameters
    ----------
    ensightTimeSet
        The Timeset.
    ensightGeometryList
        A list of evolving Ensight Geometries.
    """

    def __init__(
        self,
        ensightTimeSet: EnsightTimeSet,
        ensightGeometryList: list[EnsightGeometry] = None,
    ):
        self.timeSet = ensightTimeSet
        self.geometryList = ensightGeometryList if ensightGeometryList is not None else []


class EnsightVariableTrend:
    """Container class for the time dependent evolution of one variable,
    establishes the connection between EnsightVariable entities and a EnsighTimeSet.

    Parameters
    ----------
    ensightTimeSet
        The timeset.
    variableName
        The name of this variable.
    ensightVariableList
        The list of variables over time.
    variableType
        The Ensight valid type of this variable.
    description
        The description of this variable.
    """

    def __init__(
        self,
        ensightTimeSet: EnsightTimeSet,
        variableName: str,
        ensightVariableList: list = None,
        variableType="scalar per node",
        description="variableTrendDescription",
    ):
        self.timeSet = ensightTimeSet
        self.variableName = variableName
        self.variableList = ensightVariableList if ensightVariableList is not None else []
        self.variableType = variableType
        self.description = description


class EnsightPerNodeVariable:
    """Container class for data for one certain variable, defined for one or more parts (classification by partID), at a certain time state.
    For each part the structuretype ("coordinate" or "block") has to be defined.
    Each part-variable assignment is defined by a dictionary entry of type: { EnsightPart: np.array(variableValues) }

    Parameters
    ----------
    name
        The name of this variable.
    variableDimension
        The size of the variable per value.
    ensightPartsDict
        A dictionary defining the values for given Ensight parts.
    """

    def __init__(
        self,
        name: str,
        ensightPartsDict: dict[EnsightUnstructuredPart, np.ndarray],
        varSize: int,
    ):

        self.name = name.replace(" ", "_")
        self.description = self.name
        self.partsDict = ensightPartsDict or {}  # { EnsightPart: np.array(variableValues) }

        self.varType = ensightPerNodeVariableTypes[varSize]

    def writeToFile(
        self,
        fileHandle: TextIOBase,
    ):
        """Write the variable to a file.

        Parameters
        ----------
        fileHandle
            The file handle for writing.
        """

        f = fileHandle
        writeC80(f, self.description)
        for ensightPartID, (structureType, values) in self.partsDict.items():
            writeC80(f, "part")
            writeCInt(f, ensightPartID)
            writeC80(f, structureType)
            writeCFloat(f, values.T)


class EnsightPerElementVariable:
    """Container class for data for one certain variable, defined for one or more parts (classification by partID), at a certain time state.
    For each part the structuretype ("coordinate" or "block") has to be defined.
    Each part-variable assignment is defined by a dictionary entry of type: { EnsightPart: np.array(variableValues) }

    Parameters
    ----------
    name
        The name of this variable.
    variableDimension
        The size of this variable per entry.
    ensightPartsDict
        The dictionary containing parts and their values.
    """

    def __init__(
        self,
        name: str,
        ensightPartsDict: dict[EnsightUnstructuredPart, np.ndarray],
        varSize: int,
    ):
        self.name = name.replace(" ", "_")
        self.description = self.name
        self.partsDict = ensightPartsDict
        self.varType = ensightPerElementVariableTypes[varSize]

    def writeToFile(self, fileHandle: TextIOBase):
        """Write the variable to a file.

        Parameters
        ----------
        fileHandle
            The file handle for writing.
        """

        f = fileHandle
        writeC80(f, self.description)
        for ensightPartID, elTypeDict in self.partsDict.items():
            writeC80(f, "part")
            writeCInt(f, ensightPartID)
            for elType, values in elTypeDict.items():
                writeC80(f, elType)
                writeCFloat(f, values.T)


class EnsightChunkWiseCase:
    """An Ensight case, containg of time sets, geometry trends and variable trends,
    which can be written in chunks at certain times.

    Parameters
    ----------
    caseName
        The name of this case.
    directory
        The path to write.
    writeTransientSingleFiles
        Write single or multiple files.
    """

    def __init__(self, caseName: str, directory: str = "", writeTransientSingleFiles: bool = True):
        self.directory = directory
        self.caseName = caseName
        self.caseFileNamePrefix = os.path.join(directory, caseName)
        self.writeTransientSingleFiles = writeTransientSingleFiles
        self.timeAndFileSets = {}
        self.geometryTrends = {}
        self.variableTrends = {}
        self.fileNames = {}

        if not os.path.exists(self.caseFileNamePrefix):
            os.makedirs(self.caseFileNamePrefix)

    def setCurrentTime(self, timeAndFileSetNumber: int, timeValue: float):
        """Set the current time of the case.
        Parameters
        ----------
        timeAndFileSetNumber
            The number of the file and time set.
        timeValue
            The time value.
        """

        if timeAndFileSetNumber not in self.timeAndFileSets:
            self.timeAndFileSets[timeAndFileSetNumber] = EnsightTimeSet(timeAndFileSetNumber, "no description", 0, 1)
        tfSet = self.timeAndFileSets[timeAndFileSetNumber]
        tfSet.timeValues.append(timeValue)

    def writeGeometryTrendChunk(self, ensightGeometry: EnsightGeometryTrend, timeAndFileSetNumber: int = 1):
        """
        Write a chunk of geometry trend.

        Parameters
        ----------
        ensightGeometry
            The trend to write.
        timeAndFileSetNumber
            The associated time and fileset number.
        """

        if self.writeTransientSingleFiles:
            if ensightGeometry.name not in self.fileNames:
                fileName = os.path.join(
                    self.caseFileNamePrefix,
                    ensightGeometry.name + ".geo",
                )
                self.fileNames[ensightGeometry.name] = fileName
                # create empty file
                with open(fileName, mode="wb") as f:
                    pass

            filename = self.fileNames[ensightGeometry.name]

            with open(filename, mode="ab") as f:
                if ensightGeometry.name not in self.geometryTrends:
                    self.geometryTrends[ensightGeometry.name] = timeAndFileSetNumber
                    writeC80(f, "C Binary")

                writeC80(f, "BEGIN TIME STEP")
                ensightGeometry.writeToFile(f)
                writeC80(f, "END TIME STEP")
        else:
            if timeAndFileSetNumber not in self.timeAndFileSets:
                stepIndex = 0
            else:
                stepIndex = len(self.timeAndFileSets[timeAndFileSetNumber].timeValues)

            if ensightGeometry.name not in self.geometryTrends:
                self.geometryTrends[ensightGeometry.name] = timeAndFileSetNumber

            multiFileName = os.path.join(self.caseFileNamePrefix, f"{ensightGeometry.name}.geo_{stepIndex:04d}")
            with open(multiFileName, mode="wb") as f:
                writeC80(f, "C Binary")
                ensightGeometry.writeToFile(f)

    def writeVariableTrendChunk(self, ensightVariable: EnsightVariableTrend, timeAndFileSetNumber: int = 2):
        """
        Write a chunk of variable trend.

        Parameters
        ----------
        ensightVariable
            The trend to write.
        timeAndFileSetNumber
            The associated time and fileset number.
        """

        if self.writeTransientSingleFiles:
            if ensightVariable.name not in self.fileNames:
                # create file name
                fileName = os.path.join(self.caseFileNamePrefix, ensightVariable.name + ".var")
                # append to file names
                self.fileNames[ensightVariable.name] = fileName
                # create empty file
                with open(fileName, mode="wb") as f:
                    pass

            filename = self.fileNames[ensightVariable.name]

            with open(filename, mode="ab") as f:
                if ensightVariable.name not in self.variableTrends:
                    self.variableTrends[ensightVariable.name] = (
                        timeAndFileSetNumber,
                        ensightVariable.varType,
                    )
                    writeC80(f, "C Binary")

                writeC80(f, "BEGIN TIME STEP")
                ensightVariable.writeToFile(f)
                writeC80(f, "END TIME STEP")
        else:
            if timeAndFileSetNumber not in self.timeAndFileSets:
                stepIndex = 0
            else:
                stepIndex = len(self.timeAndFileSets[timeAndFileSetNumber].timeValues)

            if ensightVariable.name not in self.variableTrends:
                self.variableTrends[ensightVariable.name] = (
                    timeAndFileSetNumber,
                    ensightVariable.varType,
                )

            multiFileName = os.path.join(self.caseFileNamePrefix, f"{ensightVariable.name}.var_{stepIndex:04d}")
            with open(multiFileName, mode="wb") as f:
                ensightVariable.writeToFile(f)

    def finalize(self, replaceTimeValuesByEnumeration: bool = True, closeFileHandes: bool = True):
        """Write the file .case file containing all the required information."

        Parameters
        ----------
        replaceTimeValuesByEnumeration
            Remove the factor of time and make integers as discrete steps only.
        closeFileHandes
            Close the file handles after writing.
        """

        caseFName = self.caseFileNamePrefix + ".case"

        with open(caseFName, mode="w") as cf:
            cf.write("FORMAT\n")
            cf.write("type: ensight gold\n")

            cf.write("TIME\n")
            for setNum, timeSet in self.timeAndFileSets.items():
                cf.write("time set: " + str(setNum) + " no description\n")
                cf.write("number of steps: " + str(len(timeSet.timeValues)) + "\n")
                cf.write("filename start number: " + str(timeSet.fileNameStartNumber) + "\n")
                cf.write("filename increment: " + str(timeSet.fileNameNumberIncrement) + "\n")
                cf.write("time values: ")
                for i, timeVal in enumerate(timeSet.timeValues):
                    if not replaceTimeValuesByEnumeration:
                        cf.write("{:1.8e}".format(timeVal) + "\n")
                    else:
                        cf.write("{:}".format(i) + "\n")

            if self.writeTransientSingleFiles:
                cf.write("FILE\n")
                for timeSet in self.timeAndFileSets.values():
                    cf.write("file set: {:}\n".format(timeSet.number))
                    cf.write("number of steps: {:}\n".format(len(timeSet.timeValues)))

            cf.write("GEOMETRY\n")
            for geometryName, tAndFSetNum in self.geometryTrends.items():
                if self.writeTransientSingleFiles:
                    geoFile = os.path.join(self.caseFileNamePrefix, geometryName + ".geo")
                    cf.write(
                        "model: {:} {:} {:}\n".format(
                            tAndFSetNum,
                            tAndFSetNum,
                            geoFile,
                        )
                    )
                else:
                    geoFile = os.path.join(self.caseFileNamePrefix, geometryName + ".geo_****")
                    cf.write(
                        "model: {:} {:}\n".format(
                            tAndFSetNum,
                            geoFile,
                        )
                    )

            cf.write("VARIABLE\n")
            for variableName, (
                tAndFSetNum,
                variableType,
            ) in self.variableTrends.items():
                if self.writeTransientSingleFiles:
                    varFile = os.path.join(self.caseFileNamePrefix, variableName + ".var")
                    cf.write(
                        "{:}: {:} {:} {:} {:}\n".format(
                            variableType,
                            tAndFSetNum,
                            tAndFSetNum,
                            variableName,
                            varFile,
                        )
                    )
                else:
                    varFile = os.path.join(self.caseFileNamePrefix, variableName + ".var_****")
                    cf.write(
                        "{:}: {:} {:} {:}\n".format(
                            variableType,
                            tAndFSetNum,
                            variableName,
                            varFile,
                        )
                    )


def createUnstructuredPartFromElementSet(setName, elementSet: list, partID: int):
    """Determines the element and node list for an Ensightpart from an
    element set. The reduced, unique node set is generated, as well as
    the element to node index mapping for the ensight part.

    Parameters
    ----------
    elementSet
        The list of elements defining this part.
    partID
        The id of this part.
    """

    nodeCounter = 0
    partNodes = dict()
    elementDict = dict()
    for element in elementSet:
        elShape = element.ensightType
        if elShape not in elementDict:
            elementDict[elShape] = dict()
        elNodeIndices = []
        for node in element.visualizationNodes:
            # if the node is already in the dict, get its index,
            # else insert it, and get the current idx = counter. increase the counter
            idx = partNodes.setdefault(node, nodeCounter)
            elNodeIndices.append(idx)
            if idx == nodeCounter:
                # the node was just inserted, so increase the counter of inserted nodes
                nodeCounter += 1
        elementDict[elShape][element.elNumber] = elNodeIndices

    return EnsightUnstructuredPart(setName, partID, partNodes.keys(), elementDict)


def createUnstructuredPartFromNodeSet(setName, nodeSet: list, partID: int):
    """Construtcts an EnSight part for a node set. Since EnSight parts comprise nodes and elements, each node is assigned to a dummy point element.

    Parameters
    ----------
    nodeSet
        The list of nodes defining this part.
    partID
        The id of this part.
    """

    elementDict = dict()

    elementDict["point"] = {i: [i] for i in range(len(nodeSet))}

    return EnsightUnstructuredPart("NSET_" + setName, partID, list(nodeSet), elementDict)


def createUnstructuredPartFromRigidBody(bodyName, rigidBody, partID: int):
    """Determines the element and node list for an Ensightpart from a
    RigidBody. The reduced, unique node set is generated, as well as
    the element to node index mapping for the ensight part.

    Parameters
    ----------
    bodyName
        The name of the rigid body.
    rigidBody
        The rigid body object.
    partID
        The id of this part.
    """

    # the node list must follow getVisualizationNodes()'s order, since that is the
    # order in which RigidBodyFieldOutput.getVisualizationField() reports per-node
    # results - deriving it from the facets instead (e.g. by first-seen order, or
    # excluding nodes unreferenced by any facet) would misalign or drop entries.
    visualizationNodes = rigidBody.getVisualizationNodes()
    nodeIndices = {node: idx for idx, node in enumerate(visualizationNodes)}

    elementDict = dict()
    facets = rigidBody.getVisualizationElements()

    facetID = 1
    for facet in facets:
        elShape = facet["type"]
        if elShape not in elementDict:
            elementDict[elShape] = dict()
        elementDict[elShape][facetID] = [nodeIndices[node] for node in facet["nodes"]]
        facetID += 1

    return EnsightUnstructuredPart("RIGIDBODY_" + bodyName, partID, visualizationNodes, elementDict)


class OutputManager(OutputManagerBase):
    identification = "Ensight Export"

    #: Option schema for this output manager, per OptionSchemaProvider.
    schema = EnsightSchema

    def __init__(
        self,
        name: str,
        model: FEModel,
        fieldOutputController,
        journal,
        plotter,
        *,
        configuration: EnsightSchema = EnsightSchema(),
    ):
        """Constructible standalone, with no parser involvement. Options arrive as an
        already-validated, already-typed schema instance.

        Parameters
        ----------
        name
            The name of this output manager.
        model
            The model tree.
        fieldOutputController
            The field output controller instance.
        journal
            The journal instance for logging.
        plotter
            The plotter instance.
        configuration
            The options this output manager accepts, including its ``>>perNode``, ``>>perElement``
            and ``>>configuration`` blocks.
        """
        perNodeDefs = configuration.perNode
        perElementDefs = configuration.perElement
        configurations = configuration.configurations

        self.name = name

        self.model = model
        self.timeAtLastOutput = -1e16
        self.minDTForOutput = -1e16
        self.finishedSteps = 0
        self.intermediateSaveIntervalCounter = 0
        self.fieldOutputController = fieldOutputController
        self.journal = journal
        self.overwrite = True

        self.transientTAndFSetNumber = 1
        self.transientVariableTAndFSetNumber = 2
        self.staticTAndFSetNumber = 2

        self.elSetToEnsightPartMappings = {}
        self.nSetToEnsightPartMappings = {}
        self.rigidBodyToEnsightPartMappings = {}

        self._transientPerNodeVariableJobs = defaultdict(list)
        self._transientPerElementVariableJobs = defaultdict(list)

        self.exportName = name

        self.geometryParts = self._createGeometryParts(1)

        # Defaults come directly from the schema.
        defaults = EnsightConfigurationSchema()
        val = defaults.intermediateSaveInterval
        self.intermediateSaveInterval = int(val) if val is not None else None
        self.overwrite = defaults.overwrite
        transient = defaults.transient
        configSetName = None
        configIsNodeSet = None

        # A repeated `>>configuration` is not rejected: every scalar option is last-wins, but
        # `configSetName` carries over from an earlier block if the last one names neither nSet nor
        # elSet.
        for configurationBlock in configurations:
            self.intermediateSaveInterval = configurationBlock.intermediateSaveInterval
            transient = configurationBlock.transient
            self.overwrite = configurationBlock.overwrite

            if configurationBlock.nSet:
                configSetName = configurationBlock.nSet
                configIsNodeSet = True
            elif configurationBlock.elSet:
                configSetName = configurationBlock.elSet
                configIsNodeSet = False

        if not self.overwrite:
            self.exportName = "{:}_{:}".format(self.name, datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))

        # store the definitions so parts + variable jobs can be rebuilt when the mesh changes (AMR)
        self._perNodeDefs = perNodeDefs
        self._perElementDefs = perElementDefs
        self._fieldOutputController = fieldOutputController
        self._configSetName = configSetName
        self._configIsNodeSet = configIsNodeSet
        self._configPart = None
        self._resolveConfigPart()
        self._transientCfg = transient
        self._initialMeshSignature = (len(self.model.elements), len(self.model.nodes))
        self._meshSignature = None
        self._buildVariableJobs()

    def _resolveConfigPart(self):
        """(Re)resolve `self._configPart` from the configured set name/kind against the current
        nSetToEnsightPartMappings/elSetToEnsightPartMappings, so it stays consistent with the geometry
        parts after a mesh change (AMR). Leaves `self._configPart` as None if no set was configured."""
        if self._configSetName is None:
            self._configPart = None
        elif self._configIsNodeSet:
            self._configPart = self.nSetToEnsightPartMappings[self._configSetName]
        else:
            self._configPart = self.elSetToEnsightPartMappings[self._configSetName]

    def _buildVariableJobs(self):
        """(Re)create the per-node/per-element variable jobs from the stored definitions against the
        current geometry parts. Called at setup and again whenever the mesh changes (AMR)."""
        for definition in self._perNodeDefs:
            fieldOutput = self._fieldOutputController.fieldOutputs[definition.fieldOutput]
            name = fieldOutput.name.replace(" ", "_")
            self.createPerNodeOutput(fieldOutput, self._configPart, name, transient=self._transientCfg)

        for definition in self._perElementDefs:
            fieldOutput = self._fieldOutputController.fieldOutputs[definition.fieldOutput]
            name = fieldOutput.name.replace(" ", "_")
            self.createPerElementOutput(fieldOutput, self._configPart, name, transient=self._transientCfg)

    def _rebuildForMeshChange(self):
        """Rebuild geometry parts + variable jobs after an AMR mesh change, so both stay consistent
        with the current (refined) mesh."""
        self.elSetToEnsightPartMappings = {}
        self.nSetToEnsightPartMappings = {}
        self.rigidBodyToEnsightPartMappings = {}
        self._transientPerNodeVariableJobs = defaultdict(list)
        self._transientPerElementVariableJobs = defaultdict(list)
        self.geometryParts = self._createGeometryParts(1)
        self._resolveConfigPart()
        self._buildVariableJobs()

    def createPerElementOutput(
        self,
        fieldOutput: ElementFieldOutput,
        part: set = None,
        name: str = None,
        transient: bool = True,
        varSize: int = None,
    ):
        """Create a per element output job.

        Parameters
        ----------
        fieldOutput
            The field output to export.
        part
            The part to which the output belongs. If not specified, the part is determined from the field output.
        name
            The name of the output. If not specified, the name of the field output is taken.
        transient
            Whether the output is transient.
        varSize
            The size of the variable. If not specified, the size of the field output is taken --
            promoted from 2 to 3 components for a 2D-domain vector field, since Ensight expects a
            3-component vector even in 2D (the implicit z-component is 0).
        """

        variableJob = dict()
        variableJob["fieldOutput"] = fieldOutput
        variableJob["part"] = part
        if not name:
            name = fieldOutput.name
        variableJob["name"] = name
        variableJob["transient"] = transient

        nEntries, varSizeFp = self._ensureArrayIs2D(fieldOutput.getLastResult()).shape
        if self.model.domainSize == 2 and varSizeFp == 2:
            varSizeFp = 3

        if not part:
            part = self._getTargetPartForFieldOutput(fieldOutput)
        variableJob["part"] = part

        if nEntries != len(fieldOutput.associatedSet):
            raise Exception(
                "Variable {:} result size ({:}) does not match the number of nodes ({:})".format(
                    variableJob["name"], nEntries, len(variableJob["part"].nodes)
                )
            )

        if not varSize:
            varSize = varSizeFp
        variableJob["varSize"] = varSize

        variableJob["elementsOfShape"] = disassembleElsetToEnsightShapes(fieldOutput.associatedSet)

        if transient:
            self._transientPerElementVariableJobs[variableJob["name"]].append(variableJob)
        else:
            raise Exception("Only transient per element outputs are supported!")

    def createPerNodeOutput(
        self,
        fieldOutput: NodeFieldOutput,
        part: EnsightUnstructuredPart = None,
        name: str = None,
        transient: bool = True,
        varSize: int = None,
    ):
        """Create a per node output job.

        Parameters
        ----------
        fieldOutput
            The field output to export.
        part
            The part to which the output belongs. If not specified, the part is determined from the field output.
        name
            The name of the output. If not specified, the name of the field output is taken.
        transient
            Whether the output is transient.
        varSize
            The size of the variable. If not specified, the size of the field output is taken --
            promoted from 2 to 3 components for a 2D-domain vector field, since Ensight expects a
            3-component vector even in 2D (the implicit z-component is 0).
        """

        variableJob = dict()
        variableJob["fieldOutput"] = fieldOutput
        variableJob["part"] = part
        if not name:
            name = fieldOutput.name
        variableJob["name"] = name
        variableJob["transient"] = transient

        nEntries, varSizeFp = self._ensureArrayIs2D(fieldOutput.getLastResult()).shape
        if self.model.domainSize == 2 and varSizeFp == 2:
            varSizeFp = 3
        if not varSize:
            varSize = varSizeFp

        variableJob["varSize"] = varSize

        if not part:
            part = self._getTargetPartForFieldOutput(fieldOutput)
        variableJob["part"] = part

        if nEntries != len(variableJob["part"].nodes):
            raise Exception(
                "Variable {:} result size ({:}) does not match the number of nodes ({:})".format(
                    variableJob["name"], nEntries, len(variableJob["part"].nodes)
                )
            )

        if transient:
            self._transientPerNodeVariableJobs[variableJob["name"]].append(variableJob)
        else:
            raise Exception("Only transient per node outputs are supported!")

    def initializeJob(self):
        self.ensightCase = EnsightChunkWiseCase(self.exportName, writeTransientSingleFiles=False)
        # Geometry is written per output step (on the transient time set) rather than once, so it can
        # change with adaptive mesh refinement and stay 1:1 aligned with the variable time steps.

    def initializeStep(self, step):
        # intermediateSaveInterval/minDTForOutput overrides are pushed directly by
        # applyOptionsOverride as soon as a step's >>options block is constructed, so there is
        # nothing to do here.
        pass

    def applyOptionsOverride(self, fieldValues: dict) -> None:
        """Adjust the running export's intermediate-save interval and/or minimum output spacing.

        See :meth:`~edelweissfe.outputmanagers.base.outputmanagerbase.OutputManagerBase.applyOptionsOverride`
        for the calling convention. Named fields, not a generic loop over ``fieldValues``: this class
        has exactly two overridable options, and mapping schema-field-name to instance-attribute
        generically would mean ``setattr``, which this codebase's conventions forbid.

        Parameters
        ----------
        fieldValues
            Maps schema field name (``intermediateSaveInterval``, ``minDTForOutput``) to its new,
            already-coerced value; either may be absent (only whatever the user actually wrote is
            present at all).
        """

        if "intermediateSaveInterval" in fieldValues:
            self.intermediateSaveInterval = fieldValues["intermediateSaveInterval"]
        if "minDTForOutput" in fieldValues:
            self.minDTForOutput = fieldValues["minDTForOutput"]

    def finalizeIncrement(self, **kwargs):
        time = self.model.time

        # check if we should write output, i.e., if enough time has passed:
        timeSinceLastOutput = time - self.timeAtLastOutput

        if self.minDTForOutput - timeSinceLastOutput > 1e-16:
            self.journal.message("Skipping output".format(), self.identification, 1)
            return

        self.writeOutput(self.model)

    def finalizeFailedIncrement(self, **kwargs):
        pass

    def writeOutput(self, model: FEModel):
        self.timeAtLastOutput = model.time

        # rebuild parts + variable jobs if the mesh changed (AMR), so geometry and variables match
        signature = (len(model.elements), len(model.nodes))
        mesh_changed = False

        if self._meshSignature is None:
            mesh_changed = True
            if signature != self._initialMeshSignature:
                self._rebuildForMeshChange()
        elif signature != self._meshSignature:
            self._rebuildForMeshChange()
            mesh_changed = True

        self._meshSignature = signature

        # write the current geometry only if it changed
        if mesh_changed:
            geometry = EnsightGeometry("geometry", "EdelweissFE", "*export*", ensightPartList=self.geometryParts)
            self.ensightCase.writeGeometryTrendChunk(geometry, self.transientTAndFSetNumber)
            self.ensightCase.setCurrentTime(self.transientTAndFSetNumber, model.time)

        for (
            resultName,
            perNodeVariableJobs,
        ) in self._transientPerNodeVariableJobs.items():
            resultsByParts = {}
            for perNodeVariableJob in perNodeVariableJobs:
                result = self._ensureArrayIs2D(perNodeVariableJob["fieldOutput"].getLastResult())

                if self.model.domainSize == 2 and result.shape[1] == 2:
                    result = self._make2DVector3D(result)

                resultsByParts[perNodeVariableJob["part"].partNumber] = (
                    "coordinates",
                    result,
                )
            enSightVariable = EnsightPerNodeVariable(resultName, resultsByParts, perNodeVariableJob["varSize"])
            self.ensightCase.writeVariableTrendChunk(enSightVariable, self.transientVariableTAndFSetNumber)
            del enSightVariable

        for (
            resultName,
            perElementVariableJobs,
        ) in self._transientPerElementVariableJobs.items():
            resultsByParts = {}
            for perElementVariableJob in perElementVariableJobs:
                part = perElementVariableJob["part"]
                elementsOfShape = perElementVariableJob["elementsOfShape"]
                result = self._ensureArrayIs2D(perElementVariableJob["fieldOutput"].getLastResult())

                if self.model.domainSize == 2 and result.shape[1] == 2:
                    result = self._make2DVector3D(result)

                partResultsByElementShape = {
                    shape: result[elIndicesOfShape] for shape, elIndicesOfShape in elementsOfShape.items()
                }
                resultsByParts[part.partNumber] = partResultsByElementShape

            enSightVariable = EnsightPerElementVariable(resultName, resultsByParts, perElementVariableJob["varSize"])
            self.ensightCase.writeVariableTrendChunk(enSightVariable, self.transientVariableTAndFSetNumber)
            del enSightVariable

        # register the current time for variables
        self.ensightCase.setCurrentTime(self.transientVariableTAndFSetNumber, model.time)

        # Rewrite the small .case every step so the on-disk step count always matches the committed
        # data even if the job aborts before finalizeJob() (the L-panel dies on GCDP return-mapping).
        # A trailing partial geometry step written just before an abort is simply not counted here, so
        # the reader never overruns it. The .case is tiny text and the data files are already appended
        # every step, so there is no benefit to gating this behind intermediateSaveInterval.
        self.ensightCase.finalize(replaceTimeValuesByEnumeration=False, closeFileHandes=False)

    def finalizeStep(
        self,
    ):
        if self.model.time - self.timeAtLastOutput > 1e-12:
            self.writeOutput(self.model)

        self.finishedSteps += 1

    def finalizeJob(
        self,
    ):
        self.ensightCase.finalize(replaceTimeValuesByEnumeration=False)

    def getRestartData(self) -> dict[str, np.ndarray] | None:
        """This export's transient-sequence bookkeeping: each Ensight time/file set's history of
        already-written time values (flattened CSR-style, since sets can have different lengths),
        plus the mesh signature and ``timeAtLastOutput`` used to decide whether to write a fresh
        geometry chunk / throttle output. Both file numbering (``writeGeometryTrendChunk``/
        ``writeVariableTrendChunk`` derive the next chunk's index from ``len(timeValues)``) and the
        ``.case`` file's own declared step list come directly from this state -- restoring it is
        both necessary and sufficient for a resumed run to continue the same sequence rather than
        starting a fresh, colliding/orphaning one.

        ``None`` if nothing has been written yet (nothing to restore).
        """

        timeAndFileSets = self.ensightCase.timeAndFileSets
        if not timeAndFileSets:
            return None

        setNumbers = sorted(timeAndFileSets)
        sizes = [len(timeAndFileSets[n].timeValues) for n in setNumbers]
        flatTimeValues = [v for n in setNumbers for v in timeAndFileSets[n].timeValues]
        meshSignature = self._meshSignature if self._meshSignature is not None else (-1, -1)

        return {
            "setNumbers": np.array(setNumbers, dtype=int),
            "timeValueSizes": np.array(sizes, dtype=int),
            "timeValues": np.array(flatTimeValues, dtype=float),
            "meshSignature": np.array(meshSignature, dtype=int),
            "timeAtLastOutput": np.array([self.timeAtLastOutput]),
        }

    def setRestartData(self, data: dict[str, np.ndarray]):
        """Restore this export's transient-sequence bookkeeping from a restart checkpoint written
        by :meth:`getRestartData`, so the next chunk written continues the existing sequence
        (correct file numbering, correct ``.case`` step list) instead of starting a fresh one."""

        offset = 0
        for setNumber, size in zip(data["setNumbers"], data["timeValueSizes"]):
            size = int(size)
            timeValues = list(data["timeValues"][offset : offset + size])
            offset += size
            self.ensightCase.timeAndFileSets[int(setNumber)] = EnsightTimeSet(
                int(setNumber), "no description", 0, 1, timeValues
            )

        meshSignature = tuple(int(x) for x in data["meshSignature"])
        self._meshSignature = None if meshSignature == (-1, -1) else meshSignature
        self.timeAtLastOutput = float(data["timeAtLastOutput"][0])

    def _createGeometryParts(self, firstPartID: int):
        model = self.model
        elementSets = model.elementSets

        elSetParts = []
        partCounter = firstPartID
        for setName, elSet in elementSets.items():
            elSetPart = createUnstructuredPartFromElementSet(setName, elSet, partCounter)
            self.elSetToEnsightPartMappings[setName] = elSetPart
            elSetParts.append(elSetPart)
            partCounter += 1

        nodeSets = model.nodeSets

        nodeSetParts = []
        for setName, nodeSet in nodeSets.items():
            nodeSetPart = createUnstructuredPartFromNodeSet(setName, nodeSet, partCounter)
            self.nSetToEnsightPartMappings[setName] = nodeSetPart
            nodeSetParts.append(nodeSetPart)
            partCounter += 1

        rigidBodyParts = []
        for bodyName, body in model.rigidBodies.items():
            bodyPart = createUnstructuredPartFromRigidBody(bodyName, body, partCounter)
            self.rigidBodyToEnsightPartMappings[bodyName] = bodyPart
            rigidBodyParts.append(bodyPart)
            partCounter += 1

        return elSetParts + nodeSetParts + rigidBodyParts

    def _getTargetPartForFieldOutput(self, fieldOutput: _FieldOutputBase) -> EnsightUnstructuredPart:
        """
        Determine depending on the specified input,
        for which Part the result should be written.
        If no input is specified, the associated set of the :class:`FieldOutput is taken.

        Parameters
        ----------
        fieldOutput
            The :class:`FieldOutput which contains the result

        Returns
        -------
        EnsightStructuredPart
            The identified part.
        """
        theSetName = fieldOutput.associatedSet.name

        if isinstance(fieldOutput.associatedSet, NodeSet):
            return self.nSetToEnsightPartMappings[theSetName]

        elif isinstance(fieldOutput.associatedSet, ElementSet):
            return self.elSetToEnsightPartMappings[theSetName]

        elif isinstance(fieldOutput.associatedSet, RigidBody):
            return self.rigidBodyToEnsightPartMappings[theSetName]

        else:
            raise Exception(
                "Ensight Variables need to be explicitly associated with a part, or implicitly through a FieldOutput defined on ElementSets, NodeSets, or RigidBodies!"
            )

    def _ensureArrayIs2D(self, result: np.ndarray) -> np.ndarray:
        """
        Ensure that a result array is in tabular form.

        Parameters
        ----------
        result
            The result in potential 1D form.

        Returns
        -------
        np.ndarray
            The result in guaranteed 2d form.
        """
        return np.reshape(result, (len(result), -1))

    def _make2DVector3D(self, result: np.ndarray) -> np.ndarray:
        """
        Vector results in 2D consist of 2 components.
        However, in Ensight all results must be written for the 3D case.
        This function appends a zero column.

        Parameters
        ----------
        result
            The vector result in 2D form.

        Returns
        -------
        np.ndarray
            The result in 3D form.
        """

        return np.pad(
            result,
            ((0, 0), (0, 1)),
        )
