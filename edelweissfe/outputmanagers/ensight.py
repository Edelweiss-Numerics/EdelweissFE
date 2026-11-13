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
import functools
import os
from collections import defaultdict
from dataclasses import dataclass
from io import TextIOBase

import h5py
import numpy as np
from scipy.sparse import csr_matrix

from edelweissfe.elements._hexa3dnodeordering import hexa8N, hexa20N
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
perQuadraturePoint results are exported on an additional part ``<elSet>_QP``, which splits each element
into one linear sub-element per quadrature point.

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
class EnsightPerQuadraturePointSchema:
    """The options of a single ``>>perQuadraturePoint`` block."""

    fieldOutput: str | None = schemaField(
        description="Name of an element result at all quadrature points of quadrilaterals or hexahedra, e.g. "
        "quadraturePoint=0:4 for CPE8R, 0:9 for CPE8, 0:8 for C3D20R, 0:27 for C3D20; written on a separate part, "
        "which splits each element into linear sub-elements, one per quadrature point",
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
    perQuadraturePoint: tuple[EnsightPerQuadraturePointSchema, ...] = subKeywordField(
        description="Quadrature point-based Ensight export on sub-elements.",
        schema=EnsightPerQuadraturePointSchema,
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

            # The connectivity table of the whole shape in ONE write (every element of a shape has the same number of nodes, so this is the
            # same byte stream as writing the elements one by one). ndarray.tofile on a file object flushes, duplicates the descriptor and
            # synchronizes the file position at every call, about 100 microseconds on a network file system: one call per element made the
            # time of the geometry write proportional to the number of elements, and it is repeated after every mesh change.
            writeCInt(f, np.asarray(list(elements.values()), np.int32) + 1)


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
            # get the node's index, inserting it with idx = counter if new
            idx = partNodes.setdefault(node, nodeCounter)
            elNodeIndices.append(idx)
            if idx == nodeCounter:
                # the node was just inserted, so increase the counter of inserted nodes
                nodeCounter += 1
        elementDict[elShape][element.elNumber] = elNodeIndices

    return EnsightUnstructuredPart(setName, partID, partNodes.keys(), elementDict)


# natural coordinates of the Hexa20 nodes (Abaqus/Marmot ordering): 8 corners, then the edge midpoints
_hexa20NodeXi = np.array(
    [
        [-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
        [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1],
        [0, -1, -1], [1, 0, -1], [0, 1, -1], [-1, 0, -1],
        [0, -1, 1], [1, 0, 1], [0, 1, 1], [-1, 0, 1],
        [-1, -1, 0], [1, -1, 0], [1, 1, 0], [-1, 1, 0],
    ]
)  # fmt: skip

# natural coordinates of the Quad8 nodes: the bottom face of the Hexa20
_quad8NodeXi = _hexa20NodeXi[[0, 1, 2, 3, 8, 9, 10, 11], :2]


def _quad4N(xi, eta):
    """Bilinear shape functions of the Quad4 element."""
    return 0.25 * np.array([(1 - xi) * (1 - eta), (1 + xi) * (1 - eta), (1 + xi) * (1 + eta), (1 - xi) * (1 + eta)])


def _quad8N(xi, eta):
    """Serendipity shape functions of the Quad8 element."""
    return 0.25 * np.array(
        [
            (1 - xi) * (1 - eta) * (-xi - eta - 1),
            (1 + xi) * (1 - eta) * (xi - eta - 1),
            (1 + xi) * (1 + eta) * (xi + eta - 1),
            (1 - xi) * (1 + eta) * (-xi + eta - 1),
            2 * (1 - xi**2) * (1 - eta),
            2 * (1 + xi) * (1 - eta**2),
            2 * (1 - xi**2) * (1 + eta),
            2 * (1 - xi) * (1 - eta**2),
        ]
    )


# ensight shape -> (natural coordinates of the nodes, shape functions, shape of the sub-elements)
_quadraturePointSubdivisions = {
    "quad4": (_quad8NodeXi[:4], _quad4N, "quad4"),
    "quad8": (_quad8NodeXi, _quad8N, "quad4"),
    "hexa8": (_hexa20NodeXi[:8], hexa8N, "hexa8"),
    "hexa20": (_hexa20NodeXi, hexa20N, "hexa8"),
}


@functools.lru_cache
def _gaussSubdivision(n: int) -> tuple[np.ndarray, np.ndarray]:
    """The points of the Gauss rule with n points, and the n + 1 natural coordinates which split
    [-1, 1] in proportion to its weights: each point lies in the interval of its weight."""

    points, weights = np.polynomial.legendre.leggauss(n)
    breakpoints = np.concatenate([[-1.0], -1.0 + np.cumsum(weights)])
    # symmetric, with the exact values -1, 1 and, for even n, 0 of the element nodes
    breakpoints = 0.5 * (breakpoints - breakpoints[::-1])
    breakpoints[[0, -1]] = -1.0, 1.0
    return points, breakpoints


def createQuadraturePointPartFromElementSet(
    setName: str, elementSet: list, basePart: EnsightUnstructuredPart, partID: int
) -> tuple[EnsightUnstructuredPart, csr_matrix]:
    """Splits each element of an element set into one linear sub-element per quadrature point of its
    Gauss rule with n points per direction (n x n for Quad4 and Quad8, n x n x n for Hexa8 and Hexa20);
    sub-element g of an element contains its quadrature point g, in whatever order the element
    numbers them. Along each natural direction the element is split in proportion to the Gauss
    weights, so a sub-element covers the share of the element its quadrature point integrates.

    The part's nodes are those nodes of the base part which are corners of sub-elements (in the
    order of the base part), followed by the added nodes, placed by the element's interpolation.
    Added nodes on edges and faces are shared by neighboring elements.

    Parameters
    ----------
    setName
        The name of the element set.
    elementSet
        The elements, all quadrilaterals or all hexahedra, with the same number of quadrature points.
    basePart
        The part of the element set.
    partID
        The id of this part.

    Returns
    -------
    tuple[EnsightUnstructuredPart, csr_matrix]
        The part, and the sparse matrix mapping per-node results of the base part to the nodes of the part.
    """

    baseNodes = list(basePart.nodes)
    baseIndices = {node: i for i, node in enumerate(baseNodes)}
    nBase = len(baseNodes)

    # the added nodes are indexed after the nodes of the base part, of which the unused ones are dropped at the end
    addedCoordinates = []
    rows, cols, weights = [], [], []
    sharedNodeIndices = {}
    elementTree = {}
    nQp = None
    for element in elementSet:
        if element.ensightType not in _quadraturePointSubdivisions:
            raise Exception(
                "Quadrature point output on sub-elements requires Quad4, Quad8, Hexa8 or Hexa20 elements, "
                "element {:} is {:}".format(element.elNumber, element.ensightType)
            )
        nodeXi, shapeFunctions, subShape = _quadraturePointSubdivisions[element.ensightType]
        dim = nodeXi.shape[1]
        nCorners = 2**dim
        cornerXi = nodeXi[:nCorners]
        elNodes = element.visualizationNodes
        nElNodes = len(elNodes)
        elIndices = [baseIndices[node] for node in elNodes]
        X = np.asarray([node.coordinates for node in elNodes])

        if nQp is None:
            nQp = element.getNumberOfQuadraturePoints()
        elif element.getNumberOfQuadraturePoints() != nQp:
            raise Exception(
                "Quadrature point output on sub-elements requires the same number of quadrature points in all "
                "elements, element set {:} holds elements with {:} and with {:}".format(
                    setName, nQp, element.getNumberOfQuadraturePoints()
                )
            )

        # each quadrature point must be a point of the Gauss rule with n points per direction: the one of its cell
        n = round(nQp ** (1 / dim))
        cellOfQp = None
        if n**dim == nQp:
            gaussPoints, breakpoints = _gaussSubdivision(n)
            cells = np.indices(dim * (n,)).reshape(dim, -1).T
            qpExpected = np.asarray(shapeFunctions(*gaussPoints[cells].T)).T @ X
            qpCoordinates = np.reshape(element.getCoordinatesAtQuadraturePoints(), (nQp, -1))[:, :dim]
            distances = np.linalg.norm(qpCoordinates[:, None, :] - qpExpected[None, :, :], axis=2)
            nearest = distances.argmin(axis=1)
            if distances.min(axis=1).max() <= 1e-8 * np.ptp(X, axis=0).max() and len(set(nearest)) == nQp:
                cellOfQp = cells[nearest]
        if cellOfQp is None:
            raise Exception(
                "Quadrature point output on sub-elements requires a Gauss rule with the same number of quadrature "
                "points in each direction, element {:} with {:} quadrature points does not use one".format(
                    element.elNumber, nQp
                )
            )

        # node indices of the lattice of the n + 1 breakpoints per direction; the element's own nodes on it first
        lattice = np.full(dim * (n + 1,), -1, dtype=int)
        latticeIndex = {-1: 0, 1: n} | ({0: n // 2} if n % 2 == 0 else {})
        for i, xi in enumerate(nodeXi):
            if all(c in latticeIndex for c in xi):
                lattice[tuple(latticeIndex[c] for c in xi)] = elIndices[i]
        cornerIndices = (cornerXi + 1) // 2 * n
        for index in np.argwhere(lattice < 0):
            # nodes on edges and faces are shared with the neighbors, whose natural directions differ:
            # identify them by their lattice distances to the corner nodes of the edge or face
            key = None
            onBoundary = (index == 0) | (index == n)
            if onBoundary.any():
                key = frozenset(
                    (elNodes[c], tuple(sorted(int(d) for d in np.abs(index - cornerIndices[c])[~onBoundary])))
                    for c in range(nCorners)
                    if np.all(cornerIndices[c][onBoundary] == index[onBoundary])
                )
                if key in sharedNodeIndices:
                    lattice[tuple(index)] = sharedNodeIndices[key]
                    continue
            N = np.asarray(shapeFunctions(*breakpoints[index]))
            idx = nBase + len(addedCoordinates)
            addedCoordinates.append(N @ X)
            rows += [len(addedCoordinates) - 1] * nElNodes
            cols += elIndices
            weights += list(N)
            lattice[tuple(index)] = idx
            if key is not None:
                sharedNodeIndices[key] = idx

        # the sub-element of a quadrature point is its cell of the lattice, its nodes ordered as the corners
        subElements = elementTree.setdefault(subShape, {})
        for g, cell in enumerate(cellOfQp):
            subElements[nQp * element.elNumber + g] = [lattice[tuple(cell + (t + 1) // 2)] for t in cornerXi]

    if len(elementTree) > 1:
        raise Exception(
            "Quadrature point output on sub-elements requires an element set of quadrilaterals only or of "
            "hexahedra only, element set {:} holds both".format(setName)
        )

    # keep the nodes of the base part which the sub-elements use, e.g. not the edge midpoints for an odd n
    nAdded = len(addedCoordinates)
    used = np.zeros(nBase + nAdded, dtype=bool)
    used[nBase:] = True
    for subElements in elementTree.values():
        used[np.asarray(list(subElements.values()), dtype=int)] = True
    usedBase = np.flatnonzero(used[:nBase])
    newIndices = np.cumsum(used) - 1
    elementTree = {
        shape: {label: newIndices[nodeIndices] for label, nodeIndices in subElements.items()}
        for shape, subElements in elementTree.items()
    }

    nextLabel = int(basePart.nodeLabels.max()) + 1 if nBase else 1
    nodes = [baseNodes[i] for i in usedBase] + [Node(nextLabel + i, x) for i, x in enumerate(addedCoordinates)]
    mapping = csr_matrix(
        (
            np.concatenate([np.ones(len(usedBase)), weights]),
            (
                np.concatenate([np.arange(len(usedBase)), len(usedBase) + np.asarray(rows, dtype=int)]),
                np.concatenate([usedBase, np.asarray(cols, dtype=int)]),
            ),
        ),
        shape=(len(nodes), nBase),
    )
    return EnsightUnstructuredPart(setName + "_QP", partID, nodes, elementTree), mapping


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

    # the node list must follow getVisualizationNodes()'s order, in which
    # RigidBodyFieldOutput.getVisualizationField() reports per-node results.
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
        perQuadraturePointDefs = configuration.perQuadraturePoint
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
        # base part -> (sub-element part, mapping of per-node results to its nodes)
        self._quadraturePointParts = {}

        self._transientPerNodeVariableJobs = defaultdict(list)
        self._transientPerElementVariableJobs = defaultdict(list)
        self._transientPerQuadraturePointVariableJobs = {}

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

        # Repeated `>>configuration` blocks: last wins, but `configSetName` carries over from an earlier
        # block if the last one names neither nSet nor elSet.
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
        self._perQuadraturePointDefs = perQuadraturePointDefs
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

        for definition in self._perQuadraturePointDefs:
            fieldOutput = self._fieldOutputController.fieldOutputs[definition.fieldOutput]
            name = fieldOutput.name.replace(" ", "_")
            self.createPerQuadraturePointOutput(fieldOutput, name)

    def _rebuildForMeshChange(self):
        """Rebuild geometry parts + variable jobs after an AMR mesh change, so both stay consistent
        with the current (refined) mesh."""
        self.elSetToEnsightPartMappings = {}
        self.nSetToEnsightPartMappings = {}
        self.rigidBodyToEnsightPartMappings = {}
        self._quadraturePointParts = {}
        self._transientPerNodeVariableJobs = defaultdict(list)
        self._transientPerElementVariableJobs = defaultdict(list)
        self._transientPerQuadraturePointVariableJobs = {}
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

    def createPerQuadraturePointOutput(self, fieldOutput: ElementFieldOutput, name: str = None):
        """Create a per quadrature point output job, written per sub-element on a separate part
        of the element set (created on first use). Per-node results of the element set are also
        written on that part, interpolated to its added nodes.

        Parameters
        ----------
        fieldOutput
            The field output to export: a result at all quadrature points of quadrilaterals or
            hexahedra integrated by a Gauss rule.
        name
            The name of the output. If not specified, the name of the field output is taken.
        """

        if not name:
            name = fieldOutput.name
        if name in self._transientPerElementVariableJobs or name in self._transientPerQuadraturePointVariableJobs:
            raise Exception("Ensight variable {:} is defined more than once".format(name))

        if not isinstance(fieldOutput, ElementFieldOutput):
            raise Exception("Quadrature point output {:} requires an element field output".format(name))

        elSet = fieldOutput.associatedSet
        nElements = len(elSet)
        if not nElements:
            raise Exception("Quadrature point output {:} requires a non-empty element set".format(name))

        basePart = self.elSetToEnsightPartMappings[elSet.name]
        if basePart not in self._quadraturePointParts:
            partID = max(part.partNumber for part in self.geometryParts) + 1
            subPart, mapping = createQuadraturePointPartFromElementSet(elSet.name, elSet, basePart, partID)
            self.geometryParts.append(subPart)
            self._quadraturePointParts[basePart] = (subPart, mapping)
        subPart = self._quadraturePointParts[basePart][0]

        # the sub-elements follow the quadrature points 0, 1, ... of each element
        (subShape,) = subPart.elementTree.keys()
        nQp = len(subPart.elementTree[subShape]) // nElements
        if list(fieldOutput.quadraturePoints) != list(range(nQp)):
            raise Exception(
                "Quadrature point output {:} requires an element field output with quadraturePoint=0:{:}".format(
                    name, nQp
                )
            )

        # an f(x) of the field output may reduce the components, but has to keep one entry per quadrature point
        result = np.asarray(fieldOutput.getLastResult())
        if result.shape[:2] != (nElements, nQp):
            raise Exception(
                "Variable {:} result shape {:} does not start with {:} elements and {:} quadrature points".format(
                    name, result.shape, nElements, nQp
                )
            )

        varSize = result.size // (nQp * nElements)
        if self.model.domainSize == 2 and varSize == 2:
            varSize = 3

        self._transientPerQuadraturePointVariableJobs[name] = {
            "fieldOutput": fieldOutput,
            "part": subPart,
            "subShape": subShape,
            "nSubElements": nQp * nElements,
            "varSize": varSize,
        }

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
        # step overrides arrive via applyOptionsOverride
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

        # No time has passed (e.g. a resumed run's priming zero increment): a second frame would break the
        # strictly increasing .case time values. The initial timeAtLastOutput of -1e16 keeps a cold start's first frame.
        if timeSinceLastOutput <= 0.0:
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
                if perNodeVariableJob["part"] in self._quadraturePointParts:
                    subPart, mapping = self._quadraturePointParts[perNodeVariableJob["part"]]
                    resultsByParts[subPart.partNumber] = ("coordinates", mapping @ result)
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

        # one value per sub-element: element by element, quadrature point by quadrature point
        for resultName, job in self._transientPerQuadraturePointVariableJobs.items():
            result = np.reshape(job["fieldOutput"].getLastResult(), (job["nSubElements"], -1))

            if self.model.domainSize == 2 and result.shape[1] == 2:
                result = self._make2DVector3D(result)

            resultsByParts = {job["part"].partNumber: {job["subShape"]: result}}
            enSightVariable = EnsightPerElementVariable(resultName, resultsByParts, job["varSize"])
            self.ensightCase.writeVariableTrendChunk(enSightVariable, self.transientVariableTAndFSetNumber)
            del enSightVariable

        # register the current time for variables
        self.ensightCase.setCurrentTime(self.transientVariableTAndFSetNumber, model.time)

        # Rewrite the .case every step, so it matches the data on disk even if the job aborts before finalizeJob().
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

    def getRestartData(self) -> dict[str, np.ndarray]:
        """This export's transient-sequence bookkeeping: each Ensight time/file set's history of
        already-written time values (flattened CSR-style, since sets can have different lengths),
        which geometry trends have ever been written (name -> time/file set number -- unlike
        variable trends, which get (re-)registered on every write regardless of mesh change and so
        self-heal on resume, a geometry trend is only (re-)registered when the mesh actually
        changes; a resumed run whose mesh happens to be already stable would otherwise never
        re-register it, leaving the ``.case`` file's ``GEOMETRY`` section without its ``model:``
        line even though the geometry file itself exists on disk from before the resume), plus the
        mesh signature and ``timeAtLastOutput`` used to decide whether to write a fresh geometry
        chunk / throttle output. File numbering (``writeGeometryTrendChunk``/
        ``writeVariableTrendChunk`` derive the next chunk's index from ``len(timeValues)``) and the
        ``.case`` file's own declared step list come directly from ``timeAndFileSets`` -- restoring
        it is both necessary and sufficient for continuing the same *sequence*, but the geometry
        trend registration above is a separate, independently-necessary piece for the ``.case``
        file to still reference the geometry at all.

        Empty if nothing has been written yet (nothing to restore).
        """

        timeAndFileSets = self.ensightCase.timeAndFileSets
        if not timeAndFileSets:
            return {}

        setNumbers = sorted(timeAndFileSets)
        sizes = [len(timeAndFileSets[n].timeValues) for n in setNumbers]
        flatTimeValues = [v for n in setNumbers for v in timeAndFileSets[n].timeValues]
        meshSignature = self._meshSignature if self._meshSignature is not None else (-1, -1)

        geometryTrends = self.ensightCase.geometryTrends
        geometryTrendNames = list(geometryTrends.keys())
        geometryTrendSetNumbers = list(geometryTrends.values())

        return {
            "setNumbers": np.array(setNumbers, dtype=int),
            "timeValueSizes": np.array(sizes, dtype=int),
            "timeValues": np.array(flatTimeValues, dtype=float),
            "meshSignature": np.array(meshSignature, dtype=int),
            "timeAtLastOutput": np.array([self.timeAtLastOutput]),
            "geometryTrendNames": np.array(geometryTrendNames, dtype=h5py.string_dtype(encoding="utf-8")),
            "geometryTrendSetNumbers": np.array(geometryTrendSetNumbers, dtype=int),
        }

    def setRestartData(self, data: dict[str, np.ndarray]):
        """Restore this export's transient-sequence bookkeeping from a restart checkpoint written
        by :meth:`getRestartData`, so the next chunk written continues the existing sequence
        (correct file numbering, correct ``.case`` step list) instead of starting a fresh one."""

        if not data:
            return

        offset = 0
        for setNumber, size in zip(data["setNumbers"], data["timeValueSizes"]):
            size = int(size)
            timeValues = list(data["timeValues"][offset : offset + size])
            offset += size
            self.ensightCase.timeAndFileSets[int(setNumber)] = EnsightTimeSet(
                int(setNumber), "no description", 0, 1, timeValues
            )

        for name, setNumber in zip(data["geometryTrendNames"], data["geometryTrendSetNumbers"]):
            name = name.decode("utf-8") if isinstance(name, bytes) else str(name)
            self.ensightCase.geometryTrends[name] = int(setNumber)

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
