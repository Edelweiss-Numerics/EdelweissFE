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
#  Daniel Reitmair daniel.reitmair@uibk.ac.at
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

from dataclasses import dataclass
from operator import attrgetter

import numpy as np

from edelweissfe.config.elementlibrary import getElementClass
from edelweissfe.generators.base.generatorbase import GeneratorBase
from edelweissfe.journal.journal import Journal
from edelweissfe.models.femodel import FEModel
from edelweissfe.points.node import Node
from edelweissfe.sets.elementset import ElementSet
from edelweissfe.sets.nodeset import NodeSet
from edelweissfe.surfaces.entitybasedsurface import EntityBasedSurface
from edelweissfe.utils.inputlanguage import InputLanguage, Module
from edelweissfe.utils.schema import schemaField

module = Module("pipegen", "A structured hex mesh generator for pipe geometries.")

inputLanguage = InputLanguage()

keyword = "modelGenerator"
if keyword in inputLanguage:
    inputLanguage[keyword].addModule(module)

module.addOptionalArg("x0", "Origin along the x axis.", float, 0.0)
module.addOptionalArg("y0", "Origin along the y axis.", float, 0.0)
module.addOptionalArg("z0", "Origin along the z axis.", float, 0.0)

module.addOptionalArg("Ro(y)", "Outer radius of the pipe as a function of height.", str, "1.0")
module.addOptionalArg("Ri(y)", "Inner radius of the pipe as a function of height.", str, "1.0")

module.addOptionalArg("lT", "Thickness of the pipe.", float, 0.5)
module.addOptionalArg("lY", "Height of the pipe.", float, 1.0)

module.addOptionalArg("phi", "Total angle for the pipe sector.", float, 360.0)

module.addOptionalArg("nT", "Number of elements along radial direction.", int, 1)
module.addOptionalArg("nC", "Number of elements along circumferential direction.", int, 4)
module.addOptionalArg("nY", "Number of elements along longitudinal direction.", int, 4)

module.addOptionalArg("exG", "Flag to place nodes on exact geometry.", bool, True)

module.addRequiredArg("elType", "Element type.", str)
module.addOptionalArg("elProvider", "Element provider.", str, None)

documentation = [module]


@dataclass(frozen=True)
class PipegenSchema:
    """L2: the options this generator accepts, owned by this module and never mutated from
    outside it.

    Mirrors the ``module.addOptionalArg``/``module.addRequiredArg(...)`` declarations above
    one-for-one. The two declarations coexist while the migration is in progress; the ``Module``
    one goes away with the ``InputLanguage`` singleton in P5.

    ``Ro(y)``/``Ri(y)`` are not valid Python identifiers, hence the ``optionName`` indirection --
    see ``optionName`` on :func:`~edelweissfe.utils.schema.schemaField`. ``elType`` is declared
    ``required=True`` explicitly, mirroring ``addRequiredArg`` above, but is still given a
    ``default=None`` so the schema remains constructible for the L1 constructor's default argument.
    """

    x0: float = schemaField(description="Origin along the x axis.", dtype=float, default=0.0)
    y0: float = schemaField(description="Origin along the y axis.", dtype=float, default=0.0)
    z0: float = schemaField(description="Origin along the z axis.", dtype=float, default=0.0)
    Ro_y: str = schemaField(
        description="Outer radius of the pipe as a function of height.", dtype=str, default="1.0", optionName="Ro(y)"
    )
    Ri_y: str = schemaField(
        description="Inner radius of the pipe as a function of height.", dtype=str, default="1.0", optionName="Ri(y)"
    )
    lT: float = schemaField(description="Thickness of the pipe.", dtype=float, default=0.5)
    lY: float = schemaField(description="Height of the pipe.", dtype=float, default=1.0)
    phi: float = schemaField(description="Total angle for the pipe sector.", dtype=float, default=360.0)
    nT: int = schemaField(description="Number of elements along radial direction.", dtype=int, default=1)
    nC: int = schemaField(description="Number of elements along circumferential direction.", dtype=int, default=4)
    nY: int = schemaField(description="Number of elements along longitudinal direction.", dtype=int, default=4)
    exG: bool = schemaField(description="Flag to place nodes on exact geometry.", dtype=bool, default=True)
    elType: str | None = schemaField(description="Element type.", dtype=str, default=None, required=True)
    elProvider: str | None = schemaField(description="Element provider.", dtype=str, default=None)


class Generator(GeneratorBase):
    """A structured hex mesh generator for pipe geometries."""

    #: L2 schema declared for the L3 registry, per OptionSchemaProvider.
    schema = PipegenSchema

    def __init__(self, name: str, model: FEModel, journal: Journal, *, configuration: PipegenSchema = PipegenSchema()):
        """L1: constructible standalone, with no ``InputLanguage``/``Module``/parser involvement.
        Populates ``model`` directly; construction *is* the generation.

        Parameters
        ----------
        name
            The name of this generator instance, used as the prefix for the generated sets.
        model
            The model tree to populate. Mutated in place.
        journal
            Unused.
        configuration
            The options this generator accepts; ``elType`` is still required, see
            :class:`PipegenSchema`.
        """
        x0 = configuration.x0
        y0 = configuration.y0
        z0 = configuration.z0

        Roy = configuration.Ro_y
        Riy = configuration.Ri_y

        lT = configuration.lT
        lY = configuration.lY

        phi = configuration.phi

        nT = configuration.nT
        nC = configuration.nC
        nY = configuration.nY

        exG = configuration.exG

        if np.abs(phi) > 360:
            raise Exception("The angle can't be higher than 360° or lower than -360°.")

        elTypeName = configuration.elType
        elProvider = configuration.elProvider

        elType = getElementClass(elTypeName, elProvider)

        testEl = elType(elTypeName, 0)

        if nY < 4 and testEl.nNodes == 20:
            print(
                "Information: Using a hexahedral element with 20 nodes and a non constant function"
                + " for the radius with low element count over height can lead to display errors in paraview."
            )

        extraNode = 0 if phi != 360 else 1
        if testEl.nNodes == 8:
            nNodesT = nT + 1
            nNodesY = nY + 1
            nNodesC = nC + 1
        elif testEl.nNodes == 20:
            nNodesT = 2 * nT + 1
            nNodesY = 2 * nY + 1
            nNodesC = 2 * nC + 1
        else:
            return

        # coordinates of cylindrical layers
        cLayers = np.linspace(0, np.pi * phi / 180.0, nNodesC)
        yLayers = np.linspace(y0, y0 + lY, nNodesY)

        if Riy == "1.0":  # use outer
            Ry = Roy
            useouter = True
        elif Roy == "1.0" and Riy != "1.0":  # use inner
            Ry = Riy
            useouter = False
        else:
            raise Exception(
                "Only the outer or the inner radius can be defined as a function. Please use 'lT' option instead!"
            )

        def Ry_(y):
            return eval(Ry)

        try:
            Ry_ = np.vectorize(Ry_)
            rLayers = Ry_(yLayers - y0)
        except NameError:
            raise Exception("The radius function can only depend on the height y.")
        if (np.any(rLayers - lT < 1e-5) and useouter) or (np.any(rLayers < 1e-5) and not useouter):
            raise Exception("The radius must be > 1e-5 at every point.")

        nodes = []
        currentNodeLabel = 1
        if model.nodes:
            currentNodeLabel += max(model.nodes.keys())
        for iy in range(nNodesY):
            if useouter:
                tLayers = np.linspace(Ry_(yLayers[iy] - y0) - lT, Ry_(yLayers[iy] - y0), nNodesT)
            else:
                tLayers = np.linspace(Ry_(yLayers[iy] - y0), Ry_(yLayers[iy] - y0) + lT, nNodesT)
            tLayersRed = tLayers * [np.cos((np.pi * phi / 180.0) / (nNodesC - extraNode)) if not exG else 1]
            for it in range(nNodesT):
                for ic in range(nNodesC - extraNode):
                    # use reduced radius to keep element planar if exG is True
                    if testEl.nNodes == 20 and (ic % 2 != 0):
                        node = Node(
                            currentNodeLabel,
                            np.array(
                                [
                                    x0 + tLayersRed[it] * np.sin(cLayers[ic]),
                                    yLayers[iy],
                                    z0 + tLayersRed[it] * np.cos(cLayers[ic]),
                                ]
                            ),
                        )
                    else:
                        node = Node(
                            currentNodeLabel,
                            np.array(
                                [
                                    x0 + tLayers[it] * np.sin(cLayers[ic]),
                                    yLayers[iy],
                                    z0 + tLayers[it] * np.cos(cLayers[ic]),
                                ]
                            ),
                        )
                    nodes.append(node)
                    # only add node to model if it will be part of an element
                    if testEl.nNodes == 8 or (testEl.nNodes == 20 and sum(np.mod([it, iy, ic], 2)) < 2):
                        model.nodes[currentNodeLabel] = node
                        currentNodeLabel += 1

        # # 3d plot of nodes; for debugging
        # import os
        # def plotNodeList(nodeList):
        #     nodeListFile = "nodes.dat"
        #     with open(nodeListFile, "w+") as f:
        #         for node in nodeList:
        #             coords = node.coordinates
        #             line = "{:5}, {:12}, {:12}, {:12}\n".format(node.label-1, coords[0], coords[1], coords[2])
        #             f.write(line)

        #     cmd = [
        #         "gnuplot",
        #         "plotConfig",
        #         "-p",
        #         "-e",
        #         '\' filename="{}"; splot filename using 4:2:3:(sprintf("(%i)", $1)) with labels \''.format(nodeListFile),
        #     ]
        #     os.system(" ".join(cmd))

        # plotNodeList(nodes)

        elements = []
        nNodesC -= extraNode
        currentElementLabel = 1
        if model.elements:
            currentElementLabel += max(model.elements.keys())
        for it in range(nT):
            for iy in range(nY):
                for ic in range(nC):
                    if phi != 360 or ic != nC - 1:
                        fullIndex = 0
                    else:  # for the last element of a full circle
                        fullIndex = nC
                    if testEl.nNodes == 8:
                        nodeList = [
                            nodes[0 + iy * (nNodesT * nNodesC) + it * nNodesC + ic],
                            nodes[1 + iy * (nNodesT * nNodesC) + it * nNodesC + ic - fullIndex],
                            nodes[1 + (1 + iy) * (nNodesT * nNodesC) + it * nNodesC + ic - fullIndex],
                            nodes[0 + (1 + iy) * (nNodesT * nNodesC) + it * nNodesC + ic],
                            nodes[0 + iy * (nNodesT * nNodesC) + (1 + it) * nNodesC + ic],
                            nodes[1 + iy * (nNodesT * nNodesC) + (1 + it) * nNodesC + ic - fullIndex],
                            nodes[1 + (1 + iy) * (nNodesT * nNodesC) + (1 + it) * nNodesC + ic - fullIndex],
                            nodes[0 + (1 + iy) * (nNodesT * nNodesC) + (1 + it) * nNodesC + ic],
                        ]
                    elif testEl.nNodes == 20:
                        nodeList = [
                            nodes[0 + 2 * iy * (nNodesT * nNodesC) + 2 * it * nNodesC + 2 * ic],
                            nodes[2 + 2 * iy * (nNodesT * nNodesC) + 2 * it * nNodesC + 2 * ic - 2 * fullIndex],
                            nodes[2 + 2 * (1 + iy) * (nNodesT * nNodesC) + 2 * it * nNodesC + 2 * ic - 2 * fullIndex],
                            nodes[0 + 2 * (1 + iy) * (nNodesT * nNodesC) + 2 * it * nNodesC + 2 * ic],
                            nodes[0 + 2 * iy * (nNodesT * nNodesC) + 2 * (1 + it) * nNodesC + 2 * ic],
                            nodes[2 + 2 * iy * (nNodesT * nNodesC) + 2 * (1 + it) * nNodesC + 2 * ic - 2 * fullIndex],
                            nodes[
                                2 + 2 * (1 + iy) * (nNodesT * nNodesC) + 2 * (1 + it) * nNodesC + 2 * ic - 2 * fullIndex
                            ],
                            nodes[0 + 2 * (1 + iy) * (nNodesT * nNodesC) + 2 * (1 + it) * nNodesC + 2 * ic],
                            nodes[1 + 2 * iy * (nNodesT * nNodesC) + 2 * it * nNodesC + 2 * ic],
                            nodes[2 + (1 + 2 * iy) * (nNodesT * nNodesC) + 2 * it * nNodesC + 2 * ic - 2 * fullIndex],
                            nodes[1 + 2 * (1 + iy) * (nNodesT * nNodesC) + 2 * it * nNodesC + 2 * ic],
                            nodes[0 + (1 + 2 * iy) * (nNodesT * nNodesC) + 2 * it * nNodesC + 2 * ic],
                            nodes[1 + 2 * iy * (nNodesT * nNodesC) + 2 * (1 + it) * nNodesC + 2 * ic],
                            nodes[
                                2 + (1 + 2 * iy) * (nNodesT * nNodesC) + 2 * (1 + it) * nNodesC + 2 * ic - 2 * fullIndex
                            ],
                            nodes[1 + 2 * (1 + iy) * (nNodesT * nNodesC) + 2 * (1 + it) * nNodesC + 2 * ic],
                            nodes[0 + (1 + 2 * iy) * (nNodesT * nNodesC) + 2 * (1 + it) * nNodesC + 2 * ic],
                            nodes[0 + 2 * iy * (nNodesT * nNodesC) + (1 + 2 * it) * nNodesC + 2 * ic],
                            nodes[2 + 2 * iy * (nNodesT * nNodesC) + (1 + 2 * it) * nNodesC + 2 * ic - 2 * fullIndex],
                            nodes[
                                2 + 2 * (1 + iy) * (nNodesT * nNodesC) + (1 + 2 * it) * nNodesC + 2 * ic - 2 * fullIndex
                            ],
                            nodes[0 + 2 * (1 + iy) * (nNodesT * nNodesC) + (1 + 2 * it) * nNodesC + 2 * ic],
                        ]
                    else:
                        return

                    # plotNodeList( nodeList )

                    # newEl = elType(options["elType"], nodeList, currentElementLabel)
                    newEl = elType(elTypeName, currentElementLabel)
                    newEl.setNodes(nodeList)

                    elements.append(newEl)
                    model.elements[currentElementLabel] = newEl

                    # for i, node in enumerate(newEl.nodes):
                    #     node.fields.update([(f, True) for f in newEl.fields[i]])

                    currentElementLabel += 1

        # fmt: on
        # model.initializeNodeFields()
        model._populateNodeFieldVariablesFromElements()

        nG = np.asarray(nodes).reshape(nNodesY, nNodesT, nNodesC)

        # nodesets:
        nodeSets = []

        # 6 faces
        getFields = np.vectorize(attrgetter("fields"))
        getLength = np.vectorize(len)
        filterGrid = getLength(getFields(nG)) > 0

        def getFilteredNodes(s):
            return nG[s][filterGrid[s]]

        nodeSets.append(NodeSet("{:}_outer".format(name), getFilteredNodes(np.s_[:, -1, :])))
        nodeSets.append(NodeSet("{:}_inner".format(name), getFilteredNodes(np.s_[:, 0, :])))
        nodeSets.append(NodeSet("{:}_top".format(name), getFilteredNodes(np.s_[-1, :, :])))
        nodeSets.append(NodeSet("{:}_bottom".format(name), getFilteredNodes(np.s_[0, :, :])))
        nodeSets.append(NodeSet("{:}_end".format(name), getFilteredNodes(np.s_[:, :, -1])))
        nodeSets.append(NodeSet("{:}_begin".format(name), getFilteredNodes(np.s_[:, :, 0])))

        # 12 edges
        nodeSets.append(NodeSet("{:}_beginTop".format(name), getFilteredNodes(np.s_[-1, :, 0])))
        nodeSets.append(NodeSet("{:}_endTop".format(name), getFilteredNodes(np.s_[-1, :, -1])))
        nodeSets.append(NodeSet("{:}_beginBottom".format(name), getFilteredNodes(np.s_[0, :, 0])))
        nodeSets.append(NodeSet("{:}_endBottom".format(name), getFilteredNodes(np.s_[0, :, -1])))

        nodeSets.append(NodeSet("{:}_innerEnd".format(name), getFilteredNodes(np.s_[:, 0, -1])))
        nodeSets.append(NodeSet("{:}_innerBegin".format(name), getFilteredNodes(np.s_[:, 0, 0])))
        nodeSets.append(NodeSet("{:}_innerTop".format(name), getFilteredNodes(np.s_[-1, 0, :])))
        nodeSets.append(NodeSet("{:}_innerBottom".format(name), getFilteredNodes(np.s_[0, 0, :])))

        nodeSets.append(NodeSet("{:}_outerEnd".format(name), getFilteredNodes(np.s_[:, -1, -1])))
        nodeSets.append(NodeSet("{:}_outerBegin".format(name), getFilteredNodes(np.s_[:, -1, 0])))
        nodeSets.append(NodeSet("{:}_outerTop".format(name), getFilteredNodes(np.s_[-1, -1, :])))
        nodeSets.append(NodeSet("{:}_outerBottom".format(name), getFilteredNodes(np.s_[0, -1, :])))

        nodeSets.append(NodeSet("{:}_centerY".format(name), getFilteredNodes(np.s_[int(nNodesY / 2), :, :])))
        nodeSets.append(NodeSet("{:}_centerT".format(name), getFilteredNodes(np.s_[:, int(nNodesT / 2), :])))
        nodeSets.append(NodeSet("{:}_centerC".format(name), getFilteredNodes(np.s_[:, :, int(nNodesC / 2)])))

        # 8 vertices
        nodeSets.append(NodeSet("{:}_topEndInner".format(name), nG[-1, 0, -1]))
        nodeSets.append(NodeSet("{:}_topBeginInner".format(name), nG[-1, 0, 0]))
        nodeSets.append(NodeSet("{:}_bottomEndInner".format(name), nG[0, 0, -1]))
        nodeSets.append(NodeSet("{:}_bottomBeginInner".format(name), nG[0, 0, 0]))

        nodeSets.append(NodeSet("{:}_topEndOuter".format(name), nG[-1, -1, -1]))
        nodeSets.append(NodeSet("{:}_topBeginOuter".format(name), nG[-1, -1, 0]))
        nodeSets.append(NodeSet("{:}_bottomEndOuter".format(name), nG[0, -1, -1]))
        nodeSets.append(NodeSet("{:}_bottomBeginOuter".format(name), nG[0, -1, 0]))

        for nodeSet in nodeSets:
            model.nodeSets[nodeSet.name] = nodeSet

        # element sets
        elementSets = []
        elementSets.append(ElementSet("{:}_all".format(name), elements))

        elGrid = np.asarray(elements).reshape(nT, nY, nC)
        elementSets.append(ElementSet("{:}_bottom".format(name), np.ravel(elGrid[:, 0, :])))
        elementSets.append(ElementSet("{:}_top".format(name), np.ravel(elGrid[:, -1, :])))
        elementSets.append(ElementSet("{:}_outer".format(name), np.ravel(elGrid[-1, :, :])))
        elementSets.append(ElementSet("{:}_inner".format(name), np.ravel(elGrid[0, :, :])))
        elementSets.append(ElementSet("{:}_end".format(name), np.ravel(elGrid[:, :, -1])))
        elementSets.append(ElementSet("{:}_begin".format(name), np.ravel(elGrid[:, :, 0])))

        elementSets.append(ElementSet("{:}_centerSliceY".format(name), np.ravel(elGrid[int(nT / 2), :, :])))
        elementSets.append(ElementSet("{:}_centerSliceT".format(name), np.ravel(elGrid[:, int(nY / 2), :])))
        elementSets.append(ElementSet("{:}_centerSliceC".format(name), np.ravel(elGrid[:, :, int(nC / 2)])))

        nShearBand = min(nT, nY)
        if nShearBand > 3:
            shearBand = []
            for i1 in range(nShearBand):
                shearBand.extend(
                    np.ravel(
                        elGrid[
                            int(nT / 2 + i1 - nShearBand / 2),
                            int(nY / 2 + i1 - nShearBand / 2),
                            0:nC,
                        ]
                    )
                )
            elementSets.append(ElementSet("{:}_shearBandInnerToOuter".format(name), [e for e in shearBand]))
            elementSets.append(
                ElementSet(
                    "{:}_shearBandCenterInnerToOuter".format(name),
                    [e for e in shearBand[(int(nShearBand / 2) - 1) * nC : (int(nShearBand / 2) + 2) * nC]],
                )
            )

        for elementSet in elementSets:
            model.elementSets[elementSet.name] = elementSet

        # surfaces
        surfaceName = "{:}_bottom".format(name)
        model.surfaces[surfaceName] = EntityBasedSurface(surfaceName, {1: model.elementSets[surfaceName]})
        surfaceName = "{:}_top".format(name)
        model.surfaces[surfaceName] = EntityBasedSurface(surfaceName, {2: model.elementSets[surfaceName]})

        surfaceName = "{:}_outer".format(name)
        model.surfaces[surfaceName] = EntityBasedSurface(surfaceName, {5: model.elementSets[surfaceName]})
        surfaceName = "{:}_inner".format(name)
        model.surfaces[surfaceName] = EntityBasedSurface(surfaceName, {3: model.elementSets[surfaceName]})

        surfaceName = "{:}_end".format(name)
        model.surfaces[surfaceName] = EntityBasedSurface(surfaceName, {4: model.elementSets[surfaceName]})
        surfaceName = "{:}_begin".format(name)
        model.surfaces[surfaceName] = EntityBasedSurface(surfaceName, {6: model.elementSets[surfaceName]})
