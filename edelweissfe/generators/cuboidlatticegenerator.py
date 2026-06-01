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
#  Alexander Dummer alexander.dummer@uibk.ac.at
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
A mesh generator for generating cuboid lattice structure.
The following unit cell is generted and then replicated in x, y, and z direction:

.. code-block:: console

               +---------------+
              /  +---------+  /|
             /  /         /  / |
            /  /         /  /  |
           /  +---------+  /  +|
          +---------------+  /||
          |               | + ||
          |  +---------+  | | ||
          |  |         |  | | +|
          |  |         |  | |/ +
          |  |         |  | + /
          |  |         |  |  /
          |  +---------+  | /
          |               |/
          +---------------+

After generating the unit cell mesh the microstructure generator is used
to replicate the unit cell mesh in x, y, and z direction.

Example
-------

Generate meshes on the fly using the following syntax:

.. code-block:: edelweiss

    *job, name=job, domain=2d, solver=NIST

    *modelGenerator, generator=cuboidlatticegenerator, name=gen
        lX=4
        lY=8
        lZ=2
        nEleX=8
        nEleY=16
        nEleZ=4
        nEleStrutX=2
        nEleStrutY=4
        nEleStrutZ=1
        nX=3
        nY=4
        nZ=2
"""


import copy

import numpy as np

from edelweissfe.config.elementlibrary import getElementClass
from edelweissfe.generators.boxgen import generateModelData as generateBoxMesh
from edelweissfe.generators.microstructuregenerator import replicateMesh
from edelweissfe.models.femodel import FEModel
from edelweissfe.sets.elementset import ElementSet
from edelweissfe.sets.nodeset import NodeSet
from edelweissfe.utils.caseinsensitivedict import CaseInsensitiveDict
from edelweissfe.utils.inputlanguage import InputLanguage, Module
from edelweissfe.utils.misc import (
    caseInsensitiveKwargsChecker,
    castKwargsValuesAndAddDefaults,
)

module = Module("cuboidlatticegenerator", "A mesh generator for generating cuboid lattice structure.")

inputLanguage = InputLanguage()

keyword = "modelGenerator"
if keyword in inputLanguage:
    inputLanguage[keyword].addModule(module)

# module.addOptionalArg("x0", "Origin along the x axis.", float, 0.0)
# module.addOptionalArg("y0", "Origin along the y axis.", float, 0.0)
# module.addOptionalArg("z0", "Origin along the z axis.", float, 0.0)

module.addOptionalArg("lX", "Length of the body along the x axis.", float, 1.0)
module.addOptionalArg("lY", "Length of the body along the y axis.", float, 1.0)
module.addOptionalArg("lZ", "Length of the body along the z axis.", float, 1.0)

module.addOptionalArg("nEleX", "Number of elements along the x axis.", int, 1)
module.addOptionalArg("nEleY", "Number of elements along the y axis.", int, 1)
module.addOptionalArg("nEleZ", "Number of elements along the z axis.", int, 1)

module.addOptionalArg("nEleStrutX", "Number of in struts along the x axis.", int, 1)
module.addOptionalArg("nEleStrutY", "Number of in struts along the y axis.", int, 1)
module.addOptionalArg("nEleStrutZ", "Number of in struts along the z axis.", int, 1)

module.addOptionalArg("nX", "Number of replications along the x axis.", int, 1)
module.addOptionalArg("nY", "Number of replications along the y axis.", int, 1)
module.addOptionalArg("nZ", "Number of replications along the z axis.", int, 1)

module.addRequiredArg("elType", "Element type.", str)
module.addOptionalArg("elProvider", "Element provider.", str, None)

documentation = [module]


@caseInsensitiveKwargsChecker([kw.name for kw in module.requiredArgs], [kw.name for kw in module.optionalArgs])
@castKwargsValuesAndAddDefaults(module)
def generateModelData(generatorDefinition: dict, model: FEModel, journal, *args, **kwargs) -> dict:
    kwargs = CaseInsensitiveDict(kwargs)

    name = generatorDefinition.get("name", "boxGen")

    # x0 = kwargs["x0"]
    # y0 = kwargs["y0"]
    # z0 = kwargs["z0"]

    lX = kwargs["lX"]
    lY = kwargs["lY"]
    lZ = kwargs["lZ"]

    nEleX = kwargs["nEleX"]
    nEleY = kwargs["nEleY"]
    nEleZ = kwargs["nEleZ"]

    nEleStrutX = kwargs["nEleStrutX"]
    nEleStrutY = kwargs["nEleStrutY"]
    nEleStrutZ = kwargs["nEleStrutZ"]

    nX = kwargs["nX"]
    nY = kwargs["nY"]
    nZ = kwargs["nZ"]

    elementType = getElementClass(kwargs["elType"], kwargs["elProvider"])

    boxgeneratorDefinition = generatorDefinition.copy()
    boxgenKwargs = dict(
        lX=lX,
        lY=lY,
        lZ=lZ,
        nX=nEleX,
        nY=nEleY,
        nZ=nEleZ,
        elType=kwargs["elType"],
    )
    if kwargs["elProvider"] is not None:
        boxgenKwargs.update(dict(elProvider=kwargs["elProvider"]))

    boxmodel = copy.deepcopy(model)
    boxmodel = generateBoxMesh(boxgeneratorDefinition, boxmodel, journal, **boxgenKwargs)

    lStrutX = nEleStrutX * lX / nEleX
    lStrutY = nEleStrutY / (nEleY) * lY
    lStrutZ = nEleStrutZ / (nEleZ) * lZ

    # compute coordinates of nodes to keep
    xToDelete = (lStrutX, lX - lStrutX)
    yToDelete = (lStrutY, lY - lStrutY)
    zToDelete = (lStrutZ, lZ - lStrutZ)

    elements = {}
    nodes = {}

    idx = 1
    for el in boxmodel.elements.values():
        nodeCoords = np.array([el.nodes[i].coordinates for i in range(len(el.nodes))])
        xCoords = nodeCoords[:, 0]
        yCoords = nodeCoords[:, 1]
        zCoords = nodeCoords[:, 2]

        deleteElement = False
        # delete element if all x coords are in between xToDelete
        if np.all((xCoords >= xToDelete[0] - 1e-8) & (xCoords <= xToDelete[1] + 1e-8)):
            if np.all((yCoords >= yToDelete[0] - 1e-8) & (yCoords <= yToDelete[1] + 1e-8)) or np.all(
                (zCoords >= zToDelete[0] - 1e-8) & (zCoords <= zToDelete[1] + 1e-8)
            ):
                deleteElement = True

        if np.all((yCoords >= yToDelete[0] - 1e-8) & (yCoords <= yToDelete[1] + 1e-8)):
            if np.all((zCoords >= zToDelete[0] - 1e-8) & (zCoords <= zToDelete[1] + 1e-8)):
                deleteElement = True

        if not deleteElement:
            new = elementType(kwargs["elType"], idx)
            new.setNodes([node for node in el.nodes])
            elements[idx] = new
            idx += 1

    idx = 1
    for node in boxmodel.nodes.values():
        # check if node is used by any remaining element
        nodeUsed = False
        for el in elements.values():
            if node in el.nodes:
                nodeUsed = True
                break
        if nodeUsed:
            nodes[idx] = node
            idx += 1

    model.nodes = nodes
    model.elements = elements

    # get unit cell dimensions
    x_min = 0
    x_max = lX
    y_min = 0
    y_max = lY
    z_min = 0
    z_max = lZ

    elementSets = []
    elementSets.append(ElementSet("{:}_all".format(name), elements.values()))

    model.elementSets = {es.name: es for es in elementSets}

    nodel_label_to_index = {node.label: idx for idx, node in enumerate(model.nodes.values())}
    for node in model.nodes.values():
        node.label = nodel_label_to_index[node.label] + 1  # re-label nodes to have continuous numbering

    # replicate the mesh of the unit cell in x direction
    model = replicateMesh(
        model, direction=0, nReplications=nX, elementType=elementType, options=kwargs, journal=journal
    )
    model = replicateMesh(
        model, direction=1, nReplications=nY, elementType=elementType, options=kwargs, journal=journal
    )
    model = replicateMesh(
        model, direction=2, nReplications=nZ, elementType=elementType, options=kwargs, journal=journal
    )

    model._populateNodeFieldVariablesFromElements()

    # create node sets for boundary conditions
    nSet_left = set()
    nSet_right = set()
    nSet_bottom = set()
    nSet_top = set()
    nSet_front = set()
    nSet_back = set()
    # add sets for edges as well
    nSet_top_left = set()
    nSet_top_right = set()
    nSet_bottom_left = set()
    nSet_bottom_right = set()

    # create node sets for boundaries
    for nodeID, node in model.nodes.items():
        if np.isclose(node.coordinates[0], x_min, atol=1e-8):
            nSet_left.add(node)
        elif np.isclose(node.coordinates[0], x_max + (nX - 1) * lX, atol=1e-8):
            nSet_right.add(node)
        if np.isclose(node.coordinates[1], y_min, atol=1e-8):
            nSet_bottom.add(node)
            if np.isclose(node.coordinates[0], x_min, atol=1e-8):
                nSet_bottom_left.add(node)
            elif np.isclose(node.coordinates[0], x_max + (nX - 1) * lX, atol=1e-8):
                nSet_bottom_right.add(node)
        elif np.isclose(node.coordinates[1], y_max + (nY - 1) * lY, atol=1e-8):
            nSet_top.add(node)
            if np.isclose(node.coordinates[0], x_min, atol=1e-8):
                nSet_top_left.add(node)
            elif np.isclose(node.coordinates[0], x_max + (nX - 1) * lX, atol=1e-8):
                nSet_top_right.add(node)
        if np.isclose(node.coordinates[2], z_min, atol=1e-8):
            nSet_front.add(node)
        elif np.isclose(node.coordinates[2], z_max + (nZ - 1) * lZ, atol=1e-8):
            nSet_back.add(node)

    model.nodeSets[f"{name}_left"] = NodeSet(f"{name}_left", nSet_left)
    model.nodeSets[f"{name}_right"] = NodeSet(f"{name}_right", nSet_right)
    model.nodeSets[f"{name}_bottom"] = NodeSet(f"{name}_bottom", nSet_bottom)
    model.nodeSets[f"{name}_top"] = NodeSet(f"{name}_top", nSet_top)
    model.nodeSets[f"{name}_front"] = NodeSet(f"{name}_front", nSet_front)
    model.nodeSets[f"{name}_back"] = NodeSet(f"{name}_back", nSet_back)
    model.nodeSets[f"{name}_bottom_left"] = NodeSet(f"{name}_bottom_left", nSet_bottom_left)
    model.nodeSets[f"{name}_bottom_right"] = NodeSet(f"{name}_bottom_right", nSet_bottom_right)
    model.nodeSets[f"{name}_top_left"] = NodeSet(f"{name}_top_left", nSet_top_left)
    model.nodeSets[f"{name}_top_right"] = NodeSet(f"{name}_top_right", nSet_top_right)

    return model
