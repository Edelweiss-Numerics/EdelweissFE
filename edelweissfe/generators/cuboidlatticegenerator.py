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

.. code-block:: edelweiss
    :caption: Generate meshes on the fly. Example:

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
from edelweissfe.utils.misc import convertLinesToStringDictionary

documentation = {
    "lX": "Length of the unit cell in X direction.",
    "lY": "Length of the unit cell in Y direction.",
    "lZ": "Length of the unit cell in Z direction.",
    "nEleX": "Number of elements in X direction of the unit cell.",
    "nEleY": "Number of elements in Y direction of the unit cell.",
    "nEleZ": "Number of elements in Z direction of the unit cell.",
    "nEleStrutX": "Number of elements in the struts in X direction of the unit cell.",
    "nEleStrutY": "Number of elements in the struts in Y direction of the unit cell.",
    "nEleStrutZ": "Number of elements in the of struts in Z direction of the unit cell.",
    "nX": "Number of replications in X direction.",
    "nY": "Number of replications in Y direction.",
    "nZ": "Number of replications in Z direction.",
}


def generateModelData(generatorDefinition: dict, model: FEModel, journal) -> dict:

    name = generatorDefinition.get("name", "cuboidlatticegenerator")

    options = generatorDefinition["data"]
    options = convertLinesToStringDictionary(options)

    boxgeneratorDefinition = generatorDefinition.copy()
    boxgeneratorDefinition["data"] = (
        f"lX={options['lX']}",
        f"lY={options['lY']}",
        f"lZ={options['lZ']}",
        f"nX={options['nEleX']}",
        f"nY={options['nEleY']}",
        f"nZ={options['nEleZ']}",
        f"elType={options['elType']}",
    )

    boxmodel = copy.deepcopy(model)

    boxmodel = generateBoxMesh(boxgeneratorDefinition, boxmodel, journal)

    # now remove elements and nodes to create the lattice structure
    nEleStrutX = int(options["nEleStrutX"])
    nEleStrutY = int(options["nEleStrutY"])
    nEleStrutZ = int(options["nEleStrutZ"])

    nEleX = int(options["nEleX"])
    nEleY = int(options["nEleY"])
    nEleZ = int(options["nEleZ"])

    # compute coordinates of nodes to keep
    lX = float(options["lX"])
    lY = float(options["lY"])
    lZ = float(options["lZ"])

    lStrutX = nEleStrutX * lX / nEleX
    lStrutY = nEleStrutY / (nEleY) * lY
    lStrutZ = nEleStrutZ / (nEleZ) * lZ

    xToDelete = (lStrutX, lX - lStrutX)
    yToDelete = (lStrutY, lY - lStrutY)
    zToDelete = (lStrutZ, lZ - lStrutZ)

    elements = {}
    nodes = {}

    elementType = getElementClass(options.get("elType", None), options.get("elProvider", None))
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
            new = elementType(options["elType"], idx)
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

    nX = int(options.get("nX", 1))
    nY = int(options.get("nY", 1))
    nZ = int(options.get("nZ", 1))

    elementSets = []
    elementSets.append(ElementSet("{:}_all".format(name), elements.values()))

    model.elementSets = {es.name: es for es in elementSets}

    nodel_label_to_index = {node.label: idx for idx, node in enumerate(model.nodes.values())}
    for node in model.nodes.values():
        node.label = nodel_label_to_index[node.label] + 1  # re-label nodes to have continuous numbering

    # replicate the mesh of the unit cell in x direction
    model = replicateMesh(model, direction=0, nReplications=nX, elementType=elementType, options=options)
    model = replicateMesh(model, direction=1, nReplications=nY, elementType=elementType, options=options)
    model = replicateMesh(model, direction=2, nReplications=nZ, elementType=elementType, options=options)

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
    model.nodeSets[f"{name}_bottom_left"] = NodeSet(f"{name}_bottom_left", nSet_bottom_left)
    model.nodeSets[f"{name}_bottom_right"] = NodeSet(f"{name}_bottom_right", nSet_bottom_right)
    model.nodeSets[f"{name}_top_left"] = NodeSet(f"{name}_top_left", nSet_top_left)
    model.nodeSets[f"{name}_top_right"] = NodeSet(f"{name}_top_right", nSet_top_right)

    return model
