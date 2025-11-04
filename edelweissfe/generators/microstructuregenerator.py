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
A mesh generator for generating a structure from a single unit cell mesh:


.. code-block:: edelweiss
    :caption: Generate meshes on the fly. Example:

    *job, name=job, domain=2d, solver=NIST

    *modelGenerator, generator=microstructuregenerator, name=gen
        unitCellMeshFile = myUnitCellMesh.inp
        nX      =4
        nY      =8
        nZ      =2
"""


import meshio
import numpy as np

from edelweissfe.config.elementlibrary import getElementClass
from edelweissfe.models.femodel import FEModel
from edelweissfe.points.node import Node
from edelweissfe.utils.misc import convertLinesToStringDictionary

documentation = {
    "unitCellMeshFile": "path to the unit cell mesh file",
    "nX": "number of unit cells along x",
    "nY": "number of unit cells along y",
    "nZ": "number of unit cells along z",
}


def generateModelData(generatorDefinition: dict, model: FEModel, journal) -> dict:
    options = generatorDefinition["data"]
    options = convertLinesToStringDictionary(options)

    # name = generatorDefinition.get("name", "microstructuregenerator")

    unitCellMeshFile = options.get("unitCellMeshFile", None)
    nX = int(options.get("nX", 1))
    nY = int(options.get("nY", 1))
    # nZ = int(options.get("nZ", 1))
    elementType = getElementClass(options.get("elType", None), options.get("elProvioder", None))

    unitCellMesh = meshio.read(unitCellMeshFile)

    nodes = np.array(unitCellMesh.points)
    elements = unitCellMesh.cells

    all_nodes = nodes.copy()
    all_elements = [el for el in elements[0].data]

    # get unit cell dimensions
    x_min = np.min(nodes[:, 0])
    x_max = np.max(nodes[:, 0])
    y_min = np.min(nodes[:, 1])
    y_max = np.max(nodes[:, 1])

    lX = x_max - x_min
    lY = y_max - y_min

    # create nodes and elements for the unit cell
    _nodes = []
    for idx, node in enumerate(all_nodes):
        _node = Node(idx + 1, np.array([node[0], node[1]]))
        _nodes.append(_node)
        model.nodes[idx + 1] = _node

    for idx, element in enumerate(all_elements):
        newEl = elementType(options["elType"], idx + 1)
        nodeList = [_nodes[nid] for nid in element]
        # print(f"Creating element {idx + 1} with nodes {element}")
        # print(f"Node coordinates: {[node.coordinates for node in nodeList]}")
        newEl.setNodes(nodeList)

        model.elements[idx + 1] = newEl

    # replicate unit cell
    for i in range(nX):
        for j in range(nY):
            if i == 0 and j == 0:
                continue
            else:
                # shift nodes
                shift = np.array(
                    [
                        i * lX,
                        j * lY,
                    ]
                )
                new_nodes = []
                # # remove duplicate nodes at interfaces
                if i > 0 and j > 0:
                    new_nodes = [node + shift for node in nodes if node[0] > x_min + 1e-8 and node[1] > y_min + 1e-8]
                elif i > 0 and j == 0:
                    new_nodes = [node + shift for node in nodes if node[0] > x_min + 1e-8]
                elif j > 0 and i == 0:
                    new_nodes = [node + shift for node in nodes if node[1] > y_min + 1e-8]

                new_nodes = np.array(new_nodes)

                all_nodes = np.vstack((all_nodes, new_nodes))

                # ad nodes to model
                for node in new_nodes:
                    _node = Node(len(model.nodes) + 1, np.array([node[0], node[1]]))
                    model.nodes[len(model.nodes) + 1] = _node

                associated_nodes = []
                # brute force search of all associated nodes
                for i_old, node in enumerate(nodes):
                    success = False
                    for i_new, new_node in enumerate(new_nodes):

                        if np.allclose(node, new_node - shift, atol=1e-8):
                            associated_nodes.append([i_old, len(model.nodes) - len(new_nodes) + i_new])
                            success = True
                    if not success:
                        for i_new_, new_node_ in enumerate(all_nodes):
                            if np.allclose(node, new_node_ - shift, atol=1e-8):
                                associated_nodes.append([i_old, i_new_])

                associated_nodes_array = np.array(associated_nodes)

                for el in elements[0].data:
                    # print(f"Element: old_el: {el}")
                    new_el = []
                    for nid in el:
                        for k, assoc in enumerate(associated_nodes_array[:, 0]):
                            if nid == assoc:
                                new_el.append(int(associated_nodes_array[k, 1]))
                    # print(f"Element: new_el: {new_el}")
                    newEl = elementType(options["elType"], len(all_elements) + 1)
                    nodeList = [model.nodes[nid + 1] for nid in new_el]
                    # print(f"Creating element {len(all_elements) + 1} with nodes {new_el}")
                    # print(f"Node coordinates: {[node.coordinates for node in nodeList]}")
                    newEl.setNodes(nodeList)
                    model.elements[len(all_elements) + 1] = newEl

                    all_elements.append(new_el)

    model._populateNodeFieldVariablesFromElements()

    return model
