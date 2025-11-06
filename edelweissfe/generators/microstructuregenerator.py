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

import time

import meshio
import numpy as np

from edelweissfe.config.elementlibrary import getElementClass
from edelweissfe.models.femodel import FEModel
from edelweissfe.points.node import Node
from edelweissfe.sets.nodeset import NodeSet
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

    print("  Generating microstructure mesh from unit cell mesh...")
    name = generatorDefinition.get("name", "microgen")

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
        _node = Node(idx + 1, np.array(node))
        _nodes.append(_node)
        model.nodes[idx + 1] = _node

    for idx, element in enumerate(all_elements):
        newEl = elementType(options["elType"], idx + 1)
        nodeList = [_nodes[nid] for nid in element]
        newEl.setNodes(nodeList)

        model.elements[idx + 1] = newEl

    # replicate the mesh of the unit cell in x direction
    model = replicateMesh(model, direction=0, nReplications=nX, elementType=elementType, options=options)

    # replicate the already replicated mesh in y direction
    model = replicateMesh(model, direction=1, nReplications=nY, elementType=elementType, options=options)

    model._populateNodeFieldVariablesFromElements()

    # create node sets for boundary conditions
    nSet_left = set()
    nSet_right = set()
    nSet_bottom = set()
    nSet_top = set()
    # create node sets for left and bottom boundaries
    for nodeID, node in model.nodes.items():
        if np.isclose(node.coordinates[1], y_min, atol=1e-8):
            nSet_bottom.add(node)
        if np.isclose(node.coordinates[0], x_min, atol=1e-8):
            nSet_left.add(node)
        if np.isclose(node.coordinates[0], x_max + (nX - 1) * lX, atol=1e-8):
            nSet_right.add(node)
        if np.isclose(node.coordinates[1], y_max + (nY - 1) * lY, atol=1e-8):
            nSet_top.add(node)

    model.nodeSets[f"{name}_left"] = NodeSet(f"{name}_left", nSet_left)
    model.nodeSets[f"{name}_right"] = NodeSet(f"{name}_right", nSet_right)
    model.nodeSets[f"{name}_bottom"] = NodeSet(f"{name}_bottom", nSet_bottom)
    model.nodeSets[f"{name}_top"] = NodeSet(f"{name}_top", nSet_top)

    return model


def findInterfaceNodes(nodes, coordIndex, coordValue, idx_offset=0):
    interfaceNodes = set()
    ids = np.where(np.isclose(nodes[:, coordIndex], coordValue, atol=1e-5))[0] + idx_offset
    interfaceNodes.update(ids.tolist())
    return interfaceNodes


def replicateMesh(model: FEModel, direction: int, nReplications: int, elementType, options: dict):

    all_elements_in_x = [[n.label - 1 for n in el.nodes] for el in model.elements.values()]
    all_nodes_in_x = [model.nodes[i + 1].coordinates for i in range(len(model.nodes))]
    # replicate all created unit cells in y direction

    all_nodes = np.array(all_nodes_in_x)

    direction_min = np.min([node[direction] for node in all_nodes_in_x])
    direction_max = np.max([node[direction] for node in all_nodes_in_x])
    length_in_direction = direction_max - direction_min

    shift = np.zeros(len(all_nodes_in_x[0]))

    minNodes = np.array(
        [k for k, node in enumerate(all_nodes_in_x) if np.isclose(node[direction], direction_min, atol=1e-8)]
    )

    nodes_to_shift = [(k, node) for k, node in enumerate(all_nodes_in_x) if k not in minNodes]

    for j in range(1, nReplications):
        tic_total = time.time()
        # shift nodes
        shift[direction] = j * length_in_direction
        new_nodes = []
        associated_nodes = []

        for k, node in nodes_to_shift:
            new_nodes.append(node + shift)
            associated_nodes.append([k, len(model.nodes)])
            _node = Node(len(model.nodes) + 1, np.array(new_nodes[-1]))
            model.nodes[len(model.nodes) + 1] = _node

        new_nodes = np.array(new_nodes)

        all_nodes = np.vstack((all_nodes, new_nodes))

        # create (smaller) array to search in
        idx_offset = (j - 1) * (len(all_nodes_in_x) - len(minNodes))
        search_array = all_nodes[idx_offset:, :]

        for i_old in minNodes:
            i_new_ = np.where(np.all(np.abs(search_array - all_nodes[i_old] - shift) < 1e-5, axis=1))[0][0] + idx_offset
            associated_nodes.append([i_old, i_new_])
        associated_nodes_array = np.array(associated_nodes)

        for el in all_elements_in_x:
            new_el = []
            for nid in el:
                k = np.where(associated_nodes_array[:, 0] == nid)[0][0]
                new_el.append(int(associated_nodes_array[k, 1]))
            newEl = elementType(options["elType"], len(model.elements) + 1)
            nodeList = [model.nodes[nid + 1] for nid in new_el]
            newEl.setNodes(nodeList)
            model.elements[len(model.elements) + 1] = newEl

        # remove nodes that are now internal
        toc_total = time.time()
        print(f"    Replication step {j}/{nReplications - 1} in direction {direction} done.")
        print(f"      Total nodes so far: {len(model.nodes)}")
        print(f"      Total elements so far: {len(model.elements)}")
        print(f"      Total time for replication step: {round(toc_total - tic_total, 2)} seconds")

    return model
