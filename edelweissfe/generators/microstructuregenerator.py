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
The unit cell mesh must be readable by meshio (e.g., in .inp format).

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
from edelweissfe.journal.journal import Journal
from edelweissfe.models.femodel import FEModel
from edelweissfe.points.node import Node
from edelweissfe.sets.elementset import ElementSet
from edelweissfe.sets.nodeset import NodeSet
from edelweissfe.utils.caseinsensitivedict import CaseInsensitiveDict
from edelweissfe.utils.inputlanguage import InputLanguage
from edelweissfe.utils.misc import (
    caseInsensitiveKwargsChecker,
    castKwargsValuesAndAddDefaults,
)

inputLanguage = InputLanguage()
module = inputLanguage["modelGenerator"].addModule(
    "microstructuregenerator", "A mesh generator for generating a structure from a single unit cell mesh."
)

module.addOptionalArg("unitCellMeshFile", "Path to the unit cell mesh file.", str, None)

module.addOptionalArg("nX", "Number of cells along the x axis.", int, 1)
module.addOptionalArg("nY", "Number of cells along the y axis.", int, 1)
module.addOptionalArg("nZ", "Number of cells along the z axis.", int, 1)

module.addRequiredArg("elType", "Element type.", str)
module.addOptionalArg("elProvider", "Element provider.", str, None)

documentation = [module]

identification = "microgen"


@caseInsensitiveKwargsChecker([kw.name for kw in module.requiredArgs], [kw.name for kw in module.optionalArgs])
@castKwargsValuesAndAddDefaults(module)
def generateModelData(generatorDefinition: dict, model: FEModel, journal: Journal, *args, **kwargs) -> dict:
    kwargs = CaseInsensitiveDict(kwargs)

    journal.message("Generating microstructure mesh from unit cell mesh...", identification)
    name = generatorDefinition.get("name", "microgen")

    unitCellMeshFile = kwargs["unitCellMeshFile"]

    nX = kwargs["nX"]
    nY = kwargs["nY"]
    nZ = kwargs["nZ"]

    elementType = getElementClass(kwargs["elType"], kwargs["elProvider"])

    unitCellMesh = meshio.read(unitCellMeshFile)

    nodes = np.array(unitCellMesh.points)
    elements = unitCellMesh.cells

    all_nodes = nodes.copy()
    all_elements = np.array([], dtype=int).reshape(0, elements[0].data.shape[1])
    block_elements_assignments = {}
    for i, block in enumerate(elements):
        # stacke all elements together
        all_elements = np.vstack((all_elements, block.data))
        # store element ids for each block
        block_elements_assignments[i] = list(range(len(all_elements) - len(block.data), len(all_elements)))
        # create an empty element set for each block
        model.elementSets[f"{name}_block-{i + 1}"] = ElementSet(f"{name}_block-{i + 1}", set())

    # print information about the unit cell mesh
    journal.message(f"Unit cell mesh has {len(all_nodes)} nodes and {len(all_elements)} elements.", identification)
    journal.message(
        f"Element blocks in unit cell mesh: {len(block_elements_assignments)} with assignments:", identification
    )
    for block_id, el_ids in block_elements_assignments.items():
        journal.message(f"block-{block_id + 1}: {len(el_ids)} elements", identification)

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

    idx = 0
    for block_id, el_ids in block_elements_assignments.items():
        elements_per_block = []
        for local_el_id in el_ids:
            newEl = elementType(kwargs["elType"], idx + 1)
            nodeList = [_nodes[nid] for nid in all_elements[local_el_id]]
            newEl.setNodes(nodeList)

            model.elements[idx + 1] = newEl
            # add element to corresponding element set
            elements_per_block.append(newEl)
            idx += 1

        model.elementSets[f"{name}_block-{block_id + 1}"] = ElementSet(
            f"{name}_block-{block_id + 1}", set(elements_per_block)
        )

    # replicate the mesh of the unit cell in x direction
    model = replicateMesh(
        model, direction=0, nReplications=nX, elementType=elementType, options=kwargs, journal=journal
    )

    # replicate the already replicated mesh in y direction
    model = replicateMesh(
        model, direction=1, nReplications=nY, elementType=elementType, options=kwargs, journal=journal
    )

    if model.domainSize == 3:
        # replicate the already replicated mesh in z direction
        model = replicateMesh(
            model, direction=2, nReplications=nZ, elementType=elementType, options=kwargs, journal=journal
        )

    model._populateNodeFieldVariablesFromElements()

    # create node sets for boundary conditions
    nSet_left = set()
    nSet_right = set()
    nSet_bottom = set()
    nSet_top = set()
    # add sets for corners as well
    nSet_top_left = set()
    nSet_top_right = set()
    nSet_bottom_left = set()
    nSet_bottom_right = set()

    # create node sets for left and bottom boundaries
    for nodeID, node in model.nodes.items():
        if np.isclose(node.coordinates[1], y_min, atol=1e-8):
            nSet_bottom.add(node)
            if np.isclose(node.coordinates[0], x_min, atol=1e-8):
                nSet_bottom_left.add(node)
            elif np.isclose(node.coordinates[0], x_max + (nX - 1) * lX, atol=1e-8):
                nSet_bottom_right.add(node)
        if np.isclose(node.coordinates[0], x_min, atol=1e-8):
            nSet_left.add(node)
        if np.isclose(node.coordinates[0], x_max + (nX - 1) * lX, atol=1e-8):
            nSet_right.add(node)
        if np.isclose(node.coordinates[1], y_max + (nY - 1) * lY, atol=1e-8):
            nSet_top.add(node)
            if np.isclose(node.coordinates[0], x_min, atol=1e-8):
                nSet_top_left.add(node)
            elif np.isclose(node.coordinates[0], x_max + (nX - 1) * lX, atol=1e-8):
                nSet_top_right.add(node)

    model.nodeSets[f"{name}_left"] = NodeSet(f"{name}_left", nSet_left)
    model.nodeSets[f"{name}_right"] = NodeSet(f"{name}_right", nSet_right)
    model.nodeSets[f"{name}_bottom"] = NodeSet(f"{name}_bottom", nSet_bottom)
    model.nodeSets[f"{name}_top"] = NodeSet(f"{name}_top", nSet_top)
    model.nodeSets[f"{name}_bottom_left"] = NodeSet(f"{name}_bottom_left", nSet_bottom_left)
    model.nodeSets[f"{name}_bottom_right"] = NodeSet(f"{name}_bottom_right", nSet_bottom_right)
    model.nodeSets[f"{name}_top_left"] = NodeSet(f"{name}_top_left", nSet_top_left)
    model.nodeSets[f"{name}_top_right"] = NodeSet(f"{name}_top_right", nSet_top_right)

    return model


def findInterfaceNodes(nodes, coordIndex, coordValue, idx_offset=0):
    interfaceNodes = set()
    ids = np.where(np.isclose(nodes[:, coordIndex], coordValue, atol=1e-5))[0] + idx_offset
    interfaceNodes.update(ids.tolist())
    return interfaceNodes


def replicateMesh(
    model: FEModel, direction: int, nReplications: int, elementType, options: dict, journal: Journal = None
) -> FEModel:

    all_elements_to_copy = [[n.label - 1 for n in el.nodes] for el in model.elements.values()]
    all_nodes_to_copy = [model.nodes[i + 1].coordinates for i in range(len(model.nodes))]

    elements_in_block = {}
    # separate elements according to their blocks
    for elset_name, elset in model.elementSets.items():
        elements_in_block[elset_name] = []
        for el in elset.elements:
            elements_in_block[elset_name].append(el.elNumber - 1)

    all_nodes = np.array(all_nodes_to_copy)

    direction_min = np.min([node[direction] for node in all_nodes])
    direction_max = np.max([node[direction] for node in all_nodes])
    length_in_direction = direction_max - direction_min

    shift = np.zeros(len(all_nodes[0]))

    minNodes = np.array(
        [k for k, node in enumerate(all_nodes_to_copy) if np.isclose(node[direction], direction_min, atol=1e-8)]
    )

    nodes_to_shift = [(k, node) for k, node in enumerate(all_nodes_to_copy) if k not in minNodes]

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
        idx_offset = (j - 1) * (len(all_nodes_to_copy) - len(minNodes))
        search_array = all_nodes[idx_offset:, :]

        for i_old in minNodes:
            i_new_ = (
                np.where(np.all(np.abs(search_array - all_nodes_to_copy[i_old] - shift) < 1e-5, axis=1))[0][0]
                + idx_offset
            )
            associated_nodes.append([i_old, i_new_])
        associated_nodes_array = np.array(associated_nodes)

        # create elements per block
        for elset_name, el_ids in elements_in_block.items():
            new_el_ids = []
            for local_el_id in el_ids:
                el = all_elements_to_copy[local_el_id]
                new_el = []
                for nid in el:
                    k = np.where(associated_nodes_array[:, 0] == nid)[0][0]
                    new_el.append(int(associated_nodes_array[k, 1]))
                newEl = elementType(options["elType"], len(model.elements) + 1)
                nodeList = [model.nodes[nid + 1] for nid in new_el]
                newEl.setNodes(nodeList)
                model.elements[len(model.elements) + 1] = newEl
                # add element to corresponding element set
                new_el_ids.append(len(model.elements) - 1)

            # create new element set becaus elsets are immutable
            model.elementSets[elset_name] = ElementSet(
                elset_name,
                set(list(model.elementSets[elset_name].elements) + [model.elements[el_id + 1] for el_id in new_el_ids]),
            )

        # remove nodes that are now internal
        toc_total = time.time()

        if journal:
            journal.message(f"Replication step {j}/{nReplications - 1} in direction {direction} done.", identification)
            journal.message(f" Total nodes so far: {len(model.nodes)}", identification)
            journal.message(f" Total elements so far: {len(model.elements)}", identification)
            journal.message(
                f" Total time for replication step: {round(toc_total - tic_total, 2)} seconds", identification
            )

    return model
