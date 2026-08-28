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
from dataclasses import dataclass

import numpy as np

from edelweissfe.config.elementlibrary import getElementClass
from edelweissfe.generators.base.generatorbase import GeneratorBase
from edelweissfe.generators.boxgen import BoxgenSchema
from edelweissfe.generators.boxgen import Generator as BoxGenerator
from edelweissfe.generators.microstructuregenerator import replicateMesh
from edelweissfe.journal.journal import Journal
from edelweissfe.models.femodel import FEModel
from edelweissfe.sets.elementset import ElementSet
from edelweissfe.sets.nodeset import NodeSet
from edelweissfe.utils.schema import schemaField


@dataclass(frozen=True)
class CuboidLatticeGeneratorSchema:
    """The options this generator accepts, owned by this module and never mutated from outside
    it.

    ``elType`` is declared ``required=True`` explicitly, but is still given a ``default=None`` so
    the schema remains constructible for the constructor's default argument.
    """

    lX: float = schemaField(description="Length of the body along the x axis.", dtype=float, default=1.0)
    lY: float = schemaField(description="Length of the body along the y axis.", dtype=float, default=1.0)
    lZ: float = schemaField(description="Length of the body along the z axis.", dtype=float, default=1.0)
    nEleX: int = schemaField(description="Number of elements along the x axis.", dtype=int, default=1)
    nEleY: int = schemaField(description="Number of elements along the y axis.", dtype=int, default=1)
    nEleZ: int = schemaField(description="Number of elements along the z axis.", dtype=int, default=1)
    nEleStrutX: int = schemaField(description="Number of in struts along the x axis.", dtype=int, default=1)
    nEleStrutY: int = schemaField(description="Number of in struts along the y axis.", dtype=int, default=1)
    nEleStrutZ: int = schemaField(description="Number of in struts along the z axis.", dtype=int, default=1)
    nX: int = schemaField(description="Number of replications along the x axis.", dtype=int, default=1)
    nY: int = schemaField(description="Number of replications along the y axis.", dtype=int, default=1)
    nZ: int = schemaField(description="Number of replications along the z axis.", dtype=int, default=1)
    elType: str | None = schemaField(description="Element type.", dtype=str, default=None, required=True)
    elProvider: str | None = schemaField(description="Element provider.", dtype=str, default=None)


class Generator(GeneratorBase):
    """A mesh generator for generating cuboid lattice structure."""

    #: Option schema for this generator, per OptionSchemaProvider.
    schema = CuboidLatticeGeneratorSchema

    def __init__(
        self,
        name: str,
        model: FEModel,
        journal: Journal,
        *,
        configuration: CuboidLatticeGeneratorSchema = CuboidLatticeGeneratorSchema(),
    ):
        """Constructible standalone, with no parser involvement.
        Populates ``model`` directly; construction *is* the generation.

        Parameters
        ----------
        name
            The name of this generator instance, used as the prefix for the generated sets.
        model
            The model tree to populate. Mutated in place.
        journal
            The journal object for logging.
        configuration
            The options this generator accepts; ``elType`` is still required, see
            :class:`CuboidLatticeGeneratorSchema`.
        """
        lX = configuration.lX
        lY = configuration.lY
        lZ = configuration.lZ

        nEleX = configuration.nEleX
        nEleY = configuration.nEleY
        nEleZ = configuration.nEleZ

        nEleStrutX = configuration.nEleStrutX
        nEleStrutY = configuration.nEleStrutY
        nEleStrutZ = configuration.nEleStrutZ

        nX = configuration.nX
        nY = configuration.nY
        nZ = configuration.nZ

        elementType = getElementClass(configuration.elType, configuration.elProvider)

        boxmodel = copy.deepcopy(model)
        BoxGenerator(
            name,
            boxmodel,
            journal,
            configuration=BoxgenSchema(
                lX=lX,
                lY=lY,
                lZ=lZ,
                nX=nEleX,
                nY=nEleY,
                nZ=nEleZ,
                elType=configuration.elType,
                elProvider=configuration.elProvider,
            ),
        )

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
                new = elementType(configuration.elType, idx)
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
        # This generator REPLACES the element dict wholesale, with its own 1..N numbering, rather
        # than creating elements through the model. Tell the allocator about those numbers, so that
        # everything minted afterwards (contact facets, rigid-body point masses, model modifiers)
        # cannot collide with them. See FEModel.adoptSetupElementNumbers.
        model.elements = elements
        model.adoptSetupElementNumbers()

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
        # Same story as the elements above, for the node dict this generator also replaces
        # wholesale: the replications below mint their labels from the allocator.
        model.adoptSetupNodeNumbers()

        # replicate the mesh of the unit cell in x direction
        replicateMesh(
            model,
            direction=0,
            nReplications=nX,
            elementType=elementType,
            elTypeName=configuration.elType,
            journal=journal,
        )
        replicateMesh(
            model,
            direction=1,
            nReplications=nY,
            elementType=elementType,
            elTypeName=configuration.elType,
            journal=journal,
        )
        replicateMesh(
            model,
            direction=2,
            nReplications=nZ,
            elementType=elementType,
            elTypeName=configuration.elType,
            journal=journal,
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
