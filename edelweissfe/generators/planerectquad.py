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
# Created on Wed Apr 12 15:41:51 2017

# @author: Matthias Neuner
"""

A mesh generator, for rectangular geometries and structured quad meshes:


.. code-block:: console

        <-----l----->
         nX elements
         __ __ __ __
        |__|__|__|__|  A
        |__|__|__|__|  |
        |__|__|__|__|  | h
        |__|__|__|__|  | nY elements
      | |__|__|__|__|  |
      | |__|__|__|__|  V
    x0|_____
      y0

nSets, elSets, surface : 'name'_top, _bottom, _left, _right, ...
are automatically generated

Datalines:
"""

from dataclasses import dataclass

import numpy as np

from edelweissfe.config.elementlibrary import getElementClass
from edelweissfe.generators.base.generatorbase import GeneratorBase
from edelweissfe.journal.journal import Journal
from edelweissfe.models.femodel import FEModel
from edelweissfe.points.node import Node
from edelweissfe.sets.elementset import ElementSet
from edelweissfe.sets.nodeset import NodeSet
from edelweissfe.surfaces.entitybasedsurface import EntityBasedSurface
from edelweissfe.utils.schema import schemaField


@dataclass(frozen=True)
class PlaneRectQuadSchema:
    """The options this generator accepts, owned by this module and never mutated from outside
    it.

    ``length`` is spelled ``l`` in the input file -- a single-letter option name flake8 flags as
    ambiguous if used directly as a field/variable name, hence the ``optionName`` indirection.
    ``elType`` is declared ``required=True`` explicitly, but is still given a ``default=None`` so
    the schema remains constructible for the constructor's default argument.
    """

    x0: float = schemaField(description="Origin along the x axis.", dtype=float, default=0.0)
    y0: float = schemaField(description="Origin along the y axis.", dtype=float, default=0.0)
    z0: float = schemaField(description="Origin along the z axis.", dtype=float, default=0.0)
    length: float = schemaField(description="Height of the body.", dtype=float, default=1.0, optionName="l")
    h: float = schemaField(description="Length of the body.", dtype=float, default=1.0)
    nX: int = schemaField(description="Number of elements along the x axis.", dtype=int, default=1)
    nY: int = schemaField(description="Number of elements along the y axis.", dtype=int, default=1)
    nZ: int = schemaField(description="Number of elements along the z axis.", dtype=int, default=1)
    elType: str | None = schemaField(description="Element type.", dtype=str, default=None, required=True)
    elProvider: str | None = schemaField(description="Element provider.", dtype=str, default=None)


class Generator(GeneratorBase):
    """A mesh generator for rectangular geometries and structured quad meshes."""

    #: Option schema for this generator, per OptionSchemaProvider.
    schema = PlaneRectQuadSchema

    def __init__(
        self,
        name: str,
        model: FEModel,
        journal: Journal,
        *,
        configuration: PlaneRectQuadSchema = PlaneRectQuadSchema(),
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
            :class:`PlaneRectQuadSchema`.
        """
        x0 = configuration.x0
        y0 = configuration.y0

        l = configuration.length  # noqa: E741
        h = configuration.h

        nX = configuration.nX
        nY = configuration.nY

        elTypeName = configuration.elType
        elProvider = configuration.elProvider

        elType = getElementClass(elTypeName, elProvider)

        testEl = elType(elTypeName, 0)
        if testEl.nNodes == 4:
            nNodesX = nX + 1
            nNodesY = nY + 1

        if testEl.nNodes == 8:
            nNodesX = 2 * nX + 1
            nNodesY = 2 * nY + 1

        grid = np.mgrid[
            x0 : x0 + l : nNodesX * 1j,
            y0 : y0 + h : nNodesY * 1j,
        ]

        nodes = []
        # continue the numbering of any pre-existing nodes -- a fixed start at 1 would silently
        # overwrite the entries of a previously run generator in model.nodes
        currentNodeLabel = 1
        if model.nodes:
            currentNodeLabel += max(model.nodes.keys())

        for x in range(nNodesX):
            for y in range(nNodesY):
                node = Node(currentNodeLabel, grid[:, x, y])
                model.nodes[currentNodeLabel] = node
                nodes.append(node)
                currentNodeLabel += 1

        nG = np.asarray(nodes).reshape(nNodesX, nNodesY)

        # Element numbers come from the model's monotonic allocator (FEModel.reserveElementNumbers),
        # not from max(model.elements). Reserved one at a time so the count need not be predicted;
        # nothing else mints during this loop, so the numbers are consecutive exactly as before.

        elements = []
        for x in range(nX):
            for y in range(nY):
                (currentElementLabel,) = model.reserveElementNumbers(1)
                if testEl.nNodes == 4:
                    newEl = elType(elTypeName, currentElementLabel)
                    newEl.setNodes([nG[x, y], nG[x + 1, y], nG[x + 1, y + 1], nG[x, y + 1]])

                elif testEl.nNodes == 8:
                    newEl = elType(
                        elTypeName,
                        currentElementLabel,
                    )
                    newEl.setNodes(
                        [
                            nG[2 * x, 2 * y],
                            nG[2 * x + 2, 2 * y],
                            nG[2 * x + 2, 2 * y + 2],
                            nG[2 * x, 2 * y + 2],
                            nG[2 * x + 1, 2 * y],
                            nG[2 * x + 2, 2 * y + 1],
                            nG[2 * x + 1, 2 * y + 2],
                            nG[2 * x, 2 * y + 1],
                        ]
                    )
                elements.append(newEl)
                model.createElement(newEl)

        model._populateNodeFieldVariablesFromElements()

        # nodesets:
        model.nodeSets["{:}_all".format(name)] = NodeSet(
            "{:}_all".format(name), [n for n in np.ravel(nG) if len(n.fields)]
        )

        model.nodeSets["{:}_left".format(name)] = NodeSet("{:}_left".format(name), np.ravel(nG[0, :]))
        model.nodeSets["{:}_right".format(name)] = NodeSet("{:}_right".format(name), np.ravel(nG[-1, :]))
        model.nodeSets["{:}_top".format(name)] = NodeSet("{:}_top".format(name), np.ravel(nG[:, -1]))
        model.nodeSets["{:}_bottom".format(name)] = NodeSet("{:}_bottom".format(name), np.ravel(nG[:, 0]))

        model.nodeSets["{:}_leftBottom".format(name)] = NodeSet("{:}_leftBottom".format(name), nG[0, 0])
        model.nodeSets["{:}_leftTop".format(name)] = NodeSet("{:}_leftTop".format(name), nG[0, -1])
        model.nodeSets["{:}_rightBottom".format(name)] = NodeSet("{:}_rightBottom".format(name), nG[-1, 0])
        model.nodeSets["{:}_rightTop".format(name)] = NodeSet("{:}_rightTop".format(name), nG[-1, -1])

        # element sets
        elGrid = np.asarray(elements).reshape(nX, nY)
        model.elementSets["{:}_bottom".format(name)] = ElementSet("{:}_bottom".format(name), np.ravel(elGrid[:, 0]))
        model.elementSets["{:}_top".format(name)] = ElementSet("{:}_top".format(name), np.ravel(elGrid[:, -1]))
        model.elementSets["{:}_central".format(name)] = ElementSet(
            "{:}_central".format(name), elGrid[int(nX / 2), int(nY / 2)]
        )
        model.elementSets["{:}_right".format(name)] = ElementSet("{:}_right".format(name), np.ravel(elGrid[-1, :]))
        model.elementSets["{:}_left".format(name)] = ElementSet("{:}_left".format(name), np.ravel(elGrid[0, :]))

        nShearBand = min(nX, nY)
        if nShearBand > 3:
            shearBand = [
                elGrid[int(nX / 2 + i - nShearBand / 2), int(nY / 2 + i - nShearBand / 2)] for i in range(nShearBand)
            ]
            model.elementSets["{:}_shearBand".format(name)] = ElementSet(
                "{:}_shearBand".format(name), [e for e in shearBand]
            )
            model.elementSets["{:}_shearBandCenter".format(name)] = ElementSet(
                "{:}_shearBandCenter".format(name),
                [e for e in shearBand[int(nShearBand / 2) - 1 : int(nShearBand / 2) + 2]],
            )

        model.elementSets["{:}_sandwichHorizontal".format(name)] = ElementSet(
            "{:}_sandwichHorizontal".format(name), np.ravel(elGrid[1:-1, :])
        )

        model.elementSets["{:}_sandwichVertical".format(name)] = ElementSet(
            "{:}_sandwichVertical".format(name), np.ravel(elGrid[:, 1:-1])
        )

        model.elementSets["{:}_core".format(name)] = ElementSet("{:}_core".format(name), np.ravel(elGrid[1:-1, 1:-1]))

        model.elementSets["{:}_all".format(name)] = ElementSet("{:}_all".format(name), np.ravel(elGrid))
        # surfaces
        surfaceName = "{:}_bottom".format(name)
        model.surfaces[surfaceName] = EntityBasedSurface(surfaceName, {1: np.ravel(elGrid[:, 0])})
        surfaceName = "{:}_top".format(name)
        model.surfaces[surfaceName] = EntityBasedSurface(surfaceName, {3: np.ravel(elGrid[:, -1])})
        surfaceName = "{:}_right".format(name)
        model.surfaces[surfaceName] = EntityBasedSurface(surfaceName, {2: np.ravel(elGrid[-1, :])})
        surfaceName = "{:}_left".format(name)
        model.surfaces[surfaceName] = EntityBasedSurface(surfaceName, {4: np.ravel(elGrid[0, :])})
