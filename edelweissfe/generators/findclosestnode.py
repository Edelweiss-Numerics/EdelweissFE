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
#  Paul Hofer Paul.Hofer@uibk.ac.at
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
# Created on Tue Aug 9 15:41:51 2022

"""
Find the node closest to a given spatial position, and store it in an existing or new node set.
"""

from dataclasses import dataclass

import numpy as np

from edelweissfe.generators.base.generatorbase import GeneratorBase
from edelweissfe.journal.journal import Journal
from edelweissfe.models.femodel import FEModel
from edelweissfe.sets.nodeset import NodeSet
from edelweissfe.utils.exceptions import WrongDomain
from edelweissfe.utils.schema import schemaField


@dataclass(frozen=True)
class FindClosestNodeSchema:
    """The options this generator accepts, owned by this module and never mutated from outside it.

    Both fields are declared ``required=True`` explicitly, but are still given a ``default=None``
    so the schema remains constructible for the constructor's default argument.
    """

    location: str | None = schemaField(description="Query point.", dtype=str, default=None, required=True)
    storeIn: str | None = schemaField(
        description="Node set to store closest node in.", dtype=str, default=None, required=True
    )


class Generator(GeneratorBase):
    """Find the node closest to a given spatial position, and store it in an existing or new node
    set."""

    #: Option schema for this generator, per OptionSchemaProvider.
    schema = FindClosestNodeSchema

    def __init__(
        self,
        name: str,
        model: FEModel,
        journal: Journal,
        *,
        configuration: FindClosestNodeSchema = FindClosestNodeSchema(),
    ):
        """Constructible standalone, with no parser involvement.
        Populates ``model`` directly; construction *is* the generation.

        Parameters
        ----------
        name
            Unused: this generator names no sets of its own beyond ``storeIn``.
        model
            The model tree to populate. Mutated in place.
        journal
            The journal object for logging.
        configuration
            The options this generator accepts; both are still required, see
            :class:`FindClosestNodeSchema`.
        """
        loc = np.fromstring(configuration.location, sep=",", dtype=float)

        if len(loc) != model.domainSize:
            raise WrongDomain("Spatial dimension of specified location does not match model dimension")

        allNodes = np.asarray([n.coordinates for n in model.nodes.values()])

        differenceNorm = np.linalg.norm(allNodes - loc, axis=1)

        indexClosest = differenceNorm.argmin()

        closestNode = list(model.nodes.values())[indexClosest]

        storeIn = configuration.storeIn
        if storeIn in model.nodeSets:
            raise Exception(f"Nodeset {storeIn} already exists")

        model.nodeSets[storeIn] = NodeSet(storeIn, [closestNode])
