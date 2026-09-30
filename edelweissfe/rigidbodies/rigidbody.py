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

from abc import ABC, abstractmethod
from typing import Any, Dict, List

import numpy as np


class RigidBody(ABC):
    """
    Abstract Base Class for all rigid bodies in the explicit and implicit solver framework.
    """

    def __init__(self, name: str, model):
        self.name = name
        self.model = model
        self.rpNode = None

    @abstractmethod
    def updateKinematics(self, timeStep=None):
        """
        Update the kinematics of the rigid body according to its prescribed or computed motion.
        """

    def referenceCoordinatesOfMovedNodes(self) -> Dict[int, np.ndarray]:
        """The reference coordinates of the nodes whose ``coordinates`` this body moves along with it.

        A node's ``coordinates`` are normally its reference coordinates, and the topology fingerprint
        (:meth:`~edelweissfe.models.femodel.FEModel.topologyFingerprint`) hashes them. A rigid body that
        writes its current configuration into its nodes reports their reference coordinates here, so
        that the fingerprint describes the mesh and not the body's current position.

        Returns
        -------
        Dict[int, numpy.ndarray]
            The reference coordinates, keyed by node label. Empty if this body moves no nodes.
        """
        return {}

    @abstractmethod
    def getCurrentKinematics(self):
        """
        Retrieve the current kinematic state of the rigid body.
        """

    @abstractmethod
    def getVisualizationNodes(self) -> List:
        """
        Returns the nodes that should be visualized for this rigid body.
        """

    @abstractmethod
    def getVisualizationElements(self) -> List[Dict[str, Any]]:
        """
        Returns the geometric elements for visualization (e.g., facets).
        Each element is a dict with 'type' and 'nodes' keys.
        """

    @abstractmethod
    def getVisualizationField(self, fieldName: str) -> np.ndarray:
        """
        Returns the mapped results for visualization (e.g., displacement) on the nodes.
        """
