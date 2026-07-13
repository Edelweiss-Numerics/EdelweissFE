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
