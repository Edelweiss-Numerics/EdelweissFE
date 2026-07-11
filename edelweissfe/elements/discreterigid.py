import numpy as np

from edelweissfe.elements.base.baseelement import BaseElement
from edelweissfe.points.node import Node


class DiscreteRigidElement(BaseElement):
    """
    A lightweight element class to represent discrete rigid body surfaces in the FEModel tree.
    It has no constitutive behavior and does not contribute to the system's internal forces or stiffness matrix.
    Its sole purpose is to group nodes into topological shapes (e.g., tria3 or quad4) for Ensight export.
    """

    def __init__(self, elNumber: int, nodes: list[Node], model, ensightType: str):
        self._elNumber = elNumber
        self._model = model
        self._ensightType = ensightType
        self.setNodes(nodes)

    @property
    def elNumber(self) -> int:
        return self._elNumber

    def elType(self) -> str:
        return "DiscreteRigidElement"

    @property
    def hasMaterial(self) -> str:
        return True

    @property
    def nodes(self) -> list[Node]:
        return self._nodes

    def setNodes(self, nodes: list[Node]):
        self._nodes = nodes
        self._nDof = 0

    def setProperties(self, elementProperties: np.ndarray):
        pass

    def initializeElement(self):
        pass

    def setMaterial(self, materialName: str, materialProperties: np.ndarray):
        pass

    def setInitialCondition(self, stateType: str, values: np.ndarray):
        pass

    def computeDistributedLoad(
        self,
        loadType: str,
        P: np.ndarray,
        K: np.ndarray,
        faceID: int,
        load: np.ndarray,
        U: np.ndarray,
        time: np.ndarray,
        dT: float,
    ):
        pass

    def computeYourself(
        self,
        P: np.ndarray,
        K: np.ndarray,
        U: np.ndarray,
        dU: np.ndarray,
        time: np.ndarray,
        dT: float,
    ):
        pass

    def computeYourselfExplicit(
        self,
        P: np.ndarray,
        U: np.ndarray,
        dU: np.ndarray,
        time: np.ndarray,
        dT: float,
    ):
        pass

    def computeLumpedInertia(self, M: np.ndarray):
        pass

    def computeCriticalTimeStepForExplicitDynamics(self, Q: np.ndarray) -> float:
        return np.inf

    def updateOutput(self, fieldValuesList: list):
        pass

    def updateHistoryVariables(self):
        pass

    @property
    def fields(self) -> list:
        return [["displacement"] for _ in self.nodes]

    @property
    def nDof(self) -> int:
        return 0

    @property
    def visualizationNodes(self) -> list[Node]:
        return self.nodes

    @property
    def ensightType(self) -> str:
        return self._ensightType

    def acceptLastState(self):
        pass

    def computeBodyForce(self, *args, **kwargs):
        pass

    def computeInternalEnergy(self, *args, **kwargs) -> float:
        return 0.0

    @property
    def dofIndicesPermutation(self):
        return None

    def getCoordinatesAtCenter(self) -> np.ndarray:
        return np.mean([n.coordinates for n in self.nodes], axis=0)

    def getCoordinatesAtQuadraturePoints(self, *args, **kwargs):
        return []

    def getNumberOfQuadraturePoints(self, *args, **kwargs) -> int:
        return 0

    def getResultArray(self, *args, **kwargs):
        return None

    @property
    def nNodes(self) -> int:
        return len(self.nodes)

    def resetToLastValidState(self):
        pass
