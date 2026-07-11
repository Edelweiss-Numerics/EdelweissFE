import numpy as np

from edelweissfe.elements.base.baseelement import BaseElement


class PointMass(BaseElement):
    """
    A 1-node element that adds lumped mass and rotary inertia to a node.
    """

    def __init__(
        self, elNumber: int, nodes: list, model, mass: float, inertia: list = None, initial_velocity: list = None
    ):
        super().__init__("PointMass", elNumber)
        self._elNumber = elNumber
        self._nodes = nodes
        self.mass = mass
        self.inertia = inertia if inertia is not None else [0.0, 0.0, 0.0]

        # We need to initialize the velocity fields on the node if they don't exist
        node = self.nodes[0]
        if "velocity" not in node.fields:
            if "displacement" in node.fields:
                pass

        self.velocity = np.array(initial_velocity) if initial_velocity is not None else np.zeros(3)
        self.angular_velocity = np.zeros(3)

    def computeLumpedInertia(self, Me: np.ndarray):
        """
        Populate the lumped mass matrix for this element.
        Me is a 1D array of size self.nDof.
        """
        Me[:] = 0.0

        offset = 0
        node = self.nodes[0]

        if "displacement" in node.fields:
            Me[offset : offset + 3] = self.mass
            offset += 3

        if "rotation" in node.fields:
            Me[offset] = self.inertia[0]
            Me[offset + 1] = self.inertia[1]
            Me[offset + 2] = self.inertia[2]

    def computeMomentum(self, Mv_e: np.ndarray):
        """
        Populate the momentum vector for this element.
        Mv_e is a 1D array of size self.nDof.
        """
        Mv_e[:] = 0.0
        offset = 0
        node = self.nodes[0]

        if not hasattr(self, "_first_momentum_call_done"):
            vel = self.velocity
            self._first_momentum_call_done = True
        elif hasattr(node, "current_velocity") and getattr(node, "_velocity_initialized", False):
            vel = node.current_velocity
        elif "velocity" in node.fields:
            vel = node.fields["velocity"]
        else:
            vel = self.velocity

        if "displacement" in node.fields:
            Mv_e[offset : offset + 3] = self.mass * vel
            offset += 3

        if "rotation" in node.fields:
            if not hasattr(self, "_first_ang_momentum_call_done"):
                ang_vel = self.angular_velocity
                self._first_ang_momentum_call_done = True
            elif hasattr(node, "current_angular_velocity") and getattr(node, "_velocity_initialized", False):
                ang_vel = node.current_angular_velocity
            elif "angular_velocity" in node.fields:
                ang_vel = node.fields["angular_velocity"]
            else:
                ang_vel = self.angular_velocity
            Mv_e[offset] = self.inertia[0] * ang_vel[0]
            Mv_e[offset + 1] = self.inertia[1] * ang_vel[1]
            Mv_e[offset + 2] = self.inertia[2] * ang_vel[2]

    def getStructure(self):
        """
        Return the structure of the element. (Required by some parts of the framework).
        """
        offset = 0
        node = self.nodes[0]
        structure = {}
        if "displacement" in node.fields:
            structure["displacement"] = (offset, offset + 3)
            offset += 3
        if "rotation" in node.fields:
            structure["rotation"] = (offset, offset + 3)
        return structure

    # Dummy implementations for abstract methods of BaseElement
    @property
    def ensightType(self) -> str:
        return "point"

    @property
    def fields(self) -> list:
        return [["displacement", "rotation"]]

    @property
    def nDof(self) -> int:
        return 6

    @property
    def nNodes(self) -> int:
        return 1

    @property
    def elNumber(self) -> int:
        return self._elNumber

    @property
    def nodes(self) -> list:
        return self._nodes

    @property
    def dofIndicesPermutation(self):
        return None

    def setNodes(self, nodes: list):
        self._nodes = nodes

    def acceptLastState(self):
        pass

    def computeBodyForce(self, Fe: np.ndarray, bodyForceNodes: list):
        pass

    def computeCriticalTimeStepForExplicitDynamics(self) -> float:
        return 1e99

    def computeDistributedLoad(self, Pe: np.ndarray, distributedLoad: list, timeStep):
        pass

    def computeInternalEnergy(self) -> float:
        return 0.0

    def computeYourself(self, Ke: np.ndarray, Pe: np.ndarray, *args, **kwargs):
        pass

    def computeYourselfExplicit(self, Pe: np.ndarray, *args, **kwargs):
        pass

    def getCoordinatesAtCenter(self) -> np.ndarray:
        return self.nodes[0].coordinates

    def getCoordinatesAtQuadraturePoints(self) -> list:
        return [self.nodes[0].coordinates]

    def getNumberOfQuadraturePoints(self) -> int:
        return 1

    def getResultArray(self, varName: str) -> np.ndarray:
        return np.zeros(1)

    def hasMaterial(self) -> bool:
        return False

    def initializeElement(self):
        pass

    def resetToLastValidState(self):
        pass

    def setInitialCondition(self, prop: str, value: float):
        pass

    def setMaterial(self, material):
        pass

    def setProperties(self, properties: list):
        pass

    @property
    def visualizationNodes(self) -> list:
        return self.nodes
