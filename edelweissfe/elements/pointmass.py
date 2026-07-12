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
        self.model = model
        self.domainSize = model.domainSize
        self.mass = mass
        self.inertia = inertia if inertia is not None else [0.0, 0.0, 0.0]
        self._use_rotation = inertia is not None

        # We need to initialize the velocity fields on the node if they don't exist
        node = self.nodes[0]
        if "velocity" not in node.fields:
            if "displacement" in node.fields:
                pass

        self.velocity = np.array(initial_velocity) if initial_velocity is not None else np.zeros(self.domainSize)
        self.angular_velocity = np.zeros(1 if self.domainSize == 2 else 3)

    def computeLumpedInertia(self, Me: np.ndarray):
        """
        Populate the lumped mass matrix for this element.
        Me is a 1D array of size self.nDof.
        """
        Me[:] = 0.0

        offset = 0
        node = self.nodes[0]

        if "displacement" in node.fields:
            n_disp = self.domainSize
            Me[offset : offset + n_disp] = self.mass
            offset += n_disp

        if self._use_rotation and "rotation" in node.fields:
            if self.domainSize == 2:
                if isinstance(self.inertia, (list, np.ndarray)):
                    val = self.inertia[2] if len(self.inertia) == 3 else self.inertia[0]
                else:
                    val = self.inertia
                Me[offset] = val
            else:
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
        elif "velocity" in self.model.nodeFields and "U" in self.model.nodeFields["velocity"]:
            try:
                vel = self.model.nodeFields["velocity"].subset(node)["U"][0]
            except Exception:
                vel = self.velocity
        else:
            vel = self.velocity

        if "displacement" in node.fields:
            n_disp = self.domainSize
            Mv_e[offset : offset + n_disp] = self.mass * vel[:n_disp]
            offset += n_disp

        if self._use_rotation and "rotation" in node.fields:
            if not hasattr(self, "_first_ang_momentum_call_done"):
                ang_vel = self.angular_velocity
                self._first_ang_momentum_call_done = True
            elif hasattr(node, "current_angular_velocity") and getattr(node, "_velocity_initialized", False):
                ang_vel = node.current_angular_velocity
            elif "angular_velocity" in self.model.nodeFields and "U" in self.model.nodeFields["angular_velocity"]:
                try:
                    ang_vel = self.model.nodeFields["angular_velocity"].subset(node)["U"][0]
                except Exception:
                    ang_vel = self.angular_velocity
            else:
                ang_vel = self.angular_velocity

            if self.domainSize == 2:
                if isinstance(self.inertia, (list, np.ndarray)):
                    val = self.inertia[2] if len(self.inertia) == 3 else self.inertia[0]
                else:
                    val = self.inertia
                Mv_e[offset] = val * ang_vel[0]
            else:
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
            n_disp = self.domainSize
            structure["displacement"] = (offset, offset + n_disp)
            offset += n_disp
        if self._use_rotation and "rotation" in node.fields:
            n_rot = 1 if self.domainSize == 2 else 3
            structure["rotation"] = (offset, offset + n_rot)
        return structure

    # Dummy implementations for abstract methods of BaseElement
    @property
    def ensightType(self) -> str:
        return "point"

    @property
    def elType(self) -> str:
        return "PointMass"

    @property
    def fields(self) -> list:
        node = self.nodes[0]
        active_fields = []
        if "displacement" in node.fields:
            active_fields.append("displacement")
        if self._use_rotation and "rotation" in node.fields:
            active_fields.append("rotation")
        return [active_fields]

    @property
    def nDof(self) -> int:
        node = self.nodes[0]
        ndof = 0
        if "displacement" in node.fields:
            ndof += self.domainSize
        if self._use_rotation and "rotation" in node.fields:
            ndof += 1 if self.domainSize == 2 else 3
        return ndof

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

    def computeBodyForce(
        self,
        P: np.ndarray,
        K: np.ndarray,
        load: np.ndarray,
        U: np.ndarray,
        time: float,
        dTime: float,
        *args,
        **kwargs,
    ):
        pass

    def computeCriticalTimeStepForExplicitDynamics(self, Q=None, *args, **kwargs) -> float:
        return 1e99

    def computeDistributedLoad(
        self,
        loadType: str,
        P: np.ndarray,
        K: np.ndarray,
        faceID: int,
        load: np.ndarray,
        U: np.ndarray,
        time: float,
        dT: float,
        *args,
        **kwargs,
    ):
        pass

    def computeInternalEnergy(self) -> float:
        return 0.0

    def computeKernels(
        self, P: np.ndarray, K: np.ndarray, U: np.ndarray, dU: np.ndarray, time: float, dT: float, *args, **kwargs
    ):
        pass

    def computeKernelsExplicit(
        self, P: np.ndarray, U: np.ndarray, dU: np.ndarray, time: float, dT: float, *args, **kwargs
    ):
        pass

    def computeYourself(self, *args, **kwargs):
        pass

    def computeYourselfExplicit(self, *args, **kwargs):
        pass

    def getCoordinatesAtCenter(self) -> np.ndarray:
        return self.nodes[0].coordinates

    def getCoordinatesAtQuadraturePoints(self) -> list:
        return [self.nodes[0].coordinates]

    def getNumberOfQuadraturePoints(self) -> int:
        return 1

    def getResultArray(self, result: str, quadraturePoint: int, getPersistentView: bool = True) -> np.ndarray:
        return np.zeros(1)

    @property
    def hasMaterial(self) -> bool:
        return True

    def initializeElement(self):
        pass

    def resetToLastValidState(self):
        pass

    def setInitialCondition(self, prop: str, value: float):
        pass

    def setMaterial(self, materialName: str, materialProperties: np.ndarray):
        raise TypeError("PointMass elements cannot have materials assigned to them.")

    def setProperties(self, properties: list):
        pass

    @property
    def visualizationNodes(self) -> list:
        return self.nodes
