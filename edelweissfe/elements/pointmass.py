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
        self._use_rotation = inertia is not None

        # Normalize the rotary inertia to one value per rotational DOF.
        nRot = 1 if self.domainSize == 2 else 3
        if inertia is None:
            self.inertia = np.zeros(nRot)
        else:
            inertia = np.atleast_1d(np.asarray(inertia, dtype=float))
            if self.domainSize == 2:
                # A 3-component (diagonal) inertia refers to the out-of-plane axis.
                self.inertia = inertia[[2]] if inertia.shape[0] == 3 else inertia[[0]]
            else:
                if inertia.shape[0] != 3:
                    raise ValueError("PointMass in 3D requires a diagonal inertia [Ixx, Iyy, Izz].")
                self.inertia = inertia

        # The node's velocity attributes are the single source of truth for
        # the momentum computation; they are updated by the explicit solvers.
        node = self.nodes[0]
        node.current_velocity = (
            np.array(initial_velocity, dtype=float)[: self.domainSize]
            if initial_velocity is not None
            else np.zeros(self.domainSize)
        )
        node.current_angular_velocity = np.zeros(nRot)

    def computeLumpedInertia(self, Me: np.ndarray):
        """
        Populate the lumped mass matrix for this element.
        Me is a 1D array of size self.nDof.
        """
        Me[:] = 0.0

        n_disp = self.domainSize
        Me[:n_disp] = self.mass

        if self._use_rotation:
            Me[n_disp : n_disp + self.inertia.shape[0]] = self.inertia

    def computeMomentum(self, Mv_e: np.ndarray):
        """
        Populate the momentum vector for this element.
        Mv_e is a 1D array of size self.nDof.
        """
        Mv_e[:] = 0.0
        node = self.nodes[0]

        n_disp = self.domainSize
        Mv_e[:n_disp] = self.mass * node.current_velocity[:n_disp]

        if self._use_rotation:
            nRot = self.inertia.shape[0]
            Mv_e[n_disp : n_disp + nRot] = self.inertia * node.current_angular_velocity[:nRot]

    # Dummy implementations for abstract methods of BaseElement
    @property
    def ensightType(self) -> str:
        return "point"

    @property
    def elType(self) -> str:
        return "PointMass"

    @property
    def fields(self) -> list:
        active_fields = ["displacement"]
        if self._use_rotation:
            active_fields.append("rotation")
        return [active_fields]

    @property
    def nDof(self) -> int:
        ndof = self.domainSize
        if self._use_rotation:
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
