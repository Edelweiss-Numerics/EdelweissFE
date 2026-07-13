from typing import Any, Dict, List

import numpy as np

from edelweissfe.rigidbodies.rigidbody import RigidBody
from edelweissfe.utils.caseinsensitivedict import CaseInsensitiveDict


class DiscreteRigidBody(RigidBody):
    """
    Discrete Rigid Body entity.

    It represents a rigid body defined by a discrete surface mesh. Instances
    are normally created via
    :func:`~edelweissfe.generators.discreterigidbodygenerator.generateDiscreteRigidBodyFromMeshFile`,
    which handles loading the mesh and creating the surface/reference-point
    nodes -- this class itself only deals with the rigid body's kinematics.
    """

    def __init__(self, name, model, *args, **kwargs):
        super().__init__(name, model)
        self.model.rigidBodies[self.name] = self

        kwargs = CaseInsensitiveDict(kwargs)

        self.surfaceNodes = list(model.nodeSets[kwargs["nSet"]])
        rpNodeSet = model.nodeSets[kwargs["referencePoint"]]

        if len(rpNodeSet) > 1:
            raise ValueError("Reference point set must contain exactly one node!")

        self.rpNode = list(rpNodeSet)[0]
        self.domainSize = model.domainSize

        # Initialize default explicit velocities to avoid hasattr/getattr on RP node
        self.rpNode.current_velocity = np.zeros(self.domainSize)
        if self.domainSize == 3:
            self.rpNode.current_angular_velocity = np.zeros(3)

        self.surface_mesh = None
        self._query_engine = None

        # Facets replace DiscreteRigidElement.
        # Format: [{'type': 'tria3', 'nodes': [node1, node2, node3]}, ...]
        self.facets = kwargs.get("facets", [])

        # Precompute initial relative positions of surface nodes w.r.t. the RP
        self.initialRelativePositions = np.array([n.coordinates - self.rpNode.coordinates for n in self.surfaceNodes])

        self.mass = kwargs.get("mass")
        self.inertia = kwargs.get("inertia")
        self.initial_velocity = kwargs.get("initial_velocity")

        # Abstract the PointMass element
        self.point_mass_element = None
        if self.mass is not None:
            from edelweissfe.elements.pointmass import PointMass

            el_num = max(model.elements.keys()) + 1 if model.elements else 1
            self.point_mass_element = PointMass(
                el_num, [self.rpNode], model, self.mass, self.inertia, self.initial_velocity
            )
            model.elements[el_num] = self.point_mass_element

    def _getFieldU(self, fieldName, node):
        if fieldName not in node.fields:
            # Depending on if it's displacement (size domainSize) or rotation (size varies)
            from edelweissfe.config.phenomena import getFieldSize

            return np.zeros(getFieldSize(fieldName, self.domainSize))
        node_field = self.model.nodeFields.get(fieldName)
        if node_field is None or "U" not in node_field:
            from edelweissfe.config.phenomena import getFieldSize

            return np.zeros(getFieldSize(fieldName, self.domainSize))
        return node_field.subset(node)["U"][0].copy()

    def getCurrentKinematics(self):
        disp_field = self.model.nodeFields.get("displacement")
        if disp_field is not None and "U" in disp_field and self.rpNode in disp_field.nodes:
            idx = disp_field._indicesOfNodesInArray[self.rpNode]
            u_rp = disp_field["U"][idx].copy()
        else:
            u_rp = np.zeros(self.model.domainSize)

        R = np.eye(3)
        if self.model.domainSize == 3:
            rot_field = self.model.nodeFields.get("rotation")
            if rot_field is not None and "U" in rot_field and self.rpNode in rot_field.nodes:
                idx = rot_field._indicesOfNodesInArray[self.rpNode]
                theta = rot_field["U"][idx]
                R = self._getRotationMatrix3D(theta)

        # In EdelweissFE, node.coordinates are the *reference* (initial) coordinates.
        return u_rp, R, self.rpNode.coordinates

    def updateKinematics(self, timeStep=None):
        u_rp, R, rp_initial = self.getCurrentKinematics()
        rp_current = rp_initial + u_rp

        disp_field = self.model.nodeFields.get("displacement")
        has_disp = disp_field is not None and "U" in disp_field

        new_coords = rp_current + self.initialRelativePositions.dot(R.T)
        if has_disp:
            disp_u = disp_field["U"]

        for i, node in enumerate(self.surfaceNodes):
            node.coordinates[:] = new_coords[i]
            if has_disp:
                idx = disp_field._indicesOfNodesInArray[node]
                disp_u[idx] = new_coords[i] - (self.rpNode.coordinates + self.initialRelativePositions[i])

    def getAABB(self):
        coords = np.array([n.coordinates for n in self.surfaceNodes])
        return np.min(coords, axis=0), np.max(coords, axis=0)

    def _getRotationMatrix3D(self, theta):
        angle = np.linalg.norm(theta)
        if angle < 1e-12:
            return np.eye(3)
        axis = theta / angle
        K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
        R = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * np.dot(K, K)
        return R

    def getVisualizationNodes(self) -> List:
        return self.surfaceNodes

    def getVisualizationElements(self) -> List[Dict[str, Any]]:
        return self.facets

    def getVisualizationField(self, fieldName: str) -> np.ndarray:
        # Generic query to the model's nodeFields
        if fieldName not in self.model.nodeFields:
            return np.zeros((len(self.surfaceNodes), self.model.domainSize))
        return np.zeros((len(self.surfaceNodes), self.model.domainSize))
