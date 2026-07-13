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

        # The RP velocity state is owned by the PointMass mass carrier (created
        # below), not by the node -- the explicit solver keeps it in sync.
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

    def getCurrentKinematics(self):
        """Return the current rigid body motion.

        Returns
        -------
        tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]
            The total RP displacement, the 3x3 rotation matrix, and the
            *reference* (initial) RP coordinates. The current RP position is
            the sum of the first and third entries.
        """
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

        # In EdelweissFE, the RP node's coordinates are the *reference* (initial) coordinates.
        return u_rp, R, self.rpNode.coordinates

    def updateKinematics(self, timeStep=None):
        u_rp, R, rp_initial = self.getCurrentKinematics()
        rp_current = rp_initial + u_rp

        disp_field = self.model.nodeFields.get("displacement")
        has_disp = disp_field is not None and "U" in disp_field

        d = self.domainSize
        new_coords = rp_current + self.initialRelativePositions.dot(R[:d, :d].T)
        if has_disp:
            disp_u = disp_field["U"]

        # Deviating from the usual EdelweissFE convention, the surface nodes'
        # coordinates hold the *current* configuration: the output managers
        # write the transient geometry of the moving body from them, and the
        # AABB proximity checks of the contact constraints rely on them. The
        # reference configuration is retained via the RP node's coordinates
        # and initialRelativePositions.
        for i, node in enumerate(self.surfaceNodes):
            node.coordinates[:] = new_coords[i]
            if has_disp:
                idx = disp_field._indicesOfNodesInArray[node]
                disp_u[idx] = new_coords[i] - (self.rpNode.coordinates + self.initialRelativePositions[i])

    def getAABB(self):
        coords = np.array([n.coordinates for n in self.surfaceNodes])
        return np.min(coords, axis=0), np.max(coords, axis=0)

    def querySurface(self, coords: np.ndarray, proximityDistance: float = None):
        """Compute signed distances and outward face normals of the rigid
        surface, in its current configuration, for an array of query points.

        Parameters
        ----------
        coords : numpy.ndarray, shape (nPoints, 3)
            The query coordinates.
        proximityDistance : float, optional
            If given, a broadphase AABB check (the current AABB inflated by
            this distance) is performed first; points outside are assigned a
            distance of ``inf`` and a zero normal without querying the surface.

        Returns
        -------
        dists : numpy.ndarray, shape (nPoints,)
            The signed distances (negative = penetration).
        normals : numpy.ndarray, shape (nPoints, 3)
            The outward unit normals of the closest faces.
        """
        if self._query_engine is None:
            if self.surface_mesh is None:
                raise RuntimeError(f"Discrete rigid body '{self.name}' has no surface_mesh to query.")
            from edelweissfe.utils.discretesurfacequery import DiscreteSurfaceQuery

            self._query_engine = DiscreteSurfaceQuery(mesh=self.surface_mesh)

        n_points = coords.shape[0]
        if proximityDistance is not None:
            curr_min, curr_max = self.getAABB()
            aabb_min = curr_min - proximityDistance
            aabb_max = curr_max + proximityDistance
            in_aabb = np.all((coords >= aabb_min) & (coords <= aabb_max), axis=1)
            active_indices = np.where(in_aabb)[0]
            if len(active_indices) == 0:
                return np.full(n_points, np.inf), np.zeros((n_points, 3))
            coords_to_query = coords[active_indices]
        else:
            coords_to_query = coords
            active_indices = np.arange(n_points)

        u_rp, R, rp_initial = self.getCurrentKinematics()
        active_dists, active_normals = self._query_engine.query(
            coords_to_query, translation=u_rp, rotation_matrix=R, rotation_center=rp_initial
        )

        dists = np.full(n_points, np.inf)
        dists[active_indices] = active_dists
        normals = np.zeros((n_points, 3))
        normals[active_indices] = active_normals
        return dists, normals

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
        nodeField = self.model.nodeFields.get(fieldName)
        if nodeField is None or "U" not in nodeField:
            return np.zeros((len(self.surfaceNodes), self.model.domainSize))

        values = nodeField["U"]
        indices = nodeField._indicesOfNodesInArray
        result = np.zeros((len(self.surfaceNodes), values.shape[1]))
        for i, node in enumerate(self.surfaceNodes):
            if node in indices:
                result[i] = values[indices[node]]
        return result
