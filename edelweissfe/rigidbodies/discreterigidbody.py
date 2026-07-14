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
                R = self.rotationMatrixFromPseudoVector(theta)

        # In EdelweissFE, the RP node's coordinates are the *reference* (initial) coordinates.
        return u_rp, R, self.rpNode.coordinates

    def _currentAndReferenceSurfaceCoordinates(self, kinematics: tuple = None):
        """Compute the current and reference world coordinates of every surface node.

        The surface nodes are not independent degrees of freedom themselves -- they
        carry no FieldVariables and this never registers or writes to any NodeField
        -- but by default the reference point's kinematics *are* read from the
        NodeFields of its ``displacement``/``rotation`` fields, via
        :meth:`getCurrentKinematics`.

        Parameters
        ----------
        kinematics : tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray], optional
            An explicit ``(u_rp, R, rp_initial)`` triple (matching the return value of
            :meth:`getCurrentKinematics`) to use instead of the (possibly stale, since
            NodeFields only reflect the last *converged* increment) current state. Pass
            this when the caller already holds a fresher RP kinematic state -- e.g. a
            contact constraint mid-Newton-iteration, which has direct access to the
            current trial solution vector.

        Returns
        -------
        tuple[numpy.ndarray, numpy.ndarray]
            The current and reference coordinates, both of shape ``(nSurfaceNodes, domainSize)``.
        """
        u_rp, R, rp_initial = kinematics if kinematics is not None else self.getCurrentKinematics()
        d = self.domainSize
        currentCoords = (rp_initial + u_rp) + self.initialRelativePositions.dot(R[:d, :d].T)
        referenceCoords = rp_initial + self.initialRelativePositions
        return currentCoords, referenceCoords

    def updateKinematics(self, timeStep=None):
        currentCoords, _ = self._currentAndReferenceSurfaceCoordinates()

        # Deviating from the usual EdelweissFE convention, the surface nodes'
        # coordinates hold the *current* configuration: the output managers
        # write the transient geometry of the moving body from them, and the
        # AABB proximity checks of the contact constraints rely on them. The
        # reference configuration is retained via the RP node's coordinates
        # and initialRelativePositions. The surface nodes are not degrees of
        # freedom themselves -- they carry no FieldVariables and are not part
        # of any NodeField -- so nothing else needs to be updated here.
        for i, node in enumerate(self.surfaceNodes):
            node.coordinates[:] = currentCoords[i]

    def getAABB(self, kinematics: tuple = None):
        """Axis-aligned bounding box of the surface nodes' current positions.

        Parameters
        ----------
        kinematics : tuple, optional
            See :meth:`_currentAndReferenceSurfaceCoordinates`. If not given, uses
            the surface nodes' own (last-updated-by :meth:`updateKinematics`)
            ``coordinates`` directly rather than recomputing them, as a fast path
            for the common case.
        """
        if kinematics is None:
            coords = np.array([n.coordinates for n in self.surfaceNodes])
        else:
            coords, _ = self._currentAndReferenceSurfaceCoordinates(kinematics)
        return np.min(coords, axis=0), np.max(coords, axis=0)

    def querySurface(self, coords: np.ndarray, proximityDistance: float = None, kinematics: tuple = None):
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
        kinematics : tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray], optional
            An explicit ``(u_rp, R, rp_initial)`` triple to use for the rigid body's
            pose instead of :meth:`getCurrentKinematics`'s (last-*converged*-increment)
            NodeField state. A caller that already holds a fresher/current RP
            kinematic state -- e.g. a contact constraint mid-Newton-iteration, which
            has direct access to the trial solution vector -- should pass it here so
            the query is evaluated for a pose consistent with the rest of that
            caller's own current state (both this narrow-phase query and the
            broadphase AABB check use it consistently).

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
            curr_min, curr_max = self.getAABB(kinematics)
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

        u_rp, R, rp_initial = kinematics if kinematics is not None else self.getCurrentKinematics()
        active_dists, active_normals = self._query_engine.query(
            coords_to_query, translation=u_rp, rotation_matrix=R, rotation_center=rp_initial
        )

        dists = np.full(n_points, np.inf)
        dists[active_indices] = active_dists
        normals = np.zeros((n_points, 3))
        normals[active_indices] = active_normals
        return dists, normals

    @staticmethod
    def rotationMatrixFromPseudoVector(theta: np.ndarray) -> np.ndarray:
        """Convert a 3-component rotation pseudo-vector to a 3x3 rotation matrix
        via the exponential map (Rodrigues' formula)."""
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
        """Compute a field's values on the surface (visualization) nodes.

        The surface nodes carry no FieldVariables of their own -- they are fully
        determined by the reference point's already-solved kinematics -- so this
        is computed directly rather than looked up from a NodeField.

        Parameters
        ----------
        fieldName
            The name of the field. Currently only ``"displacement"`` is supported;
            any other field returns zeros.
        """
        if fieldName != "displacement":
            return np.zeros((len(self.surfaceNodes), self.model.domainSize))

        currentCoords, referenceCoords = self._currentAndReferenceSurfaceCoordinates()
        return currentCoords - referenceCoords
