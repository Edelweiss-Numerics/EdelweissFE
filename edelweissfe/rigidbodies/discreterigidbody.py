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

from typing import Any, Dict, List

import numpy as np

from edelweissfe.rigidbodies.rigidbody import RigidBody
from edelweissfe.utils.caseinsensitivedict import CaseInsensitiveDict
from edelweissfe.utils.rotations import rightJacobianSO3, rotationMatrixFromPseudoVector


class DiscreteRigidBody(RigidBody):
    """
    Discrete Rigid Body entity.

    It represents a rigid body defined by a discrete surface mesh. Instances
    are normally created via
    :func:`~edelweissfe.generators.discreterigidbodygenerator.generateDiscreteRigidBodyFromMeshFile`,
    which handles loading the mesh and creating the surface/reference-point
    nodes -- this class itself only deals with the rigid body's kinematics.
    """

    def __init__(self, name, model, surfaceMesh, *args, **kwargs):
        super().__init__(name, model)
        self.model.rigidBodies[self.name] = self

        kwargs = CaseInsensitiveDict(kwargs)

        self.surfaceNodes = list(model.nodeSets[kwargs["nSet"]])
        rpNodeSet = model.nodeSets[kwargs["referencePoint"]]

        if len(rpNodeSet) > 1:
            raise ValueError("Reference point set must contain exactly one node!")

        self.rpNode = list(rpNodeSet)[0]
        self.domainSize = model.domainSize

        self.surfaceMesh = surfaceMesh
        self._outerSurface = None
        self._referenceTriangles = None

        # Facets replace DiscreteRigidElement.
        # Format: [{'type': 'tria3', 'nodes': [node1, node2, node3]}, ...]
        self.facets = kwargs.get("facets", [])

        # The surface nodes' coordinates follow the body (see updateKinematics), so keep their reference
        # coordinates for the topology fingerprint.
        self.referenceSurfaceCoordinates = np.array([n.coordinates for n in self.surfaceNodes], dtype=float)

        # Precompute initial relative positions of surface nodes w.r.t. the RP
        self.initialRelativePositions = np.array([n.coordinates - self.rpNode.coordinates for n in self.surfaceNodes])

        self.mass = kwargs.get("mass")
        self.inertia = kwargs.get("inertia")

        # Abstract the PointMass element
        self.pointMassElement = None
        if self.mass is not None:
            from edelweissfe.elements.pointmass import PointMass

            # Element numbers come from the model allocator (topology pipeline).
            (el_num,) = model.topology.reserveElementNumbers(1)
            self.pointMassElement = PointMass(el_num, [self.rpNode], model, self.mass, self.inertia)
            model.createElement(self.pointMassElement)

    def referenceCoordinatesOfMovedNodes(self) -> dict:
        """The reference coordinates of the surface nodes, which :meth:`updateKinematics` moves.

        Returns
        -------
        dict
            The reference coordinates, keyed by surface node label.
        """
        return dict(zip((node.label for node in self.surfaceNodes), self.referenceSurfaceCoordinates))

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
        if disp_field is None or self.rpNode not in disp_field.nodes:
            raise RuntimeError(
                f"Discrete rigid body '{self.name}': reference point node is not part of the model's "
                "'displacement' NodeField -- the model may not be fully initialized yet."
            )
        u_rp = disp_field["U"][disp_field.indexOfNode(self.rpNode)].copy()

        R = np.eye(3)
        rot_field = self.model.nodeFields.get("rotation")
        if rot_field is not None and self.rpNode in rot_field.nodes:
            theta = rot_field["U"][rot_field.indexOfNode(self.rpNode)]
            R = rotationMatrixFromPseudoVector(theta)

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
        currentCoords = (rp_initial + u_rp) + self.initialRelativePositions.dot(R.T)
        referenceCoords = rp_initial + self.initialRelativePositions
        return currentCoords, referenceCoords

    def updateKinematics(self, timeStep=None):
        currentCoords, _ = self._currentAndReferenceSurfaceCoordinates()

        # Unlike the usual EdelweissFE convention, the surface nodes' coordinates hold the *current*
        # configuration (used by output); they carry no field variables.
        for i, node in enumerate(self.surfaceNodes):
            node.coordinates[:] = currentCoords[i]

    def poseFromDofs(self, rpDisplacement: np.ndarray, rpRotation: np.ndarray) -> tuple:
        """The rigid body pose for given reference point DOF values.

        A contact constraint evaluates the pose from its own current trial solution, not from the
        last converged NodeField state that :meth:`getCurrentKinematics` reads.

        Parameters
        ----------
        rpDisplacement
            The displacement of the reference point.
        rpRotation
            The rotation pseudo-vector of the reference point.

        Returns
        -------
        tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]
            The current reference point position, the rotation matrix :math:`R`, and
            ``dPhysicalSpin_dTheta`` :math:`= R \\, J_r(\\boldsymbol{\\theta})`, which maps a perturbation
            of the stored pseudo-vector DOF onto the spatial rotation it produces (the identity only
            at zero rotation), see :func:`~edelweissfe.utils.rotations.rightJacobianSO3`.
        """
        rotationMatrix = rotationMatrixFromPseudoVector(rpRotation)
        dPhysicalSpin_dTheta = rotationMatrix @ rightJacobianSO3(rpRotation)
        currentRpPosition = self.rpNode.coordinates + rpDisplacement
        return currentRpPosition, rotationMatrix, dPhysicalSpin_dTheta

    def outerSurface(self) -> tuple[Any, np.ndarray]:
        """The outer surface of the rigid body's mesh and the outward unit normals of its cells, made on first use.

        The outward normals are established from the mesh topology
        (``compute_normals(..., auto_orient_normals=True)``) rather than from the face winding of the file, so they are
        reliable for any correctly closed mesh.

        Returns
        -------
        tuple[pyvista.PolyData, numpy.ndarray]
            The outer surface, and the outward unit normals of its cells, of shape (nCells, 3).
        """
        if self._outerSurface is None:
            surface = self.surfaceMesh.extract_surface(algorithm="dataset_surface")
            surface.compute_normals(cell_normals=True, point_normals=False, auto_orient_normals=True, inplace=True)
            self._outerSurface = (surface, np.array(surface.cell_normals))
        return self._outerSurface

    def referenceTriangles(self) -> tuple[np.ndarray, np.ndarray]:
        """The triangles of the rigid surface and their outward unit normals, in the reference
        configuration (the body frame).

        The rigid surface must be closed, so that outside and inside are well defined, and must not
        contain triangles of zero area, whose normal is undefined. Both are checked here, since a
        violation otherwise shows up only as unphysical contact forces much later in a run.

        Returns
        -------
        tuple[numpy.ndarray, numpy.ndarray]
            The triangle vertex coordinates, of shape (nTriangles, 3, 3), and the outward unit
            normals, of shape (nTriangles, 3).
        """
        if self._referenceTriangles is None:
            surface, outwardNormals = self.outerSurface()
            if not surface.is_all_triangles:
                raise ValueError(f"Discrete rigid body '{self.name}': the surface must consist of triangles only.")
            if surface.n_open_edges > 0:
                raise ValueError(
                    f"Discrete rigid body '{self.name}': the surface is not closed "
                    f"({surface.n_open_edges} open edges), so its outward normals are undefined."
                )
            triangleCoordinates = np.asarray(surface.points)[surface.regular_faces]
            doubleAreas = np.linalg.norm(
                np.cross(
                    triangleCoordinates[:, 1] - triangleCoordinates[:, 0],
                    triangleCoordinates[:, 2] - triangleCoordinates[:, 0],
                ),
                axis=1,
            )
            degenerate = np.flatnonzero(doubleAreas <= 1e-12 * doubleAreas.max())
            if degenerate.size:
                raise ValueError(
                    f"Discrete rigid body '{self.name}': {degenerate.size} triangle(s) of zero area "
                    f"(first: {degenerate[0]}), whose normals are undefined."
                )
            self._referenceTriangles = (triangleCoordinates, outwardNormals)
        return self._referenceTriangles

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
            The name of the field. Currently only ``"displacement"`` is supported.

        Raises
        ------
        ValueError
            If ``fieldName`` is not ``"displacement"``.
        """
        if fieldName != "displacement":
            raise ValueError(
                f"Discrete rigid body '{self.name}': unsupported field '{fieldName}' for visualization "
                "output; only 'displacement' is currently supported."
            )

        currentCoords, referenceCoords = self._currentAndReferenceSurfaceCoordinates()
        return currentCoords - referenceCoords
