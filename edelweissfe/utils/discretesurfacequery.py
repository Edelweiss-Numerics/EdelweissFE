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

import numpy as np


class DiscreteSurfaceQuery:
    def __init__(self, mesh):
        """
        Initializes the query engine for a given PyVista mesh.

        Uses static VTK locators and evaluators to avoid memory leaks during
        repeated distance evaluations.

        Parameters
        ----------
        mesh : pyvista.PolyData
            The PyVista mesh to query.
        """
        import vtk

        self._vtk = vtk

        self.mesh = mesh

        # Extract the outer surface to ensure we have a PolyData object
        self.mesh = self.mesh.extract_surface(algorithm="dataset_surface")

        # Ensure outward normals are computed for each cell. auto_orient_normals determines the
        # true outward direction from the mesh topology itself (requires a closed, manifold
        # surface -- true for any solid rigid body's contact surface), rather than trusting the
        # source file's face winding order, which is not otherwise guaranteed to be outward.
        self.mesh.compute_normals(cell_normals=True, point_normals=False, auto_orient_normals=True, inplace=True)
        self.mesh_cell_normals = np.array(self.mesh.cell_normals)

        # Initialize implicit distance evaluator once to avoid memory leaks
        self.implicit_dist = vtk.vtkImplicitPolyDataDistance()
        self.implicit_dist.SetInput(self.mesh)

        # Initialize cell locator once to avoid memory leaks
        self.locator = vtk.vtkStaticCellLocator()
        self.locator.SetDataSet(self.mesh)
        self.locator.BuildLocator()

    def query(
        self,
        coords: np.ndarray,
        translation: np.ndarray = None,
        rotation_matrix: np.ndarray = None,
        rotation_center: np.ndarray = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Batched query to compute signed distance and closest face normals for an array of points.

        The coordinate transform and result assembly are vectorized, but the actual
        VTK distance/closest-cell evaluation loops over points one at a time -- this
        is a potential performance bottleneck for very large point counts.

        Parameters
        ----------
        coords : numpy.ndarray
            Array of shape (N, 3) containing the query coordinates.
        translation : numpy.ndarray, optional
            A translation vector of the rigid body RP.
        rotation_matrix : numpy.ndarray, optional
            A 3x3 rotation matrix of the rigid body.
        rotation_center : numpy.ndarray, optional
            The center of rotation (initial position of the RP).

        Returns
        -------
        dists : numpy.ndarray
            Array of shape (N,) containing the signed distance (negative = inside/penetration).
        normals : numpy.ndarray
            Array of shape (N, 3) containing the outward normal vectors on the closest faces.

        .. note::
            Note on the rotation convention: both ``localCoords``/``normals`` here and e.g.
            :meth:`~edelweissfe.rigidbodies.discreterigidbody.DiscreteRigidBody.getCurrentKinematics`
            use the *same* forward convention: a body-frame vector ``v`` is mapped to the world frame
            via ``v.dot(R.T)``. Because these are *row* vectors (shape ``(N, 3)``), ``v.dot(M)`` computes
            ``M.T @ v`` per row, not ``M @ v`` -- so ``v.dot(R.T)`` is ``R @ v`` (forward: body -> world),
            and ``v.dot(R)`` is ``R.T @ v`` (inverse: world -> body). This is the opposite of what a
            naive read of ``.dot(rotation_matrix)`` vs. ``.dot(rotation_matrix.T)`` suggests -- verified
            numerically against a known rotated mesh before relying on it, since it is easy to get backwards
            (do not "fix" the transpose here without re-deriving/re-checking this convention first).
        """
        # Inverse transform query coordinates to the local (static mesh) frame
        localCoords = coords.copy()

        if translation is not None and rotation_center is not None:
            # RP current position
            rp_current = rotation_center + translation
            localCoords -= rp_current

        if rotation_matrix is not None:
            # Inverse rotation R^T * (P - RP_current) -- see the rotation-convention note above;
            # .dot(rotation_matrix), NOT .dot(rotation_matrix.T), is the correct inverse here.
            localCoords = localCoords.dot(rotation_matrix)

        if rotation_center is not None:
            # P_local = RP_initial + R^T * (P_global - RP_current)
            localCoords += rotation_center

        n_points = localCoords.shape[0]
        dists = np.empty(n_points, dtype=np.float64)
        closest_cells = np.empty(n_points, dtype=np.int64)

        # Pre-allocate reusable VTK out arguments for speed
        closest_point = [0.0, 0.0, 0.0]
        sub_id = self._vtk.reference(0)
        dist2 = self._vtk.reference(0.0)

        for i in range(n_points):
            pt = localCoords[i]
            dists[i] = self.implicit_dist.EvaluateFunction(pt)

            cell_id = self._vtk.reference(0)
            self.locator.FindClosestPoint(pt, closest_point, cell_id, sub_id, dist2)
            closest_cells[i] = int(cell_id)

        normals = self.mesh_cell_normals[closest_cells]

        # Forward rotation R * n (body -> world frame); see "Note on the rotation convention" in
        # this method's docstring for why .dot(rotation_matrix.T), not .dot(rotation_matrix), is
        # the correct forward transform for these row vectors.
        if rotation_matrix is not None:
            normals = normals.dot(rotation_matrix.T)

        return dists, normals
