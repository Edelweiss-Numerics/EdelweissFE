import numpy as np


class DiscreteSurfaceQuery:
    def __init__(self, filename: str = None, mesh=None, initial_offset: np.ndarray = None):
        """
        Initializes the query engine by loading an Exodus mesh or using a provided PyVista mesh.

        Uses static VTK locators and evaluators to avoid memory leaks during
        repeated distance evaluations.

        Parameters
        ----------
        filename : str, optional
            Path to the Exodus mesh file for the rigid body.
        mesh : pyvista.PolyData, optional
            A direct PyVista mesh object to use. If provided, `filename` is ignored.
        initial_offset : numpy.ndarray, optional
            A translation vector applied to the mesh points before building
            the VTK locators.
        """
        global pv, vtk
        import pyvista as pv
        import vtk

        if mesh is not None:
            self.mesh = mesh
        elif filename is not None:
            self.mesh = pv.read(filename)
            if isinstance(self.mesh, pv.MultiBlock):
                self.mesh = self.mesh.combine()
        else:
            raise ValueError("Must provide either 'filename' or 'mesh'")

        # Extract the outer surface to ensure we have a PolyData object
        self.mesh = self.mesh.extract_surface()

        # Apply the initial offset so that the contact surface sits at its
        # physical starting position (consistent with the visualization nodes).
        if initial_offset is not None:
            self.mesh.points = self.mesh.points + np.asarray(initial_offset, dtype=np.float64)

        # Ensure outward normals are computed for each cell
        self.mesh.compute_normals(cell_normals=True, point_normals=False, inplace=True)
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
        Vectorized query to compute signed distance and closest face normals for an array of points.

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
        """
        # Inverse transform query coordinates to the local (static mesh) frame
        local_coords = coords.copy()

        if translation is not None and rotation_center is not None:
            # RP current position
            rp_current = rotation_center + translation
            local_coords -= rp_current

        if rotation_matrix is not None:
            # R^T * (P - RP_current)
            local_coords = local_coords.dot(rotation_matrix)

        if rotation_center is not None:
            # P_local = RP_initial + R^T * (P_global - RP_current)
            local_coords += rotation_center

        n_points = local_coords.shape[0]
        dists = np.empty(n_points, dtype=np.float64)
        closest_cells = np.empty(n_points, dtype=np.int64)

        # Pre-allocate reusable VTK out arguments for speed
        closest_point = [0.0, 0.0, 0.0]
        sub_id = vtk.reference(0)
        dist2 = vtk.reference(0.0)

        for i in range(n_points):
            pt = local_coords[i]
            dists[i] = self.implicit_dist.EvaluateFunction(pt)

            cell_id = vtk.reference(0)
            self.locator.FindClosestPoint(pt, closest_point, cell_id, sub_id, dist2)
            closest_cells[i] = int(cell_id)

        normals = self.mesh_cell_normals[closest_cells]

        # Forward transform normals to global frame
        if rotation_matrix is not None:
            normals = normals.dot(rotation_matrix.T)

        return dists, normals
