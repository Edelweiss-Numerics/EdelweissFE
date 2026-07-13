"""
Exact mass properties (volume, mass, center of mass, and inertia tensor) of an
arbitrary closed polyhedral surface mesh, computed via divergence-theorem
integration over signed tetrahedra.

This module has no dependency on any particular mesh library (PyVista, VTK,
...) or on EdelweissFE's model/entity classes: it operates purely on raw
vertex coordinates and face connectivity, so it can be reused wherever exact
mass properties of a solid described by its boundary surface are needed
(e.g. rigid bodies, indenters, generators reading mesh files).
"""

from typing import NamedTuple, Sequence

import numpy as np


class PolyhedronMassProperties(NamedTuple):
    """Mass properties of a homogeneous solid polyhedron.

    Attributes
    ----------
    volume : float
        The enclosed volume of the polyhedron.
    mass : float
        The total mass, ``density * volume``.
    centerOfMass : numpy.ndarray, shape (3,)
        The center of mass, in the same coordinate frame as the input vertices.
    inertia : numpy.ndarray, shape (3, 3)
        The (symmetric) inertia tensor about the center of mass, expressed in
        the same (global, non-rotated) axes as the input vertices.
    """

    volume: float
    mass: float
    centerOfMass: np.ndarray
    inertia: np.ndarray


def _triangulateFaces(faces: Sequence[Sequence[int]]) -> np.ndarray:
    """Fan-triangulate a sequence of planar, convex polygonal faces.

    Parameters
    ----------
    faces : sequence of sequence of int
        Each entry lists the vertex indices of one face, in consistent
        (counter-clockwise, viewed from outside the solid) winding order.
        Faces may have different numbers of vertices (e.g. a mix of
        triangles and quadrilaterals).

    Returns
    -------
    numpy.ndarray, shape (nTriangles, 3)
        The vertex-index triples of the triangulated faces. Triangular faces
        are passed through unchanged; any other planar convex polygon is
        fan-triangulated from its first vertex.

    Raises
    ------
    ValueError
        If any face references fewer than 3 vertices.
    """
    triangles = []
    for face in faces:
        face = np.asarray(face, dtype=int)
        if face.shape[0] < 3:
            raise ValueError("Each face must reference at least 3 vertices.")
        for i in range(1, face.shape[0] - 1):
            triangles.append((face[0], face[i], face[i + 1]))
    return np.asarray(triangles, dtype=int)


def computePolyhedronMassProperties(
    vertices: np.ndarray,
    faces: Sequence[Sequence[int]],
    density: float = 1.0,
) -> PolyhedronMassProperties:
    """Compute the exact mass properties of a closed polyhedral surface mesh.

    Computes the volume, mass, center of mass, and inertia tensor of the
    homogeneous solid enclosed by a closed (watertight), consistently
    oriented polyhedral surface, via divergence-theorem integration over
    signed tetrahedra formed between the coordinate origin and each face
    [1]_. The result is exact (up to floating-point round-off) for any
    polyhedron describable by its boundary faces -- it is not limited to
    canonical primitives such as boxes, cylinders, or spheres.

    Parameters
    ----------
    vertices : numpy.ndarray, shape (nVertices, 3)
        The Cartesian coordinates of all vertices referenced by `faces`.
    faces : sequence of sequence of int
        The faces of the closed surface, each given as the (0-based) indices
        of its vertices in `vertices`, in consistent winding order (all
        faces oriented outward, or all oriented inward -- a single globally
        reversed mesh is detected and corrected automatically). Faces may be
        triangles, quadrilaterals, or other planar convex polygons.
    density : float, optional
        The (uniform) mass density of the solid. Default is 1.0, in which
        case `mass` numerically equals `volume`.

    Returns
    -------
    PolyhedronMassProperties
        A named tuple with fields `volume`, `mass`, `centerOfMass`, and
        `inertia` (the latter about the center of mass).

    Raises
    ------
    ValueError
        If the computed volume is (numerically) zero, which indicates that
        `faces` does not describe a closed, watertight surface.

    Notes
    -----
    The surface must be closed (watertight) and consistently wound for the
    result to be meaningful; this is not verified here, as doing so requires
    a topological check of the mesh (e.g. that every edge is shared by
    exactly two oppositely-wound faces), which is outside the scope of this
    purely geometric utility.

    References
    ----------
    .. [1] B. Mirtich, "Fast and accurate computation of polyhedral mass
           properties," Journal of Graphics Tools, 1(2), 1996.

    Examples
    --------
    >>> import numpy as np
    >>> vertices = np.array([[0,0,0],[1,0,0],[1,1,0],[0,1,0],
    ...                       [0,0,1],[1,0,1],[1,1,1],[0,1,1]], dtype=float)
    >>> faces = [[0,3,2,1],[4,5,6,7],[0,1,5,4],[1,2,6,5],[2,3,7,6],[3,0,4,7]]
    >>> props = computePolyhedronMassProperties(vertices, faces)
    >>> round(props.volume, 6)
    1.0
    >>> np.allclose(props.centerOfMass, [0.5, 0.5, 0.5])
    True
    """
    verts = np.asarray(vertices, dtype=float)
    triangles = _triangulateFaces(faces)

    a = verts[triangles[:, 0]]
    b = verts[triangles[:, 1]]
    c = verts[triangles[:, 2]]

    # Signed volume of the tetrahedron spanned by the origin and each face.
    signedVolumes = np.einsum("ij,ij->i", a, np.cross(b, c)) / 6.0
    volume = np.sum(signedVolumes)

    if np.isclose(volume, 0.0):
        raise ValueError(
            "The computed volume is (numerically) zero; `faces` does not "
            "describe a closed, watertight polyhedral surface."
        )

    # First moment of each tetrahedron about the origin: V * centroid, where
    # the centroid of a tetrahedron with one vertex at the origin is the
    # average of its four vertices.
    firstMoment = np.sum(signedVolumes[:, None] * (a + b + c) / 4.0, axis=0)

    # Second moment tensor of each tetrahedron about the origin, via the
    # closed-form integral over the reference simplex:
    #   int x_i x_j dV = (V / 20) * (S_i S_j + a_i a_j + b_i b_j + c_i c_j)
    # with S = a + b + c.
    S = a + b + c
    outerProducts = (
        np.einsum("fi,fj->fij", S, S)
        + np.einsum("fi,fj->fij", a, a)
        + np.einsum("fi,fj->fij", b, b)
        + np.einsum("fi,fj->fij", c, c)
    )
    secondMomentOrigin = np.einsum("f,fij->ij", signedVolumes / 20.0, outerProducts)

    if volume < 0:
        # The mesh is consistently wound, but inward (all normals point into
        # the solid); flip all accumulated quantities to correct it.
        volume = -volume
        firstMoment = -firstMoment
        secondMomentOrigin = -secondMomentOrigin

    centerOfMass = firstMoment / volume
    mass = float(density * volume)
    volume = float(volume)

    # Convert the raw second-moment tensor into the standard inertia tensor
    # about the origin, I = density * (trace(M) * Identity - M), then shift
    # to the center of mass via the parallel axis theorem.
    inertiaOrigin = density * (np.trace(secondMomentOrigin) * np.eye(3) - secondMomentOrigin)
    d = centerOfMass
    inertiaCenterOfMass = inertiaOrigin - mass * (np.dot(d, d) * np.eye(3) - np.outer(d, d))

    return PolyhedronMassProperties(
        volume=volume,
        mass=mass,
        centerOfMass=centerOfMass,
        inertia=inertiaCenterOfMass,
    )
