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
"""
A generator for discrete rigid bodies from surface mesh files (Exodus, STL,
OBJ, or any other format readable by PyVista).

Loading the mesh, creating the surface/reference-point nodes, and mutating
the model are all handled here -- mirroring how every other model-populating
generator in EdelweissFE/EdelweissMeshfree works -- so that
:class:`~edelweissfe.rigidbodies.discreterigidbody.DiscreteRigidBody` itself
only has to deal with rigid body kinematics, not with how it is instantiated.
"""

import numpy as np

from edelweissfe.points.node import Node
from edelweissfe.rigidbodies.discreterigidbody import DiscreteRigidBody
from edelweissfe.sets.nodeset import NodeSet
from edelweissfe.utils.caseinsensitivedict import CaseInsensitiveDict
from edelweissfe.utils.exceptions import WrongDomain
from edelweissfe.utils.inputlanguage import InputLanguage, Module
from edelweissfe.utils.misc import (
    caseInsensitiveKwargsChecker,
    castKwargsValuesAndAddDefaults,
)
from edelweissfe.utils.polyhedronmassproperties import computePolyhedronMassProperties

module = Module(
    "discreteRigidBodyGenerator",
    "Generates a discrete rigid body from a surface mesh file (Exodus, STL, OBJ, or any other format readable by PyVista).",
)

inputLanguage = InputLanguage()

keyword = "modelGenerator"
if keyword in inputLanguage:
    inputLanguage[keyword].addModule(module)

module.addRequiredArg("filename", "The file path to the surface mesh (e.g., Exodus, STL, OBJ).", str)

module.addOptionalArg(
    "translation",
    "A comma-separated 3D vector to translate the mesh globally upon initialization.",
    str,
    None,
)
module.addOptionalArg(
    "density",
    "The (uniform) mass density of the rigid body; if given, mass and rotary inertia are computed exactly "
    "from the mesh geometry.",
    float,
    None,
)
module.addOptionalArg("mass", "The total mass of the rigid body. Overrides the density-based computation.", float, None)
module.addOptionalArg(
    "inertia",
    "A comma-separated diagonal rotary inertia [Ixx, Iyy, Izz]. Overrides the density-based computation.",
    str,
    None,
)
module.addOptionalArg("initialVelocity", "A comma-separated initial velocity vector [vx, vy, vz].", str, None)
module.addOptionalArg(
    "rpCoordinate",
    "A comma-separated explicit global coordinate for the reference point. Defaults to the (exact or "
    "approximate) center of mass.",
    str,
    None,
)

documentation = [module]


def _parseVector(value: str):
    return np.fromstring(value, sep=",", dtype=np.double) if value is not None else None


@caseInsensitiveKwargsChecker([kw.name for kw in module.requiredArgs], [kw.name for kw in module.optionalArgs])
@castKwargsValuesAndAddDefaults(module)
def generateModelData(generatorDefinition: dict, model, journal, *args, **kwargs) -> dict:
    """Entry point for the ``*modelGenerator, generator=discreteRigidBodyGenerator`` input file keyword.

    Thin wrapper around :func:`generateDiscreteRigidBodyFromMeshFile`, translating the ``.inp``
    keyword arguments (comma-separated vector strings) to the native Python types it expects.
    """

    kwargs = CaseInsensitiveDict(kwargs)

    # The rigid body's surface mesh and the node-to-discrete-rigid-body contact are inherently 3D.
    if model.domainSize != 3:
        raise WrongDomain("discreteRigidBodyGenerator is only available for 3D models.")

    name = generatorDefinition.get("name", "discreteRigidBody")

    translation = _parseVector(kwargs["translation"])
    inertia = _parseVector(kwargs["inertia"])
    initial_velocity = _parseVector(kwargs["initialVelocity"])
    rp_coordinate = _parseVector(kwargs["rpCoordinate"])

    # All of these are 3-component quantities (Cartesian vectors, or the diagonal inertia
    # [Ixx, Iyy, Izz]). Validate up front so a mistyped option fails clearly here rather than
    # silently creating wrong-sized coordinate/inertia arrays that break downstream.
    for argName, vector in (
        ("translation", translation),
        ("inertia", inertia),
        ("initialVelocity", initial_velocity),
        ("rpCoordinate", rp_coordinate),
    ):
        if vector is not None and vector.shape[0] != 3:
            raise WrongDomain(f"discreteRigidBodyGenerator option '{argName}' must have 3 components.")

    generateDiscreteRigidBodyFromMeshFile(
        model,
        journal,
        name=name,
        filename=kwargs["filename"],
        translation=translation,
        density=kwargs["density"],
        mass=kwargs["mass"],
        inertia=inertia,
        initial_velocity=initial_velocity,
        rp_coordinate=rp_coordinate,
    )

    return model


def generateDiscreteRigidBodyFromMeshFile(
    model,
    journal,
    name: str,
    filename: str,
    translation: np.ndarray = None,
    density: float = None,
    mass: float = None,
    inertia: list = None,
    initial_velocity: list = None,
    rp_coordinate: np.ndarray = None,
    start_label: int = None,
) -> DiscreteRigidBody:
    """Create a :class:`DiscreteRigidBody` from a surface mesh file and register it in the model.

    Reads a surface mesh (Exodus/NetCDF, or anything else PyVista can read),
    creates the surface and reference-point (RP) nodes and node sets in
    `model`, computes mass and rotary inertia from the mesh geometry if a
    `density` is given, and instantiates the corresponding
    :class:`~edelweissfe.rigidbodies.discreterigidbody.DiscreteRigidBody`.

    Parameters
    ----------
    model : edelweissfe.models.femodel.FEModel
        The model to populate.
    journal : edelweissfe.journal.journal.Journal
        The journal instance used to report progress and warnings.
    name : str
        The identifier name for the discrete rigid body.
    filename : str
        The file path to the surface mesh (e.g., Exodus, STL, OBJ).
    translation : numpy.ndarray, optional
        A 3D vector to translate the mesh globally upon initialization.
    density : float, optional
        The (uniform) mass density of the rigid body. If given, the mass and
        rotary inertia are computed exactly from the mesh geometry via
        :func:`~edelweissfe.utils.polyhedronmassproperties.computePolyhedronMassProperties`.
        Ignored if not given -- in that case `mass`/`inertia` are used as-is
        (both `None` by default, giving a purely kinematically driven rigid
        body with no dynamic response).
    mass : float, optional
        The total mass of the rigid body. Overrides the density-based
        computation.
    inertia : list, optional
        The diagonal rotary inertia `[Ixx, Iyy, Izz]`. Overrides the
        density-based computation. Note that
        :class:`~edelweissfe.elements.pointmass.PointMass` only supports a
        diagonal (axis-aligned) rotary inertia -- see Notes.
    initial_velocity : list, optional
        The initial velocity vector [vx, vy, vz].
    rp_coordinate : numpy.ndarray, optional
        The explicit global coordinates for the reference point. If `None`,
        it defaults to the exact center of mass (if `density` was given) or
        otherwise the mesh's approximate center of mass.
    start_label : int, optional
        The starting label for newly generated nodes. Defaults to one past
        the highest existing node label in `model`.

    Returns
    -------
    DiscreteRigidBody
        The created discrete rigid body. It is also registered in
        `model.rigidBodies[name]`.

    Notes
    -----
    The exact inertia tensor computed from the mesh geometry generally has
    non-zero off-diagonal (product-of-inertia) terms unless the body's
    principal axes happen to be aligned with the global axes. Only the
    diagonal is passed on, since the underlying
    :class:`~edelweissfe.elements.pointmass.PointMass` element does not
    support a fully populated inertia tensor. A warning is issued via
    `journal` if the discarded off-diagonal terms are not negligible.
    """

    journal.message(f"Reading discrete rigid body surface mesh from: {filename}", "discreteRigidBody", 1)

    filenameLower = filename.lower()
    if filenameLower.endswith(".exo") or filenameLower.endswith(".nc"):
        points, faces, elementTypes, surf = _readExodusSurfaceMesh(filename, translation)
    else:
        points, faces, elementTypes, surf = _readGenericSurfaceMesh(filename, translation)

    if density is not None:
        massProperties = computePolyhedronMassProperties(points, faces, density)

        offDiagonal = massProperties.inertia - np.diag(np.diag(massProperties.inertia))
        offDiagonalMagnitude = np.max(np.abs(offDiagonal))
        diagonalMagnitude = np.max(np.abs(np.diag(massProperties.inertia)))
        if diagonalMagnitude > 0.0 and offDiagonalMagnitude > 1e-3 * diagonalMagnitude:
            journal.message(
                f"Discrete rigid body '{name}': the exact inertia tensor has non-negligible "
                "off-diagonal (product-of-inertia) terms, but only its diagonal is used, since "
                "PointMass only supports axis-aligned rotary inertia. Results will be approximate "
                "unless the body's principal axes are aligned with the global axes.",
                "discreteRigidBody",
                0,
            )

        if mass is None:
            mass = massProperties.mass
        if inertia is None:
            inertia = list(np.diag(massProperties.inertia))
        if rp_coordinate is None:
            rp_coordinate = massProperties.centerOfMass

    journal.message(f"Discrete rigid body '{name}': {len(points)} surface nodes, mass={mass}.", "discreteRigidBody", 1)

    rigidNodes = []
    nodeLabel = start_label if start_label is not None else (max(model.nodes.keys()) + 1 if model.nodes else 1)
    for point in points:
        node = Node(nodeLabel, point.copy())
        model.nodes[node.label] = node
        rigidNodes.append(node)
        nodeLabel += 1

    surfaceNodeSetName = f"{name}_surface_nodes"
    model.nodeSets[surfaceNodeSetName] = NodeSet(surfaceNodeSetName, rigidNodes)

    facets = [
        {"type": elementType, "nodes": [rigidNodes[idx] for idx in face]}
        for face, elementType in zip(faces, elementTypes)
    ]

    if rp_coordinate is None:
        rp_coordinate = surf.center_of_mass()

    referencePoint = Node(nodeLabel, np.asarray(rp_coordinate))
    model.nodes[referencePoint.label] = referencePoint

    rpNodeSetName = f"{name}_rp"
    model.nodeSets[rpNodeSetName] = NodeSet(rpNodeSetName, [referencePoint])

    if "all" in model.nodeSets:
        allNodes = list(model.nodeSets["all"])
        allNodes.extend(rigidNodes)
        allNodes.append(referencePoint)
        model.nodeSets["all"] = NodeSet("all", allNodes)

    rigidBody = DiscreteRigidBody(
        name,
        model,
        nSet=surfaceNodeSetName,
        referencePoint=rpNodeSetName,
        mass=mass,
        inertia=inertia,
        initial_velocity=initial_velocity,
        facets=facets,
    )
    rigidBody.surface_mesh = surf

    return rigidBody


def _readExodusSurfaceMesh(filename: str, translation: np.ndarray = None):
    """Read a surface mesh from an Exodus/NetCDF file.

    Parameters
    ----------
    filename : str
        The file path to the Exodus/NetCDF surface mesh.
    translation : numpy.ndarray, optional
        A 3D vector to translate the mesh globally.

    Returns
    -------
    points : numpy.ndarray, shape (nNodes, 3)
        The (translated) vertex coordinates.
    faces : list of numpy.ndarray
        The vertex-index list of each face.
    elementTypes : list of str
        The EdelweissFE/Ensight element type ("tria3" or "quad4") of each face.
    surf : pyvista.PolyData
        The assembled surface, with outward face normals computed.
    """
    import netCDF4
    import pyvista as pv

    nc = netCDF4.Dataset(filename, "r")
    try:
        x = nc.variables["coordx"][:]
        y = nc.variables["coordy"][:]
        z = nc.variables["coordz"][:] if "coordz" in nc.variables else np.zeros_like(x)
        points = np.column_stack((x, y, z))

        if translation is not None:
            points = points + np.asarray(translation)

        if "connect1" not in nc.variables:
            raise ValueError("No connect1 variable found in NetCDF/Exodus file.")

        connectivity = nc.variables["connect1"][:] - 1  # 0-indexed
    finally:
        nc.close()

    numElements, numNodesPerElement = connectivity.shape

    faces = []
    elementTypes = []
    pyvistaFaces = []
    for i in range(numElements):
        face = connectivity[i]
        faces.append(face)

        pyvistaFaces.append(numNodesPerElement)
        pyvistaFaces.extend(face)

        if numNodesPerElement == 3:
            elementTypes.append("tria3")
        elif numNodesPerElement == 4:
            elementTypes.append("quad4")
        else:
            raise ValueError(f"Unsupported number of nodes {numNodesPerElement} for surface mesh.")

    surf = pv.PolyData(points, np.array(pyvistaFaces))
    surf.compute_normals(cell_normals=True, point_normals=False, inplace=True)

    return points, faces, elementTypes, surf


def _readGenericSurfaceMesh(filename: str, translation: np.ndarray = None):
    """Read a surface mesh via PyVista (STL, OBJ, VTK, ...).

    Parameters
    ----------
    filename : str
        The file path to the surface mesh.
    translation : numpy.ndarray, optional
        A 3D vector to translate the mesh globally.

    Returns
    -------
    points : numpy.ndarray, shape (nNodes, 3)
        The (translated) vertex coordinates.
    faces : list of numpy.ndarray
        The vertex-index list of each face.
    elementTypes : list of str
        The EdelweissFE/Ensight element type ("tria3" or "quad4") of each face.
    surf : pyvista.PolyData
        The extracted surface, with outward face normals computed.
    """
    import pyvista as pv

    mesh = pv.read(filename)
    if isinstance(mesh, pv.MultiBlock):
        mesh = mesh.combine()

    surf = mesh.extract_surface()
    surf.compute_normals(cell_normals=True, point_normals=False, inplace=True)

    # points must come from surf, not mesh, since extract_surface() can drop/renumber
    # points relative to the input mesh - faces below index into surf's point array.
    points = surf.points.copy()
    if translation is not None:
        points = points + np.asarray(translation)
        surf.points = points

    # PyVista renamed PolyData.cells -> PolyData.faces for this flat VTK cell-array representation.
    cells = surf.faces
    faces = []
    elementTypes = []

    i = 0
    cellIndex = 0
    while i < len(cells):
        n = cells[i]
        face = cells[i + 1 : i + 1 + n]
        faces.append(face)

        vtkType = surf.GetCellType(cellIndex)
        # 5 = VTK_TRIANGLE, 9 = VTK_QUAD, 7 = VTK_POLYGON
        if vtkType == 5:
            elementTypes.append("tria3")
        elif vtkType == 9:
            elementTypes.append("quad4")
        elif vtkType == 7:
            if n == 3:
                elementTypes.append("tria3")
            elif n == 4:
                elementTypes.append("quad4")
            else:
                raise ValueError(f"Unsupported VTK_POLYGON with {n} nodes for discrete rigid body.")
        else:
            if n == 3:
                elementTypes.append("tria3")
            elif n == 4:
                elementTypes.append("quad4")
            else:
                raise ValueError(f"Unsupported VTK cell type {vtkType} with {n} nodes.")

        i += 1 + n
        cellIndex += 1

    return np.asarray(points), faces, elementTypes, surf
