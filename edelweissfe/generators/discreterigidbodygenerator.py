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

from dataclasses import dataclass

import numpy as np
import pyvista as pv

from edelweissfe.generators.base.generatorbase import GeneratorBase
from edelweissfe.journal.journal import Journal
from edelweissfe.models.femodel import FEModel
from edelweissfe.points.node import Node
from edelweissfe.rigidbodies.discreterigidbody import DiscreteRigidBody
from edelweissfe.sets.nodeset import NodeSet
from edelweissfe.utils.exceptions import WrongDomain
from edelweissfe.utils.inputlanguage import InputLanguage, Module
from edelweissfe.utils.polyhedronmassproperties import computePolyhedronMassProperties
from edelweissfe.utils.schema import schemaField

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


@dataclass(frozen=True)
class DiscreteRigidBodyGeneratorSchema:
    """L2: the options this generator accepts, owned by this module and never mutated from
    outside it.

    Mirrors the ``module.addRequiredArg``/``module.addOptionalArg(...)`` declarations above
    one-for-one. The two declarations coexist while the migration is in progress; the ``Module``
    one goes away with the ``InputLanguage`` singleton in P5.

    ``filename`` is declared ``required=True`` explicitly, mirroring ``addRequiredArg`` above, but
    is still given a ``default=None`` so the schema remains constructible for the L1 constructor's
    default argument. The comma-separated vector options stay ``str`` here, as they did in the
    legacy grammar -- parsing them (:func:`_parseVector`) is the L1 constructor's job, not the
    schema's.
    """

    filename: str | None = schemaField(
        description="The file path to the surface mesh (e.g., Exodus, STL, OBJ).",
        dtype=str,
        default=None,
        required=True,
    )
    translation: str | None = schemaField(
        description="A comma-separated 3D vector to translate the mesh globally upon initialization.",
        dtype=str,
        default=None,
    )
    density: float | None = schemaField(
        description="The (uniform) mass density of the rigid body; if given, mass and rotary "
        "inertia are computed exactly from the mesh geometry.",
        dtype=float,
        default=None,
    )
    mass: float | None = schemaField(
        description="The total mass of the rigid body. Overrides the density-based computation.",
        dtype=float,
        default=None,
    )
    inertia: str | None = schemaField(
        description="A comma-separated diagonal rotary inertia [Ixx, Iyy, Izz]. Overrides the "
        "density-based computation.",
        dtype=str,
        default=None,
    )
    rpCoordinate: str | None = schemaField(
        description="A comma-separated explicit global coordinate for the reference point. "
        "Defaults to the (exact or approximate) center of mass.",
        dtype=str,
        default=None,
    )


class Generator(GeneratorBase):
    """Generates a discrete rigid body from a surface mesh file (Exodus, STL, OBJ, or any other
    format readable by PyVista).

    Loading the mesh, creating the surface/reference-point nodes, and mutating the model are all
    handled here -- mirroring how every other model-populating generator in
    EdelweissFE/EdelweissMeshfree works -- so that
    :class:`~edelweissfe.rigidbodies.discreterigidbody.DiscreteRigidBody` itself only has to deal
    with rigid body kinematics, not with how it is instantiated.
    """

    #: L2 schema declared for the L3 registry, per OptionSchemaProvider.
    schema = DiscreteRigidBodyGeneratorSchema

    def __init__(
        self,
        name: str,
        model: FEModel,
        journal: Journal,
        *,
        configuration: DiscreteRigidBodyGeneratorSchema = DiscreteRigidBodyGeneratorSchema(),
    ):
        """L1: constructible standalone, with no ``InputLanguage``/``Module``/parser involvement.
        Populates ``model`` directly (via :func:`generateDiscreteRigidBodyFromMeshFile`);
        construction *is* the generation.

        Parameters
        ----------
        name
            The identifier name for the discrete rigid body.
        model
            The model tree to populate. Mutated in place.
        journal
            The journal instance used to report progress and warnings.
        configuration
            The options this generator accepts; ``filename`` is still required, see
            :class:`DiscreteRigidBodyGeneratorSchema`.
        """
        # The rigid body's surface mesh and the node-to-discrete-rigid-body contact are inherently 3D.
        if model.domainSize != 3:
            raise WrongDomain("discreteRigidBodyGenerator is only available for 3D models.")

        translation = _parseVector(configuration.translation)
        inertia = _parseVector(configuration.inertia)
        rpCoordinate = _parseVector(configuration.rpCoordinate)

        # All of these are 3-component quantities (Cartesian vectors, or the diagonal inertia
        # [Ixx, Iyy, Izz]). Validate up front so a mistyped option fails clearly here rather than
        # silently creating wrong-sized coordinate/inertia arrays that break downstream.
        for argName, vector in (
            ("translation", translation),
            ("inertia", inertia),
            ("rpCoordinate", rpCoordinate),
        ):
            if vector is not None and vector.shape[0] != 3:
                raise WrongDomain(f"discreteRigidBodyGenerator option '{argName}' must have 3 components.")

        generateDiscreteRigidBodyFromMeshFile(
            model,
            journal,
            name=name,
            filename=configuration.filename,
            translation=translation,
            density=configuration.density,
            mass=configuration.mass,
            inertia=inertia,
            rpCoordinate=rpCoordinate,
        )


def generateDiscreteRigidBodyFromMeshFile(
    model,
    journal,
    name: str,
    filename: str,
    translation: np.ndarray = None,
    density: float = None,
    mass: float = None,
    inertia: list = None,
    rpCoordinate: np.ndarray = None,
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
    rpCoordinate : numpy.ndarray, optional
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
        if rpCoordinate is None:
            rpCoordinate = massProperties.centerOfMass

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

    if rpCoordinate is None:
        rpCoordinate = surf.center_of_mass()

    referencePoint = Node(nodeLabel, np.asarray(rpCoordinate))
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
        surf,
        nSet=surfaceNodeSetName,
        referencePoint=rpNodeSetName,
        mass=mass,
        inertia=inertia,
        facets=facets,
    )

    return rigidBody


def _readGenericSurfaceMesh(filename: str, translation: np.ndarray = None):
    """Read a surface mesh via PyVista (Exodus, STL, OBJ, VTK, ...).

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
    mesh = pv.read(filename)
    if isinstance(mesh, pv.MultiBlock):
        mesh = mesh.combine()

    surf = mesh.extract_surface(algorithm="dataset_surface")
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
