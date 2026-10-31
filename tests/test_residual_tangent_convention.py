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
"""The residual and tangent convention, checked entity by entity.

Every element, load and constraint adds a residual contribution ``P`` and its tangent ``K = dP/dU``: an element its
internal forces, a load the negative of the external load, a constraint the negative of the force it exerts on its
nodes. The solvers then form ``R = f_ext - f_int`` as the negative of the summed contributions.

- **Constraints**: ``K`` is compared with central differences of ``P``, at a state where every switch (contact, stick
  or slip) stays on one side under the perturbation. A ``P`` of the wrong sign fails this as surely as a wrong ``K``.
  The two integrated surface contact constraints are checked the same way in their own modules,
  ``test_surfacetodeformablesurfacepenalty.py`` and ``test_surfacetodiscreterigidbodypenalty.py``.
- **Element loads**: a dead load has no tangent, so the finite-difference check cannot see its sign. Its residual
  contribution is compared with the external load integrated independently instead. A load that follows the
  deformation (a pressure on an updated-Lagrangian element) has a tangent, and gets the finite-difference check too.
"""

from pathlib import Path

import numpy as np
import pytest
from _residualtangent import (  # noqa: E402  (tests/ is on sys.path)
    centralDifferences,
    finiteDifferenceTangent,
    localVector,
    residualContributionAndTangent,
)

from edelweissfe.config.phenomena import getFieldSize
from edelweissfe.elements.pointmass import PointMass
from edelweissfe.helpers.inputfilehelpers import fillFEModelFromInputFile
from edelweissfe.journal.journal import Journal
from edelweissfe.models.femodel import FEModel
from edelweissfe.points.node import Node
from edelweissfe.timesteppers.timestep import TimeStep
from edelweissfe.utils.inputfileparser import parseInputFile

_MATERIAL = """
*material, name=LinearElastic, id=linearelastic, provider=edelweiss
100.0, 0.3
"""

#: Two by two plane-strain quads on the unit square.
_PLANE_BLOCK = (
    _MATERIAL
    + """
*modelGenerator, generator=planeRectQuad, name=gen
x0=0, y0=0, l=1.0, h=1.0, nX=2, nY=2, elType=CPE4, elProvider=edelweiss

*section, name=section1, thickness=1.0, material=linearelastic, type=plane
gen_all
"""
)

#: One hexahedron on the unit cube.
_SOLID_BLOCK = (
    _MATERIAL
    + """
*modelGenerator, generator=boxGen, name=gen
x0=0, y0=0, z0=0, lX=1.0, lY=1.0, lZ=1.0, nX=1, nY=1, nZ=1, elType=C3D8, elProvider=edelweiss

*section, name=section1, material=linearelastic, type=solid
gen_all
"""
)

#: A plane block above a wider one, with a gap of 0.05; the upper block's slave nodes project into the interior of
#: the lower block's master facets.
_TWO_PLANE_BLOCKS = (
    _MATERIAL
    + """
*modelGenerator, generator=planeRectQuad, name=lower
x0=0, y0=0, l=2.0, h=1.0, nX=2, nY=1, elType=CPE4, elProvider=edelweiss

*modelGenerator, generator=planeRectQuad, name=upper
x0=0.7, y0=1.05, l=0.6, h=1.0, nX=1, nY=1, elType=CPE4, elProvider=edelweiss

*modelGenerator, generator=surfaceElementGenerator, name=slaveFacets
surface=upper_bottom
name=slaveSurf

*modelGenerator, generator=surfaceElementGenerator, name=masterFacets
surface=lower_top
name=masterSurf

*section, name=section1, thickness=1.0, material=linearelastic, type=plane
lower_all
upper_all
"""
)

#: The 3D counterpart: the upper block's bottom corners lie off both diagonals of the lower block's top face, so each
#: projects into the interior of one master triangle.
_TWO_SOLID_BLOCKS = (
    _MATERIAL
    + """
*modelGenerator, generator=boxGen, name=lower
x0=0, y0=0, z0=0, lX=1.0, lY=1.0, lZ=1.0, nX=1, nY=1, nZ=1, elType=C3D8, elProvider=edelweiss

*modelGenerator, generator=boxGen, name=upper
x0=0.3, y0=1.05, z0=0.15, lX=0.4, lY=0.5, lZ=0.45, nX=1, nY=1, nZ=1, elType=C3D8, elProvider=edelweiss

*modelGenerator, generator=surfaceElementGenerator, name=slaveFacets
surface=upper_bottom
name=slaveSurf

*modelGenerator, generator=surfaceElementGenerator, name=masterFacets
surface=lower_top
name=masterSurf

*section, name=section1, material=linearelastic, type=solid
lower_all
upper_all
"""
)


def _writeBoxStl(path: Path, lower: tuple, upper: tuple):
    """A closed box, as an ASCII STL with outward-oriented triangles."""

    # corner i takes the upper bound in x, y, z where bit 0, 1, 2 of i is set
    bounds = (lower, upper)
    corners = [np.array([bounds[(i >> axis) & 1][axis] for axis in range(3)], dtype=float) for i in range(8)]
    quads = [(0, 2, 3, 1), (4, 5, 7, 6), (0, 1, 5, 4), (2, 6, 7, 3), (0, 4, 6, 2), (1, 3, 7, 5)]
    center = 0.5 * (np.array(lower) + np.array(upper))

    lines = ["solid box"]
    for quad in quads:
        for a, b, c in ((quad[0], quad[1], quad[2]), (quad[0], quad[2], quad[3])):
            pa, pb, pc = corners[a], corners[b], corners[c]
            normal = np.cross(pb - pa, pc - pa)
            if normal @ ((pa + pb + pc) / 3.0 - center) < 0.0:
                pb, pc, normal = pc, pb, -normal
            lines += [" facet normal {:} {:} {:}".format(*normal / np.linalg.norm(normal)), "  outer loop"]
            lines += ["   vertex {:} {:} {:}".format(*p) for p in (pa, pb, pc)] + ["  endloop", " endfacet"]
    path.write_text("\n".join(lines + ["endsolid box"]) + "\n")


def _buildModel(tmp_path: Path, deck: str, domainSize: int) -> FEModel:
    """The model of ``deck``, set up as the driver does before the first step, without solving anything."""

    inpPath = tmp_path / "model.inp"
    inpPath.write_text(deck)
    inputfile = parseInputFile(str(inpPath))

    journal = Journal(verbose=False)
    model = FEModel(domainSize)
    model = fillFEModelFromInputFile(model, inputfile, journal)
    model.prepareYourself(journal)
    model.advanceToTime(0.0)

    for nodeField in model.nodeFields.values():
        nodeField.createFieldValueEntry("U")
        nodeField.createFieldValueEntry("P")
    model._linkFieldVariableObjects(model.nodeSets["all"])

    return model


def _noise(node: Node, field: str, scale: float) -> np.ndarray:
    """A reproducible perturbation of the values of ``field`` at ``node``, different for every node."""

    size = getFieldSize(field, len(node.coordinates))
    return scale * np.random.default_rng(node.label).standard_normal(size)


def _localIndices(constraint, isWanted) -> np.ndarray:
    """The local indices of the components of the (node, field) pairs for which ``isWanted(node, field)`` holds."""

    indices, offset = [], 0
    for node, fields in zip(constraint.nodes, constraint.fieldsOnNodes):
        for field in fields:
            size = getFieldSize(field, len(node.coordinates))
            if isWanted(node, field):
                indices += range(offset, offset + size)
            offset += size
    return np.array(indices, dtype=int)


def _assertTangentIsTheDerivative(constraint, U, dU, timeStep=None, rows=None, excluded=None, h=1e-7):
    """Assert ``K == dP/dU``, on the given rows, with the given (row, column) blocks excluded."""

    P, K = residualContributionAndTangent(constraint, U, dU, timeStep)
    dP_dU = finiteDifferenceTangent(constraint, U, dU, timeStep, h)

    if excluded is not None:
        K[np.ix_(excluded, excluded)] = 0.0
        dP_dU[np.ix_(excluded, excluded)] = 0.0
    if rows is not None:
        K, dP_dU = K[rows], dP_dU[rows]

    scale = np.abs(K).max()
    assert scale > 0.0, "the check is vacuous: the tangent is zero"
    np.testing.assert_allclose(K, dP_dU, rtol=0.0, atol=1e-6 * scale)
    return P


# --- constraints


def test_directional_spring_penalty(tmp_path):
    model = _buildModel(
        tmp_path,
        _PLANE_BLOCK
        + """
*constraint, type=directionalSpringPenalty, name=theConstraint
nSet=gen_top
field=displacement
component=1
penalty=10.0
""",
        2,
    )
    constraint = model.constraints["theConstraint"]
    U = localVector(constraint, lambda node, field: _noise(node, field, 1e-2))

    P = _assertTangentIsTheDerivative(constraint, U, U)

    # the spring pulls the nodes back: its residual contribution has the sign of the displacement, like a spring element
    np.testing.assert_allclose(P[1::2], 10.0 * U[1::2], rtol=1e-14)


def test_equal_value_penalty(tmp_path):
    model = _buildModel(
        tmp_path,
        _PLANE_BLOCK
        + """
*constraint, type=equalValuePenalty, name=theConstraint
nSet=gen_top
field=displacement
component=1
penalty=10.0
""",
        2,
    )
    constraint = model.constraints["theConstraint"]
    U = localVector(constraint, lambda node, field: _noise(node, field, 1e-2))

    _assertTangentIsTheDerivative(constraint, U, U)


def test_equal_value_lagrangian(tmp_path):
    model = _buildModel(
        tmp_path,
        _PLANE_BLOCK
        + """
*constraint, type=equalValueLagrangian, name=theConstraint
nSet=gen_top
field=displacement
component=1
""",
        2,
    )
    constraint = model.constraints["theConstraint"]
    U = localVector(constraint, lambda node, field: _noise(node, field, 1e-2), scalarValues=[0.3, -0.7])

    _assertTangentIsTheDerivative(constraint, U, U)


@pytest.mark.parametrize("contactType", ["linear", "quadratic"])
def test_node_to_rigid_surface_penalty(tmp_path, contactType):
    model = _buildModel(
        tmp_path,
        _PLANE_BLOCK
        + f"""
*constraint, type=nodeToRigidSurfacePenalty, name=theConstraint
nSet=gen_top
field=displacement
component=1
penalty=1e3
value=0.0
direction=1.0
type={contactType}
""",
        2,
    )
    constraint = model.constraints["theConstraint"]

    # every top node has moved past the boundary at y = 0 by far more than the perturbation
    U = localVector(constraint, lambda node, field: np.array([0.0, 0.02]) + _noise(node, field, 1e-3))

    P = _assertTangentIsTheDerivative(constraint, U, U)

    # the boundary pushes the nodes back (-y), so their residual contribution points in +y
    assert np.all(P[1::2] > 0.0)


def test_penalty_indirect_control(tmp_path):
    model = _buildModel(
        tmp_path,
        _PLANE_BLOCK
        + """
*constraint, type=penaltyIndirectControl, name=theConstraint
field=displacement
cVector='0, 1, 0, 0, 0, -1'
constrainedNSet=gen_left
loadNSet=gen_rightTop
loadVector='0, -1'
length=-1
penaltyStiffness=1e3
""",
        2,
    )
    constraint = model.constraints["theConstraint"]
    U = localVector(constraint, lambda node, field: _noise(node, field, 1e-2))
    timeStep = TimeStep(1, 0.1, 0.5, 0.1, 0.5, 0.5)

    _assertTangentIsTheDerivative(constraint, U, U, timeStep)


@pytest.mark.parametrize("domainSize", [2, 3])
def test_linearized_rigid_body(tmp_path, domainSize):
    """Only the rows of the nodes are compared: the rows of the Lagrange multipliers receive no residual, since the
    constraint relies on the linear constraint equation staying satisfied (see the constraint's ``computeKernels``)."""

    model = _buildModel(
        tmp_path,
        (_PLANE_BLOCK if domainSize == 2 else _SOLID_BLOCK)
        + """
*constraint, type=linearizedRigidBody, name=theConstraint
nSet=gen_right
referencePoint={:}
""".format(
            "gen_rightTop" if domainSize == 2 else "gen_topRightFront"
        ),
        domainSize,
    )
    constraint = model.constraints["theConstraint"]
    U = localVector(
        constraint,
        lambda node, field: _noise(node, field, 1e-2),
        scalarValues=np.linspace(-1.0, 1.0, constraint.nConstraints),
    )

    _assertTangentIsTheDerivative(constraint, U, U, rows=slice(0, constraint.nDof - constraint.nConstraints))


def test_rigid_body(tmp_path):
    model = _buildModel(
        tmp_path,
        _SOLID_BLOCK
        + """
*constraint, type=rigidBody, name=theConstraint
nSet=gen_right
referencePoint=gen_bottomLeftBack
""",
        3,
    )
    constraint = model.constraints["theConstraint"]
    rotation = np.array([0.1, -0.2, 0.15])
    U = localVector(
        constraint,
        lambda node, field: _noise(node, field, 1e-2) if field == "displacement" else rotation,
        scalarValues=np.linspace(-1.0, 1.0, constraint.nConstraints),
    )

    _assertTangentIsTheDerivative(constraint, U, U)


def test_node_to_discrete_rigid_body_penalty(tmp_path):
    """The rotation-rotation block of the reference point is excluded: the constraint omits it (it needs the second
    derivative of the exponential map), as documented in its class docstring."""

    _writeBoxStl(tmp_path / "support.stl", (-1.0, -1.0, -1.0), (2.0, 0.0, 2.0))
    model = _buildModel(
        tmp_path,
        _SOLID_BLOCK
        + f"""
*modelGenerator, generator=discreteRigidBodyGenerator, name=support, executeAfterManualGeneration=True
filename={tmp_path / 'support.stl'}
rpCoordinate='0.5, -0.5, 0.5'
mass=1.0
inertia="1.0, 1.0, 1.0"

*constraint, type=nodeToDiscreteRigidBodyPenalty, name=theConstraint
nSet=gen_bottom
rigidBody=support
penalty=1e3
searchDistance=1.0
""",
        3,
    )
    constraint = model.constraints["theConstraint"]
    rpNode = constraint.rpNode

    def nodalValue(node, field):
        if node is rpNode:
            return np.array([1e-3, 2e-3, -1e-3]) if field == "displacement" else np.array([0.01, -0.02, 0.015])
        # the slave nodes penetrate the support's top face by far more than the perturbation
        return np.array([0.0, -0.05, 0.0]) + _noise(node, field, 1e-3)

    U = localVector(constraint, nodalValue)
    rpRotation = _localIndices(constraint, lambda node, field: node is rpNode and field == "rotation")

    _assertTangentIsTheDerivative(constraint, U, U, excluded=rpRotation)
    assert constraint.totalNormalForce > 0.0, "the contact must be active"


def _nodeToDeformableSurfaceCase(tmp_path, deck: str, domainSize: int, options: str, tangential: float):
    """The node-to-deformable-surface constraint between the two blocks of ``deck``, and a state in which the upper
    block has moved down by 0.1 -- penetrating the lower one by 0.05 -- and sideways by ``tangential``."""

    model = _buildModel(
        tmp_path,
        deck
        + f"""
*constraint, name=theConstraint, type=nodeToDeformableSurfacePenalty
slaveSurface=slaveSurf_facets, masterSurface=masterSurf_facets, penalty=1e3, searchDistance=2.0, {options}
""",
        domainSize,
    )
    constraint = model.constraints["theConstraint"]
    constraint.updateConnectivity(model)

    def nodalValue(node, field):
        motion = np.zeros(domainSize)
        if node.coordinates[1] > 1.02:
            motion[0] = tangential
            motion[1] = -0.1
        return motion + _noise(node, field, 1e-4)

    U = localVector(constraint, nodalValue)
    return constraint, U


@pytest.mark.parametrize(
    "options, tangential",
    [
        ("type=linear, sliding=small", 0.0),
        ("type=quadratic, sliding=small", 0.0),
        # stick: the tangential trial force stays well below the friction limit mu * N
        ("type=linear, sliding=small, mu=0.3, tangentPenalty=1e3", 1e-3),
        # slip: the tangential trial force exceeds the friction limit by far
        ("type=linear, sliding=small, mu=0.3, tangentPenalty=1e3", 5e-2),
        ("type=quadratic, sliding=finite", 0.0),
    ],
    ids=["small-linear", "small-quadratic", "friction-stick", "friction-slip", "finite-quadratic"],
)
def test_node_to_deformable_surface_penalty_2d(tmp_path, options, tangential):
    constraint, U = _nodeToDeformableSurfaceCase(tmp_path, _TWO_PLANE_BLOCKS, 2, options, tangential)

    _assertTangentIsTheDerivative(constraint, U, U)
    assert np.all(constraint._normalForceCurrent < 0.0), "every slave node must be in contact"


def test_node_to_deformable_surface_penalty_3d_finite_sliding(tmp_path):
    constraint, U = _nodeToDeformableSurfaceCase(tmp_path, _TWO_SOLID_BLOCKS, 3, "type=quadratic, sliding=finite", 0.0)

    _assertTangentIsTheDerivative(constraint, U, U)
    assert np.all(constraint._normalForceCurrent < 0.0), "every slave node must be in contact"


# --- element loads


def test_python_element_body_force_is_the_negative_external_load(tmp_path):
    """On a rectangular bilinear quad, a constant body force ``b`` puts ``b * area * thickness / 4`` on each node."""

    model = _buildModel(
        tmp_path,
        _MATERIAL
        + """
*modelGenerator, generator=planeRectQuad, name=gen
x0=0, y0=0, l=2.0, h=1.0, nX=1, nY=1, elType=CPE4, elProvider=edelweiss

*section, name=section1, thickness=3.0, material=linearelastic, type=plane
gen_all
""",
        2,
    )
    (element,) = model.elements.values()
    load = np.array([0.3, -0.7])

    P = np.zeros(element.nDof)
    K = np.zeros(element.nDof**2)
    element.computeBodyForce(P, K, load, np.zeros(element.nDof), 0.0, 1.0)

    externalLoadPerNode = load * 2.0 * 1.0 * 3.0 / 4.0
    np.testing.assert_allclose(P.reshape((-1, 2)), np.tile(-externalLoadPerNode, (4, 1)), rtol=1e-14)
    np.testing.assert_array_equal(K, 0.0)


def test_point_mass_body_force_is_the_negative_external_load():
    """A point mass reads the body force as an acceleration: the external load is mass times it."""

    model = FEModel(3)
    pointMass = PointMass(1, [Node(1, np.zeros(3))], model, mass=2.5, inertia=[1.0, 1.0, 1.0])
    gravity = np.array([0.0, -9.81, 0.0])

    P = np.zeros(pointMass.nDof)
    pointMass.computeBodyForce(P, np.zeros(pointMass.nDof**2), gravity, np.zeros(pointMass.nDof), 0.0, 1.0)

    np.testing.assert_array_equal(P[:3], -2.5 * gravity)
    np.testing.assert_array_equal(P[3:], 0.0)


#: Marmot's linear elastic material, for the Marmot element loads.
_MARMOT_MATERIAL = """
*material, name=LinearElastic, id=linearelastic
100.0, 0.3
"""


def test_marmot_element_body_force_is_the_negative_external_load(tmp_path):
    """On a brick, a constant body force ``b`` puts ``b * volume / 8`` on each node of a trilinear hexahedron."""

    pytest.importorskip("edelweissfe.elements.marmotelement.element")
    model = _buildModel(
        tmp_path,
        _MARMOT_MATERIAL
        + """
*modelGenerator, generator=boxGen, name=gen
x0=0, y0=0, z0=0, lX=2.0, lY=1.0, lZ=0.5, nX=1, nY=1, nZ=1, elType=C3D8

*section, name=section1, material=linearelastic, type=solid
gen_all
""",
        3,
    )
    (element,) = model.elements.values()
    load = np.array([0.3, -0.7, 0.2])

    P = np.zeros(element.nDof)
    K = np.zeros(element.nDof**2)
    element.computeBodyForce(P, K, load, np.zeros(element.nDof), 0.0, 1.0)

    externalLoadPerNode = load * 2.0 * 1.0 * 0.5 / 8.0
    np.testing.assert_allclose(P.reshape((-1, 3)), np.tile(-externalLoadPerNode, (8, 1)), rtol=1e-13)
    np.testing.assert_array_equal(K, 0.0)


@pytest.mark.parametrize("loadType, load", [("pressure", [0.4]), ("surface traction", [0.3, -0.7])])
def test_marmot_element_distributed_load_is_the_negative_external_load(tmp_path, loadType, load):
    """A constant pressure ``p`` on the top face of a rectangle acts against the outward normal, a traction ``t`` acts
    as given; each puts half of its resultant on each of the two face nodes."""

    pytest.importorskip("edelweissfe.elements.marmotelement.element")
    model = _buildModel(
        tmp_path,
        _MARMOT_MATERIAL
        + """
*modelGenerator, generator=planeRectQuad, name=gen
x0=0, y0=0, l=2.0, h=1.0, nX=1, nY=1, elType=CPE4

*section, name=section1, thickness=3.0, material=linearelastic, type=plane
gen_all
""",
        2,
    )
    ((faceID, (element,)),) = model.surfaces["gen_top"].items()
    load = np.array(load)

    P = np.zeros(element.nDof)
    K = np.zeros(element.nDof**2)
    element.computeDistributedLoad(loadType, P, K, faceID, load, np.zeros(element.nDof), 0.0, 1.0)

    faceLength, thickness = 2.0, 3.0
    resultant = -load[0] * np.array([0.0, 1.0]) if loadType == "pressure" else load
    externalLoad = np.zeros((4, 2))
    for i, node in enumerate(element.nodes):
        if node.coordinates[1] == 1.0:
            externalLoad[i] = 0.5 * resultant * faceLength * thickness

    np.testing.assert_allclose(P.reshape((-1, 2)), -externalLoad, rtol=1e-13, atol=1e-15)


def test_marmot_follower_pressure_tangent_is_the_derivative_of_its_residual_contribution():
    """A pressure on the current surface of an updated-Lagrangian element follows the deformation, so its residual
    contribution has a tangent, which Marmot computes together with the load; checked by central differences on every
    face of a brick in a deformed state."""

    elementModule = pytest.importorskip("edelweissfe.elements.marmotelement.element")
    try:
        element = elementModule.MarmotElementWrapper("C3D8UL", 1)
    except NotImplementedError:
        pytest.skip("this Marmot build has no C3D8UL")

    corners = [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0), (0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1)]
    element.setNodes([Node(i, np.array(x, dtype=float)) for i, x in enumerate(corners)])
    element.setMaterial("COMPRESSIBLENEOHOOKE", np.array([3500.0, 1500.0]))
    element.initializeElement()
    nDof = element.nDof
    U = 0.05 * np.random.default_rng(0).standard_normal(nDof)

    def pressureContributionAndTangent(faceID, U):
        P = np.zeros(nDof)
        K = np.zeros(nDof * nDof)
        element.computeDistributedLoad("pressure", P, K, faceID, np.array([2.0]), U, 0.0, 1.0)
        return P, K.reshape((nDof, nDof), order="F")

    for faceID in range(1, 7):
        _, K = pressureContributionAndTangent(faceID, U)
        dP_dU = centralDifferences(lambda U: pressureContributionAndTangent(faceID, U)[0], U)

        scale = np.abs(K).max()
        assert scale > 0.0
        np.testing.assert_allclose(K, dP_dU, rtol=0.0, atol=1e-6 * scale, err_msg=f"face {faceID}")
