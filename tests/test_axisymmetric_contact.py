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
"""Contact and tie in axisymmetric models, where every contact point stands for the ring it sweeps about the axis.

The first tests need no element formulation: two rings of quadratic elements overlap across an annulus or a cylinder,
and the areas of the contact points and the forces of their springs are checked against the areas of the rings and the
force of a uniform pressure on them; on the axis, where a point stands for no area, the end faces of two solid
cylinders. The tests with Marmot's axisymmetric elements (CX8RUL, CAX8R) solve: tied rings transmit a homogeneous state
as the merged mesh does, a contact patch transmits a pressure whose force the reactions of the elements balance, also
on a disk reaching the axis, the critical time step of an explicit analysis with a tie is estimated below the exact
one, and a resumed explicit contact is bitwise the uninterrupted one.
"""

import warnings
from pathlib import Path

import numpy as np
import pytest
from test_surfacetodeformablesurfacepenalty import (  # noqa: E402  (tests/ is on sys.path)
    assertTheBoundIsTheOneOfTheTangent,
)

from edelweissfe.config.configurator import loadConfiguration
from edelweissfe.constraints.base.contactpoints import (
    axisTolerance,
    circumferenceAt,
    outOfPlaneExtentAt,
)
from edelweissfe.constraints.base.surfacecontactpenaltyschema import (
    contactPointsOfTheSlaveSurface,
)
from edelweissfe.constraints.nodetodeformablesurfacepenalty import (
    Constraint as NodeToDeformableSurfacePenalty,
)
from edelweissfe.constraints.nodetodeformablesurfacepenalty import (
    NodeToDeformableSurfacePenaltySchema,
)
from edelweissfe.constraints.surfacetodeformablesurfacepenalty import (
    Constraint as SurfaceToDeformableSurfacePenalty,
)
from edelweissfe.constraints.surfacetodeformablesurfacepenalty import (
    SurfaceToDeformableSurfacePenaltySchema,
)
from edelweissfe.drivers.inputfiledrivensimulation import finiteElementSimulation
from edelweissfe.elements.displacementelement.element import DisplacementElement
from edelweissfe.generators.surfaceelementgenerator import buildContactFacets
from edelweissfe.helpers.inputfilehelpers import fillFEModelFromInputFile
from edelweissfe.journal.journal import Journal
from edelweissfe.models.femodel import FEModel
from edelweissfe.points.node import Node
from edelweissfe.sections.plane import PlaneSectionSchema
from edelweissfe.sections.plane import Section as PlaneSection
from edelweissfe.sets.elementset import ElementSet
from edelweissfe.solvers.nonlinearexplicitdynamic import NED
from edelweissfe.steps.adaptivestepforexplicitsimulations import (
    AdaptiveStepForExplicitSimulations,
)
from edelweissfe.steps.stepmanager import StepActionCollection
from edelweissfe.utils.fieldoutput import FieldOutputController
from edelweissfe.utils.inputfileparser import parseInputFile
from edelweissfe.utils.residualtangentcheck import residualContributionAndTangent

#: The rings span the radii RI to RO and the height H; the slave ring overlaps the master by DELTA.
RI, RO, H, DELTA = 1.0, 2.0, 1.0, 1e-3
#: The radius of the cylindrical interface.
R = 1.5
#: The penalty of the unit tests.
PENALTY = 1e4

#: The faces of a quad8 element, per the generator's face tables.
_BOTTOM, _RIGHT, _TOP, _LEFT = 1, 2, 3, 4


def _quad8Ring(model: FEModel, origin: tuple, size: tuple, divisions: tuple) -> np.ndarray:
    """A structured ring of 8-node elements (the section r-z of a ring), with the element formulation the facets need
    alone; returned as the grid of its elements, (divisions along r, along z). The elements are plane ones (CPE8), never
    evaluated: they give the facets their nodes, faces and sections. In an axisymmetric model they stand in for
    axisymmetric elements, which nothing checks (:attr:`~edelweissfe.models.femodel.FEModel.axisymmetric`).

    Parameters
    ----------
    model
        The model, which takes the nodes and the elements.
    origin
        The corner (r, z) of the section with the smallest coordinates.
    size
        The extent of the section along r and z.
    divisions
        The number of elements along r and z.
    """

    nR, nZ = divisions
    r = np.linspace(origin[0], origin[0] + size[0], 2 * nR + 1)
    z = np.linspace(origin[1], origin[1] + size[1], 2 * nZ + 1)
    nodeAt = {}
    for i in range(2 * nR + 1):
        for j in range(2 * nZ + 1):
            if i % 2 and j % 2:
                continue  # the center of an element, which a quad8 has no node at
            label = len(model.nodes) + 1
            model.nodes[label] = nodeAt[i, j] = Node(label, np.array([r[i], z[j]]))

    grid = np.empty(divisions, dtype=object)
    for a in range(nR):
        for b in range(nZ):
            i, j = 2 * a, 2 * b
            corners = [(i, j), (i + 2, j), (i + 2, j + 2), (i, j + 2)]
            midsides = [(i + 1, j), (i + 2, j + 1), (i + 1, j + 2), (i, j + 1)]
            (elNumber,) = model.topology.reserveElementNumbers(1)
            element = DisplacementElement("CPE8", elNumber)
            element.setNodes([nodeAt[ij] for ij in corners + midsides])
            model.createElement(element)
            grid[a, b] = element
    return grid


def _twoRings(interface: str, axisymmetric: bool = True, thickness: float = 7.0, innerRadius: float = RI) -> tuple:
    """Two rings that overlap by :data:`DELTA` across an ``"annulus"`` (z = 0, innerRadius <= r <= RO; the lower ring
    is the slave; with innerRadius = 0 two solid cylinders, the annulus a disk) or a ``"cylinder"`` (r = R, 0 <= z <= H;
    the inner ring is the slave), with different meshes on either side, in a plane section of the given thickness, which
    an axisymmetric model does not use. In a plane model they are two blocks of that thickness.

    Returns
    -------
    tuple
        The model, the slave facets, the master facets, and the area of the ring of the slave surface (in a plane
        model, the length of the slave line times the thickness).
    """

    journal = Journal(verbose=False)
    model = FEModel(2, axisymmetric=axisymmetric)
    with model.topology.changes():
        if interface == "annulus":
            slave = _quad8Ring(model, (innerRadius, -H + DELTA), (RO - innerRadius, H), (2, 2))
            master = _quad8Ring(model, (innerRadius, 0.0), (RO - innerRadius, H), (3, 2))
            model.surfaces["slaveFace"] = {_TOP: ElementSet("s", list(slave[:, -1]))}
            model.surfaces["masterFace"] = {_BOTTOM: ElementSet("m", list(master[:, 0]))}
            ringArea = np.pi * (RO**2 - innerRadius**2) if axisymmetric else (RO - innerRadius) * thickness
        else:
            slave = _quad8Ring(model, (RI, 0.0), (R + DELTA - RI, H), (2, 2))
            master = _quad8Ring(model, (R, 0.0), (RO - R, H), (2, 3))
            model.surfaces["slaveFace"] = {_RIGHT: ElementSet("s", list(slave[-1, :]))}
            model.surfaces["masterFace"] = {_LEFT: ElementSet("m", list(master[0, :]))}
            ringArea = 2 * np.pi * (R + DELTA) * H if axisymmetric else H * thickness
        model.sections["rings"] = PlaneSection(
            "rings",
            model,
            {},
            [ElementSet("rings", list(slave.ravel()) + list(master.ravel()))],
            configuration=PlaneSectionSchema(thickness=thickness),
        )
        slaveSetName, _ = buildContactFacets(model, "slaveFace", "slv", "corner", "facetConsistent", journal)
        masterSetName, _ = buildContactFacets(model, "masterFace", "mst", "corner", "facetConsistent", journal)
    return model, slaveSetName, masterSetName, ringArea


def _constraint(model: FEModel, slaveSetName: str, masterSetName: str, **options) -> SurfaceToDeformableSurfacePenalty:
    """The constraint between the slave and the master facets, built as an input file builds it, and paired."""

    configuration = SurfaceToDeformableSurfacePenaltySchema(
        slaveSurface=slaveSetName, masterSurface=masterSetName, penalty=PENALTY, searchDistance=0.5, **options
    )
    constraint = SurfaceToDeformableSurfacePenalty(
        "theContact",
        model,
        contactPointsOfTheSlaveSurface(configuration, model, "theContact"),
        model.elementSets[masterSetName],
        Journal(verbose=False),
        configuration=configuration,
    )
    constraint.updateConnectivity(model)
    return constraint


def _evaluate(constraint, U: np.ndarray):
    """Evaluate the springs at the solution ``U`` of the constraint's block, for the outputs."""

    constraint.computeKernels(U, np.zeros_like(U), np.zeros(constraint.nDof), None, None)


#: The kinds of contact points of a slave surface, and the quadrature rules of the quadrature points.
_CONTACT_POINTS = [
    dict(contactPoints="quadraturePoints", nQuadraturePoints=1),
    dict(contactPoints="quadraturePoints", nQuadraturePoints=2),
    dict(contactPoints="quadraturePoints", nQuadraturePoints=3),
    dict(contactPoints="nodes"),
]
_CONTACT_POINTS_IDS = ["1 quadrature point", "2 quadrature points", "3 quadrature points", "nodes"]


def test_the_out_of_plane_extent_is_one_the_thickness_or_the_circumference():
    """:math:`b` of :math:`\\mathrm{d}V = b\\, \\mathrm{d}A`: 1 in 3D, the thickness of the element's section in a
    plane model, :math:`2 \\pi r` in an axisymmetric model, 0 on the axis."""

    coordinates = np.array([[2.0, 7.0], [0.5, -1.0], [0.0, 3.0]])
    for axisymmetric, expected in ((False, np.full(3, 7.0)), (True, 2 * np.pi * coordinates[:, 0])):
        model, _, _, _ = _twoRings("annulus", axisymmetric=axisymmetric, thickness=7.0)
        elements = [next(iter(model.sections["rings"].elSets[0]))] * 3
        np.testing.assert_array_equal(outOfPlaneExtentAt(coordinates, elements, model), expected)
    np.testing.assert_array_equal(outOfPlaneExtentAt(np.zeros((2, 3)), [None, None], FEModel(3)), np.ones(2))


def test_a_point_within_the_round_off_of_the_axis_is_on_it_and_one_further_below_is_refused():
    """The rule of Marmot's axisymmetric elements for their nodes: :math:`r \\geq -\\epsilon h` with the size
    :math:`h` of the element is the round-off of a mesher, on the axis; further below, a mesh drawn across the axis."""

    model, _, _, _ = _twoRings("annulus")
    element = next(iter(model.sections["rings"].elSets[0]))
    corners = np.array([node.coordinates for node in element.nodes])
    h = np.linalg.norm(corners.max(axis=0) - corners.min(axis=0))

    onTheAxis = np.array([[-1e-17, 0.0], [-0.5 * axisTolerance * h, 0.0]])
    np.testing.assert_array_equal(outOfPlaneExtentAt(onTheAxis, [element] * 2, model), 0.0)
    with pytest.raises(ValueError, match="r >= 0"):
        outOfPlaneExtentAt(np.array([[-2 * axisTolerance * h, 0.0]]), [element], model)


def test_the_circumference_on_radii_and_sizes_is_the_rule_of_the_axis_for_every_kind_of_point():
    """:func:`circumferenceAt` takes the radii and the sizes of the bodies the points lie on, so that a kind of point
    of another library, whose bodies are not elements, has the same rule of the axis: :math:`2 \\pi r`, 0 within the
    round-off :math:`\\epsilon h` of the body below the axis, a refusal further below that names the body."""

    sizes = np.array([1.0, 1.0, 1.0e3, 1.0])
    radii = np.array([0.5, 0.0, -0.5 * axisTolerance * 1.0e3, 2.0])
    np.testing.assert_array_equal(circumferenceAt(radii, sizes, str), 2 * np.pi * np.array([0.5, 0.0, 0.0, 2.0]))
    with pytest.raises(ValueError, match="on particle 1; .* r >= 0"):
        circumferenceAt(np.array([1.0, -2 * axisTolerance]), np.ones(2), lambda p: f"particle {p}")


@pytest.mark.parametrize("interface", ["annulus", "cylinder"])
@pytest.mark.parametrize("axisymmetric", [True, False], ids=["axisymmetric", "plane"])
@pytest.mark.parametrize("contactPoints", _CONTACT_POINTS, ids=_CONTACT_POINTS_IDS)
def test_the_areas_of_the_points_add_up_to_the_area_of_the_ring_of_the_slave_surface(
    interface, axisymmetric, contactPoints
):
    """Every rule integrates the radius, which is linear along a straight facet, exactly: the areas of the points add up
    to the area of the ring of the slave surface, :math:`\\pi (r_o^2 - r_i^2)` of an annulus, :math:`2 \\pi R H` of a
    cylinder, whatever the thickness of the section -- and in a plane model to the length of the slave line times the
    thickness."""

    model, slaveSetName, masterSetName, ringArea = _twoRings(interface, axisymmetric)
    constraint = _constraint(model, slaveSetName, masterSetName, **contactPoints)

    assert constraint.slave.axisymmetric == axisymmetric
    assert constraint.slave.areas.sum() == pytest.approx(ringArea, rel=1e-14)


@pytest.mark.parametrize("interface", ["annulus", "cylinder"])
@pytest.mark.parametrize("contactPoints", _CONTACT_POINTS, ids=_CONTACT_POINTS_IDS)
def test_a_uniform_overlap_transmits_the_force_of_a_uniform_pressure_on_the_ring(interface, contactPoints):
    """The slave ring overlaps the master by :math:`\\delta` everywhere, so the pressure is :math:`k \\delta` at every
    point, and the force across the interface is that pressure on the area of the ring: :math:`\\pi (r_o^2 - r_i^2) k
    \\delta` along the axis across an annulus, :math:`2 \\pi R H k \\delta` radially across a cylinder (the force on the
    rings, whose radial forces cancel about the axis). The nodal forces on either side carry it, equal and opposite."""

    model, slaveSetName, masterSetName, ringArea = _twoRings(interface)
    constraint = _constraint(model, slaveSetName, masterSetName, **contactPoints)
    U = np.zeros(constraint.nDof)
    _evaluate(constraint, U)

    assert constraint.numberOfPointsWithASpring == constraint.nPoints
    np.testing.assert_allclose(constraint.getNormalPressures(), PENALTY * DELTA, rtol=1e-10)

    axis = 1 if interface == "annulus" else 0
    expected = np.zeros(2)
    expected[axis] = -PENALTY * DELTA * ringArea
    np.testing.assert_allclose(constraint.contactForceOnSlave, expected, rtol=1e-10, atol=1e-12 * PENALTY * DELTA)
    np.testing.assert_allclose(constraint.contactForceOnMaster, -expected, rtol=1e-10, atol=1e-12 * PENALTY * DELTA)

    P = np.zeros(constraint.nDof)
    constraint.computeKernels(U, np.zeros_like(U), P, None, None)
    forcesOnTheNodes = -P.reshape((-1, 2))
    onTheSlave = np.array([node in constraint.slave.finiteElementNodes() for node in constraint.nodes])
    np.testing.assert_allclose(forcesOnTheNodes[onTheSlave].sum(axis=0), expected, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(forcesOnTheNodes[~onTheSlave].sum(axis=0), -expected, rtol=1e-10, atol=1e-10)

    assertTheBoundIsTheOneOfTheTangent(constraint, U)


@pytest.mark.parametrize("interface", ["annulus", "cylinder"])
def test_a_tie_holds_the_ring_with_the_force_of_a_uniform_traction_on_it(interface):
    """A tie paired at rest, its slave ring moved by :math:`\\boldsymbol{d}`: the traction is :math:`-k
    \\boldsymbol{d}` at every point, the force on the slave ring that traction on its area."""

    model, slaveSetName, masterSetName, ringArea = _twoRings(interface)
    constraint = _constraint(model, slaveSetName, masterSetName, contactType="tied")
    d = np.array([3e-4, -2e-4])

    U = np.zeros(constraint.nDof)
    slaveNodes = constraint.slave.finiteElementNodes()
    for i, node in enumerate(constraint.nodes):
        if node in slaveNodes:
            U[2 * i : 2 * i + 2] = d
    _evaluate(constraint, U)

    np.testing.assert_allclose(constraint.getTractions(), np.tile(-PENALTY * d, (constraint.nPoints, 1)), rtol=1e-10)
    np.testing.assert_allclose(constraint.contactForceOnSlave, -PENALTY * d * ringArea, rtol=1e-10)
    assertTheBoundIsTheOneOfTheTangent(constraint, U)


@pytest.mark.parametrize("contactPoints", _CONTACT_POINTS, ids=_CONTACT_POINTS_IDS)
def test_a_point_on_the_axis_stands_for_no_area_and_reports_no_traction(contactPoints):
    """The end faces of two solid cylinders, 0 <= r <= RO, overlap by :math:`\\delta`. The axis is a line, not a
    surface of the body: a point on it stands for a ring of radius 0, no area, and carries no force. The areas still add
    up to the disk, :math:`\\pi R^2`, the force across it is the pressure :math:`k \\delta` on the disk, and the
    pressures and tractions are finite everywhere: :math:`k \\delta` off the axis, zero on it, without a division by
    zero. The quadrature points lie inside their facets, never on the axis; the node at r = 0 lies on it."""

    model, slaveSetName, masterSetName, diskArea = _twoRings("annulus", innerRadius=0.0)
    constraint = _constraint(model, slaveSetName, masterSetName, **contactPoints)
    U = np.zeros(constraint.nDof)
    _evaluate(constraint, U)

    assert diskArea == np.pi * RO**2
    assert constraint.slave.areas.sum() == pytest.approx(diskArea, rel=1e-14)
    assert constraint.numberOfPointsWithASpring == constraint.nPoints
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        pressures, tractions = constraint.getNormalPressures(), constraint.getTractions()

    onTheAxis = constraint.slave.referenceCoordinates()[:, 0] == 0.0
    assert onTheAxis.sum() == (1 if contactPoints["contactPoints"] == "nodes" else 0)
    assert np.all(constraint.slave.areas[onTheAxis] == 0.0)
    assert np.all(constraint._normalForcesOfPoints()[onTheAxis] == 0.0)
    assert np.all(pressures[onTheAxis] == 0.0) and np.all(tractions[onTheAxis] == 0.0)
    np.testing.assert_allclose(pressures[~onTheAxis], PENALTY * DELTA, rtol=1e-10)
    np.testing.assert_allclose(
        tractions[~onTheAxis],
        np.tile([0.0, -PENALTY * DELTA], ((~onTheAxis).sum(), 1)),
        rtol=1e-10,
        atol=1e-12 * PENALTY * DELTA,
    )
    np.testing.assert_allclose(
        constraint.contactForceOnSlave, [0.0, -PENALTY * DELTA * diskArea], rtol=1e-10, atol=1e-12
    )


def test_a_node_within_the_round_off_of_the_axis_is_on_it():
    """A node a mesher put at r = -1e-17 is on the axis, as the axisymmetric elements accept it: its area is 0. A node
    at r = -1e-3 is a mesh drawn across the axis, and refused."""

    for radius, accepted in ((-1e-17, True), (-1e-3, False)):
        model, slaveSetName, masterSetName, _ = _twoRings("annulus", innerRadius=0.0)
        for node in model.nodes.values():
            if node.coordinates[0] == 0.0:
                node.coordinates[0] = radius
        if accepted:
            constraint = _constraint(model, slaveSetName, masterSetName, contactPoints="nodes")
            onTheAxis = constraint.slave.referenceCoordinates()[:, 0] < 0.0
            assert onTheAxis.sum() == 1 and constraint.slave.areas[onTheAxis] == 0.0
        else:
            with pytest.raises(ValueError, match="r >= 0"):
                _constraint(model, slaveSetName, masterSetName, contactPoints="nodes")


def test_the_node_based_contact_refuses_an_axisymmetric_model():
    model, slaveSetName, masterSetName, _ = _twoRings("annulus")

    with pytest.raises(NotImplementedError, match="surfaceToDeformableSurfacePenalty"):
        NodeToDeformableSurfacePenalty(
            "theContact",
            model,
            model.elementSets[slaveSetName],
            model.elementSets[masterSetName],
            Journal(verbose=False),
            configuration=NodeToDeformableSurfacePenaltySchema(
                slaveSurface=slaveSetName, masterSurface=masterSetName, penalty=PENALTY
            ),
        )


def test_an_axisymmetric_model_is_two_dimensional():
    with pytest.raises(ValueError, match="two-dimensional"):
        FEModel(3, axisymmetric=True)


# ---------------------------------------------------------------------------------------------------------------------
# With Marmot's axisymmetric elements
# ---------------------------------------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def marmotAxisymmetricElements():
    """Skip unless Marmot provides the axisymmetric elements CX8RUL and CAX8R."""

    elementModule = pytest.importorskip("edelweissfe.elements.marmotelement.element")
    for elType in ("CX8RUL", "CAX8R"):
        try:
            elementModule.MarmotElementWrapper(elType, 1)
        except Exception:
            pytest.skip(f"Marmot has no {elType} element")


#: E = 1000, nu = 0, density 1: the material of the solved tests, Neo-Hooke (K = E / 3, G = E / 2) for the finite
#: strain element CX8RUL, linear elastic for the small strain element CAX8R; both with the id ``material``.
E = 1000.0
_MATERIAL = {
    "CX8RUL": f"*material, name=CompressibleNeoHooke, id=material\n{E / 3!r}, {E / 2!r}, 1.0\n",
    "CAX8R": f"*material, name=LinearElastic, id=material\n{E!r}, 0.0, 1.0\n",
}


def _ring(name: str, origin: tuple, size: tuple, divisions: tuple, elType: str = "CX8RUL") -> str:
    """A ring of axisymmetric elements, from the plane rectangle generator: its x is the radius, its y the axis."""

    return f"""
*modelGenerator, generator=planeRectQuad, name={name}
x0={origin[0]}
y0={origin[1]}
l={size[0]}
h={size[1]}
nX={divisions[0]}
nY={divisions[1]}
elType={elType}
"""


def _facets(surface: str, name: str) -> str:
    return f"""
*modelGenerator, generator=surfaceElementGenerator, name={name}Generator
surface={surface}
name={name}
"""


def _section(*elementSets: str) -> str:
    return "*section, name=section1, thickness=1.0, material=material, type=plane\n" + "\n".join(elementSets) + "\n"


def _tie(penalty: float) -> str:
    return f"""
*constraint, name=tie, type=surfaceToDeformableSurfacePenalty
slaveSurface=slaveSurf_facets, masterSurface=masterSurf_facets, penalty={penalty}, type=tied
"""


#: The domain 1 <= r <= 2, 0 <= z <= 1 as one mesh, and as two rings split horizontally (z = 0.5) or vertically
#: (r = 1.5): the generators, the element sets, and the node sets of the faces of the domain.
_MESHES = {
    "merged horizontally": (
        _ring("whole", (1, 0), (1, 1), (2, 4)),
        ["whole_all"],
        dict(bottom=["whole_bottom"], top=["whole_top"], inner=["whole_left"], outer=["whole_right"]),
    ),
    "merged vertically": (
        _ring("whole", (1, 0), (1, 1), (4, 2)),
        ["whole_all"],
        dict(bottom=["whole_bottom"], top=["whole_top"], inner=["whole_left"], outer=["whole_right"]),
    ),
    "tied horizontally": (
        _ring("lower", (1, 0), (1, 0.5), (2, 2))
        + _ring("upper", (1, 0.5), (1, 0.5), (2, 2))
        + _facets("upper_bottom", "slaveSurf")
        + _facets("lower_top", "masterSurf"),
        ["lower_all", "upper_all"],
        dict(
            bottom=["lower_bottom"],
            top=["upper_top"],
            inner=["lower_left", "upper_left"],
            outer=["lower_right", "upper_right"],
        ),
    ),
    "tied vertically": (
        _ring("inner", (1, 0), (0.5, 1), (2, 2))
        + _ring("outer", (1.5, 0), (0.5, 1), (2, 2))
        + _facets("outer_left", "slaveSurf")
        + _facets("inner_right", "masterSurf"),
        ["inner_all", "outer_all"],
        dict(
            bottom=["inner_bottom", "outer_bottom"],
            top=["inner_top", "outer_top"],
            inner=["inner_left"],
            outer=["outer_right"],
        ),
    ),
}

#: The prescribed displacements of the states, per face of the domain: (face, component, value).
_STATES = {
    "axial tension": [("bottom", 2, 0.0), ("top", 2, 1e-4)],
    "radial expansion": [("inner", 1, 1e-4 * 1.0), ("outer", 1, 1e-4 * 2.0), ("bottom", 2, 0.0)],
    "axial shear": [
        ("inner", 1, 0.0),
        ("inner", 2, 0.0),
        ("outer", 1, 0.0),
        ("outer", 2, 1e-4),
        ("bottom", 1, 0.0),
        ("top", 1, 0.0),
    ],
}


def _solveTheState(directory: Path, mesh: str, state: str, penalty: float | None = None) -> dict:
    """Solve a state on a mesh (implicitly, in one step), and return the displacements of the nodes by where they
    are: a list per position, two entries where a tie has a node on either side."""

    generators, elementSets, faces = _MESHES[mesh]
    dirichlets = "\n".join(
        f">>dirichlet, name=bc{i}{k}, nSet={nodeSet}, field=displacement, {component}={value!r}"
        for i, (face, component, value) in enumerate(_STATES[state])
        for k, nodeSet in enumerate(faces[face])
    )
    deck = f"""
{_MATERIAL["CX8RUL"]}
*job, name=axisymmetricState, domain=axisymmetric
*solver, solver=NIST, name=theSolver
{generators}
{_section(*elementSets)}
{_tie(penalty) if penalty is not None else ""}
*step, solver=theSolver
maxInc=1, minInc=1e-6, maxNumInc=10, maxIter=25, stepLength=1
>>options, name=theSolver, extrapolation=off
{dirichlets}
"""
    path = directory / "state.inp"
    path.write_text(deck)
    model, _ = finiteElementSimulation(parseInputFile(str(path)), verbose=False, suppressPlots=True)

    displacement = model.nodeFields["displacement"]
    displacementsAt = {}
    for node in model.nodeSets["all"]:
        if "displacement" not in node.fields:
            continue  # the center of an element of the generator's grid, which a quad8 has no node at
        position = tuple(np.round(node.coordinates, 12))
        displacementsAt.setdefault(position, []).append(np.array(displacement["U"][displacement.indexOfNode(node)]))
    return displacementsAt


def _deviationFromTheMergedMesh(tied: dict, merged: dict) -> float:
    """The largest difference of a displacement of the tied rings from the merged mesh at the same place, relative to
    the largest displacement of the merged mesh."""

    largest = max(np.abs(u).max() for displacements in merged.values() for u in displacements)
    return max(np.abs(u - merged[position][0]).max() for position, us in tied.items() for u in us) / largest


@pytest.mark.parametrize("state", list(_STATES))
@pytest.mark.parametrize("split", ["horizontally", "vertically"])
def test_tied_rings_transmit_a_state_as_the_merged_mesh_does(marmotAxisymmetricElements, tmp_path, split, state):
    """Two rings of CX8RUL elements tied across a horizontal (annulus) or a vertical (cylindrical) interface, against
    the same mesh merged. A tie of penalty :math:`k` holds the interface with springs that a traction :math:`t` opens
    by :math:`t / k`, so the tied rings differ from the merged mesh by an opening that falls as :math:`1 / k`, and by
    nothing else: tenfold the penalty, a tenth of the deviation. Where the state has no traction across the interface
    -- the radial expansion across the annulus (:math:`\\sigma_{zz} = 0` at :math:`\\nu = 0`), the axial tension across
    the cylinder (:math:`\\sigma_{rr} = 0`) -- the tie carries nothing, and the tied rings are the merged mesh.

    The springs stand for the areas of their rings: weighted otherwise than the rings of the elements, they would not
    balance the elements' forces at the interface, and the deviation would not vanish with the penalty."""

    merged = _solveTheState(tmp_path, f"merged {split}", state)
    deviation = [
        _deviationFromTheMergedMesh(_solveTheState(tmp_path, f"tied {split}", state, penalty), merged)
        for penalty in (1e6, 1e7)
    ]

    noTractionAcrossTheInterface = (split, state) in (
        ("horizontally", "radial expansion"),
        ("vertically", "axial tension"),
    )
    if noTractionAcrossTheInterface:
        assert max(deviation) < 1e-8
    else:
        assert 1e-6 < deviation[0] < 1e-2
        assert deviation[0] / deviation[1] == pytest.approx(10.0, rel=0.05)


_CONTACT_PATCH = Path(__file__).parents[1] / "testfiles" / "marmot" / "AxisymmetricSurfaceContactPatch" / "test.inp"


def test_an_axisymmetric_contact_patch_transmits_the_pressure_of_the_annulus(
    marmotAxisymmetricElements, tmp_path, monkeypatch
):
    """The deck ``AxisymmetricSurfaceContactPatch``: two rings over :math:`1 \\le r \\le 2`, a gap :math:`g_0 =
    10^{-3}` apart, pressed together by :math:`\\delta = 3 \\cdot 10^{-3}`. The gap closes, the pressure is uniform,
    :math:`\\sigma = (\\delta - g_0) / (2 h / E + 1 / k)` to the finite strain of the material, and its force on the
    annulus, :math:`\\pi (2^2 - 1^2) \\sigma`, is what the elements' ring-weighted reactions at the bottom carry."""

    monkeypatch.chdir(tmp_path)
    model, _ = finiteElementSimulation(parseInputFile(str(_CONTACT_PATCH)), verbose=False, suppressPlots=True)
    contact = model.constraints["contact"]

    assert model.axisymmetric
    assert contact.numberOfPointsWithASpring == contact.nPoints

    pressures = contact.getNormalPressures()
    sigma = (3e-3 - 1e-3) / (2 * 1.0 / E + 1 / 1e5)
    assert np.ptp(pressures) < 1e-6 * pressures.mean()
    assert pressures.mean() == pytest.approx(sigma, rel=5e-3)

    # The force on the upper ring pushes it up; the support of the lower ring carries it, the reaction at the supported
    # nodes.
    forceAcrossTheInterface = contact.contactForceOnSlave[1]
    assert forceAcrossTheInterface == pytest.approx(np.pi * 3.0 * pressures.mean(), rel=1e-10)
    assert forceAcrossTheInterface == pytest.approx(np.pi * 3.0 * sigma, rel=5e-3)
    displacement = model.nodeFields["displacement"]
    reaction = sum(displacement["P"][displacement.indexOfNode(node), 1] for node in model.nodeSets["lower_bottom"])
    assert reaction == pytest.approx(forceAcrossTheInterface, rel=1e-8)


@pytest.mark.parametrize("contactPoints", ["quadraturePoints", "nodes"])
def test_a_contact_patch_reaching_the_axis_transmits_the_pressure_of_the_disk(
    marmotAxisymmetricElements, tmp_path, monkeypatch, contactPoints
):
    """Two solid cylinders of CAX8R elements, 0 <= r <= 2, a gap :math:`g_0 = 10^{-3}` apart, pressed together by
    :math:`\\delta = 3 \\cdot 10^{-3}`: the axisymmetric elements accept their nodes on the axis, and so do the
    contact points. The reactions of the elements balance the force across the disk for either kind of point, and no
    pressure is undefined. The quadrature points give the consistent load of the rings, so the state is homogeneous, at
    the pressure :math:`\\sigma = (\\delta - g_0) / (2 h / E + 1 / k)` (linear elastic, small strain, :math:`\\nu = 0`)
    on the disk, :math:`\\pi 2^2 \\sigma`. The node on the axis stands for no area and reports zero."""

    deck = f"""
{_MATERIAL["CAX8R"]}
*job, name=axisymmetricContactOnTheAxis, domain=axisymmetric
*solver, solver=NIST, name=theSolver
{_ring("lower", (0, 0), (2, 1), (2, 2), "CAX8R")}
{_ring("upper", (0, 1.001), (2, 1), (2, 2), "CAX8R")}
{_facets("upper_bottom", "slaveSurf")}
{_facets("lower_top", "masterSurf")}
{_section("lower_all", "upper_all")}
*constraint, name=contact, type=surfaceToDeformableSurfacePenalty
slaveSurface=slaveSurf_facets, masterSurface=masterSurf_facets, penalty=1e5, type=linear, searchDistance=0.5, contactPoints={contactPoints}
*step, solver=theSolver
maxInc=0.25, minInc=1e-6, maxNumInc=100, maxIter=25, stepLength=1
>>options, name=theSolver, extrapolation=off
>>dirichlet, name=fixBottom, nSet=lower_bottom, field=displacement, 2=0.0
>>dirichlet, name=pushTop, nSet=upper_top, field=displacement, 2=-3e-3
"""
    monkeypatch.chdir(tmp_path)
    path = tmp_path / "axis.inp"
    path.write_text(deck)
    model, _ = finiteElementSimulation(parseInputFile(str(path)), verbose=False, suppressPlots=True)
    contact = model.constraints["contact"]

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        pressures, tractions = contact.getNormalPressures(), contact.getTractions()
    assert np.isfinite(pressures).all() and np.isfinite(tractions).all()
    onTheAxis = contact.slave.referenceCoordinates()[:, 0] == 0.0
    assert np.all(pressures[onTheAxis] == 0.0)

    forceAcrossTheInterface = contact.contactForceOnSlave[1]
    displacement = model.nodeFields["displacement"]
    reaction = sum(displacement["P"][displacement.indexOfNode(node), 1] for node in model.nodeSets["lower_bottom"])
    assert reaction == pytest.approx(forceAcrossTheInterface, rel=1e-8)

    if contactPoints == "quadraturePoints":
        sigma = (3e-3 - 1e-3) / (2 * 1.0 / E + 1 / 1e5)
        np.testing.assert_allclose(pressures, sigma, rtol=1e-10)
        assert forceAcrossTheInterface == pytest.approx(np.pi * 2.0**2 * sigma, rel=1e-10)
    else:
        assert onTheAxis.sum() == 1


def _tiedRingsForTheExplicitSolver(directory: Path, penalty: float) -> FEModel:
    """An inner and an outer ring of CX8RUL elements, with different meshes, tied across the cylinder r = 1.5 by the
    penalty tie; free, at rest, and prepared."""

    deck = f"""
{_MATERIAL["CX8RUL"]}
*job, name=axisymmetricTie, domain=axisymmetric
{_ring("inner", (1, 0), (0.5, 1), (1, 1))}
{_ring("outer", (1.5, 0), (0.5, 1), (1, 2))}
{_facets("outer_left", "slaveSurf")}
{_facets("inner_right", "masterSurf")}
{_section("inner_all", "outer_all")}
{_tie(penalty)}
"""
    path = directory / "tie.inp"
    path.write_text(deck)
    inputFile = parseInputFile(str(path))
    journal = Journal(verbose=False)
    model = fillFEModelFromInputFile(FEModel(2, axisymmetric=True), inputFile, journal)
    model.prepareYourself(journal)
    return model


def _exactCriticalTimeStep(model: FEModel) -> float:
    """:math:`2 / \\omega_\\mathrm{max}` of the assembled stiffness of the elements and the tie at rest, with the lumped
    masses of the elements: from the eigenvalues of :math:`M^{-1/2} K M^{-1/2}`."""

    nodes = [node for node in model.nodeSets["all"] if "displacement" in node.fields]
    indexOfNode = {node: i for i, node in enumerate(nodes)}
    K = np.zeros((2 * len(nodes), 2 * len(nodes)))
    M = np.zeros(2 * len(nodes))

    def dofsOf(nodesOfTheBlock, permutation=None):
        dofs = np.array([2 * indexOfNode[node] + component for node in nodesOfTheBlock for component in range(2)])
        return dofs if permutation is None else dofs[np.asarray(permutation)]

    for element in model.elements.values():
        if not element.hasKernels:
            continue
        dofs = dofsOf(element.nodes, element.dofIndicesPermutation)
        Ke, Pe, Me = np.zeros(element.nDof**2), np.zeros(element.nDof), np.zeros(element.nDof)
        zero = np.zeros(element.nDof)
        element.computeKernels(Ke, Pe, zero, zero, 0.0, 1e-3)
        element.computeLumpedInertia(Me)
        np.add.at(K, (dofs[:, None], dofs[None, :]), Ke.reshape((element.nDof, element.nDof)))
        np.add.at(M, dofs, Me)

    tie = model.constraints["tie"]
    zero = np.zeros(tie.nDof)
    _, Kc = residualContributionAndTangent(tie, zero, zero)
    dofs = dofsOf(tie.nodes)
    np.add.at(K, (dofs[:, None], dofs[None, :]), Kc)

    s = 1.0 / np.sqrt(M)
    return 2.0 / np.sqrt(np.max(np.linalg.eigvalsh(s[:, None] * 0.5 * (K + K.T) * s[None, :])))


@pytest.mark.parametrize("penalty", [1e3, 1e6], ids=["the elements bind", "the tie binds"])
def test_the_critical_time_step_of_an_axisymmetric_tie_is_estimated_below_the_exact_one(
    marmotAxisymmetricElements, tmp_path, penalty
):
    """``NED`` on two rings tied by a penalty: the critical time step it estimates -- the elements', and the one of the
    degrees of freedom the tie couples, whose row sums are scaled by the lumped masses -- is at most the exact one of
    the assembled system, and not far below it. The springs and the masses are both those of the full rings: weighted
    differently, the row sums would misjudge the frequencies of the tied nodes."""

    model = _tiedRingsForTheExplicitSolver(tmp_path, penalty)
    journal = Journal(verbose=False)
    fieldOutputController = FieldOutputController(model, journal)
    fieldOutputController.initializeJob()
    jobInfo = loadConfiguration(dict())
    solver = NED(jobInfo, journal)
    AdaptiveStepForExplicitSimulations(
        0,
        model,
        fieldOutputController,
        journal,
        jobInfo,
        solver,
        [],
        StepActionCollection(),
        stepLength=1e-4,
        startInc=1.0,
        maxInc=1.0,
        maxNumInc=100000,
    ).solve()
    fieldOutputController.finalizeJob()

    estimate = solver._criticalTimeStepOfTheModel()
    exact = _exactCriticalTimeStep(model)

    assert estimate <= exact * (1 + 1e-12)
    assert estimate > 0.3 * exact
    if penalty == 1e6:
        assert solver._criticalTimeStepWithThePenaltyConstraints.value < solver._criticalTimeStepOfTheElements.value


def _explicitContactDeck(directory: Path, maxNumInc: int, extra: str = "") -> Path:
    """Two rings of CAX8R elements over the annulus 1 <= r <= 2, the upper one pushed onto the lower one under ``NED``,
    with a throttled contact search; the step is longer than ``maxNumInc`` increments, the gap closes within the first
    few dozen."""

    deck = f"""
{_MATERIAL["CAX8R"]}
*job, name=axisymmetricContactRestart, domain=axisymmetric
*solver, solver=NED, name=theSolver
courant-number=0.8
output-frequency=5
contact-update-frequency=7
{_ring("lower", (1, 0), (1, 0.5), (2, 1), "CAX8R")}
{_ring("upper", (1, 0.501), (1, 0.5), (3, 1), "CAX8R")}
{_facets("upper_bottom", "slaveSurf")}
{_facets("lower_top", "masterSurf")}
{_section("lower_all", "upper_all")}
*constraint, name=contact, type=surfaceToDeformableSurfacePenalty
slaveSurface=slaveSurf_facets, masterSurface=masterSurf_facets, penalty=1e5, type=linear, searchDistance=0.1
{extra}
*step, type=adaptiveForExplicitSimulations, solver=theSolver
maxInc=1, minInc=1e-12, maxNumInc={maxNumInc}, maxIter=25, stepLength=1
>>dirichlet, name=fixLower, nSet=lower_bottom, field=displacement, 2=0
>>dirichlet, name=pushUpper, nSet=upper_top, field=displacement, 2=-0.03
"""
    path = directory / f"contact{maxNumInc}{'restart' if extra else ''}.inp"
    path.write_text(deck)
    return path


def _fieldsOf(path: Path) -> dict:
    model, _ = finiteElementSimulation(parseInputFile(str(path)), verbose=False, suppressPlots=True)
    return {(name, entry): np.array(field[entry]) for name, field in model.nodeFields.items() for entry in "UP"}


def test_a_resumed_axisymmetric_explicit_contact_is_bitwise_the_uninterrupted_one(marmotAxisymmetricElements, tmp_path):
    """The areas of the points are geometry, rebuilt with the model; the frozen projection is adopted from the
    checkpoint, as in a plane model. With elements whose critical time step does not depend on their state (CAX8R), so
    that the resumed run takes the time increments of the uninterrupted one."""

    nFull, nTruncated = 120, 90
    reference = _fieldsOf(_explicitContactDeck(tmp_path, nFull))
    writer = (
        f"*output, type=restart, name=restart\nwriteInterval=1, baseName={tmp_path / 'ckpt'}, numberOfFilesToKeep=1"
    )
    _fieldsOf(_explicitContactDeck(tmp_path, nTruncated, writer))
    (checkpoint,) = tmp_path.glob("ckpt_*.h5")
    resumed = _fieldsOf(_explicitContactDeck(tmp_path, nFull, f"*restart, readFrom={checkpoint}"))

    assert np.abs(reference["displacement", "U"]).max() > 0.0
    for key, values in reference.items():
        assert np.array_equal(resumed[key], values), key
