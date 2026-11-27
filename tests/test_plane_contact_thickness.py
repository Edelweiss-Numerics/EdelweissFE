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
"""Contact and tie in plane models, a slice of the thickness of their sections: every contact point stands for the
length of its share of the slave line times the thickness of the section of its element, as the plane elements
integrate times the thickness.

The blocks are those of the axisymmetric tests (:mod:`test_axisymmetric_contact`), in a plane model: a horizontal
(``"annulus"``) or a vertical (``"cylinder"``) interface. A uniform overlap transmits the pressure on the length times
the thickness, a tie the traction on it; a node between elements of different thicknesses stands for its share of the
area of each; an element without a plane section is refused. Two tied blocks, solved, transmit the same stress at any
thickness, with forces proportional to it.
"""

from pathlib import Path

import numpy as np
import pytest
from test_axisymmetric_contact import (  # noqa: E402  (tests/ is on sys.path)
    _BOTTOM,
    _CONTACT_POINTS,
    _CONTACT_POINTS_IDS,
    _TOP,
    DELTA,
    PENALTY,
    RI,
    RO,
    H,
    _constraint,
    _evaluate,
    _quad8Ring,
    _twoRings,
)

from edelweissfe.constraints.base.contactpointsonslavesurface import (
    ContactPointsOnSlaveSurface,
)
from edelweissfe.constraints.surfacetodeformablesurfacepenalty import (
    Constraint as SurfaceToDeformableSurfacePenalty,
)
from edelweissfe.constraints.surfacetodeformablesurfacepenalty import (
    SurfaceToDeformableSurfacePenaltySchema,
)
from edelweissfe.drivers.inputfiledrivensimulation import finiteElementSimulation
from edelweissfe.generators.surfaceelementgenerator import buildContactFacets
from edelweissfe.journal.journal import Journal
from edelweissfe.models.femodel import FEModel
from edelweissfe.sections.plane import PlaneSectionSchema
from edelweissfe.sections.plane import Section as PlaneSection
from edelweissfe.sets.elementset import ElementSet
from edelweissfe.utils.inputfileparser import parseInputFile

#: The length of the slave line of the blocks of :func:`_twoRings`, by interface.
_LENGTH = {"annulus": RO - RI, "cylinder": H}

#: The direction normal to the interface, by interface.
_NORMAL_AXIS = {"annulus": 1, "cylinder": 0}


@pytest.mark.parametrize("thickness", [1.0, 2.0])
@pytest.mark.parametrize("interface", ["annulus", "cylinder"])
@pytest.mark.parametrize("contactPoints", _CONTACT_POINTS, ids=_CONTACT_POINTS_IDS)
def test_a_uniform_overlap_transmits_the_pressure_on_the_length_times_the_thickness(
    thickness, interface, contactPoints
):
    """The slave block overlaps the master by :math:`\\delta` everywhere: the pressure is :math:`p = k \\delta` at every
    point, whatever the thickness, and the force across the interface is :math:`p L t`."""

    model, slaveSetName, masterSetName, _ = _twoRings(interface, axisymmetric=False, thickness=thickness)
    constraint = _constraint(model, slaveSetName, masterSetName, **contactPoints)
    U = np.zeros(constraint.nDof)
    _evaluate(constraint, U)

    assert constraint.numberOfPointsWithASpring == constraint.nPoints
    np.testing.assert_allclose(constraint.getNormalPressures(), PENALTY * DELTA, rtol=1e-10)

    expected = np.zeros(2)
    expected[_NORMAL_AXIS[interface]] = -PENALTY * DELTA * _LENGTH[interface] * thickness
    atol = 1e-12 * PENALTY * DELTA
    np.testing.assert_allclose(constraint.contactForceOnSlave, expected, rtol=1e-10, atol=atol)
    np.testing.assert_allclose(constraint.contactForceOnMaster, -expected, rtol=1e-10, atol=atol)


@pytest.mark.parametrize("thickness", [1.0, 2.0])
@pytest.mark.parametrize("interface", ["annulus", "cylinder"])
def test_a_tie_holds_the_block_with_the_traction_on_the_length_times_the_thickness(thickness, interface):
    """A tie paired at rest, its slave block moved by :math:`\\boldsymbol{d}`: the traction is :math:`-k
    \\boldsymbol{d}` at every point, whatever the thickness, the force on the slave block :math:`-k \\boldsymbol{d} L
    t`."""

    model, slaveSetName, masterSetName, _ = _twoRings(interface, axisymmetric=False, thickness=thickness)
    constraint = _constraint(model, slaveSetName, masterSetName, contactType="tied")
    d = np.array([3e-4, -2e-4])

    U = np.zeros(constraint.nDof)
    slaveNodes = constraint.slave.finiteElementNodes()
    for i, node in enumerate(constraint.nodes):
        if node in slaveNodes:
            U[2 * i : 2 * i + 2] = d
    _evaluate(constraint, U)

    np.testing.assert_allclose(constraint.getTractions(), np.tile(-PENALTY * d, (constraint.nPoints, 1)), rtol=1e-10)
    np.testing.assert_allclose(
        constraint.contactForceOnSlave, -PENALTY * d * _LENGTH[interface] * thickness, rtol=1e-10
    )


def _blocksOfTwoThicknesses() -> tuple:
    """The blocks of the annulus, in a plane model, the slave block of two columns of elements, the inner in a section
    of thickness 1, the outer in one of thickness 2: on the slave line, 1 <= x <= 1.5 is 1 thick, 1.5 <= x <= 2 is 2
    thick.

    Returns
    -------
    tuple
        The model, the slave facets and the master facets.
    """

    journal = Journal(verbose=False)
    model = FEModel(2)
    with model.topology.changes():
        slave = _quad8Ring(model, (RI, -H + DELTA), (RO - RI, H), (2, 2))
        master = _quad8Ring(model, (RI, 0.0), (RO - RI, H), (3, 2))
        model.surfaces["slaveFace"] = {_TOP: ElementSet("s", list(slave[:, -1]))}
        model.surfaces["masterFace"] = {_BOTTOM: ElementSet("m", list(master[:, 0]))}
        slaveSetName, _ = buildContactFacets(model, "slaveFace", "slv", "corner", "facetConsistent", journal)
        masterSetName, _ = buildContactFacets(model, "masterFace", "mst", "corner", "facetConsistent", journal)
    for name, elements, thickness in (
        ("inner", list(slave[0, :]) + list(master.ravel()), 1.0),
        ("outer", list(slave[1, :]), 2.0),
    ):
        model.sections[name] = PlaneSection(
            name, model, {}, [ElementSet(name, elements)], configuration=PlaneSectionSchema(thickness=thickness)
        )
    return model, slaveSetName, masterSetName


@pytest.mark.parametrize("contactPoints", _CONTACT_POINTS, ids=_CONTACT_POINTS_IDS)
def test_a_point_between_elements_of_two_thicknesses_stands_for_its_share_of_each(contactPoints):
    """A quadrature point stands for its length times the thickness of its own element; the node at x = 1.5, between
    the two, for its share of the length of either element times the thickness of each, :math:`\\frac{1}{4} \\cdot 1 +
    \\frac{1}{4} \\cdot 2`. The areas add up to the area of the slave line, :math:`\\frac{1}{2} \\cdot 1 + \\frac{1}{2}
    \\cdot 2`."""

    model, slaveSetName, masterSetName = _blocksOfTwoThicknesses()
    constraint = _constraint(model, slaveSetName, masterSetName, **contactPoints)

    assert constraint.slave.areas.sum() == pytest.approx(1.5, rel=1e-14)
    if contactPoints["contactPoints"] == "nodes":
        areaAt = {
            node.coordinates[0]: area for node, area in zip(constraint.slave.surfaceNodes, constraint.slave.areas)
        }
        assert areaAt == pytest.approx({1.0: 0.25, 1.5: 0.25 + 0.5, 2.0: 0.5}, rel=1e-14)
    else:
        origins = np.concatenate(
            [N @ X for N, X in zip(constraint.slave.shapeFunctions, constraint.slave.parentReferenceCoordinates)]
        )
        lengths = np.concatenate(constraint.slave.integrationWeights)
        np.testing.assert_allclose(constraint.slave.areas, lengths * np.where(origins[:, 0] < 1.5, 1.0, 2.0))


def test_an_element_of_the_slave_surface_without_a_plane_section_is_refused():
    model, slaveSetName, masterSetName, _ = _twoRings("annulus", axisymmetric=False)
    del model.sections["rings"]

    with pytest.raises(ValueError, match="no section with a thickness"):
        _constraint(model, slaveSetName, masterSetName)


class _PointsOfAPlaneBody(ContactPointsOnSlaveSurface):
    """Points of another library, measured as their own bodies integrate when they are made: here as the plane slice
    of unit thickness, whatever the model."""

    def __init__(self, slaveSurface: ElementSet, model: FEModel):
        super().__init__(slaveSurface, 3, model, "theContact")
        self.areas = np.concatenate(self.integrationWeights)
        self.axisymmetric = False


def test_points_made_for_a_model_of_another_kind_are_refused():
    model, slaveSetName, masterSetName, _ = _twoRings("annulus")
    configuration = SurfaceToDeformableSurfacePenaltySchema(
        slaveSurface=slaveSetName, masterSurface=masterSetName, penalty=PENALTY
    )

    with pytest.raises(ValueError, match="axisymmetric"):
        SurfaceToDeformableSurfacePenalty(
            "theContact",
            model,
            _PointsOfAPlaneBody(model.elementSets[slaveSetName], model),
            model.elementSets[masterSetName],
            Journal(verbose=False),
            configuration=configuration,
        )


def _solveTheTiedBlocks(directory: Path, thickness: float, state: str) -> tuple:
    """Two blocks of plane strain elements in a section of the given thickness, tied by a penalty across y = 0.5 with
    different meshes on either side, the bottom held and the top pulled up or sideways; solved in one increment.

    Returns
    -------
    tuple
        The displacements of the nodes, by label, and the reaction force on the bottom.
    """

    pull = {"tension": "2=1e-3", "shear": "1=1e-3, 2=0.0"}[state]
    deck = f"""
*material, name=LinearElastic, id=linearelastic, provider=edelweiss
1.0e4, 0.2

*job, name=planeTie, domain=2d
*solver, solver=NIST, name=theSolver

*modelGenerator, generator=planeRectQuad, name=lower
nX=2
nY=2
x0=0
y0=0
l=1
h=0.5
elType=CPE8
elProvider=edelweiss

*modelGenerator, generator=planeRectQuad, name=upper
nX=3
nY=2
x0=0
y0=0.5
l=1
h=0.5
elType=CPE8
elProvider=edelweiss

*modelGenerator, generator=surfaceElementGenerator, name=gen1
surface=upper_bottom
name=slaveSurf

*modelGenerator, generator=surfaceElementGenerator, name=gen2
surface=lower_top
name=masterSurf

*section, name=section1, thickness={thickness!r}, material=linearelastic, type=plane
lower_all
upper_all

*constraint, name=tie, type=surfaceToDeformableSurfacePenalty
slaveSurface=slaveSurf_facets, masterSurface=masterSurf_facets, penalty=1e5, type=tied

*step, solver=theSolver
maxInc=1, minInc=1e-6, maxNumInc=10, maxIter=25, stepLength=1
>>options, name=theSolver, extrapolation=off
>>dirichlet, name=fixBottom, nSet=lower_bottom, field=displacement, 1=0.0, 2=0.0
>>dirichlet, name=pullTop, nSet=upper_top, field=displacement, {pull}
"""
    path = directory / f"planeTie_{thickness}_{state}.inp"
    path.write_text(deck)
    model, _ = finiteElementSimulation(parseInputFile(str(path)), verbose=False, suppressPlots=True)

    displacement = model.nodeFields["displacement"]
    displacements = {
        node.label: np.array(displacement["U"][displacement.indexOfNode(node)])
        for node in model.nodeSets["all"]
        if "displacement" in node.fields
    }
    reaction = sum(
        np.array(displacement["P"][displacement.indexOfNode(node)]) for node in model.nodeSets["lower_bottom"]
    )
    return displacements, reaction


@pytest.mark.parametrize("state", ["tension", "shear"])
def test_tied_blocks_transmit_the_same_stress_at_any_thickness(tmp_path, state):
    """The blocks and the springs of the tie both scale with the thickness, so the solution is the same at any thickness
    -- the same displacements, the same stresses -- and the forces are proportional to it. Springs per unit thickness
    would be half as stiff, relative to blocks twice as thick, and open the interface twice as far."""

    displacementsOfTheThin, reactionOfTheThin = _solveTheTiedBlocks(tmp_path, 1.0, state)
    displacementsOfTheThick, reactionOfTheThick = _solveTheTiedBlocks(tmp_path, 2.0, state)

    largest = max(np.abs(u).max() for u in displacementsOfTheThin.values())
    for label, u in displacementsOfTheThin.items():
        np.testing.assert_allclose(displacementsOfTheThick[label], u, rtol=0, atol=1e-10 * largest)
    assert np.abs(reactionOfTheThin).max() > 0.0
    np.testing.assert_allclose(reactionOfTheThick, 2.0 * reactionOfTheThin, rtol=1e-8, atol=1e-10)
